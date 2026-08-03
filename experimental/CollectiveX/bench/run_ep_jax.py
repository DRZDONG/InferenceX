#!/usr/bin/env python3
"""CollectiveX EP benchmark entrypoint for JAX/TPU.

Sibling of bench/run_ep.py. It accepts the SAME argv the neutral case codec
(runtime/config.py ``case-args``) emits, so a TPU case is scheduled through the same
matrix, and it writes the SAME ``case-attempt`` artifact, so summarize.py and every
downstream consumer read it without a special case.

It is a separate entrypoint rather than a fifth ep_backend adapter because the execution
model differs at the root: one process owns all local devices, there is no
torch.distributed group, and the collective lives inside a compiled XLA program. See
bench/ep_jax.py for the transport and the timing caveat.

Differences from a GPU artifact, all of them recorded IN the artifact rather than left
for a reader to infer:
  * ``measurement.timing_source`` is ``host-wallclock-blocked``, not CUDA events.
  * ``components.stage`` is ``unavailable``: there is no separate staging pass to time
    (the permute is fused into dispatch).
  * ``implementation.kernel_generation`` names the transport that actually ran.
  * ``implementation.oracle`` names the probe oracle (source-ID identity of every
    dispatched copy + exact unweighted rank-sum on combine), which is narrower than the
    GPU harness's full per-expert transform oracle.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import numpy as np  # noqa: E402

import ep_harness  # noqa: E402
import ep_jax  # noqa: E402
import routing_np  # noqa: E402
import xprof  # noqa: E402


TIMING_SOURCE = "host-wallclock-blocked"
COMPONENTS = ("dispatch", "combine", "roundtrip")
# The reference program is traced alongside them but is NOT a published component: it is
# the cross-check against the host-timed reference figure, and the gate below deliberately
# does not require it (a failed reference capture must not fail a case).
REFERENCE = "reference"
TRACED = COMPONENTS + (REFERENCE,)


def traced_operation(point, component: str):
    """The callable whose device time a traced capture measures."""
    if component == REFERENCE:
        return point.reference_program()[0]
    return point.timed_operation(component)


def traced_markers(component: str, tokens_per_rank: int):
    """(outer scope, inner scopes) to attribute a capture of `component` to.

    The reference program IS the collective -- there is no permute inside it to separate
    out -- so its inner scope is the collective HLO itself. Comparing that against
    dispatch's `transport_us` is what settles whether the host-timed reference figure
    sitting at ~2x the collective is real device work or host overhead.
    """
    if component == REFERENCE:
        return ep_jax.REFERENCE_MARKER, ep_jax.REFERENCE_HLO
    return (ep_jax.scope_name(component, tokens_per_rank),
            (ep_jax.transport_scope(component, tokens_per_rank),) + ep_jax.TRACED_HLO)


def transport_key(component: str, tokens_per_rank: int) -> str:
    """The `by_hlo` key holding just the collective for `component`."""
    if component == REFERENCE:
        return ep_jax.REFERENCE_HLO[0]
    return ep_jax.transport_scope(component, tokens_per_rank)


def profile_session(jax, directory: str):
    """`jax.profiler.trace` with host-side tracing turned down where supported.

    Only the DEVICE plane carries the spans this publishes; host TraceMe and the Python
    tracer contribute the bulk of a ~400k-event capture and nothing that is read back.
    Trimming them is the cheapest way to reduce the volume implicated in captures that
    arrive with no scope metadata. Guarded because ProfileOptions is not in every JAX
    version -- an older runtime silently gets the default capture rather than an error.
    """
    try:
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 1
        options.python_tracer_level = 0
        return jax.profiler.trace(directory, profiler_options=options)
    except Exception:
        return jax.profiler.trace(directory)


def record_collective_bandwidth(entry: dict, transport, span, egress_bytes,
                                combine_transport=None) -> dict:
    """Publish every device figure for the collective. They disagree; that is the point.

    Measured at T=8192 (us/iteration, op_inventory, with both full-buffer reshapes
    removed from the reference program):

        reference program   6680 call-done + 3280 ragged_all_to_all.8 + 363 zeros
        combine  transport  6690 call-done + 3283 ragged_all_to_all.6
        dispatch transport  6683 call-done

    `transport_us` sums the ops whose name carries the hyphenated collective HLO, exactly
    and consistently. What is NOT settled is whether the 3,280us `ragged_all_to_all.8`
    beside it is part of the collective's lowering or a buffer copy XLA inserts around
    it. It survives removing the output `[None]` and the operand `[0]`, and its cost
    matches a full copy of the 0.624GB output; donating that buffer added an op instead
    of removing one.

    So all three figures ship with the ops behind them, and this docstring deliberately
    does not nominate one as the collective's true cost. `op_inventory` on each component
    is the evidence; a reader who needs the distinction can see exactly what differs.
    """
    entry["collective_device_us"] = transport
    entry["collective_device_us_combine"] = combine_transport
    entry["collective_device_us_span"] = span
    entry["collective_device_us_note"] = (
        "Three device figures for the same collective, and they differ. "
        "collective_device_us is dispatch's transport scope; _combine is combine's, "
        "which additionally contains a ~3,280us op named for the collective primitive; "
        "_span is a bare-collective program, which contains that op plus a ~363us output "
        "memset. Whether that op is the collective's lowering or a buffer copy around it "
        "is unresolved -- see device_timing.<component>.op_inventory for the ops behind "
        "each figure rather than relying on one number."
    )
    entry["collective_bandwidth_gbps_per_device"] = (
        egress_bytes / (transport * 1e-6) / 1e9 if transport else None
    )
    entry["collective_bandwidth_gbps_per_device_combine"] = (
        egress_bytes / (combine_transport * 1e-6) / 1e9 if combine_transport else None
    )
    entry["collective_bandwidth_gbps_per_device_span"] = (
        egress_bytes / (span * 1e-6) / 1e9 if span else None
    )
    return entry


def xprof_points(spec: str, ladder: list) -> list:
    """Ladder points to capture device timings for.

    Tracing every point is what pushed the prefill shard past its wall-clock budget: the
    profiler instruments every op on every device, and prefill ops are long. The ends
    characterise what the device numbers are for -- the floor's share at small T and the
    bandwidth-bound regime at large T -- for a fraction of the cost.
    """
    if not ladder:
        return []
    if spec == "all":
        return list(ladder)
    if spec == "ends":
        return sorted({ladder[0], ladder[-1]})
    wanted = {int(value) for value in spec.replace(",", " ").split() if value.strip()}
    return [tokens for tokens in ladder if tokens in wanted]


DEVICE_TIMING_SOURCE = "xla-device-trace-span"


def _case_timing_source(rows: list) -> str:
    """One label for the case, or `mixed` when its rows disagree.

    Never collapses a mixture to the better of the two: the per-row `timing_source`
    stays authoritative and this only summarises it.
    """
    seen = {source for row in rows for source in row["timing_source"].values()}
    if not seen:
        return TIMING_SOURCE
    return seen.pop() if len(seen) == 1 else "mixed"


def timing_source(from_device: bool) -> str:
    """Which measurement produced the published percentiles.

    `xla-device-trace-span` is the per-occurrence span between the first and last op of
    the component on the slowest device -- the direct equivalent of the CUDA-event pair
    the GPU SKUs use, and the only TPU number of the same KIND as theirs. The host
    fallback carries a ~450us per-call dispatch floor and is not comparable to them.
    """
    return DEVICE_TIMING_SOURCE if from_device else TIMING_SOURCE
VALUE_BYTES = ep_jax.JaxEPTransport.dispatch_value_bytes


def _runtime_info(jax, vendor: str, device) -> dict:
    """Runtime versions needed to compare and debug a TPU result."""
    try:
        import jaxlib  # noqa: PLC0415

        accelerator_runtime = getattr(jaxlib, "__version__", None)
    except ImportError:
        accelerator_runtime = None
    return {
        # There is no separate collective library on TPU: interconnect collectives are
        # compiled into the XLA program, so the "library" is the compiler itself.
        "accelerator_runtime": accelerator_runtime,
        "collective_library": {
            "kind": "xla-ici", "version": getattr(jax, "__version__", None),
        },
        "framework": f"jax-{getattr(jax, '__version__', 'unknown')}",
        "vendor": vendor,
        "device_kind": getattr(device, "device_kind", None),
    }


def _routing_stats(idx_g, args, experts_per_rank: int, ep_size: int, tokens: int) -> dict:
    stats = routing_np.routing_stats(idx_g, args.experts, experts_per_rank)
    stats["locality"] = routing_np.routing_locality(
        idx_g, experts_per_rank, ep_size, max(1, tokens),
        args.gpus_per_node, args.scale_up_domain,
    )
    return stats


def _expected_combine(activations: np.ndarray, layout: ep_jax.Layout,
                      rank: int) -> np.ndarray:
    """fanout(token) * x[token] -- the unweighted rank-sum of identity experts.

    The probe's expert is the identity, so the exact expected combine for a source token
    is its activation scaled by the number of distinct destination ranks it reached. Each
    activation encodes its own global source ID in the first 32 columns, so a copy that
    comes back carrying the wrong token is caught by value, not merely by count.
    """
    counts = np.bincount(
        layout.send_index[rank, :int(layout.send_total[rank])],
        minlength=layout.tokens_per_rank,
    ).astype(np.float32)
    return activations.astype(np.float32) * counts[:, None]


def _check_dispatch(received: np.ndarray, layout: ep_jax.Layout, seed: int,
                    ep_size: int) -> bool:
    """Every dispatched copy carries the exact bytes of the token it claims to be, and
    lands in the slot the exchange plan assigned it."""
    for rank in range(ep_size):
        rows = int(layout.recv_total[rank])
        if rows == 0:
            continue
        payload = received[rank, :rows].astype(np.float32)
        try:
            source = routing_np.decode_source_ids(payload, seed)
        except ValueError:
            return False
        expected = routing_np.activations_for_source_ids(source, payload.shape[1], seed)
        if not np.array_equal(payload, expected):
            return False
        # A transport that delivers the right bytes to the wrong offset still fails here.
        for src in range(ep_size):
            size = int(layout.recv_sizes[rank, src])
            if not size:
                continue
            start = int(layout.recv_offsets[rank, src])
            sent_from = int(layout.input_offsets[src, rank])
            local = layout.send_index[src, sent_from:sent_from + size]
            want = local.astype(np.int64) + src * layout.tokens_per_rank
            if not np.array_equal(np.sort(source[start:start + size]), np.sort(want)):
                return False
    return True


def _combine_error(combined: np.ndarray, activations: np.ndarray,
                   layout: ep_jax.Layout, ep_size: int) -> float:
    """Max magnitude-floored relative error against the exact expected combine."""
    worst = 0.0
    for rank in range(ep_size):
        expected = _expected_combine(activations[rank], layout, rank)
        got = combined[rank].astype(np.float32)
        denominator = np.maximum(np.abs(expected), ep_harness.COMBINE_MAG_FLOOR)
        worst = max(worst, float(np.max(np.abs(got - expected) / denominator)))
    return worst


def _ladder_within_budget(ladder, layout_for, hidden: int, budget_bytes: int):
    """Split the ladder into points that fit the per-device payload budget and those that
    do not. Oversized points are REPORTED as dropped, never silently truncated -- the same
    contract EPBackend.buffer_cap gives the GPU backends."""
    kept, dropped = [], []
    for tokens in ladder:
        layout = layout_for(tokens)
        # dispatch receive buffer + combine return buffer + the fp32 accumulator
        need = ((layout.max_recv + layout.max_send) * hidden * VALUE_BYTES
                + tokens * hidden * 4)
        (kept if need <= budget_bytes else dropped).append(tokens)
    return kept, dropped


def _by_rank(array, ep_size: int) -> np.ndarray:
    """(ep*rows, hidden) -> (ep, rows, hidden). A view; nothing is copied or timed."""
    values = np.asarray(array)
    if values.ndim == 3:  # already per-rank (stubs in the test suite)
        return values
    return values.reshape(ep_size, -1, values.shape[-1])


def _oracle(point, layout, activations, seed, ep_size, jax):
    """Run both halves of the probe oracle; returns (dispatch_ok, max_relative_error)."""
    # The device programs return (ep*rows, hidden): shard_map concatenates the per-shard
    # 2D results, because expanding them to a leading axis of size 1 inside the program
    # cost a full re-materialisation of the buffer on every call. Reshaping back is a
    # numpy view on the host, outside anything timed.
    received = _by_rank(jax.block_until_ready(point.dispatch()), ep_size)
    combined = _by_rank(jax.block_until_ready(point.combine()), ep_size)
    return (
        _check_dispatch(received, layout, seed, ep_size),
        _combine_error(combined, activations, layout, ep_size),
    )


def _scheduled_case(args, ep_size: int, nodes: int, ladder: list[int]) -> dict:
    return {
        "backend": args.backend,
        "ep": ep_size,
        "experts": args.experts,
        "gpus_per_node": args.gpus_per_node,
        "hidden": args.hidden,
        "ladder": " ".join(map(str, ladder)),
        "mode": args.mode,
        "nodes": nodes,
        "phase": args.phase,
        "precision": args.precision,
        "routing": args.routing,
        "scale_up_domain": args.scale_up_domain,
        "scale_up_transport": args.scale_up_transport,
        "scale_out_transport": args.scale_out_transport or None,
        "scope": args.scope,
        "suite": args.suite,
        "topk": args.topk,
        "topology_class": args.topology_class,
        "transport": args.transport,
        "workload": args.workload_name,
    }



def _attempt_ordinal() -> int:
    try:
        attempt_ordinal = int(os.environ.get("COLLX_ATTEMPT_ID", "1"))
    except ValueError:
        attempt_ordinal = 0
    if attempt_ordinal <= 0:
        raise ValueError("COLLX_ATTEMPT_ID must be a positive integer")
    return attempt_ordinal


def _write_terminal_failure(args, reason: str, vendor: str, ep_size: int,
                            nodes: int) -> int:
    ladder, _ = ep_harness.token_ladder(args.tokens_ladder, None)
    scheduled_case = _scheduled_case(args, ep_size, nodes, ladder)
    computed = ep_harness.case_id(args.runner, scheduled_case)
    if args.case_id != computed:
        print(f"ERROR: scheduled case ID does not match realized factors: "
              f"{args.case_id} != {computed}", file=sys.stderr)
        return 2
    try:
        attempt_ordinal = _attempt_ordinal()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    source_sha = (os.environ.get("COLLECTIVEX_SOURCE_SHA")
                  or os.environ.get("GITHUB_SHA"))
    doc = {
        "version": args.version,
        "record_type": "case-attempt",
        "generated_at": _dt.datetime.now().astimezone().isoformat(),
        "identity": {
            "allocation_factors": {
                "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "run_id": os.environ.get("GITHUB_RUN_ID"),
                "source_sha": source_sha,
            },
            "attempt_ordinal": attempt_ordinal,
            "case_factors": {"case": scheduled_case, "sku": args.runner},
            "case_id": args.case_id,
        },
        "workload": {"cross_rank_consistent": False},
        "measurement": {
            "rows": [],
            "sampling": {
                "iterations_per_trial": args.iters,
                "samples_per_component": 0,
                "trials": args.trials,
                "warmup_iterations": args.warmup,
            },
            "timing_source": timing_source(False),
        },
        "implementation": {
            "fp8_consume": None,
            "kernel_generation": "jax-ragged-all-to-all",
            "name": args.backend,
            "oracle": "probe-source-identity-and-exact-rank-sum",
        },
        "topology": {
            "device_product": None,
            "gpus_per_node": args.gpus_per_node,
            "nodes": nodes,
            "placement": "packed",
            "scale_up_domain": args.scale_up_domain,
            "scale_up_transport": args.scale_up_transport,
            "scale_out_transport": args.scale_out_transport or None,
            "scope": args.scope,
            "topology_class": args.topology_class,
            "transport": args.transport,
            "world_size": ep_size,
        },
        "runtime": {"vendor": vendor},
        "provenance": {
            "image": os.environ.get("COLLECTIVEX_IMAGE", "") or None,
            "source_sha": source_sha,
        },
        "outcome": {"reasons": [reason], "status": "failed"},
    }
    ep_harness._write_json_atomic(args.out, doc)
    print(f"{args.backend} ep-dispatch-combine [{args.phase}/{args.mode}]: "
          f"status=failed reason={reason} -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The TPU entrypoint's full argv contract.

    Module-level so tests exercise the REAL parser instead of a hand-kept mirror of it --
    a mirror silently stops covering every flag added after it was written.
    """
    ap = argparse.ArgumentParser(
        description="CollectiveX EP dispatch/combine sweep (JAX/TPU)"
    )
    ap.add_argument("--backend", required=True, choices=["jax-ragged-a2a"])
    ap.add_argument("--xprof", action="store_true", default=True,
                    help="capture a profiler trace and publish device-side spans as the "
                         "PRIMARY latency. This is the CUDA-event equivalent and the only "
                         "TPU measurement of the same kind as the GPU SKUs'")
    ap.add_argument("--no-xprof", dest="xprof", action="store_false",
                    help="fall back to host wall-clock, which carries a ~450us per-call "
                         "dispatch floor and is NOT comparable to the GPU SKUs")
    ap.add_argument("--allow-host-fallback", action="store_true",
                    help="accept host wall-clock for points the profiler did not cover. "
                         "Off by default: a silent fallback publishes a ~450us-floored "
                         "latency under a device label")
    ap.add_argument("--xprof-iters", type=int, default=30,
                    help="calls per component per traced ladder point; kept small because "
                         "the profiler records every op on every device")
    ap.add_argument("--reference-method", action="store_true", default=True,
                    help="also time the bare collective the way "
                         "AI-Hypercomputer/accelerator-microbenchmarks does: the "
                         "collective alone as the whole jitted program over a pre-staged "
                         "input, host wall-clock per call, IQR-filtered, with per-device "
                         "egress bandwidth. Directly comparable to published TPU numbers, "
                         "unlike the composite dispatch/combine components")
    ap.add_argument("--no-reference-method", dest="reference_method",
                    action="store_false")
    ap.add_argument("--xprof-points", default="all",
                    help="which ladder points to trace: 'all' (the default, now that "
                         "device spans are the published latency), 'ends', or a "
                         "comma-list")
    ap.add_argument("--max-payload-gib", type=float, default=16.0,
                    help="per-device payload budget; ladder points needing more are "
                         "dropped and reported, not truncated")
    ap.add_argument("--terminal-failure", choices=["case-timeout", "case-process-failed"],
                    help="write a terminal failed artifact without initializing JAX")
    ep_harness.add_common_args(ap)
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if not ep_harness.is_case_id(args.case_id):
        print(f"ERROR: invalid native case ID {args.case_id!r}", file=sys.stderr)
        return 2
    if args.mode != "normal":
        print(f"ERROR: {args.backend} realizes only normal mode, not {args.mode!r}",
              file=sys.stderr)
        return 2
    if args.precision != "bf16":
        print(f"ERROR: {args.backend} is BF16-only; got {args.precision!r}",
              file=sys.stderr)
        return 2
    if min(args.iters, args.trials, args.warmup) <= 0:
        print("ERROR: iters/trials/warmup must be positive", file=sys.stderr)
        return 2

    vendor = os.environ.get("COLLX_VENDOR", "").strip().lower()
    runtime_kind = os.environ.get("COLLX_RUNTIME", "").strip().lower()
    if not vendor:
        print("ERROR: COLLX_VENDOR is required", file=sys.stderr)
        return 2
    if runtime_kind != "tpu":
        print(f"ERROR: configured runtime {runtime_kind!r} is not 'tpu'", file=sys.stderr)
        return 2

    nodes = max(1, int(os.environ.get("COLLX_NODES", "1")))
    ep_size = args.gpus_per_node * nodes
    if args.terminal_failure:
        return _write_terminal_failure(
            args, args.terminal_failure, vendor, ep_size, nodes,
        )

    try:
        jax, jnp, shard_map = ep_jax.import_jax()
    except Exception as exc:  # pragma: no cover - image problem, not a logic path
        print(f"ERROR: jax unavailable: {exc!r}", file=sys.stderr)
        return 3

    devices = jax.devices()
    if len(devices) < ep_size:
        kind = devices[0].device_kind if devices else "none"
        print(f"ERROR: case needs EP{ep_size} but this process sees {len(devices)} "
              f"device(s) of kind {kind!r}", file=sys.stderr)
        return 2
    devices = devices[:ep_size]
    if args.experts % ep_size:
        print(f"ERROR: experts ({args.experts}) must divide EP{ep_size}", file=sys.stderr)
        return 2
    experts_per_rank = args.experts // ep_size

    mesh = jax.sharding.Mesh(np.asarray(devices), ("ep",))
    try:
        transport = ep_jax.JaxEPTransport(
            jax, jnp, shard_map, mesh, ep_size, args.hidden,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    ladder, _ = ep_harness.token_ladder(args.tokens_ladder, None)
    if not ladder:
        print(f"ERROR: empty token ladder (phase={args.phase})", file=sys.stderr)
        return 2

    # Routing traces are deterministic in (T, ep, experts, topk, seed), so each point's
    # trace is built once and reused for the layout, the oracle, and the published stats.
    traces, layouts = {}, {}

    def layout_for(tokens: int) -> ep_jax.Layout:
        if tokens not in layouts:
            idx_g, weights_g = routing_np.build_global_routing(
                tokens * ep_size, args.experts, args.topk, args.routing, args.seed,
            )
            traces[tokens] = (idx_g, weights_g)
            layouts[tokens] = ep_jax.build_layout(
                idx_g, tokens, ep_size, experts_per_rank,
            )
        return layouts[tokens]

    ladder, dropped = _ladder_within_budget(
        ladder, layout_for, args.hidden, int(args.max_payload_gib * (1 << 30)),
    )
    if dropped:
        print(f"NOTE: dropped tokens/rank {dropped} — exceed the "
              f"{args.max_payload_gib} GiB per-device payload budget "
              f"(hidden={args.hidden}); not silently truncated.")
    if not ladder:
        print("ERROR: every ladder point exceeds the per-device payload budget",
              file=sys.stderr)
        return 2

    print(f"[run_ep_jax] backend={args.backend} "
          f"phase={args.phase} mode={args.mode} ep_size={ep_size} "
          f"hidden={args.hidden} topk={args.topk} experts={args.experts} "
          f"precision={args.precision} routing={args.routing} seed={args.seed} "
          f"device={devices[0].device_kind}")

    # ---- Pass 1: materialize, compile, and gate every point -------------------
    points, activations, gates = {}, {}, {}
    for tokens in ladder:
        layout = layout_for(tokens)
        activations[tokens] = np.stack([
            routing_np.rank_activations(tokens, args.hidden, args.seed, rank)
            for rank in range(ep_size)
        ])
        point = transport.build(layout, activations[tokens])
        # Compile and settle before anything gate-bearing runs.
        jax.block_until_ready(point.roundtrip())
        dispatch_ok, max_rel = _oracle(
            point, layout, activations[tokens], args.seed, ep_size, jax,
        )
        points[tokens] = point
        gates[tokens] = {
            "rstats": _routing_stats(
                traces[tokens][0], args, experts_per_rank, ep_size, tokens,
            ),
            "dispatch_ok": dispatch_ok,
            "max_rel": max_rel,
        }

    # ---- Pass 2: timed trials, same rotated orders as the GPU harness ---------
    pools = {tokens: {name: [] for name in COMPONENTS} for tokens in ladder}
    # ---- Reference method: the bare collective, timed as the reference times it ----
    reference = {}
    if args.reference_method:
        for tokens in ladder:
            built = points[tokens].reference_program()
            if built is None:
                break
            operation, _ = built
            samples = ep_jax.reference_timing(jax, operation)
            metrics = ep_jax.iqr_metrics(samples)
            # DENSE all_to_all pads every chunk to the largest (src, dst) pair, so it
            # moves slightly more than the ragged exchange. Bill what it actually moved,
            # and record the delta rather than treating it as negligible.
            ragged_egress = ep_jax.egress_bytes(layout_for(tokens), args.hidden)
            dense_egress = getattr(points[tokens], "reference_egress_bytes", None)
            egress = {"mean_per_device": dense_egress or ragged_egress["mean_per_device"]}
            latency_s = (metrics["avg_ms"] / 1000.0) if metrics else 0.0
            reference[tokens] = {
                "method": "accelerator-microbenchmarks/AllToAllBenchmark",
                "note": "host-timed, so it carries a ~450us per-call floor. Two device "
                        "figures sit below and they DISAGREE by up to ~1.5x; see "
                        "collective_device_us_note, and op_inventory for the ops behind "
                        "each. Which one is the collective's true cost is unresolved",
                "timed": "DENSE jax.lax.all_to_all(split_axis=0, concat_axis=0, "
                         "tiled=True) inside one named scope, operand in and result out "
                         "with no indexing -- AllToAllBenchmark transcribed. Dense takes "
                         "no output buffer, so no allocation, memset or reshape is inside "
                         "the timed region",
                "padding_vs_ragged": {
                    "note": "dense pads every (src,dst) chunk to the largest, so it moves "
                            "this much more than the probe's ragged exchange",
                    "dense_egress_bytes_per_device": dense_egress,
                    "ragged_egress_bytes_per_device": ragged_egress["mean_per_device"],
                    "ratio": (dense_egress / ragged_egress["mean_per_device"]
                              if dense_egress and ragged_egress["mean_per_device"] else None),
                },
                "latency": metrics,
                "egress_bytes_per_device": egress["mean_per_device"],
                "bandwidth_gbps_per_device": (
                    egress["mean_per_device"] / latency_s / 1e9 if latency_s > 0 else None
                ),
            }
        if reference:
            worst = max(reference)
            entry = reference[worst]
            print(f"[run_ep_jax] reference method T={worst}: "
                  f"avg={entry['latency']['avg_ms'] * 1000:.1f}us "
                  f"p50={entry['latency']['p50_ms'] * 1000:.1f}us "
                  f"({entry['latency']['kept_after_iqr']}/{entry['latency']['samples']} "
                  f"samples kept) -> "
                  f"{entry['bandwidth_gbps_per_device']:.1f} GB/s per device egress")


    # Host timing is now the SANITY CHECK, not the published latency, so it uses the
    # reference harness's light sampling rather than the shared 2048-sample profile: at a
    # ~450us per-call floor the full profile costs an hour of wall clock to produce
    # numbers the device trace supersedes.
    for tokens in ladder:
        point = points[tokens]
        for name in COMPONENTS:
            pools[tokens][name] = ep_jax.reference_timing(
                jax, point.timed_operation(name), warmup_tries=5, num_runs=10,
                min_duration_s=0.0,
            )
            pools[tokens][name] = [value * 1000.0 for value in pools[tokens][name]]

    # ---- Traced pass: device-side spans, the published latency -----------------
    # ONE TRACE PER LADDER POINT, not one over the whole ladder. A single trace filled up
    # at ~648k events and silently stopped recording partway through, so every point after
    # the fifth had no device data and fell back to host timing while the artifact still
    # claimed `xla-device-trace-span`. Per-point traces are individually small, bound the
    # failure to the point that caused it, and make find_trace unambiguous.
    device_timing = {}
    traced_points = xprof_points(args.xprof_points, ladder) if args.xprof else []
    calls = max(1, args.xprof_iters)
    if traced_points:
        print(f"[run_ep_jax] xprof tracing {len(traced_points)} point(s), "
              f"{calls} calls/component, one trace per component")
    for tokens in traced_points:
        point = points[tokens]
        parsed_point = {}
        # ONE PROFILER SESSION PER COMPONENT, not one per point. Three components in a
        # single session produced ~400k events, and past the fifth traced point the
        # capture began returning traces with every op recorded but NO CollectiveX scope
        # metadata attached (`scopes=[]` over 400k timed events) -- intermittently, on the
        # same code that had covered the full ladder on an earlier run. Splitting the
        # session cuts per-capture volume ~3x and bounds a bad capture to one component
        # rather than a whole point. It also makes each trace hold exactly one marker, so
        # attribution cannot collide.
        for name in TRACED:
            with tempfile.TemporaryDirectory(
                    prefix=f"collectivex-xprof-t{tokens}-{name}-") as directory:
                try:
                    operation = traced_operation(point, name)
                    with profile_session(jax, directory):
                        for _ in range(calls):
                            jax.block_until_ready(operation())
                except Exception as exc:  # a failed capture must not lose the case
                    print(f"[run_ep_jax] xprof capture failed at T={tokens} "
                          f"{name}: {exc!r}", file=sys.stderr)
                    parsed_point[name] = {"error": repr(exc)}
                    continue
                marker, hlo = traced_markers(name, tokens)
                # The nested transport scope isolates the collective from the
                # permute/scatter around it, which is what makes combine comparable
                # to a GPU backend whose kernel fuses the reduction.
                parsed_point[name] = xprof.parse_trace_durations(
                    directory, marker, calls, hlo)
        for name, parsed in parsed_point.items():
            # Everything in the component that is not the collective -- the permute for
            # dispatch, the scatter-add for combine -- by subtraction. Naming those ops is
            # ambiguous; subtraction is not.
            total = parsed.get("op_per_iteration_us")
            transport = ((parsed.get("by_hlo") or {})
                         .get(transport_key(name, tokens), {})
                         .get("per_iteration_us"))
            if total is not None and transport is not None:
                parsed["transport_us"] = transport
                parsed["non_transport_us"] = total - transport
        device_timing[tokens] = parsed_point
        summary = " ".join(
            f"{name}={parsed['percentiles_us']['p50']:.1f}"
            for name, parsed in parsed_point.items()
            if parsed.get("percentiles_us")
        )
        print(f"[run_ep_jax] device span p50 T={tokens}: {summary or 'NONE'}")
        # A diagnostic PER MISSING COMPONENT. Reporting only when the whole point missed
        # hid the decisive fact for a full run: `reference` was captured at all 14 points
        # while the other three were captured at 5, which is what identified the compile
        # cache. A partial miss is the interesting case, so it must not be the silent one.
        for name, parsed in parsed_point.items():
            if parsed.get("percentiles_us"):
                continue
            probe = parsed.get("diagnostic") or {}
            print(f"[run_ep_jax]   {name} UNATTRIBUTED marker={parsed.get('marker')!r} "
                  f"error={parsed.get('error')!r} "
                  f"files={probe.get('trace_files')} "
                  f"timed_events={probe.get('events_with_duration')} "
                  f"devices={probe.get('devices_seen')} "
                  f"scopes={probe.get('collectivex_scopes_present')} "
                  f"all_to_all={probe.get('all_to_all_events')} "
                  f"top={probe.get('top_event_names')}")

    # A point that was asked for device timing and did not get it would otherwise publish
    # a host latency under a device label. Fail the case instead: silent substitution is
    # how a green run ends up carrying numbers that do not mean what they say.
    if traced_points:
        missing = [
            tokens for tokens in traced_points
            if not all((device_timing.get(tokens, {}).get(name) or {}).get("percentiles_us")
                       for name in COMPONENTS)
        ]
        if missing and not args.allow_host_fallback:
            print(f"ERROR: no device spans at tokens/rank {missing} despite --xprof. "
                  f"The published latency would silently be host wall-clock, which "
                  f"carries a ~450us dispatch floor and is not comparable to the GPU "
                  f"SKUs. Rerun with --allow-host-fallback to accept mixed provenance.",
                  file=sys.stderr)
            return 6
        if missing:
            print(f"[run_ep_jax] WARNING: host-timed fallback at T={missing}")

    # Achieved collective bandwidth from DEVICE time for the collective alone. Must run
    # AFTER the traced pass -- the reference block above is measured before tracing, so it
    # cannot see device_timing when it is built.
    for tokens, entry in reference.items():
        transport = ((device_timing.get(tokens, {}).get("dispatch") or {})
                     .get("transport_us"))
        egress_bytes = entry["egress_bytes_per_device"]
        span = ((device_timing.get(tokens, {}).get(REFERENCE) or {})
                .get("percentiles_us") or {}).get("p50")
        combine_transport = ((device_timing.get(tokens, {}).get("combine") or {})
                             .get("transport_us"))
        record_collective_bandwidth(entry, transport, span, egress_bytes,
                                    combine_transport)

    # ---- Pass 3: re-run the oracle after timing, then build the rows ---------
    rows = []
    for tokens in ladder:
        point, gate, layout = points[tokens], gates[tokens], layout_for(tokens)
        post_ok, post_rel = _oracle(
            point, layout, activations[tokens], args.seed, ep_size, jax,
        )
        max_rel = max(gate["max_rel"], post_rel)
        passed = bool(
            gate["dispatch_ok"] and post_ok and max_rel <= ep_harness.COMBINE_REL_TOL
        )
        rstats = gate["rstats"]
        pool = pools[tokens]
        # PUBLISHED latency is the device span -- the CUDA-event equivalent, and the only
        # TPU number of the same kind as the GPU SKUs'. Host wall-clock is the fallback
        # when a trace is unavailable, and it is NOT comparable to them: it carries a
        # ~450us per-call dispatch floor the GPU numbers do not have.
        measured = device_timing.get(tokens) or {}
        row_sources = {}

        def _percentiles(name):
            from_device = (measured.get(name) or {}).get("percentiles_us")
            if from_device:
                row_sources[name] = DEVICE_TIMING_SOURCE
                return from_device, len((measured[name] or {}).get(
                    "per_occurrence_us", []))
            row_sources[name] = TIMING_SOURCE
            samples = pool[name]
            return ep_harness._pcts(samples), len(samples)

        (dispatch_pcts, dispatch_n) = _percentiles("dispatch")
        (combine_pcts, combine_n) = _percentiles("combine")
        (roundtrip_pcts, roundtrip_n) = _percentiles("roundtrip")
        isolated = (
            {key: dispatch_pcts[key] + combine_pcts[key] for key in dispatch_pcts}
            if dispatch_pcts and combine_pcts else None
        )
        global_tokens = tokens * ep_size
        dispatch_bytes = ep_harness.logical_byte_provenance(
            rstats["routed_copies"], args.hidden, VALUE_BYTES, 0,
        )
        combine_bytes = ep_harness.logical_byte_provenance(
            rstats["routed_copies"], args.hidden,
        )
        rows.append({
            "components": {
                "combine": ep_harness._component(combine_pcts, combine_n),
                "dispatch": ep_harness._component(dispatch_pcts, dispatch_n),
                "isolated_sum": ep_harness._component(isolated, 0, derived=True),
                "roundtrip": ep_harness._component(roundtrip_pcts, roundtrip_n),
                # No separate staging pass exists on this transport: the permute is fused
                # into dispatch, so there is nothing to time.
                "stage": ep_harness._component(None, 0),
            },
            "correctness": {
                "max_relative_error": max_rel,
                "passed": passed,
            },
            "global_tokens": global_tokens,
            "byte_provenance": {
                "combine": combine_bytes,
                "dispatch": dispatch_bytes,
                "roundtrip": {
                    field: dispatch_bytes[field] + combine_bytes[field]
                    for field in dispatch_bytes
                },
                "stage": dict.fromkeys(dispatch_bytes, 0),
            },
            "receive": {
                "max": int(layout.recv_total.max()),
                "mean": float(layout.recv_total.mean()),
                "min": int(layout.recv_total.min()),
                "total": int(layout.recv_total.sum()),
            },
            # Device-side durations from the profiler when --xprof ran: the only
            # numbers here free of the host dispatch floor, and the only ones that
            # separate combine's transport from its scatter-add (see by_hlo).
            "device_timing": device_timing.get(tokens),
            # Provenance PER COMPONENT. The case-level measurement.timing_source cannot
            # carry this: the profiler can cover some components or points and not
            # others, and a row that fell back to host timing must never be readable as
            # a device measurement (a whole-case label once claimed device timing for
            # rows that were host-timed with a ~450us floor).
            "timing_source": dict(row_sources),
            # Host wall-clock for the same components, kept as a cross-check on the
            # device spans. Includes the per-call dispatch floor.
            "host_latency_us": {
                name: ep_harness._pcts(pool[name]) for name in COMPONENTS
            },
            # The bare collective timed the reference way -- comparable to published TPU
            # collective figures in a way the composite components are not.
            "reference_transport": reference.get(tokens),
            "routing": {key: rstats[key] for key in ep_harness._ROUTING_FIELDS},
            "token_rate_at_latency_percentile": {
                name: global_tokens / (latency * 1e-6)
                for name, latency in (roundtrip_pcts or {}).items()
            },
            "tokens_per_rank": tokens,
        })
        print(f"  T={tokens:<5} "
              f"disp p50/p99={dispatch_pcts['p50']:7.1f}/{dispatch_pcts['p99']:7.1f} "
              f"comb {combine_pcts['p50']:6.1f}/{combine_pcts['p99']:6.1f} "
              f"RT p50/p99={roundtrip_pcts['p50']:7.1f}/{roundtrip_pcts['p99']:7.1f}us "
              f"fanout={rstats['fanout_mean']:.2f} "
              f"recv[min/max]={int(layout.recv_total.min())}/"
              f"{int(layout.recv_total.max())} correct={passed}")

    all_ok = bool(rows) and all(row["correctness"]["passed"] for row in rows)
    scheduled_case = _scheduled_case(args, ep_size, nodes, ladder)
    # Recompute the case ID from the realized factors and refuse to publish under a
    # scheduled identity it does not match (same gate as ep_harness.run_sweep).
    computed = ep_harness.case_id(args.runner, scheduled_case)
    if args.case_id != computed:
        print(f"ERROR: scheduled case ID does not match realized factors: "
              f"{args.case_id} != {computed}", file=sys.stderr)
        return 2
    try:
        attempt_ordinal = _attempt_ordinal()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    source_sha = (os.environ.get("COLLECTIVEX_SOURCE_SHA")
                  or os.environ.get("GITHUB_SHA"))
    doc = {
        "version": args.version,
        "record_type": "case-attempt",
        "generated_at": _dt.datetime.now().astimezone().isoformat(),
        "identity": {
            "allocation_factors": {
                "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "run_id": os.environ.get("GITHUB_RUN_ID"),
                "source_sha": source_sha,
            },
            "attempt_ordinal": attempt_ordinal,
            "case_factors": {"case": scheduled_case, "sku": args.runner},
            "case_id": args.case_id,
        },
        "workload": {
            # One process builds the single global trace every device is sliced from, so
            # cross-rank identity is structural here rather than proven by an all-reduce
            # as it is on the GPU path.
            "cross_rank_consistent": True,
        },
        "measurement": {
            "combine_dtype": ep_jax.JaxEPTransport.combine_dtype,
            "combine_semantics": "activation-only",
            "dispatch_dtype": ep_jax.JaxEPTransport.dispatch_dtype,
            "payload_unit": "token-rank",
            "rows": rows,
            "sampling": {
                # Occurrences behind the published device percentiles. The scheduled
                # iters/trials profile is recorded too, but it governs only the host
                # sanity check now.
                "device_occurrences_per_point": args.xprof_iters,
                "iterations_per_trial": args.iters,
                "trials": args.trials,
                "warmup_iterations": args.warmup,
            },
            # NOT comparable to the GPU family's CUDA-event latencies without accounting
            # for host dispatch overhead. See bench/ep_jax.py.
            "timing_source": _case_timing_source(rows),
        },
        "implementation": {
            "fp8_consume": None,
            "kernel_generation": "jax-ragged-all-to-all",
            "name": args.backend,
            "oracle": "probe-source-identity-and-exact-rank-sum",
        },
        "topology": {
            "device_product": devices[0].device_kind,
            "gpus_per_node": args.gpus_per_node,
            "nodes": nodes,
            "placement": "packed",
            "scale_up_domain": args.scale_up_domain,
            "scale_up_transport": args.scale_up_transport,
            "scale_out_transport": args.scale_out_transport or None,
            "scope": args.scope,
            "topology_class": args.topology_class,
            "transport": args.transport,
            "world_size": ep_size,
        },
        "runtime": _runtime_info(jax, vendor, devices[0]),
        "provenance": {
            "image": os.environ.get("COLLECTIVEX_IMAGE", "") or None,
            "source_sha": source_sha,
        },
        "outcome": {
            "reasons": [] if all_ok else ["semantic correctness failed"],
            "status": "success" if all_ok else "invalid",
        },
    }
    ep_harness._write_json_atomic(args.out, doc)
    print(f"{args.backend} ep-dispatch-combine [{args.phase}/{args.mode}]: "
          f"status={doc['outcome']['status']} {len(rows)} pts -> {args.out}")
    # A captured `invalid` outcome must fail the leg, never ride as a green success.
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
