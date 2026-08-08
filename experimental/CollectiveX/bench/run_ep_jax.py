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
  * ``measurement.timing_source`` is ``xla-device-trace-span``, not CUDA events: the
    device time comes from spans in the XLA trace, reduced MAX across devices per
    occurrence. ``host-wallclock-blocked`` is the fallback, and it is opt-in
    (``--allow-host-fallback``) precisely so a host figure cannot be published as if it
    were comparable to a GPU row.
  * ``components.stage`` is ``unavailable`` ON BF16 ROWS: the permute is fused into dispatch
    and there is no conversion, so there is nothing to time. FP8 rows DO publish ``stage`` --
    the fp8->bf16 conversion, hoisted out of both combine and the chained roundtrip and
    measured on its own, which is what makes ``roundtrip + stage`` reconstruct the
    mismatched-config cost.
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
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import numpy as np  # noqa: E402

import ep_harness  # noqa: E402
import ep_jax  # noqa: E402
import routing_np  # noqa: E402
import xprof  # noqa: E402


TIMING_SOURCE = "host-wallclock-blocked"
COMPONENTS = ("dispatch", "combine", "roundtrip")
# fp8 additionally measures `stage`: the fp8->bf16 conversion, hoisted out of combine and
# out of the chained roundtrip and timed on its own. Same component the GPU harness puts
# that work in (for deepep-v2 and uccl-ep, `stage` IS the fp8 conversion), which is what
# keeps `dequant roundtrip ~= roundtrip + stage` derivable from a published row. BF16 has no
# conversion at all, so its `stage` stays unavailable.
STAGE = "stage"


def components_for(precision: str) -> tuple:
    return COMPONENTS + ((STAGE,) if precision == "fp8" else ())
# The reference program is traced alongside them but is NOT a published component: it is
# the cross-check against the host-timed reference figure, and the gate below deliberately
# does not require it (a failed reference capture must not fail a case).
REFERENCE = "reference"
TRACED = COMPONENTS + (REFERENCE,)


def traced_for(precision: str) -> tuple:
    return components_for(precision) + (REFERENCE,)


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
            (ep_jax.transport_scope(component, tokens_per_rank),
             ep_jax.permute_scope(component, tokens_per_rank),
             ep_jax.quantize_scope(component, tokens_per_rank)) + ep_jax.TRACED_HLO)


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
                                combine_transport=None, bf16_egress_bytes=None) -> dict:
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
    # Combine and the reference span are BF16 exchanges on BOTH precisions, so they are
    # billed at the BF16 basis. Passing the dispatch basis here understated them ~1.94x
    # under fp8 while looking entirely plausible.
    reference_bytes = bf16_egress_bytes if bf16_egress_bytes is not None else egress_bytes
    entry["collective_bandwidth_gbps_per_device_combine"] = (
        reference_bytes / (combine_transport * 1e-6) / 1e9 if combine_transport else None
    )
    entry["collective_bandwidth_gbps_per_device_span"] = (
        reference_bytes / (span * 1e-6) / 1e9 if span else None
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
#: The host sanity loop's ACTUAL counts. Named constants because the artifact publishes them:
#: it previously published the case's scheduled `iters`/`trials`/`warmup` profile (8/256/32)
#: beside a comment saying that profile "governs only the host sanity check", which was false
#: -- the loop has always hardcoded these, and the scheduled numbers govern nothing here. A
#: reader would have concluded the host cross-check rested on 2048 samples; it rests on 10.
HOST_WARMUP_TRIES = 5
HOST_NUM_RUNS = 10


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


# Two e4m3 grid steps. Chosen against measurement rather than argued: on the real
# activations, under this exact metric (relative, floored at COMBINE_MAG_FLOOR),
#
#   legitimate one-grid-step divergence            0.125   <- must pass, and does
#   swap two ADJACENT columns                      2.000   <- tightest corruption tried
#   zero the last 8 columns                        1.000
#   scale everything past the ID prefix by -0.37   1.370
#   shift the row by one column                   50.781
#   shift the row by one 128-block                93.750
#
# So the band sits 2x above the largest legitimate divergence and 8x below the subtlest
# corruption -- a column swap, which is as close to a no-op as a real stride bug gets.
TIMED_VALUE_REL_TOL = 2 * 2.0 ** -3


def _host_dequantize(values, scales, hidden: int) -> np.ndarray:
    """values * per-128-block scale, on the host. The inverse the device applies."""
    flat = np.asarray(values, dtype=np.float32)
    blocks = np.asarray(scales, dtype=np.float32)
    view = flat.reshape(flat.shape[0], blocks.shape[-1], ep_jax.QUANT_BLOCK)
    return (view * blocks[:, :, None]).reshape(flat.shape[0], hidden)


def _check_timed_values_match(received, oracle_values, oracle_scales,
                              layout: ep_jax.Layout, ep_size: int) -> bool:
    """The TIMED program delivered the same VALUES the oracle program did.

    `_check_timed_scales_match` covers only the scales, which leaves a real corruption
    class green: scramble the timed program's delivered values ANYWHERE PAST THE SOURCE-ID
    PREFIX and every other check still passes. Demonstrated against the real oracle --
    multiplying `received[:, :, SOURCE_ID_COLUMNS:]` by -0.37 in the timed program alone
    returned ok=True, max_rel=0.0. The ID decode reads only prefix signs, placement sorts
    IDs, the permutation check reads the oracle program's own output, and the combine
    expectation is built FROM the corrupted rows, so both sides carry the corruption and
    agree exactly. A stride or partial-row bug in the values exchange looks like this.

    Compared with TOLERANCE, not exactly: unlike the scales, `view * (448/amax)` is
    reassociable, so two compilations may legitimately differ by an e4m3 grid step.
    `received` is the device's bf16 dequantization of the timed output; the right-hand side
    is the host's f32 dequantization of the oracle output. bf16 rounding is ~2^-8 relative,
    far inside the tolerance.
    """
    hidden = received.shape[-1]
    values = _by_rank(oracle_values, ep_size)
    scales = _by_rank(oracle_scales, ep_size)
    for rank in range(ep_size):
        rows = int(layout.recv_total[rank])
        if not rows:
            continue
        want = _host_dequantize(values[rank, :rows], scales[rank, :rows], hidden)
        got = np.asarray(received[rank, :rows], dtype=np.float32)
        denominator = np.maximum(np.abs(want), ep_harness.COMBINE_MAG_FLOOR)
        relative = np.abs(got - want) / denominator
        worst = float(relative.max())
        if worst > TIMED_VALUE_REL_TOL:
            flat = int(np.argmax(relative))
            row, column = divmod(flat, hidden)
            print(f"[oracle] TIMED program's values differ from the oracle program's at "
                  f"rank={rank} row={row} col={column}: rel={worst:.6f} > "
                  f"{TIMED_VALUE_REL_TOL:.6f} (got {got[row, column]:.6f}, want "
                  f"{want[row, column]:.6f}). The published fp8 latency is not the program "
                  f"the byte check validated.", file=sys.stderr)
            return False
    return True


def _ulp_distance(left, right) -> np.ndarray:
    """Distance in representable f32 steps. Both inputs are positive here (amax/448)."""
    a = np.asarray(left, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(right, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)


def _check_timed_scales_match(timed_scales, oracle_scales, layout: ep_jax.Layout,
                              ep_size: int) -> bool:
    """The TIMED program delivered the same block scales the ORACLE program did.

    `_check_dispatch_permutation` compares arrived against staged inside the oracle
    program. That program shares `_quantized_dispatch_local` with the timed one but is
    still its own compilation, so on its own it would leave the timed program's scales
    exchange unchecked -- and a corrupted scales exchange there is invisible to everything
    else: the source ID lives in the SIGN of the values, and placement is checked on the
    values too.

    Numerics cannot police it. Measured on this activation generator, the per-128-block
    amax takes only FOUR distinct values across 229,376 blocks (1.953125 to 2.0), so every
    possible scale mix-up lands within 2.4% -- inside e4m3's own 12.5% grid step and inside
    COMBINE_REL_TOL. Any tolerance loose enough to admit correct fp8 admits the corruption
    too. So this compares EXACTLY.

    Compared to within ONE f32 ulp rather than bit-exactly. A scale is
    `clip(max(|block|), 1e-4) / 448`: the max is associative and exact for floats, so
    reduction order cannot change it, and both programs trace the same Python. But 448 is
    not a power of two, and a divide-by-constant may lower through a reciprocal multiply --
    a context-free rewrite that should fire identically in both compilations, though not
    one this can prove. Measured: an EXACT comparison passed on all 14 fp8 ladder points in
    run 30873922946, so the two compilations do agree bitwise today. The 1-ulp band is kept
    anyway because it retires the risk across future XLA versions at zero cost in teeth: the
    smallest
    separation between two DIFFERENT amax values on this lattice is ~0.4%, which is tens of
    thousands of ulps at these magnitudes. Mix-ups between blocks whose amax is identical
    are invisible to any comparison, exact or tolerant, and harmless -- identical scales
    dequantize identically.
    """
    timed = _by_rank(timed_scales, ep_size)
    oracle = _by_rank(oracle_scales, ep_size)
    for rank in range(ep_size):
        rows = int(layout.recv_total[rank])
        if not rows:
            continue
        distance = _ulp_distance(timed[rank, :rows], oracle[rank, :rows])
        if distance.max() > 1:
            differing = int((distance > 1).sum())
            print(f"[oracle] TIMED program's scales differ from the oracle program's at "
                  f"rank={rank}: {differing}/{distance.size} blocks, worst "
                  f"{int(distance.max())} ulp. One ulp would be compiler rounding; this is "
                  f"not. The published fp8 latency is not the program the byte check "
                  f"validated.", file=sys.stderr)
            return False
    return True


def _expected_combine_from_received(received: np.ndarray, layout: ep_jax.Layout,
                                    rank: int, seed: int) -> np.ndarray:
    """The rank-sum combine SHOULD return, built from what dispatch actually delivered.

    Not from a second quantize of x. That was the defect: `codec_activations` was a separate
    compiled program, and XLA rounds this lattice's e4m3 midpoints differently there than
    inside the fused dispatch -- measured at rank 0 token 26 col 52, the dispatch produced
    161/256 where the standalone codec produced 152/256, one full e4m3 grid step, 5.92%.
    Any expectation re-derived through a second compilation inherits that coin flip.

    So sum the rows that were really received: for each destination, the chunk this rank
    sent it maps back to this rank's tokens. That validates combine's return transport and
    its scatter-add -- which is combine's job -- while dispatch's own fidelity is validated
    separately, and exactly, by the staged-vs-arrived permutation check.

    Each row is attributed to the token its PAYLOAD claims to be, decoded from the source
    ID in the sign bits, NOT to the token at its position in send_index. Those two agree on
    a correct transport and disagree exactly when a row is out of place, which is the whole
    point: combine pairs positionally, so anchoring the expectation positionally too makes
    a reordering cancel out of both sides. Confirmed by replaying a swap of two rows inside
    one (dst, src) chunk offline -- the positional expectation scored it 0.000000 while the
    ID-anchored one scores it 2.046875. Nothing else in the fp8 path catches that swap: the
    permutation check sees it identically on both sides when the permute staged it, step 1
    decodes the ID from the moved payload so ID and data stay consistent, and step 3 sorts.
    Under bf16 the pristine expectation caught it; fp8 gave up that anchor and this
    restores it. The ID survives quantization because it is carried in the SIGN.

    Returns None when the delivered rows cannot be attributed at all -- a prefix that will
    not decode, or an ID belonging to another rank. Both mean the transport is broken, and
    both must read as a FAILED oracle rather than an exception: `_combine_error` runs before
    `_check_dispatch` (whose own decode is guarded), so raising here would kill the process
    and take the localising diagnostic with it, on exactly the runs that need it.
    """
    expected = np.zeros((layout.tokens_per_rank, received.shape[-1]), dtype=np.float32)
    for destination in range(layout.ep_size):
        size = int(layout.recv_sizes[destination, rank])
        if not size:
            continue
        here = int(layout.recv_offsets[destination, rank])
        rows = received[destination, here:here + size].astype(np.float32)
        try:
            tokens = routing_np.decode_source_ids(rows, seed).astype(np.int64)
        except ValueError as exc:
            print(f"[oracle] combine expectation cannot decode the rows delivered to "
                  f"dst={destination} for rank={rank}: {exc}", file=sys.stderr)
            return None
        local = tokens - rank * layout.tokens_per_rank
        # Bounds, NOT clipping: a row carrying another rank's token means the transport
        # misdelivered across ranks. A negative index would wrap silently under np.add.at
        # and quietly corrupt the expectation into agreeing with the corruption.
        if local.min() < 0 or local.max() >= layout.tokens_per_rank:
            print(f"[oracle] combine expectation: dst={destination} delivered a token "
                  f"outside rank {rank}'s range [0, {layout.tokens_per_rank}): "
                  f"[{int(local.min())}, {int(local.max())}]", file=sys.stderr)
            return None
        np.add.at(expected, local, rows)
    return expected


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


# Minimum share of the dispatch's device work a genuine chained roundtrip must add on top
# of the standalone combine. Measured separation is 0.659..0.816 genuine vs -0.021..0.020
# deleted, so this sits >30x clear of both.
ROUNDTRIP_DISPATCH_MIN_SHARE = 0.25
# The per-case wall budget the launcher enforces (COLLX_RUN_TIMEOUT). The chain runs twice
# per ladder point on top of the fresh-entry components, so `capture_s` is published
# against this: a raised --chain-iters should be a decision, not a timeout discovered at
# the last and largest point of a 45-minute case.
CHAIN_CAPTURE_BUDGET_S = int(os.environ.get("COLLX_RUN_TIMEOUT", "2700"))


def gather_chain_periods(chain: dict, expected_devices: int):
    """Fold the other host's devices into the period matrix before it is reduced.

    At EP16 each process's profiler covers only its own 8 devices, and the missing half is
    precisely where inter-host stragglers live on a slice that spans hosts -- a local median
    would report the faster half's rate and an understated spread.

    The gather is UNCONDITIONAL and fixed-shape. Both hosts reach it exactly once per point
    whatever their trace held, because a collective one host skips is a deadlock, not a
    missing field: a host with no usable series contributes NaN columns and the other host's
    devices still count. That is the same rule `to_host()` documents.
    """
    if _PROCESS_COUNT[0] == 1:
        # Still stamp it. `null` reads as "nobody checked"; 8 of 8 is the honest EP8 answer
        # and makes the EP16 case (16 expected, fewer gathered) legible by contrast.
        chain["devices_expected"] = expected_devices
        return chain
    import numpy as _np  # noqa: PLC0415
    from jax.experimental import multihost_utils  # noqa: PLC0415

    local = chain.get("per_device_periods") or []
    width = max(1, expected_devices // max(1, _PROCESS_COUNT[0]))
    length = max((len(series) for series in local), default=0)
    length = int(multihost_utils.process_allgather(_np.array([length])).max())
    if length == 0:
        chain["devices_expected"] = expected_devices
        return chain
    block = _np.full((length, width), _np.nan, dtype=_np.float64)
    for column, series in enumerate(local[:width]):
        block[:len(series[:length]), column] = series[:length]
    gathered = _np.asarray(multihost_utils.process_allgather(block))
    # (hosts, length, width) -> per-device columns across every host
    matrix = gathered.reshape(-1, length, width).transpose(1, 0, 2).reshape(length, -1)
    # float(), not the numpy scalar. `process_allgather` round-trips through JAX, which
    # with x64 disabled returns float32 -- and a numpy float32 raises out of json.dumps at
    # the very END of the run, after every timing pass has been paid for. Measured on run
    # 31147054940: both EP16 bf16 cases died with "Object of type float32 is not JSON
    # serializable" while the fp8 cases, which are not chained, wrote fine. That is the same
    # dead-process-at-the-last-step failure the allow_nan=False guard was added to remove.
    #
    # The float32 round-trip itself is acceptable and stays: spacing at a 25,000us period is
    # ~0.002us, three orders below the smallest quantity published from these series (a
    # -1.6us settle drift). Reporting more digits than that would be the real error.
    columns = [[float(v) for v in matrix[:, c]] for c in range(matrix.shape[1])
               if not _np.isnan(matrix[:, c]).all()]
    if not columns:
        chain["devices_expected"] = expected_devices
        return chain
    chain.update(xprof.chain_stats(columns))
    chain["devices_expected"] = expected_devices
    chain["gathered_across_hosts"] = True
    return chain


def _local_rows(jax, array):
    """This process's shards, in device order, flattened to (rows, hidden).

    `device_get` on a multi-host global array raises -- it spans devices this process
    cannot address. Measured on run 31156743315: every EP16 bf16 row came back
    `chain_regime_passed=False` with "Fetching value for jax.Array that spans
    non-addressable devices", i.e. the ORACLE could not run, on a transport whose drained
    oracle had just scored 0.0. Reading the addressable shards is the supported route, and
    sorting them by device id makes the two arrays line up rank for rank even though one
    is (ep, tokens, hidden) and the other (ep*tokens, hidden).
    """
    import numpy as _np  # noqa: PLC0415

    shards = sorted(array.addressable_shards, key=lambda shard: shard.device.id)
    if not shards:
        return None
    blocks = [_np.asarray(jax.device_get(shard.data), dtype=_np.float32) for shard in shards]
    return _np.concatenate([block.reshape(-1, block.shape[-1]) for block in blocks], axis=0)


def chain_identity(jax, produced, point) -> dict:
    """Score the chained program's OUTPUT against its input. Exactness is the point.

    With combine's unweighted rank-sum divided by each token's copy count, one pair is the
    identity BITWISE -- the fp32 sum of `d` copies of a bf16 value is exact, and IEEE
    division is correctly rounded, so `(d*v)/d` is `v`. That holds for any iteration count,
    which is what makes it usable on the 64-iteration program we actually time.

    Before this, the chain had no oracle at all, and could not have had a useful one: the
    carry grew by `d_t` per pair and saturated to +/-inf by iteration ~43, so any check
    against an expected value would have compared inf to inf and passed for a chain that
    had dropped rows, mis-routed chunks, or reversed offsets.

    Three-valued. `ok=True` passed; `ok=False` means the oracle RAN and disagreed, which
    fails the row; `ok=None` means it could not be evaluated, which withholds the period
    but does not condemn the drained components measured in the same case. Collapsing the
    last two cost 10 good EP16 rows once already.

    Every process reaches the cross-host reduction, unconditionally and at fixed shape:
    a collective one host skips is a deadlock, not a missing field.
    """
    import numpy as _np  # noqa: PLC0415

    try:
        got = _local_rows(jax, produced)
        want = _local_rows(jax, point.x)
        if got is None or want is None or got.shape != want.shape:
            local = {"ok": None, "max_rel": None,
                     "reason": f"chain oracle could not align shards: "
                               f"{None if got is None else got.shape} vs "
                               f"{None if want is None else want.shape}"}
        elif not _np.isfinite(got).all():
            bad = int((~_np.isfinite(got)).sum())
            local = {"ok": False, "max_rel": None,
                     "reason": f"{bad} non-finite element(s) in the chain output"}
        else:
            scale = float(_np.abs(want).max()) or 1.0
            local = {"ok": bool((got == want).all()),
                     "max_rel": float(_np.abs(got - want).max() / scale),
                     "reason": None}
    except Exception as exc:
        local = {"ok": None, "max_rel": None,
                 "reason": f"chain oracle could not run: {exc!r}"}
    if _PROCESS_COUNT[0] == 1:
        if local["ok"] is False and local["reason"] is None:
            local["reason"] = "chain output is not bitwise its input"
        return local
    from jax.experimental import multihost_utils  # noqa: PLC0415

    # [evaluated?, passed?, max_rel] -- fixed shape, every host, every point.
    flags = _np.array([0.0 if local["ok"] is None else 1.0,
                       1.0 if local["ok"] else 0.0,
                       -1.0 if local["max_rel"] is None else local["max_rel"]],
                      dtype=_np.float64)
    merged = _np.asarray(multihost_utils.process_allgather(flags)).reshape(-1, 3)
    if not merged[:, 0].all():
        return {"ok": None, "max_rel": None,
                "reason": f"chain oracle unevaluated on {int((merged[:, 0] == 0).sum())} "
                          f"of {merged.shape[0]} host(s): {local['reason']}"}
    scored = merged[:, 2][merged[:, 2] >= 0]
    return {"ok": bool(merged[:, 1].all()),
            "max_rel": float(scored.max()) if scored.size else None,
            "reason": (None if merged[:, 1].all()
                       else "chain output is not bitwise its input on at least one host")}


def _chain_anchor(trace_dir: str, marker: str):
    """The collective op to anchor periods on, chosen from what the trace actually holds.

    Not a constant, because the name is a compiler artifact and differs by precision:
    measured at T=8192, bf16 dispatch/combine and fp8 combine all carry a hyphenated
    `ragged-all-to-all...call-done`, while fp8 dispatch carries ONLY underscored
    `ragged_all_to_all.38`/`.24`. A hardcoded hyphenated anchor finds nothing under fp8 and
    the chained fields would go unavailable on every fp8 row while appearing to work.

    Picks the most frequent matching op name so the anchor is one homogeneous op -- a chain
    body contains several collectives and consecutive-start deltas across a mixture would
    not be a period at all.
    """
    trace = xprof.find_trace(trace_dir)
    if trace is None:
        return None
    try:
        events = xprof.load_events(trace)
    except (OSError, ValueError, EOFError):
        return None
    counts: dict[str, int] = {}
    for event in events:
        name = str(event.get("name", ""))
        if not any(family in name for family in xprof.CHAIN_ANCHOR_FAMILIES):
            continue
        if not xprof.marker_in(str((event.get("args") or {}).get("tf_op", "")), marker):
            continue
        counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else None


def roundtrip_contains_dispatch(roundtrip: dict | None, combine: dict | None,
                                dispatch: dict | None):
    """Did the chained roundtrip actually run its dispatch? None when unjudgeable.

    It is not enough to write the program: under fp8 nothing in the roundtrip CONSUMES the
    dispatch result, so XLA is free to delete it, and it did. Run 30890386745 published a
    roundtrip of 16,511.5us against combine's 16,511.4 -- a lone combine wearing the
    roundtrip's label, green, correct=True, and 8.5ms too fast. The fix (returning the
    dispatch results, so they cannot be elided) is not something an offline test can confirm
    held; only a trace can.

    So compare total device work: a roundtrip that ran its dispatch costs materially more
    than the standalone combine, by a fair fraction of the dispatch. The threshold is read
    off the data rather than guessed. Over all 28 ladder points of run 30890386745, where
    bf16 chains genuinely and fp8's dispatch was deleted:

        (roundtrip - combine) / dispatch     bf16  0.659 .. 0.816     (genuine)
                                             fp8  -0.021 .. 0.020    (deleted)

    0.25 sits more than 30x clear of both clusters. It is well under 1.0 because chaining
    really does amortise -- bf16's genuine roundtrip runs ~0.70 of dispatch+combine, and a
    stricter bound would fail honest rows.

    Comparing op NAMES instead does not work, and an earlier version of this that did was
    wrong on exactly the run it was written for: XLA's numeric suffixes differ between
    compilations (`ragged_all_to_all.8` in the roundtrip against `.6` in combine), so a
    set-difference is never empty and the check passed a roundtrip whose inventory was
    otherwise byte-for-byte combine's.
    """
    def total(parsed):
        entries = (parsed or {}).get("op_inventory")
        if not entries:
            return None
        return sum(entry.get("per_iteration_us") or 0.0 for entry in entries)

    here, combined, dispatch_cost = total(roundtrip), total(combine), total(dispatch)
    if not (here and combined and dispatch_cost):
        return None
    return (here - combined) / dispatch_cost >= ROUNDTRIP_DISPATCH_MIN_SHARE


def row_passed(dispatch_ok: bool, post_ok: bool, max_rel: float,
               roundtrip_intact) -> bool:
    """A row passes only if the transport is correct AND the roundtrip measured itself.

    `roundtrip_intact is False` means XLA deleted the chained dispatch, so the row's
    headline `roundtrip` is a lone combine. That is not a transport correctness failure --
    dispatch and combine both verified -- but a row whose published number measures a
    different program must not read as passed. None (no trace to judge) does not fail it.
    """
    return bool(
        dispatch_ok and post_ok and max_rel <= ep_harness.COMBINE_REL_TOL
        and roundtrip_intact is not False
    )


def fp8_provenance(precision: str) -> dict:
    """How this row consumed FP8, and what `roundtrip` therefore means.

    `native` matches the GPU default and this workload: deepseek-v3 block-fp8 experts
    consume the dispatched fp8 + per-128-block scales DIRECTLY and emit BF16, so no
    standalone conversion sits between the two collectives. SGLang's DeepEP dispatcher has
    no dequant at all; vLLM skips it when the expert's block shape matches DeepEP's 128,
    which this workload's does. `dequant` is vLLM's fallback for a quant-format MISMATCH --
    a real path, but not this one's, and the GPU side keeps it strictly as a verification
    hatch.

    This probe published `dequant` for two runs. That charged the conversion to the chained
    roundtrip, comparing fp8 and bf16 through structurally different pipelines, and made
    fp8's roundtrip look WORSE than bf16's while its dispatch was 1.53x faster. The GPU
    corpus measured the same inversion in 39 of 51 comparisons before hoisting its stage.

    `stage_excluded_from_roundtrip` is now True on BOTH precisions, as on every GPU row:
    roundtrip means dispatch->combine, transport only. The conversion is measured as the
    `stage` component instead, which keeps `dequant roundtrip ~= roundtrip + stage`
    derivable -- in that direction only.
    """
    return {
        "fp8_consume": "native" if precision == "fp8" else None,
        # bf16 chains today; fp8 does not, because its combine needs BF16 and a chain body
        # would carry the conversion inside the loop -- the period would then include work
        # the `native` contract says production does not do standalone.
        "chained_period": precision != "fp8",
        # Constant on this backend BY CONSTRUCTION: an XLA SPMD program sequences collectives
        # identically on every device, so there is no valve to open. Emitted for schema
        # parity so the frontend badge needs no TPU special case.
        "chain_barrier": False,
        "stage_excluded_from_roundtrip": True,
        "fp8_dequant_inside_roundtrip": False,
    }


def _sampling_provenance(args) -> dict:
    """What the published numbers actually rest on.

    `device_*` govern the headline: the device percentiles come from that many profiled
    occurrences per point. `host_*` govern the per-row `host_latency_us` cross-check, and are
    the loop's real counts rather than the case's scheduled profile.

    `scheduled_*` is the profile the matrix asked for. On this backend it governs NOTHING --
    it is recorded so a reader can see what was scheduled and what was measured are different
    questions, instead of inferring the host figures rest on trials x iterations samples.
    """
    return {
        "device_occurrences_per_point": args.xprof_iters,
        # The GPU family emits this key too (ep_harness: iters x trials), so a consumer that
        # indexes it by name still finds it here -- with the value that is TRUE for this
        # backend rather than a scheduled figure nothing used.
        "samples_per_component": HOST_NUM_RUNS,
        "host_samples_per_point": HOST_NUM_RUNS,
        "host_warmup_per_point": HOST_WARMUP_TRIES,
        "scheduled_iterations_per_trial": args.iters,
        "scheduled_trials": args.trials,
        "scheduled_warmup_iterations": args.warmup,
        "scheduled_profile_governs": None,
        "chain_iterations_per_trial": args.chain_iters,
        # 1 on this backend, and that is a fact rather than a placeholder: ONE capture holds
        # the entire chain, so there is no second trial to average over.
        "chain_trials": 1,
        "chain_drop": args.chain_drop,
        "chain_governs": "pair_period, chain_floor_us, chain_health",
    }


CHAIN_TIMING_SOURCE = "xla-device-trace-chain"


def scalar_component(value, origin):
    """One scalar in the family's component shape.

    The GPU harness publishes every `chain_health` field through `_component`, so a
    consumer reads `["percentiles_us"]["p50"]` uniformly. These are single reduced scalars
    rather than distributions, so every percentile carries the same value -- which is
    honest (the reduction produced one number) and keeps the access path identical.
    """
    if value is None:
        return {"availability": "unavailable", "origin": None,
                "percentiles_us": None, "sample_count": 0}
    return {"availability": "measured", "origin": origin,
            "percentiles_us": {key: value for key in ("p50", "p90", "p95", "p99")},
            "sample_count": 1}


def _chain_fields(chain: dict | None) -> dict:
    """The chained-regime row fields, in the GPU family's shapes.

    Additive and presence-keyed: a row without a usable chain still publishes the blocks, as
    `unavailable` with the reason, so a consumer never has to distinguish "absent" from
    "failed". The origin strings are the GPU's verbatim -- `chained-median` and
    `chained-cross-rank-min` -- even though the reduction here is cross-DEVICE: the string
    names the statistic and the frontend treats it as opaque, so forking it would fork the
    consumer for no gain.

    Chained per-op medians and MAXes are absent by design, not omission. In a free-running
    chain the inter-device wait parks in whichever op window absorbs it, bistably, so only
    the period and the floors are stable.
    """
    def block(reason):
        # Component-SHAPED, not a bare None. The GPU family's unavailable blocks carry
        # availability/origin/percentiles_us/sample_count, and a consumer indexing
        # `.percentiles_us` must not have to special-case TPU by finding null there.
        return {"availability": "unavailable", "origin": None, "percentiles_us": None,
                "sample_count": 0, "reason": reason}

    if not chain or chain.get("availability") != "measured":
        reason = (chain or {}).get("reason", "not captured")
        return {
            "pair_period": block(reason),
            "chain_floor_us": {"dispatch": block(reason), "combine": block(reason)},
            "chain_health": {"pair_spread_us": scalar_component(None, None),
                             "interpair_gap_us": scalar_component(None, None),
                             "settle_drift_us": scalar_component(None, None),
                             "devices": None,
                             "devices_expected": (chain or {}).get("devices_expected"),
                             "gathered_across_hosts": False,
                             # Same keys in both states: a consumer must not have to
                             # branch on whether the chain ran to know the shape.
                             "renorm_us": scalar_component(None, None),
                             "op_inventory": None, "capture_s": None,
                             "capture_budget_s": None,
                             "anchor": {"period": None, "floor_dispatch": None,
                                        "floor_combine": None, "phase": None,
                                        "phase_note": None}},
        }

    floors = chain.get("floors") or {}

    def floor_block(side):
        got = floors.get(side) or {}
        if got.get("availability") != "measured":
            return block(got.get("reason", "no per-direction anchor"))
        return {"availability": "measured", "origin": got.get("origin"),
                "percentiles_us": got.get("percentiles_us"),
                "sample_count": got.get("sample_count", 0)}

    return {
        "pair_period": {
            "availability": "measured",
            "origin": chain["origin"],
            "percentiles_us": chain["percentiles_us"],
            "sample_count": chain["sample_count"],
        },
        # Measured PER DIRECTION. An earlier cut published one series under both names --
        # byte-identical p50/p90/p95/p99 for dispatch and combine -- which invited a reader
        # to sum a number with itself. Where the directions cannot be told apart, BOTH are
        # unavailable with the reason; null here means "did not measure", never "same as the
        # other one".
        "chain_floor_us": {"dispatch": floor_block("dispatch"),
                           "combine": floor_block("combine")},
        # GPU parity: these three are `_component` blocks there, not bare scalars, because
        # they share the block shape while being differently-reduced statistics. A consumer
        # reading `["percentiles_us"]["p50"]` uniformly across the family got a float here.
        "chain_health": {
            "pair_spread_us": scalar_component(chain.get("pair_spread_us"),
                                               "chained-cross-device-spread"),
            # Period minus in-iteration work: near zero == genuinely free-running.
            "interpair_gap_us": scalar_component(chain.get("interpair_gap_us"),
                                                 "chained-start-to-start-minus-window"),
            # Late-half minus early-half of the period series: defends or indicts `drop`.
            "settle_drift_us": scalar_component(chain.get("settle_drift_us"),
                                                "chained-signed-max-magnitude"),
            "devices": chain.get("devices"),
            "devices_expected": chain.get("devices_expected"),
            "gathered_across_hosts": bool(chain.get("gathered_across_hosts")),
            # The renormalisation is work production does not do, and it lands between
            # combine and the next dispatch -- INSIDE the period, since the period is
            # anchor-start to anchor-start. Published rather than asserted to be free.
            "renorm_us": scalar_component(chain.get("renorm_us"), "chained-per-iteration"),
            # What the chain body contains, per op. fp8's exclusion from chaining rests on
            # a claim about this inventory; publishing it makes the claim checkable.
            "op_inventory": chain.get("op_inventory"),
            # Capture cost against the launcher's own per-case budget, so raising
            # --chain-iters is a decision rather than a timeout found at the last point.
            "capture_s": chain.get("capture_s"),
            "capture_budget_s": chain.get("capture_budget_s"),
            # WHICH op each number came from. Three different ops in general.
            "anchor": {
                "period": chain.get("anchor"),
                "floor_dispatch": (floors.get("dispatch") or {}).get("anchor"),
                "floor_combine": (floors.get("combine") or {}).get("anchor"),
                # Where the combine anchor starts within the period, as a fraction. Two
                # names alternating is ALSO what two ops of one direction look like
                # (d1a, d1b, d2a, d2b), so alternation cannot separate the cases and this
                # can: ~0.5 is a real pair, ~0 or ~1 means both floors are one direction.
                "phase": chain.get("anchor_phase"),
                "phase_note": chain.get("anchor_phase_note"),
            },
        },
    }


def _correctness(max_rel: float, passed: bool, chain_regime=None) -> dict:
    """The artifact's correctness block, JSON-safe by construction.

    `_combine_error` returns inf when the delivered rows cannot be attributed at all. The
    artifact is written with `allow_nan=False`, so publishing that raw raises out of
    json.dumps at the very END of the run -- after every timing pass has been paid for --
    and the process dies with a traceback instead of writing a red row. That is the same
    dead-process failure the guarded decode was added to remove, merely relocated by five
    minutes, so the number is nulled here and `passed` carries the verdict. `unattributable`
    says WHY the number is missing, rather than leaving a reader to read null as
    "not measured".
    """
    # The chained regime is gated separately, as the GPU family does: the drained passes
    # only ever check one pair entered from idle, so a transport that corrupts under
    # free-running pairs would present as the fastest in the suite while every drained
    # check stayed green. Tri-state: None means the backend was not chained (fp8 here),
    # never "it passed".
    # None survives as None: "the oracle could not be evaluated" is not "the chain is
    # wrong", and treating it as failure invalidated 10 EP16 rows whose drained oracle had
    # scored 0.0 -- because `device_get` cannot read a multi-host array, not because
    # anything was corrupt.
    chain_regime = None if chain_regime is None else chain_regime.get("ok")
    if chain_regime is False:
        passed = False
    finite = bool(np.isfinite(max_rel))
    return {
        "max_relative_error": float(max_rel) if finite else None,
        "unattributable": not finite,
        "chain_regime_passed": chain_regime,
        "passed": bool(passed),
    }


def _check_dispatch(received: np.ndarray, layout: ep_jax.Layout, seed: int,
                    ep_size: int, verify_values=None) -> bool:
    """Every dispatched copy carries the bytes of the token it claims to be, and lands in
    the slot the exchange plan assigned it.

    Three steps, and only the middle one depends on precision:
      1. decode the source ID from the payload (the ID is the SIGN of the first columns);
      2. check the payload really is that token's activation row;
      3. check that ID landed where the exchange plan says it should.

    Step 3 is what polices MISROUTING, and it anchors against the layout, not against the
    payload -- so a transport that delivers self-consistent but wrong rows still fails.
    `verify_values` replaces step 2 when the wire was lossy: under fp8 an exact comparison
    against the un-quantized source would fail on a correct transport. It must NOT be used
    to skip step 3; an earlier draft of this ran a separate fp8 checker that did only step
    2, so misrouting would have published green rows once step 2 was made to pass."""
    for rank in range(ep_size):
        rows = int(layout.recv_total[rank])
        if rows == 0:
            continue
        payload = received[rank, :rows].astype(np.float32)
        try:
            source = routing_np.decode_source_ids(payload, seed)
        except ValueError as exc:
            prefix = payload[:, :routing_np.SOURCE_ID_COLUMNS]
            print(f"[oracle] decode FAILED rank={rank} rows={rows}: {exc}; "
                  f"|prefix| min={np.abs(prefix).min():.4f} "
                  f"max={np.abs(prefix).max():.4f} "
                  f"below_guard={(np.abs(prefix) < 0.25).sum()}", file=sys.stderr)
            return False
        expected = routing_np.activations_for_source_ids(source, payload.shape[1], seed)
        if verify_values is None:
            if not np.array_equal(payload, expected):
                return False
        elif not verify_values(expected, rank, rows):
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
                print(f"[oracle] PLACEMENT differs rank={rank} src={src} size={size} "
                      f"start={start}", file=sys.stderr)
                return False
    return True


def _combine_error(combined: np.ndarray, activations: np.ndarray,
                   layout: ep_jax.Layout, ep_size: int, label: str = "",
                   received: np.ndarray | None = None,
                   seed: int | None = None) -> float:
    """Max magnitude-floored relative error against the exact expected combine.

    `received`, when given (fp8), replaces the activation-derived expectation with one
    built from the delivered rows -- see _expected_combine_from_received.
    """
    worst, worst_at = 0.0, None
    for rank in range(ep_size):
        expected = (_expected_combine(activations[rank], layout, rank)
                    if received is None
                    else _expected_combine_from_received(received, layout, rank, seed))
        if expected is None:
            # Unattributable rows: the transport is broken. inf reads naturally through
            # every downstream comparison and keeps the diagnostic above on stderr.
            return float("inf")
        got = combined[rank].astype(np.float32)
        denominator = np.maximum(np.abs(expected), ep_harness.COMBINE_MAG_FLOOR)
        relative = np.abs(got - expected) / denominator
        here = float(np.max(relative))
        if here > worst:
            worst = here
            flat = int(np.argmax(relative))
            token, column = divmod(flat, expected.shape[1])
            counts = np.bincount(
                layout.send_index[rank, :int(layout.send_total[rank])],
                minlength=layout.tokens_per_rank,
            )
            worst_at = (rank, token, column, float(expected[token, column]),
                        float(got[token, column]),
                        float(activations[rank][token, column]),
                        int(counts[token]))
    if label and worst_at is not None:
        rank, token, column, want, got_v, source, fanout = worst_at
        print(f"[oracle] {label} worst rel={worst:.6f} at rank={rank} token={token} "
              f"col={column}: expected={want!r} got={got_v!r} "
              f"activation={source!r} fanout={fanout} "
              f"(got/activation={got_v / source if source else float('nan'):.4f})",
              file=sys.stderr)
    return worst


def _ladder_within_budget(ladder, layout_for, hidden: int, budget_bytes: int,
                          value_bytes: int = 2, scale_bytes_per_copy: int = 0,
                          holds_dequantized: bool = False):
    """Split the ladder into points that fit the per-device payload budget and those that
    do not. Oversized points are REPORTED as dropped, never silently truncated -- the same
    contract EPBackend.buffer_cap gives the GPU backends.

    fp8 does NOT simply halve the footprint. It holds the 1-byte values, the FP32 block
    scales, AND the dequantized BF16 that combine consumes, all at once -- so billing it at
    a single `value_bytes` under-counts and the budget would admit a point that OOMs. The
    combine return buffer and the accumulator are BF16/FP32 on both precisions."""
    kept, dropped = [], []
    for tokens in ladder:
        layout = layout_for(tokens)
        receive = layout.max_recv * (hidden * value_bytes + scale_bytes_per_copy)
        dequantized = layout.max_recv * hidden * 2 if holds_dequantized else 0
        combine_return = layout.max_send * hidden * 2
        accumulator = tokens * hidden * 4
        need = receive + dequantized + combine_return + accumulator
        (kept if need <= budget_bytes else dropped).append(tokens)
    return kept, dropped


def to_host(array):
    """A fully-addressable numpy copy of a possibly multi-process device array.

    Single process: `np.asarray`, unchanged, and no collective.

    Multi process (EP16 spans two hosts): the oracle verifies EVERY rank's rows on host, but
    each process addresses only its own shards, so `np.asarray` on the global array raises
    rather than returning a partial answer. `process_allgather` is the gather that makes the
    whole thing readable. It is a host-side collective and every process must reach it the
    same number of times -- so it lives in this ONE helper rather than being sprinkled
    through the oracle, where an early `return False` on one process would deadlock the
    others.
    """
    if not hasattr(array, "sharding") or _process_count() == 1:
        return np.asarray(array)
    from jax.experimental import multihost_utils  # noqa: PLC0415

    return np.asarray(multihost_utils.process_allgather(array, tiled=True))


def writes_artifact(jax=None) -> bool:
    """Only ONE process may write the artifact, and it must be a PREDICTABLE one.

    Under EP16 every host runs this entrypoint and computes the same rows from the same
    byte-stable trace, so two processes writing one path would race and the harvester would
    read whichever landed last.

    The writer is chosen by the POD's completion index, NOT by `jax.process_index()`.
    Measured on the real slice: the two do not agree -- pod `ep16-probe-0` (TPU_WORKER_ID=0)
    reported `jax.process_index()==1` and pod `ep16-probe-1` reported `0`. JAX's process
    order is its own business. Gating on it would still yield exactly one writer, but WHICH
    pod that is would vary between runs, and the launcher harvests the result out of a
    specific pod's log -- so half the runs would harvest the pod that wrote nothing and
    report "the Job returned no result payload" on an otherwise good sweep.

    `JOB_COMPLETION_INDEX` is set by Kubernetes on every pod of an Indexed Job;
    `TPU_WORKER_ID` is GKE's equivalent and is the fallback. Absent both, this is a
    single-process run and it writes.
    """
    for name in ("JOB_COMPLETION_INDEX", "TPU_WORKER_ID"):
        index = os.environ.get(name)
        if index not in (None, ""):
            return index.strip() == "0"
    if jax is None or _PROCESS_COUNT[0] == 1:
        return True
    # Multi-process without either index: fall back to JAX's order so that exactly one
    # process still writes, and say so, because the harvester may look in the wrong pod.
    print("[run_ep_jax] WARNING: multi-process run with neither JOB_COMPLETION_INDEX nor "
          "TPU_WORKER_ID set; falling back to jax.process_index(), which may not be the "
          "pod the harvester reads", file=sys.stderr)
    return jax.process_index() == 0


def _trace(nodes: int, message: str) -> None:
    """Flushed progress, multi-host only.

    A single-host run reaches its first per-point line in seconds, so it needs none of
    this. A multi-host run can block in operand placement, in a cold cross-host compile, or
    in the oracle's host-side allgather -- and until these existed, a hung EP16 shard was
    killed having reported only the setup banner, which named none of the three.
    """
    if nodes > 1:
        print(f"[run_ep_jax] {message}", flush=True)


def _env_int(name: str):
    """An int from the environment, or None so JAX falls back to slice metadata."""
    raw = os.environ.get(name)
    return int(raw) if raw else None


def _process_count() -> int:
    """1 unless a multi-host run initialised jax.distributed."""
    return _PROCESS_COUNT[0]


#: Set once in main(), so helpers need no jax handle threaded through them.
_PROCESS_COUNT = [1]
#: Whether this PROCESS has joined the JAX cluster. One shard runs every case in one
#: process, and jax.distributed.initialize() is once-per-process, not once-per-case.
_DISTRIBUTED_READY = [False]


def _by_rank(array, ep_size: int) -> np.ndarray:
    """(ep*rows, hidden) -> (ep, rows, hidden). A view; nothing is copied or timed."""
    values = to_host(array)
    if values.ndim == 3:  # already per-rank (stubs in the test suite)
        return values
    return values.reshape(ep_size, -1, values.shape[-1])


def _values_already_checked(expected, rank: int, rows: int) -> bool:
    """Step 2 is satisfied elsewhere under fp8.

    `_check_dispatch_permutation` compares the arrived bytes against the staged bytes,
    which subsumes "the payload is this token's data" without re-deriving it through a
    second quantize. Steps 1 and 3 still run, so the decoded ID is still checked against
    the layout.
    """
    return True


def _check_dispatch_permutation(values, scales, staged_values, staged_scales,
                                layout: ep_jax.Layout, ep_size: int) -> bool:
    """Every (dst, src) chunk that arrived is byte-identical to the chunk src staged.

    Checks identity AND placement in one comparison, exactly: a transport that moved the
    wrong rows, or the right rows to the wrong offset, differs from the staged bytes. Immune
    to quantize numerics by construction -- it never quantizes anything, and both sides come
    out of the SAME compiled execution, so it does not care how that quantize was compiled.
    """
    for dst in range(ep_size):
        for src in range(ep_size):
            size = int(layout.recv_sizes[dst, src])
            if not size:
                continue
            here = int(layout.recv_offsets[dst, src])
            there = int(layout.input_offsets[src, dst])
            got_values = values[dst, here:here + size]
            want_values = staged_values[src, there:there + size]
            got_scales = scales[dst, here:here + size]
            want_scales = staged_scales[src, there:there + size]
            if not (np.array_equal(got_values, want_values)
                    and np.array_equal(got_scales, want_scales)):
                bad = int((np.asarray(got_values, dtype=np.float32)
                           != np.asarray(want_values, dtype=np.float32)).sum())
                print(f"[oracle] fp8 chunk dst={dst} src={src} size={size} differs from "
                      f"staged: {bad}/{np.asarray(got_values).size} value bytes, "
                      f"scales_equal={np.array_equal(got_scales, want_scales)}",
                      file=sys.stderr)
                return False
    return True


def _oracle(point, layout, activations, seed, ep_size, jax):
    """Run both halves of the probe oracle; returns (dispatch_ok, max_relative_error)."""
    # The device programs return (ep*rows, hidden): shard_map concatenates the per-shard
    # 2D results, because expanding them to a leading axis of size 1 inside the program
    # cost a full re-materialisation of the buffer on every call. Reshaping back is a
    # numpy view on the host, outside anything timed.
    dispatched = jax.block_until_ready(point.dispatch())
    if point.t.fp8:
        # Compare what ARRIVED against what was STAGED, chunk by chunk. Bit-exact by
        # actual construction: the transport either moved those bytes to that offset or it
        # did not, and no numeric assumption is involved.
        #
        # The previous design re-quantized the expected activations and compared bits. That
        # premise -- identity by construction, because both sides call quantize_local -- is
        # false on TPU: the wire's quantize is XLA-compiled inside a fused shard_map and the
        # oracle's ran eager, and `view * (448/amax)` may reassociate to `(view * 448)/amax`
        # between the two, one rounding against two. Measured offline, algebraically
        # equivalent forms disagree on 116-196 of 771 lattice values, and this lattice hits
        # exact e4m3 midpoints where one f32 ulp flips the output byte. The GPU harness
        # relies on bitwise identity only because it ASSERTS it on metal first; this port
        # kept the assumption and dropped the assertion.
        # ONE execution yields both sides of the comparison.
        values, scales, staged_values, staged_scales = point.dispatch_with_staged()
        # ...but that execution is the ORACLE program. It shares
        # `_quantized_dispatch_local` with the timed one, so the collective and its barrier
        # have a single definition, yet it is still its own compilation. Tie the two
        # together on the scales, which are the part nothing else can police: exact, and
        # safe to compare exactly. See _check_timed_scales_match.
        #
        # Every tensor the fp8 oracle reads is gathered HERE, exactly once each. `to_host`
        # is a cross-host collective at EP16 (a no-op at EP8, where the shard is one
        # process), so a helper that re-gathers an array a previous helper already pulled
        # pays for the whole buffer again over the host network. Measured on run
        # 31194062843: `values` was gathered twice and `scales` three times, 11.9GB of the
        # 62.7GB the EP16 fp8 prefill case moved -- ~19 minutes of a 100-minute case, on a
        # shard that finished within 9 minutes of its timeout.
        #
        # Straight-line and unconditional, which is what keeps it safe: `to_host` must be
        # reached the same number of times by every process, so these may never move under
        # a branch or an early return. The helpers still call `_by_rank` internally and
        # that stays correct -- it passes a host array straight back (`to_host` short-
        # circuits on anything without `.sharding`, then the ndim==3 branch returns it
        # unchanged), so they remain usable standalone and under the test stubs.
        timed_scales_host = _by_rank(dispatched[1], ep_size)
        values_host = _by_rank(values, ep_size)
        scales_host = _by_rank(scales, ep_size)
        staged_values_host = _by_rank(staged_values, ep_size)
        staged_scales_host = _by_rank(staged_scales, ep_size)
        timed_scales_ok = _check_timed_scales_match(
            timed_scales_host, scales_host, layout, ep_size)
        received = _by_rank(jax.block_until_ready(point.combine_input()), ep_size)
        # `received` is the dequantized TIMED output, so this compares the two programs'
        # values. Both checks run unconditionally: when one fails, whether the OTHER also
        # failed is the first thing worth knowing -- shared-program corruption shows up in
        # both, a compilation divergence in only one.
        timed_values_ok = _check_timed_values_match(
            received, values_host, scales_host, layout, ep_size)
        # The combine oracle must expect rank-sums of the DEQUANTIZED activations. The
        # codec's error (~6.25% worst case for e4m3) exceeds COMBINE_REL_TOL (3.125%), which
        # was set for BF16, so comparing against the pristine values fails a correct fp8
        # transport for being lossy -- which is what fp8 is. Two runs read correct=False for
        # this reason while the dispatch half was passing, and the diagnostic I added
        # instrumented the dispatch half, so it printed nothing: the absence localised it.
        permutation_ok = _check_dispatch_permutation(
            values_host, scales_host, staged_values_host, staged_scales_host,
            layout, ep_size,
        )
        fp8_ok = timed_scales_ok and timed_values_ok and permutation_ok
        verify_values = _values_already_checked
    else:
        received = _by_rank(dispatched, ep_size)
        fp8_ok, verify_values = True, None
    combined = _by_rank(jax.block_until_ready(point.combine()), ep_size)
    error = _combine_error(combined, activations, layout, ep_size,
                           "fp8" if point.t.fp8 else "",
                           received=received if point.t.fp8 else None, seed=seed)
    return (
        fp8_ok and _check_dispatch(received, layout, seed, ep_size, verify_values),
        error,
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
            # Same provenance shape as a successful document, so a consumer reads one
            # schema. `samples_per_component: 0` is the terminal-failure fact: nothing ran.
            # Nothing ran, so neither count describes anything: the backfill invokes this
            # entrypoint without --xprof-iters, so device_occurrences_per_point would
            # otherwise report the parser default for a capture that never happened.
            "sampling": {**_sampling_provenance(args), "samples_per_component": 0,
                         "device_occurrences_per_point": None,
                         "host_samples_per_point": 0},
            "timing_source": timing_source(False),
        },
        "implementation": {
            **fp8_provenance(args.precision),
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
    ap.add_argument("--chain-iters", type=int, default=64,
                    help="dispatch->combine pairs per chained capture. STATIC in the "
                         "compiled program, which is what keeps a multi-host slice in "
                         "lockstep; never derive it from anything host-local")
    ap.add_argument("--chain-drop", type=int, default=1,
                    help="leading periods discarded as pipeline fill")
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
    if args.precision not in ("bf16", "fp8"):
        print(f"ERROR: {args.backend} realizes bf16 and fp8, not {args.precision!r}",
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

    # Multi-host EP (EP16 = two 8-device hosts in ONE ICI slice) needs every process in the
    # slice to join one JAX cluster BEFORE anything touches a device. On Cloud TPU the
    # coordinator, process count and index all come from the slice metadata, so the no-arg
    # form is correct here; the env overrides exist for a local reproduction.
    if nodes > 1 and _DISTRIBUTED_READY[0]:
        # Second and later cases of the same shard. `run_ep_jax_shard.py` calls main() once
        # per case IN ONE PROCESS, and jax.distributed.initialize() may be called only once
        # per process -- a second call raises "must be called before any JAX calls that might
        # initialise the XLA backend". EP16's decode case passed and its prefill case died on
        # exactly that. The cluster from the first case is still joined, so nothing to redo.
        print(f"[run_ep_jax] EP{ep_size}: already joined to the "
              f"{_PROCESS_COUNT[0]}-process cluster", flush=True)
    elif nodes > 1:
        # Progress markers, flushed, because a multi-host shard that hangs is otherwise
        # undiagnosable: the first EP16 attempt was SIGKILLed at its shard timeout having
        # printed nothing, and every stage below is a plausible place to block -- joining the
        # cluster waits on a peer, and the first cross-host compile has no cache to hit.
        print(f"[run_ep_jax] EP{ep_size}: joining the JAX cluster across {nodes} hosts "
              f"(JOB_COMPLETION_INDEX={os.environ.get('JOB_COMPLETION_INDEX')} "
              f"TPU_WORKER_ID={os.environ.get('TPU_WORKER_ID')} "
              f"hosts={os.environ.get('TPU_WORKER_HOSTNAMES')})", flush=True)
        try:
            jax.distributed.initialize(
                coordinator_address=os.environ.get("JAX_COORDINATOR_ADDRESS") or None,
                num_processes=_env_int("JAX_NUM_PROCESSES"),
                process_id=_env_int("JAX_PROCESS_ID"),
            )
        except Exception as exc:
            print(f"ERROR: EP{ep_size} spans {nodes} hosts but jax.distributed would not "
                  f"initialise: {exc!r}", file=sys.stderr)
            return 3
        _DISTRIBUTED_READY[0] = True
    _PROCESS_COUNT[0] = jax.process_count()
    if nodes > 1:
        print(f"[run_ep_jax] joined: process {jax.process_index()} of "
              f"{_PROCESS_COUNT[0]}, {len(jax.devices())} global devices, "
              f"{len(jax.local_devices())} local", flush=True)
    if _PROCESS_COUNT[0] != nodes:
        print(f"ERROR: EP{ep_size} expects {nodes} JAX process(es), the cluster reports "
              f"{_PROCESS_COUNT[0]}", file=sys.stderr)
        return 3
    # GLOBAL devices, deliberately: jax.devices() is the whole slice in canonical order and
    # is identical on every process, so every rank builds the same mesh and the same layout.
    # jax.local_devices() would silently build an EP8 mesh per host and measure two
    # independent 8-way exchanges instead of one 16-way one.
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
    if nodes > 1:
        print(f"[run_ep_jax] mesh built over {len(devices)} devices; building transport",
              flush=True)
    try:
        transport = ep_jax.JaxEPTransport(
            jax, jnp, shard_map, mesh, ep_size, args.hidden,
            precision=args.precision,
        )
        # Captured HERE, not read at row-build time: `transport` was shadowed further down
        # by an xprof local of the same name, so reading it late silently picked up a float.
        value_bytes = transport.dispatch_value_bytes
        scale_bytes_per_copy = transport.dispatch_scale_bytes_per_copy
        # Instance, not class: the fp8 override lives on the instance, so reading the class
        # would label an fp8 artifact "bf16" while its byte_provenance said fp8.
        dispatch_dtype = transport.dispatch_dtype
        combine_dtype = transport.combine_dtype
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
        value_bytes, scale_bytes_per_copy, holds_dequantized=args.precision == "fp8",
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
        _trace(nodes, f"T={tokens}: placing operands")
        point = transport.build(layout, activations[tokens])
        # Compile and settle before anything gate-bearing runs.
        _trace(nodes, f"T={tokens}: compiling and running roundtrip")
        jax.block_until_ready(point.roundtrip())
        _trace(nodes, f"T={tokens}: roundtrip ok; running the oracle")
        dispatch_ok, max_rel = _oracle(
            point, layout, activations[tokens], args.seed, ep_size, jax,
        )
        _trace(nodes, f"T={tokens}: oracle done (ok={dispatch_ok})")
        points[tokens] = point
        gates[tokens] = {
            "rstats": _routing_stats(
                traces[tokens][0], args, experts_per_rank, ep_size, tokens,
            ),
            "dispatch_ok": dispatch_ok,
            "max_rel": max_rel,
        }

    # ---- Pass 2: timed trials, same rotated orders as the GPU harness ---------
    measured_components = components_for(args.precision)
    pools = {tokens: {name: [] for name in measured_components} for tokens in ladder}
    # ---- Reference method: the bare collective, timed as the reference times it ----
    reference = {}
    if args.reference_method:
        for tokens in ladder:
            built = points[tokens].reference_program()
            if built is None:
                break
            operation, _ = built
            # lockstep: this loop drives a collective over the whole slice, so a
            # duration-driven iteration count would desynchronise the hosts.
            samples = ep_jax.reference_timing(
                jax, operation, lockstep=_PROCESS_COUNT[0] > 1)
            metrics = ep_jax.iqr_metrics(samples)
            # DENSE all_to_all pads every chunk to the largest (src, dst) pair, so it
            # moves slightly more than the ragged exchange. Bill what it actually moved,
            # and record the delta rather than treating it as negligible.
            # PER DIRECTION. Only dispatch changes with precision: it moves 1 byte per
            # value plus one FP32 scale per 128-block under fp8. Combine moves the
            # dequantized BF16 rows on BOTH precisions (the expert emits BF16), and the
            # reference program stages self.x in BF16 regardless of precision. Billing one
            # case-level figure against all three understated combine and the reference by
            # ~1.94x at hidden=7168 -- an earlier fix here corrected dispatch and broke
            # those two, which is the same mistake in the other direction.
            dispatch_egress = ep_jax.egress_bytes(
                layout_for(tokens), args.hidden, value_bytes, scale_bytes_per_copy,
            )
            bf16_egress = ep_jax.egress_bytes(layout_for(tokens), args.hidden, 2, 0)
            egress = {"mean_per_device": bf16_egress["mean_per_device"]}
            latency_s = (metrics["avg_ms"] / 1000.0) if metrics else 0.0
            reference[tokens] = {
                "method": "accelerator-microbenchmarks/AllToAllBenchmark",
                "note": "host-timed, so it carries a per-call floor -- and that floor is "
                        "PAYLOAD-DEPENDENT, not the ~450us a fixed dispatch cost would "
                        "give: measured host-minus-span is 440-500us to T=1024, ~750us at "
                        "T=2048-4096 and ~2,300us at T=8192, and dispatch shows the same, "
                        "so it is a harness property rather than something in this "
                        "program. Two device figures sit below and they DISAGREE by up to "
                        "~1.5x; see collective_device_us_note, and op_inventory for the "
                        "ops behind each",
                "timed": "RAGGED jax.lax.ragged_all_to_all over a take-staged BF16 "
                         "operand -- the same collective the components measure, so this "
                         "is comparable to dispatch.transport_us. The upstream benchmark's "
                         "METHOD is what is transcribed (per-call host timing loop, IQR "
                         "filter, per-device egress convention), not its program: it times "
                         "a DENSE all_to_all, which was tried here and could not be "
                         "reconciled -- different primitive, ~11% cheaper per byte on 2% "
                         "more bytes. BF16 on both precisions, because it stages self.x "
                         "unquantized. The output-buffer memset (~363us at T=8192) IS "
                         "inside the timed scope, unavoidably: ragged needs the output "
                         "buffer that dense does not, so upstream's timed region has no "
                         "equivalent. See device_timing.reference.op_inventory",
                "latency": metrics,
                # BF16: the reference program transports self.x unquantized on both
                # precisions, so an fp8 byte basis here would understate it ~1.94x.
                "egress_bytes_per_device": egress["mean_per_device"],
                "wire": "ragged-bf16",
                "dispatch_egress_bytes_per_device": dispatch_egress["mean_per_device"],
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
        for name in measured_components:
            pools[tokens][name] = ep_jax.reference_timing(
                jax, point.timed_operation(name),
                warmup_tries=HOST_WARMUP_TRIES, num_runs=HOST_NUM_RUNS,
                min_duration_s=0.0, lockstep=_PROCESS_COUNT[0] > 1,
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
        for name in traced_for(args.precision):
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
            transport_us = ((parsed.get("by_hlo") or {})
                            .get(transport_key(name, tokens), {})
                            .get("per_iteration_us"))
            if total is not None and transport_us is not None:
                parsed["transport_us"] = transport_us
                parsed["non_transport_us"] = total - transport_us
                # transport_us is a SUM over the scope's ops, which is only a transport
                # time if they do not overlap. Measured, not assumed -- see overlap_us().
                block = (parsed.get("by_hlo") or {}).get(transport_key(name, tokens), {})
                parsed["transport_ops_disjoint"] = block.get("disjoint")
                parsed["transport_overlap_us"] = block.get("overlap_us")
            # Measured directly, where the scope exists, so `non_transport_us` can be
            # checked against it instead of standing in for it.
            permute = ((parsed.get("by_hlo") or {})
                       .get(ep_jax.permute_scope(name, tokens), {})
                       .get("per_iteration_us"))
            if permute is not None:
                parsed["permute_us"] = permute
            # The bf16->fp8 cast, measured rather than left in the subtraction residual.
            # Without this `non_transport_us` silently means different things on bf16 and
            # fp8 rows -- a reader comparing the two would charge the quantize to the
            # permute. Absent (not zero) on bf16, where the scope does not exist.
            quantize = ((parsed.get("by_hlo") or {})
                        .get(ep_jax.quantize_scope(name, tokens), {})
                        .get("per_iteration_us"))
            if quantize is not None:
                parsed["quantize_us"] = quantize
        # ---- chained pair period: a separate capture, a different quantity -------
        # Fresh-entry spans measure one pair entered from idle; the period measures the
        # steady-state rate of a pipeline that never drains. Both are published, and
        # `timing_source` tells them apart per row.
        chain_call = point.chain(args.chain_iters) if args.chain_iters > 0 else None
        if chain_call is None:
            # Say WHY, rather than leaving the generic "not captured" that a reader cannot
            # distinguish from a failed capture. fp8 is excluded by design and the READMEs
            # promise the reason travels with the row.
            parsed_point["chain"] = {
                "availability": "unavailable",
                "reason": ("fp8 is not chained: its combine needs BF16, so a chained body "
                           "would carry the fp8->bf16 conversion inside the loop and the "
                           "period would include work the `native` contract says production "
                           "does not do standalone" if point.t.fp8
                           else f"chaining disabled (--chain-iters {args.chain_iters})"),
            }
        if chain_call is not None:
            try:
                chain_started = time.time()
                warm = chain_call()
                jax.block_until_ready(warm)                  # warm: compile + fabric
                identity = chain_identity(jax, warm, point)
                del warm
                with tempfile.TemporaryDirectory(
                        prefix=f"collectivex-chain-t{tokens}-") as directory:
                    with profile_session(jax, directory):
                        jax.block_until_ready(chain_call())
                    marker = ep_jax.chain_scope(tokens)
                    anchor = _chain_anchor(directory, marker)
                    if anchor is None:
                        parsed_point["chain"] = {
                            "availability": "unavailable",
                            "reason": "no collective anchor under the chain scope",
                            "marker": marker,
                        }
                    else:
                        chain = xprof.parse_chain(directory, marker, anchor,
                                                  args.chain_iters, args.chain_drop)
                        chain = gather_chain_periods(chain, ep_size)
                        # DIFFERENT anchors, one per direction: the period needs an op
                        # firing once per iteration, a floor needs the op that is the
                        # transfer, and dispatch and combine are distinct instructions.
                        period = ((chain.get("percentiles_us") or {}).get("p50")
                                  if chain.get("availability") == "measured" else None)
                        split = xprof.floor_anchors(directory, marker, args.chain_iters)
                        chain["anchor_phase"] = split.get("phase")
                        chain["anchor_phase_note"] = split.get("phase_note")
                        if split.get("ok"):
                            chain["floors"] = {
                                side: xprof.chain_floors(directory, marker, split[side],
                                                         args.chain_drop, period)
                                for side in ("dispatch", "combine")
                            }
                        else:
                            # Both unavailable, never one series wearing two labels.
                            # `row_groups` is how the ops were distributed over device
                            # rows -- the fact the anchor decision turns on, and the one
                            # the artifact never carried while three selection rules were
                            # being tried against it.
                            block = {"availability": "unavailable",
                                     "reason": split.get("reason"),
                                     "row_groups": split.get("row_groups")}
                            chain["floors"] = {"dispatch": block, "combine": block}
                        # What the chain body actually contains, per op. The reason fp8
                        # is not chained is a CLAIM about this inventory -- that a chained
                        # fp8 body would have to carry the fp8->bf16 conversion inside the
                        # loop, so its period would include work the `native` contract says
                        # production does not do standalone. Publishing the bf16 inventory
                        # makes that claim checkable against a trace instead of an argument.
                        chain["op_inventory"] = (xprof.parse_trace_durations(
                            directory, marker, args.chain_iters).get("op_inventory"))
                        # Capture budget: the chain runs twice (warm, then traced) per
                        # point, on top of the fresh-entry components. Record the margin so
                        # a raised --chain-iters is a decision rather than a surprise
                        # timeout at some later ladder point.
                        # Independent of the floor gate: the window is the scope's own
                        # extent, so a run whose anchors are unusable still reports a gap.
                        chain["interpair_gap_us"] = xprof.chain_pair_gap(
                            directory, marker, anchor, args.chain_drop)
                        chain["capture_s"] = round(time.time() - chain_started, 2)
                        chain["capture_budget_s"] = CHAIN_CAPTURE_BUDGET_S
                        # The renormalisation is work production does not do, and it lands
                        # between combine and the next dispatch -- i.e. INSIDE the period,
                        # by construction, since the period is anchor-start to anchor-start.
                        # Publish what it costs rather than assert it is free.
                        chain["renorm_us"] = (xprof.parse_trace_durations(
                            directory, ep_jax.chain_norm_scope(tokens), args.chain_iters)
                            .get("per_iteration_us"))
                        chain["identity"] = identity
                        if identity.get("ok") is not True:
                            # The period describes a program that did not compute what the
                            # chain is defined to compute. Withhold it; keep the diagnosis.
                            chain["availability"] = "unavailable"
                            chain["reason"] = identity.get("reason")
                            chain["percentiles_us"] = None
                        parsed_point["chain"] = chain
            except Exception as exc:  # a chain failure must not cost the fresh-entry rows
                parsed_point["chain"] = {"availability": "unavailable",
                                         "reason": f"chain capture failed: {exc!r}"}
            _trace(nodes, f"T={tokens}: chain "
                          f"{parsed_point['chain'].get('availability')}")
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
            # Every PUBLISHED component, `stage` included under fp8. A stage that fell back
            # to host wall-clock would carry the ~450us dispatch floor into the one figure a
            # reader adds to `roundtrip`.
            if not all((device_timing.get(tokens, {}).get(name) or {}).get("percentiles_us")
                       for name in components_for(args.precision))
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
        dispatch_transport_us = ((device_timing.get(tokens, {}).get("dispatch") or {})
                                 .get("transport_us"))
        # NOT named `egress_bytes`: that is the imported ep_jax function, and rebinding a
        # module-level name to a float in a local scope is how `transport` came to be read
        # as a float in the row builder twice.
        bf16_basis = entry["egress_bytes_per_device"]
        span = ((device_timing.get(tokens, {}).get(REFERENCE) or {})
                .get("percentiles_us") or {}).get("p50")
        combine_transport = ((device_timing.get(tokens, {}).get("combine") or {})
                             .get("transport_us"))
        dispatch_basis = entry.get("dispatch_egress_bytes_per_device")
        record_collective_bandwidth(
            entry, dispatch_transport_us, span,
            # `is None`, NOT `or`: a legitimate 0.0 (every copy destined for its own rank,
            # so nothing egresses) would fall through and rebill dispatch at the BF16 basis
            # -- reintroducing exactly the mismatch this split exists to remove.
            dispatch_basis if dispatch_basis is not None else bf16_basis,
            combine_transport, bf16_egress_bytes=bf16_basis,
        )

    # ---- Pass 3: re-run the oracle after timing, then build the rows ---------
    rows = []
    for tokens in ladder:
        point, gate, layout = points[tokens], gates[tokens], layout_for(tokens)
        post_ok, post_rel = _oracle(
            point, layout, activations[tokens], args.seed, ep_size, jax,
        )
        max_rel = max(gate["max_rel"], post_rel)
        if not (gate["dispatch_ok"] and post_ok and
                max_rel <= ep_harness.COMBINE_REL_TOL):
            # Name WHICH of the three conjuncts failed. Two runs were spent guessing
            # because only one of them was instrumented.
            print(f"[oracle] T={tokens} FAIL dispatch_ok={gate['dispatch_ok']} "
                  f"post_ok={post_ok} max_rel={max_rel:.6f} "
                  f"tol={ep_harness.COMBINE_REL_TOL:.6f} "
                  f"gate_rel={gate['max_rel']:.6f} post_rel={post_rel:.6f}",
                  file=sys.stderr)
        rstats = gate["rstats"]
        pool = pools[tokens]
        # PUBLISHED latency is the device span -- the CUDA-event equivalent, and the only
        # TPU number of the same kind as the GPU SKUs'. Host wall-clock is the fallback
        # when a trace is unavailable, and it is NOT comparable to them: it carries a
        # ~450us per-call dispatch floor the GPU numbers do not have.
        measured = device_timing.get(tokens) or {}
        row_sources = {}
        rt_intact = roundtrip_contains_dispatch(
            measured.get("roundtrip"), measured.get("combine"),
            measured.get("dispatch"))
        if rt_intact is False:
            print(f"[gate] T={tokens} roundtrip does NOT contain its dispatch: the trace's "
                  f"op device time exceeds combine's by under "
                  f"{ROUNDTRIP_DISPATCH_MIN_SHARE:.0%} of the dispatch's. XLA deleted "
                  f"the dispatch and this roundtrip is a lone combine.", file=sys.stderr)
        passed = row_passed(gate["dispatch_ok"], post_ok, max_rel, rt_intact)

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
        (stage_pcts, stage_n) = (
            _percentiles(STAGE) if STAGE in measured_components else (None, 0))
        isolated = (
            {key: dispatch_pcts[key] + combine_pcts[key] for key in dispatch_pcts}
            if dispatch_pcts and combine_pcts else None
        )
        global_tokens = tokens * ep_size
        # fp8 dispatch moves 1 byte per value plus one FP32 scale per 128-element block;
        # bf16 moves 2 and no scales. Read off the transport so the artifact cannot claim a
        # precision the wire did not carry. Combine is BF16 on both, as on every GPU
        # backend -- the expert emits BF16.
        dispatch_bytes = ep_harness.logical_byte_provenance(
            rstats["routed_copies"], args.hidden,
            value_bytes, scale_bytes_per_copy,
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
                # BF16: unavailable, and correctly so -- the permute is fused into
                # dispatch and there is no conversion, so there is nothing to time.
                # FP8: the fp8->bf16 conversion, hoisted out of combine AND out of the
                # chained roundtrip and measured here, which is the component the GPU
                # harness puts the same work in. `roundtrip + stage` then reconstructs the
                # mismatched-config cost; the reverse direction does not hold.
                "stage": ep_harness._component(stage_pcts, stage_n),
            },
            **_chain_fields(measured.get("chain")),
            "correctness": _correctness(
                max_rel, passed, (measured.get("chain") or {}).get("identity")),
            # Whether the chained roundtrip's trace shows work the standalone combine's does
            # not. False means XLA deleted the dispatch and this row's `roundtrip` is a lone
            # combine; None means no trace to judge. Published, not merely gated, so a
            # consumer can see which rows the question was answerable for.
            "roundtrip_contains_dispatch": rt_intact,
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
                name: ep_harness._pcts(pool[name]) for name in measured_components
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
            "combine_dtype": combine_dtype,
            "combine_semantics": "activation-only",
            "dispatch_dtype": dispatch_dtype,
            "payload_unit": "token-rank",
            "rows": rows,
            "sampling": _sampling_provenance(args),
            # The device spans ARE the same kind of number as the GPU family's CUDA events:
            # chip time, MAX-reduced across devices per occurrence, raw percentiles, no host
            # dispatch cost. `host_latency_us` on each row is the cross-check and carries a
            # payload-dependent per-call floor, which is why it is never the headline and why
            # falling back to it takes --allow-host-fallback.
            "timing_source": _case_timing_source(rows),
        },
        "implementation": {
            **fp8_provenance(args.precision),
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
    if writes_artifact(jax):
        ep_harness._write_json_atomic(args.out, doc)
        print(f"{args.backend} ep-dispatch-combine [{args.phase}/{args.mode}]: "
              f"status={doc['outcome']['status']} {len(rows)} pts -> {args.out}")
    else:
        print(f"{args.backend} ep-dispatch-combine [{args.phase}/{args.mode}]: "
              f"status={doc['outcome']['status']} {len(rows)} pts "
              f"(process {jax.process_index()} defers the write to process 0)")
    # A captured `invalid` outcome must fail the leg, never ride as a green success.
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
