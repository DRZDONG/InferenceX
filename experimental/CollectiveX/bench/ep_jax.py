#!/usr/bin/env python3
"""JAX/TPU expert-parallel dispatch/combine transport for the CollectiveX probe.

This is deliberately NOT an ``ep_backend.EPBackend`` subclass. That base class is built
around a CUDA/HIP execution model that TPU does not have: one OS process per rank, a
``torch.distributed`` process group, per-call host-side ``dispatch()``/``combine()``
entry points, and ``torch.cuda.Event`` timing. On TPU a single process owns every local
device and the collective is fused inside one compiled XLA program, so the transport is
expressed as jitted functions over a device mesh instead.

What IS shared with the GPU backends, deliberately and exactly:
  * the workload (deepseek-v3 shape) and the byte-stable routing trace, via routing_np,
    which is a parity-tested port of bench/routing.py;
  * the payload unit -- one copy per unique (token, destination rank) pair, deduplicated
    across top-k, so the logical bytes are computed the same way;
  * the combine contract -- ``unweighted-rank-sum``.

The transport is ``jax.lax.ragged_all_to_all``: variable-length per-destination chunks,
the primitive JAX MoE stacks use for EP dispatch/combine. A build without it fails closed
rather than substituting a padded fixed-capacity ``all_to_all``, which would move padding
and quietly measure something else.

Timing caveat (this is the headline methodological difference from the GPU harness):
there is no public JAX equivalent of ``torch.cuda.Event``, so latencies here are HOST
wall-clock around a ``block_until_ready`` on a jitted program. That includes dispatch
overhead the CUDA-event numbers exclude -- tens of microseconds, which is negligible at
prefill sizes and NOT negligible at T=1. The emitted artifact records
``measurement.timing_source`` so a consumer cannot mistake these for event timings.

The per-destination layout (offsets, sizes, gather indices) is precomputed on host and
excluded from the timed region, matching the GPU harness's treatment of DeepEP's layout
pass. The on-device permute gather and the combine scatter-add ARE inside the timed
region, because production pays them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import routing_np


# jax.named_scope labels wrapped around each timed component. bench/xprof.py matches
# these as substrings in a profiler trace to recover device-side durations, so the prefix
# must stay distinctive enough not to collide with framework op names.
SCOPE_PREFIX = "collectivex"
# HLO ops broken out when a trace is parsed. Exactly one entry, deliberately: matching by
# name is ambiguous and an earlier four-entry list published rows that looked like four
# independent components but were not. "all-to-all" is a SUBSTRING of "ragged-all-to-all"
# so it duplicated it exactly, and "fusion" and "scatter" matched the same fused ops
# (observed at T=8192: scatter 4446.3us, fusion 4411.2us -- the same work counted twice).
#
# What is NOT the collective is therefore derived by SUBTRACTION from the component total
# rather than named: dispatch's remainder is its permute gather, combine's is its fp32
# scatter-add. Subtraction cannot double-count and needs no guess about lowering.
TRACED_HLO = ("ragged-all-to-all",)
# The reference program is DENSE all_to_all (AllToAllBenchmark, transcribed), so its HLO
# is `all-to-all`. Matching that inside the components would be ambiguous -- it is a
# substring of `ragged-all-to-all` -- but the reference program contains no ragged op, so
# there it is exact.
REFERENCE_HLO = ("all-to-all",)


def transport_scope(component: str, tokens_per_rank: int) -> str:
    """Trace label wrapping ONLY the collective, nested inside the component scope.

    The reference implementation (accelerator-microbenchmarks `AllToAllBenchmark`) scopes
    the collective alone and returns it directly, so every event under its marker is the
    same op and a per-occurrence series is well defined. This probe's component scope is
    necessarily wider -- dispatch also permutes, combine also scatter-adds -- so it needs
    both: the outer scope sums to the component's device time, this inner one isolates the
    transport. Nested scopes concatenate in the trace, so an event here matches both.
    """
    return f"{SCOPE_PREFIX}-xport-{component}-t{int(tokens_per_rank)}"


def scope_name(component: str, tokens_per_rank: int) -> str:
    """Trace label for one component at one ladder point.

    The ladder point is part of the label so a single trace covering the whole ladder
    can be attributed per point; without it, occurrences from different T values would
    pool into one indistinguishable series.
    """
    return f"{SCOPE_PREFIX}-{component}-t{int(tokens_per_rank)}"


def import_jax():
    """Import jax and resolve the shard_map spelling this build uses."""
    import jax
    import jax.numpy as jnp

    shard_map = getattr(jax, "shard_map", None)
    if shard_map is None:  # JAX < 0.6 kept it under jax.experimental
        from jax.experimental.shard_map import shard_map  # noqa: PLC0415
    return jax, jnp, shard_map


def ragged_all_to_all_available(jax) -> bool:
    return hasattr(jax.lax, "ragged_all_to_all")



@dataclass
class Layout:
    """Host-computed per-device exchange layout for one token-ladder point.

    Every array is indexed [device, ...] and is identical on every process (there is
    one process). ``send_index`` is the permute gather: for device s it lists, in
    destination-grouped order, which LOCAL token index each outgoing copy carries.
    """

    tokens_per_rank: int
    ep_size: int
    send_index: np.ndarray        # [ep, max_send]  int32, padded with 0
    send_sizes: np.ndarray        # [ep, ep]        int32, [src, dst]
    input_offsets: np.ndarray     # [ep, ep]        int32, prefix over dst within src
    output_offsets: np.ndarray    # [ep, ep]        int32, offset in the DESTINATION buffer
    recv_sizes: np.ndarray        # [ep, ep]        int32, [dst, src]
    recv_offsets: np.ndarray      # [ep, ep]        int32, prefix over src within dst
    return_offsets: np.ndarray    # [ep, ep]        int32, combine-direction remote offsets
    send_total: np.ndarray        # [ep]            int32
    recv_total: np.ndarray        # [ep]            int32
    max_send: int
    max_recv: int
    routed_copies: int

    def payload_bytes(self, hidden: int, value_bytes: int = 2) -> int:
        return int(self.routed_copies) * int(hidden) * int(value_bytes)


def build_layout(global_idx: np.ndarray, tokens_per_rank: int, ep_size: int,
                 experts_per_rank: int) -> Layout:
    """Derive the deduplicated (token, destination-rank) exchange plan on host.

    One copy per unique destination rank per token -- the same payload unit the GPU
    backends bill, so the two families' logical bytes are comparable.
    """
    destinations = routing_np._destination_onehot(global_idx, experts_per_rank, ep_size)
    token, dest = np.nonzero(destinations)
    src = np.minimum(token // max(1, tokens_per_rank), ep_size - 1)
    local_token = (token - src * tokens_per_rank).astype(np.int32)

    send_sizes = np.zeros((ep_size, ep_size), dtype=np.int32)
    np.add.at(send_sizes, (src, dest), 1)
    input_offsets = np.zeros_like(send_sizes)
    input_offsets[:, 1:] = np.cumsum(send_sizes, axis=1)[:, :-1]
    send_total = send_sizes.sum(axis=1).astype(np.int32)

    recv_sizes = send_sizes.T.copy()
    recv_offsets = np.zeros_like(recv_sizes)
    recv_offsets[:, 1:] = np.cumsum(recv_sizes, axis=1)[:, :-1]
    recv_total = recv_sizes.sum(axis=1).astype(np.int32)

    # output_offsets[s][d] is where s's chunk lands in d's RECEIVE buffer; d groups its
    # receives by source rank, so that is d's prefix over sources up to s.
    output_offsets = recv_offsets.T.copy()
    # Combine runs the same exchange backwards: the chunk s returns to source i must
    # land where i originally staged it, i.e. at i's own input offset for destination s.
    return_offsets = input_offsets.T.copy()

    max_send = int(send_total.max()) if send_total.size else 0
    max_recv = int(recv_total.max()) if recv_total.size else 0
    send_index = np.zeros((ep_size, max(1, max_send)), dtype=np.int32)
    # Destination-grouped ordering per source: sort by (dest) with a stable sort so the
    # token order inside each chunk stays ascending and the plan is reproducible.
    for rank in range(ep_size):
        selected = src == rank
        order = np.argsort(dest[selected], kind="stable")
        rows = local_token[selected][order]
        send_index[rank, :rows.size] = rows

    return Layout(
        tokens_per_rank=tokens_per_rank,
        ep_size=ep_size,
        send_index=send_index,
        send_sizes=send_sizes,
        input_offsets=input_offsets,
        output_offsets=output_offsets,
        recv_sizes=recv_sizes,
        recv_offsets=recv_offsets,
        return_offsets=return_offsets,
        send_total=send_total,
        recv_total=recv_total,
        max_send=max_send,
        max_recv=max_recv,
        routed_copies=int(dest.size),
    )


class JaxEPTransport:
    """Dispatch/combine over one 1-D device mesh, for a single ladder point."""

    name = "jax-ragged-a2a"
    combine_weight_semantics = "unweighted-rank-sum"
    dispatch_dtype = "bf16"
    combine_dtype = "bf16"
    dispatch_value_bytes = 2
    dispatch_scale_bytes_per_copy = 0

    def __init__(self, jax, jnp, shard_map, mesh, ep_size: int, hidden: int):
        if not ragged_all_to_all_available(jax):
            raise RuntimeError(
                "this JAX build has no jax.lax.ragged_all_to_all; upgrade the image"
            )
        self.jax, self.jnp, self.shard_map = jax, jnp, shard_map
        self.mesh = mesh
        self.ep_size = ep_size
        self.hidden = hidden
        self.axis = "ep"
        self._P = jax.sharding.PartitionSpec

    # ---- placement helpers ----------------------------------------------------

    def _shard(self, array):
        """Put a [ep, ...] host array on the mesh, sharded over the leading axis."""
        sharding = self.jax.sharding.NamedSharding(self.mesh, self._P(self.axis))
        return self.jax.device_put(self.jnp.asarray(array), sharding)

    def _mapped(self, fn, in_specs, out_specs, donate=()):
        P = self._P
        return self.jax.jit(
            self.shard_map(
                fn, mesh=self.mesh,
                in_specs=tuple(P(self.axis) if s else P() for s in in_specs),
                out_specs=P(self.axis) if out_specs else P(),
            ),
            donate_argnums=tuple(donate),
        )

    # ---- transport ------------------------------------------------------------

    def build(self, layout: Layout, activations: np.ndarray):
        """Materialize device state for one ladder point; returns a Point."""
        jnp = self.jnp
        x = self._shard(activations.astype(np.float32)).astype(jnp.bfloat16)
        return Point(
            transport=self,
            layout=layout,
            x=x,
            send_index=self._shard(layout.send_index),
            input_offsets=self._shard(layout.input_offsets),
            send_sizes=self._shard(layout.send_sizes),
            output_offsets=self._shard(layout.output_offsets),
            recv_sizes=self._shard(layout.recv_sizes),
            recv_offsets=self._shard(layout.recv_offsets),
            return_offsets=self._shard(layout.return_offsets),
        )


class Point:
    """One fully-materialized ladder point: device inputs plus its jitted programs."""

    def __init__(self, transport: JaxEPTransport, layout: Layout, x, send_index,
                 input_offsets, send_sizes, output_offsets, recv_sizes, recv_offsets,
                 return_offsets):
        self.t = transport
        self.layout = layout
        self.x = x
        self.send_index = send_index
        self.input_offsets = input_offsets
        self.send_sizes = send_sizes
        self.output_offsets = output_offsets
        self.recv_sizes = recv_sizes
        self.recv_offsets = recv_offsets
        self.return_offsets = return_offsets
        jax, jnp = transport.jax, transport.jnp
        self._jax, self._jnp = jax, jnp
        axis = transport.axis
        hidden = transport.hidden
        max_recv = max(1, layout.max_recv)
        max_send = max(1, layout.max_send)

        # PLAN_ORDER is the single ordering every mapped function and every call site
        # shares, so the seven layout arrays cannot drift out of correspondence.
        def _plan(arrays):
            """Unpack one device's slice of the plan (shard_map hands us [1, ...])."""
            index, in_off, send_sz, out_off, recv_sz, recv_off, ret_off = (
                a[0] for a in arrays
            )
            return index, in_off, send_sz, out_off, recv_sz, recv_off, ret_off

        def _dispatch_local(x_l, plan):
            index, in_off, send_sz, out_off, recv_sz, recv_off, _ = plan
            # Permute gather: stage this device's outgoing copies in destination order.
            # Production pays this, so it stays inside the timed region.
            staged = jnp.take(x_l, index, axis=0)
            out = jnp.zeros((max_recv, hidden), dtype=x_l.dtype)
            with jax.named_scope(transport_scope("dispatch", layout.tokens_per_rank)):
                return jax.lax.ragged_all_to_all(
                    staged, out, in_off, send_sz, out_off, recv_sz, axis_name=axis,
                )

        def _combine_local(recv_l, plan):
            index, in_off, send_sz, _, recv_sz, recv_off, ret_off = plan
            back = jnp.zeros((max_send, hidden), dtype=recv_l.dtype)
            with jax.named_scope(transport_scope("combine", layout.tokens_per_rank)):
                returned = jax.lax.ragged_all_to_all(
                    recv_l, back, recv_off, recv_sz, ret_off, send_sz, axis_name=axis,
                )
            # Unweighted rank-sum, accumulated in fp32: the wire stays bf16, but summing
            # up to top-k bf16 copies is not exact and production reduces in fp32 too.
            zeros = jnp.zeros((layout.tokens_per_rank, hidden), dtype=jnp.float32)
            return zeros.at[index].add(returned.astype(jnp.float32))

        # Each component runs inside a named scope so a profiler trace can attribute
        # device time to it (bench/xprof.py matches these labels). Scopes are labels
        # only -- they do not change the compiled computation or the host timing.
        # NO `[None]` on any result. Expanding the per-shard output to give shard_map a
        # leading mapped axis re-materialises the whole buffer on every call -- measured
        # in the reference program at 3,232us of a 15,084us figure at T=8192, and these
        # three carried the identical op (`broadcast_in_dim.19`, 3,231us in dispatch).
        # That is measurement overhead inside a PUBLISHED latency: the leading axis of
        # size 1 is an artifact of this file's shard_map plumbing, not something a
        # production MoE pays. Returning the 2D result lets shard_map concatenate for
        # free; the global result is (ep*rows, hidden) and the oracle reshapes it back on
        # the host, outside the timed region.
        #
        # `x_g[0]` stays: squeezing the OPERAND was tried in the reference program and
        # moved its span by 0.8us, so it costs nothing worth restructuring for.
        def dispatch(x_g, *plan_g):
            with jax.named_scope(scope_name("dispatch", layout.tokens_per_rank)):
                return _dispatch_local(x_g[0], _plan(plan_g))

        # recv_g arrives 2D now, being dispatch's own output fed back in.
        def combine(recv_g, *plan_g):
            with jax.named_scope(scope_name("combine", layout.tokens_per_rank)):
                return _combine_local(recv_g, _plan(plan_g))

        def roundtrip(x_g, *plan_g):
            plan = _plan(plan_g)
            with jax.named_scope(scope_name("roundtrip", layout.tokens_per_rank)):
                return _combine_local(_dispatch_local(x_g[0], plan), plan)

        self._dispatch = transport._mapped(dispatch, (True,) * 8, True)
        self._combine = transport._mapped(combine, (True,) * 8, True)
        self._roundtrip = transport._mapped(roundtrip, (True,) * 8, True)
        self._dispatched = None

        self._plan_fn = _plan
        self._program_cache = {}
        self._reference = None

    def reference_program(self):
        """`jit(shard_map(named_scope(all_to_all)))` -- AllToAllBenchmark, transcribed.

        Read from the source (collectives.py, AllToAllBenchmark._setup_jit_fn), the
        reference times exactly this:

            jax.lax.all_to_all(a, axis_name=..., split_axis=0, concat_axis=0, tiled=True)

        inside one named scope, with the operand passed straight in and the result
        returned straight out. DENSE all_to_all takes no output buffer, so there is no
        allocation, no memset and no reshape on either side of the timed region -- the
        structural reason its figure is the collective and nothing else.

        This probe previously timed `ragged_all_to_all` here, which needs an output buffer
        and therefore carried a 363us memset plus a further 3,280us op no experiment could
        remove (donation was honoured and cost +2,900us; removing the output `[None]` and
        the operand `[0]` changed 3,232us and 0.8us respectively). Calling that
        "AllToAllBenchmark" in `reference_transport.method` was inaccurate, and it is the
        reason the figure sat at ~12,800us where the bare collective is ~6,700us.

        Dense is a fair comparison here because this workload's per-pair sizes are nearly
        uniform: at T=8192 receive counts span 43,255..43,555, a 0.7% spread, so padding
        every chunk to the max moves ~0.7% more than the ragged exchange. That delta is
        recorded in the artifact rather than assumed negligible.

        Returns (callable, send_buffer).
        """
        if getattr(self, "_reference", None) is not None:
            return self._reference
        jax, jnp = self._jax, self._jnp
        axis, hidden = self.t.axis, self.t.hidden
        ep_size = self.t.ep_size

        # The dense chunk: every (src, dst) pair pads to the largest, as all_to_all
        # requires equal splits. Built ONCE here, outside anything timed.
        chunk = max(1, int(self.layout.send_sizes.max()))
        rows = ep_size * chunk
        max_send = max(1, self.layout.max_send)

        def _fit(g):
            if max_send >= rows:
                return g[:rows]
            return jnp.pad(g, ((0, rows - max_send), (0, 0)))

        # Operand staged 3D as (ep, chunk, hidden) per shard -- the shape upstream passes
        # (AllToAllBenchmark uses (dim, _BASE_N, _BASE_K)), with tiled=True as upstream
        # does. split_axis=0 then has exactly the axis size, so the split is one slice per
        # device and no local rearrangement is needed.
        #
        # A 2D (ep*chunk, hidden) operand cost two extra ops -- 897.7us and 900.5us at
        # T=8192, bracketing the 6,962us exchange -- because the split axis had to be
        # carved out of the leading dimension and folded back. A matched pre/post pair at
        # ~13% each is a reshape signature, not transfer, and staging 3D removed both
        # (span 8,760us -> 7,416us). The reshape happens once here, outside the timed
        # region.
        staged = self.t._mapped(
            lambda x_g, index_g: _fit(
                jnp.take(x_g[0], index_g[0], axis=0)
            ).reshape(ep_size, chunk, hidden),
            (True, True), True,
        )(self.x, self.send_index)
        staged = jax.block_until_ready(staged)

        def transport(send_g):
            with jax.named_scope(REFERENCE_MARKER):
                return jax.lax.all_to_all(
                    send_g, axis_name=axis, split_axis=0, concat_axis=0, tiled=True,
                )

        program = self.t._mapped(transport, (True,), True)

        def call():
            return program(staged)

        self.reference_chunk_rows = chunk
        self.reference_egress_bytes = float(chunk * (ep_size - 1) * hidden
                                            * self.t.dispatch_value_bytes)
        self._reference = (call, staged)
        return self._reference

    @property
    def _plan_arrays(self):
        return (self.send_index, self.input_offsets, self.send_sizes,
                self.output_offsets, self.recv_sizes, self.recv_offsets,
                self.return_offsets)

    # ---- callable operations (each returns a device array) --------------------

    def dispatch(self):
        return self._dispatch(self.x, *self._plan_arrays)

    def combine(self, received=None):
        source = self.dispatched() if received is None else received
        return self._combine(source, *self._plan_arrays)

    def roundtrip(self):
        return self._roundtrip(self.x, *self._plan_arrays)

    def dispatched(self):
        """The dispatch result, materialized once and reused (untimed prep)."""
        if self._dispatched is None:
            self._dispatched = self._jax.block_until_ready(self.dispatch())
        return self._dispatched

    def timed_operation(self, component: str):
        """The callable that runs `component` once. One call, one occurrence.

        The device trace supplies the latency, so there is nothing to amortize: chip time
        carries no host dispatch cost to divide away.
        """
        return {
            "dispatch": self.dispatch,
            "combine": lambda: self.combine(self.dispatched()),
            "roundtrip": self.roundtrip,
        }[component]



def time_us(jax, fn, warmup: int, iters: int, per_call: int = 1) -> list[float]:
    """Host wall-clock microseconds per operation, blocking on each call.

    Secondary to the device trace: this carries a ~450us per-call dispatch floor that
    chip time does not. Kept as a sanity check on the device numbers, not as the headline.

    Every device runs inside one program, so a single measurement already covers the
    slowest device -- there is no cross-rank reduction to do (the GPU harness takes a
    per-iteration MAX across ranks for exactly this reason).

    `per_call` is how many operations the callable performs internally. Each returned
    sample is the call's elapsed time divided by it, so the unit stays microseconds per
    operation.
    """
    if per_call < 1:
        raise ValueError("per_call must be >= 1")
    for _ in range(max(0, warmup)):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter_ns() - start) / 1000.0 / per_call)
    return samples


# ---------------------------------------------------------------------------------
# Reference method: accelerator-microbenchmarks' AllToAllBenchmark, transcribed.
#
# That benchmark times the COLLECTIVE AND NOTHING ELSE -- `run_op` is
# `jit(shard_map(named_scope(all_to_all)))` over a pre-prepared input, with the result
# returned directly. This probe's dispatch/combine are deliberately wider: they include
# the permute and the scatter-add because production pays them and because DeepEP's
# kernels do too, which is what makes the GPU comparison fair. Both are worth having, so
# the reference method runs alongside rather than replacing them:
#
#   * it is directly comparable to published TPU collective numbers, which the composite
#     components are not;
#   * the collective is the program's output, so liveness needs no barrier and no
#     fold-back -- the hazard that produced a hollow measurement cannot arise;
#   * its scope holds one op, so a per-occurrence trace series is well defined;
#   * it isolates transport from the scatter-add without decomposing anything.
# ---------------------------------------------------------------------------------

# Single marker, as the reference uses -- this program contains exactly one collective,
# so it needs no per-ladder-point disambiguation.
REFERENCE_MARKER = f"{SCOPE_PREFIX}-reference-transport"


def reference_timing(jax, fn, warmup_tries: int = 10, num_runs: int = 10,
                     min_duration_s: float = 1.0) -> list:
    """The reference harness's measurement loop (core/base.py), transcribed.

    Host wall-clock around a blocking call, per iteration -- no in-jit chaining. Warmup
    and measurement both continue past their try counts until a duration floor is met, so
    a fast operation still gets a statistically useful sample count.
    """
    warmup_floor = min(1.0, min_duration_s / 5)
    started = time.perf_counter()
    tries = 0
    while tries < warmup_tries or (time.perf_counter() - started) < warmup_floor:
        jax.block_until_ready(fn())
        tries += 1
    samples = []
    loop_started = time.perf_counter()
    runs = 0
    # `held` keeps the result alive PAST the timer and is cleared after it, so a ~5GB
    # deallocation cannot land inside a measurement. The upstream loop writes
    # `block_until_ready(fn())` and discards, freeing the previous output while the timer
    # runs -- harmless for its small fixed-shape all_to_all, not obviously so here.
    #
    # This did NOT explain the payload-scaling gap it was written for. Measured:
    #     before  +2423 / +2465 / +2634us at T=8192   after  +2455us
    # i.e. no change. Deallocation is ruled out as the cause; the gap between the
    # reference's host wall-clock and its own device span at the largest payload
    # (~2.4ms, against ~500us for every point below T=512) is device idle between
    # iterations that the host is not filling, and remains unexplained. Kept anyway,
    # because timing a deallocation is wrong on its own terms -- but it buys nothing.
    held = None
    while runs < num_runs or (time.perf_counter() - loop_started) < min_duration_s:
        start = time.perf_counter()
        held = jax.block_until_ready(fn())
        end = time.perf_counter()
        held = None  # freed outside the window, not inside the next measurement
        runs += 1
        if runs <= 1000:  # the reference retains at most 1000 samples
            samples.append((end - start) * 1000.0)
    return samples


def iqr_metrics(times_ms: list) -> dict | None:
    """avg/p50/p90/std after IQR outlier removal, as the reference reports them.

    CollectiveX's own components publish raw percentiles with no filtering; this keeps the
    reference's statistics so its numbers mean what its numbers mean. Both the raw and
    kept sample counts are recorded so the filtering is auditable.
    """
    if not times_ms:
        return None
    ordered = sorted(times_ms)
    kept = ordered
    if len(ordered) > 3:
        def _percentile(fraction):
            position = fraction * (len(ordered) - 1)
            low = int(position)
            high = min(low + 1, len(ordered) - 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

        q1, q3 = _percentile(0.25), _percentile(0.75)
        spread = q3 - q1
        bounded = [
            value for value in ordered
            if q1 - 1.5 * spread <= value <= q3 + 1.5 * spread
        ]
        kept = bounded or ordered  # the reference falls back if filtering empties the set
    mean = sum(kept) / len(kept)
    variance = sum((value - mean) ** 2 for value in kept) / len(kept)
    return {
        "avg_ms": mean,
        "p50_ms": kept[len(kept) // 2],
        "p90_ms": kept[min(len(kept) - 1, int(0.9 * (len(kept) - 1)))],
        "std_ms": variance ** 0.5,
        "samples": len(times_ms),
        "kept_after_iqr": len(kept),
    }


def egress_bytes(layout: Layout, hidden: int, value_bytes: int = 2) -> dict:
    """Per-device bytes that actually leave the device, the reference's byte convention.

    `AllToAllBenchmark` counts `chunk_size_bytes * (num_devices - 1)`: each device keeps
    its own chunk and sends the rest. The ragged analogue is each device's send total
    minus the copies whose destination is itself. Counting the self-copy would inflate
    bandwidth by ~1/ep_size against a published figure that excludes it.
    """
    per_device = [
        int(layout.send_total[rank] - layout.send_sizes[rank, rank]) * hidden * value_bytes
        for rank in range(layout.ep_size)
    ]
    return {
        "mean_per_device": sum(per_device) / len(per_device),
        "max_per_device": max(per_device),
        "total": sum(per_device),
    }
