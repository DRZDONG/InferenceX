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

Timing: there is no public JAX equivalent of ``torch.cuda.Event``, so the published latency
comes from the XLA profiler trace instead -- SPANS (first start to last end of a component's
scope, never a sum of op durations), reduced MAX across devices per occurrence, raw
percentiles. That is the same KIND of number as the GPU family's CUDA events: chip time,
carrying no host dispatch cost. ``measurement.timing_source`` is ``xla-device-trace-span``.

Host wall-clock around ``block_until_ready`` survives as the per-row ``host_latency_us``
cross-check and as an OPT-IN fallback (``--allow-host-fallback``). It is never the headline:
its per-call dispatch floor is payload-dependent and reaches thousands of microseconds, so a
point that yields no device span fails the case rather than quietly publishing a host figure
as though it were comparable. An earlier revision amortized N operations inside one program to
divide that floor; device spans made that unnecessary and the machinery is gone.

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
# The reference times the SAME ragged collective the probe does, so it matches the same
# HLO. It was briefly a DENSE all_to_all transcribed from AllToAllBenchmark, which needed
# its own `all-to-all` pattern; that made the reference a different collective from the one
# under test, which is the opposite of what a reference is for.
REFERENCE_HLO = TRACED_HLO


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


# FP8 codec, pinned to match the GPU backends so the precision axis is comparable:
# blockwise e4m3fn with one FP32 scale per 128-element block, which is DeepEP's
# `per_token_cast_to_fp8` (amax over the block, clamped at 1e-4, scale = amax/448).
# deepep-v2, uccl-ep and flashinfer-ep all use this recipe; a different block size or a
# per-tensor scale would move different bytes and stop the fp8 rows being comparable.
QUANT_BLOCK = 128
E4M3_MAX = 448.0
QUANT_AMAX_FLOOR = 1e-4


def quantize_scope(component: str, tokens_per_rank: int) -> str:
    """Trace label wrapping ONLY the bf16->fp8 cast.

    Production quantises once per forward pass immediately before the dispatch collective,
    so the cast is INSIDE the timed region -- the GPU harness charges it for the same
    reason. Scoped separately so what it costs is measured rather than folded into the
    permute, and because it is otherwise easy to misread: on the GPU side it is ~65us of
    the fp8-minus-bf16 delta at T=1 and negative by T=512, which reads as a fixed per-call
    charge. It is NOT fixed on TPU -- measured 12.5us at T=512 and 292.6us at T=8192,
    roughly linear in payload -- which is exactly why this scope publishes it per row
    instead of letting the platforms share one assumption about it.
    """
    return f"{SCOPE_PREFIX}-quantize-{component}-t{int(tokens_per_rank)}"


def permute_scope(component: str, tokens_per_rank: int) -> str:
    """Trace label wrapping ONLY the permute gather, a sibling of the transport scope.

    The component's non-collective time was derived by SUBTRACTION (component minus
    transport), on the reasoning that naming those ops is ambiguous. That reasoning broke
    down: dispatch's transport scope reports 6,689us where combine's reports 9,973us for
    the same collective moving the same bytes in the other direction, because XLA fuses
    the collective's second op into whatever is adjacent and dispatch has a gather to fuse
    into while combine does not. Subtraction then charges that op to the permute.

    Scoping the permute explicitly turns the question into a measurement: whichever scope
    the ~3,280us op lands in is the one it belongs to. The bare-collective reference
    program, which has NO permute, emits an op of the same size -- so the expectation is
    that it is collective work -- but expectation is what has been wrong five times here.
    """
    return f"{SCOPE_PREFIX}-permute-{component}-t{int(tokens_per_rank)}"


def chain_norm_scope(tokens_per_rank: int) -> str:
    """Trace label wrapping the chain's per-token renormalisation.

    Combine is an UNWEIGHTED rank-sum, so one chained pair multiplies token `t` by `d_t`,
    its number of unique destination ranks -- measured 3..7 (mean 5.35) at EP8 and 4..8
    (mean 6.52) at EP16 for deepseek-v3 top-8. Over the default 64 iterations that is
    `d_t**64`: bf16 saturates to +/-inf at iteration ~43 for d=8 and ~64 even for d=4, so
    EVERY token was infinite for the last third of the chain whose period we published.

    Dividing by `d_t` makes the pair exactly norm-preserving, which is what gives the chain
    an oracle at all: the sum of `d_t` copies of a bf16 value is exact in fp32, and IEEE
    division is correctly rounded, so `(d_t*v)/d_t` returns `v` BITWISE. The chain's output
    must therefore equal its input exactly, for any iteration count -- a check an all-inf
    output could never fail meaningfully.

    It is work production does not do, and it sits between combine and the next dispatch,
    i.e. INSIDE the measured period. So it gets its own scope and is published, rather than
    hidden in the period or asserted to be free.
    """
    return f"{SCOPE_PREFIX}-chainnorm-t{int(tokens_per_rank)}"


def chain_scope(tokens_per_rank: int) -> str:
    """Trace label wrapping the whole free-running dispatch->combine chain.

    The chained regime measures a different quantity from the fresh-entry components: the
    steady-state PERIOD of one pair in a pipeline that never drains, rather than the latency
    of one pair entered from idle. Its own scope keeps the two from pooling in a trace.
    """
    return f"{SCOPE_PREFIX}-chain-t{int(tokens_per_rank)}"


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
    copies_per_token: np.ndarray  # [ep, tokens_per_rank] int32, unique dest ranks per token
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

    # How many copies of each token combine will sum back -- the chain's growth factor.
    # Built by scatter-add over (src, local_token) rather than by reshaping `destinations`,
    # because `src` is CLAMPED above and a trailing partial rank would misalign a reshape.
    copies_per_token = np.zeros((ep_size, max(1, tokens_per_rank)), dtype=np.int32)
    np.add.at(copies_per_token, (src, local_token), 1)

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
        copies_per_token=copies_per_token,
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
    maturity = "candidate"
    combine_weight_semantics = "unweighted-rank-sum"
    dispatch_dtype = "bf16"
    combine_dtype = "bf16"
    dispatch_value_bytes = 2
    dispatch_scale_bytes_per_copy = 0
    # Class defaults so a transport built without going through __init__ (the test suite
    # constructs some via __new__) still answers the precision questions.
    precision = "bf16"
    fp8 = False
    # `native`: the expert consumes fp8 + scales directly, so no conversion pass sits
    # between the two collectives. See Point.roundtrip_fp8.
    fp8_consume = None

    def __init__(self, jax, jnp, shard_map, mesh, ep_size: int, hidden: int,
                 precision: str = "bf16"):
        if not ragged_all_to_all_available(jax):
            raise RuntimeError(
                "this JAX build has no jax.lax.ragged_all_to_all; upgrade the image"
            )
        if precision not in ("bf16", "fp8"):
            raise ValueError(f"unsupported precision {precision!r}")
        if precision == "fp8" and hidden % QUANT_BLOCK:
            raise ValueError(
                f"fp8 needs hidden divisible by {QUANT_BLOCK}; got {hidden}"
            )
        self.jax, self.jnp, self.shard_map = jax, jnp, shard_map
        self.mesh = mesh
        self.ep_size = ep_size
        self.hidden = hidden
        self.axis = "ep"
        self._P = jax.sharding.PartitionSpec
        self.precision = precision
        self.fp8 = precision == "fp8"
        if self.fp8:
            # Instance overrides: 1 byte per value, plus one FP32 scale per 128-block.
            # Combine stays BF16 -- the expert emits BF16 -- as it does on every GPU
            # backend, so only the dispatch direction changes.
            self.dispatch_dtype = "fp8-e4m3fn"
            self.dispatch_value_bytes = 1
            self.dispatch_scale_bytes_per_copy = (hidden // QUANT_BLOCK) * 4

    @property
    def quant_blocks(self) -> int:
        return self.hidden // QUANT_BLOCK

    def quantize_local(self, x_l):
        """bf16 -> (e4m3fn values, FP32 per-128-block scales), DeepEP's recipe.

        The wire's ONLY quantize. The oracle does not call this to build an expectation and
        must not start: identity between two XLA compilations of this expression does NOT
        hold on TPU, because `view * (448/amax)` may reassociate to `(view * 448) / amax`,
        one rounding against two, and the probe lattice hits exact e4m3 midpoints where a
        single f32 ulp flips the byte. Measured: the dispatch program produced 161/256 where
        a standalone program produced 152/256 for the same input. The GPU harness compares
        re-quantized bits only because assert_quantize_identity establishes that premise on
        metal first; this port has no such assertion, so correctness is anchored on arrived
        -vs-staged bytes and on source IDs instead.
        """
        jnp = self.jnp
        rows = x_l.shape[0]
        view = x_l.astype(jnp.float32).reshape(rows, self.quant_blocks, QUANT_BLOCK)
        amax = jnp.clip(jnp.max(jnp.abs(view), axis=2), QUANT_AMAX_FLOOR, None)
        values = (view * (E4M3_MAX / amax[:, :, None])).astype(jnp.float8_e4m3fn)
        return values.reshape(rows, self.hidden), (amax / E4M3_MAX)

    def dequantize_local(self, values, scales):
        """(e4m3fn, FP32 block scales) -> bf16, the inverse of quantize_local."""
        jnp = self.jnp
        rows = values.shape[0]
        view = values.astype(jnp.float32).reshape(rows, self.quant_blocks, QUANT_BLOCK)
        return (view * scales[:, :, None]).reshape(rows, self.hidden).astype(jnp.bfloat16)

    # ---- placement helpers ----------------------------------------------------

    def _shard(self, array):
        """Put a [ep, ...] host array on the mesh, sharded over the leading axis.

        Single process: `device_put` of the whole array, which is what EP8 has always done
        and stays byte-identical.

        Multi process (EP16 spans two hosts): each process may only place the shards it
        ADDRESSES. Every rank here holds the WHOLE [ep, ...] array, because the routing trace
        is byte-stable and every process derives the same layout from the same seed -- so the
        right API is `make_array_from_callback`, which asks for each addressable shard by
        index and lets this host answer from the global array it already has.

        NOT `make_array_from_process_local_data`: that takes the process's OWN slice, so
        handing it the global array from every process concatenates them. Measured on the
        2x2x2 slice, a 16-row global array became 32 rows and the collective rejected the
        operands -- `input_offsets must be rank 1 ... but got shape (2, 16)`, two rows per
        device where there should be one.
        """
        sharding = self.jax.sharding.NamedSharding(self.mesh, self._P(self.axis))
        if self.jax.process_count() == 1:
            return self.jax.device_put(self.jnp.asarray(array), sharding)
        host = np.asarray(array)
        return self.jax.make_array_from_callback(
            host.shape, sharding, lambda index: host[index],
        )

    def _mapped(self, fn, in_specs, out_specs, donate=()):
        """`out_specs` may be a tuple, for a program returning several arrays.

        fp8 dispatch returns (values, block scales) -- shard_map needs one spec per
        output, and passing a single spec for a tuple output fails at trace time rather
        than producing a wrong answer.
        """
        P = self._P
        sharded = P(self.axis)
        if isinstance(out_specs, tuple):
            mapped_out = tuple(sharded if spec else P() for spec in out_specs)
        else:
            mapped_out = sharded if out_specs else P()
        return self.jax.jit(
            self.shard_map(
                fn, mesh=self.mesh,
                in_specs=tuple(P(self.axis) if s else P() for s in in_specs),
                out_specs=mapped_out,
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
            # float32 and NOT a precomputed reciprocal: 1/d is inexact in fp32 for
            # d in {3,5,6,7}, so multiplying would lose the bitwise identity that is the
            # entire basis of the chain oracle. Correctly-rounded DIVISION keeps it.
            copies_per_token=self._shard(
                layout.copies_per_token.astype(np.float32)[:, :, None]),
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
                 return_offsets, copies_per_token=None):
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
        self.copies_per_token = copies_per_token
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

        def _quantized_stage_local(x_l, plan):
            """Quantize then permute: everything the fp8 dispatch does BEFORE the wire.

            Exposed so the oracle can compare what arrived against what was staged, which
            is bit-exact by actual construction. Re-deriving the wire bits by quantizing
            again cannot be: the wire's quantize is XLA-compiled inside a fused shard_map
            and the oracle's would be a second compilation, and `view * (448/amax)` is free
            to reassociate to `(view * 448) / amax` between them -- one rounding against
            two. This lattice manufactures exact e4m3 midpoints (block amax 4/64 with value
            3/64 scales to 336, precisely between the representable 320 and 352), where a
            single f32 ulp flips the output byte. The GPU harness only ever relied on
            bitwise identity because it ASSERTED it on metal first; this port inherited the
            assumption and dropped the assertion.
            """
            index, _, _, _, _, _, _ = plan
            with jax.named_scope(quantize_scope("dispatch", layout.tokens_per_rank)):
                values_l, scales_l = transport.quantize_local(x_l)
            with jax.named_scope(permute_scope("dispatch", layout.tokens_per_rank)):
                return (jnp.take(values_l, index, axis=0),
                        jnp.take(scales_l, index, axis=0))

        def _quantized_dispatch_local(x_l, plan):
            """fp8 dispatch: quantize, permute, then exchange values AND block scales.

            Production quantises once per forward immediately before the collective, so the
            cast is inside the timed region (the GPU harness charges it for the same
            reason). Two ragged exchanges, values and scales, which is the same tuple
            DeepEP's fp8 dispatch moves.

            Returns the staged operands alongside the received ones. The timed wrapper
            drops them; the oracle wrapper publishes them. Both therefore run THIS function
            -- one definition of the collective, one barrier -- so the bytes the oracle
            validates come off the same source as the bytes the profiler times. An earlier
            draft gave the oracle its own copy of these two collectives, which meant the
            byte-for-byte check covered a program whose latency was never published.
            """
            _, in_off, send_sz, out_off, recv_sz, _, _ = plan
            staged_values, staged_scales = _quantized_stage_local(x_l, plan)
            staged_values, staged_scales = jax.lax.optimization_barrier(
                (staged_values, staged_scales)
            )
            out_values = jnp.zeros((max_recv, hidden), dtype=staged_values.dtype)
            out_scales = jnp.zeros(
                (max_recv, transport.quant_blocks), dtype=staged_scales.dtype
            )
            with jax.named_scope(transport_scope("dispatch", layout.tokens_per_rank)):
                return (
                    jax.lax.ragged_all_to_all(
                        staged_values, out_values, in_off, send_sz, out_off, recv_sz,
                        axis_name=axis,
                    ),
                    jax.lax.ragged_all_to_all(
                        staged_scales, out_scales, in_off, send_sz, out_off, recv_sz,
                        axis_name=axis,
                    ),
                    staged_values,
                    staged_scales,
                )

        def _dispatch_local(x_l, plan):
            index, in_off, send_sz, out_off, recv_sz, recv_off, _ = plan
            # Permute gather: stage this device's outgoing copies in destination order.
            # Production pays this, so it stays inside the timed region -- but in its own
            # scope, so what it costs is measured rather than inferred by subtraction.
            with jax.named_scope(permute_scope("dispatch", layout.tokens_per_rank)):
                staged = jnp.take(x_l, index, axis=0)
            # Block fusion across the permute/collective boundary. LOAD-BEARING for
            # attribution: without it XLA fuses part of the collective into the gather and
            # dispatch's transport scope reports 6,685us where combine's reports 9,974us
            # for the same collective. Both readings of dispatch's span add up, so the span
            # alone could not decide between them:
            #     6,685 transport + 5,262 permute + 363 memset = 12,310
            #     9,965 transport + 1,979 permute + 363 memset = 12,307
            # The barrier decided it, and the answer was that the fusion is metadata-only:
            # measured, transport_us rose to 9,966 (matching combine's 9,974 to 0.08%),
            # permute_us fell to ~1,980, and the span did not rise -- it FELL by 466us. So
            # this is not measurement perturbing the thing measured, and removing it
            # silently restores a 1.49x under-attribution of the collective.
            staged = jax.lax.optimization_barrier(staged)
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
                if transport.fp8:
                    # Drop the staged operands: this is the program the profiler times, and
                    # it must carry no output the production path would not.
                    return _quantized_dispatch_local(x_g[0], _plan(plan_g))[:2]
                return _dispatch_local(x_g[0], _plan(plan_g))

        # recv_g arrives 2D now, being dispatch's own output fed back in.
        def combine(recv_g, *plan_g):
            with jax.named_scope(scope_name("combine", layout.tokens_per_rank)):
                return _combine_local(recv_g, _plan(plan_g))

        def roundtrip(x_g, *plan_g):
            plan = _plan(plan_g)
            with jax.named_scope(scope_name("roundtrip", layout.tokens_per_rank)):
                return _combine_local(_dispatch_local(x_g[0], plan), plan)

        def roundtrip_fp8(x_g, staged_g, *plan_g):
            """dispatch -> combine, with the fp8->bf16 conversion HOISTED OUT.

            `roundtrip` must mean transport-to-transport in every row or it cannot be
            compared, and under this workload the conversion is not transport. deepseek-v3
            block-fp8 takes the `native` path: the expert consumes the dispatched fp8 and
            per-128-block scales DIRECTLY and emits BF16, so no standalone conversion pass
            sits between the two collectives. SGLang's DeepEP dispatcher contains no dequant
            at all; vLLM skips it whenever the expert's block shape matches DeepEP's 128,
            which this workload's does. A standalone dequant is vLLM's fallback for a
            quant-format MISMATCH -- a real path, but not this one's, and the GPU harness
            keeps it strictly as a verification hatch.

            Charging it here compared fp8 and bf16 through structurally different pipelines
            and made fp8's roundtrip look WORSE than bf16's despite dispatch being 1.53x
            faster. The GPU side measured the same inversion in 39 of 51 comparisons before
            hoisting its `stage` out.

            So combine reads `staged_g`, the BF16 the expert would have handed it,
            materialized ONCE outside the timed region -- exactly what the GPU harness does
            by hoisting stage() out of the chain. The conversion is still measured, as the
            `stage` component, so `dequant roundtrip ~= roundtrip + stage` stays derivable.

            Keeping the dispatch ALIVE takes both of the things below, and a barrier alone
            is not enough. Measured, run 30890386745: with only the barrier, the fp8
            roundtrip came back at 16,511.5us against combine's 16,511.4 -- a lone combine.
            Its op_inventory was byte-for-byte combine's, and every op characteristic of
            dispatch (`ragged_all_to_all.38` at 3,347us, `ragged_all_to_all.24` at 1,002us,
            the quantize fusions) was absent. `optimization_barrier` constrains ORDERING; it
            does not make an unused tuple output live, so XLA deleted the whole dispatch.

            So:
              * the dispatch results are RETURNED, which XLA cannot eliminate. This is the
                same materialization the standalone dispatch program already pays, so it
                measures the same work rather than a cheaper variant of it.
              * the barrier stays, for ordering: without it the two collectives are
                independent and XLA is free to overlap them, which would understate a chain
                production runs in sequence.

            Verified per run, not assumed: dispatch's collectives must appear in the
            roundtrip's `op_inventory` beside combine's. Under fp8 they do not share
            combine's op NAME, so the check is that the inventory is not merely combine's --
            the bf16 roundtrip is the positive control at 2 collectives.
            """
            plan = _plan(plan_g)
            with jax.named_scope(scope_name("roundtrip", layout.tokens_per_rank)):
                values, scales, _, _ = _quantized_dispatch_local(x_g[0], plan)
                values, scales, staged = jax.lax.optimization_barrier(
                    (values, scales, staged_g)
                )
                return _combine_local(staged, plan), values, scales

        self._dispatch = transport._mapped(
            dispatch, (True,) * 8, (True, True) if transport.fp8 else True,
        )
        self._combine = transport._mapped(combine, (True,) * 8, True)
        # fp8's roundtrip takes the pre-staged BF16 as a second operand; see roundtrip_fp8.
        self._roundtrip = transport._mapped(
            roundtrip_fp8 if transport.fp8 else roundtrip,
            (True,) * (9 if transport.fp8 else 8),
            # fp8 returns the dispatch results too, so XLA cannot delete the dispatch.
            (True, True, True) if transport.fp8 else True,
        )
        def chain(x_g, copies_g, *plan_g, _iters):
            """`_iters` dispatch->combine pairs, free-running, in ONE compiled program.

            The carry is REAL: each iteration's combine output becomes the next iteration's
            dispatch operand. That data dependency is what keeps XLA from hoisting or CSE-ing
            the pair -- an `optimization_barrier` would NOT be enough, as measured in run
            30890386745 where a dispatch feeding only a barrier was deleted outright and the
            program published a lone combine as a roundtrip.

            `_iters` is a Python int closed over at trace time, so the loop count is baked
            into the compiled program. That is load-bearing on a multi-host slice: both
            processes then issue an identical sequence of slice-wide collectives. A count
            derived from anything host-local (a clock, a memory reading, that host's own
            trace) desynchronises them and libtpu kills the slice --
            SLICE_FAILURE_SW_INJECT_ERROR, as happened when reference_timing extended its
            loops by wall clock.
            """
            plan = _plan(plan_g)
            copies = copies_g[0]

            def body(_index, carry):
                received = _dispatch_local(carry, plan)
                summed = _combine_local(received, plan)
                # In fp32, before the cast back: combine already accumulates in fp32, and
                # `(d*v)/d == v` exactly there. Casting first would round twice.
                with jax.named_scope(chain_norm_scope(layout.tokens_per_rank)):
                    return (summed / copies).astype(carry.dtype)

            with jax.named_scope(chain_scope(layout.tokens_per_rank)):
                return jax.lax.fori_loop(0, _iters, body, x_g[0])

        # fp8 is deliberately NOT chained yet: its combine needs BF16, so a chain body would
        # have to carry the fp8->bf16 conversion inside the loop, and the period would then
        # include work the `native` contract says production does not do standalone. Getting
        # that wrong would publish a period that is not comparable to the GPU family's.
        # bf16 first, measured; fp8 once the contract question is settled rather than assumed.
        self._chain_builder = None if transport.fp8 else chain
        self._chain_programs = {}
        self._dispatched = None
        self._combine_input = None

        # Oracle-only, untimed. Returns the received AND the staged tensors from ONE
        # execution, which is the whole point: the comparison then tests the TRANSPORT, not
        # the quantize. A separate staging program would be a second compilation of the
        # same recipe, and compiled-vs-compiled divergence is precisely the premise this
        # redesign exists to stop relying on -- so this calls the SAME
        # `_quantized_dispatch_local` the timed program calls, and differs from it only in
        # which of that function's four outputs it keeps. It is still its own compilation,
        # which is why the received SCALES are additionally compared against the timed
        # program's own output; see _check_timed_scales_match in run_ep_jax.py.
        def _dispatch_with_staged(x_g, *plan_g):
            return _quantized_dispatch_local(x_g[0], _plan(plan_g))

        self._dispatch_oracle = transport._mapped(
            _dispatch_with_staged, (True,) * 8, (True, True, True, True),
        ) if transport.fp8 else None
        # fp8: the BF16 the expert would hand combine. Its own program so it can be
        # materialized ONCE, untimed, and hoisted out of both the standalone combine and the
        # chained roundtrip -- exactly what the GPU harness does with stage().
        #
        # Scoped as `stage`, which is where the GPU puts the same work: for deepep-v2 and
        # uccl-ep `stage` IS the fp8 conversion. Naming it that keeps
        # `dequant roundtrip ~= roundtrip + stage` derivable from a published row. The
        # derivation only runs in that direction: reconstructing native as
        # `dequant - stage` errs -11.6% at T=1 on the GPU corpus and only converges by
        # T=256, i.e. worst in exactly the decode regime the headline reports.
        def stage(v_g, s_g):
            with jax.named_scope(scope_name("stage", layout.tokens_per_rank)):
                return transport.dequantize_local(v_g, s_g)

        self._dequant = transport._mapped(
            stage, (True, True), True,
        ) if transport.fp8 else None

        self._plan_fn = _plan
        self._program_cache = {}
        self._reference = None

    def reference_program(self):
        """`jit(shard_map(named_scope(ragged_all_to_all)))` -- the reference METHOD, applied
        to the collective under test.

        What is transcribed from AllToAllBenchmark (read from source, collectives.py
        `_setup_jit_fn`) is its MEASUREMENT METHOD: the per-call host timing loop, the IQR
        outlier filter, and the per-device egress byte convention that excludes
        self-destined copies. Not its program. Upstream times a DENSE
        `jax.lax.all_to_all(split_axis=0, concat_axis=0, tiled=True)`, and that was tried
        here: it is a different primitive, ~11% cheaper per byte on 2% more bytes, so its
        figure could never be reconciled with the ragged `transport_us` the components
        publish. A cross-check has to time the same collective or it checks nothing.

        So this is RAGGED, and BF16 on both precisions -- it stages `self.x` unquantized, so
        an fp8 byte basis would understate its bandwidth ~1.94x.

        What IS inside the timed scope: the collective and its output-buffer memset
        (~363us at T=8192). What is NOT, having been measured and removed: the `[None]` on
        the result, which re-materialised the whole output every call (3,232us), and the
        `[0]` squeeze of the operand, which cost 0.8us and was removed anyway. Donating the
        output buffer was tried, honoured, and cost +2,900us. All of it is visible in
        `device_timing.reference.op_inventory` rather than asserted here.

        Returns (callable, staged_operand).
        """
        if getattr(self, "_reference", None) is not None:
            return self._reference
        jax, jnp = self._jax, self._jnp
        axis, hidden = self.t.axis, self.t.hidden
        max_recv = max(1, self.layout.max_recv)
        # RAGGED, the same collective the probe measures. The reference is the upstream
        # measurement METHOD -- per-call host timing loop, IQR filter, per-device egress
        # convention -- applied to the collective under test. Timing a DENSE all_to_all
        # instead was faithful to AllToAllBenchmark's program but benchmarked a different
        # primitive, which makes it uncomparable to dispatch.transport_us: dense ran ~11%
        # cheaper per byte and 2% heavier in bytes, so the two could not be reconciled.
        #
        # Everything that made the ragged version wrong before is gone: the operand is
        # staged 2D so no `[0]` squeeze copies it per call, and the result is returned 2D
        # so no `[None]` re-materialises it. What remains inside the scope is the
        # collective and its ~363us output memset, which the reference implementation
        # also pays.
        staged = self.t._mapped(
            lambda x_g, index_g: jnp.take(x_g[0], index_g[0], axis=0),
            (True, True), True,
        )(self.x, self.send_index)
        staged = jax.block_until_ready(staged)

        def transport(staged_g, in_off_g, send_sz_g, out_off_g, recv_sz_g):
            with jax.named_scope(REFERENCE_MARKER):
                out = jnp.zeros((max_recv, hidden), dtype=staged_g.dtype)
                return jax.lax.ragged_all_to_all(
                    staged_g, out, in_off_g[0], send_sz_g[0], out_off_g[0],
                    recv_sz_g[0], axis_name=axis,
                )

        program = self.t._mapped(transport, (True,) * 5, True)
        plan = (self.input_offsets, self.send_sizes,
                self.output_offsets, self.recv_sizes)

        def call():
            return program(staged, *plan)

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
        source = self.combine_input() if received is None else received
        return self._combine(source, *self._plan_arrays)

    def dispatch_with_staged(self):
        """(recv_values, recv_scales, staged_values, staged_scales) from ONE execution.

        fp8 only, oracle only, untimed. Comparing the first pair against the second tests
        whether the transport moved the staged bytes to the right offsets -- bit-exact by
        construction, and independent of how the quantize happened to be compiled, because
        both sides of the comparison came out of the same compiled program.
        """
        return self._jax.block_until_ready(
            self._dispatch_oracle(self.x, *self._plan_arrays)
        )

    def combine_input(self):
        """The BF16 combine input, dequantized once under fp8 (untimed prep).

        Combine is BF16 on every precision because the expert emits BF16, so under fp8 the
        dispatch result has to be dequantized before it. Doing it here, once, keeps that
        cast out of combine's timed region -- the same boundary the GPU harness draws with
        stage(). The CHAINED roundtrip cannot use this: see the note there.
        """
        received = self.dispatched()
        if not self.t.fp8:
            return received
        if self._combine_input is None:
            values, scales = received
            self._combine_input = self._jax.block_until_ready(
                self._dequant(values, scales)
            )
        return self._combine_input

    def chain(self, iters: int):
        """The compiled `iters`-pair chain, or None where the backend does not chain."""
        if self._chain_builder is None:
            return None
        if iters not in self._chain_programs:
            import functools  # noqa: PLC0415

            self._chain_programs[iters] = self.t._mapped(
                functools.partial(self._chain_builder, _iters=int(iters)),
                (True,) * 9, True,
            )
        program = self._chain_programs[iters]
        return lambda: program(self.x, self.copies_per_token, *self._plan_arrays)

    def roundtrip(self):
        """dispatch -> combine. Under fp8 the pre-staged BF16 rides along as an operand.

        `combine_input()` materializes it once and caches, so the conversion is paid outside
        every timed call -- the same boundary the GPU harness draws by hoisting stage().
        """
        if not self.t.fp8:
            return self._roundtrip(self.x, *self._plan_arrays)
        return self._roundtrip(self.x, self.combine_input(), *self._plan_arrays)

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
        operations = {
            "dispatch": self.dispatch,
            "combine": lambda: self.combine(self.combine_input()),
            "roundtrip": self.roundtrip,
        }
        if self.t.fp8:
            # The fp8->bf16 conversion, hoisted out of both combine and roundtrip and
            # measured on its own. Reads the CACHED dispatch output, so what is timed is the
            # conversion, not a re-dispatch.
            operations["stage"] = lambda: self._dequant(*self.dispatched())
        return operations[component]



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
                     min_duration_s: float = 1.0, lockstep: bool = False) -> list:
    """The reference harness's measurement loop (core/base.py), transcribed.

    Host wall-clock around a blocking call, per iteration -- no in-jit chaining. Warmup
    and measurement both continue past their try counts until a duration floor is met, so
    a fast operation still gets a statistically useful sample count.

    `lockstep` DISABLES those floors, and multi-process callers must set it. `fn` runs a
    collective over every device in the slice, so both hosts have to call it the SAME number
    of times; the floors make the count depend on each host's own wall clock, so one host
    ends up blocked in a collective the other never issues and libtpu aborts the slice with
    SLICE_FAILURE_SW_INJECT_ERROR. That is what broke EP16 at the Pass 1 -> Pass 2 boundary
    in runs 30944020533 and 30950584737 while 30945710335 got past it -- flaky exactly
    because it depends on whether the two clocks happen to agree on an iteration count.

    Under lockstep the sample count is fixed rather than duration-driven, so the host figure
    rests on fewer samples. That is acceptable here and only here: the published latency is
    the device span, and this host loop is its cross-check.
    """
    if lockstep:
        # Zeroing this zeroes the warmup floor with it, so BOTH loops fall back to their
        # fixed try counts and neither depends on this host's clock.
        min_duration_s = 0.0
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


def egress_bytes(layout: Layout, hidden: int, value_bytes: int = 2,
                 scale_bytes_per_copy: int = 0) -> dict:
    """Per-device bytes that actually leave the device, the reference's byte convention.

    `AllToAllBenchmark` counts `chunk_size_bytes * (num_devices - 1)`: each device keeps
    its own chunk and sends the rest. The ragged analogue is each device's send total
    minus the copies whose destination is itself. Counting the self-copy would inflate
    bandwidth by ~1/ep_size against a published figure that excludes it.
    """
    per_device = [
        int(layout.send_total[rank] - layout.send_sizes[rank, rank])
        * (hidden * value_bytes + scale_bytes_per_copy)
        for rank in range(layout.ep_size)
    ]
    return {
        "mean_per_device": sum(per_device) / len(per_device),
        "max_per_device": max(per_device),
        "total": sum(per_device),
    }
