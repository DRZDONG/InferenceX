#!/usr/bin/env python3
"""NumPy port of bench/routing.py for runtimes without PyTorch.

bench/routing.py is the reference generator, but it imports torch at module scope and
the TPU (JAX) image carries no usable torch. This module reproduces the SAME byte-stable
counter scheme -- keyed BLAKE2b over the coordinate tuple -- on numpy, so a TPU case and
a GPU case benchmark the identical routing trace and the identical activation bytes.

The two halves are a parity contract, not a coincidence: tests/test_tpu_probe.py asserts
element-for-element equality against the torch reference whenever torch is importable,
and pins a golden digest so the port cannot drift on machines where it is not. Any edit
here must be mirrored in bench/routing.py (and vice versa).

Activations are produced in float32. Every value the lattice emits is k/64 for integer
k in [-128, 128], which needs at most 8 significand bits, so the caller's cast to
bfloat16 is exact and matches the torch path bit for bit.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

_MASK64 = (1 << 64) - 1

SOURCE_ID_BITS = 32
SOURCE_ID_COLUMNS = SOURCE_ID_BITS


def build_global_routing(global_tokens: int, experts: int, topk: int, routing: str, seed: int):
    """Return one byte-stable counter-generated routing window (indices, weights)."""
    if routing != "uniform":
        raise ValueError(f"unknown routing {routing!r} (uniform)")
    if global_tokens <= 0 or experts <= 0 or topk <= 0 or topk > experts:
        raise ValueError("global_tokens/experts/topk must be positive and topk <= experts")

    key = (int(seed) & _MASK64).to_bytes(8, "little")

    def counter(token: int, slot: int, attempt: int, stream: int) -> int:
        message = struct.pack("<4Q", token, slot, attempt, stream)
        return int.from_bytes(
            hashlib.blake2b(message, key=key, digest_size=8).digest(), "little"
        )

    indices, weights = [], []
    for token in range(int(global_tokens)):
        selected, used = [], set()
        for slot in range(int(topk)):
            attempt = 0
            while True:
                expert = counter(token, slot, attempt, 0) % int(experts)
                if expert not in used:
                    used.add(expert)
                    selected.append(expert)
                    break
                attempt += 1
        raw = [1 + counter(token, slot, 0, 1) % 65535 for slot in range(int(topk))]
        denominator = float(sum(raw))
        indices.append(selected)
        weights.append([value / denominator for value in raw])
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def rank_slice(idx, weights, rank: int, tokens_per_rank: int):
    lo = rank * tokens_per_rank
    return (
        np.ascontiguousarray(idx[lo:lo + tokens_per_rank]),
        np.ascontiguousarray(weights[lo:lo + tokens_per_rank]),
    )


def rank_activations(tokens: int, hidden: int, seed: int, rank: int):
    """Exact counter-derived inputs with a quantization-safe source-token prefix."""
    source = np.arange(tokens, dtype=np.int64) + rank * tokens
    return activations_for_source_ids(source, hidden, seed)


def activations_for_source_ids(source, hidden: int, seed: int):
    """Materialize canonical float32 activations for global source-token IDs."""
    if hidden < SOURCE_ID_COLUMNS:
        raise ValueError(f"hidden must be at least {SOURCE_ID_COLUMNS}")
    source = np.asarray(source, dtype=np.int64)
    if source.size and (int(source.min()) < 0 or int(source.max()) >= (1 << SOURCE_ID_BITS)):
        raise ValueError("source token ID is outside the bounded identity contract")
    column = np.arange(hidden, dtype=np.int64)
    # Same frozen integer lattice as bench/routing.py: 131/17/19 are odd multipliers
    # coprime to the prime modulus 257, so distinct (source, column, seed) land on
    # distinct residues; %257-128 yields k in [-128, 128] and k/64 is exactly
    # representable in bfloat16.
    values = (source[:, None] * 131 + column[None, :] * 17 + int(seed) * 19) % 257 - 128
    output = values.astype(np.float32) * np.float32(1.0 / 64.0)
    source_columns = np.arange(SOURCE_ID_BITS, dtype=np.int64)
    source_bits = ((source[:, None] >> source_columns[None, :]) & 1) * 2 - 1
    output[:, :SOURCE_ID_BITS] = source_bits.astype(np.float32)
    return output


def decode_source_ids(payload, seed: int):
    """Decode and validate the source IDs carried by rank_activations."""
    payload = np.asarray(payload, dtype=np.float32)
    if payload.ndim != 2 or payload.shape[1] < SOURCE_ID_COLUMNS:
        raise ValueError("received payload cannot carry the source-token prefix")
    prefix = payload[:, :SOURCE_ID_COLUMNS]
    if not bool(np.isfinite(prefix).all()) or bool((np.abs(prefix) < 0.25).any()):
        raise ValueError("received source-token prefix is not quantization-stable")
    bits = (prefix >= 0).astype(np.int64)
    powers = (1 << np.arange(SOURCE_ID_BITS, dtype=np.int64))
    return (bits * powers[None, :]).sum(axis=1)


def _destination_onehot(idx, experts_per_rank: int, ep_size: int):
    """[global_tokens, ep_size] bool: which ranks a token has at least one expert on."""
    global_tokens = idx.shape[0]
    assignments = np.minimum(idx // experts_per_rank, ep_size - 1)
    destinations = np.zeros((global_tokens, ep_size), dtype=bool)
    np.put_along_axis(destinations, assignments, True, axis=1)
    return destinations


def routing_locality(idx, experts_per_rank: int, ep_size: int, tokens_per_rank: int,
                     gpus_per_node: int, scale_up_domain: int = None) -> dict:
    """Locality of rank-deduplicated payload copies under packed placement."""
    destinations = _destination_onehot(idx, experts_per_rank, ep_size)
    token, dest = np.nonzero(destinations)
    src = np.minimum(token // max(1, tokens_per_rank), ep_size - 1)
    sud = scale_up_domain or (gpus_per_node * ep_size)  # default: all one domain
    local = dest == src
    same_node = (dest // gpus_per_node) == (src // gpus_per_node)
    same_dom = (dest // sud) == (src // sud)
    return {
        "placement": "packed",
        "local_rank_fraction": float(local.mean()),
        "same_node_fraction": float(same_node.mean()),
        "same_scaleup_domain_fraction": float(same_dom.mean()),
        "cross_node_fraction": float((~same_node).mean()),
        "cross_domain_fraction": float((~same_dom).mean()),
        "gpus_per_node": gpus_per_node, "scale_up_domain": sud, "copies": int(dest.size),
    }


def routing_stats(idx, experts: int, experts_per_rank: int) -> dict:
    """Realized routing properties for the GLOBAL trace; idx is [global_tokens, topk]."""
    ep = max(1, experts // max(1, experts_per_rank))
    ranks = idx // experts_per_rank
    onehot = _destination_onehot(idx, experts_per_rank, ep)
    fanout = onehot.sum(axis=1)
    hist = np.bincount(fanout, minlength=ep + 1)[1:ep + 1].tolist()
    load = np.bincount(idx.reshape(-1), minlength=experts).astype(np.float64)
    # Expert assignments (compute load) stay separate from rank-deduplicated payload
    # copies (network load); conflating them overstates traffic when two experts share
    # a rank.
    assignment_load = np.bincount(
        np.minimum(ranks, ep - 1).reshape(-1), minlength=ep
    ).astype(np.float64)
    payload_load = onehot.sum(axis=0).astype(np.float64)

    def _cv(values):
        mean = float(values.mean())
        return float(values.std() / mean) if mean > 0 else 0.0

    mean_load = float(load.mean())
    return {
        "fanout_mean": float(fanout.mean()),
        "fanout_min": int(fanout.min()), "fanout_max": int(fanout.max()),
        "fanout_histogram": hist,
        "expert_assignments_per_rank": [int(x) for x in assignment_load.tolist()],
        "payload_copies_per_rank": [int(x) for x in payload_load.tolist()],
        "routed_copies": int(fanout.sum()),
        "expert_load_min": int(load.min()), "expert_load_max": int(load.max()),
        "expert_load_mean": mean_load, "expert_load_cv": _cv(load),
        "expert_assignment_rank_cv": _cv(assignment_load),
        "payload_rank_cv": _cv(payload_load),
        "hotspot_ratio": float(load.max() / mean_load) if mean_load > 0 else 0.0,
        "empty_expert_count": int((load == 0).sum()),
        "empty_rank_count": int((payload_load == 0).sum()),
    }
