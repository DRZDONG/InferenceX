#!/usr/bin/env python3
"""Tests for the TPU (JAX) probe: routing parity, exchange-plan algebra, and wiring.

The transport itself needs a TPU, so what is testable off-cluster is everything the
transport depends on being right:

  * routing_np is byte-identical to the torch reference bench/routing.py (asserted
    directly when torch is importable, pinned by digest when it is not);
  * the host-computed exchange plan is self-consistent and its offset algebra really does
    round-trip -- simulated in pure numpy, so a sign or transpose error in
    ep_jax.build_layout fails here rather than on a TPU node twenty minutes later;
  * the neutral case codec's argv parses in run_ep_jax.py, and the registry entry points
    at a launcher that exists.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

COLLX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COLLX / "bench"))
sys.path.insert(0, str(COLLX / "runtime"))

import numpy as np  # noqa: E402

import config  # noqa: E402
import ep_harness  # noqa: E402
import ep_jax  # noqa: E402
import routing_np  # noqa: E402
import run_ep_jax  # noqa: E402
import xprof  # noqa: E402
import run_ep_jax_shard  # noqa: E402

try:
    import torch  # noqa: E402

    import routing as routing_torch  # noqa: E402
except ImportError:  # the TPU image and most dev boxes have no torch
    torch = None
    routing_torch = None


# A small but non-degenerate trace: 4 ranks, 8 experts (2 per rank), top-3 over 4 tokens
# per rank. Small enough to simulate exhaustively; chosen because at this size the plan
# contains BOTH an empty (src, dst) pair and pairs of differing size, which is exactly
# the ragged shape a uniform full-size trace would hide.
TRACE = dict(tokens_per_rank=4, ep_size=4, experts=8, topk=3, seed=67)


def _trace():
    return routing_np.build_global_routing(
        TRACE["tokens_per_rank"] * TRACE["ep_size"], TRACE["experts"], TRACE["topk"],
        "uniform", TRACE["seed"],
    )


def _layout():
    idx, _ = _trace()
    return idx, ep_jax.build_layout(
        idx, TRACE["tokens_per_rank"], TRACE["ep_size"],
        TRACE["experts"] // TRACE["ep_size"],
    )


class RoutingParityTests(unittest.TestCase):
    """routing_np must be the same generator as bench/routing.py, not merely similar."""

    # Digest of the numpy port's own output for TRACE. It pins the port against silent
    # drift on machines without torch; the torch tests below are what establish that the
    # pinned bytes are the REFERENCE bytes.
    GOLDEN = "6c6c3350805252e7f587b208c9d9192a9725cacaeeb1a35215fd1216a71e6765"

    @staticmethod
    def _digest(*arrays) -> str:
        digest = hashlib.sha256()
        for array in arrays:
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def test_golden_digest_pins_the_generator(self) -> None:
        idx, weights = _trace()
        activations = routing_np.rank_activations(
            TRACE["tokens_per_rank"], 64, TRACE["seed"], 2,
        )
        self.assertEqual(self._digest(idx, weights, activations), self.GOLDEN)

    @unittest.skipUnless(torch is not None, "torch is unavailable")
    def test_routing_matches_the_torch_reference(self) -> None:
        idx, weights = _trace()
        idx_t, weights_t = routing_torch.build_global_routing(
            TRACE["tokens_per_rank"] * TRACE["ep_size"], TRACE["experts"],
            TRACE["topk"], "uniform", TRACE["seed"],
        )
        np.testing.assert_array_equal(idx, idx_t.numpy())
        np.testing.assert_array_equal(weights, weights_t.numpy())

    @unittest.skipUnless(torch is not None, "torch is unavailable")
    def test_activations_match_the_torch_reference_exactly(self) -> None:
        # float32 here, bfloat16 there: every lattice value is k/64 with |k| <= 128, so
        # the cast is exact and the comparison is bit-for-bit, not approximate.
        for rank in range(TRACE["ep_size"]):
            with self.subTest(rank=rank):
                mine = routing_np.rank_activations(
                    TRACE["tokens_per_rank"], 128, TRACE["seed"], rank,
                )
                reference = routing_torch.rank_activations(
                    TRACE["tokens_per_rank"], 128, TRACE["seed"], rank, "cpu",
                )
                np.testing.assert_array_equal(
                    mine, reference.to(torch.float32).numpy()
                )

    @unittest.skipUnless(torch is not None, "torch is unavailable")
    def test_routing_stats_and_locality_match_the_torch_reference(self) -> None:
        idx, _ = _trace()
        experts_per_rank = TRACE["experts"] // TRACE["ep_size"]
        mine = routing_np.routing_stats(idx, TRACE["experts"], experts_per_rank)
        reference = routing_torch.routing_stats(
            torch.from_numpy(idx), TRACE["experts"], experts_per_rank,
        )
        self.assertEqual(set(mine), set(reference))
        for key in reference:
            with self.subTest(stat=key):
                if isinstance(reference[key], float):
                    self.assertAlmostEqual(mine[key], reference[key], places=6)
                else:
                    self.assertEqual(mine[key], reference[key])
        mine_locality = routing_np.routing_locality(
            idx, experts_per_rank, TRACE["ep_size"], TRACE["tokens_per_rank"], 4, 4,
        )
        reference_locality = routing_torch.routing_locality(
            torch.from_numpy(idx), experts_per_rank, TRACE["ep_size"],
            TRACE["tokens_per_rank"], 4, 4,
        )
        for key in reference_locality:
            with self.subTest(locality=key):
                if isinstance(reference_locality[key], float):
                    self.assertAlmostEqual(
                        mine_locality[key], reference_locality[key], places=6
                    )
                else:
                    self.assertEqual(mine_locality[key], reference_locality[key])

    def test_source_ids_survive_a_decode(self) -> None:
        activations = routing_np.rank_activations(
            TRACE["tokens_per_rank"], 64, TRACE["seed"], 3,
        )
        decoded = routing_np.decode_source_ids(activations, TRACE["seed"])
        expected = np.arange(TRACE["tokens_per_rank"]) + 3 * TRACE["tokens_per_rank"]
        np.testing.assert_array_equal(decoded, expected)


class ExchangePlanTests(unittest.TestCase):
    """The offset algebra ep_jax.build_layout hands to ragged_all_to_all."""

    def test_plan_is_internally_consistent(self) -> None:
        idx, layout = _layout()
        ep = TRACE["ep_size"]
        # One copy per unique (token, destination rank) pair, deduplicated across top-k.
        destinations = routing_np._destination_onehot(
            idx, TRACE["experts"] // ep, ep,
        )
        self.assertEqual(layout.routed_copies, int(destinations.sum()))
        self.assertEqual(int(layout.send_sizes.sum()), layout.routed_copies)
        # recv is the transpose of send, and both offset tables are exclusive prefixes.
        np.testing.assert_array_equal(layout.recv_sizes, layout.send_sizes.T)
        for sizes, offsets in ((layout.send_sizes, layout.input_offsets),
                               (layout.recv_sizes, layout.recv_offsets)):
            expected = np.zeros_like(offsets)
            expected[:, 1:] = np.cumsum(sizes, axis=1)[:, :-1]
            np.testing.assert_array_equal(offsets, expected)
        # The remote-offset tables are the transposes the two directions need.
        np.testing.assert_array_equal(layout.output_offsets, layout.recv_offsets.T)
        np.testing.assert_array_equal(layout.return_offsets, layout.input_offsets.T)
        np.testing.assert_array_equal(layout.send_total, layout.send_sizes.sum(axis=1))
        np.testing.assert_array_equal(layout.recv_total, layout.recv_sizes.sum(axis=1))
        # A trace this wide must exercise both an empty pair and an uneven one, or the
        # plan is not actually being tested against the ragged case.
        self.assertGreater(int((layout.send_sizes == 0).sum()), 0)
        self.assertGreater(layout.send_sizes.max(), layout.send_sizes.min())

    def test_send_index_names_the_right_tokens_in_destination_order(self) -> None:
        idx, layout = _layout()
        ep, per_rank = TRACE["ep_size"], TRACE["experts"] // TRACE["ep_size"]
        destinations = routing_np._destination_onehot(idx, per_rank, ep)
        token, dest = np.nonzero(destinations)
        src = token // TRACE["tokens_per_rank"]
        for rank in range(ep):
            for destination in range(ep):
                size = int(layout.send_sizes[rank, destination])
                start = int(layout.input_offsets[rank, destination])
                chunk = layout.send_index[rank, start:start + size]
                expected = np.sort(
                    token[(src == rank) & (dest == destination)]
                    - rank * TRACE["tokens_per_rank"]
                )
                np.testing.assert_array_equal(np.sort(chunk), expected)

    def test_plan_round_trips_in_a_pure_numpy_simulation(self) -> None:
        """Simulate both ragged_all_to_all calls on host and check the identity.

        This is the test that would have caught a transposed output_offsets: it walks the
        exact same six arrays the device program passes to the primitive, moves the rows
        by hand, and requires that combine reconstructs fanout(token) * x[token].
        """
        idx, layout = _layout()
        ep, tokens, hidden = TRACE["ep_size"], TRACE["tokens_per_rank"], 64
        activations = np.stack([
            routing_np.rank_activations(tokens, hidden, TRACE["seed"], rank)
            for rank in range(ep)
        ])

        # Dispatch: gather into destination order, then place each chunk into the
        # DESTINATION's receive buffer at output_offsets[src][dst].
        staged = np.stack([
            activations[rank][layout.send_index[rank]] for rank in range(ep)
        ])
        received = np.zeros((ep, max(1, layout.max_recv), hidden), dtype=np.float32)
        for source in range(ep):
            for destination in range(ep):
                size = int(layout.send_sizes[source, destination])
                if not size:
                    continue
                start = int(layout.input_offsets[source, destination])
                landing = int(layout.output_offsets[source, destination])
                received[destination, landing:landing + size] = \
                    staged[source, start:start + size]

        # Every received row must decode to the source token the plan promised.
        for rank in range(ep):
            rows = int(layout.recv_total[rank])
            source_ids = routing_np.decode_source_ids(
                received[rank, :rows], TRACE["seed"],
            )
            for source in range(ep):
                size = int(layout.recv_sizes[rank, source])
                if not size:
                    continue
                start = int(layout.recv_offsets[rank, source])
                sent_from = int(layout.input_offsets[source, rank])
                want = (layout.send_index[source, sent_from:sent_from + size]
                        + source * tokens)
                np.testing.assert_array_equal(
                    np.sort(source_ids[start:start + size]), np.sort(want)
                )

        # Combine: the same exchange backwards, landing at return_offsets[dst][src].
        returned = np.zeros((ep, max(1, layout.max_send), hidden), dtype=np.float32)
        for holder in range(ep):
            for origin in range(ep):
                size = int(layout.recv_sizes[holder, origin])
                if not size:
                    continue
                start = int(layout.recv_offsets[holder, origin])
                landing = int(layout.return_offsets[holder, origin])
                returned[origin, landing:landing + size] = \
                    received[holder, start:start + size]
        # Round-tripping must restore each device's own staged buffer exactly.
        for rank in range(ep):
            total = int(layout.send_total[rank])
            np.testing.assert_array_equal(
                returned[rank, :total], staged[rank, :total]
            )

        # Unweighted rank-sum, and the expected value the probe oracle asserts.
        for rank in range(ep):
            combined = np.zeros((tokens, hidden), dtype=np.float32)
            total = int(layout.send_total[rank])
            np.add.at(combined, layout.send_index[rank, :total], returned[rank, :total])
            expected = run_ep_jax._expected_combine(activations[rank], layout, rank)
            np.testing.assert_array_equal(combined, expected)

    def test_payload_budget_reports_dropped_points_instead_of_truncating(self) -> None:
        idx, _ = _layout()
        layouts = {}

        def layout_for(tokens: int):
            if tokens not in layouts:
                trace, _ = routing_np.build_global_routing(
                    tokens * TRACE["ep_size"], TRACE["experts"], TRACE["topk"],
                    "uniform", TRACE["seed"],
                )
                layouts[tokens] = ep_jax.build_layout(
                    trace, tokens, TRACE["ep_size"],
                    TRACE["experts"] // TRACE["ep_size"],
                )
            return layouts[tokens]

        kept, dropped = run_ep_jax._ladder_within_budget(
            [1, 2, 4, 8], layout_for, 7168, 64 * 1024,
        )
        self.assertEqual(sorted(kept + dropped), [1, 2, 4, 8])
        self.assertTrue(dropped, "a 64 KiB budget must drop the larger points")
        # Kept points are a prefix: the budget is monotone in tokens per rank.
        self.assertEqual(kept, sorted(kept))
        self.assertLess(max(kept, default=0), min(dropped))


def _uncommented(text: str, needle: str) -> bool:
    """Is `needle` present on a line where nothing comments it out?

    A plain `assertIn` passed a mutation that inserted `#` directly before the directive,
    because the directive sits inside a shell string assignment so the `#` lands mid-line
    and neither a substring test nor a "line starts with #" test sees it.
    """
    for line in text.splitlines():
        head, separator, _tail = line.partition(needle)
        if separator and "#" not in head:
            return True
    return False


class TpuWiringTests(unittest.TestCase):
    """The registry entry, the codec, and the entrypoint have to agree."""

    REGISTRY = json.loads(
        (COLLX / "configs" / "platform_config.json").read_text(encoding="utf-8")
    )["platforms"]

    def test_tpu_runtime_is_registered_with_an_entrypoint_and_a_launcher(self) -> None:
        self.assertIn("tpu", config.ACCELERATOR_RUNTIMES)
        tpu_skus = {
            name: entry for name, entry in self.REGISTRY.items()
            if entry["runtime"] == "tpu"
        }
        self.assertTrue(tpu_skus, "expected at least one tpu-runtime SKU")
        for name, entry in tpu_skus.items():
            with self.subTest(sku=name):
                launcher = COLLX / "launchers" / f"launch_{entry['launcher']}.sh"
                self.assertTrue(launcher.is_file(), f"missing {launcher.name}")
                # A tpu SKU must only offer backends the JAX entrypoint accepts, or the
                # matrix would schedule a case nothing can run.
                for backend in entry["backends"]:
                    self.assertIn(backend, _jax_backends())
                # A cross-host EP degree may only be offered if the launch path can
                # actually provide a multi-host slice AND the entrypoint joins one JAX
                # cluster across it. The matrix scheduling something the runtime refuses is
                # exactly how an fp8 shard once claimed a TPU pod and died 48s in.
                text = launcher.read_text()
                entrypoint = (COLLX / "bench" / "run_ep_jax.py").read_text()
                cross_host = any(degree > entry["gpus_per_node"]
                                 for degrees in entry["backends"].values()
                                 for degree in degrees)
                if cross_host:
                    for needed in ("completionMode: Indexed", "clusterIP: None",
                                   "parallelism:", "subdomain:"):
                        self.assertTrue(
                            _uncommented(text, needed),
                            f"{launcher.name} offers cross-host EP without an active "
                            f"{needed!r}")
                    self.assertIn("jax.distributed.initialize", entrypoint,
                                  "cross-host EP without joining one JAX cluster would "
                                  "measure N independent single-host exchanges")
                    self.assertIn("jax.devices()", entrypoint)
                else:
                    self.assertFalse(_uncommented(text, "completionMode: Indexed"))

    def test_case_args_round_trips_through_the_jax_parser(self) -> None:
        case = {
            "backend": "jax-ragged-a2a", "mode": "normal", "precision": "bf16",
            "phase": "decode", "routing": "uniform", "ep": 8, "nodes": 1,
            "gpus_per_node": 8, "scale_up_domain": 8, "scope": "scale-up",
            "scale_up_transport": "ici", "scale_out_transport": None,
            "transport": "ici", "topology_class": "tpuv7-ici-island",
            "hidden": 7168, "topk": 8, "experts": 256, "seed": 67,
            "ladder": "1 2 4", "timing": "8:256:32",
            "suite": "ep-core", "workload": "deepseek-v3",
        }
        case["case_id"] = ep_harness.case_id("tpuv7", case)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.json"
            path.write_text(json.dumps({"version": 1, "cases": [case]}))
            result = subprocess.run(
                [sys.executable, str(COLLX / "runtime" / "config.py"), "case-args",
                 str(path), "0", "tpuv7", "TS", "8", "1", "8", "8"],
                capture_output=True, check=True,
            )
        parts = result.stdout.split(b"\0")
        self.assertEqual(parts[-1], b"")
        argv = [part.decode() for part in parts[:-1]]
        args = _jax_parser().parse_args(argv)
        self.assertEqual(args.backend, "jax-ragged-a2a")
        self.assertEqual(args.case_id, case["case_id"])
        self.assertEqual(args.scale_up_transport, "ici")
        self.assertEqual(args.gpus_per_node, 8)
        self.assertEqual(args.xprof_iters, 20)
        self.assertEqual(args.out, f"results/{case['case_id']}_TS-c000.json")

    def test_terminal_failure_document_needs_no_jax_initialization(self) -> None:
        case = {
            "backend": "jax-ragged-a2a", "mode": "normal", "precision": "bf16",
            "phase": "prefill", "routing": "uniform", "ep": 8, "nodes": 1,
            "gpus_per_node": 8, "scale_up_domain": 8, "scope": "scale-up",
            "scale_up_transport": "ici", "scale_out_transport": None,
            "transport": "ici", "topology_class": "tpuv7-ici-island",
            "hidden": 7168, "topk": 8, "experts": 256, "seed": 67,
            "ladder": "1 2 4", "timing": "8:256:32",
            "suite": "ep-core", "workload": "deepseek-v3",
        }
        case["case_id"] = ep_harness.case_id("tpuv7", case)
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard.json"
            out = Path(directory) / "failed.json"
            shard.write_text(json.dumps({"version": 1, "cases": [case]}))
            argv = config.case_argv(
                str(shard), 0, "tpuv7", "TS", "8", "1", "8", "8",
            )
            argv[argv.index("--out") + 1] = str(out)
            environment = {
                **os.environ,
                "COLLX_ATTEMPT_ID": "1",
                "COLLX_NODES": "1",
                "COLLX_RUNTIME": "tpu",
                "COLLX_VENDOR": "google",
                # If terminal mode accidentally initializes JAX this invalid backend
                # makes the regression fail rather than silently using local CPU.
                "JAX_PLATFORMS": "definitely-not-a-platform",
            }
            subprocess.run(
                [
                    sys.executable,
                    str(COLLX / "bench" / "run_ep_jax.py"),
                    "--terminal-failure",
                    "case-timeout",
                    *argv,
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            document = json.loads(out.read_text())
            summary = subprocess.run(
                [
                    sys.executable,
                    str(COLLX / "summarize.py"),
                    "--results-dir",
                    directory,
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        self.assertEqual(document["outcome"], {
            "reasons": ["case-timeout"], "status": "failed",
        })
        self.assertEqual(document["measurement"]["rows"], [])
        self.assertEqual(document["identity"]["case_id"], case["case_id"])
        self.assertEqual(document["runtime"]["vendor"], "google")
        self.assertIn("| failed | - | - | - | - | - |", summary)
        self.assertEqual(document["workload"]["ladder_measured"], [])
        self.assertEqual(document["workload"]["ladder_dropped"], [])
        self.assertIsNone(document["workload"]["ladder_cap"])
        self.assertEqual(document["implementation"]["combine_reduction"], "domain-fp32")
        self.assertIsNone(document["implementation"]["library_version"])

    def test_shard_runner_reuses_one_process_for_every_case(self) -> None:
        observed = []

        def fake_case_argv(_path, index, *_placement):
            return ["--phase", ["decode", "prefill"][index]]

        def fake_main():
            observed.append((list(sys.argv), os.environ["COLLX_ATTEMPT_ID"]))
            return 0

        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard.json"
            shard.write_text(json.dumps({
                "version": 1,
                "cases": [
                    {"backend": "jax-ragged-a2a"},
                    {"backend": "jax-ragged-a2a"},
                ],
            }))
            argv = [
                "run_ep_jax_shard.py",
                "--shard", str(shard),
                "--runner", "tpuv7",
                "--timestamp", "TS",
                "--ngpus", "8",
                "--nodes", "1",
                "--gpus-per-node", "8",
                "--scale-up-domain", "8",
            ]
            with (
                mock.patch.object(run_ep_jax_shard.config, "case_argv",
                                  side_effect=fake_case_argv),
                mock.patch.object(run_ep_jax_shard.run_ep_jax, "main",
                                  side_effect=fake_main),
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("COLLX_ATTEMPT_ID", None)
                self.assertEqual(run_ep_jax_shard.main(), 0)
                self.assertNotIn("COLLX_ATTEMPT_ID", os.environ)

        self.assertEqual([attempt for _argv, attempt in observed], ["1", "2"])
        self.assertEqual(
            [argv[-2:] for argv, _attempt in observed],
            [["--phase", "decode"], ["--phase", "prefill"]],
        )
        # The shard driver forwards the neutral codec's argv verbatim and prepends only
        # the entrypoint path -- it adds no per-case flags of its own now that
        # amortization is gone.
        for argv, _attempt in observed:
            self.assertTrue(argv[0].endswith("run_ep_jax.py"))
            self.assertEqual(len(argv), 3, f"unexpected extra flags: {argv}")

    def test_matrix_schedules_the_tpu_sku(self) -> None:
        sys.path.insert(0, str(COLLX))
        import sweep_matrix  # noqa: PLC0415

        document = sweep_matrix.resolve_matrix(backend="all", only_sku="tpuv7")
        runnable = [
            item for item in document["requested_cases"]
            if item["disposition"] == "runnable"
        ]
        self.assertTrue(runnable)
        for item in runnable:
            self.assertEqual(item["case"]["backend"], "jax-ragged-a2a")
            self.assertIn(item["case"]["ep"], (8, 16))
            # BOTH degrees are scale-up on TPU, and that is the whole point of the "ep"
            # domain: ICI reaches every chip in a slice, so EP16 spanning two hosts does
            # NOT leave the scale-up fabric. A flat domain of 8 would have labelled it
            # scale-out over DCN, which is false, and would have made the row look like a
            # b200/h100 EP16 (which really is RDMA) instead of a gb200 EP16 (which is not).
            self.assertEqual(item["case"]["scope"], "scale-up")
            self.assertEqual(item["case"]["scale_up_domain"], item["case"]["ep"])
            self.assertIsNone(item["case"]["scale_out_transport"])
        # BOTH precisions schedule now. This asserted bf16-only, which encoded the absence
        # of a quantized dispatch path as though it were a property of the SKU -- TPU v7
        # runs FP8 models elsewhere in this repo.
        self.assertEqual({item["case"]["precision"] for item in runnable},
                         {"bf16", "fp8"})
        # Both degrees now schedule; nothing is left as an unsupported coverage row.
        self.assertEqual({item["case"]["ep"] for item in runnable}, {8, 16})
        self.assertEqual(
            [item for item in document["requested_cases"]
             if item["disposition"] == "unsupported"], [])
        # EP16 is a two-host slice, so it must shard separately from EP8.
        self.assertEqual({shard["nodes"] for shard in document["include"]}, {1, 2})
        for shard in document["include"]:
            self.assertEqual(shard["launcher"], "tpu-gke")

    def test_tpu_launcher_uses_one_jax_process_and_records_missing_cases(self) -> None:
        launcher = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertIn("python3 -u bench/run_ep_jax_shard.py", launcher)
        self.assertIn("--terminal-failure", launcher)
        self.assertIn("case-timeout", launcher)

    def test_tpu_launcher_does_not_reuse_a_cross_run_compile_cache(self) -> None:
        """A cache hit can return a binary whose op metadata predates this source.

        The profiler attributes spans by the CollectiveX scope baked into that metadata,
        so reusing the node-local cache across runs silently cost device timing on every
        ladder point whose program came back from the cache -- pinned at decode T<=16 /
        prefill T<=2048 for three consecutive runs, while a just-edited program was
        captured at all 14. Opt-in only, and the export must stay behind the guard.
        """
        launcher = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertIn('COMPILE_CACHE="${COLLX_TPU_COMPILE_CACHE:-0}"', launcher)
        for line in launcher.splitlines():
            if "export JAX_COMPILATION_CACHE_DIR" in line:
                self.assertRegex(line, r"^\s{16,}export",
                                 "the export must be indented inside the guard, not run "
                                 "unconditionally")
        guard = launcher.index('if [ "${COMPILE_CACHE}" = "1" ]')
        self.assertLess(guard, launcher.index("export JAX_COMPILATION_CACHE_DIR="),
                        "the guard must precede the export")


class HostTimingTests(unittest.TestCase):
    """Host wall-clock timing, now the fallback rather than the published latency.

    The device trace supplies the headline; these cover the sanity-check path and the
    reference harness's statistics, both of which still have to be right.
    """

    @staticmethod
    def _fake_jax(*, barrier: bool = True, ragged: bool = True):
        lax = types.SimpleNamespace()
        if barrier:
            lax.optimization_barrier = lambda values: values
        if ragged:
            lax.ragged_all_to_all = lambda *a, **k: None
        return types.SimpleNamespace(lax=lax, block_until_ready=lambda value: value)






    def test_case_label_reports_mixed_rather_than_the_better_source(self) -> None:
        """Regression: a whole-case label claimed device timing for host-timed rows.

        Run 30741278607 published `xla-device-trace-span` for the case while half its
        decode rows had silently fallen back to host wall-clock — identical to
        `host_latency_us`, ~500 µs floor and all. Provenance is per component now, and
        the case label must summarise honestly rather than pick the flattering one.
        """
        device = run_ep_jax.DEVICE_TIMING_SOURCE
        host = run_ep_jax.TIMING_SOURCE
        all_device = [{"timing_source": {"dispatch": device, "combine": device}}]
        self.assertEqual(run_ep_jax._case_timing_source(all_device), device)
        all_host = [{"timing_source": {"dispatch": host, "combine": host}}]
        self.assertEqual(run_ep_jax._case_timing_source(all_host), host)
        # One host-timed component anywhere makes the case mixed, never "device".
        mixed = [
            {"timing_source": {"dispatch": device, "combine": device}},
            {"timing_source": {"dispatch": host, "combine": device}},
        ]
        self.assertEqual(run_ep_jax._case_timing_source(mixed), "mixed")
        self.assertEqual(run_ep_jax._case_timing_source([]), host)

    def test_point_programs_construct_against_stubs(self) -> None:
        """Walk every program-building path with stubs, catching attribute typos.

        `reference_program` shipped calling `self._shard` when `_shard` lives on the
        transport, not on Point -- an AttributeError that only surfaced on the TPU after
        a full run. Nothing here needs a real device: constructing the programs exercises
        the attribute graph, which is where that class of mistake lives.
        """
        stub = _stub_jax()
        transport = ep_jax.JaxEPTransport.__new__(ep_jax.JaxEPTransport)
        transport.jax, transport.jnp = stub, stub.numpy
        transport.mesh, transport.ep_size, transport.hidden = object(), 4, 64
        transport.axis = "ep"
        transport._P = stub.sharding.PartitionSpec
        transport.shard_map = lambda fn, **kwargs: fn
        _, layout = _layout()

        point = ep_jax.Point(
            transport=transport, layout=layout, x=stub.array(),
            send_index=stub.array(), input_offsets=stub.array(),
            send_sizes=stub.array(), output_offsets=stub.array(),
            recv_sizes=stub.array(), recv_offsets=stub.array(),
            return_offsets=stub.array(),
        )
        for component in ("dispatch", "combine", "roundtrip"):
            self.assertTrue(callable(point.timed_operation(component)))
        call, staged = point.reference_program()
        self.assertTrue(callable(call))
        self.assertIsNotNone(staged)

    def test_reference_program_does_not_reshape_the_collective_result(self) -> None:
        """No `[None]` on the collective's output: it costs a full copy of the buffer.

        Expanding the per-shard result to give shard_map a leading mapped axis
        re-materialised all 0.624GB of the output on every call -- 3,232us of a 15,084us
        figure at T=8192, measured by op_inventory. Returning the 2D result and letting
        shard_map concatenate is free. This is indexing behaviour, visible without a TPU,
        and it was mistaken for a memset for three hardware runs.
        """
        stub = _stub_jax()
        indexed = []

        class Result:
            dtype = "bfloat16"

            def __getitem__(self, index):
                indexed.append(index)
                return self

            def reshape(self, *_a, **_k):
                return self

        result = Result()
        stub.lax.all_to_all = lambda *a, **k: result

        transport = ep_jax.JaxEPTransport.__new__(ep_jax.JaxEPTransport)
        transport.jax, transport.jnp = stub, stub.numpy
        transport.mesh, transport.ep_size, transport.hidden = object(), 4, 64
        transport.axis = "ep"
        transport._P = stub.sharding.PartitionSpec
        transport.shard_map = lambda fn, **kwargs: fn
        _, layout = _layout()
        point = ep_jax.Point(
            transport=transport, layout=layout, x=stub.array(),
            send_index=stub.array(), input_offsets=stub.array(),
            send_sizes=stub.array(), output_offsets=stub.array(),
            recv_sizes=stub.array(), recv_offsets=stub.array(),
            return_offsets=stub.array(),
        )
        call, _ = point.reference_program()
        call()
        self.assertNotIn(None, indexed,
                         "the collective result must be returned as-is, not [None]")

    def test_reference_program_does_not_reshape_the_collective_operand(self) -> None:
        """No `[0]` on the staged operand either: it copies the whole input per call.

        The mirror image of the output `[None]`. Staging the input as
        (ep, max_send, hidden) forces the timed region to squeeze it back with
        `staged_g[0]`, copying ~0.6GB every call -- 3,279us at T=8192, against 3,232us for
        the output side. Staging it 2D removes the squeeze; the reshape happens once,
        outside the timed region.

        The staged operand is a DISTINCT object from the plan arrays, which are indexed
        [0] legitimately (they are tiny). An earlier version of this test stubbed `jit`
        so the transport body never ran and passed against a mutant that squeezed.
        """
        stub = _stub_jax()
        squeezed = []

        class Operand:
            dtype = "bfloat16"

            def __getitem__(self, index):
                squeezed.append(index)
                return self

            def reshape(self, *_a, **_k):
                return self

        # Staging is take -> _fit (slice or pad), so both must yield the watched object
        # or the assertIsInstance guard below will catch the test watching nothing.
        operand = Operand()
        stub.numpy.take = lambda *a, **k: operand
        stub.numpy.pad = lambda *a, **k: operand

        transport = ep_jax.JaxEPTransport.__new__(ep_jax.JaxEPTransport)
        transport.jax, transport.jnp = stub, stub.numpy
        transport.mesh, transport.ep_size, transport.hidden = object(), 4, 64
        transport.axis = "ep"
        transport._P = stub.sharding.PartitionSpec
        transport.shard_map = lambda fn, **kwargs: fn
        _, layout = _layout()
        point = ep_jax.Point(
            transport=transport, layout=layout, x=stub.array(),
            send_index=stub.array(), input_offsets=stub.array(),
            send_sizes=stub.array(), output_offsets=stub.array(),
            recv_sizes=stub.array(), recv_offsets=stub.array(),
            return_offsets=stub.array(),
        )
        call, staged = point.reference_program()
        self.assertIsInstance(staged, Operand, "the test must be watching the operand")
        # Indexing during STAGING is fine -- that runs once, outside any measurement.
        # What must not happen is indexing inside the timed call, so watch only that.
        squeezed.clear()
        call()
        self.assertEqual(squeezed, [],
                         "the staged operand must arrive ready to use; indexing it inside "
                         "the timed call copies the whole buffer every iteration")

    def test_main_runs_end_to_end_against_stubs(self) -> None:
        """Execute main()'s whole control flow offline.

        Two consecutive runs were lost to errors that never needed a TPU: an
        AttributeError building a program, then an UnboundLocalError reading
        `device_timing` before the traced pass assigned it. Neither is caught by
        py_compile, and no linter is available here. Executing main() is what catches
        them: the oracle fails on stub data (rc=1), which is fine -- the point is that
        every statement runs.
        """
        for extra, expected, why in (
            (["--no-xprof"], 1, "host path: oracle fails on stub data"),
            ([], 6, "device path: the gate refuses when no spans were captured"),
        ):
            with self.subTest(mode=why):
                self.assertEqual(self._run_main(extra), expected, why)

    @staticmethod
    def _run_main(extra_argv, profiler=None):
        """Run run_ep_jax.main() with a stubbed jax; returns its exit code."""
        stub = _stub_jax()
        devices = [types.SimpleNamespace(device_kind="TPU7x-stub") for _ in range(8)]
        stub.devices = lambda: devices
        stub.sharding.Mesh = lambda *a, **k: object()
        stub.profiler = profiler or types.SimpleNamespace(
            trace=lambda *a, **k: contextlib.nullcontext()
        )

        class FakePoint:
            """Shaped numpy returns, so the oracle runs and fails rather than raising."""

            def __init__(self, layout, hidden):
                self.layout, self.hidden = layout, hidden
                # The oracle asks the transport which precision the wire carried.
                self.t = types.SimpleNamespace(fp8=False, hidden=hidden)

            def _recv(self):
                return np.zeros((layout_ep(self.layout),
                                 max(1, self.layout.max_recv), self.hidden),
                                dtype=np.float32)

            def _tokens(self):
                return np.zeros((layout_ep(self.layout), self.layout.tokens_per_rank,
                                 self.hidden), dtype=np.float32)

            dispatched = _recv
            dispatch = _recv
            roundtrip = _tokens

            def chain(self, iters):
                # The stub models a backend that does not chain; the chained blocks then
                # publish `unavailable` with a reason rather than vanishing.
                return None

            def combine(self, received=None):
                return self._tokens()

            def timed_operation(self, name):
                return self.roundtrip

            def reference_program(self):
                return (self.roundtrip, None)

        def layout_ep(layout):
            return layout.ep_size

        with tempfile.TemporaryDirectory() as directory:
            argv = ["run_ep_jax.py", *_tpu_argv(), *extra_argv]
            argv[argv.index("--out") + 1] = str(Path(directory) / "out.json")
            argv[argv.index("--trials") + 1] = "1"
            with (
                mock.patch.object(run_ep_jax.ep_jax, "import_jax",
                                  return_value=(stub, stub.numpy, lambda f, **k: f)),
                mock.patch.object(run_ep_jax.ep_jax.JaxEPTransport, "__init__",
                                  lambda self, *a, **k: None),
                mock.patch.object(run_ep_jax.ep_jax.JaxEPTransport, "build",
                                  lambda self, layout, acts: FakePoint(layout, 7168)),
                mock.patch.object(run_ep_jax.ep_jax, "reference_timing",
                                  return_value=[1.0, 1.0]),
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ,
                                {"COLLX_VENDOR": "google", "COLLX_RUNTIME": "tpu"}),
            ):
                return run_ep_jax.main()

    def test_each_traced_component_gets_its_own_capture(self) -> None:
        """One profiler session per component per point, into distinct directories.

        Three components sharing one session produced ~400k-event captures, and past the
        fifth traced point the capture started coming back with every op recorded and no
        CollectiveX scope metadata at all. Splitting the session is the mitigation, so the
        split itself is worth pinning: a refactor that folds these back into one session
        would silently reinstate the failure on hardware only.
        """
        directories = []
        profiler = types.SimpleNamespace(
            trace=lambda directory, **k: (directories.append(directory),
                                          contextlib.nullcontext())[1]
        )
        self._run_main(["--xprof-points", "1,2"], profiler=profiler)
        self.assertEqual(len(directories), 8, "2 points x 4 captures")
        self.assertEqual(len(set(directories)), len(directories),
                         "each capture needs its own directory or find_trace is ambiguous")
        for component in ("dispatch", "combine", "roundtrip", "reference"):
            for tokens in (1, 2):
                self.assertTrue(
                    any(f"-t{tokens}-{component}-" in d for d in directories),
                    f"no capture for {component} at T={tokens}: {directories}")

    def test_reference_program_is_traced_as_a_cross_check_not_a_component(self) -> None:
        """The reference is captured, but a failed capture must not fail a case.

        Its host-timed figure sits at ~2x the collective's device time across the prefill
        ladder; tracing its own scope is what turns that from an inference into a
        measurement. It stays out of COMPONENTS so the device-timing gate never blocks a
        case on it.
        """
        self.assertIn(run_ep_jax.REFERENCE, run_ep_jax.TRACED)
        self.assertNotIn(run_ep_jax.REFERENCE, run_ep_jax.COMPONENTS)

        marker, inner = run_ep_jax.traced_markers(run_ep_jax.REFERENCE, 8192)
        self.assertEqual(marker, run_ep_jax.ep_jax.REFERENCE_MARKER)
        # The reference times the SAME ragged collective the components do -- that is
        # what makes it comparable to dispatch.transport_us at all. A dense all_to_all
        # reference was tried and could not be reconciled: different primitive, ~11%
        # cheaper per byte on 2% more bytes.
        self.assertEqual(run_ep_jax.transport_key(run_ep_jax.REFERENCE, 8192),
                         run_ep_jax.ep_jax.REFERENCE_HLO[0])
        self.assertIn(run_ep_jax.ep_jax.REFERENCE_HLO[0], inner)
        self.assertEqual(run_ep_jax.ep_jax.REFERENCE_HLO,
                         run_ep_jax.ep_jax.TRACED_HLO)

        marker, inner = run_ep_jax.traced_markers("dispatch", 8192)
        self.assertEqual(marker, run_ep_jax.ep_jax.scope_name("dispatch", 8192))
        self.assertEqual(run_ep_jax.transport_key("dispatch", 8192),
                         run_ep_jax.ep_jax.transport_scope("dispatch", 8192))

    def test_profile_session_trims_host_tracing_where_supported(self) -> None:
        """Turn down the planes we never read; degrade rather than fail without them.

        Host TraceMe and the Python tracer dominate a ~400k-event capture and contribute
        nothing to a device span. ProfileOptions is not in every JAX version, so its
        absence must fall back to a default capture, not crash a case.
        """
        captured = {}

        class Options:
            pass

        rich = types.SimpleNamespace(
            ProfileOptions=Options,
            trace=lambda directory, profiler_options=None: captured.update(
                directory=directory, options=profiler_options) or contextlib.nullcontext(),
        )
        run_ep_jax.profile_session(types.SimpleNamespace(profiler=rich), "/tmp/x")
        self.assertEqual(captured["directory"], "/tmp/x")
        self.assertEqual(captured["options"].host_tracer_level, 1)
        self.assertEqual(captured["options"].python_tracer_level, 0)

        calls = []

        def older(directory, **kwargs):
            if kwargs:
                raise TypeError("profiler_options is not supported in this version")
            calls.append(directory)
            return contextlib.nullcontext()

        legacy = types.SimpleNamespace(trace=older)
        run_ep_jax.profile_session(types.SimpleNamespace(profiler=legacy), "/tmp/y")
        self.assertEqual(calls, ["/tmp/y"])

    def test_timing_source_names_which_measurement_published(self) -> None:
        """A consumer must see whether a latency is chip time or host time.

        Only the device span is the same KIND of measurement as the GPU SKUs' CUDA
        events; the host fallback carries a per-call dispatch floor they do not have.
        """
        self.assertEqual(run_ep_jax.timing_source(True), "xla-device-trace-span")
        self.assertEqual(run_ep_jax.timing_source(False), "host-wallclock-blocked")
        self.assertNotEqual(run_ep_jax.timing_source(True),
                            run_ep_jax.timing_source(False))



class _Fp8Array(np.ndarray):
    """numpy that tolerates the e4m3 cast, so quantize_local's ARITHMETIC can be run.

    The cast itself is a no-op here: e4m3 rounding is the device's job, and NOTHING checks
    the rounding itself -- the oracle deliberately never re-derives the wire bits, because
    two compilations of this recipe round midpoints differently. What these tests exercise
    is everything around the cast -- the reduction axis, the clamp, the direction of the
    scale, and the cast TARGET and ORDER via `_cast_to` -- which is where a mutation
    publishes a wrong number rather than crashing.
    """

    # Both device-side narrowing casts are identities here. e4m3 and bf16 ROUNDING is the
    # device's job and the oracle checks it on hardware by comparing transported bits; what
    # is under test is the arithmetic around them.
    _DEVICE_CASTS = ("float8_e4m3fn", "bfloat16")
    #: every dtype any instance was asked to cast to, so a test can assert the cast TARGET.
    #: Identity-casting both device dtypes without recording them made "the e4m3 cast was
    #: deleted" undetectable -- and that mutation is wrong-but-GREEN on hardware, because
    #: the oracle compares against the same quantize_local while the bytes still claim 1
    #: byte per value.
    #: (dtype, shape) so a test can assert WHICH array was cast to what. Recording the
    #: dtype alone let a cast SWAP through -- values narrowed to bf16 and scales to e4m3
    #: still satisfied "e4m3 was requested" and "scales are f32" (the shim returns the f32
    #: array unchanged for any device cast), while the wire moved 2 bytes per value billed
    #: at 1: dispatch bandwidth understated ~1.94x, oracle green.
    requested = []

    def astype(self, dtype, *args, **kwargs):
        _Fp8Array.requested.append((dtype, tuple(self.shape)))
        if dtype in self._DEVICE_CASTS:
            narrowed = self.view(_Fp8Array)
            narrowed._cast_to = dtype
            return narrowed
        return np.ndarray.astype(self, dtype, *args, **kwargs).view(_Fp8Array)

    # `requested` records that a cast HAPPENED; `_cast_to` records that it happened LAST.
    # Without the second, `view.astype(e4m3) * (448/amax)` -- cast before scale instead of
    # after -- still "requests e4m3 for a value-shaped array" and passes. In real JAX that
    # mutation promotes f8*f32 back to FLOAT32, so the wire carries 4 bytes per value while
    # byte_provenance bills 1: dispatch bandwidth overstated 4x, and green, because the
    # shim cannot see a dtype the device would have produced. Views and reshapes carry the
    # tag (the production code reshapes after casting); arithmetic clears it.
    def __array_finalize__(self, obj):
        self._cast_to = getattr(obj, "_cast_to", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Route through plain ndarrays: ndarray.__array_ufunc__ returns NotImplemented when
        # an input overrides it. The result then finalizes from a plain array, so _cast_to
        # comes back None -- which is exactly the clearing this needs.
        plain = tuple(np.asarray(value) if isinstance(value, _Fp8Array) else value
                      for value in inputs)
        if "out" in kwargs:
            kwargs["out"] = tuple(
                np.asarray(value) if isinstance(value, _Fp8Array) else value
                for value in kwargs["out"])
        result = getattr(ufunc, method)(*plain, **kwargs)
        return result.view(_Fp8Array) if isinstance(result, np.ndarray) else result


def _fp8_transport(hidden=256, ep_size=4, stub=None):
    stub = _stub_jax() if stub is None else stub
    numpy_like = types.SimpleNamespace(
        float32=np.float32, bfloat16="bfloat16", float8_e4m3fn="float8_e4m3fn",
        clip=np.clip, max=np.max, abs=np.abs, zeros=np.zeros, take=np.take,
    )
    return ep_jax.JaxEPTransport(
        stub, numpy_like, lambda fn, **k: fn, object(), ep_size, hidden, precision="fp8",
    )


class Fp8PermutationOracleTests(unittest.TestCase):
    """The fp8 oracle compares arrived bytes against STAGED bytes, not re-quantized ones.

    Re-deriving the wire bits assumed bitwise identity between two XLA compilations of the
    same recipe. That is false on TPU -- `view * (448/amax)` may reassociate to
    `(view * 448) / amax`, one rounding against two, and this lattice hits exact e4m3
    midpoints where a single f32 ulp flips the byte. A permutation check needs no such
    assumption.
    """

    def _plan(self, ep_size=2, size=3, hidden=4, blocks=2):
        """Two ranks, each sending `size` rows to each of the two."""
        layout = types.SimpleNamespace(
            recv_sizes=np.full((ep_size, ep_size), size, dtype=np.int32),
            recv_offsets=np.array([[0, size]] * ep_size, dtype=np.int32),
            input_offsets=np.array([[0, size]] * ep_size, dtype=np.int32),
        )
        rows = ep_size * size
        staged_values = (np.arange(ep_size * rows * hidden, dtype=np.float32)
                         .reshape(ep_size, rows, hidden))
        staged_scales = (np.arange(ep_size * rows * blocks, dtype=np.float32)
                         .reshape(ep_size, rows, blocks))
        return layout, staged_values, staged_scales, ep_size, size

    def _delivered(self, layout, staged_values, staged_scales, ep_size, size):
        """What a CORRECT transport delivers: dst gets src's chunk at recv_offsets."""
        values = np.zeros_like(staged_values)
        scales = np.zeros_like(staged_scales)
        for dst in range(ep_size):
            for src in range(ep_size):
                here = int(layout.recv_offsets[dst, src])
                there = int(layout.input_offsets[src, dst])
                values[dst, here:here + size] = staged_values[src, there:there + size]
                scales[dst, here:here + size] = staged_scales[src, there:there + size]
        return values, scales

    def test_a_correct_transport_passes(self) -> None:
        layout, sv, ss, ep, size = self._plan()
        values, scales = self._delivered(layout, sv, ss, ep, size)
        self.assertTrue(run_ep_jax._check_dispatch_permutation(
            values, scales, sv, ss, layout, ep))

    def test_a_swapped_chunk_fails(self) -> None:
        """The failure the old design could not see: self-consistent misrouting."""
        layout, sv, ss, ep, size = self._plan()
        values, scales = self._delivered(layout, sv, ss, ep, size)
        values[0, 0:size], values[0, size:2 * size] = (
            values[0, size:2 * size].copy(), values[0, 0:size].copy())
        self.assertFalse(run_ep_jax._check_dispatch_permutation(
            values, scales, sv, ss, layout, ep))

    def test_corrupted_scales_fail_even_when_values_are_right(self) -> None:
        layout, sv, ss, ep, size = self._plan()
        values, scales = self._delivered(layout, sv, ss, ep, size)
        scales[1, 0, 0] += 1.0
        self.assertFalse(run_ep_jax._check_dispatch_permutation(
            values, scales, sv, ss, layout, ep))

    def test_one_corrupted_value_byte_fails(self) -> None:
        layout, sv, ss, ep, size = self._plan()
        values, scales = self._delivered(layout, sv, ss, ep, size)
        values[1, 2, 3] += 1.0
        self.assertFalse(run_ep_jax._check_dispatch_permutation(
            values, scales, sv, ss, layout, ep))

    def test_an_empty_chunk_is_skipped_not_silently_passed(self) -> None:
        """A zero-size chunk has nothing to compare; a NON-zero one must still be checked."""
        layout, sv, ss, ep, size = self._plan()
        values, scales = self._delivered(layout, sv, ss, ep, size)
        layout.recv_sizes = np.array([[0, size], [size, size]], dtype=np.int32)
        values[1, 0, 0] += 1.0          # inside a chunk that is still sized
        self.assertFalse(run_ep_jax._check_dispatch_permutation(
            values, scales, sv, ss, layout, ep))


def _replay(hidden: int = 64, swap: tuple[int, int] | None = None):
    """Replay a whole correct dispatch on the host: (layout, activations, received).

    `swap` exchanges two adjacent rows INSIDE one (dst, src) chunk, which is what a permute
    that gathered in the wrong order would produce. Rows carry real encoded source IDs, so
    the oracle's ID decode has something to read.
    """
    idx, layout = _layout()
    ep, tpr = TRACE["ep_size"], TRACE["tokens_per_rank"]
    seed = TRACE["seed"]
    acts = np.stack([routing_np.rank_activations(tpr, hidden, seed, r) for r in range(ep)])
    staged = np.zeros((ep, layout.send_index.shape[1], hidden), np.float32)
    for src in range(ep):
        n = int(layout.send_total[src])
        staged[src, :n] = acts[src][layout.send_index[src, :n].astype(np.int64)]
    if swap is not None:
        src, dst = swap
        at = int(layout.input_offsets[src, dst])
        staged[src, [at, at + 1]] = staged[src, [at + 1, at]]
    blocks = max(1, hidden // ep_jax.QUANT_BLOCK)
    # Distinct per (rank, slot) so a mis-delivered scale block is actually distinguishable
    # here -- unlike on the real lattice, where amax takes only four values.
    staged_scales = (np.arange(1, ep * staged.shape[1] * blocks + 1, dtype=np.float32)
                     .reshape(ep, staged.shape[1], blocks))
    received = np.zeros((ep, int(layout.recv_total.max()), hidden), np.float32)
    received_scales = np.zeros((ep, int(layout.recv_total.max()), blocks), np.float32)
    for dst in range(ep):
        for src in range(ep):
            size = int(layout.recv_sizes[dst, src])
            if size:
                here = int(layout.recv_offsets[dst, src])
                there = int(layout.input_offsets[src, dst])
                received[dst, here:here + size] = staged[src, there:there + size]
                received_scales[dst, here:here + size] = (
                    staged_scales[src, there:there + size])
    # `received`/`staged` are the DEQUANTIZED rows -- what the device hands combine. The
    # `_values` arrays are the wire payload that dequantizes back to them, so the two are
    # mutually consistent and `_check_timed_values_match` can dequantize one and get the
    # other. Without that, the correct case would fail the values check.
    def unquantized(rows, scale):
        blocks = scale.shape[-1]
        view = rows.reshape(rows.shape[0], rows.shape[1], blocks, -1)
        return (view / scale[:, :, :, None]).reshape(rows.shape)

    return types.SimpleNamespace(
        layout=layout, activations=acts, received=received, staged=staged,
        received_scales=received_scales, staged_scales=staged_scales, hidden=hidden,
        received_values=unquantized(received, received_scales),
        staged_values=unquantized(staged, staged_scales),
    )


def _combine_actual(layout, received, rank: int, hidden: int):
    """What combine really returns: row i of a chunk pairs with send_index[there + i]."""
    out = np.zeros((layout.tokens_per_rank, hidden), np.float32)
    for dst in range(layout.ep_size):
        size = int(layout.recv_sizes[dst, rank])
        if not size:
            continue
        here = int(layout.recv_offsets[dst, rank])
        there = int(layout.input_offsets[rank, dst])
        np.add.at(out, layout.send_index[rank, there:there + size].astype(np.int64),
                  received[dst, here:here + size])
    return out


class CombineExpectationTests(unittest.TestCase):
    """Under fp8 the combine expectation comes from DELIVERED rows, not a second quantize.

    Re-deriving it by quantizing x again in a separate program is a coin flip on this
    lattice: XLA rounds e4m3 midpoints differently inside the fused dispatch than in a
    standalone codec program. Measured at rank 0 token 26 col 52 -- dispatch produced
    161/256, the standalone codec 152/256, one full grid step, 5.92%.

    Each delivered row is attributed to the token its PAYLOAD claims to be, not to the
    token at its POSITION. See the swap test below for why that distinction is the entire
    correctness value of this function.
    """

    def test_a_correct_transport_scores_exactly_zero(self) -> None:
        hidden = 64
        replay = _replay(hidden)
        layout, received = replay.layout, replay.received
        for rank in range(layout.ep_size):
            expected = run_ep_jax._expected_combine_from_received(
                received, layout, rank, TRACE["seed"])
            np.testing.assert_array_equal(
                expected, _combine_actual(layout, received, rank, hidden))

    def test_a_row_swapped_inside_one_chunk_is_caught(self) -> None:
        """The hole this function exists to close, and the only thing that closes it.

        A swap confined to one (dst, src) chunk passes EVERY other fp8 check: the
        permutation oracle sees it identically on both sides when the permute staged it,
        step 1 decodes the source ID from the moved payload so ID and data stay mutually
        consistent, and step 3 sorts before comparing. Under bf16 the pristine expectation
        caught it. An expectation anchored on send_index POSITION would bake the swap into
        both sides and score 0.
        """
        hidden = 64
        pair = next((s, d) for s in range(TRACE["ep_size"])
                    for d in range(TRACE["ep_size"])
                    if _layout()[1].recv_sizes[d, s] >= 2)
        replay = _replay(hidden, swap=pair)
        layout, received = replay.layout, replay.received
        worst = max(
            float(np.abs(_combine_actual(layout, received, rank, hidden)
                         - run_ep_jax._expected_combine_from_received(
                             received, layout, rank, TRACE["seed"])).max())
            for rank in range(layout.ep_size)
        )
        self.assertGreater(worst, 0.0)

    def test_a_destination_that_received_nothing_contributes_nothing(self) -> None:
        ep, tokens, hidden = 2, 2, routing_np.SOURCE_ID_COLUMNS + 1
        layout = types.SimpleNamespace(
            ep_size=ep, tokens_per_rank=tokens,
            recv_sizes=np.array([[1, 0], [0, 0]], dtype=np.int32),
            recv_offsets=np.zeros((ep, ep), dtype=np.int32),
            input_offsets=np.zeros((ep, ep), dtype=np.int32),
            send_index=np.array([[0, 0], [0, 0]], dtype=np.int32),
        )
        row = routing_np.activations_for_source_ids(np.array([0]), hidden, TRACE["seed"])
        received = np.stack([row, np.zeros_like(row)]).astype(np.float32)
        expected = run_ep_jax._expected_combine_from_received(
            received, layout, 0, TRACE["seed"])
        np.testing.assert_array_equal(expected[0], row[0])
        np.testing.assert_array_equal(expected[1], np.zeros(hidden, np.float32))


class TimedProgramScalesTests(unittest.TestCase):
    """The oracle must police the program whose latency is PUBLISHED.

    `_check_dispatch_permutation` compares arrived against staged inside the oracle
    program. That program shares `_quantized_dispatch_local` with the timed one, so the
    collective and its barrier have a single definition -- but it is still its own
    compilation, and an earlier draft duplicated the collectives outright, so the
    byte-for-byte guarantee covered a program that was never timed.

    Nothing else can police the timed scales exchange: the source ID rides in the SIGN of
    the VALUES, placement is checked on the values, and numerics are useless here because
    the real lattice's per-128-block amax takes only FOUR distinct values across 229,376
    blocks -- every possible mix-up lands within 2.4%, inside e4m3's own grid step.
    """

    def _drive(self, corrupt_timed_scales: bool = False, corrupt_timed_values: bool = False,
               zero_a_delivered_row: bool = False,
               misdeliver_across_ranks: bool = False):
        """Run the REAL _oracle against a point that replays a correct dispatch."""
        replay = _replay(hidden=2 * ep_jax.QUANT_BLOCK)
        ep = replay.layout.ep_size
        timed_scales = replay.received_scales.copy()
        if corrupt_timed_scales:
            # The LAST block of the LAST delivered row of the LAST rank. Corrupting [0,0,0]
            # cannot distinguish a full check from one truncated to the first row or rank.
            last = ep - 1
            timed_scales[last, int(replay.layout.recv_total[last]) - 1, -1] += 1.0
        timed_values = replay.received
        if corrupt_timed_values:
            # ONE row -- the last delivered row of the last rank -- past the source-ID
            # prefix, so the IDs still decode and placement still passes. This returned
            # ok=True before _check_timed_values_match existed.
            #
            # One row, not all of them, or the test cannot tell a full check from one
            # truncated to the first row or the first rank. And a 1.5x scale rather than a
            # dramatic one: it scores 0.5 under the check's metric, which pins
            # TIMED_VALUE_REL_TOL between 0.25 and 0.5. Corrupting everything scored 1.371
            # and left a 4x loosening of the tolerance undetected.
            timed_values = replay.received.copy()
            last = ep - 1
            row = int(replay.layout.recv_total[last]) - 1
            timed_values[last, row, routing_np.SOURCE_ID_COLUMNS:] *= 1.5
        if zero_a_delivered_row:
            timed_values = replay.received.copy()
            timed_values[0, 0, :] = 0.0      # prefix no longer decodes at all
        if misdeliver_across_ranks:
            # A row that decodes cleanly but carries a token belonging to ANOTHER rank.
            # `tokens - rank * tokens_per_rank` then leaves [0, tokens_per_rank): without a
            # bounds check the negative index wraps silently under np.add.at and the
            # expectation quietly absorbs the corruption.
            timed_values = replay.received.copy()
            foreign = (ep - 1) * replay.layout.tokens_per_rank
            timed_values[0, int(replay.layout.recv_offsets[0, 0])] = (
                routing_np.activations_for_source_ids(
                    np.array([foreign]), replay.hidden, TRACE["seed"])[0])
        combined = np.stack([
            _combine_actual(replay.layout, timed_values, rank, replay.hidden)
            for rank in range(ep)
        ])
        point = types.SimpleNamespace(
            t=types.SimpleNamespace(fp8=True),
            # The TIMED program's output. The oracle program's stays pristine throughout.
            dispatch=lambda: (timed_values, timed_scales),
            dispatch_with_staged=lambda: (replay.received_values, replay.received_scales,
                                          replay.staged_values, replay.staged_scales),
            combine_input=lambda: timed_values,
            combine=lambda: combined,
        )
        ok, error = run_ep_jax._oracle(
            point, replay.layout, replay.activations, TRACE["seed"], ep,
            types.SimpleNamespace(block_until_ready=lambda value: value),
        )
        return ok, error

    def test_a_consistent_timed_program_passes(self) -> None:
        ok, error = self._drive(corrupt_timed_scales=False)
        self.assertTrue(ok)
        self.assertEqual(error, 0.0)

    def test_corrupt_scales_in_the_TIMED_program_fail_the_case(self) -> None:
        """Corrupt only the timed program's scales; the oracle program stays perfect.

        Without the timed-vs-oracle tie this returns True: the permutation check reads the
        oracle program's own output and is satisfied, the values are untouched so ID decode
        and placement pass, and the combine expectation is built from delivered rows.
        """
        ok, _ = self._drive(corrupt_timed_scales=True)
        self.assertFalse(ok, "a corrupt timed-program scales exchange published as correct")

    def test_corrupt_values_past_the_id_prefix_fail_the_case(self) -> None:
        """The scales check alone left this green -- measured ok=True, max_rel=0.0.

        Scrambling the delivered values anywhere past the source-ID prefix survives every
        other check: the ID decode reads prefix signs only, placement sorts IDs, the
        permutation check reads the ORACLE program's output, and the combine expectation is
        built from the corrupted rows so both sides agree. A stride or partial-row bug in
        the timed values exchange looks exactly like this.
        """
        ok, _ = self._drive(corrupt_timed_values=True)
        self.assertFalse(ok, "a corrupt timed-program values exchange published as correct")

    def test_an_undecodable_row_fails_the_oracle_instead_of_crashing(self) -> None:
        """A broken transport must read as correct=False, not as a dead process.

        `_combine_error` runs BEFORE `_check_dispatch`, so an unguarded decode here raised
        ValueError out of _oracle and the case died as a process failure -- still red, but
        without the diagnostic that localises it.
        """
        ok, error = self._drive(zero_a_delivered_row=True)
        self.assertFalse(ok)
        self.assertEqual(error, float("inf"))

    def test_a_row_carrying_another_ranks_token_fails_the_oracle(self) -> None:
        """Cross-rank misdelivery must not wrap into a negative index.

        The row decodes cleanly, so the guard above does not fire. `tokens - rank *
        tokens_per_rank` then lands outside [0, tokens_per_rank), and np.add.at wraps a
        negative index silently -- the expectation would absorb the corruption and agree
        with the broken combine.
        """
        ok, error = self._drive(misdeliver_across_ranks=True)
        self.assertFalse(ok)
        self.assertEqual(error, float("inf"))


class _Sharded(np.ndarray):
    """An ndarray that `to_host` treats as a device array (it has `.sharding`)."""

    sharding = object()


class OracleGatherCountTests(unittest.TestCase):
    """Each tensor the fp8 oracle reads is gathered across hosts EXACTLY once.

    `to_host` is free at EP8 -- one process, so it is a local `np.asarray` -- and a
    multi-GB cross-host collective at EP16. Nothing in the oracle's structure makes a
    second gather of the same array visible: it returns the same values, so every
    correctness test passes either way. The cost shows up only as wall clock on hardware.

    Measured on run 31194062843 before this was pinned: `values` was gathered twice and
    `scales` three times, because `_check_timed_values_match` and
    `_check_dispatch_permutation` each re-gathered arrays the other had already pulled.
    11.9GB of the 62.7GB that the EP16 fp8 prefill case moved -- roughly 19 minutes of a
    100-minute case, on a shard that finished within 9 minutes of its hard timeout.

    Counting is by object identity at the `process_allgather` boundary, which is the real
    predicate: a gathered result comes back as a base ndarray with no `.sharding`, so
    re-entering `_by_rank` with it is genuinely free and must NOT be counted as a repeat.
    """

    def _count_gathers(self):
        import types as _t
        gathered = []

        def process_allgather(value, tiled=False):
            gathered.append(value)
            return np.asarray(value)

        utils = _t.ModuleType("jax.experimental.multihost_utils")
        utils.process_allgather = process_allgather
        experimental = _t.ModuleType("jax.experimental")
        experimental.multihost_utils = utils
        jax_mod = _t.ModuleType("jax")
        jax_mod.experimental = experimental

        # Every array the stub point hands back looks like a device array, so `to_host`
        # takes the collective branch exactly where it would at EP16 -- but anything a
        # gather already RETURNED is a host array and must stay one, or the fixture
        # re-gathers its own output and reports 11 where the code performs 7. That is not
        # a detail of the stub: it is the property the fix relies on, since the helpers
        # still call `_by_rank` on arguments their caller has already pulled.
        original = run_ep_jax._by_rank
        produced = set()

        def sharded_by_rank(array, ep_size):
            if (isinstance(array, np.ndarray) and not hasattr(array, "sharding")
                    and id(array) not in produced):
                array = array.view(_Sharded)
            result = original(array, ep_size)
            produced.add(id(result))
            return result

        run_ep_jax._PROCESS_COUNT[0] = 2
        try:
            with mock.patch.dict(sys.modules, {
                    "jax": jax_mod, "jax.experimental": experimental,
                    "jax.experimental.multihost_utils": utils}), \
                 mock.patch.object(run_ep_jax, "_by_rank", sharded_by_rank):
                # Reuses the replay harness rather than a second fixture, so this counts
                # gathers on the SAME execution path the correctness tests validate.
                ok, error = TimedProgramScalesTests._drive(self)
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1
        return ok, error, gathered

    def test_the_oracle_still_passes_under_the_two_host_path(self) -> None:
        """Guards the guard: a counting test over a BROKEN oracle proves nothing."""
        ok, error, gathered = self._count_gathers()
        self.assertTrue(ok)
        self.assertEqual(error, 0.0)
        self.assertTrue(gathered, "no gather happened at all; the fixture is inert")

    def test_no_tensor_crosses_the_host_boundary_twice(self) -> None:
        _, _, gathered = self._count_gathers()
        seen = {}
        for array in gathered:
            seen[id(array)] = seen.get(id(array), 0) + 1
        repeats = {key: count for key, count in seen.items() if count > 1}
        self.assertEqual(
            repeats, {},
            f"{len(repeats)} tensor(s) gathered more than once: {sorted(repeats.values())} "
            f"-- each repeat re-sends the whole buffer over the host network at EP16")

    def test_the_gather_count_is_the_number_of_distinct_tensors(self) -> None:
        """Seven: timed scales, oracle values/scales, both staged, combine_input, combine.

        Pinning the total as well as the duplicate-freedom is what stops a future helper
        from gathering an eighth tensor that nothing needs.
        """
        _, _, gathered = self._count_gathers()
        self.assertEqual(len(gathered), 7, f"gathered {len(gathered)} tensors, expected 7")


class CorrectnessBlockTests(unittest.TestCase):
    """The correctness block must survive the artifact writer.

    `ep_harness._write_json_atomic` serializes with `allow_nan=False`. A raw inf from an
    unattributable oracle therefore raised out of json.dumps at the END of the run, after
    the full timed sweep, killing the process with a traceback instead of writing a red
    row -- the same dead-process failure the guarded decode removed, relocated. Asserting
    the value is None in memory is NOT enough to catch that; these serialize.
    """

    @staticmethod
    def _dumps(block):
        return json.dumps(block, allow_nan=False)

    def test_a_measured_error_is_published_as_a_number(self) -> None:
        block = run_ep_jax._correctness(0.0123, True)
        self.assertAlmostEqual(block["max_relative_error"], 0.0123)
        self.assertFalse(block["unattributable"])
        self.assertTrue(block["passed"])
        self.assertIn("0.0123", self._dumps(block))

    def test_an_unattributable_oracle_still_serializes(self) -> None:
        block = run_ep_jax._correctness(float("inf"), False)
        self.assertIsNone(block["max_relative_error"])
        self.assertTrue(block["unattributable"])
        self.assertFalse(block["passed"])
        # The real writer's flags. Raw inf raises here; that was the defect.
        self.assertEqual(json.loads(self._dumps(block))["max_relative_error"], None)

    def test_nan_is_treated_the_same_as_inf(self) -> None:
        block = run_ep_jax._correctness(float("nan"), False)
        self.assertIsNone(block["max_relative_error"])
        self.assertTrue(block["unattributable"])
        self._dumps(block)

    def test_the_whole_artifact_writer_accepts_an_unattributable_row(self) -> None:
        """End to end through the ACTUAL writer, not a local json.dumps."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "case-attempt.json")
            ep_harness._write_json_atomic(
                path, {"correctness": run_ep_jax._correctness(float("inf"), False)})
            with open(path) as handle:
                written = json.load(handle)
        self.assertIsNone(written["correctness"]["max_relative_error"])
        self.assertFalse(written["correctness"]["passed"])


class _AtArray(np.ndarray):
    """numpy plus JAX's `.at[idx].add(v)`, so `_combine_local`'s scatter-add can be traced.

    The scatter itself is not under test here -- the combine oracle covers that on hardware.
    What these tests need is for the roundtrip PROGRAM to build without a device.
    """

    class _Indexer:
        def __init__(self, array):
            self._array = array

        def __getitem__(self, _index):
            return self

        def add(self, _value):
            return self._array

    @property
    def at(self):
        return _AtArray._Indexer(self)


class NativeFp8RoundtripTests(unittest.TestCase):
    """`roundtrip` must mean dispatch->combine on BOTH precisions.

    deepseek-v3 block-fp8 takes the `native` path: the expert consumes the dispatched fp8
    and per-128-block scales directly and emits BF16, so no standalone conversion sits
    between the two collectives. Charging one to the chained roundtrip compares fp8 and bf16
    through structurally different pipelines -- it made fp8's roundtrip look WORSE than
    bf16's while its dispatch was 1.53x faster, the same inversion the GPU corpus measured
    in 39 of 51 comparisons before hoisting its stage out.
    """

    def _traced_fp8_roundtrip(self):
        """Trace the real fp8 roundtrip program, recording scopes and barrier operands."""
        events, barriers = [], []
        stub = _stub_jax()
        stub.named_scope = lambda name: _recording_scope(events, name)
        stub.lax.optimization_barrier = lambda values: (barriers.append(values) or values)
        stub.lax.ragged_all_to_all = lambda operand, *a, **k: operand
        hidden = 2 * ep_jax.QUANT_BLOCK
        _, layout = _layout()
        transport = _fp8_transport(hidden=hidden, ep_size=layout.ep_size, stub=stub)
        transport.jnp.zeros = lambda shape, dtype=None: np.zeros(
            shape, dtype=np.float32).view(_AtArray)
        dequantized = []
        original = transport.dequantize_local
        transport.dequantize_local = lambda v, s: (dequantized.append(True)
                                                   or original(v, s))
        x = np.ones((1, layout.tokens_per_rank, hidden), np.float32).view(_Fp8Array)
        plan = {name: getattr(layout, name).astype(np.int32) for name in (
            "send_index", "input_offsets", "send_sizes", "output_offsets",
            "recv_sizes", "recv_offsets", "return_offsets")}
        point = ep_jax.Point(transport=transport, layout=layout, x=x, **plan)
        staged = np.full((layout.max_recv, hidden), 7.0, np.float32)
        point._combine_input = staged
        point._dispatched = (x, x)
        self._last_result = point.roundtrip()
        return events, barriers, dequantized, staged

    def test_the_fp8_roundtrip_contains_no_conversion(self) -> None:
        _, _, dequantized, _ = self._traced_fp8_roundtrip()
        self.assertEqual(dequantized, [],
                         "the fp8 roundtrip dequantized inside the timed region; that is "
                         "the mismatch path, not this workload's")

    def test_the_fp8_roundtrip_still_dispatches(self) -> None:
        """The conversion was the data dependency. Removing it must not remove dispatch."""
        events, barriers, _, _ = self._traced_fp8_roundtrip()
        tokens = _layout()[1].tokens_per_rank
        self.assertIn(ep_jax.transport_scope("dispatch", tokens), events)
        self.assertIn(ep_jax.transport_scope("combine", tokens), events)
        self.assertIn(ep_jax.quantize_scope("dispatch", tokens), events)

    def test_the_dispatch_results_are_program_OUTPUTS(self) -> None:
        """The only thing that actually keeps the dispatch alive.

        The barrier alone does not. Measured on hardware, run 30890386745: the fp8 roundtrip
        came back at 16,511.5us against combine's 16,511.4, its op_inventory byte-for-byte
        combine's, with every dispatch op absent. `optimization_barrier` constrains ORDERING
        and does not make an unused tuple output live, so XLA deleted the dispatch outright.
        Returning the results is what XLA cannot elide.
        """
        stub = _stub_jax()
        specs = {}
        transport = _fp8_transport(hidden=2 * ep_jax.QUANT_BLOCK, ep_size=4, stub=stub)
        original = transport._mapped

        def recording(fn, in_specs, out_specs, donate=()):
            specs[getattr(fn, "__name__", "?")] = out_specs
            return original(fn, in_specs, out_specs, donate)

        transport._mapped = recording
        _, layout = _layout()
        plan = {name: getattr(layout, name).astype(np.int32) for name in (
            "send_index", "input_offsets", "send_sizes", "output_offsets",
            "recv_sizes", "recv_offsets", "return_offsets")}
        ep_jax.Point(transport=transport, layout=layout, x=stub.array(), **plan)
        self.assertEqual(specs.get("roundtrip_fp8"), (True, True, True),
                         "the fp8 roundtrip must return the dispatch results, or XLA "
                         "deletes the dispatch and the roundtrip becomes a lone combine")

    def test_the_traced_fp8_roundtrip_returns_three_arrays(self) -> None:
        events, barriers, _, staged = self._traced_fp8_roundtrip()
        self.assertEqual(len(self._last_result), 3)
        # The combine result first, then the two dispatch buffers held live.
        self.assertIsNot(self._last_result[1], staged)

    def test_the_barrier_carries_the_dispatch_result_and_the_staged_operand(self) -> None:
        """What keeps XLA from deleting the dispatch, now that nothing consumes it.

        Both collectives' results must enter the barrier alongside the staged buffer, or the
        dispatch has no consumer and the roundtrip silently becomes a lone combine.
        """
        _, barriers, _, staged = self._traced_fp8_roundtrip()
        self.assertTrue(barriers, "no optimization_barrier in the fp8 roundtrip")
        operands = barriers[-1]
        self.assertEqual(len(operands), 3,
                         "the barrier must tie values, scales AND the staged operand")
        self.assertIs(operands[2], staged)

    def test_stage_is_a_timed_component_under_fp8_only(self) -> None:
        self.assertEqual(run_ep_jax.components_for("fp8"),
                         run_ep_jax.COMPONENTS + ("stage",))
        self.assertEqual(run_ep_jax.components_for("bf16"), run_ep_jax.COMPONENTS)
        self.assertIn(run_ep_jax.REFERENCE, run_ep_jax.traced_for("fp8"))
        self.assertIn("stage", run_ep_jax.traced_for("fp8"))
        self.assertNotIn("stage", run_ep_jax.traced_for("bf16"))

    def test_the_published_provenance_says_native_on_both_artifacts(self) -> None:
        """A row that measures native must not be labelled `dequant`, or vice versa.

        The label is the only thing distinguishing this row from one that charged the
        conversion to the roundtrip, and the two differ by 8.8ms at T=8192. Both the success
        and terminal-failure documents build it from this one helper, so they cannot drift.
        """
        fp8 = run_ep_jax.fp8_provenance("fp8")
        self.assertEqual(fp8["fp8_consume"], "native")
        self.assertTrue(fp8["stage_excluded_from_roundtrip"])
        self.assertFalse(fp8["fp8_dequant_inside_roundtrip"])
        bf16 = run_ep_jax.fp8_provenance("bf16")
        self.assertIsNone(bf16["fp8_consume"])
        self.assertTrue(bf16["stage_excluded_from_roundtrip"])
        # roundtrip means the same thing on both precisions -- that is the whole point.
        self.assertEqual(fp8["stage_excluded_from_roundtrip"],
                         bf16["stage_excluded_from_roundtrip"])
        source = Path(run_ep_jax.__file__).read_text()
        self.assertEqual(source.count('"fp8_consume"'), 1,
                         "fp8_consume is built in more than one place and can drift")

    def test_the_stage_operation_reads_the_cached_dispatch(self) -> None:
        """Timing `stage` must time the conversion, not a re-dispatch."""
        calls = []
        stub = _stub_jax()
        _, layout = _layout()
        transport = _fp8_transport(hidden=2 * ep_jax.QUANT_BLOCK,
                                   ep_size=layout.ep_size, stub=stub)
        plan = {name: getattr(layout, name).astype(np.int32) for name in (
            "send_index", "input_offsets", "send_sizes", "output_offsets",
            "recv_sizes", "recv_offsets", "return_offsets")}
        point = ep_jax.Point(transport=transport, layout=layout,
                             x=stub.array(), **plan)
        point._dispatched = ("cached-values", "cached-scales")
        point._dequant = lambda *args: calls.append(args) or "staged"
        point.dispatch = lambda: self.fail("stage re-dispatched instead of reusing the "
                                           "cached output")
        self.assertEqual(point.timed_operation("stage")(), "staged")
        self.assertEqual(calls, [("cached-values", "cached-scales")])


class DispatchOutputOrderTests(unittest.TestCase):
    """The timed and oracle wrappers must keep the four outputs in the right order.

    `_quantized_dispatch_local` returns (values, scales, staged_values, staged_scales); the
    timed wrapper slices `[:2]`. Swapping the two collectives in the return tuple, or
    slicing `[2:]` so the published program returns the STAGED tensors, leaves the whole
    offline suite green and fails only late and noisily on hardware.
    """

    def _dispatch_outputs(self):
        stub = _stub_jax()
        marks = {}

        def ragged(operand, output, *_a, **_k):
            # Tag by the output buffer's width: values are `hidden` wide, scales `blocks`.
            mark = f"a2a<-{'values' if output.shape[-1] > 4 else 'scales'}"
            marks[mark] = mark
            return mark

        stub.lax.ragged_all_to_all = ragged
        stub.lax.optimization_barrier = lambda values: values
        stub.numpy.zeros = lambda shape, dtype=None: types.SimpleNamespace(shape=shape,
                                                                          dtype=dtype)
        hidden = 2 * ep_jax.QUANT_BLOCK
        _, layout = _layout()
        transport = _fp8_transport(hidden=hidden, ep_size=layout.ep_size, stub=stub)
        transport.jnp.zeros = stub.numpy.zeros
        x = np.ones((1, layout.tokens_per_rank, hidden), np.float32).view(_Fp8Array)
        plan = {name: getattr(layout, name).astype(np.int32) for name in (
            "send_index", "input_offsets", "send_sizes", "output_offsets",
            "recv_sizes", "recv_offsets", "return_offsets")}
        point = ep_jax.Point(transport=transport, layout=layout, x=x, **plan)
        return point.dispatch(), point.dispatch_with_staged()

    def test_the_timed_program_returns_the_received_pair_in_order(self) -> None:
        timed, _ = self._dispatch_outputs()
        self.assertEqual(len(timed), 2, "the timed program must publish exactly two arrays")
        self.assertEqual(timed[0], "a2a<-values")
        self.assertEqual(timed[1], "a2a<-scales")

    def test_the_oracle_program_returns_received_then_staged(self) -> None:
        _, oracle = self._dispatch_outputs()
        self.assertEqual(len(oracle), 4)
        self.assertEqual((oracle[0], oracle[1]), ("a2a<-values", "a2a<-scales"))
        # The staged pair are real arrays, not collective results.
        for staged in oracle[2:]:
            self.assertNotIsInstance(staged, str)


class Fp8CodecRecipeTests(unittest.TestCase):
    """quantize_local's arithmetic, run for real against numpy.

    The earlier version of these tests asserted the recipe in the abstract and never called
    the production code, so it caught none of: inverting the scale, reducing amax on the
    wrong axis, or dropping the clamp. Each of those publishes a wrong number.
    """

    def test_scale_is_amax_over_448_not_the_inverse(self) -> None:
        transport = _fp8_transport(hidden=256)
        x = (np.arange(2 * 256, dtype=np.float32).reshape(2, 256) / 64.0).view(_Fp8Array)
        values, scales = transport.quantize_local(x)
        blocks = x.reshape(2, 2, 128)
        want = np.abs(blocks).max(axis=2) / 448.0
        np.testing.assert_allclose(np.asarray(scales), want, rtol=1e-6)
        # The inverse (448/amax) is >= 1 here while amax/448 is far below it, so an
        # inversion cannot masquerade as this.
        self.assertLess(float(np.asarray(scales).max()), 1.0)

    def test_amax_reduces_over_the_128_block_not_the_row(self) -> None:
        """Guards axis=2. Reducing over axis=1 would give one scale per column position."""
        transport = _fp8_transport(hidden=256)
        x = np.zeros((3, 256), dtype=np.float32)
        x[:, 0] = 8.0        # block 0 peaks at 8
        x[:, 128] = 2.0      # block 1 peaks at 2
        values, scales = transport.quantize_local(x.view(_Fp8Array))
        self.assertEqual(np.asarray(scales).shape, (3, 2))
        np.testing.assert_allclose(np.asarray(scales)[:, 0], 8.0 / 448.0, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(scales)[:, 1], 2.0 / 448.0, rtol=1e-6)

    def test_the_clamp_keeps_an_all_zero_block_finite(self) -> None:
        """Without the 1e-4 floor an empty block divides by zero and poisons the payload."""
        transport = _fp8_transport(hidden=256)
        values, scales = transport.quantize_local(
            np.zeros((2, 256), dtype=np.float32).view(_Fp8Array))
        self.assertTrue(np.isfinite(np.asarray(values)).all())
        self.assertTrue(np.isfinite(np.asarray(scales)).all())
        np.testing.assert_allclose(np.asarray(scales), 1e-4 / 448.0, rtol=1e-6)

    def test_dequantize_is_the_algebraic_inverse(self) -> None:
        """With the cast as identity, dequantize(quantize(x)) must return x exactly.

        Catches a wrong reshape or a scale broadcast on the wrong axis, which would
        otherwise show up only as a correctness failure on hardware.
        """
        transport = _fp8_transport(hidden=256)
        x = ((np.arange(4 * 256, dtype=np.float32).reshape(4, 256) % 257 - 128)
             / 64.0).view(_Fp8Array)
        values, scales = transport.quantize_local(x)
        back = transport.dequantize_local(values, scales)
        np.testing.assert_allclose(np.asarray(back, dtype=np.float32),
                                   np.asarray(x, dtype=np.float32), rtol=1e-5, atol=1e-6)

    def test_the_fp8_byte_basis_is_read_off_a_real_fp8_transport(self) -> None:
        """The previous test of this name only checked the BF16 class defaults."""
        transport = _fp8_transport(hidden=7168)
        self.assertTrue(transport.fp8)
        self.assertEqual(transport.dispatch_value_bytes, 1)
        self.assertEqual(transport.dispatch_scale_bytes_per_copy, (7168 // 128) * 4)
        self.assertEqual(transport.dispatch_dtype, "fp8-e4m3fn")
        self.assertEqual(transport.combine_dtype, "bf16")

    def test_fp8_budget_counts_the_dequantized_copy_too(self) -> None:
        """fp8 is not simply half the footprint.

        It holds 1-byte values, FP32 block scales AND the dequantized BF16 combine input
        simultaneously. Billing it at one `value_bytes` under-counts, and the budget would
        then admit a ladder point that OOMs on the device.
        """
        _, layout = _layout()
        ladder = [layout.tokens_per_rank]
        layout_for = {layout.tokens_per_rank: layout}.__getitem__
        hidden, huge = 7168, 1 << 62

        def need(**kwargs):
            captured = []
            original = run_ep_jax._ladder_within_budget

            class _Probe:
                def __getitem__(self, tokens):
                    return layout

            kept, _ = original(ladder, layout_for, hidden, huge, **kwargs)
            self.assertEqual(kept, ladder)
            # Binary-search the smallest budget that still admits the point: that IS the
            # computed requirement, without exposing an internal.
            low, high = 0, huge
            while low < high:
                middle = (low + high) // 2
                if original(ladder, layout_for, hidden, middle, **kwargs)[0]:
                    high = middle
                else:
                    low = middle + 1
            return low

        bf16 = need(value_bytes=2, scale_bytes_per_copy=0, holds_dequantized=False)
        fp8 = need(value_bytes=1, scale_bytes_per_copy=(hidden // 128) * 4,
                   holds_dequantized=True)
        # fp8's receive side is 1 byte + scales + a 2-byte dequantized copy = MORE than
        # bf16's 2 bytes, so its requirement must not come out lower.
        self.assertGreater(fp8, bf16)

    def test_values_are_cast_to_e4m3_and_scales_stay_fp32(self) -> None:
        """The cast TARGET, not just the arithmetic around it.

        If `.astype(jnp.float8_e4m3fn)` were deleted or widened, the oracle would still
        pass -- it compares arrived bytes against staged bytes, and a wider wire moves the
        wider bytes consistently to both -- while byte_provenance still claimed 1 byte per
        value. The published fp8 bandwidth would then be fp8 logical bytes over the wall
        time of a wider exchange: wrong AND green. Nothing else in the suite can see that.
        """
        transport = _fp8_transport(hidden=256)
        x = ((np.arange(2 * 256, dtype=np.float32).reshape(2, 256) % 257 - 128)
             / 64.0).view(_Fp8Array)
        _Fp8Array.requested.clear()
        values, scales = transport.quantize_local(x)
        rows, hidden = x.shape
        blocks = hidden // ep_jax.QUANT_BLOCK

        # e4m3 must be requested for the VALUES, identified by shape. The values are cast
        # while still block-shaped, so accept either (rows, blocks, 128) or (rows, hidden).
        value_shapes = {(rows, blocks, ep_jax.QUANT_BLOCK), (rows, hidden)}
        e4m3_targets = {shape for dtype, shape in _Fp8Array.requested
                        if dtype == "float8_e4m3fn"}
        self.assertTrue(e4m3_targets & value_shapes,
                        f"e4m3 was not requested for the values; got {e4m3_targets}")
        # And NOTHING may narrow the scales: the byte basis charges 4 bytes each.
        narrowed_scales = {(dtype, shape) for dtype, shape in _Fp8Array.requested
                           if shape == (rows, blocks)
                           and dtype in _Fp8Array._DEVICE_CASTS}
        self.assertEqual(narrowed_scales, set(),
                         "scales were narrowed; a swap moves 2 bytes per value billed at 1")
        # And NOTHING may request bfloat16 anywhere in quantize_local. The recipe needs
        # only float32 and float8_e4m3fn, so this is clean -- and it closes WIDEN-AFTER
        # (`.astype(e4m3).astype(bfloat16)`), which satisfies every shape-based assertion
        # above while putting 2 bytes per value on a wire billed at 1.
        self.assertEqual([entry for entry in _Fp8Array.requested
                          if entry[0] == "bfloat16"], [],
                         "quantize_local must not widen back to bf16")
        self.assertEqual(np.asarray(scales).dtype, np.float32)
        # The e4m3 cast must be the LAST thing that touched the values, not merely present
        # somewhere in the expression. `view.astype(e4m3) * (448/amax)` requests e4m3 for a
        # value-shaped array and satisfies every assertion above, while in real JAX the
        # f8*f32 product promotes back to float32 -- 4 bytes per value on a wire billed at
        # 1. Only provenance can see that; the shim cannot return a narrower dtype.
        self.assertEqual(getattr(values, "_cast_to", None), "float8_e4m3fn",
                         "the values returned are not the direct result of the e4m3 cast; "
                         "something scaled or widened them after it")
        self.assertIsNone(getattr(scales, "_cast_to", None),
                          "the scales were produced by a narrowing cast")

    def test_amax_uses_magnitude_so_negatives_cannot_be_dropped(self) -> None:
        """Guards `jnp.abs`. Every other vector here is non-negative, so dropping abs
        would pass them all; a block whose extreme is negative would then get a negative
        scale and flip the sign of every value in it -- including the source-ID prefix the
        oracle decodes."""
        transport = _fp8_transport(hidden=256)
        x = np.zeros((2, 256), dtype=np.float32)
        x[:, 0] = -8.0        # block 0's extreme is NEGATIVE
        x[:, 5] = 1.0
        x[:, 128] = 2.0
        _, scales = transport.quantize_local(x.view(_Fp8Array))
        np.testing.assert_allclose(np.asarray(scales)[:, 0], 8.0 / 448.0, rtol=1e-6)
        # Block 1's extreme is POSITIVE, so `amax = -min(x)` would floor-clamp it instead
        # of returning 2/448. Asserting both blocks makes this test pin `abs` on its own
        # rather than only jointly with the axis test.
        np.testing.assert_allclose(np.asarray(scales)[:, 1], 2.0 / 448.0, rtol=1e-6)
        self.assertTrue((np.asarray(scales) > 0).all(),
                        "a negative scale would invert every value in the block")

    def test_bandwidth_bills_each_direction_at_its_own_basis(self) -> None:
        """dispatch is fp8; combine and the reference span are BF16 on both precisions.

        One case-level byte figure understated combine and the reference by ~1.94x at
        hidden=7168. An earlier fix corrected dispatch and broke those two -- the same
        mistake in the other direction -- so pin all three.
        """
        fp8_bytes, bf16_bytes = 7392.0, 14336.0     # per copy: 7168+224 vs 7168*2
        entry = run_ep_jax.record_collective_bandwidth(
            {}, transport=1000.0, span=2000.0, egress_bytes=fp8_bytes,
            combine_transport=4000.0, bf16_egress_bytes=bf16_bytes,
        )
        self.assertAlmostEqual(entry["collective_bandwidth_gbps_per_device"],
                               fp8_bytes / 1e-3 / 1e9)
        self.assertAlmostEqual(entry["collective_bandwidth_gbps_per_device_combine"],
                               bf16_bytes / 4e-3 / 1e9)
        self.assertAlmostEqual(entry["collective_bandwidth_gbps_per_device_span"],
                               bf16_bytes / 2e-3 / 1e9)

    def test_a_zero_dispatch_basis_is_not_treated_as_absent(self) -> None:
        """`or` would fall through on 0.0 and rebill dispatch at the BF16 basis.

        Zero egress is legitimate -- every copy destined for its own rank -- and it must
        publish 0 bandwidth, not silently switch byte bases.
        """
        entry = run_ep_jax.record_collective_bandwidth(
            {}, transport=1000.0, span=2000.0, egress_bytes=0.0,
            combine_transport=4000.0, bf16_egress_bytes=14336.0,
        )
        self.assertEqual(entry["collective_bandwidth_gbps_per_device"], 0.0)
        # ...while combine still uses the BF16 basis it was given.
        self.assertAlmostEqual(entry["collective_bandwidth_gbps_per_device_combine"],
                               14336.0 / 4e-3 / 1e9)

    def test_bandwidth_falls_back_to_one_basis_when_no_bf16_figure_is_given(self) -> None:
        """bf16 rows pass a single basis; the two must then agree."""
        entry = run_ep_jax.record_collective_bandwidth(
            {}, transport=1000.0, span=2000.0, egress_bytes=14336.0,
            combine_transport=4000.0,
        )
        self.assertAlmostEqual(entry["collective_bandwidth_gbps_per_device_combine"],
                               14336.0 / 4e-3 / 1e9)

    def test_egress_bills_values_and_scales(self) -> None:
        """fp8 egress must not be billed at bf16 bytes: that published ~1.9x the
        achieved bandwidth, which can exceed the physical link rate."""
        _, layout = _layout()
        bf16 = ep_jax.egress_bytes(layout, 7168, 2, 0)
        fp8 = ep_jax.egress_bytes(layout, 7168, 1, (7168 // 128) * 4)
        self.assertLess(fp8["mean_per_device"], bf16["mean_per_device"])
        # 1 byte/value + 224 scale bytes against 2 bytes/value.
        self.assertAlmostEqual(fp8["mean_per_device"] / bf16["mean_per_device"],
                               (7168 + 224) / (7168 * 2), places=6)


class Fp8CodecTests(unittest.TestCase):
    """The fp8 codec, checked against the recipe the GPU backends use.

    Blockwise e4m3fn with one FP32 scale per 128 elements -- DeepEP's
    `per_token_cast_to_fp8`: amax over the block clamped at 1e-4, scale = amax/448. A
    different block size or a per-tensor scale would move different bytes and stop the fp8
    rows being comparable with deepep-v2/uccl-ep/flashinfer-ep.
    """

    def test_byte_basis_matches_the_codec(self) -> None:
        """1 byte per value plus one FP32 scale per 128-block; bf16 carries no scales."""
        self.assertEqual(ep_jax.QUANT_BLOCK, 128)
        self.assertEqual(ep_jax.E4M3_MAX, 448.0)
        transport = ep_jax.JaxEPTransport.__new__(ep_jax.JaxEPTransport)
        self.assertEqual(transport.dispatch_value_bytes, 2)
        self.assertEqual(transport.dispatch_scale_bytes_per_copy, 0)
        self.assertFalse(transport.fp8)

    def test_every_precision_the_matrix_schedules_is_accepted_by_the_entrypoint(self) -> None:
        """The matrix and the argv gate must agree on which precisions exist.

        They did not: sweep_matrix was widened to (bf16, fp8) while run_ep_jax kept a
        `precision != "bf16"` guard, so the fp8 shard was scheduled, launched a TPU pod, and
        died on "jax-ragged-a2a is BF16-only" 48 seconds in. It failed closed rather than
        publishing anything, but it burned a hardware run to find a disagreement two files
        apart that this test settles offline.
        """
        sys.path.insert(0, str(COLLX))
        import sweep_matrix  # noqa: PLC0415

        scheduled = sweep_matrix.BACKEND_PRECISIONS["jax-ragged-a2a"]
        source = (COLLX / "bench" / "run_ep_jax.py").read_text()
        for precision in scheduled:
            self.assertIn(f'"{precision}"', source,
                          f"{precision} is scheduled but the entrypoint never names it")
        # And the gate must still reject something, or it is not a gate.
        self.assertIn('not in ("bf16", "fp8")', source)

    def test_fp8_rejects_a_hidden_the_block_does_not_divide(self) -> None:
        """Silently truncating a partial block would move fewer bytes than it claims."""
        stub = _stub_jax()
        with self.assertRaises(ValueError):
            ep_jax.JaxEPTransport(stub, stub.numpy, lambda f, **k: f, object(),
                                  8, 7168 + 1, precision="fp8")
        with self.assertRaises(ValueError):
            ep_jax.JaxEPTransport(stub, stub.numpy, lambda f, **k: f, object(),
                                  8, 7168, precision="int4")

    @staticmethod
    def _block_amax(x, block=128):
        rows = x.shape[0]
        view = x.reshape(rows, x.shape[1] // block, block).astype(np.float32)
        return np.clip(np.abs(view).max(axis=2), 1e-4, None)

    @staticmethod
    def _e4m3_spacing(magnitude):
        """Gap between adjacent e4m3fn normals near `magnitude`.

        4 exponent bits, 3 mantissa bits, max 448 = 1.75 x 2^8. A value is exactly
        representable iff it is a multiple of this spacing.
        """
        exponent = np.clip(np.floor(np.log2(np.maximum(magnitude, 1e-30))), -6, 8)
        return np.exp2(exponent - 3)

    def test_source_id_prefix_clears_the_stability_guard_under_the_codec(self) -> None:
        """The oracle decodes the source ID from the SIGN of the first columns.

        It does NOT round-trip exactly, which was my first assumption and it was wrong: the
        block amax is max(1.0, lattice max), and for a block whose lattice peaks at
        127/64 the prefix scales to 225.76 and rounds to 224, coming back as 0.9922.

        What has to hold is weaker and has huge margin. e4m3 keeps 3 mantissa bits, so
        rounding costs at most a relative 2^-4, and the prefix is +/-1.0, so it returns with
        magnitude >= 0.9375 against `decode_source_ids`' 0.25 guard -- and the scale is
        positive, so the sign is untouched. Asserted as arithmetic; an e4m3 emulator written
        here would only test the emulator.
        """
        hidden, rows = 256, 8
        x = routing_np.activations_for_source_ids(
            np.arange(rows, dtype=np.int64) + 1, hidden, seed=67)
        amax = self._block_amax(x)

        # The lattice is k/64 for k in [-128, 128], so no block amax can exceed 2.0, and
        # the prefix itself is 1.0, so no amax is below it.
        self.assertTrue((amax <= 2.0).all())
        self.assertTrue((amax >= 1.0).all())

        worst_relative_error = 2.0 ** -4
        floor = 1.0 * (1.0 - worst_relative_error)
        self.assertGreaterEqual(floor, 0.25 * 3,
                                "the prefix must clear the 0.25 guard with real margin")
        # And the scaled prefix stays inside e4m3's normal range, so it cannot land on a
        # subnormal where the relative error would be larger than 2^-4.
        scaled = 448.0 / amax[:, 0]
        self.assertTrue(((scaled >= 2.0 ** -6) & (scaled <= 448.0)).all())

    def test_the_lattice_columns_are_lossy_under_the_codec(self) -> None:
        """Guards WHY the fp8 combine is scored against delivered rows, not pristine x.

        If the payload round-tripped exactly, the codec would be doing nothing and the
        pristine bf16 expectation would still work. At least one lattice value must fail to
        land on an e4m3 grid point -- that loss is what exceeds COMBINE_REL_TOL and forces
        the delivered-rows expectation.
        """
        hidden, rows = 256, 8
        x = routing_np.activations_for_source_ids(
            np.arange(rows, dtype=np.int64) + 1, hidden, seed=67)
        amax = self._block_amax(x)
        view = x.reshape(rows, hidden // 128, 128).astype(np.float32)
        scaled = np.abs(view) * (448.0 / amax[:, :, None])
        off_grid = np.mod(scaled, self._e4m3_spacing(scaled)) != 0.0
        self.assertTrue(off_grid.any(),
                        "no value needs rounding: the cast would be lossless, so the "
                        "fp8 leg would be measuring bf16 with extra steps")


class PermuteScopeTests(unittest.TestCase):
    """The permute is scoped, so its cost is measured rather than left to subtraction.

    `non_transport_us` was derived as component-minus-transport. That silently charged the
    permute for whatever the transport scope missed -- and dispatch's scope reports
    6,689us where combine's reports 9,973us for the same collective, because XLA fuses the
    collective's second op into an adjacent gather that only dispatch has.
    """

    def test_permute_and_transport_are_distinct_sibling_scopes(self) -> None:
        permute = ep_jax.permute_scope("dispatch", 8192)
        transport = ep_jax.transport_scope("dispatch", 8192)
        self.assertNotEqual(permute, transport)
        # Neither may be a substring of the other, or marker_in would match both and the
        # two costs would pool -- the same defect that once made four HLO rows duplicate.
        self.assertNotIn(permute, transport)
        self.assertNotIn(transport, permute)
        self.assertTrue(permute.startswith(ep_jax.SCOPE_PREFIX))

    def test_the_permute_scope_is_traced_for_components(self) -> None:
        _, inner = run_ep_jax.traced_markers("dispatch", 8192)
        self.assertIn(ep_jax.permute_scope("dispatch", 8192), inner)
        self.assertIn(ep_jax.transport_scope("dispatch", 8192), inner)
        # The reference program has no permute, so it must not claim one.
        _, reference_inner = run_ep_jax.traced_markers(run_ep_jax.REFERENCE, 8192)
        self.assertNotIn(ep_jax.permute_scope(run_ep_jax.REFERENCE, 8192), reference_inner)

    def test_the_permute_is_scoped_inside_dispatch(self) -> None:
        """The gather must actually be wrapped, not just the helper exist."""
        scopes = []
        stub = _stub_jax()
        stub.named_scope = lambda name: _recording_scope(scopes, name)
        transport = ep_jax.JaxEPTransport.__new__(ep_jax.JaxEPTransport)
        transport.jax, transport.jnp = stub, stub.numpy
        transport.mesh, transport.ep_size, transport.hidden = object(), 4, 64
        transport.axis = "ep"
        transport._P = stub.sharding.PartitionSpec
        transport.shard_map = lambda fn, **kwargs: fn
        _, layout = _layout()
        point = ep_jax.Point(
            transport=transport, layout=layout, x=stub.array(),
            send_index=stub.array(), input_offsets=stub.array(),
            send_sizes=stub.array(), output_offsets=stub.array(),
            recv_sizes=stub.array(), recv_offsets=stub.array(),
            return_offsets=stub.array(),
        )
        point.dispatch()
        self.assertIn(ep_jax.permute_scope("dispatch", layout.tokens_per_rank), scopes)


class OptimizationBarrierTests(unittest.TestCase):
    """The barrier at the permute/collective boundary must actually be emitted.

    It is the whole substance of the transport-attribution fix: without it XLA fuses part
    of the collective into the gather and dispatch's transport scope reads 6,685us against
    combine's 9,974us for the same collective -- a 1.49x under-attribution of the published
    figure. Deleting both calls used to leave the entire offline suite green, so a refactor
    could restore that silently. Nothing here needs a device; the barrier's PRESENCE is a
    property of the traced program.
    """

    def _point(self, fp8: bool):
        calls = []
        stub = _stub_jax()
        stub.lax.optimization_barrier = lambda values: (calls.append(values) or values)
        _, layout = _layout()
        if fp8:
            # fp8 traces through the real codec, so it needs arrays with a dtype and a
            # shape rather than the opaque stub -- same shim the codec tests use.
            hidden = 2 * ep_jax.QUANT_BLOCK
            transport = _fp8_transport(hidden=hidden, ep_size=layout.ep_size, stub=stub)
            x = np.ones((1, layout.tokens_per_rank, hidden), np.float32).view(_Fp8Array)
            plan = {name: getattr(layout, name).astype(np.int32) for name in (
                "send_index", "input_offsets", "send_sizes", "output_offsets",
                "recv_sizes", "recv_offsets", "return_offsets")}
        else:
            transport = ep_jax.JaxEPTransport.__new__(ep_jax.JaxEPTransport)
            transport.jax, transport.jnp = stub, stub.numpy
            transport.mesh, transport.ep_size, transport.hidden = object(), 4, 64
            transport.axis = "ep"
            transport._P = stub.sharding.PartitionSpec
            transport.shard_map = lambda fn, **kwargs: fn
            x = stub.array()
            plan = {name: stub.array() for name in (
                "send_index", "input_offsets", "send_sizes", "output_offsets",
                "recv_sizes", "recv_offsets", "return_offsets")}
        return calls, ep_jax.Point(transport=transport, layout=layout, x=x, **plan)

    def test_bf16_dispatch_barriers_the_staged_operand(self) -> None:
        calls, point = self._point(fp8=False)
        point.dispatch()
        self.assertEqual(len(calls), 1, "the bf16 permute/collective barrier is missing")

    def test_fp8_dispatch_barriers_both_staged_operands(self) -> None:
        """fp8 stages values AND scales, and barriers them as one tuple."""
        calls, point = self._point(fp8=True)
        point.dispatch()
        self.assertEqual(len(calls), 1, "the fp8 permute/collective barrier is missing")
        self.assertEqual(len(calls[0]), 2, "fp8 must barrier values and scales together")

    def test_the_barrier_is_between_the_permute_and_the_collective(self) -> None:
        """Ordering, not merely presence: a barrier after the collective buys nothing."""
        events = []
        stub = _stub_jax()
        stub.named_scope = lambda name: _recording_scope(events, name)
        stub.lax.optimization_barrier = lambda values: (events.append("BARRIER") or values)
        transport = ep_jax.JaxEPTransport.__new__(ep_jax.JaxEPTransport)
        transport.jax, transport.jnp = stub, stub.numpy
        transport.mesh, transport.ep_size, transport.hidden = object(), 4, 64
        transport.axis = "ep"
        transport._P = stub.sharding.PartitionSpec
        transport.shard_map = lambda fn, **kwargs: fn
        _, layout = _layout()
        point = ep_jax.Point(
            transport=transport, layout=layout, x=stub.array(),
            send_index=stub.array(), input_offsets=stub.array(),
            send_sizes=stub.array(), output_offsets=stub.array(),
            recv_sizes=stub.array(), recv_offsets=stub.array(),
            return_offsets=stub.array(),
        )
        point.dispatch()
        permute = events.index(ep_jax.permute_scope("dispatch", layout.tokens_per_rank))
        collective = events.index(
            ep_jax.transport_scope("dispatch", layout.tokens_per_rank))
        self.assertLess(permute, events.index("BARRIER"))
        self.assertLess(events.index("BARRIER"), collective)


class ByRankTests(unittest.TestCase):
    """Regrouping shard_map's concatenated output back to per-rank.

    The components return per-shard 2D results now, because expanding them to a leading
    axis of size 1 inside the program re-materialised the whole buffer every call
    (3,231us in dispatch at T=8192). shard_map concatenates them, so the oracle has to
    regroup -- and a wrong regroup would not crash, it would feed the oracle correctly
    shaped but mis-grouped data and report a correctness failure on a working transport.
    """

    def test_regrouping_inverts_the_concatenation_exactly(self) -> None:
        ep, rows, hidden = 4, 5, 3
        per_rank = np.arange(ep * rows * hidden, dtype=np.float32).reshape(
            ep, rows, hidden)
        # What shard_map produces: each shard's (rows, hidden) block, concatenated.
        concatenated = np.concatenate([per_rank[rank] for rank in range(ep)], axis=0)
        self.assertEqual(concatenated.shape, (ep * rows, hidden))
        np.testing.assert_array_equal(
            run_ep_jax._by_rank(concatenated, ep), per_rank)

    def test_an_already_grouped_array_passes_through(self) -> None:
        values = np.zeros((4, 5, 3), dtype=np.float32)
        self.assertIs(run_ep_jax._by_rank(values, 4), values)

    def test_regrouping_uses_the_rank_count_it_is_given(self) -> None:
        # Guards the reshape argument order: (ep, -1, hidden), not (-1, ep, hidden).
        # Both are legal reshapes of the same buffer and only one is right.
        ep, rows, hidden = 2, 3, 2
        per_rank = np.arange(ep * rows * hidden, dtype=np.float32).reshape(
            ep, rows, hidden)
        concatenated = np.concatenate([per_rank[rank] for rank in range(ep)], axis=0)
        regrouped = run_ep_jax._by_rank(concatenated, ep)
        self.assertEqual(regrouped.shape, (ep, rows, hidden))
        np.testing.assert_array_equal(regrouped[0], per_rank[0])
        np.testing.assert_array_equal(regrouped[1], per_rank[1])


class ReferenceLoopTests(unittest.TestCase):
    """The timed window must contain the collective and nothing else.

    Freeing the previous ~5GB output inside the window read as a host "floor" that grew
    with payload -- +2,634us at T=8192 against ~480us below T=512. Holding the result
    past the timer fixes that, but must not turn into retaining every result: at 67
    iterations that would be hundreds of GB.
    """

    def test_no_previous_result_is_still_alive_when_the_next_call_starts(self) -> None:
        """Checked DURING the loop, not after.

        A post-hoc weakref sweep cannot distinguish "released each iteration" from
        "retained in a local that died when the function returned" -- a retain-everything
        mutant passed that version of this test. Counting live results at call time is
        what separates them, and it is also the property that matters: at 67 iterations
        of a ~5GB output, retaining is hundreds of GB.
        """
        import weakref

        class Result:
            pass

        refs, live_at_call = [], []

        def fn():
            live_at_call.append(sum(1 for ref in refs if ref() is not None))
            result = Result()
            refs.append(weakref.ref(result))
            return result

        jax = types.SimpleNamespace(block_until_ready=lambda value: value)
        samples = ep_jax.reference_timing(
            jax, fn, warmup_tries=1, num_runs=3, min_duration_s=0.0)
        self.assertGreaterEqual(len(samples), 3)
        self.assertGreater(len(live_at_call), 3)
        self.assertEqual(max(live_at_call), 0,
                         f"a previous result was still alive at call time: {live_at_call}")
        self.assertEqual([ref for ref in refs if ref() is not None], [])


class CollectiveBandwidthTests(unittest.TestCase):
    """Both device denominators ship, bracketing the collective from either side.

    Bandwidth was derived from `transport_us` alone, which counts only ops whose name
    carries the HYPHENATED HLO and misses an underscored variant, so it read optimistic.
    The span overshoots instead: it includes output-buffer materialisation that is not
    collective work. Neither is the collective, so the note must present them as bounds
    rather than nominate a winner -- an earlier version said "prefer the span", which was
    wrong in the other direction.
    """

    def test_every_denominator_ships_and_the_note_names_the_trustworthy_one(self) -> None:
        entry = run_ep_jax.record_collective_bandwidth(
            {}, transport=6682.8, span=10322.8, egress_bytes=544038656.0,
            combine_transport=9972.9)
        self.assertAlmostEqual(entry["collective_device_us"], 6682.8)
        self.assertAlmostEqual(entry["collective_device_us_combine"], 9972.9)
        self.assertAlmostEqual(entry["collective_device_us_span"], 10322.8)
        named = entry["collective_bandwidth_gbps_per_device"]
        combined = entry["collective_bandwidth_gbps_per_device_combine"]
        self.assertAlmostEqual(named, 81.4, places=0)
        self.assertAlmostEqual(combined, 54.6, places=0)
        # Dispatch's denominator flatters the result by ~1.5x because it loses an op to
        # XLA folding; a reader taking the first field they find must be told which one.
        self.assertGreater(named, combined)
        note = entry["collective_device_us_note"]
        # The note must present the disagreement and point at the evidence, NOT nominate
        # a winner: two successive explanations for the differing op were each disproved
        # by the next run, so the artifact states what was measured and stops there.
        self.assertIn("unresolved", note)
        self.assertIn("op_inventory", note)
        self.assertNotIn("Prefer", note)

    def test_a_missing_combine_capture_yields_null_not_a_wrong_number(self) -> None:
        entry = run_ep_jax.record_collective_bandwidth(
            {}, transport=1.0, span=None, egress_bytes=1e9, combine_transport=None)
        self.assertIsNone(entry["collective_bandwidth_gbps_per_device_combine"])
        self.assertIsNone(entry["collective_bandwidth_gbps_per_device_span"])

    def test_a_missing_capture_yields_null_not_a_wrong_number(self) -> None:
        entry = run_ep_jax.record_collective_bandwidth(
            {}, transport=None, span=None, egress_bytes=1e9)
        self.assertIsNone(entry["collective_bandwidth_gbps_per_device"])
        self.assertIsNone(entry["collective_bandwidth_gbps_per_device_span"])


class OpInventoryTests(unittest.TestCase):
    """Itemising what summed op time is made of, by name.

    `non_transport_us` was read as "the permute". The reference program disproves that:
    its only source-level operation is the collective, yet ops carrying the collective's
    name account for half its device time. The remainder has to be named, not narrated.
    """

    @staticmethod
    def _event(name, pid, start, dur):
        return {"name": name, "pid": pid, "ts": start, "dur": dur,
                "args": {"device_duration_ps": str(int(dur * 1e6))}}

    def test_inventory_itemises_the_slowest_device_biggest_first(self) -> None:
        events = [
            self._event("ragged-all-to-all", 1, 0, 100),
            self._event("fusion.1", 1, 100, 60),
            self._event("fusion.1", 1, 200, 40),
            self._event("copy.2", 1, 300, 10),
            # A faster device must not contribute: mixing devices would stop the
            # inventory adding up to op_total_device_us.
            self._event("ragged-all-to-all", 2, 0, 5),
            self._event("only-on-fast-device", 2, 10, 5),
        ]
        inventory = xprof.op_inventory(events, occurrences=2)
        self.assertEqual([entry["name"] for entry in inventory],
                         ["ragged-all-to-all", "fusion.1", "copy.2"])
        self.assertNotIn("only-on-fast-device",
                         [entry["name"] for entry in inventory])
        self.assertAlmostEqual(inventory[0]["per_iteration_us"], 50.0)
        self.assertAlmostEqual(inventory[1]["per_iteration_us"], 50.0)
        self.assertAlmostEqual(inventory[1]["events_per_iteration"], 1.0)

        # It must reconcile with the number it explains, or it is decoration.
        summed = xprof.sum_across_devices(events, occurrences=2)
        self.assertAlmostEqual(sum(e["per_iteration_us"] for e in inventory),
                               summed["per_iteration_us"])

    def test_inventory_is_empty_rather_than_raising_without_events(self) -> None:
        self.assertEqual(xprof.op_inventory([], occurrences=4), [])
        self.assertEqual(xprof.op_inventory(
            [self._event("x", 1, 0, 1)], occurrences=0), [])


class XprofParserTests(unittest.TestCase):
    """Device-duration extraction from a profiler trace.

    Unlike the amortization -- whose failure only existed on hardware and which therefore
    shipped broken past eight passing tests -- this parser consumes a plain gzipped
    Chrome-trace JSON, so its full behaviour including the cross-device reduction is
    exercised offline against synthetic traces.
    """

    @staticmethod
    def _write_trace(directory, events, name="plugins/profile/run/host.trace.json.gz"):
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"traceEvents": events}, handle)
        return path

    @staticmethod
    def _event(name, pid, ts, *, device_ps=None, dur=None, tf_op=""):
        event = {"name": name, "pid": pid, "ts": ts, "args": {"tf_op": tf_op}}
        if device_ps is not None:
            event["args"]["device_duration_ps"] = device_ps
        if dur is not None:
            event["dur"] = dur
        return event

    def test_a_ladder_label_does_not_match_a_longer_one(self) -> None:
        """Regression: `-t1` is a substring of `-t16`, `-t128`, `-t1024`.

        A substring match silently pooled four ladder points into one series -- observed
        on hardware as T=1 and T=16 reporting byte-identical device durations. The whole
        decode ladder is checked, not just the pair that happened to be caught.
        """
        ladder = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        colliding = [
            (short, long)
            for short in ladder for long in ladder
            if short != long
            and ep_jax.scope_name("dispatch", short) in ep_jax.scope_name("dispatch", long)
        ]
        self.assertTrue(colliding, "expected raw substrings to collide; test is vacuous")
        for short, long in colliding:
            with self.subTest(short=short, long=long):
                self.assertFalse(
                    xprof.marker_in(
                        f"jit(f)/{ep_jax.scope_name('dispatch', long)}/ragged-all-to-all",
                        ep_jax.scope_name("dispatch", short),
                    ),
                    f"T={short} must not match T={long}'s events",
                )

    def test_a_label_still_matches_its_own_events_with_suffixes(self) -> None:
        marker = ep_jax.scope_name("dispatch", 1)
        for text in (marker, f"jit(f)/{marker}/ragged-all-to-all",
                     f"{marker}.2", f"{marker}_1", f"a/{marker}/b"):
            with self.subTest(text=text):
                self.assertTrue(xprof.marker_in(text, marker))

    def test_end_to_end_parse_isolates_colliding_ladder_points(self) -> None:
        short, long = ep_jax.scope_name("dispatch", 1), ep_jax.scope_name("dispatch", 16)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                self._event("ragged-all-to-all", pid=0, ts=1, device_ps=5_000_000,
                            tf_op=f"jit(f)/{short}/ragged-all-to-all"),
                self._event("ragged-all-to-all", pid=0, ts=2, device_ps=900_000_000,
                            tf_op=f"jit(f)/{long}/ragged-all-to-all"),
            ])
            small = xprof.parse_trace_durations(directory, short, occurrences=1)
            large = xprof.parse_trace_durations(directory, long, occurrences=1)
        self.assertEqual(small["op_per_iteration_us"], 5.0)
        self.assertEqual(large["op_per_iteration_us"], 900.0)

    def test_a_scope_totals_its_ops_rather_than_taking_a_median(self) -> None:
        """Regression: a scope holds several HLO ops per iteration.

        Zipping a scope's events positionally treats element k as iteration k, which it
        is not -- it produced a median over unrelated ops, observed on hardware as
        roundtrip reporting the same duration as dispatch when it must be roughly double.
        """
        marker = ep_jax.scope_name("roundtrip", 512)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                # One iteration: two collectives plus a fusion. Total 1000us.
                self._event("ragged-all-to-all.1", pid=0, ts=1, device_ps=400_000_000,
                            tf_op=marker),
                self._event("ragged-all-to-all.2", pid=0, ts=2, device_ps=400_000_000,
                            tf_op=marker),
                self._event("fusion.7", pid=0, ts=3, device_ps=200_000_000,
                            tf_op=marker),
            ])
            parsed = xprof.parse_trace_durations(
                directory, marker, occurrences=1,
                hlo_substrings=("ragged-all-to-all", "fusion"),
            )
        self.assertEqual(parsed["op_total_device_us"], 1000.0)
        self.assertEqual(parsed["op_per_iteration_us"], 1000.0)  # NOT the 400us median
        self.assertEqual(parsed["op_matched_events"], 3)
        # The collective alone is 800 of that 1000 -- the decomposition combine needs.
        self.assertEqual(parsed["by_hlo"]["ragged-all-to-all"]["total_device_us"], 800.0)
        self.assertEqual(parsed["by_hlo"]["fusion"]["total_device_us"], 200.0)

    def test_per_iteration_divides_by_the_occurrence_count(self) -> None:
        marker = ep_jax.scope_name("dispatch", 64)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                self._event("op", pid=0, ts=index, device_ps=100_000_000, tf_op=marker)
                for index in range(8)
            ])
            parsed = xprof.parse_trace_durations(directory, marker, occurrences=8)
        self.assertEqual(parsed["op_total_device_us"], 800.0)
        self.assertEqual(parsed["op_per_iteration_us"], 100.0)

    def test_scope_total_takes_the_slowest_device(self) -> None:
        marker = ep_jax.scope_name("dispatch", 32)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                self._event("a", pid=0, ts=1, device_ps=100_000_000, tf_op=marker),
                self._event("b", pid=0, ts=2, device_ps=100_000_000, tf_op=marker),
                self._event("a", pid=1, ts=1, device_ps=500_000_000, tf_op=marker),
                self._event("b", pid=1, ts=2, device_ps=100_000_000, tf_op=marker),
            ])
            parsed = xprof.parse_trace_durations(directory, marker, occurrences=1)
        self.assertEqual(parsed["op_devices"], 2)
        self.assertEqual(parsed["op_total_device_us"], 600.0)  # device 1, not device 0

    def test_reduces_with_max_across_devices_not_the_first_device(self) -> None:
        """The reference implementation keeps min(pid) only; a collective finishes when
        its SLOWEST participant does, so this must take the max."""
        marker = ep_jax.scope_name("dispatch", 512)
        # occurrence 0 then 1, on two devices; device 1 is slower both times.
        reduced = xprof.reduce_across_devices([
            self._event(marker, pid=0, ts=10, device_ps=1_000_000),   # 1us
            self._event(marker, pid=0, ts=30, device_ps=2_000_000),   # 2us
            self._event(marker, pid=1, ts=11, device_ps=5_000_000),   # 5us
            self._event(marker, pid=1, ts=31, device_ps=9_000_000),   # 9us
        ])
        self.assertEqual(reduced["devices"], 2)
        self.assertEqual(reduced["per_occurrence_us"], [5.0, 9.0])
        self.assertEqual(reduced["dropped_occurrences"], 0)

    def test_orders_occurrences_by_timestamp_within_a_device(self) -> None:
        marker = ep_jax.scope_name("roundtrip", 8)
        reduced = xprof.reduce_across_devices([
            self._event(marker, pid=0, ts=99, device_ps=7_000_000),
            self._event(marker, pid=0, ts=1, device_ps=3_000_000),
        ])
        self.assertEqual(reduced["per_occurrence_us"], [3.0, 7.0])

    def test_uneven_occurrence_counts_are_reported_not_silently_dropped(self) -> None:
        marker = ep_jax.scope_name("combine", 64)
        reduced = xprof.reduce_across_devices([
            self._event(marker, pid=0, ts=1, device_ps=1_000_000),
            self._event(marker, pid=0, ts=2, device_ps=1_000_000),
            self._event(marker, pid=1, ts=1, device_ps=4_000_000),
        ])
        self.assertEqual(reduced["per_occurrence_us"], [4.0])
        self.assertEqual(reduced["dropped_occurrences"], 1)

    def test_falls_back_to_wall_duration_when_no_device_counter(self) -> None:
        marker = ep_jax.scope_name("dispatch", 1)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                self._event(marker, pid=0, ts=1, dur=12.5),  # `dur` is already us
            ])
            parsed = xprof.parse_trace_durations(directory, marker, occurrences=1)
        self.assertEqual(parsed["op_total_device_us"], 12.5)

    def test_matches_the_marker_in_tf_op_as_well_as_name(self) -> None:
        marker = ep_jax.scope_name("dispatch", 2)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                self._event("fusion.3", pid=0, ts=1, device_ps=6_000_000,
                            tf_op=f"{marker}/ragged-all-to-all"),
            ])
            parsed = xprof.parse_trace_durations(directory, marker, occurrences=1)
        self.assertEqual(parsed["op_total_device_us"], 6.0)

    def test_breaks_out_named_hlo_ops_so_combine_can_be_decomposed(self) -> None:
        """combine's host latency fuses transport with a scatter-add; the per-HLO
        breakdown is the only thing that separates them."""
        marker = ep_jax.scope_name("combine", 512)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                self._event("ragged-all-to-all.1", pid=0, ts=1, device_ps=800_000_000,
                            tf_op=marker),
                self._event("scatter.2", pid=0, ts=2, device_ps=300_000_000,
                            tf_op=marker),
            ])
            parsed = xprof.parse_trace_durations(
                directory, marker, 1, ("ragged-all-to-all", "scatter"),
            )
        self.assertEqual(sorted(parsed["by_hlo"]), ["ragged-all-to-all", "scatter"])
        self.assertEqual(parsed["by_hlo"]["ragged-all-to-all"]["per_occurrence_us"], [800.0])
        self.assertEqual(parsed["by_hlo"]["scatter"]["per_occurrence_us"], [300.0])

    def test_missing_trace_and_missing_marker_degrade_rather_than_raise(self) -> None:
        # A failed capture must not fail the case; the host-timed measurement stands.
        with tempfile.TemporaryDirectory() as directory:
            empty = xprof.parse_trace_durations(directory, "collectivex-dispatch-t1", 1)
            self.assertIn("error", empty)
            self.assertIsNone(empty.get("total_device_us"))
            self._write_trace(directory, [self._event("unrelated", pid=0, ts=1, dur=1)])
            absent = xprof.parse_trace_durations(directory, "collectivex-dispatch-t1", 1)
        self.assertIn("error", absent)
        self.assertIsNone(absent["op_total_device_us"])

    def test_every_trace_file_is_searched_not_just_the_first(self) -> None:
        """A session can write several trace files; the marker may not be in the first.

        Picking the alphabetically-first file silently loses the marker when the device
        rows land elsewhere -- a plausible cause of coverage that was 14/14 on one run
        and 7/14 on the next with identical code.
        """
        marker = ep_jax.scope_name("dispatch", 128)
        with tempfile.TemporaryDirectory() as directory:
            # sorts first, but holds nothing relevant
            self._write_trace(directory, [self._event("unrelated", pid=0, ts=1, dur=1.0)],
                              name="a/host.trace.json.gz")
            # sorts later, holds the device rows
            self._write_trace(directory, [
                self._event("ragged-all-to-all", pid=0, ts=1, device_ps=700_000_000,
                            tf_op=marker),
            ], name="z/device.trace.json.gz")
            parsed = xprof.parse_trace_durations(directory, marker, occurrences=1)
        self.assertNotIn("error", parsed)
        self.assertEqual(parsed["per_occurrence_us"], [700.0])
        self.assertEqual(parsed["trace"], "device.trace.json.gz")
        self.assertEqual(parsed["traces_searched"], 2)

    def test_trace_file_choice_is_deterministic(self) -> None:
        marker = ep_jax.scope_name("dispatch", 4)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [self._event(marker, pid=0, ts=1, dur=1.0)],
                              name="b/second.trace.json.gz")
            self._write_trace(directory, [self._event(marker, pid=0, ts=1, dur=2.0)],
                              name="a/first.trace.json.gz")
            first = xprof.parse_trace_durations(directory, marker, occurrences=1)
            second = xprof.parse_trace_durations(directory, marker, occurrences=1)
        self.assertEqual(first["op_total_device_us"], second["op_total_device_us"])
        self.assertEqual(first["trace"], "first.trace.json.gz")

    def test_transport_scope_isolates_the_collective_inside_the_component(self) -> None:
        """The reference scopes the collective alone; this probe needs both scopes.

        Nested scopes concatenate in the trace, so a collective event matches the inner
        transport marker AND the outer component marker -- the component total still sums
        everything while the transport is separable. The two labels must not alias.
        """
        outer = ep_jax.scope_name("combine", 512)
        inner = ep_jax.transport_scope("combine", 512)
        self.assertNotEqual(outer, inner)
        self.assertFalse(xprof.marker_in(inner, outer),
                         "the component marker must not match the transport label alone")
        nested = f"jit(f)/{outer}/{inner}/ragged-all-to-all"
        self.assertTrue(xprof.marker_in(nested, outer))
        self.assertTrue(xprof.marker_in(nested, inner))
        # And the transport label is ladder-point safe for the same reason as the outer.
        self.assertFalse(
            xprof.marker_in(f"jit(f)/{ep_jax.transport_scope('combine', 5120)}/x",
                            ep_jax.transport_scope("combine", 512)),
        )

    def test_transport_scope_yields_a_separable_collective_total(self) -> None:
        outer = ep_jax.scope_name("combine", 64)
        inner = ep_jax.transport_scope("combine", 64)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                # collective (inside both scopes) + scatter (component scope only)
                self._event("ragged-all-to-all", pid=0, ts=1, device_ps=700_000_000,
                            tf_op=f"{outer}/{inner}"),
                self._event("scatter.4", pid=0, ts=2, device_ps=300_000_000,
                            tf_op=outer),
            ])
            parsed = xprof.parse_trace_durations(
                directory, outer, occurrences=1, hlo_substrings=(inner,),
            )
        self.assertEqual(parsed["op_per_iteration_us"], 1000.0)
        self.assertEqual(parsed["by_hlo"][inner]["per_iteration_us"], 700.0)

    def test_traced_hlo_entries_do_not_overlap_each_other(self) -> None:
        """Regression: a four-entry list published rows that double-counted.

        `all-to-all` is a substring of `ragged-all-to-all` so it duplicated it exactly,
        and `fusion` matched the same fused ops as `scatter` (observed at T=8192: 4446.3us
        and 4411.2us for what is one piece of work). What is not the collective is derived
        by subtraction now, so the list stays minimal and non-overlapping.
        """
        for one in ep_jax.TRACED_HLO:
            for other in ep_jax.TRACED_HLO:
                if one is not other:
                    self.assertNotIn(
                        one, other,
                        f"{one!r} matches everything {other!r} does; rows would duplicate",
                    )
        self.assertIn("ragged-all-to-all", ep_jax.TRACED_HLO)

    def test_iqr_filtering_matches_the_reference_statistics(self) -> None:
        """CollectiveX publishes raw percentiles; the reference filters outliers first.

        Keeping its statistics is what makes the reference number mean what the
        reference's numbers mean, so the filter is transcribed rather than approximated.
        """
        clean = [10.0] * 20
        with_outlier = clean + [10_000.0]
        filtered = ep_jax.iqr_metrics(with_outlier)
        self.assertEqual(filtered["samples"], 21)
        self.assertEqual(filtered["kept_after_iqr"], 20)
        self.assertAlmostEqual(filtered["avg_ms"], 10.0)
        # Three or fewer samples are never filtered (the reference's guard).
        tiny = ep_jax.iqr_metrics([1.0, 2.0, 900.0])
        self.assertEqual(tiny["kept_after_iqr"], 3)
        self.assertIsNone(ep_jax.iqr_metrics([]))

    def test_iqr_filtering_falls_back_when_it_would_empty_the_set(self) -> None:
        # The reference explicitly guards this; a bimodal set must not yield no samples.
        metrics = ep_jax.iqr_metrics([1.0, 1.0, 100.0, 100.0])
        self.assertEqual(metrics["kept_after_iqr"], 4)
        self.assertGreater(metrics["avg_ms"], 0)

    def test_egress_excludes_the_copy_that_never_leaves_the_device(self) -> None:
        """The reference counts chunk * (num_devices - 1): the self-chunk stays home.

        Counting it would inflate bandwidth by ~1/ep_size against any published figure.
        """
        _, layout = _layout()
        hidden = 64
        egress = ep_jax.egress_bytes(layout, hidden)
        expected_total = sum(
            int(layout.send_total[rank] - layout.send_sizes[rank, rank])
            for rank in range(layout.ep_size)
        ) * hidden * 2
        self.assertEqual(egress["total"], expected_total)
        # Strictly less than the full payload, by exactly the self-destined copies.
        self.assertLess(egress["total"], layout.payload_bytes(hidden))
        self_copies = sum(
            int(layout.send_sizes[rank, rank]) for rank in range(layout.ep_size)
        )
        self.assertGreater(self_copies, 0, "trace must exercise a local copy")
        self.assertEqual(
            layout.payload_bytes(hidden) - egress["total"], self_copies * hidden * 2,
        )
        self.assertGreaterEqual(egress["max_per_device"], egress["mean_per_device"])

    def test_reference_timing_honours_its_run_and_duration_floors(self) -> None:
        jax = types.SimpleNamespace(block_until_ready=lambda value: value)
        calls = []
        samples = ep_jax.reference_timing(
            jax, lambda: calls.append(1), warmup_tries=3, num_runs=5,
            min_duration_s=0.0,
        )
        self.assertEqual(len(samples), 5)
        self.assertEqual(len(calls), 8)  # 3 warmup + 5 measured
        for sample in samples:
            self.assertGreaterEqual(sample, 0.0)

    def test_traced_point_selection_defaults_to_the_ladder_ends(self) -> None:
        """Regression: tracing every point overran the prefill shard budget.

        Run 30736428447's prefill case died on case-timeout with the profiler enabled,
        having completed without it. The device numbers calibrate the host ladder, so the
        ends carry the information -- the floor's share at small T, the bandwidth-bound
        regime at large T -- at a fraction of the trace volume.
        """
        decode = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        prefill = [1024, 2048, 4096, 8192]
        self.assertEqual(run_ep_jax.xprof_points("ends", decode), [1, 512])
        self.assertEqual(run_ep_jax.xprof_points("ends", prefill), [1024, 8192])
        self.assertEqual(run_ep_jax.xprof_points("all", prefill), prefill)
        self.assertEqual(run_ep_jax.xprof_points("1,64,512", decode), [1, 64, 512])
        # An explicit list is filtered to the ladder, never inventing a point.
        self.assertEqual(run_ep_jax.xprof_points("64,99999", decode), [64])
        self.assertEqual(run_ep_jax.xprof_points("ends", []), [])
        # A single-point ladder must not trace it twice.
        self.assertEqual(run_ep_jax.xprof_points("ends", [8]), [8])

    def test_a_miss_reports_what_the_trace_actually_holds(self) -> None:
        """A coverage failure must carry evidence, not just the absence of a marker.

        Two coverage failures have now been diagnosed by hypothesis and fixed wrongly.
        The diagnostic distinguishes the cases that matter: scope never recorded, scope
        present under a different label, or a trace with no device rows at all.
        """
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                self._event("ragged-all-to-all", pid=0, ts=1, device_ps=5_000_000,
                            tf_op="jit(f)/collectivex-dispatch-t16/ragged-all-to-all"),
                self._event("unrelated", pid=1, ts=2, dur=3.0),
            ])
            parsed = xprof.parse_trace_durations(
                directory, ep_jax.scope_name("dispatch", 512), occurrences=1,
            )
        self.assertIn("error", parsed)
        probe = parsed["diagnostic"]
        self.assertEqual(probe["events_with_duration"], 2)
        self.assertEqual(probe["devices_seen"], 2)
        # The scope that IS present is named, which is what tells them apart.
        self.assertEqual(probe["collectivex_scopes_present"], ["collectivex-dispatch-t16"])
        # When scope metadata is missing, the HLO's own name may still be matchable --
        # observed on hardware: 400k timed events, zero CollectiveX scopes.
        self.assertEqual(probe["all_to_all_events"], 1)
        self.assertTrue(probe["top_event_names"])
        self.assertTrue(probe["trace_files"])

    def test_unaligned_per_hlo_series_is_suppressed_not_published(self) -> None:
        """Devices can log different event counts for the same HLO.

        Observed: 480 matched events over 16 devices with 160 dropped -- some devices
        emit one event per iteration, others two. Positional pairing then compares
        different iterations across devices, so the distribution is meaningless. The
        summed totals do not depend on the pairing and survive.
        """
        marker = ep_jax.scope_name("dispatch", 512)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                # device 0 logs one event per iteration, device 1 logs two
                self._event("ragged-all-to-all", pid=0, ts=1, device_ps=100_000_000,
                            tf_op=marker),
                self._event("ragged-all-to-all", pid=0, ts=3, device_ps=100_000_000,
                            tf_op=marker),
                self._event("ragged-all-to-all", pid=1, ts=1, device_ps=100_000_000,
                            tf_op=marker),
                self._event("ragged-all-to-all", pid=1, ts=2, device_ps=100_000_000,
                            tf_op=marker),
                self._event("ragged-all-to-all", pid=1, ts=3, device_ps=100_000_000,
                            tf_op=marker),
                self._event("ragged-all-to-all", pid=1, ts=4, device_ps=100_000_000,
                            tf_op=marker),
            ])
            parsed = xprof.parse_trace_durations(
                directory, marker, occurrences=2,
                hlo_substrings=("ragged-all-to-all",),
            )
        entry = parsed["by_hlo"]["ragged-all-to-all"]
        self.assertGreater(entry["dropped_occurrences"], 0)
        self.assertNotIn("per_occurrence_us", entry)
        self.assertIn("per_occurrence_unreliable", entry)
        # The pairing-independent totals survive: device 1 summed 400us over 2 iterations.
        self.assertEqual(entry["total_device_us"], 400.0)
        self.assertEqual(entry["per_iteration_us"], 200.0)

    def test_span_brackets_the_region_including_idle_gaps(self) -> None:
        """The span is what a CUDA event pair measures, and why it is the GPU-comparable
        number: it includes idle time between ops inside the region, which summing op
        durations silently drops.
        """
        marker = ep_jax.scope_name("dispatch", 512)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                # one occurrence: op at 100-200, 300us gap, op at 500-600 -> span 500
                self._event("a", pid=0, ts=100, dur=100.0, tf_op=marker),
                self._event("b", pid=0, ts=500, dur=100.0, tf_op=marker),
            ])
            parsed = xprof.parse_trace_durations(directory, marker, occurrences=1)
        self.assertEqual(parsed["per_occurrence_us"], [500.0])
        # Summed op time misses the gap entirely -- 200us against a 500us region.
        self.assertEqual(parsed["op_total_device_us"], 200.0)

    def test_span_takes_the_slowest_device_per_occurrence(self) -> None:
        marker = ep_jax.scope_name("roundtrip", 64)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                self._event("a", pid=0, ts=0, dur=10.0, tf_op=marker),
                self._event("a", pid=0, ts=100, dur=10.0, tf_op=marker),
                self._event("a", pid=1, ts=0, dur=50.0, tf_op=marker),   # slower
                self._event("a", pid=1, ts=100, dur=10.0, tf_op=marker),
            ])
            parsed = xprof.parse_trace_durations(directory, marker, occurrences=2)
        self.assertEqual(parsed["per_occurrence_us"], [50.0, 10.0])
        self.assertEqual(parsed["span_devices"], 2)

    def test_spans_split_into_one_chunk_per_occurrence(self) -> None:
        marker = ep_jax.scope_name("combine", 8)
        with tempfile.TemporaryDirectory() as directory:
            self._write_trace(directory, [
                # 3 occurrences x 2 ops; each occurrence spans 20us
                self._event("x", pid=0, ts=base, dur=10.0, tf_op=marker)
                for base in (0, 10, 1000, 1010, 2000, 2010)
            ])
            parsed = xprof.parse_trace_durations(directory, marker, occurrences=3)
        self.assertEqual(parsed["per_occurrence_us"], [20.0, 20.0, 20.0])
        self.assertEqual(parsed["percentiles_us"]["p50"], 20.0)
        self.assertEqual(parsed["uneven_devices"], 0)

    def test_scope_names_separate_ladder_points(self) -> None:
        # One trace covers the whole ladder, so the label must disambiguate T.
        self.assertNotEqual(ep_jax.scope_name("dispatch", 1),
                            ep_jax.scope_name("dispatch", 512))
        self.assertTrue(ep_jax.scope_name("dispatch", 512).endswith("-t512"))

    def test_summarize_matches_the_artifact_percentile_shape(self) -> None:
        self.assertIsNone(xprof.summarize([]))
        percentiles = xprof.summarize([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(sorted(percentiles), ["p50", "p90", "p95", "p99"])
        self.assertEqual(percentiles["p50"], ep_harness.percentile([10, 20, 30, 40], 50))


@contextlib.contextmanager
def _recording_scope(sink, name):
    sink.append(name)
    yield


def _stub_jax():
    """A jax/jnp stand-in recording nothing: enough to build programs, not run them."""

    class Array:
        dtype = "bfloat16"

        def __getitem__(self, _index):
            return self

        def __call__(self, *a, **k):
            return self

        def astype(self, *_a, **_k):
            return self

        def reshape(self, *_a, **_k):
            return self

    array = Array()

    def anything(*_a, **_k):
        return array

    numpy = types.SimpleNamespace(zeros=anything, take=anything, arange=anything,
                                  asarray=anything, searchsorted=anything, pad=anything,
                                  bfloat16="bfloat16", float32="float32")
    lax = types.SimpleNamespace(ragged_all_to_all=anything, all_to_all=anything,
                                fori_loop=anything, optimization_barrier=anything)
    sharding = types.SimpleNamespace(
        PartitionSpec=lambda *a, **k: object(),
        NamedSharding=lambda *a, **k: object(),
    )
    return types.SimpleNamespace(
        numpy=numpy, lax=lax, sharding=sharding,
        # Single process unless a test says otherwise: EP8 must stay exactly as measured.
        process_count=lambda: 1, process_index=lambda: 0,
        distributed=types.SimpleNamespace(initialize=lambda **k: None),
        make_array_from_process_local_data=lambda sharding, data: data,
        make_array_from_callback=lambda shape, sharding, fn: fn(
            tuple(slice(None) for _ in shape)),
        jit=lambda fn, **k: fn, device_put=anything,
        block_until_ready=lambda value: value,
        named_scope=lambda _name: contextlib.nullcontext(),
        array=lambda: array,
    )


def _tpu_argv() -> list:
    """A minimal valid TPU case argv for parser-level tests."""
    return [
        "--backend", "jax-ragged-a2a", "--mode", "normal", "--precision", "bf16",
        "--phase", "decode", "--tokens-ladder", "1 2 4", "--hidden", "7168",
        "--topk", "8", "--experts", "256", "--routing", "uniform",
        # A real case ID: main() recomputes it from the realized factors and refuses to
        # publish under an identity that does not match.
        "--case-id", ep_harness.case_id("tpuv7", {
            "backend": "jax-ragged-a2a", "workload": "deepseek-v3", "mode": "normal",
            "phase": "decode", "ep": 8, "routing": "uniform", "precision": "bf16",
        }),
        "--suite", "ep-core", "--workload-name", "deepseek-v3",
        "--seed", "67", "--version", "1", "--warmup", "32", "--iters", "8",
        "--trials", "256", "--runner", "tpuv7", "--topology-class", "tpuv7-ici-island",
        "--transport", "ici", "--scope", "scale-up", "--scale-up-transport", "ici",
        "--scale-out-transport", "", "--gpus-per-node", "8", "--scale-up-domain", "8",
        "--out", "results/x.json",
    ]


def _jax_backends() -> tuple:
    """The --backend choices bench/run_ep_jax.py declares, read from its parser."""
    for action in _jax_parser()._actions:
        if action.dest == "backend":
            return tuple(action.choices)
    raise AssertionError("run_ep_jax.py declares no --backend")


def _jax_parser() -> argparse.ArgumentParser:
    # The REAL parser, not a mirror: a hand-kept copy silently stops covering flags added
    # after it was written (it did exactly that when --in-program-iters landed).
    return run_ep_jax.build_parser()


if __name__ == "__main__":
    unittest.main()


class RoundtripContainsDispatchTests(unittest.TestCase):
    """The run-level check that the chained roundtrip really ran its dispatch.

    Offline tests can assert the PROGRAM returns the dispatch results; only a trace can
    show XLA kept them. Run 30890386745 published 14 fp8 rows whose roundtrip was a lone
    combine -- green, correct=True, and 8.5ms too fast at T=8192 -- because
    `optimization_barrier` constrains ordering without making an unused output live.

    The numbers below are that run's real op-inventory totals.
    """

    @staticmethod
    def _inv(total):
        return {"op_inventory": [{"name": "op", "per_iteration_us": total}]}

    def test_a_genuine_chain_passes(self) -> None:
        # bf16 T=8192, run 30890386745: dispatch 11837.9, combine 14782.5, rt 22951.9
        self.assertTrue(run_ep_jax.roundtrip_contains_dispatch(
            self._inv(22951.9), self._inv(14782.5), self._inv(11837.9)))

    def test_a_deleted_dispatch_fails(self) -> None:
        # fp8 T=8192, same run: the roundtrip is combine to within 0.1us.
        self.assertFalse(run_ep_jax.roundtrip_contains_dispatch(
            self._inv(14785.9), self._inv(14785.8), self._inv(6885.6)))

    def test_every_point_of_the_real_broken_run_is_classified(self) -> None:
        """The full ladder, both precisions, from run 30890386745."""
        genuine = [(12.8, 14.6, 24.5), (33.7, 36.5, 58.9), (168.7, 240.2, 351.3),
                   (1410.0, 1789.2, 2775.5), (5945.9, 7051.9, 11345.1),
                   (11837.9, 14782.5, 22951.9)]
        deleted = [(12.9, 14.3, 14.3), (14.5, 20.1, 19.8), (22.8, 36.6, 36.3),
                   (103.2, 239.7, 240.0), (721.1, 1790.2, 1790.0),
                   (2806.2, 7053.3, 7052.6), (6885.6, 14785.8, 14785.9)]
        for d, c, rt in genuine:
            self.assertTrue(run_ep_jax.roundtrip_contains_dispatch(
                self._inv(rt), self._inv(c), self._inv(d)), f"d={d} c={c} rt={rt}")
        for d, c, rt in deleted:
            self.assertFalse(run_ep_jax.roundtrip_contains_dispatch(
                self._inv(rt), self._inv(c), self._inv(d)), f"d={d} c={c} rt={rt}")

    def test_op_names_are_not_what_decides_it(self) -> None:
        """An earlier version compared op NAME sets and was wrong on this exact case.

        XLA's numeric suffixes differ between compilations, so a set-difference is never
        empty. Identical names with a genuine cost gap must still pass; differing names with
        no cost gap must still fail.
        """
        self.assertTrue(run_ep_jax.roundtrip_contains_dispatch(
            {"op_inventory": [{"name": "ragged_all_to_all.6", "per_iteration_us": 22951.9}]},
            {"op_inventory": [{"name": "ragged_all_to_all.6", "per_iteration_us": 14782.5}]},
            self._inv(11837.9)))
        self.assertFalse(run_ep_jax.roundtrip_contains_dispatch(
            {"op_inventory": [{"name": "ragged_all_to_all.8", "per_iteration_us": 14785.9}]},
            {"op_inventory": [{"name": "ragged_all_to_all.6", "per_iteration_us": 14785.8}]},
            self._inv(6885.6)))

    def test_a_missing_inventory_is_unjudgeable_rather_than_a_failure(self) -> None:
        for args in (({}, self._inv(1.0), self._inv(1.0)),
                     (self._inv(1.0), {}, self._inv(1.0)),
                     (self._inv(1.0), self._inv(1.0), {}),
                     (None, None, None)):
            self.assertIsNone(run_ep_jax.roundtrip_contains_dispatch(*args))


class RowVerdictTests(unittest.TestCase):
    """A row whose roundtrip lost its dispatch must not publish as passed.

    Run 30890386745 published 14 such fp8 rows as correct=True: the transport WAS correct,
    dispatch and combine both verified, and the headline roundtrip measured a lone combine.
    Correctness and "the row measured what it says" are separate questions.
    """

    OK = dict(dispatch_ok=True, post_ok=True, max_rel=0.0)

    def test_a_clean_row_with_an_intact_roundtrip_passes(self) -> None:
        self.assertTrue(run_ep_jax.row_passed(**self.OK, roundtrip_intact=True))

    def test_a_deleted_dispatch_fails_an_otherwise_perfect_row(self) -> None:
        self.assertFalse(run_ep_jax.row_passed(**self.OK, roundtrip_intact=False))

    def test_no_trace_to_judge_does_not_fail_the_row(self) -> None:
        """Absence of evidence is not failure: host-timed rows have no inventory."""
        self.assertTrue(run_ep_jax.row_passed(**self.OK, roundtrip_intact=None))

    def test_correctness_failures_still_fail_regardless(self) -> None:
        for override in ({"dispatch_ok": False}, {"post_ok": False},
                         {"max_rel": ep_harness.COMBINE_REL_TOL * 2},
                         {"max_rel": float("inf")}):
            self.assertFalse(run_ep_jax.row_passed(
                **{**self.OK, **override}, roundtrip_intact=True), override)


class MultiHostEpTests(unittest.TestCase):
    """EP16 spans two 8-device hosts inside ONE ICI slice.

    The trap is silence: without joining one JAX cluster, each host builds an EP8 mesh from
    its own `local_devices()` and the run measures two independent 8-way exchanges while
    labelling them EP16. Nothing in the numbers would look wrong.
    """

    def test_single_process_placement_is_unchanged(self) -> None:
        """EP8 must be byte-identical to what was measured; it takes the device_put path."""
        stub = _stub_jax()
        placed = []
        stub.device_put = lambda array, sharding: placed.append(("device_put", sharding)) \
            or array
        stub.make_array_from_process_local_data = lambda *a: self.fail(
            "single-process EP8 must not take the multi-process placement path")
        transport = ep_jax.JaxEPTransport.__new__(ep_jax.JaxEPTransport)
        transport.jax, transport.jnp = stub, stub.numpy
        transport.mesh, transport.axis = object(), "ep"
        transport._P = stub.sharding.PartitionSpec
        transport._shard(np.zeros((8, 4), np.float32))
        self.assertEqual(len(placed), 1)

    def test_multi_process_places_by_addressable_shard(self) -> None:
        """`device_put` of a global array is not the multi-process API, and neither is
        `make_array_from_process_local_data`.

        That second one is the trap, because it looks right. It takes the process's OWN
        slice, so handing it the global array from every process concatenates them. Measured
        on the 2x2x2 slice: a 16-row global array became 32 rows and the collective rejected
        the operands with `input_offsets must be rank 1 ... but got shape (2, 16)` -- two
        rows per device where there should be one.

        `make_array_from_callback` asks for each addressable shard BY INDEX, which is
        answerable from the global array every rank already holds.
        """
        stub = _stub_jax()
        stub.process_count = lambda: 2
        asked = []

        def callback_api(shape, sharding, fn):
            asked.append(shape)
            # Two devices' worth of index, to prove the callback indexes rather than
            # returning the whole thing.
            return np.stack([fn((slice(0, 1),)), fn((slice(1, 2),))])

        stub.make_array_from_callback = callback_api
        stub.make_array_from_process_local_data = lambda *a, **k: self.fail(
            "process_local_data concatenates the global array across processes")
        stub.device_put = lambda *a, **k: self.fail(
            "multi-process placement must not go through device_put")
        transport = ep_jax.JaxEPTransport.__new__(ep_jax.JaxEPTransport)
        transport.jax, transport.jnp = stub, stub.numpy
        transport.mesh, transport.axis = object(), "ep"
        transport._P = stub.sharding.PartitionSpec
        source = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)
        placed = transport._shard(source)
        self.assertEqual(asked, [(16, 4)], "the GLOBAL shape must be declared")
        # Row 0 and row 1 of the source, fetched by index -- not the whole array twice.
        np.testing.assert_array_equal(placed[0][0], source[0])
        np.testing.assert_array_equal(placed[1][0], source[1])

    def test_to_host_gathers_only_when_multi_process(self) -> None:
        run_ep_jax._PROCESS_COUNT[0] = 1
        try:
            plain = np.arange(6, dtype=np.float32).reshape(3, 2)
            np.testing.assert_array_equal(run_ep_jax.to_host(plain), plain)
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1

    def test_only_process_zero_writes_the_artifact(self) -> None:
        """Two hosts writing one path would race and the harvester reads the loser."""
        run_ep_jax._PROCESS_COUNT[0] = 2
        try:
            for index, expected in ((0, True), (1, False)):
                stub = types.SimpleNamespace(process_index=lambda index=index: index)
                self.assertEqual(run_ep_jax.writes_artifact(stub), expected)
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1
        # Single process always writes, whatever the index says.
        self.assertTrue(run_ep_jax.writes_artifact(
            types.SimpleNamespace(process_index=lambda: 3)))

    def test_the_matrix_and_the_launcher_agree_on_host_count(self) -> None:
        sys.path.insert(0, str(COLLX))
        import sweep_matrix  # noqa: PLC0415

        document = sweep_matrix.resolve_matrix(backend="all", only_sku="tpuv7")
        by_ep = {item["case"]["ep"]: item["case"] for item in document["requested_cases"]
                 if item["disposition"] == "runnable"}
        self.assertEqual(by_ep[8]["nodes"], 1)
        self.assertEqual(by_ep[16]["nodes"], 2)
        # gpus_per_node counts JAX DEVICES (8), not chips (4) -- a tpu7x chip surfaces as
        # two devices, so EP16 is 8 chips across two hosts.
        self.assertEqual(by_ep[16]["gpus_per_node"], 8)

    def test_a_multi_host_run_refuses_the_single_host_topology(self) -> None:
        launcher = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertIn("2x2x1", launcher)
        self.assertIn("COLLX_TPU_TOPOLOGY", launcher)


class ScaleUpDomainTests(unittest.TestCase):
    """`scale_up_domain: "ep"` must apply ONLY where the domain really is the slice.

    Applying it everywhere would relabel every GPU EP16 row as scale-up, erasing the RDMA
    hop those SKUs actually take -- the single most load-bearing distinction in the corpus.
    """

    @staticmethod
    def _platforms():
        sys.path.insert(0, str(COLLX))
        import sweep_matrix  # noqa: PLC0415
        return sweep_matrix, json.loads(
            (COLLX / "configs" / "platform_config.json").read_text())["platforms"]

    def test_an_eight_wide_nvlink_domain_still_goes_scale_out_at_ep16(self) -> None:
        matrix, platforms = self._platforms()
        for sku in ("b200-nscale", "b300", "h100-dgxc", "h200-dgxc", "mi355x"):
            with self.subTest(sku=sku):
                topology = matrix._topology(platforms[sku], 16)
                self.assertEqual(topology["scope"], "scale-out")
                self.assertEqual(topology["scale_up_domain"], 8)
                self.assertIsNotNone(topology["scale_out_transport"])

    def test_a_wide_mnnvl_domain_stays_scale_up_at_ep16(self) -> None:
        matrix, platforms = self._platforms()
        for sku in ("gb200", "gb300"):
            with self.subTest(sku=sku):
                topology = matrix._topology(platforms[sku], 16)
                self.assertEqual(topology["scope"], "scale-up")
                self.assertEqual(topology["scale_up_domain"], 72)

    def test_tpu_follows_the_ep_degree_at_both_sizes(self) -> None:
        matrix, platforms = self._platforms()
        for ep in (8, 16):
            topology = matrix._topology(platforms["tpuv7"], ep)
            self.assertEqual(topology["scale_up_domain"], ep)
            self.assertEqual(topology["scope"], "scale-up")
            self.assertEqual(topology["transport"], "ici")
            self.assertIsNone(topology["scale_out_transport"])

    def test_only_tpu_declares_the_ep_domain(self) -> None:
        _, platforms = self._platforms()
        following = {sku for sku, entry in platforms.items()
                     if entry["scale_up_domain"] == "ep"}
        self.assertEqual(following, {"tpuv7"})


class ArtifactWriterSelectionTests(unittest.TestCase):
    """Exactly one process writes, and it is the pod the harvester reads.

    Measured on the real 2x2x2 slice: `jax.process_index()` does NOT track the pod index --
    pod `ep16-probe-0` (TPU_WORKER_ID=0) reported process_index 1, and pod `ep16-probe-1`
    reported 0. Gating the write on JAX's order still yields one writer, but a
    nondeterministic one, while the launcher harvests a specific pod's log.
    """

    @staticmethod
    def _jax(process_index: int):
        return types.SimpleNamespace(process_index=lambda: process_index)

    def test_the_pod_index_decides_not_the_jax_index(self) -> None:
        run_ep_jax._PROCESS_COUNT[0] = 2
        try:
            # The real observation: pod 0 carries jax index 1. Pod 0 must still write.
            with mock.patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "0"}, clear=False):
                self.assertTrue(run_ep_jax.writes_artifact(self._jax(1)))
            with mock.patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "1"}, clear=False):
                self.assertFalse(run_ep_jax.writes_artifact(self._jax(0)))
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1

    def test_exactly_one_pod_of_a_slice_writes(self) -> None:
        run_ep_jax._PROCESS_COUNT[0] = 2
        try:
            writers = 0
            for pod in ("0", "1"):
                with mock.patch.dict(os.environ, {"JOB_COMPLETION_INDEX": pod},
                                     clear=False):
                    writers += run_ep_jax.writes_artifact(self._jax(0))
            self.assertEqual(writers, 1)
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1

    def test_gke_tpu_worker_id_is_the_fallback(self) -> None:
        run_ep_jax._PROCESS_COUNT[0] = 2
        try:
            env = {k: v for k, v in os.environ.items() if k != "JOB_COMPLETION_INDEX"}
            with mock.patch.dict(os.environ, env, clear=True):
                os.environ["TPU_WORKER_ID"] = "0"
                self.assertTrue(run_ep_jax.writes_artifact(self._jax(1)))
                os.environ["TPU_WORKER_ID"] = "1"
                self.assertFalse(run_ep_jax.writes_artifact(self._jax(0)))
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1

    def test_a_single_process_run_always_writes(self) -> None:
        env = {k: v for k, v in os.environ.items()
               if k not in ("JOB_COMPLETION_INDEX", "TPU_WORKER_ID")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(run_ep_jax.writes_artifact(self._jax(3)))

    def test_the_launcher_harvests_the_writing_pod(self) -> None:
        launcher = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertIn("job-completion-index=0", launcher,
                      "multi-host harvest must name the pod that writes")


class TopologyPerHostCountTests(unittest.TestCase):
    """One coordinator runs BOTH the EP8 shard (2x2x1) and the EP16 shard (2x2x2).

    A single flat COLLX_TPU_TOPOLOGY cannot serve both: set it to 2x2x2 and the EP8 shard
    asks for a slice its nodes do not have; leave it at 2x2x1 and EP16 is refused. So the
    launcher keys the topology on the shard's HOST COUNT.
    """

    @staticmethod
    def _launcher_lines() -> str:
        """The launcher's OWN topology lines, not a copy of them.

        A test that reimplements the selection cannot catch the launcher drifting away from
        it -- and the first version of this test did exactly that, reproducing the logic with
        the eval escaping wrong so it silently selected the default every time.
        """
        text = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        lines = [line for line in text.splitlines()
                 if line.startswith("eval \"TOPOLOGY_FOR_NODES=")
                 or line.startswith("TOPOLOGY=\"${TOPOLOGY_FOR_NODES")]
        assert len(lines) == 2, f"expected 2 topology lines, found {lines}"
        return "\n".join(lines)

    def _select(self, nodes, **env):
        script = f"NODES={nodes}\n{self._launcher_lines()}\necho \"$TOPOLOGY\""
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True,
            env={**{k: v for k, v in os.environ.items()
                    if not k.startswith("COLLX_TPU_TOPOLOGY")},
                 **{k: v for k, v in env.items() if v is not None}})
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_single_host_keeps_the_default_even_when_a_multi_host_entry_exists(self) -> None:
        self.assertEqual(self._select(1, COLLX_TPU_TOPOLOGY_N2="2x2x2"), "2x2x1")

    def test_two_hosts_pick_the_two_host_entry(self) -> None:
        self.assertEqual(self._select(2, COLLX_TPU_TOPOLOGY_N2="2x2x2"), "2x2x2")

    def test_the_per_host_entry_beats_the_flat_one(self) -> None:
        self.assertEqual(
            self._select(2, COLLX_TPU_TOPOLOGY_N2="2x2x2",
                         COLLX_TPU_TOPOLOGY="4x4x4"), "2x2x2")

    def test_the_launcher_uses_the_per_host_variable(self) -> None:
        launcher = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertIn("COLLX_TPU_TOPOLOGY_N", launcher)
        # And must not re-assign TOPOLOGY flatly afterwards, which would clobber it.
        self.assertNotIn('TOPOLOGY="${COLLX_TPU_TOPOLOGY:-2x2x1}"', launcher)


class InPodScriptTests(unittest.TestCase):
    """The in-pod script is a heredoc STRING; `bash -n` on the launcher never sees it.

    That gap shipped a real break: a comment placed between a `\\` continuation and the line
    it continued, which ends the command -- `timeout` would have run with no arguments and
    Python would never have started. The launcher still passed `bash -n`, because the outer
    file is syntactically fine either way.
    """

    @staticmethod
    def _pod_script() -> str:
        text = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        start = text.index("          args:\n            - |\n") + len(
            "          args:\n            - |\n")
        end = text.index("          env:\n", start)
        body = text[start:end]
        lines = [line[14:] if line.startswith(" " * 14) else line
                 for line in body.splitlines()]
        script = "\n".join(lines)
        # The heredoc is expanded by the coordinator's shell: `\$x` reaches the pod as `$x`
        # and `\\` as `\`. Undo that so what is checked is what the pod actually runs.
        return script.replace("\\$", "$").replace("\\\\", "\\")

    def test_the_pod_script_is_valid_bash(self) -> None:
        script = self._pod_script()
        self.assertIn("run_ep_jax_shard.py", script)
        result = subprocess.run(["bash", "-n"], input=script,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"in-pod script is not valid bash:\n{result.stderr}")

    def test_no_comment_follows_a_line_continuation(self) -> None:
        """A comment after `\\` silently truncates the command it was continuing."""
        lines = self._pod_script().splitlines()
        for index, line in enumerate(lines[:-1]):
            if line.rstrip().endswith("\\"):
                nxt = lines[index + 1].strip()
                self.assertFalse(
                    nxt.startswith("#"),
                    f"line {index + 1} continues into a comment, which ends the command:\n"
                    f"  {line}\n  {lines[index + 1]}")

    def test_python_runs_unbuffered(self) -> None:
        """Buffered stdout plus a SIGKILL loses every diagnostic the shard produced."""
        script = self._pod_script()
        self.assertIn("PYTHONUNBUFFERED=1", script)
        self.assertIn("python3 -u bench/run_ep_jax_shard.py", script)


class HeredocExpansionTests(unittest.TestCase):
    """The Job manifest is an UNQUOTED heredoc, so the coordinator expands its body.

    A single stray backtick pair in a comment aborted the whole manifest with
    "unexpected EOF while looking for matching" -> "no objects passed to apply" ->
    "FATAL: cannot create the bench Job", and it took down EVERY shard including the EP8
    ones that had been green. Neither `bash -n` on the launcher nor the in-pod `bash -n`
    check sees it: the first does not evaluate heredoc bodies, and the second checks the
    body as a standalone script rather than as text about to be expanded.
    """

    @staticmethod
    def _heredoc_body() -> list:
        lines = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text().splitlines()
        start = next(i for i, line in enumerate(lines)
                     if line.startswith("cat <<EOF | $KUBECTL"))
        end = next(i for i, line in enumerate(lines[start:], start) if line == "EOF")
        return [(i + 1, lines[i]) for i in range(start + 1, end)]

    def test_no_backticks_in_the_manifest_heredoc(self) -> None:
        offenders = [(n, line) for n, line in self._heredoc_body() if "`" in line]
        self.assertEqual(offenders, [],
                         "backticks in an unquoted heredoc are command substitution and "
                         "abort the manifest")

    def test_every_dollar_in_the_heredoc_is_deliberate(self) -> None:
        """An unescaped $VAR is expanded by the COORDINATOR, not the pod.

        Both are legitimate -- the manifest interpolates ${JOB}, ${NODES} and friends on
        purpose -- so this pins the in-pod ones: anything the POD must evaluate has to be
        written \\$, and the shard's own runtime variables are the ones that matter.
        """
        pod_runtime = ("shard_rc", "rc", "index", "arg_index", "ep_args", "out_path",
                       "argv_file", "produced", "failure_reason", "shard_timeout")
        for number, line in self._heredoc_body():
            for name in pod_runtime:
                self.assertNotIn(
                    f"${name}", line.replace(f"\\${name}", ""),
                    f"line {number}: ${name} would be expanded by the coordinator, not "
                    f"the pod; write \\${name}")


class MultiHostProgressTests(unittest.TestCase):
    """The multi-host progress markers are diagnostics, and they are load-bearing.

    Two EP16 runs were SIGKILLed having reported only the setup banner, so the stalling
    stage was unknowable. These markers are what localized it to the first ladder point --
    and what will localize the next one. A single-host run reaches its first per-point line
    in seconds and needs none of them, so they must stay off there.
    """

    def _emitted(self, nodes):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_ep_jax._trace(nodes, "T=8: compiling")
        return buffer.getvalue()

    def test_multi_host_emits_the_marker(self) -> None:
        self.assertIn("T=8: compiling", self._emitted(2))

    def test_single_host_stays_quiet(self) -> None:
        self.assertEqual(self._emitted(1), "")

    def test_every_stage_of_the_first_ladder_point_is_marked(self) -> None:
        """Placement, compile+run, and the oracle are the three candidate stalls."""
        source = Path(run_ep_jax.__file__).read_text()
        for stage in ("placing operands", "compiling and running roundtrip",
                      "roundtrip ok; running the oracle", "oracle done"):
            self.assertIn(stage, source, f"no marker for {stage!r}")


class ClusterJoinOncePerProcessTests(unittest.TestCase):
    """One shard runs EVERY case in ONE process, so the cluster is joined once.

    `jax.distributed.initialize()` raises on a second call -- "must be called before any JAX
    calls that might initialise the XLA backend". EP16's decode case passed all 10 ladder
    points and its prefill case died on exactly that, so the shard reported 1/2 cases failed
    on a run whose measurement was sound.
    """

    def setUp(self) -> None:
        self._saved = run_ep_jax._DISTRIBUTED_READY[0]
        run_ep_jax._DISTRIBUTED_READY[0] = False

    def tearDown(self) -> None:
        run_ep_jax._DISTRIBUTED_READY[0] = self._saved

    def test_a_second_case_does_not_rejoin(self) -> None:
        calls = []
        stub = _stub_jax()
        stub.process_count = lambda: 2
        stub.distributed = types.SimpleNamespace(
            initialize=lambda **kwargs: calls.append(kwargs))
        # First case joins.
        self.assertFalse(run_ep_jax._DISTRIBUTED_READY[0])
        stub.distributed.initialize()
        run_ep_jax._DISTRIBUTED_READY[0] = True
        # Second case must NOT call initialize again.
        self.assertTrue(run_ep_jax._DISTRIBUTED_READY[0])
        self.assertEqual(len(calls), 1)

    def test_the_guard_is_read_before_initialising(self) -> None:
        """Source-level, because the failure is an exception ordering, not a value."""
        source = Path(run_ep_jax.__file__).read_text()
        guard = source.index("if nodes > 1 and _DISTRIBUTED_READY[0]:")
        # The indented CALL, not the many prose mentions of it in comments -- the first
        # version of this test matched a comment and compared the wrong positions.
        joins = source.index("\n            jax.distributed.initialize(")
        self.assertLess(guard, joins,
                        "the already-joined check must precede the join")
        self.assertIn("_DISTRIBUTED_READY[0] = True", source,
                      "nothing records that the process joined")

    def test_single_host_never_touches_the_flag(self) -> None:
        source = Path(run_ep_jax.__file__).read_text()
        # The join and its bookkeeping both live under a nodes > 1 branch.
        self.assertIn("elif nodes > 1:", source)


class MultiHostShardMergeTests(unittest.TestCase):
    """A multi-host TPU slice is ONE indivisible allocation, so it gets ONE shard.

    Run 30942031527 ran the fp8 and bf16 EP16 shards concurrently. Four pods contended for
    the two pod slots a 2x2x2 pool has, Kubernetes gave each Job one of its two pods, and
    both half-formed slices aborted with SLICE_FAILURE_SW_INJECT_ERROR. Merging removes the
    contention instead of sequencing it, and holds for a manual dispatch too.
    """

    @staticmethod
    def _matrix():
        sys.path.insert(0, str(COLLX))
        import sweep_matrix  # noqa: PLC0415
        return sweep_matrix.resolve_matrix(backend="all", only_sku="tpuv7")

    def test_one_shard_covers_the_whole_multi_host_slice(self) -> None:
        shards = [s for s in self._matrix()["include"] if s["nodes"] > 1]
        self.assertEqual(len(shards), 1, "a multi-host slice must not be split across Jobs")
        self.assertEqual(
            sorted({case["precision"] for case in shards[0]["cases"]}), ["bf16", "fp8"])

    def test_single_host_shards_still_split_by_precision(self) -> None:
        """EP8 has six independent nodes; serialising it would halve throughput for nothing."""
        shards = [s for s in self._matrix()["include"] if s["nodes"] == 1]
        self.assertEqual(len(shards), 2)
        for shard in shards:
            self.assertEqual(
                len({case["precision"] for case in shard["cases"]}), 1,
                "single-host shards are per-precision")

    def test_every_case_still_appears_exactly_once(self) -> None:
        """Merging must not drop or duplicate coverage."""
        document = self._matrix()
        runnable = [item["case"]["case_id"] for item in document["requested_cases"]
                    if item["disposition"] == "runnable"]
        sharded = [case["case_id"] for shard in document["include"]
                   for case in shard["cases"]]
        self.assertEqual(sorted(sharded), sorted(runnable))
        self.assertEqual(len(set(sharded)), len(sharded), "a case is scheduled twice")

    def test_only_the_tpu_launcher_merges(self) -> None:
        """GPU EP16 spans nodes over RDMA, where shards are independent allocations."""
        sys.path.insert(0, str(COLLX))
        import sweep_matrix  # noqa: PLC0415
        for sku in ("b200-nscale", "h100-dgxc", "mi355x"):
            with self.subTest(sku=sku):
                self.assertEqual(
                    sweep_matrix._shard_precision_key(sku, 2, "fp8"), "fp8")
        self.assertEqual(sweep_matrix._shard_precision_key("tpuv7", 2, "fp8"), "all")
        self.assertEqual(sweep_matrix._shard_precision_key("tpuv7", 1, "fp8"), "fp8")


class MultiHostFailureDiagnosticsTests(unittest.TestCase):
    """On a slice failure the cause is usually in a pod whose log is never harvested.

    libtpu reports SLICE_FAILURE_SW_INJECT_ERROR on the worker that NOTICED a peer die, so
    the pod that actually failed is a different one -- and only completion index 0 is
    harvested, because that is the pod that writes the artifact. Run 30944020533 aborted
    that way with the real error invisible.
    """

    @staticmethod
    def _launcher() -> str:
        return (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()

    def test_other_pods_are_dumped_on_multi_host_failure(self) -> None:
        text = self._launcher()
        self.assertTrue(_uncommented(text, "per-pod tails"))
        self.assertTrue(_uncommented(text, 'logs "$_pod" -c bench'),
                        "the dump must fetch each pod's log")

    def test_the_dump_cannot_skip_every_pod(self) -> None:
        """Its first version read a completion index that came back empty and skipped all.

        Nothing may filter the pod list on a value that can be absent -- a diagnostic that
        silently prints nothing is worse than none, because it reads as "no peer output".
        """
        text = self._launcher()
        self.assertFalse(_uncommented(text, '= 0 ] && continue'),
                         "a filter on a possibly-empty index skips every pod")

    def test_pod_termination_reasons_are_dumped(self) -> None:
        """A libtpu abort, an eviction and an OOMKill are identical in the log."""
        self.assertTrue(_uncommented(self._launcher(), "state.terminated.reason"))

    def test_the_dump_is_failure_only_and_covers_single_host(self) -> None:
        """Any failed shard, one host or two.

        An EP8 shard failed twice reporting only "the Job returned no result payload",
        which distinguishes none of: a crash, a scheduling refusal, an evicted pod.
        A green shard still pays nothing.
        """
        text = self._launcher()
        guard = 'if [ "$JOB_RC" != 0 ]; then'
        self.assertIn(guard, text)
        self.assertFalse(_uncommented(text, '"$NODES" != 1 ] && [ "$JOB_RC"'),
                         "the dump must not be restricted to multi-host")
        # And it must come after JOB_RC is known, or it can never fire.
        self.assertLess(text.index("JOB_RC=1"), text.index(guard))




class ShardTimeoutBudgetTests(unittest.TestCase):
    """The shard's own timeout must fit inside the coordinator's wait and the job timeout.

    A multi-host slice is ONE allocation, so all of its cases share ONE budget: the merged
    EP16 shard has four. At 900s/case that is 3600s, and the shard published its whole decode
    ladder and was SIGKILLed mid-prefill. Raising the per-case allowance is only safe while
    cases x allowance still fits the outer limits, which is what this pins.
    """

    @staticmethod
    def _per_case() -> int:
        text = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        match = re.search(r'RUN_TIMEOUT="\$\{COLLX_RUN_TIMEOUT:-(\d+)\}"', text)
        assert match, "cannot find the per-case timeout"
        return int(match.group(1))

    @staticmethod
    def _coordinator_wait_seconds() -> int:
        text = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        match = re.search(r'WAIT_MAX="\$\{COLLX_TPU_WAIT_ITERS:-(\d+)\}"', text)
        assert match, "cannot find the coordinator wait"
        return int(match.group(1)) * 15

    def test_the_multi_host_shard_budget_fits_the_coordinator_wait(self) -> None:
        sys.path.insert(0, str(COLLX))
        import sweep_matrix  # noqa: PLC0415
        shards = [s for s in sweep_matrix.resolve_matrix(
            backend="all", only_sku="tpuv7")["include"] if s["nodes"] > 1]
        self.assertTrue(shards)
        worst = max(len(shard["cases"]) for shard in shards) * self._per_case()
        self.assertLess(worst, self._coordinator_wait_seconds(),
                        "the shard can outlive the coordinator's wait, which would report a "
                        "timeout on a shard still doing useful work")

    def test_the_budget_fits_the_workflow_job_timeout(self) -> None:
        workflow = (COLLX.parent.parent / ".github" / "workflows"
                    / "collectivex-sweep.yml").read_text()
        minutes = int(re.search(r"timeout-minutes:\s*(\d+)", workflow).group(1))
        self.assertLess(self._coordinator_wait_seconds(), minutes * 60)

    def test_the_allowance_covers_the_measured_worst_case(self) -> None:
        """Measured: the 4-case EP16 shard ran 7252s, hitting a 4x1800=7200s ceiling with
        three cases banked. The ceiling is per SHARD, so the allowance must cover the worst
        case rather than the mean."""
        self.assertGreaterEqual(self._per_case(), 2700)


class LockstepTimingTests(unittest.TestCase):
    """Both hosts must call a slice-wide collective the SAME number of times.

    `reference_timing` extends its warmup and measurement loops until a wall-clock floor is
    met. `fn` drives a collective over every device in the slice, so a duration-driven count
    makes the iteration count depend on each host's own clock: one host ends up blocked in a
    collective the other never issues, and libtpu aborts with SLICE_FAILURE_SW_INJECT_ERROR.
    Flaky by nature -- it depends on whether the clocks happen to agree.
    """

    @staticmethod
    def _count_calls(**kwargs):
        calls = []
        slow = [0.0]

        def fn():
            calls.append(1)
            slow[0] += 0.05          # a slow host: floors would extend the loop
            return None

        jax = types.SimpleNamespace(block_until_ready=lambda value: value)
        with mock.patch.object(ep_jax.time, "perf_counter", lambda: slow[0]):
            ep_jax.reference_timing(jax, fn, **kwargs)
        return len(calls)

    def test_lockstep_iterates_a_fixed_number_of_times(self) -> None:
        """Same count regardless of how long each iteration took."""
        fast = self._count_calls(warmup_tries=3, num_runs=4, min_duration_s=10.0,
                                 lockstep=True)
        self.assertEqual(fast, 3 + 4)

    def test_without_lockstep_the_clock_extends_the_loop(self) -> None:
        """The behaviour the GPU/single-host path wants, and why it cannot be the default
        for a multi-host slice."""
        extended = self._count_calls(warmup_tries=3, num_runs=4, min_duration_s=10.0,
                                     lockstep=False)
        self.assertGreater(extended, 3 + 4)

    def test_both_multi_host_call_sites_pass_lockstep(self) -> None:
        source = Path(run_ep_jax.__file__).read_text()
        self.assertEqual(source.count("lockstep=_PROCESS_COUNT[0] > 1"), 2,
                         "every reference_timing call on the multi-host path must be "
                         "lockstep, or the hosts desynchronise")
        self.assertEqual(source.count("ep_jax.reference_timing("), 2,
                         "a new reference_timing call site needs a lockstep decision")


class JobCompletionWaitTests(unittest.TestCase):
    """The coordinator must recognise a finished MULTI-POD Job.

    A multi-host slice runs an Indexed Job with completions=$NODES, so .status.succeeded
    reaches 2. Breaking on `= 1` never matches: run 30960700207 polled for 277 minutes after
    its EP16 shard had finished, the TTL deleted the Job at 900s, and a completed sweep was
    reported as a timeout with its results gone.
    """

    @staticmethod
    def _launcher() -> str:
        return (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()

    def test_success_is_measured_against_the_host_count(self) -> None:
        text = self._launcher()
        self.assertTrue(_uncommented(text, '[ "${succeeded:-0}" = "$NODES" ] && break'))
        self.assertFalse(_uncommented(text, '[ "${succeeded:-0}" = 1 ]'),
                         "a fixed 1 never matches an Indexed Job with completions>1")

    def test_a_vanished_job_is_not_waited_out(self) -> None:
        """Otherwise a deleted Job costs the full WAIT_MAX before reporting anything."""
        self.assertTrue(_uncommented(self._launcher(), "no longer exists"))

    def test_single_host_still_breaks_on_one(self) -> None:
        """With no completions set a Job defaults to 1, which is also $NODES=1."""
        script = 'NODES=1\nsucceeded=1\n[ "${succeeded:-0}" = "$NODES" ] && echo BREAK'
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "BREAK")


class JobEventDiagnosticsTests(unittest.TestCase):
    """When the pod list is EMPTY, only events explain why.

    An EP8 shard failed printing a bare "per-pod tails" header with nothing under it: the pod
    either never existed or was already collected, and `get pods` cannot distinguish those
    from each other or from a scheduling refusal. FailedScheduling, FailedCreate, Evicted and
    preemption live in events only -- and this cluster is shared, so "another workload took
    the node" is a real answer that needs to be visible.
    """

    @staticmethod
    def _launcher() -> str:
        return (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()

    def test_job_conditions_are_dumped(self) -> None:
        self.assertTrue(_uncommented(self._launcher(), "status.conditions"))

    def test_events_are_dumped(self) -> None:
        text = self._launcher()
        self.assertTrue(_uncommented(text, "get events"))
        self.assertTrue(_uncommented(text, "involvedObject.name=$JOB"))

    def test_the_diagnostics_do_not_abort_the_launcher(self) -> None:
        """Every probe is best-effort: a missing Job must not turn a red shard into a crash
        before its reason is printed."""
        text = self._launcher()
        block = text[text.index('if [ "$JOB_RC" != 0 ]; then'):]
        block = block[:block.index("\nfi\n")]
        for line in block.splitlines():
            if line.strip().startswith("$KUBECTL"):
                self.assertTrue(
                    "|| true" in line or line.rstrip().endswith("\\"),
                    f"unguarded kubectl in the diagnostic block: {line.strip()}")


class JobRetryPolicyTests(unittest.TestCase):
    """Retry once on a single host; never on a multi-host slice.

    This cluster is shared. EP8 fp8 failed in 2 of 4 runs while EP8 bf16 passed 4 of 4 --
    exposure, not precision: the fp8 shard traces four components instead of three, runs
    longer, and meets more contention. A retry cannot mask a real failure, because a shard
    that RAN emits an artifact per case (red included, via the terminal-failure backfill),
    so a genuine red re-runs, goes red again, and the Job still ends failed.

    Multi-host must NOT retry: an Indexed Job retries the failed index alone, and its peer
    has already exited, so the replacement waits in a collective for a process that is gone.
    """

    @staticmethod
    def _launcher() -> str:
        return (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()

    def test_single_host_default_retries_once(self) -> None:
        text = self._launcher()
        self.assertTrue(_uncommented(text, "JOB_BACKOFF_LIMIT=1"))
        # The default must be set before the multi-host branch can override it.
        self.assertLess(text.index("JOB_BACKOFF_LIMIT=1"),
                        text.index("JOB_BACKOFF_LIMIT=0"))

    def test_multi_host_never_retries(self) -> None:
        text = self._launcher()
        branch = text[text.index('if [ "$NODES" != 1 ]; then', text.index("POD_SUBDOMAIN_SPEC")):]
        branch = branch[:branch.index("\nfi\n")]
        self.assertIn("JOB_BACKOFF_LIMIT=0", branch,
                      "a multi-host slice must not retry a single index")

    def test_the_manifest_uses_the_variable(self) -> None:
        self.assertTrue(_uncommented(self._launcher(),
                                     "backoffLimit: ${JOB_BACKOFF_LIMIT}"))
        self.assertFalse(_uncommented(self._launcher(), "backoffLimit: 0\n"),
                         "a hardcoded 0 would defeat the single-host retry")


class XprofIsMandatoryTests(unittest.TestCase):
    """Device tracing is unconditional, and the launcher must not claim otherwise.

    `--xprof` defaults True and nothing on the launcher path passes `--no-xprof`, so the old
    `COLLX_TPU_XPROF=1` gate only ever changed `--xprof-iters` while its comment said tracing
    was "off by default". Wiring the knob up would have been worse than leaving it dead:
    device spans ARE the published latency and a point without them fails the case, so
    disabling tracing turns every case red rather than saving a pass.
    """

    def test_the_launcher_always_passes_xprof(self) -> None:
        text = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertTrue(_uncommented(text, 'XPROF_ARGS="--xprof --xprof-iters'))
        self.assertFalse(_uncommented(text, 'XPROF_ARGS=""'),
                         "an empty XPROF_ARGS implies tracing can be switched off here")

    def test_the_launcher_does_not_claim_tracing_is_optional(self) -> None:
        text = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertNotIn("Off by default: it is the only measurement", text)

    def test_the_entrypoint_default_and_the_launcher_agree(self) -> None:
        source = (COLLX / "bench" / "run_ep_jax.py").read_text()
        self.assertIn('ap.add_argument("--xprof", action="store_true", default=True', source)
        # --no-xprof still exists for a hand-run, but must be paired with the fallback flag.
        self.assertIn('"--no-xprof"', source)
        self.assertIn("allow-host-fallback", source)


class SamplingProvenanceTests(unittest.TestCase):
    """The artifact must publish the counts the code actually uses.

    It previously published the case's scheduled 8 iters x 256 trials x 32 warmup beside a
    comment claiming that profile "governs only the host sanity check". It governed nothing:
    the host loop hardcodes its own counts, and `args.iters/trials/warmup` had no other use
    than a positivity check. A reader would have concluded the host cross-check rested on
    2048 samples when it rests on 10.
    """

    ARGS = types.SimpleNamespace(xprof_iters=20, iters=8, trials=256, warmup=32,
                                 chain_iters=64, chain_drop=1)

    def test_host_counts_are_the_loop_s_own(self) -> None:
        block = run_ep_jax._sampling_provenance(self.ARGS)
        self.assertEqual(block["host_samples_per_point"], run_ep_jax.HOST_NUM_RUNS)
        self.assertEqual(block["host_warmup_per_point"], run_ep_jax.HOST_WARMUP_TRIES)

    def test_the_loop_uses_the_published_constants(self) -> None:
        """The drift guard: publishing a constant is worthless if the loop hardcodes another."""
        source = Path(run_ep_jax.__file__).read_text()
        self.assertIn("warmup_tries=HOST_WARMUP_TRIES, num_runs=HOST_NUM_RUNS", source,
                      "the host loop must take its counts from the published constants")
        self.assertNotIn("warmup_tries=5, num_runs=10", source,
                         "a hardcoded count can drift from what the artifact claims")

    def test_the_scheduled_profile_is_labelled_as_governing_nothing(self) -> None:
        block = run_ep_jax._sampling_provenance(self.ARGS)
        self.assertEqual(block["scheduled_iterations_per_trial"], 8)
        self.assertEqual(block["scheduled_trials"], 256)
        self.assertIsNone(block["scheduled_profile_governs"])
        # `samples_per_component` is shared with the GPU family deliberately, and carries
        # the count this backend really uses so a consumer indexing it is not misled.
        self.assertEqual(block["samples_per_component"], run_ep_jax.HOST_NUM_RUNS)
        # The other GPU key names must NOT be reused: they would imply they govern.
        for implied in ("iterations_per_trial", "trials", "warmup_iterations"):
            self.assertNotIn(implied, block,
                             f"{implied!r} reads as governing the published numbers")

    def test_device_occurrences_come_from_xprof_iters(self) -> None:
        self.assertEqual(
            run_ep_jax._sampling_provenance(self.ARGS)["device_occurrences_per_point"], 20)

    def test_both_documents_share_one_provenance_builder(self) -> None:
        """Failure and success artifacts drifted apart once already."""
        source = Path(run_ep_jax.__file__).read_text()
        # Call sites only -- `def _sampling_provenance(args)` matches the same substring.
        calls = source.count("_sampling_provenance(args)") - source.count(
            "def _sampling_provenance(args)")
        self.assertEqual(calls, 2, "both artifact documents must build sampling the same way")


def _chain_trace(path, marker, anchor, starts_by_device, duration_us=5.0):
    """A minimal Chrome trace: one anchor event per (device, start)."""
    events = []
    for pid, starts in starts_by_device.items():
        for start in starts:
            events.append({
                "ph": "X", "pid": pid, "tid": 1, "ts": start,
                "name": anchor,
                "args": {"tf_op": marker, "device_duration_ps": int(duration_us * 1e6)},
            })
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)


class ParseChainTests(unittest.TestCase):
    """The chained pair period, and the fail-closed check that guards it.

    A silently-wrong period is worse than no period: it would read as the pipeline's
    steady-state rate. So the occurrence check runs before any arithmetic, and XLA unrolling
    or deduplicating the loop must surface as `unavailable`, never as a number.
    """

    MARKER, ANCHOR = "collectivex-chain-t8", "ragged_all_to_all.7"

    def _parse(self, starts, iters, drop=1):
        with tempfile.TemporaryDirectory() as directory:
            _chain_trace(Path(directory) / "t.json.gz", self.MARKER, self.ANCHOR, starts)
            return xprof.parse_chain(directory, self.MARKER, self.ANCHOR, iters, drop)

    def test_a_clean_chain_yields_the_period(self) -> None:
        # Two devices, 5 pairs, 100us apart. drop=1 discards the fill delta.
        starts = {0: [0, 100, 200, 300, 400], 1: [1, 101, 201, 301, 401]}
        got = self._parse(starts, iters=5)
        self.assertEqual(got["availability"], "measured")
        self.assertEqual(got["origin"], "chained-median")
        self.assertEqual(got["sample_count"], 3)      # 4 deltas minus drop=1
        self.assertAlmostEqual(got["percentiles_us"]["p50"], 100.0)
        self.assertEqual(got["devices"], 2)

    def test_the_period_is_a_MEDIAN_not_a_MAX(self) -> None:
        """One device hiccuping must not become the pipeline's published rate."""
        starts = {0: [0, 100, 200, 300], 1: [0, 100, 200, 300], 2: [0, 100, 900, 1000]}
        got = self._parse(starts, iters=4, drop=0)
        # p50 does NOT discriminate here -- one hiccuping step washes out of the median of
        # the SERIES under either reduction, which is why the first version of this test
        # passed with a MAX cross-device reduction. p99 is where it surfaces: MEDIAN across
        # devices leaves every period at 100, MAX would publish the 800 as the pipeline rate.
        self.assertEqual(got["percentiles_us"]["p50"], 100.0)
        self.assertEqual(got["percentiles_us"]["p99"], 100.0)
        self.assertGreater(got["pair_spread_us"], 100.0)   # the hiccup shows up HERE

    def test_an_unrolled_loop_fails_closed(self) -> None:
        """The compiler emitting a different occurrence count must not divide by wrong N."""
        got = self._parse({0: [0, 100, 200], 1: [0, 100, 200]}, iters=8)
        self.assertEqual(got["availability"], "unavailable")
        self.assertIn("occurrences", got["reason"])

    def test_devices_disagreeing_on_the_count_fails_closed(self) -> None:
        got = self._parse({0: [0, 100, 200, 300], 1: [0, 100, 200]}, iters=4)
        self.assertEqual(got["availability"], "unavailable")
        self.assertEqual(sorted(got["occurrences_per_device"].values()), [3, 4])

    def test_a_missing_anchor_fails_closed_and_names_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _chain_trace(Path(directory) / "t.json.gz", self.MARKER,
                         "some_other_op", {0: [0, 100]})
            got = xprof.parse_chain(directory, self.MARKER, self.ANCHOR, 2)
        self.assertEqual(got["availability"], "unavailable")
        self.assertIn(self.ANCHOR, got["reason"])

    def test_floors_are_a_cross_device_MIN(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.json.gz"
            events = []
            for pid, dur in ((0, 9.0), (1, 4.0)):        # device 1 waits less
                for start in (0, 100, 200):
                    events.append({"ph": "X", "pid": pid, "tid": 1, "ts": start,
                                   "name": self.ANCHOR,
                                   "args": {"tf_op": self.MARKER,
                                            "device_duration_ps": int(dur * 1e6)}})
            with gzip.open(path, "wt") as handle:
                json.dump({"traceEvents": events}, handle)
            got = xprof.chain_floors(directory, self.MARKER, self.ANCHOR, drop=1)
        self.assertEqual(got["origin"], "chained-cross-rank-min")
        self.assertAlmostEqual(got["percentiles_us"]["p50"], 4.0)

    def test_both_anchor_spellings_are_recognised(self) -> None:
        """fp8 dispatch carries ONLY the underscored family; bf16 carries the hyphenated."""
        self.assertIn("ragged_all_to_all", xprof.CHAIN_ANCHOR_FAMILIES)
        self.assertIn("ragged-all-to-all", xprof.CHAIN_ANCHOR_FAMILIES)


class ChainFieldsTests(unittest.TestCase):
    """Chained rows publish the GPU family's shapes, and say so when they cannot."""

    def test_an_unusable_chain_publishes_unavailable_with_a_reason(self) -> None:
        got = run_ep_jax._chain_fields({"availability": "unavailable", "reason": "unrolled"})
        self.assertEqual(got["pair_period"]["availability"], "unavailable")
        self.assertEqual(got["pair_period"]["reason"], "unrolled")
        # Component-shaped, as on the GPU side -- unavailable, not a bare None, so a
        # consumer reading ["percentiles_us"] gets the same access path in both states.
        for field in ("pair_spread_us", "interpair_gap_us", "settle_drift_us"):
            block = got["chain_health"][field]
            self.assertEqual(block["availability"], "unavailable", field)
            self.assertIsNone(block["percentiles_us"], field)

    def test_a_missing_chain_still_publishes_the_blocks(self) -> None:
        """Presence-keyed: a consumer must not have to tell absent from failed."""
        for absent in (None, {}):
            got = run_ep_jax._chain_fields(absent)
            self.assertIn("pair_period", got)
            self.assertIn("chain_floor_us", got)
            self.assertIn("chain_health", got)

    def test_a_measured_chain_keeps_the_gpu_origin_strings(self) -> None:
        got = run_ep_jax._chain_fields({
            "availability": "measured", "origin": "chained-median",
            "percentiles_us": {"p50": 100.0}, "sample_count": 30,
            "pair_spread_us": 3.0, "anchor": "ragged_all_to_all.7", "devices": 8,
            "floors": {
                "dispatch": {"availability": "measured",
                             "origin": "chained-cross-rank-min",
                             "percentiles_us": {"p50": 40.0}, "sample_count": 30,
                             "anchor": "ragged_all_to_all.38"},
                "combine": {"availability": "measured",
                            "origin": "chained-cross-rank-min",
                            "percentiles_us": {"p50": 45.0}, "sample_count": 30,
                            "anchor": "ragged_all_to_all.24"},
            },
        })
        self.assertEqual(got["pair_period"]["origin"], "chained-median")
        self.assertEqual(got["chain_floor_us"]["dispatch"]["origin"],
                         "chained-cross-rank-min")
        self.assertEqual(got["chain_health"]["anchor"]["period"], "ragged_all_to_all.7")

    def test_the_two_directions_are_measured_separately(self) -> None:
        """An earlier cut published ONE series under both names, byte-identical."""
        got = run_ep_jax._chain_fields({
            "availability": "measured", "origin": "chained-median",
            "percentiles_us": {"p50": 100.0}, "sample_count": 30,
            "floors": {
                "dispatch": {"availability": "measured", "origin": "x",
                             "percentiles_us": {"p50": 40.0}, "sample_count": 30,
                             "anchor": "d_op"},
                "combine": {"availability": "measured", "origin": "x",
                            "percentiles_us": {"p50": 45.0}, "sample_count": 30,
                            "anchor": "c_op"},
            },
        })
        self.assertNotEqual(got["chain_floor_us"]["dispatch"]["percentiles_us"],
                            got["chain_floor_us"]["combine"]["percentiles_us"])
        self.assertEqual(got["chain_health"]["anchor"]["floor_dispatch"], "d_op")
        self.assertEqual(got["chain_health"]["anchor"]["floor_combine"], "c_op")

    def test_indistinguishable_directions_publish_BOTH_unavailable(self) -> None:
        """Never one series wearing two labels."""
        reason = "only 1 distinct collective op"
        got = run_ep_jax._chain_fields({
            "availability": "measured", "origin": "chained-median",
            "percentiles_us": {"p50": 100.0}, "sample_count": 30,
            "floors": {"dispatch": {"availability": "unavailable", "reason": reason},
                       "combine": {"availability": "unavailable", "reason": reason}},
        })
        for side in ("dispatch", "combine"):
            self.assertEqual(got["chain_floor_us"][side]["availability"], "unavailable")
            self.assertEqual(got["chain_floor_us"][side]["reason"], reason)

    def test_unavailable_blocks_are_component_shaped(self) -> None:
        """A consumer indexing .percentiles_us must not have to special-case TPU."""
        got = run_ep_jax._chain_fields(None)
        for probe in (got["pair_period"], got["chain_floor_us"]["dispatch"],
                      got["chain_floor_us"]["combine"]):
            self.assertEqual(
                set(probe) >= {"availability", "origin", "percentiles_us", "sample_count"},
                True, probe)

    def test_no_chained_per_op_median_or_max_is_published(self) -> None:
        """Absent by design: inter-device wait parks bistably in one op window."""
        got = run_ep_jax._chain_fields({
            "availability": "measured", "origin": "chained-median",
            "percentiles_us": {"p50": 1.0}, "sample_count": 1, "floors": {}})
        flat = json.dumps(got)
        for banned in ("chained-median-per-op", "chained_max", "per_op_median"):
            self.assertNotIn(banned, flat)

    def test_fp8_does_not_claim_a_chained_period(self) -> None:
        self.assertFalse(run_ep_jax.fp8_provenance("fp8")["chained_period"])
        self.assertTrue(run_ep_jax.fp8_provenance("bf16")["chained_period"])

    def test_chain_barrier_is_constant_false_for_schema_parity(self) -> None:
        for precision in ("bf16", "fp8"):
            self.assertIs(run_ep_jax.fp8_provenance(precision)["chain_barrier"], False)

    def test_the_chain_budget_is_published_and_says_what_it_governs(self) -> None:
        block = run_ep_jax._sampling_provenance(types.SimpleNamespace(
            xprof_iters=20, iters=8, trials=256, warmup=32, chain_iters=64, chain_drop=1))
        self.assertEqual(block["chain_iterations_per_trial"], 64)
        self.assertEqual(block["chain_trials"], 1)
        self.assertEqual(block["chain_drop"], 1)
        self.assertIn("pair_period", block["chain_governs"])

    def test_row_components_nest_pair_period_and_sum_fp8_stage(self) -> None:
        pcts = {
            "dispatch": {"p50": 10.0, "p90": 11.0, "p95": 12.0, "p99": 13.0},
            "stage": {"p50": 2.0, "p90": 3.0, "p95": 4.0, "p99": 5.0},
            "combine": {"p50": 20.0, "p90": 21.0, "p95": 22.0, "p99": 23.0},
            "roundtrip": {"p50": 25.0, "p90": 26.0, "p95": 27.0, "p99": 28.0},
        }
        period = {"availability": "measured", "percentiles_us": {"p50": 7.0}}
        components = run_ep_jax._row_components(
            pcts,
            {"dispatch": 4, "stage": 4, "combine": 4, "roundtrip": 4},
            period,
        )

        self.assertIs(components["pair_period"], period)
        self.assertEqual(components["isolated_sum"]["percentiles_us"]["p50"], 32.0)

    def test_row_components_treat_unavailable_stage_as_zero(self) -> None:
        pcts = {
            "dispatch": {"p50": 10.0},
            "stage": None,
            "combine": {"p50": 20.0},
            "roundtrip": {"p50": 25.0},
        }
        components = run_ep_jax._row_components(
            pcts,
            {"dispatch": 1, "stage": 0, "combine": 1, "roundtrip": 1},
            {"availability": "unavailable", "percentiles_us": None},
        )

        self.assertEqual(components["isolated_sum"]["percentiles_us"]["p50"], 30.0)
        self.assertEqual(components["stage"]["availability"], "unavailable")


def _deepseek_layout(ep_size: int, tokens: int, experts: int = 256, top_k: int = 8):
    """The layout the sweep actually runs, not the small fixture."""
    rng = np.random.default_rng(0)
    idx = np.stack([rng.choice(experts, size=top_k, replace=False) for _ in range(tokens)])
    return ep_jax.build_layout(idx.astype(np.int32), tokens // ep_size, ep_size,
                               experts // ep_size)


class ChainProgramTests(unittest.TestCase):
    """The chain body must be loop-CARRIED; a barrier alone does not keep it alive."""

    def test_the_loop_count_is_static_in_the_program(self) -> None:
        """Load-bearing on a multi-host slice: both processes must issue one sequence."""
        source = Path(ep_jax.__file__).read_text()
        self.assertIn("jax.lax.fori_loop(0, _iters, body", source)
        self.assertIn("_iters", source)

    def test_the_carry_is_the_renormalised_combine_output(self) -> None:
        """The divide is what gives the chain an oracle; without it the carry runs to inf.

        Also pins the ORDER: the divide happens in fp32, before the cast back, because
        casting first would round twice and lose the bitwise identity.
        """
        source = Path(ep_jax.__file__).read_text()
        self.assertIn("summed = _combine_local(received, plan)", source)
        self.assertIn("return (summed / copies).astype(carry.dtype)", source)

    def test_one_chained_pair_is_bitwise_the_identity(self) -> None:
        """The oracle's premise, checked against numpy rather than asserted.

        Combine sums `d_t` copies of a bf16 value in fp32 (exact, d_t <= 8) and IEEE
        division is correctly rounded, so `(d_t*v)/d_t` returns `v`. Run for the full
        default 64 iterations: any drift would compound and show.
        """
        layout = _deepseek_layout(ep_size=16, tokens=512)
        ep, tpr = layout.ep_size, layout.tokens_per_rank
        rng = np.random.default_rng(7)
        # bf16-representable: keep only the top 8 mantissa bits of a float32.
        x = rng.standard_normal((ep, tpr, 4), dtype=np.float32)
        x = (x.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
        copies = layout.copies_per_token.astype(np.float32)[:, :, None]
        self.assertGreater(copies.min(), 0, "a token with no destination divides by zero")

        carry = x.copy()
        for _ in range(64):
            summed = np.zeros_like(carry, dtype=np.float32)
            for rank in range(ep):
                staged = carry[rank][layout.send_index[rank][:layout.send_total[rank]]]
                # dispatch->combine with an identity expert returns the staged rows
                np.add.at(summed[rank], layout.send_index[rank][:layout.send_total[rank]],
                          staged)
            carry = summed / copies
        np.testing.assert_array_equal(carry, x)

    def test_the_device_gets_the_COUNT_not_its_reciprocal(self) -> None:
        """Multiplying by a precomputed 1/d would be cheaper and would break the oracle.

        Measured over 200k bf16-representable float32 values: `(d*v)/d == v` for every
        d in 1..8, but `(d*v) * float32(1/d)` disagrees for d == 7 on 58% of them (and
        only d == 7 -- the other counts happen to round back). d == 7 is not hypothetical:
        it is inside the measured 3..7 range at EP8 and 4..8 at EP16, so a reciprocal here
        would fail the chain identity on real tokens, on real layouts.
        """
        v = np.random.default_rng(3).standard_normal(20000, dtype=np.float32)
        v = (v.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
        seven = np.float32(7)
        self.assertTrue((((seven * v) / seven) == v).all())
        self.assertFalse((((seven * v) * (np.float32(1) / seven)) == v).all())

        source = Path(ep_jax.__file__).read_text()
        self.assertIn("layout.copies_per_token.astype(np.float32)[:, :, None]", source)
        self.assertNotIn("1.0 / layout.copies_per_token", source)
        self.assertIn("summed / copies", source)

    def test_seven_destinations_really_occurs(self) -> None:
        """The premise of the test above, on the layout the sweep runs."""
        for ep in (8, 16):
            counts = set(_deepseek_layout(ep_size=ep, tokens=512).copies_per_token.ravel())
            self.assertIn(7, counts, f"EP{ep}")

    def test_without_the_divide_the_carry_saturates(self) -> None:
        """Why the divide is not cosmetic: measured growth is d_t per pair, d_t up to 8.

        bf16 tops out at 3.39e38, so the UNnormalised chain this replaced was computing
        infinities for the last third of its 64 iterations -- and an oracle comparing inf
        to inf passes for a transport that has dropped every row.
        """
        layout = _deepseek_layout(ep_size=16, tokens=512)
        copies = layout.copies_per_token
        # deepseek-v3 top-8 over 256 experts: measured 4..8 unique destinations per token.
        self.assertEqual(int(copies.max()), 8)
        self.assertGreaterEqual(int(copies.min()), 1)
        # The MILDEST token in this layout already overflows bf16 within the 64 the
        # sweep runs, so the saturation was total rather than confined to hot tokens.
        self.assertGreater(float(copies.min()) ** 64, 3.39e38)

    def test_fp8_has_no_chain_program(self) -> None:
        transport = _fp8_transport(hidden=2 * ep_jax.QUANT_BLOCK, ep_size=4)
        _, layout = _layout()
        plan = {name: getattr(layout, name).astype(np.int32) for name in (
            "send_index", "input_offsets", "send_sizes", "output_offsets",
            "recv_sizes", "recv_offsets", "return_offsets")}
        x = np.ones((1, layout.tokens_per_rank, 2 * ep_jax.QUANT_BLOCK),
                    np.float32).view(_Fp8Array)
        point = ep_jax.Point(transport=transport, layout=layout, x=x, **plan)
        self.assertIsNone(point.chain(8))


class ChainFloorAnchorTests(unittest.TestCase):
    """A floor must come from the collective, never from a marker op.

    Measured on the first chained run: `_chain_anchor` picked
    `prepare_start_ragged-all-to-all.3.cloned.1.call-start` -- most frequent, because markers
    are frequent -- and its 0.048us duration was published as `chain_floor_us` with origin
    `chained-cross-rank-min`, i.e. as per-op device time. The PERIOD survived (start-to-start
    deltas need only an op firing once per iteration); the floor was fabricated.
    """

    MARKER = "collectivex-chain-t8"

    def _trace(self, directory, ops):
        events = []
        for name, dur, count, *rest in ops:
            offset = rest[0] if rest else 0.0
            for pid in (0, 1):
                for i in range(count):
                    events.append({"ph": "X", "pid": pid, "tid": 1,
                                   "ts": i * 100.0 + offset,
                                   "name": name,
                                   "args": {"tf_op": self.MARKER,
                                            "device_duration_ps": int(dur * 1e6)}})
        with gzip.open(Path(directory) / "t.json.gz", "wt") as handle:
            json.dump({"traceEvents": events}, handle)

    def test_the_floor_anchor_skips_marker_ops(self) -> None:
        """The marker must lose even when it wins on TOTAL device time.

        A max-total heuristic alone is not enough: a start marker firing many times per
        iteration can out-total the collective, and the first version of this test let the
        real op win on total anyway, so it passed with the name filter removed.
        """
        with tempfile.TemporaryDirectory() as d:
            self._trace(d, [("prepare_start_ragged-all-to-all.3.call-start", 5.0, 4096),
                            ("ragged_all_to_all.8", 40.0, 64, 0.0),
                            ("ragged_all_to_all.9", 30.0, 64, 50.0)])
            got = xprof.floor_anchors(d, self.MARKER, 64)
        self.assertTrue(got["ok"], got)
        self.assertEqual((got["dispatch"], got["combine"]),
                         ("ragged_all_to_all.8", "ragged_all_to_all.9"))

    def test_call_done_is_NOT_treated_as_a_marker(self) -> None:
        """Measured at 6692.6us in a real dispatch inventory -- excluding it would discard
        the floor. Only START markers are excluded."""
        with tempfile.TemporaryDirectory() as d:
            self._trace(d, [("ragged-all-to-all.2.cloned.1.call-done", 6692.6, 64)])
            got = xprof.floor_anchors(d, self.MARKER, 64)
            # Only ONE collective name here, so direction cannot be assigned -- but the
            # call-done op must not have been filtered out as a marker, which the reason
            # names it to prove.
            self.assertFalse(got["ok"])
            self.assertIn("ragged-all-to-all.2.cloned.1.call-done", str(got.get("names")))

    def test_a_marker_sized_floor_fails_closed(self) -> None:
        """0.16% of the period is not a floor; publishing it would be a fabricated number."""
        with tempfile.TemporaryDirectory() as d:
            self._trace(d, [("ragged_all_to_all.8", 0.048, 64)])
            got = xprof.chain_floors(d, self.MARKER, "ragged_all_to_all.8",
                                     drop=1, period_us=30443.9)
        self.assertEqual(got["availability"], "unavailable")
        self.assertIn("marker op", got["reason"])

    def test_a_real_floor_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self._trace(d, [("ragged_all_to_all.8", 12000.0, 64)])
            got = xprof.chain_floors(d, self.MARKER, "ragged_all_to_all.8",
                                     drop=1, period_us=30443.9)
        self.assertEqual(got["availability"], "measured")
        self.assertAlmostEqual(got["percentiles_us"]["p50"], 12000.0)
        self.assertEqual(got["anchor"], "ragged_all_to_all.8")

    def test_no_period_means_no_share_check_but_still_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self._trace(d, [("ragged_all_to_all.8", 5.0, 8)])
            got = xprof.chain_floors(d, self.MARKER, "ragged_all_to_all.8", drop=1)
        self.assertEqual(got["availability"], "measured")


class FloorDirectionTests(unittest.TestCase):
    """Dispatch and combine floors must be two measurements or none."""

    MARKER = "collectivex-chain-t8"

    def _trace(self, directory, series):
        """series: {name: [(pid, ts, dur), ...]}"""
        events = []
        for name, rows in series.items():
            for pid, ts, dur in rows:
                events.append({"ph": "X", "pid": pid, "tid": 1, "ts": ts, "name": name,
                               "args": {"tf_op": self.MARKER,
                                        "device_duration_ps": int(dur * 1e6)}})
        with gzip.open(Path(directory) / "t.json.gz", "wt") as handle:
            json.dump({"traceEvents": events}, handle)

    def _two_ops(self, interleaved=True, iters=4):
        d, c = [], []
        for pid in (0, 1):
            for i in range(iters):
                d.append((pid, i * 100.0, 30.0))
                c.append((pid, i * 100.0 + (50.0 if interleaved else -10.0), 20.0))
        return {"ragged_all_to_all.38": d, "ragged_all_to_all.24": c}

    def test_direction_comes_from_start_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._trace(directory, self._two_ops())
            got = xprof.floor_anchors(directory, self.MARKER, 4)
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["dispatch"], "ragged_all_to_all.38")   # runs first
        self.assertEqual(got["combine"], "ragged_all_to_all.24")

    def test_non_interleaving_ops_fail_closed(self) -> None:
        """d,d,c,c is not a chain of pairs; start order cannot assign direction."""
        rows = {"ragged_all_to_all.38": [(0, 0.0, 30.0), (0, 10.0, 30.0)],
                "ragged_all_to_all.24": [(0, 20.0, 20.0), (0, 30.0, 20.0)]}
        with tempfile.TemporaryDirectory() as directory:
            self._trace(directory, rows)
            got = xprof.floor_anchors(directory, self.MARKER, 2)
        self.assertFalse(got["ok"])
        self.assertIn("interleave", got["reason"])

    def test_a_single_collective_name_cannot_be_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._trace(directory, {"ragged_all_to_all.8": [(0, 0.0, 30.0),
                                                            (0, 100.0, 30.0)]})
            got = xprof.floor_anchors(directory, self.MARKER, 2)
        self.assertFalse(got["ok"])
        self.assertIn("cannot tell dispatch from combine", got["reason"])

    def test_a_wrong_occurrence_count_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._trace(directory, self._two_ops(iters=3))
            got = xprof.floor_anchors(directory, self.MARKER, 64)
        self.assertFalse(got["ok"])
        self.assertIn("not 64", got["reason"])


class ChainGatherTests(unittest.TestCase):
    """At EP16 the local trace holds half the slice, and it is the faster half."""

    def test_single_process_is_a_no_op(self) -> None:
        run_ep_jax._PROCESS_COUNT[0] = 1
        chain = {"per_device_periods": [[1.0, 2.0]], "devices": 1}
        self.assertIs(run_ep_jax.gather_chain_periods(chain, 8), chain)

    def test_multi_process_folds_in_the_other_host(self) -> None:
        """The reduction must cover 16 devices, not the 8 this host profiled."""
        import types as _t
        run_ep_jax._PROCESS_COUNT[0] = 2
        local = [[100.0, 100.0], [100.0, 100.0]]          # this host: 2 devices, fast

        def process_allgather(value, tiled=False):
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape == (1,):                          # the length agreement
                return np.array([arr[0], arr[0]])
            remote = np.where(np.isnan(arr), np.nan, 400.0)  # other host: stragglers
            return np.stack([arr, remote])

        utils = _t.ModuleType("jax.experimental.multihost_utils")
        utils.process_allgather = process_allgather
        experimental = _t.ModuleType("jax.experimental")
        experimental.multihost_utils = utils
        jax_mod = _t.ModuleType("jax")
        jax_mod.experimental = experimental
        try:
            with mock.patch.dict(sys.modules, {
                    "jax": jax_mod, "jax.experimental": experimental,
                    "jax.experimental.multihost_utils": utils}):
                got = run_ep_jax.gather_chain_periods(
                    {"per_device_periods": local, "availability": "measured"}, 16)
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1
        self.assertTrue(got["gathered_across_hosts"])
        self.assertEqual(got["devices_expected"], 16)
        self.assertEqual(got["devices"], 4)                # 2 local + 2 remote, no NaN pad
        # The straggler half must move the median off the local-only value.
        self.assertGreater(got["percentiles_us"]["p50"], 100.0)


class ChainIdentityOracleTests(unittest.TestCase):
    """The chained program's output is scored, and a failure withholds the period.

    Reads `.addressable_shards`, not `device_get` on the global array: on a multi-host
    slice the latter raises, and the first EP16 run to reach this code returned
    `chain_regime_passed=False` at all 10 points because the ORACLE could not run, on a
    transport whose drained oracle had just scored 0.0.
    """

    class _Shard:
        def __init__(self, data, device_id):
            self.data = data
            self.device = types.SimpleNamespace(id=device_id)

    class _Array:
        """Sharded over `n` devices along axis 0, like the real thing."""
        def __init__(self, data, n=2, ids=None):
            chunks = np.array_split(np.asarray(data), n)
            ids = ids if ids is not None else range(n)
            self.addressable_shards = [
                ChainIdentityOracleTests._Shard(c, i) for c, i in zip(chunks, ids)]

    def _jax(self):
        return types.SimpleNamespace(device_get=lambda v: v)

    def _point(self, x):
        return types.SimpleNamespace(x=self._Array(x))

    def setUp(self) -> None:
        run_ep_jax._PROCESS_COUNT[0] = 1

    def test_an_exact_chain_passes(self) -> None:
        x = np.arange(24, dtype=np.float32).reshape(6, 4)
        got = run_ep_jax.chain_identity(self._jax(), self._Array(x), self._point(x))
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["max_rel"], 0.0)

    def test_shards_are_aligned_by_device_id_not_list_order(self) -> None:
        """Shard listing order is not guaranteed; misaligning them would score a correct
        chain as corrupt, and a corrupt one as correct."""
        x = np.arange(24, dtype=np.float32).reshape(6, 4)
        produced = self._Array(x)                      # shard k really is on device k
        produced.addressable_shards.reverse()          # ...but listed backwards
        got = run_ep_jax.chain_identity(self._jax(), produced, self._point(x))
        self.assertTrue(got["ok"], got)

    def test_one_perturbed_element_fails(self) -> None:
        """Not a tolerance: the claim is bitwise, so one ulp is a failure."""
        x = np.arange(24, dtype=np.float32).reshape(6, 4) + 1.0
        bad = x.copy()
        bad[3, 2] = np.nextafter(bad[3, 2], np.float32(1e9))
        got = run_ep_jax.chain_identity(self._jax(), self._Array(bad), self._point(x))
        self.assertIs(got["ok"], False)
        self.assertGreater(got["max_rel"], 0.0)

    def test_the_saturated_case_is_reported_not_scored(self) -> None:
        x = np.ones((6, 4), dtype=np.float32)
        bad = np.full((6, 4), np.inf, dtype=np.float32)
        got = run_ep_jax.chain_identity(self._jax(), self._Array(bad), self._point(x))
        self.assertIs(got["ok"], False)
        self.assertIn("non-finite", got["reason"])
        self.assertIsNone(got["max_rel"])

    def test_an_oracle_that_cannot_run_is_None_not_False(self) -> None:
        """`None` withholds the period; `False` condemns the row. A multi-host read error
        is the first, and collapsing it into the second cost 10 good EP16 rows."""
        broken = types.SimpleNamespace()                # no addressable_shards at all
        got = run_ep_jax.chain_identity(
            self._jax(), broken, self._point(np.ones((6, 4), np.float32)))
        self.assertIsNone(got["ok"])
        self.assertIn("could not run", got["reason"])

    def test_a_shape_disagreement_is_unevaluated_not_failed(self) -> None:
        got = run_ep_jax.chain_identity(
            self._jax(), self._Array(np.ones((4, 4), np.float32)),
            self._point(np.ones((6, 4), np.float32)))
        self.assertIsNone(got["ok"])
        self.assertIn("could not align", got["reason"])

    def test_a_failed_identity_withholds_the_period(self) -> None:
        source = Path(run_ep_jax.__file__).read_text()
        self.assertIn('if identity.get("ok") is not True:', source)
        self.assertIn('chain["availability"] = "unavailable"', source)
        self.assertIn('chain["percentiles_us"] = None', source)

    def test_only_a_RUN_disagreement_fails_the_row(self) -> None:
        self.assertIsNone(run_ep_jax._correctness(0.0, True, {"ok": None})["chain_regime_passed"])
        self.assertTrue(run_ep_jax._correctness(0.0, True, {"ok": None})["passed"],
                        "an unevaluable oracle condemned a row whose drained checks passed")
        self.assertFalse(run_ep_jax._correctness(0.0, True, {"ok": False})["passed"])
        self.assertTrue(run_ep_jax._correctness(0.0, True, {"ok": True})["passed"])


class VanishedJobDetectionTests(unittest.TestCase):
    """A transient kubectl failure is not a deleted Job.

    Measured, run 31109742122: two of three shards aborted with "no longer exists" at 11.5
    and 31.5 minutes, both mid-case with the pod doing real work and neither old enough for
    the 900s TTL. The old guard was `! kubectl get job`, true for ANY kubectl failure, and
    the two status reads above it swallow errors with `|| true` -- so a single API blip made
    all three conditions hold and killed a healthy shard. The flakes this produced were
    previously attributed to another team's sweep on the same cluster.
    """

    @staticmethod
    def _fragment() -> str:
        """The REAL decision, sliced out of the launcher rather than restated here."""
        text = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        start = text.index('  if [ -z "$succeeded" ] && [ -z "$failed" ]; then')
        end = text.index('collx_log "ERROR: bench Job $JOB no longer exists', start)
        return text[start:end].rsplit("if [", 1)[0]

    def _run(self, kubectl_body: str, polls: int = 4) -> list:
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / "kubectl"
            stub.write_text("#!/usr/bin/env bash\n" + kubectl_body + "\n")
            stub.chmod(0o755)
            script = (f'set -uo pipefail\nKUBECTL={stub}\nNS=ns\nJOB=j\n'
                      f'succeeded=""\nfailed=""\nVANISHED=0\n'
                      f'for _ in $(seq 1 {polls}); do\n{self._fragment()}\n'
                      f'echo "$VANISHED"\ndone\n')
            out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return [int(line) for line in out.stdout.split()]

    def test_a_transient_api_failure_never_counts(self) -> None:
        counts = self._run('echo "Unable to connect to the server: dial tcp: i/o timeout" >&2\nexit 1')
        self.assertEqual(counts, [0, 0, 0, 0], "an API blip was read as a deleted Job")

    def test_throttling_never_counts(self) -> None:
        counts = self._run('echo "error: the server has asked for the client to provide credentials" >&2\nexit 1')
        self.assertEqual(counts, [0, 0, 0, 0])

    def test_a_real_notfound_accumulates_to_the_threshold(self) -> None:
        counts = self._run('echo "Error from server (NotFound): jobs.batch \\"j\\" not found" >&2\nexit 1')
        self.assertEqual(counts, [1, 2, 3, 4])

    def test_a_living_job_resets_the_counter(self) -> None:
        """NotFound twice then present again must not abort on the next NotFound."""
        body = ('n=$(cat /tmp/cx-vanish-n 2>/dev/null || echo 0); echo $((n+1)) > /tmp/cx-vanish-n\n'
                'if [ "$n" -lt 2 ]; then echo "Error from server (NotFound): nope" >&2; exit 1; fi\n'
                'echo j; exit 0')
        Path("/tmp/cx-vanish-n").unlink(missing_ok=True)
        try:
            counts = self._run(body)
        finally:
            Path("/tmp/cx-vanish-n").unlink(missing_ok=True)
        self.assertEqual(counts, [1, 2, 0, 0])

    def test_the_threshold_is_more_than_one_poll(self) -> None:
        text = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertTrue(_uncommented(text, 'VANISH_CONFIRMATIONS="${COLLX_TPU_VANISH_CONFIRMATIONS:-3}"'))
        self.assertTrue(_uncommented(text, '[ "$VANISHED" -ge "$VANISH_CONFIRMATIONS" ]'))
        self.assertFalse(_uncommented(text, '! $KUBECTL -n "$NS" get job "$JOB" >/dev/null 2>&1'),
                         "the bare-failure guard is what mistook a blip for a deletion")


class TransportOverlapTests(unittest.TestCase):
    """`transport_us` sums a scope's ops; that is a transport time only if they are disjoint.

    At T=8192 combine the scope holds `...call-done` (6690.6us) and `ragged_all_to_all.6`
    (3284.6us), summed to a published 9975.2. Both scale exactly 2.00x per doubling of the
    payload, so duration alone cannot distinguish "two disjoint transfers" from "an async
    span enclosing its own collective" -- and the second would mean the published figure is
    ~1.5x the truth. This measures it on the timeline.
    """

    @staticmethod
    def _events(spans, pid=0):
        return [{"ph": "X", "pid": pid, "tid": 1, "ts": ts, "dur": dur, "name": "op",
                 "args": {"device_duration_ps": int(dur * 1e6)}} for ts, dur in spans]

    def test_back_to_back_ops_are_disjoint(self) -> None:
        got = xprof.overlap_us(self._events([(0.0, 100.0), (100.0, 50.0)]))
        self.assertTrue(got["disjoint"])
        self.assertEqual(got["overlap_us"], 0.0)
        self.assertEqual(got["union_us"], 150.0)

    def test_an_enclosing_span_is_caught(self) -> None:
        """call-done enclosing the inner collective: the sum would double-count it."""
        got = xprof.overlap_us(self._events([(0.0, 100.0), (20.0, 40.0)]))
        self.assertFalse(got["disjoint"])
        self.assertEqual(got["overlap_us"], 40.0)     # the whole inner op
        self.assertEqual(got["union_us"], 100.0)

    def test_partial_concurrency_is_caught(self) -> None:
        got = xprof.overlap_us(self._events([(0.0, 100.0), (75.0, 50.0)]))
        self.assertFalse(got["disjoint"])
        self.assertEqual(got["overlap_us"], 25.0)

    def test_the_worst_device_is_reported(self) -> None:
        """MAX across devices, the same rule the spans use.

        Every device overlaps here, and the worst is not the first: a scan that stops at
        the first nonzero would report 10.0 and understate the double-count 9x.
        """
        events = (self._events([(0.0, 100.0), (90.0, 20.0)], pid=0)     # 10
                  + self._events([(0.0, 100.0), (10.0, 90.0)], pid=1)   # 90
                  + self._events([(0.0, 100.0), (80.0, 40.0)], pid=2))  # 20
        got = xprof.overlap_us(events)
        self.assertEqual(got["devices"], 3)
        self.assertEqual(got["overlap_us"], 90.0)
        self.assertFalse(got["disjoint"])

    def test_placement_and_occupancy_are_not_mixed(self) -> None:
        """Overlap must come from ts/dur alone. `device_duration_ps` measures how long an
        op RAN, not where it sat -- differencing one against the other invents overlap."""
        events = self._events([(0.0, 50.0), (50.0, 50.0)])
        for event in events:            # occupancy 4x the wall placement it sat in
            event["args"]["device_duration_ps"] = int(200.0 * 1e6)
        got = xprof.overlap_us(events)
        self.assertTrue(got["disjoint"],
                        "occupancy differenced against placement invents an overlap")
        self.assertEqual(got["overlap_us"], 0.0)

    def test_an_empty_scope_reports_unknown_not_disjoint(self) -> None:
        got = xprof.overlap_us([])
        self.assertIsNone(got["disjoint"])
        self.assertIsNone(got["overlap_us"])


class ChainInventoryAndBudgetTests(unittest.TestCase):
    """Items 5 and 9: what the chain body contains, and what capturing it costs."""

    @staticmethod
    def _row(**extra):
        chain = {"availability": "measured", "origin": "chained-median",
                 "percentiles_us": {"p50": 10.0}, "sample_count": 4, "floors": {}}
        chain.update(extra)
        return run_ep_jax._chain_fields(chain)["chain_health"]

    def test_the_op_inventory_REACHES_the_row(self) -> None:
        """Computing it and dropping it on the floor is what happened first: the value was
        assigned onto the chain dict and `_chain_fields` never emitted it, so run
        31147054940 published nulls while the code that produced it ran every point. The
        original test asserted the assignment existed, which it did."""
        inventory = [{"name": "ragged_all_to_all.8", "per_iteration_us": 12.5}]
        self.assertEqual(self._row(op_inventory=inventory)["op_inventory"], inventory)

    def test_the_renorm_cost_REACHES_the_row(self) -> None:
        got = self._row(renorm_us=1.75)["renorm_us"]
        self.assertEqual(got["availability"], "measured")
        self.assertEqual(got["percentiles_us"]["p50"], 1.75)

    def test_the_capture_margin_REACHES_the_row(self) -> None:
        got = self._row(capture_s=48.5, capture_budget_s=2700)
        self.assertEqual(got["capture_s"], 48.5)
        self.assertEqual(got["capture_budget_s"], 2700)

    def test_an_unavailable_chain_still_carries_the_keys(self) -> None:
        """A consumer must not have to branch on whether the chain ran."""
        health = run_ep_jax._chain_fields(
            {"availability": "unavailable", "reason": "x"})["chain_health"]
        for key in ("renorm_us", "op_inventory", "capture_s", "capture_budget_s"):
            self.assertIn(key, health)

    def test_the_capture_is_timed_from_before_the_warm_call(self) -> None:
        """The warm call runs the same 64 pairs as the traced one; starting the clock after
        it would halve the reported cost of the thing being budgeted."""
        source = Path(run_ep_jax.__file__).read_text()
        started = source.index("chain_started = time.time()")
        warm = source.index("warm = chain_call()", started - 200)
        self.assertLess(started, warm, "the warm call must be inside the measured window")

    def test_the_budget_follows_the_launcher(self) -> None:
        """COLLX_RUN_TIMEOUT is what actually kills the case, so the margin is against it."""
        self.assertEqual(run_ep_jax.CHAIN_CAPTURE_BUDGET_S, 2700)
        source = Path(run_ep_jax.__file__).read_text()
        self.assertIn('os.environ.get("COLLX_RUN_TIMEOUT", "2700")', source)
        launcher = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertTrue(_uncommented(launcher, 'RUN_TIMEOUT="${COLLX_RUN_TIMEOUT:-2700}"'),
                        "the probe's budget and the launcher's timeout have diverged")


class InterpairGapTests(unittest.TestCase):
    """`interpair_gap_us` = start-to-start MINUS the pair window. GPU parity.

    Three bases were tried on this side and the first two were wrong in the same direction,
    each leaving real in-iteration work inside the "gap":

      * `period - (dispatch_floor + combine_floor)` -- only the two collectives subtracted.
        Read 47% of the period at T=8192 and could never approach zero.
      * anchor-start to anchor-END -- misses the permute BEFORE dispatch's collective and
        the scatter-add AFTER combine's: 1,769us + 3,363us, 7.3ms together at T=8192.
        Read 20..30%.
      * the chain scope's own extent per iteration, which is what this is.

    The third is also independent of which ops the floor gate picked, so a point whose
    anchors are unusable still reports a gap.
    """

    MARKER = "collectivex-chain-t8"
    ANCHOR = "prepare_start_ragged-all-to-all.3.call-start"

    def _gap(self, period, spans, iters=8, drop=1, devices=(0,)):
        """`spans` are (offset, duration) within each iteration, besides the anchor."""
        events = []
        for pid in devices:
            for i in range(iters):
                base = i * period
                events.append((self.ANCHOR, base, 0.05, pid))
                for offset, dur in spans:
                    events.append(("op", base + offset, dur, pid))
        payload = [{"ph": "X", "pid": pid, "tid": 1, "ts": ts, "dur": dur, "name": name,
                    "args": {"tf_op": self.MARKER,
                             "device_duration_ps": int(dur * 1e6)}}
                   for name, ts, dur, pid in events]
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "t.json.gz", "wt") as handle:
                json.dump({"traceEvents": payload}, handle)
            return xprof.chain_pair_gap(directory, self.MARKER, self.ANCHOR, drop)

    def test_a_free_running_chain_reports_near_zero(self) -> None:
        self.assertAlmostEqual(self._gap(100.0, [(0.0, 100.0)]), 0.0, places=6)

    def test_a_drained_chain_reports_the_drain(self) -> None:
        self.assertAlmostEqual(self._gap(100.0, [(0.0, 60.0)]), 40.0, places=6)

    def test_work_outside_the_collectives_is_NOT_gap(self) -> None:
        """A permute at the front and a scatter-add at the back are in-iteration work; an
        anchor-to-anchor window would charge both to the gap."""
        spans = [(0.0, 20.0), (20.0, 40.0), (60.0, 35.0)]   # permute, collective, scatter
        self.assertAlmostEqual(self._gap(100.0, spans), 5.0, places=6)

    def test_the_window_runs_to_the_LAST_op_end(self) -> None:
        self.assertAlmostEqual(self._gap(100.0, [(0.0, 10.0), (80.0, 15.0)]), 5.0, places=6)

    def test_it_is_reduced_across_devices(self) -> None:
        self.assertAlmostEqual(
            self._gap(100.0, [(0.0, 70.0)], devices=(0, 1, 2)), 30.0, places=6)

    def test_the_drop_is_honoured(self) -> None:
        self.assertIsNotNone(self._gap(100.0, [(0.0, 100.0)], iters=8, drop=4))

    def test_it_does_not_depend_on_the_floor_gate(self) -> None:
        source = Path(run_ep_jax.__file__).read_text()
        self.assertIn("xprof.chain_pair_gap(\n                            directory, marker, anchor",
                      source)
        self.assertNotIn('split["dispatch"],\n', source)

    def test_a_trace_without_the_anchor_is_unknown_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "t.json.gz", "wt") as handle:
                json.dump({"traceEvents": []}, handle)
            self.assertIsNone(
                xprof.chain_pair_gap(directory, self.MARKER, self.ANCHOR))

    def test_no_trace_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(xprof.chain_pair_gap(directory, self.MARKER, self.ANCHOR))


class AnchorPhaseTests(unittest.TestCase):
    """Alternation cannot tell a dispatch/combine pair from two ops of one direction.

    Each component's inventory carries more than one collective op -- measured, both
    `...call-done` and a `ragged_all_to_all.N`. So the two highest-total names may both be
    DISPATCH's, and their starts would read d1a, d1b, d2a, d2b: perfectly alternating and
    perfectly mislabelled. Where the second op starts within the period separates them.
    """

    MARKER = "collectivex-chain-t8"

    def _phase(self, offset, period=100.0, iters=6):
        events = []
        for pid in (0, 1):
            for i in range(iters):
                base = i * period
                events.append(("ragged_all_to_all.A", base, 30.0, pid))
                events.append(("ragged_all_to_all.B", base + offset, 20.0, pid))
        with tempfile.TemporaryDirectory() as directory:
            payload = [{"ph": "X", "pid": pid, "tid": 1, "ts": ts, "dur": dur, "name": name,
                        "args": {"tf_op": self.MARKER,
                                 "device_duration_ps": int(dur * 1e6)}}
                       for name, ts, dur, pid in events]
            with gzip.open(Path(directory) / "t.json.gz", "wt") as handle:
                json.dump({"traceEvents": payload}, handle)
            return xprof.floor_anchors(directory, self.MARKER, iters)

    def test_a_real_pair_sits_near_half_the_period(self) -> None:
        """Period deliberately not 100: the fraction must come from the MEASURED period,
        not a literal, or it is meaningless at every ladder point but one."""
        got = self._phase(offset=140.0, period=280.0)
        self.assertTrue(got["ok"], got)
        self.assertAlmostEqual(got["phase"], 0.5, places=6)
        self.assertIn("consistent with", got["phase_note"])
        # ... and the same offset against a different period is a different phase
        self.assertAlmostEqual(self._phase(offset=140.0, period=700.0)["phase"], 0.2,
                               places=6)

    def test_two_ops_of_one_direction_are_flagged(self) -> None:
        """Still alternating, still passes the interleave gate -- and still wrong."""
        got = self._phase(offset=4.0)
        self.assertTrue(got["ok"], got)      # the gate does not catch it
        self.assertAlmostEqual(got["phase"], 0.04, places=6)
        self.assertIn("SUSPECT", got["phase_note"])

    def test_an_anchor_late_in_the_period_is_also_flagged(self) -> None:
        got = self._phase(offset=96.0)
        self.assertGreater(got["phase"], 0.75)
        self.assertIn("SUSPECT", got["phase_note"])

    def test_the_phase_is_published_not_gated(self) -> None:
        """A threshold would need a value nobody has measured on this fabric yet."""
        self.assertTrue(self._phase(offset=4.0)["ok"])


class DevicesStampTests(unittest.TestCase):
    """`devices_expected` must say what was expected, including when nothing was gathered.

    null reads as "nobody checked". 8 of 8 is the honest EP8 answer, and it is what makes
    the EP16 case -- 16 expected, fewer columns surviving -- legible by contrast.
    """

    def test_single_host_still_stamps_the_expectation(self) -> None:
        run_ep_jax._PROCESS_COUNT[0] = 1
        got = run_ep_jax.gather_chain_periods({"per_device_periods": [[1.0]]}, 8)
        self.assertEqual(got["devices_expected"], 8)

    def test_it_reaches_the_artifact(self) -> None:
        source = Path(run_ep_jax.__file__).read_text()
        self.assertIn('"devices_expected": chain.get("devices_expected")', source)


class GpuContractParityTests(unittest.TestCase):
    """Schema parity with the GPU family (PR 2489, `collectivex-fixes`).

    These are consumer contracts: a reader walks `chain_health.<field>.percentiles_us.p50`
    uniformly across SKUs. Publishing a bare float here would break that walk on TPU rows
    only, which is the kind of divergence nobody notices until a cross-SKU chart is wrong.
    """

    def test_chain_health_fields_are_component_blocks(self) -> None:
        chain = {"availability": "measured", "origin": "chained-median",
                 "percentiles_us": {"p50": 10.0}, "sample_count": 4,
                 "pair_spread_us": 1.5, "interpair_gap_us": 0.25,
                 "settle_drift_us": -0.75, "devices": 8, "devices_expected": 8,
                 "floors": {}}
        health = run_ep_jax._chain_fields(chain)["chain_health"]
        for field, value in (("pair_spread_us", 1.5), ("interpair_gap_us", 0.25),
                             ("settle_drift_us", -0.75)):
            block = health[field]
            self.assertEqual(block["availability"], "measured", field)
            self.assertIsNotNone(block["origin"], field)
            self.assertEqual(block["percentiles_us"]["p50"], value, field)
            # Every percentile carries the reduced scalar: it IS one number, and saying so
            # beats inventing a spread the reduction never produced.
            self.assertEqual(set(block["percentiles_us"]), {"p50", "p90", "p95", "p99"})

    def test_the_origin_strings_are_the_gpu_family_s(self) -> None:
        source = Path(xprof.__file__).read_text()
        self.assertIn('"origin": "chained-median"', source)
        self.assertIn('"origin": "chained-cross-rank-min"', source)

    def test_correctness_carries_the_chain_regime_gate(self) -> None:
        """Tri-state. None means the backend was not chained, never "it passed"."""
        self.assertIsNone(run_ep_jax._correctness(0.0, True)["chain_regime_passed"])
        self.assertIsNone(
            run_ep_jax._correctness(0.0, True, None)["chain_regime_passed"])
        ok = run_ep_jax._correctness(0.0, True, {"ok": True})
        self.assertTrue(ok["chain_regime_passed"])
        self.assertTrue(ok["passed"])

    def test_a_failed_chain_regime_fails_the_row(self) -> None:
        """The drained passes only check one pair entered from idle, so a transport that
        corrupts under free-running pairs would otherwise be the fastest in the suite."""
        got = run_ep_jax._correctness(0.0, True, {"ok": False, "reason": "not bitwise"})
        self.assertFalse(got["chain_regime_passed"])
        self.assertFalse(got["passed"], "a corrupt chained regime published as passing")

    def test_settle_drift_is_signed_max_magnitude_across_devices(self) -> None:
        """GPU: `max(drifts, key=abs)` per rank. Reducing to a median series FIRST hides
        the one device still filling its pipeline behind the seven that have settled."""
        settled = [10.0] * 8
        drifting = [10.0, 10.0, 10.0, 10.0, 40.0, 40.0, 40.0, 40.0]
        got = xprof.chain_stats([settled] * 7 + [drifting])
        self.assertAlmostEqual(got["settle_drift_us"], 30.0)

    def test_the_drift_keeps_its_sign(self) -> None:
        speeding = [40.0, 40.0, 40.0, 40.0, 10.0, 10.0, 10.0, 10.0]
        got = xprof.chain_stats([[10.0] * 8] * 7 + [speeding])
        self.assertAlmostEqual(got["settle_drift_us"], -30.0)


class OrphanedJobTests(unittest.TestCase):
    """A Job must be able to end without the coordinator that launched it.

    Measured: GHA cancelled a shard mid-run (run 31113422768), the cleanup trap did not
    survive the kill, and the Job was still Running at 159 minutes with both pods alive --
    holding the whole multi-host slice, with no coordinator left to harvest anything. Any
    later EP16 shard would have queued behind garbage. `ttlSecondsAfterFinished` cannot
    help: it deletes a Job that has FINISHED, and this one never would.
    """

    @staticmethod
    def _launcher() -> str:
        return (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()

    def test_the_job_carries_a_hard_deadline(self) -> None:
        text = self._launcher()
        self.assertTrue(_uncommented(text, "activeDeadlineSeconds: ${SHARD_DEADLINE}"))
        self.assertTrue(_uncommented(text, "ttlSecondsAfterFinished: 900"),
                        "the TTL still does the deletion once the deadline ends the Job")

    def test_the_deadline_covers_every_case_in_the_shard(self) -> None:
        """Per-case budget x cases, plus a case of slack. Firing early would kill work the
        in-pod script still considers live."""
        # The REAL assignment, sliced out of the launcher and evaluated -- restating the
        # arithmetic here would pass no matter what the launcher actually computes.
        text = self._launcher()
        line = next(l for l in text.splitlines()
                    if l.startswith('SHARD_DEADLINE="${COLLX_TPU_SHARD_DEADLINE:-'))
        out = subprocess.run(
            ["bash", "-c", f'RUN_TIMEOUT=2700\nEXPECTED_CASES=4\n{line}\n'
                           'echo "$SHARD_DEADLINE"'],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        deadline = int(out.stdout.strip())
        self.assertGreater(deadline, 2700 * 4,
                           "the deadline fires before the shard's own budget is spent")
        self.assertLessEqual(deadline, 2700 * 6, "an unbounded deadline strands the slice")

    def test_the_deadline_is_computed_after_the_case_count(self) -> None:
        text = self._launcher()
        self.assertLess(text.index("EXPECTED_CASES=\"$(python3"),
                        text.index("SHARD_DEADLINE=\"${COLLX_TPU_SHARD_DEADLINE:-"),
                        "the deadline would expand an empty EXPECTED_CASES")

    def test_a_malformed_deadline_is_refused(self) -> None:
        """An empty or non-numeric value would render as `activeDeadlineSeconds:` and the
        Job would be rejected at apply time, or worse, created without a bound."""
        self.assertTrue(_uncommented(
            self._launcher(),
            '[[ "$SHARD_DEADLINE" =~ ^[1-9][0-9]*$ ]] || collx_die'))


class ChainJsonSafetyTests(unittest.TestCase):
    """Everything the chain publishes must survive `json.dumps(allow_nan=False)`.

    The artifact is written at the very END of a case, so a numpy scalar anywhere in these
    fields kills the process after every timing pass has been paid for. Measured on run
    31147054940: both EP16 bf16 cases died with "Object of type float32 is not JSON
    serializable" -- `process_allgather` round-trips through JAX, which returns float32 with
    x64 disabled -- while the unchained fp8 cases wrote fine.
    """

    def test_gathered_periods_are_python_floats(self) -> None:
        import types as _t
        run_ep_jax._PROCESS_COUNT[0] = 2

        def process_allgather(value, tiled=False):
            arr = np.asarray(value)
            if arr.shape == (1,):
                return np.array([arr[0], arr[0]])
            # float32, exactly as JAX returns it with x64 disabled
            return np.stack([arr, arr]).astype(np.float32)

        utils = _t.ModuleType("jax.experimental.multihost_utils")
        utils.process_allgather = process_allgather
        experimental = _t.ModuleType("jax.experimental")
        experimental.multihost_utils = utils
        jax_mod = _t.ModuleType("jax")
        jax_mod.experimental = experimental
        try:
            with mock.patch.dict(sys.modules, {
                    "jax": jax_mod, "jax.experimental": experimental,
                    "jax.experimental.multihost_utils": utils}):
                got = run_ep_jax.gather_chain_periods(
                    {"per_device_periods": [[100.0, 101.0], [102.0, 103.0]]}, 16)
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1
        # The whole point: this raised TypeError before.
        json.dumps(got, allow_nan=False)
        for series in got.get("per_device_periods") or []:
            for value in series:
                self.assertIs(type(value), float, "a numpy scalar reached the artifact")

    def test_the_chain_row_is_serialisable(self) -> None:
        chain = {"availability": "measured", "origin": "chained-median",
                 "percentiles_us": {"p50": np.float32(10.0).item()}, "sample_count": 4,
                 "pair_spread_us": 1.0, "interpair_gap_us": 2.0, "settle_drift_us": -3.0,
                 "renorm_us": 0.5, "capture_s": 12.0, "capture_budget_s": 2700,
                 "devices": 16, "devices_expected": 16, "floors": {}}
        json.dumps(run_ep_jax._chain_fields(chain), allow_nan=False)


class BestTraceTests(unittest.TestCase):
    """Every chained measurement must read the SAME trace file, chosen by evidence.

    A profiler session emits several files and the marker's device rows land in one of
    them. `parse_trace_durations` has picked the file with the most matches for a while;
    the chained path used `find_trace()` -- the alphabetically FIRST file -- for the
    period, the floors, the gap and the anchors alike.

    Measured on run 31156743315: `floor_anchors` chose `ragged-all-to-all.2` as a floor
    anchor at T=512, an op absent from the chain scope's own published op inventory,
    because the inventory came from the best file and the anchor from the first. The two
    then disagreed about which device rows exist, which is what surfaced as the
    disjoint-device failure.
    """

    MARKER = "collectivex-chain-t8"

    def _dir(self, directory, files):
        """files: {basename: [(name, pid, ts, dur, marker), ...]}"""
        for base, rows in files.items():
            payload = [{"ph": "X", "pid": pid, "tid": 1, "ts": ts, "dur": dur, "name": name,
                        "args": {"tf_op": marker,
                                 "device_duration_ps": int(dur * 1e6)}}
                       for name, pid, ts, dur, marker in rows]
            with gzip.open(Path(directory) / base, "wt") as handle:
                json.dump({"traceEvents": payload}, handle)

    def test_the_file_with_the_marker_wins_over_the_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._dir(directory, {
                # alphabetically first, and holds nothing for this marker
                "aaa.json.gz": [("noise", 0, 0.0, 1.0, "other-scope")] * 3,
                "zzz.json.gz": [("ragged_all_to_all.8", 0, i * 100.0, 30.0, self.MARKER)
                                for i in range(4)],
            })
            path, matched = xprof.best_trace(directory, self.MARKER)
            self.assertTrue(path.endswith("zzz.json.gz"), path)
            self.assertEqual(len(matched), 4)

    def test_the_chained_family_agrees_on_the_file(self) -> None:
        """floor_anchors and chain_pair_gap must not resolve to different files."""
        rows = []
        for pid in (0, 1):
            for i in range(4):
                rows.append(("ragged_all_to_all.D", pid, i * 100.0, 30.0, self.MARKER))
                rows.append(("ragged_all_to_all.C", pid, i * 100.0 + 50.0, 20.0, self.MARKER))
        with tempfile.TemporaryDirectory() as directory:
            self._dir(directory, {
                "aaa.json.gz": [("ragged_all_to_all.2", 9, 0.0, 999.0, self.MARKER)],
                "zzz.json.gz": rows,
            })
            got = xprof.floor_anchors(directory, self.MARKER, 4)
        # The decoy in the first file out-totals everything and lives on a lone device row;
        # reading it would reproduce the measured failure exactly.
        self.assertTrue(got["ok"], got)
        self.assertEqual({got["dispatch"], got["combine"]},
                         {"ragged_all_to_all.D", "ragged_all_to_all.C"})

    def test_no_readable_trace_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, matched = xprof.best_trace(directory, self.MARKER)
            self.assertIsNone(path)
            self.assertEqual(matched, [])

    def test_the_chained_path_no_longer_uses_find_trace(self) -> None:
        source = Path(xprof.__file__).read_text()
        self.assertNotIn("trace = find_trace(trace_dir)", source,
                         "a chained measurement is back on the alphabetically-first file")


class ChainIdentityMultiHostTests(unittest.TestCase):
    """The verdict covers every host, not the one that happens to write the artifact.

    At EP16 this process sees 8 of 16 shards. A chain corrupted only on the other host's
    devices would score 0.0 here, and publishing that as the verdict would be exactly the
    half-slice basis this work removed from the period.
    """

    def _run(self, local_flags, remote_flags):
        import types as _t

        def process_allgather(value, tiled=False):
            return np.stack([np.asarray(value, dtype=np.float64),
                             np.asarray(remote_flags, dtype=np.float64)])

        utils = _t.ModuleType("jax.experimental.multihost_utils")
        utils.process_allgather = process_allgather
        experimental = _t.ModuleType("jax.experimental")
        experimental.multihost_utils = utils
        jax_mod = _t.ModuleType("jax")
        jax_mod.experimental = experimental
        jax_mod.device_get = lambda v: v

        x = np.asarray(local_flags["data"], dtype=np.float32)
        produced = ChainIdentityOracleTests._Array(local_flags["got"])
        point = types.SimpleNamespace(x=ChainIdentityOracleTests._Array(x))
        run_ep_jax._PROCESS_COUNT[0] = 2
        try:
            with mock.patch.dict(sys.modules, {
                    "jax": jax_mod, "jax.experimental": experimental,
                    "jax.experimental.multihost_utils": utils}):
                return run_ep_jax.chain_identity(jax_mod, produced, point)
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1

    def test_a_remote_failure_fails_the_verdict(self) -> None:
        clean = np.arange(24, dtype=np.float32).reshape(6, 4)
        got = self._run({"data": clean, "got": clean},
                        remote_flags=[1.0, 0.0, 0.75])   # evaluated, failed, max_rel .75
        self.assertIs(got["ok"], False,
                      "a chain corrupted on the other host was published as passing")
        self.assertAlmostEqual(got["max_rel"], 0.75)

    def test_both_hosts_clean_passes(self) -> None:
        clean = np.arange(24, dtype=np.float32).reshape(6, 4)
        got = self._run({"data": clean, "got": clean}, remote_flags=[1.0, 1.0, 0.0])
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["max_rel"], 0.0)

    def test_a_host_that_could_not_evaluate_makes_it_unevaluated(self) -> None:
        """Not a pass. One host with no verdict means the slice has no verdict."""
        clean = np.arange(24, dtype=np.float32).reshape(6, 4)
        got = self._run({"data": clean, "got": clean}, remote_flags=[0.0, 0.0, -1.0])
        self.assertIsNone(got["ok"])
        self.assertIn("unevaluated on 1 of 2 host", got["reason"])


class MultiHostContractTests(unittest.TestCase):
    """Offline coverage for the code that only ever runs on two hosts.

    Every EP16 defect in this work cost a full hardware cycle because nothing offline
    exercised the multi-host path: a stranded slice, then `process_allgather` returning
    float32 and killing `json.dumps`, then `device_get` raising on a global array and
    condemning 10 good rows. Three cycles, three APIs that behave differently with two
    processes, none reachable from a single-process test.

    This fixture makes the two-host contract hostile in exactly the ways the hardware is:

      * a global array RAISES on `device_get`, and only `.addressable_shards` can be read
      * `process_allgather` returns float32, as JAX does with x64 disabled

    Anything that violates either now fails here rather than two hours into a sweep.
    """

    class _Global:
        """A jax.Array spanning devices this process cannot address."""

        def __init__(self, data, hosts=2, host=0, per_host=2):
            blocks = np.array_split(np.asarray(data), hosts * per_host)
            mine = blocks[host * per_host:(host + 1) * per_host]
            self.addressable_shards = [
                types.SimpleNamespace(data=b,
                                      device=types.SimpleNamespace(id=host * per_host + i))
                for i, b in enumerate(mine)]

    @staticmethod
    def _jax():
        def device_get(value):
            if isinstance(value, MultiHostContractTests._Global):
                raise RuntimeError(
                    "Fetching value for `jax.Array` that spans non-addressable "
                    "(non process local) devices is not possible.")
            return value
        return types.SimpleNamespace(device_get=device_get)

    @staticmethod
    def _multihost(hosts=2):
        import types as _t

        def process_allgather(value, tiled=False):
            arr = np.asarray(value)
            stacked = np.stack([arr] * hosts)
            # float32, exactly as JAX returns it with x64 disabled
            return stacked.astype(np.float32) if stacked.dtype.kind == "f" else stacked

        utils = _t.ModuleType("jax.experimental.multihost_utils")
        utils.process_allgather = process_allgather
        experimental = _t.ModuleType("jax.experimental")
        experimental.multihost_utils = utils
        jax_mod = _t.ModuleType("jax")
        jax_mod.experimental = experimental
        return {"jax": jax_mod, "jax.experimental": experimental,
                "jax.experimental.multihost_utils": utils}

    def test_the_whole_chained_block_survives_two_hosts(self) -> None:
        """The composition, not the pieces: oracle -> gather -> fields -> json.

        Both EP16 crashes were in this composition and neither was in a piece that lacked
        a test. `chain_identity` had four; `gather_chain_periods` had three.
        """
        x = np.arange(64, dtype=np.float32).reshape(16, 4)
        run_ep_jax._PROCESS_COUNT[0] = 2
        try:
            with mock.patch.dict(sys.modules, self._multihost()):
                identity = run_ep_jax.chain_identity(
                    self._jax(), self._Global(x), types.SimpleNamespace(x=self._Global(x)))
                chain = run_ep_jax.gather_chain_periods(
                    {"availability": "measured", "origin": "chained-median",
                     "percentiles_us": {"p50": 10.0}, "sample_count": 4, "floors": {},
                     "per_device_periods": [[10.0, 10.5], [10.2, 10.4]]}, 16)
        finally:
            run_ep_jax._PROCESS_COUNT[0] = 1
        self.assertTrue(identity["ok"], identity)
        self.assertEqual(chain["devices_expected"], 16)
        self.assertTrue(chain["gathered_across_hosts"])
        fields = run_ep_jax._chain_fields(chain)
        # The float32 crash was here, at the very end of a case.
        json.dumps({**fields, "correctness": run_ep_jax._correctness(0.0, True, identity)},
                   allow_nan=False)

    def test_reading_the_global_array_directly_is_the_failure_being_prevented(self) -> None:
        """The fixture must actually bite -- otherwise the test above proves nothing."""
        with self.assertRaises(RuntimeError):
            self._jax().device_get(self._Global(np.ones((8, 2), np.float32)))

    def test_the_oracle_never_touches_the_global_array(self) -> None:
        source = Path(run_ep_jax.__file__).read_text()
        block = source[source.index("def _local_rows("):source.index("def _chain_anchor(")]
        self.assertIn("array.addressable_shards", block)
        self.assertNotIn("jax.device_get(produced)", block)
        self.assertNotIn("jax.device_get(point.x)", block)


class ChainedDocsTests(unittest.TestCase):
    """The README's chained section is a consumer contract; pin it to the code.

    It went unwritten through 61 commits while every field in it was being changed, which
    is how `--transport-impl padded` -- a flag that never existed, documented as doing the
    opposite of what the code did -- got into an earlier revision of this same file.
    """

    @staticmethod
    def _readme(name="README.md"):
        return (COLLX / name).read_text()

    def test_both_languages_carry_the_section(self) -> None:
        self.assertIn("### The chained pair period", self._readme())
        self.assertIn("### 链式 pair period", self._readme("README_zh.md"))

    def test_the_default_iteration_count_is_the_documented_one(self) -> None:
        source = Path(ep_harness.__file__).read_text()
        self.assertIn('"--chain-iters", type=int, default=128', source)
        self.assertIn("default 128", self._readme())
        self.assertIn("默认 128", self._readme("README_zh.md"))

    def test_the_tri_state_is_documented_in_both(self) -> None:
        """Documenting it as a boolean would invite a consumer to read null as false."""
        self.assertIn("Tri-state", self._readme())
        self.assertIn("三态", self._readme("README_zh.md"))

    def test_the_documented_origins_are_the_published_ones(self) -> None:
        source = Path(xprof.__file__).read_text()
        for origin in ("chained-median", "chained-cross-rank-min"):
            self.assertIn(f'"origin": "{origin}"', source)
        self.assertIn("chained-cross-rank-min", self._readme())

    def test_renorm_is_documented_as_unavailable_not_free(self) -> None:
        """It is bounded by differencing, not measured. Saying otherwise would publish a
        number nobody has."""
        readme = self._readme()
        self.assertIn("renorm_us", readme)
        window = readme[readme.index("| `renorm_us` |"):][:400]
        self.assertIn("unavailable", window)

    def test_the_fp8_exclusion_is_documented_with_its_reason(self) -> None:
        source = Path(run_ep_jax.__file__).read_text()
        self.assertIn('"chained_period": precision != "fp8"', source)
        self.assertIn("FP8 is deliberately not chained", self._readme())

    def test_the_documented_anchor_rule_is_the_one_the_code_runs(self) -> None:
        """The selection rule was replaced and the prose was not.

        Both READMEs described anchors chosen by row-set SIZE for three commits after
        that rule had been measured WORSE than what it replaced (2 of 10 points against
        5 of 10, run 31180043680) and swapped for busiest-pairable-group. Every other
        test here passed throughout: none of them read this sentence. A consumer porting
        to a new SKU would have implemented the rejected rule.
        """
        source = Path(xprof.__file__).read_text()
        self.assertIn("if len(members) >= 2}", source,
                      "the code selects among groups that can pair at all")
        self.assertIn("max(eligible, key=", source,
                      "and takes the busiest of those, not the widest")
        for name in ("README.md", "README_zh.md"):
            readme = self._readme(name)
            self.assertNotIn("widest device row set", readme, name)
            self.assertNotIn("最宽的", readme, name)
            self.assertIn("row", readme.lower(), name)


class UnchainedReasonTests(unittest.TestCase):
    """An unchained row must say WHY, not just that nothing was captured.

    Measured on run 31167366589, EP16 fp8: `pair_period.reason` read "not captured", which
    a consumer cannot tell apart from a capture that failed -- while both READMEs promise
    the fp8 exclusion travels with the row. The artifact and the documentation disagreed.
    """

    def test_fp8_names_the_contract_reason(self) -> None:
        source = Path(run_ep_jax.__file__).read_text()
        block = source[source.index("chain_call = point.chain("):][:900]
        self.assertIn("fp8 is not chained", block)
        self.assertIn("point.t.fp8", block)
        self.assertIn("native", block, "the reason must name the contract it follows from")

    def test_a_disabled_chain_says_so_separately(self) -> None:
        """`--chain-iters 0` is an operator choice, not a property of the backend."""
        source = Path(run_ep_jax.__file__).read_text()
        block = source[source.index("chain_call = point.chain("):][:900]
        self.assertIn("chaining disabled", block)

    def test_the_reason_reaches_pair_period(self) -> None:
        got = run_ep_jax._chain_fields(
            {"availability": "unavailable", "reason": "fp8 is not chained: ..."})
        self.assertEqual(got["pair_period"]["availability"], "unavailable")
        self.assertIn("fp8 is not chained", got["pair_period"]["reason"])

    def test_the_readme_claim_is_the_artifact_behaviour(self) -> None:
        """The doc says the fields are unavailable WITH that reason; pin both ends."""
        readme = (COLLX / "README.md").read_text()
        self.assertIn("`unavailable` with that reason", readme)
        self.assertIn("not silently absent", readme)


class AnchorRowSetTests(unittest.TestCase):
    """Both anchors must come from ONE device row set, and which set matters.

    tpu7x logs several core kinds -- the chain scope's own inventory carries a
    `sparse-core-data-format-call...` beside the TensorCore collectives -- so the top two
    ops by duration routinely live on different rows and cannot be the two directions of
    one pair.

    Two rules were tried and measured before this one:

      * widest row SET: a `...cloned.1.call-done` and a bare `ragged-all-to-all.N` have
        equal-sized but DISJOINT sets, so both survived. Floors fell to 2 of 10 points
        (run 31180043680), worse than the 5 of 10 the trace-file fix alone had given.
      * busiest row set: a single large op alone on a sparse-core row wins on total, and
        the point then fails with one candidate.

    The rule that holds: group candidates by row set, keep only groups that can form a
    pair AT ALL (>= 2 distinct ops), and take the busiest of those. The two anchors are
    then pairable by construction rather than by luck.
    """

    MARKER = "collectivex-chain-t8"

    def _anchors(self, rows, iters):
        events = [{"ph": "X", "pid": pid, "tid": 1, "ts": ts, "dur": dur, "name": name,
                   "args": {"tf_op": self.MARKER, "device_duration_ps": int(dur * 1e6)}}
                  for name, pid, ts, dur in rows]
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "t.json.gz", "wt") as handle:
                json.dump({"traceEvents": events}, handle)
            return xprof.floor_anchors(directory, self.MARKER, iters)

    @staticmethod
    def _pair(pids, iters=4, a="ragged-all-to-all.3.cloned.1.call-done",
              b="ragged-all-to-all.5.cloned.1.call-done", da=40.0, db=30.0):
        return [(n, pid, i * 100.0 + off, dur)
                for pid in pids for i in range(iters)
                for n, off, dur in ((a, 0.0, da), (b, 50.0, db))]

    def test_a_genuine_pair_passes_with_its_phase(self) -> None:
        got = self._anchors(self._pair((0, 1)), 4)
        self.assertTrue(got["ok"], got)
        self.assertAlmostEqual(got["phase"], 0.5, places=6)

    def test_a_lone_giant_on_its_own_row_cannot_win(self) -> None:
        """100x the total of either real anchor, and unpairable, so it must not be
        selected -- the failure mode of choosing the busiest row set outright."""
        rows = self._pair((0, 1, 2, 3))
        rows += [("ragged-all-to-all.2", 9, i * 100.0 + 25.0, 40000.0) for i in range(4)]
        got = self._anchors(rows, 4)
        self.assertTrue(got["ok"], got)
        self.assertEqual({got["dispatch"], got["combine"]},
                         {"ragged-all-to-all.3.cloned.1.call-done",
                          "ragged-all-to-all.5.cloned.1.call-done"})

    def test_equal_sized_disjoint_rows_fail_closed(self) -> None:
        """The measured shape: call-done on one core's rows, the bare op on another's.
        Same cardinality, so a widest-set rule keeps both and pairs them wrongly."""
        rows = [("ragged-all-to-all.3.cloned.1.call-done", 0, i * 100.0, 40.0)
                for i in range(4)]
        rows += [("ragged-all-to-all.4", 1, i * 100.0 + 50.0, 30.0) for i in range(4)]
        got = self._anchors(rows, 4)
        self.assertFalse(got["ok"], got)
        self.assertIn("busiest pairable device row set", got["reason"])
        # ... and the reason must still NAME what it saw, or it is not a diagnosis.
        self.assertIn("ragged-all-to-all.3.cloned.1.call-done", got["reason"])
        self.assertIn("ragged-all-to-all.4", got["reason"])

    def test_one_shared_and_one_lonely_row_fails_closed(self) -> None:
        rows = [("ragged-all-to-all.3.cloned.1.call-done", pid, i * 100.0, 40.0)
                for pid in (0, 1) for i in range(4)]
        rows += [("ragged-all-to-all.4", 0, i * 100.0 + 50.0, 30.0) for i in range(4)]
        got = self._anchors(rows, 4)
        self.assertFalse(got["ok"], got)

    def test_the_busier_of_two_pairable_groups_wins(self) -> None:
        """Both groups can pair; the one carrying the collective work is the right one."""
        # Both decoys must survive the collective-family filter, or the test proves
        # nothing about group SELECTION -- the first version used `sparse-core-*` names,
        # which are dropped before grouping, so it passed with the choice reversed.
        rows = self._pair((0, 1))                                   # 2*(40+30)*4 = 560
        rows += self._pair((7, 8), a="ragged-all-to-all.9.cloned.1.call-done",
                           b="ragged-all-to-all.10", da=5.0, db=4.0)   # 2*(5+4)*4 =  72
        got = self._anchors(rows, 4)
        self.assertTrue(got["ok"], got)
        self.assertEqual({got["dispatch"], got["combine"]},
                         {"ragged-all-to-all.3.cloned.1.call-done",
                          "ragged-all-to-all.5.cloned.1.call-done"},
                         "the quieter row set was selected")


class FloorFailureDiagnosisTests(unittest.TestCase):
    """A rejected floors point must diagnose itself.

    Three anchor-selection rules were tried against this hardware and each rejection cost a
    run to interpret, because the artifact named the ops it saw but never said how they
    were distributed over device rows -- the single fact the decision turns on. The second
    rule was chosen from a hypothesis and made coverage WORSE (5 of 10 points to 2 of 10).
    """

    MARKER = "collectivex-chain-t8"

    def _reject(self):
        rows = [("ragged-all-to-all.3.cloned.1.call-done", 0, i * 100.0, 40.0)
                for i in range(4)]
        rows += [("ragged-all-to-all.4", 1, i * 100.0 + 50.0, 30.0) for i in range(4)]
        events = [{"ph": "X", "pid": pid, "tid": 1, "ts": ts, "dur": dur, "name": name,
                   "args": {"tf_op": self.MARKER, "device_duration_ps": int(dur * 1e6)}}
                  for name, pid, ts, dur in rows]
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "t.json.gz", "wt") as handle:
                json.dump({"traceEvents": events}, handle)
            return xprof.floor_anchors(directory, self.MARKER, 4)

    def test_the_rejection_carries_the_row_grouping(self) -> None:
        got = self._reject()
        self.assertFalse(got["ok"])
        groups = got["row_groups"]
        self.assertEqual(len(groups), 2, groups)
        self.assertEqual({tuple(g["rows"]) for g in groups}, {("0",), ("1",)})
        self.assertEqual({g["ops"][0] for g in groups},
                         {"ragged-all-to-all.3.cloned.1.call-done", "ragged-all-to-all.4"})

    def test_the_groups_are_ordered_by_device_time(self) -> None:
        """The QUIETEST group appears first in the trace, so an unsorted list would come
        back ascending. The shared fixture happens to be descending already, which is why
        this needs its own -- a sort test whose input is pre-sorted tests nothing."""
        rows = [("ragged-all-to-all.3.cloned.1.call-done", 0, i * 100.0, 10.0)
                for i in range(4)]                                    # first seen, 40us
        rows += [("ragged-all-to-all.4", 1, i * 100.0 + 50.0, 90.0)
                 for i in range(4)]                                   # later, 360us
        events = [{"ph": "X", "pid": pid, "tid": 1, "ts": ts, "dur": dur, "name": name,
                   "args": {"tf_op": self.MARKER, "device_duration_ps": int(dur * 1e6)}}
                  for name, pid, ts, dur in rows]
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "t.json.gz", "wt") as handle:
                json.dump({"traceEvents": events}, handle)
            groups = xprof.floor_anchors(directory, self.MARKER, 4)["row_groups"]
        self.assertEqual([g["total_us"] for g in groups], [360.0, 40.0])

    def test_it_reaches_the_artifact(self) -> None:
        source = Path(run_ep_jax.__file__).read_text()
        self.assertIn('"row_groups": split.get("row_groups")', source)

    def test_it_is_json_safe(self) -> None:
        """Device ids are stringified: a numpy or non-str pid would raise at the write."""
        json.dumps(self._reject()["row_groups"], allow_nan=False)
        for group in self._reject()["row_groups"]:
            for row in group["rows"]:
                self.assertIs(type(row), str)


class PhaseFloorCorroborationTests(unittest.TestCase):
    """The documented cross-check must hold on synthetic data too.

    `phase` comes from start timestamps and the dispatch floor from op durations. If the
    anchors are a real back-to-back pair, `phase` tracks `floor_dispatch / period`. Nothing
    in the code enforces the relationship -- which is exactly why agreement on hardware is
    evidence, rather than a tautology.
    """

    MARKER = "collectivex-chain-t8"

    def test_the_ratio_and_the_phase_agree_by_construction(self) -> None:
        period, dispatch_dur = 1000.0, 300.0
        rows = []
        for pid in (0, 1):
            for i in range(6):
                base = i * period
                rows.append(("ragged-all-to-all.3.cloned.1.call-done", pid, base, dispatch_dur))
                # combine starts exactly when dispatch's collective ends
                rows.append(("ragged-all-to-all.5.cloned.1.call-done", pid,
                             base + dispatch_dur, 280.0))
        events = [{"ph": "X", "pid": pid, "tid": 1, "ts": ts, "dur": dur, "name": name,
                   "args": {"tf_op": self.MARKER, "device_duration_ps": int(dur * 1e6)}}
                  for name, pid, ts, dur in rows]
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "t.json.gz", "wt") as handle:
                json.dump({"traceEvents": events}, handle)
            split = xprof.floor_anchors(directory, self.MARKER, 6)
            floors = xprof.chain_floors(directory, self.MARKER, split["dispatch"],
                                        drop=1, period_us=period)
        self.assertTrue(split["ok"], split)
        self.assertAlmostEqual(split["phase"], dispatch_dur / period, places=6)
        self.assertAlmostEqual(floors["percentiles_us"]["p50"] / period,
                               split["phase"], places=6)

    def test_the_documented_table_is_in_both_readmes(self) -> None:
        for name in ("README.md", "README_zh.md"):
            text = (COLLX / name).read_text()
            self.assertIn("floor_dispatch / period", text, name)
            self.assertIn("31194062843", text, name)
