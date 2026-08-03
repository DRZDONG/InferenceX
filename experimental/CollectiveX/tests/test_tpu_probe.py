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
import json
import os
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
                # Single-host slices only, for now: the node pool provides no ICI path
                # between hosts, so no cross-host EP degree may be marked runnable.
                for degrees in entry["backends"].values():
                    self.assertEqual(
                        set(degrees), {entry["gpus_per_node"]},
                        "cross-host EP needs a multi-host TPU slice",
                    )

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
        self.assertTrue(args.out.startswith("results/tpuv7_jax-ragged-a2a_bf16_decode_"))

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
        self.assertIn("| failed | n/a | n/a | n/a |", summary)

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
            self.assertEqual(item["case"]["precision"], "bf16")
            self.assertEqual(item["case"]["ep"], 8)
            self.assertEqual(item["case"]["scope"], "scale-up")
        # EP16 is recorded as an unsupported coverage row, not silently absent.
        unsupported = {
            item["case"]["ep"] for item in document["requested_cases"]
            if item["disposition"] == "unsupported"
        }
        self.assertEqual(unsupported, {16})
        for shard in document["include"]:
            self.assertEqual(shard["launcher"], "tpu-gke")

    def test_tpu_launcher_uses_one_jax_process_and_records_missing_cases(self) -> None:
        launcher = (COLLX / "launchers" / "launch_tpu-gke.sh").read_text()
        self.assertIn("python3 bench/run_ep_jax_shard.py", launcher)
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
        # The reference is DENSE all_to_all, so its HLO is `all-to-all`, NOT the
        # components' `ragged-all-to-all`. Using the components' marker here would match
        # nothing and silently drop the reference's transport breakdown.
        self.assertEqual(run_ep_jax.transport_key(run_ep_jax.REFERENCE, 8192),
                         run_ep_jax.ep_jax.REFERENCE_HLO[0])
        self.assertIn(run_ep_jax.ep_jax.REFERENCE_HLO[0], inner)
        self.assertNotEqual(run_ep_jax.ep_jax.REFERENCE_HLO[0],
                            run_ep_jax.ep_jax.TRACED_HLO[0])

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
