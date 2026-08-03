#!/usr/bin/env python3
"""Run every TPU case in one Python process.

The TPU runtime may not initialize cleanly in a second OS process on the same
allocated node. Keeping the shard in one interpreter lets JAX reuse its initialized
backend and compilation cache across decode and prefill while preserving the neutral
per-case argv and result documents.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COLLX = HERE.parent
sys.path[:0] = [str(HERE), str(COLLX / "runtime")]

import config  # noqa: E402
import run_ep_jax  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one TPU CollectiveX shard")
    parser.add_argument("--shard", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--ngpus", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--gpus-per-node", required=True)
    parser.add_argument("--scale-up-domain", required=True)
    # Device-side timing overlay; off by default because it captures a trace.
    parser.add_argument("--xprof", action="store_true")
    parser.add_argument("--xprof-iters", default="20")
    args = parser.parse_args()

    document = config.load(args.shard)
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        print("ERROR: TPU shard has no cases", file=sys.stderr)
        return 2

    failed = 0
    original_argv = sys.argv
    original_attempt = os.environ.get("COLLX_ATTEMPT_ID")
    try:
        for index in range(len(cases)):
            case_argv = config.case_argv(
                args.shard,
                index,
                args.runner,
                args.timestamp,
                args.ngpus,
                args.nodes,
                args.gpus_per_node,
                args.scale_up_domain,
            )
            print(
                f"[collectivex] EP{args.ngpus}[{index + 1}/{len(cases)}] "
                f"{cases[index].get('backend', 'unknown')}",
                flush=True,
            )
            os.environ["COLLX_ATTEMPT_ID"] = str(index + 1)
            sys.argv = [
                str(HERE / "run_ep_jax.py"),
                *(["--xprof", "--xprof-iters", str(args.xprof_iters)]
                  if args.xprof else []),
                *case_argv,
            ]
            try:
                rc = run_ep_jax.main()
            except Exception as exc:  # the launcher emits a terminal artifact below
                print(f"ERROR: TPU case {index} raised {exc!r}", file=sys.stderr)
                rc = 1
            if rc != 0:
                failed += 1
    finally:
        sys.argv = original_argv
        if original_attempt is None:
            os.environ.pop("COLLX_ATTEMPT_ID", None)
        else:
            os.environ["COLLX_ATTEMPT_ID"] = original_attempt

    if failed:
        print(f"ERROR: {failed}/{len(cases)} TPU case(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
