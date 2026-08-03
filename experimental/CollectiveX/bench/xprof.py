#!/usr/bin/env python3
"""Read device-side operation durations out of a JAX/TPU profiler trace.

Why this exists: JAX has no inline event timer (no `torch.cuda.Event`), so the probe's
primary latencies are host wall-clock and carry a per-call dispatch floor. The profiler
DOES record true on-device durations per operation occurrence, and reading them back is
cheap -- the trace is already a gzipped Chrome-trace JSON, so this is stdlib only: no
protobuf, no xplane reader, no tensorboard/xprof dependency.

Modelled on AI-Hypercomputer/accelerator-microbenchmarks
(`core/profiler.py: parse_xprof_durations`), the reference implementation for TPU
collective benchmarking, with three deliberate differences:

  * CROSS-DEVICE REDUCTION. The reference keeps only `min(pid)` -- one device, in
    practice TPU:0 -- and discards the rest. A collective finishes when its SLOWEST
    participant finishes, so per-device skew is exactly the quantity of interest. This
    reduces with MAX across devices, the same reduction the GPU harness applies across
    ranks (`ep_harness.run_sweep`'s per-iteration `ReduceOp.MAX`).
  * ALL TRACE FILES SEARCHED, choosing the one with the most matching events,
    rather than whichever file `os.walk` yields first. A session can emit several
    and the device rows do not always land in the first by name.
  * MICROSECONDS, and mismatched occurrence counts reported rather than truncated.

TWO CORRECTNESS RULES, both learned from getting them wrong on hardware:

  1. A marker matches a SEGMENT, not a substring. Scope labels end in the ladder point
     (`...-t1`, `...-t16`), and `-t1` is a substring of `-t16`, `-t128`, `-t1024`. A
     substring match silently pools four ladder points into one series -- observed as
     T=1 and T=16 reporting byte-identical durations. A marker therefore only matches
     when the character following it is not a digit.
  2. A SCOPE IS NOT AN OPERATION. One scope contains several HLO ops per iteration
     (roundtrip holds two collectives plus fusions), so its events cannot be zipped
     positionally as though element k were iteration k -- that yields a median over
     unrelated ops, observed as roundtrip reporting the same duration as dispatch when
     it must be roughly double. A component's device time is therefore the SUM of every
     matched op, reduced with max across devices, divided by the iteration count the
     caller performed. Positional zipping is valid only within a single HLO, where the
     occurrences really are homogeneous.
"""
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict


def find_traces(trace_dir: str) -> list:
    """Every gzipped Chrome-trace JSON under the directory, in a stable order."""
    candidates = []
    for root, _, files in os.walk(trace_dir):
        candidates.extend(
            os.path.join(root, name) for name in files if name.endswith(".json.gz")
        )
    return sorted(candidates)


def find_trace(trace_dir: str) -> str | None:
    """The first trace file, for diagnostics that do not care which one."""
    candidates = find_traces(trace_dir)
    return candidates[0] if candidates else None


def load_events(trace_path: str) -> list[dict]:
    with open(trace_path, "rb") as handle:
        with gzip.GzipFile(fileobj=handle) as stream:
            document = json.loads(stream.read())
    return document.get("traceEvents", []) or []


def duration_us(event: dict) -> float | None:
    """Device-side duration in microseconds, preferring the device counter.

    `device_duration_ps` is what the TPU profiler attaches to an XLA op and is the true
    on-device time. `dur` is the event's wall duration in microseconds, the fallback when
    the device counter is absent.
    """
    picoseconds = (event.get("args") or {}).get("device_duration_ps")
    if picoseconds:
        try:
            return float(picoseconds) / 1e6
        except (TypeError, ValueError):
            return None
    if "dur" in event:
        try:
            return float(event["dur"])
        except (TypeError, ValueError):
            return None
    return None


def marker_in(text: str, marker: str) -> bool:
    """Whether `marker` appears in `text` as a label rather than a numeric prefix.

    Rule 1 above. `collectivex-dispatch-t1` must not match
    `collectivex-dispatch-t16/ragged-all-to-all`, but must still match when the profiler
    appends a separator or a uniquifying suffix (`-t1/...`, `-t1.2`, `-t1_1`). So the
    match is rejected only when the very next character is a digit.
    """
    if not marker:
        return False
    index = text.find(marker)
    while index != -1:
        end = index + len(marker)
        if end == len(text) or not text[end].isdigit():
            return True
        index = text.find(marker, index + 1)
    return False


def _matches(event: dict, needle: str) -> bool:
    args = event.get("args") or {}
    return marker_in(str(args.get("tf_op", "")), needle) or marker_in(
        str(event.get("name", "")), needle
    )


def _by_device(events: list[dict]) -> dict:
    grouped: dict[object, list[dict]] = defaultdict(list)
    for event in events:
        if duration_us(event) is not None:
            grouped[event.get("pid")].append(event)
    return grouped


def op_inventory(events: list[dict], occurrences: int, limit: int = 8) -> list:
    """Per-iteration device time by OP NAME on the slowest device, biggest first.

    `transport_us` attributes only ops whose name carries the collective's HLO name, and
    the remainder is published as `non_transport_us`. Reading that remainder as "the
    permute" turned out to be wrong: in the reference program, whose only source-level
    operation IS the collective, the named ops account for just half the device time
    (T=8192: 6,684us named against a 13,554us span, and the 0.624GB output buffer is far
    too small to explain the rest at 91GB/s implied). So the remainder must be itemised
    rather than described, or a reader will attribute the collective's own unnamed
    lowering to surrounding work.

    Grouped on the slowest device alone, matching `sum_across_devices`, so the numbers
    here add up to `op_total_device_us` instead of mixing devices.
    """
    grouped = _by_device(events)
    if not grouped or occurrences <= 0:
        return []
    totals = {
        device: sum(duration_us(event) for event in items)
        for device, items in grouped.items()
    }
    slowest = max(totals, key=lambda key: totals[key])
    by_name: dict = defaultdict(lambda: [0.0, 0])
    for event in grouped[slowest]:
        entry = by_name[str(event.get("name", ""))[:64]]
        entry[0] += duration_us(event)
        entry[1] += 1
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:limit]
    return [{"name": name, "per_iteration_us": total / occurrences,
             "events_per_iteration": count / occurrences}
            for name, (total, count) in ranked]


def sum_across_devices(events: list[dict], occurrences: int) -> dict:
    """Total device time attributable to a set of ops, per iteration.

    Rule 2 above. Sums every matched op's duration WITHIN a device -- so a scope holding
    two collectives and three fusions contributes all five -- then takes the max across
    devices, because the component is done when the slowest device is done. Dividing by
    the caller's iteration count gives mean device microseconds per iteration.
    """
    grouped = _by_device(events)
    if not grouped:
        return {"devices": 0, "matched_events": 0, "total_device_us": None,
                "per_iteration_us": None}
    totals = {
        device: sum(duration_us(event) for event in items)
        for device, items in grouped.items()
    }
    slowest = max(totals.values())
    return {
        "devices": len(grouped),
        "matched_events": sum(len(items) for items in grouped.values()),
        "total_device_us": slowest,
        "per_iteration_us": (slowest / occurrences) if occurrences > 0 else None,
        "slowest_device": str(max(totals, key=lambda key: totals[key])),
    }


def _end_us(event: dict) -> float | None:
    """Where the event finishes on the device timeline, in trace microseconds.

    Spans need placement on the timeline, so this uses `ts` + `dur` rather than the
    device counter: `device_duration_ps` is how long the op ran, not where it sat.
    """
    start = event.get("ts")
    if start is None:
        return None
    length = event.get("dur")
    if length is None:
        picoseconds = (event.get("args") or {}).get("device_duration_ps")
        length = (float(picoseconds) / 1e6) if picoseconds else 0.0
    try:
        return float(start) + float(length)
    except (TypeError, ValueError):
        return None


def spans_across_devices(events: list[dict], occurrences: int) -> dict:
    """Per-occurrence device SPANS: first op's start to last op's end, max across devices.

    This is the CUDA-event equivalent, and it is what makes a TPU latency comparable to
    the GPU SKUs. `torch.cuda.Event` brackets a REGION of the stream, so the GPU number
    includes any idle gaps between kernels inside it. Summing op durations
    (`sum_across_devices`) silently omits those gaps -- about 10% at the largest ladder
    point -- and would flatter TPU against a GPU number measured the other way.

    Events on one device are ordered by timestamp and split into `occurrences` equal
    chunks: every occurrence runs the same program, so each chunk is one execution of it.
    The slowest device's span wins, mirroring the GPU harness's per-iteration MAX across
    ranks.
    """
    grouped = _by_device(events)
    if not grouped or occurrences <= 0:
        return {"per_occurrence_us": [], "devices": 0, "uneven_devices": 0}
    series, uneven = [], 0
    for device in sorted(grouped, key=str):
        items = sorted(grouped[device], key=lambda item: item.get("ts", 0))
        per_occurrence = len(items) // occurrences
        if per_occurrence == 0:
            continue
        if len(items) % occurrences:
            # The op count should divide evenly; report rather than hide a mismatch.
            uneven += 1
        spans = []
        for index in range(occurrences):
            chunk = items[index * per_occurrence:(index + 1) * per_occurrence]
            starts = [item.get("ts") for item in chunk if item.get("ts") is not None]
            ends = [end for end in (_end_us(item) for item in chunk) if end is not None]
            if starts and ends:
                spans.append(max(ends) - min(starts))
        if spans:
            series.append(spans)
    if not series:
        return {"per_occurrence_us": [], "devices": 0, "uneven_devices": uneven}
    width = min(len(values) for values in series)
    return {
        "per_occurrence_us": [
            max(values[index] for values in series) for index in range(width)
        ],
        "devices": len(series),
        "uneven_devices": uneven,
    }


def reduce_across_devices(events: list[dict]) -> dict:
    """Per-occurrence MAX across devices, valid ONLY for a single homogeneous op.

    Events are grouped by `pid`, ordered by timestamp within a device, and zipped
    positionally: occurrence k on every device is the same instance of the same op. Do
    not use this across a whole scope -- see rule 2.
    """
    grouped = _by_device(events)
    if not grouped:
        return {"per_occurrence_us": [], "devices": 0, "dropped_occurrences": 0}
    series = [
        [duration_us(item)
         for item in sorted(grouped[device], key=lambda e: e.get("ts", 0))]
        for device in sorted(grouped, key=str)
    ]
    width = min(len(values) for values in series)
    dropped = sum(len(values) - width for values in series)
    return {
        "per_occurrence_us": [
            max(values[index] for values in series) for index in range(width)
        ],
        "devices": len(series),
        "dropped_occurrences": dropped,
    }


SCOPE_PREFIX = "collectivex"


def _describe(events: list[dict], trace_dir: str) -> dict:
    """What a trace actually holds, for when an expected marker is absent.

    Reports every trace file present (so a multi-file capture is visible), how many
    events carry a device duration, and the distinct CollectiveX scope labels found --
    the three things that separate "the scope was never recorded", "it is there under a
    different name", and "this trace has no device rows".
    """
    scopes, devices = set(), set()
    names: dict[str, int] = defaultdict(int)
    collectives = 0
    with_duration = 0
    for event in events:
        timed = duration_us(event) is not None
        if timed:
            with_duration += 1
            devices.add(str(event.get("pid")))
            # Which op names ARE present. When scope metadata is missing the HLO's own
            # name may still be there, which decides whether matching can fall back to it.
            names[str(event.get("name", ""))[:48]] += 1
            if "all-to-all" in str(event.get("name", "")):
                collectives += 1
        for field in (str((event.get("args") or {}).get("tf_op", "")),
                      str(event.get("name", ""))):
            index = field.find(SCOPE_PREFIX)
            if index != -1:
                scopes.add(field[index:index + 48].split("/")[0])
    files = [os.path.basename(path) for path in find_traces(trace_dir)]
    return {
        "trace_files": sorted(files),
        "events_with_duration": with_duration,
        "devices_seen": len(devices),
        "collectivex_scopes_present": sorted(scopes)[:24],
        # If scopes are absent but these are not, matching can fall back to the HLO name.
        "all_to_all_events": collectives,
        "top_event_names": sorted(names.items(), key=lambda kv: -kv[1])[:12],
    }


def parse_trace_durations(trace_dir: str, marker: str, occurrences: int,
                          hlo_substrings: tuple = ()) -> dict:
    """Device time for one component, plus an optional per-HLO breakdown.

    `marker` is the `jax.named_scope` label wrapped around the component. `occurrences`
    is how many times the caller executed it -- needed because a component's device time
    is a sum over its ops, which only becomes a per-iteration latency once divided.

    `hlo_substrings` names individual HLO ops to break out. This is the reason the trace
    is worth reading for `combine`, whose host-timed latency fuses transport with a
    scatter-add: matching `ragged-all-to-all` alone separates them. Within a single HLO
    the occurrences are homogeneous, so those carry a per-occurrence distribution as well
    as a total.

    Never raises: a missing or malformed trace returns an `error` and no durations, so a
    failed capture degrades the measurement instead of failing the case.
    """
    # Search EVERY trace file, not just the first. A profiler session can emit more than
    # one, and picking the alphabetically-first one silently loses the marker whenever the
    # device rows land in another -- a plausible cause of the intermittent coverage seen
    # on this pool, where identical code covered 14/14 points on one run and 7/14 on the
    # next. Choosing the file with the most matches is deterministic and never worse.
    candidates = find_traces(trace_dir)
    if not candidates:
        return {"error": f"no .json.gz trace under {trace_dir}", "marker": marker}
    trace_path, events, matched, failures = None, [], [], []
    for path in candidates:
        try:
            found = load_events(path)
        except (OSError, ValueError, EOFError) as exc:
            failures.append(f"{os.path.basename(path)}: {exc!r}")
            continue
        hits = [event for event in found if _matches(event, marker)]
        if trace_path is None or len(hits) > len(matched):
            trace_path, events, matched = path, found, hits
    if trace_path is None:
        return {"error": f"no readable trace under {trace_dir}: {failures}",
                "marker": marker}
    spans = spans_across_devices(matched, occurrences)
    result = {
        "marker": marker,
        "trace": os.path.basename(trace_path),
        "traces_searched": len(candidates),
        "total_events": len(events),
        "occurrences": occurrences,
        # The headline: per-occurrence spans, the CUDA-event equivalent. Raw percentiles,
        # not the reference's IQR-filtered mean, because the GPU SKUs publish raw
        # percentiles and a filtered TPU number would not sit beside them honestly.
        "per_occurrence_us": spans["per_occurrence_us"],
        "percentiles_us": summarize(spans["per_occurrence_us"]),
        "span_devices": spans["devices"],
        "uneven_devices": spans["uneven_devices"],
        # Retained alongside: summed op time excludes intra-region gaps, so comparing it
        # against the span shows how much of the region was idle.
        **{f"op_{key}": value for key, value in
           sum_across_devices(matched, occurrences).items()},
        # What the summed op time is actually made of, by name. See op_inventory.
        "op_inventory": op_inventory(matched, occurrences),
    }
    if not matched:
        result["error"] = f"no {marker!r} events in {os.path.basename(trace_path)}"
        # Say what the trace DOES contain. Twice now a coverage failure has been
        # diagnosed by hypothesis and fixed wrongly; a miss should carry the evidence to
        # settle it -- whether the scope is absent, present under another name, or the
        # trace holds no device rows at all.
        result["diagnostic"] = _describe(events, trace_dir)

    breakdown = {}
    for needle in hlo_substrings:
        hits = [event for event in matched if _matches(event, needle)]
        if not hits:
            continue
        entry = sum_across_devices(hits, occurrences)
        aligned = reduce_across_devices(hits)
        # Positional zipping across devices assumes every device logged the same number
        # of events for this op. Observed in practice: 480 matched over 16 devices with
        # 160 dropped -- some devices emit one event per iteration and others two, so
        # element k on device A is a DIFFERENT iteration from element k on device B and
        # the resulting distribution is meaningless. The summed totals above do not
        # depend on that pairing, so they stay; the per-occurrence series does not.
        if aligned["dropped_occurrences"]:
            aligned["per_occurrence_unreliable"] = (
                f"{aligned['dropped_occurrences']} occurrence(s) dropped across "
                f"{aligned['devices']} devices: event counts differ per device, so a "
                f"positional pairing does not align iterations"
            )
            aligned.pop("per_occurrence_us", None)
        entry.update(aligned)
        breakdown[needle] = entry
    if breakdown:
        result["by_hlo"] = breakdown
    return result


def summarize(durations: list) -> dict | None:
    """p50/p90/p95/p99 over per-occurrence device durations, in the artifact's shape."""
    if not durations:
        return None
    import ep_harness  # local import: keeps this module stdlib-only for reuse

    return ep_harness._pcts(list(durations))
