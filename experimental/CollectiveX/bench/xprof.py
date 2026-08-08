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
    """The first trace file, for diagnostics that do not care which one.

    Almost nothing should use this. A profiler session emits more than one file and the
    marker's device rows may land in any of them, so "first" is an alphabetical accident.
    Use `best_trace()` wherever the answer depends on the events found.
    """
    candidates = find_traces(trace_dir)
    return candidates[0] if candidates else None


def best_trace(trace_dir: str, marker: str) -> tuple:
    """The trace file holding the MOST events for `marker`, and those events.

    `parse_trace_durations` has selected this way for a while, for a measured reason: a
    session emits several files, the marker's device rows land in one of them, and taking
    the alphabetically-first silently loses them -- the suspected cause of coverage that
    was 14/14 on one run and 7/14 on the next.

    The chained path was never given the same treatment, and read `find_trace()` -- the
    FIRST file -- for the period, the floors, the gap and the anchors alike. Measured on
    run 31156743315: `floor_anchors` chose `ragged-all-to-all.2` as a floor anchor at
    T=512, an op that does not appear anywhere in the chain scope's own op inventory,
    because the inventory came from the best file and the anchor from the first one. The
    two then disagreed about which device rows exist, which is what the disjoint-device
    gate reported. Same input, two files, two answers.
    """
    chosen, events, matched = None, [], []
    for path in find_traces(trace_dir):
        try:
            found = load_events(path)
        except (OSError, ValueError, EOFError):
            continue
        hits = [event for event in found if _matches(event, marker)]
        if chosen is None or len(hits) > len(matched):
            chosen, events, matched = path, found, hits
    return chosen, matched


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
        entry.update(overlap_us(hits))
        breakdown[needle] = entry
    if breakdown:
        result["by_hlo"] = breakdown
    return result


def overlap_us(events: list[dict]) -> dict:
    """How much of a scope's op time is double-counted, i.e. nested or concurrent.

    `transport_us` SUMS every op under the transport scope, and at T=8192 combine that is
    `...call-done` (6690.6us) plus `ragged_all_to_all.6` (3284.6us) = 9975.2. That sum is
    only a transport time if the two ops do not overlap: if `call-done` were an enclosing
    async span, the inner collective would be counted twice and the published figure would
    be ~1.5x the truth. Durations alone cannot tell the two apart -- both scale exactly
    with payload -- so this measures it on the timeline instead of arguing it.

    Uses `ts`/`dur` for BOTH terms rather than `device_duration_ps`: placement and
    occupancy are different quantities, and mixing them would manufacture an overlap.
    Reported as the MAX across devices, the same rule the spans use.
    """
    grouped = _by_device(events)
    if not grouped:
        return {"overlap_us": None, "disjoint": None, "devices": 0}
    worst, worst_union = 0.0, None
    for items in grouped.values():
        spans = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur") or 0.0))
                       for e in items if e.get("ts") is not None)
        if not spans:
            continue
        total = sum(end - start for start, end in spans)
        union, (cur_start, cur_end) = 0.0, spans[0]
        for start, end in spans[1:]:
            if start > cur_end:
                union += cur_end - cur_start
                cur_start, cur_end = start, end
            else:
                cur_end = max(cur_end, end)
        union += cur_end - cur_start
        if total - union >= worst:
            worst, worst_union = total - union, union
    return {"overlap_us": worst, "union_us": worst_union,
            # A hair of jitter is not nesting; a nested op contributes its whole duration.
            "disjoint": bool(worst < 1.0), "devices": len(grouped)}


def summarize(durations: list) -> dict | None:
    """p50/p90/p95/p99 over per-occurrence device durations, in the artifact's shape."""
    if not durations:
        return None
    import ep_harness  # local import: keeps this module stdlib-only for reuse

    return ep_harness._pcts(list(durations))


#: Collective op-name families the chain anchor may use. BOTH spellings are required, and
#: that is measured, not defensive: at T=8192 (run 31008868325) bf16 dispatch/combine and fp8
#: combine all carry a hyphenated `ragged-all-to-all.2.cloned.1.call-done`, while **fp8
#: dispatch carries only underscored `ragged_all_to_all.38`/`.24` and no hyphenated op at
#: all**. Anchoring on the hyphenated name alone -- which `TRACED_HLO` uses -- finds zero
#: occurrences under fp8 dispatch, so the chained fields would go unavailable on every fp8
#: row while appearing to work everywhere else. Matching is on the family prefix, never the
#: loose substring "all-to-all", which is itself a substring of "ragged-all-to-all".
CHAIN_ANCHOR_FAMILIES = ("ragged_all_to_all", "ragged-all-to-all")


def _anchor_starts(events: list[dict]) -> dict:
    """Per-device START timestamps of the anchor op, in occurrence order."""
    grouped: dict[object, list[float]] = defaultdict(list)
    for event in events:
        start = event.get("ts")
        if start is None or duration_us(event) is None:
            continue
        grouped[event.get("pid")].append(float(start))
    return {device: sorted(starts) for device, starts in grouped.items()}


def parse_chain(trace_dir: str, chain_marker: str, anchor: str, iters: int,
                drop: int = 1) -> dict:
    """Pair PERIOD and per-op FLOORS from one free-running chain capture.

    The period is a RATE, so it reduces across devices with MEDIAN and publishes (max - min)
    as the spread -- deliberately NOT the MAX this module uses everywhere else. A lockstep
    program makes every device agree on the pipeline's speed; taking the MAX would publish
    one device's hiccup as the pipeline's rate. The floors are per-op device durations
    reduced with MIN across devices, which is the cost with inter-device waiting excluded.

    Chained per-op MEDIANS and MAXes are never computed, let alone published: in a
    free-running chain the inter-device wait parks in whichever op window happens to absorb
    it, bistably, so only the period and the floors are stable statistics.

    FAILS CLOSED. Every device must show exactly `iters` anchor occurrences. XLA unrolling,
    deduplicating or re-scheduling the loop shows up here first, and a mismatch returns an
    unavailable block rather than dividing by the wrong N. That check runs BEFORE any
    arithmetic, because a silently-wrong period is worse than no period.
    """
    trace, scoped = best_trace(trace_dir, chain_marker)
    if trace is None:
        return {"availability": "unavailable", "reason": "no readable trace",
                "marker": chain_marker}
    anchored = [e for e in scoped
                if any(family in str(e.get("name", "")) for family in (anchor,))]
    starts = _anchor_starts(anchored)
    if not starts:
        return {"availability": "unavailable",
                "reason": f"no anchor {anchor!r} under {chain_marker!r}",
                "marker": chain_marker, "anchor": anchor}

    counts = {device: len(series) for device, series in starts.items()}
    if set(counts.values()) != {int(iters)}:
        return {"availability": "unavailable",
                "reason": f"anchor occurrences {sorted(set(counts.values()))} != iters "
                          f"{iters}; the compiler transformed the loop",
                "marker": chain_marker, "anchor": anchor,
                "occurrences_per_device": counts}

    per_device = [[series[i + 1] - series[i] for i in range(len(series) - 1)][drop:]
                  for series in starts.values()]
    if not per_device or not per_device[0]:
        return {"availability": "unavailable",
                "reason": f"no periods survive drop={drop} at iters={iters}",
                "marker": chain_marker, "anchor": anchor}

    stats = chain_stats(per_device)
    return {
        "availability": "measured",
        "origin": "chained-median",
        "marker": chain_marker,
        "anchor": anchor,
        "iterations": int(iters),
        "dropped": int(drop),
        # The RAW per-device series, so a multi-host caller can gather the other host's
        # devices before reducing. At EP16 each process's profiler sees only its own 8, and
        # the missing half is exactly where inter-host stragglers live -- reducing locally
        # would publish a median over the faster half of a scale-out exchange.
        "per_device_periods": per_device,
        **stats,
    }


def chain_pair_gap(trace_dir: str, chain_marker: str, period_anchor: str,
                   drop: int = 1) -> float | None:
    """Start-to-start MINUS the pair window -- the GPU family's `interpair_gap_us`.

    The window is the chain scope's own extent within one iteration: earliest op start to
    latest op end, over EVERY op under the marker. Not anchor-to-anchor. Each direction
    carries work outside its collective -- measured at T=8192, a 1,769us gather before
    dispatch's collective and a 3,363us scatter-add after combine's, 7.3ms of the two
    together -- and an anchor-to-anchor window drops all of it into the "gap". That is the
    same mistake as subtracting only the two floors, one order smaller: it read 20..30% of
    the period where the floors basis read 47%.

    Defining the window by the scope's extent also makes this independent of WHICH ops the
    floor gate picked, so a run whose anchors are unusable still reports a gap.

    Iterations are delimited by the period anchor's starts, the one op known to fire once
    per iteration. Cross-device MEDIAN, as the GPU reduces it.
    """
    trace, scoped = best_trace(trace_dir, chain_marker)
    if trace is None:
        return None
    bounds, spans = defaultdict(list), defaultdict(list)
    for event in scoped:
        if event.get("ts") is None:
            continue
        begin, finish = float(event["ts"]), _end_us(event)
        if finish is None:
            continue
        spans[event.get("pid")].append((begin, finish))
        if str(event.get("name", "")) == period_anchor:
            bounds[event.get("pid")].append(begin)
    per_device = []
    for pid, marks in bounds.items():
        marks = sorted(marks)[drop:]
        if len(marks) < 2:
            continue
        ordered = sorted(spans.get(pid) or [])
        gaps = []
        for i in range(len(marks) - 1):
            lo, hi = marks[i], marks[i + 1]
            inside = [(b, e) for b, e in ordered if lo <= b < hi]
            if not inside:
                continue
            window = max(e for _, e in inside) - min(b for b, _ in inside)
            gaps.append((hi - lo) - window)
        pcts = _pcts(gaps)
        if pcts:
            per_device.append(pcts["p50"])
    if not per_device:
        return None
    ordered = sorted(per_device)
    middle = len(ordered) // 2
    return (ordered[middle] if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0)


def chain_stats(per_device: list) -> dict:
    """Reduce a per-device period matrix: MEDIAN across devices, plus the drift diagnostic.

    `settle_drift_us` is late-half mean minus early-half mean of the period series. It is
    what decides whether `drop` is big enough: a chain still filling its pipeline drifts, a
    settled one does not. Measure rather than argue the drop count.
    """
    usable = [series for series in per_device if series]
    if not usable:
        return {"percentiles_us": None, "pair_spread_us": None, "sample_count": 0,
                "devices": 0, "settle_drift_us": None}
    periods, spreads = [], []
    for step in zip(*usable):
        ordered = sorted(step)
        middle = len(ordered) // 2
        periods.append(ordered[middle] if len(ordered) % 2
                       else (ordered[middle - 1] + ordered[middle]) / 2.0)
        spreads.append(ordered[-1] - ordered[0])
    # Per DEVICE, then signed max-magnitude across devices -- the GPU family's reduction
    # (`max(drifts, key=abs)` over per-rank late-half minus early-half p50). Computing the
    # drift on the already-median-reduced series instead, as this did, hides exactly the
    # case the field exists to catch: one device still filling its pipeline is averaged
    # away by the seven that have settled, and `drop` is then defended by a number that
    # could not have indicted it.
    drifts = []
    for series in usable:
        half = len(series) // 2
        if half:
            early, late = _pcts(list(series[:half])), _pcts(list(series[half:]))
            if early and late:
                drifts.append(late["p50"] - early["p50"])
    drift = max(drifts, key=abs) if drifts else None
    return {
        "percentiles_us": _pcts(periods),
        "pair_spread_us": max(spreads) if spreads else None,
        "sample_count": len(periods),
        "devices": len(usable),
        "settle_drift_us": drift,
    }


def _pcts(values: list) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    return {"p50": at(0.50), "p90": at(0.90), "p95": at(0.95), "p99": at(0.99)}


#: Ops whose device duration is a MARKER, not work. Measured: `_chain_anchor` picked
#: `prepare_start_ragged-all-to-all.3.cloned.1.call-start` -- most frequent, because markers
#: are frequent -- and its 0.048us duration was published as a chain FLOOR, i.e. as "per-op
#: device time with waiting excluded". The period survived that (start-to-start deltas only
#: need an op firing once per iteration) but the floor was a fabricated number.
#: `call-done` is deliberately NOT here. It looks like a marker by name, but measured it
#: carries substantial device time -- `ragged-all-to-all.2.cloned.1.call-done` at 6692.6us in
#: a fresh-entry dispatch inventory -- so excluding it could discard the real floor. Only the
#: START markers are excluded, which measured 0.048us. Whether call-done is the transfer or a
#: completion wait is unresolved; the share check below is what protects the number either way.
_MARKER_OP_FRAGMENTS = ("call-start", "call_start", "prepare_start")
#: A floor below this fraction of the period is a marker, not the collective.
_FLOOR_MIN_SHARE_OF_PERIOD = 0.01


def floor_anchors(trace_dir: str, chain_marker: str, iters: int) -> dict:
    """The dispatch and combine collectives, told apart -- or nothing.

    XLA numbers distinct instructions, so the chain body's two collectives are usually two
    concrete names (fp8 dispatch alone shows `ragged_all_to_all.38` and `.24`). Take the two
    highest-total distinct names, then assign direction by per-iteration START ORDER: the
    instance that runs first in each iteration is dispatch.

    Gated, because a mislabeled direction is worse than no direction. Each chosen name must
    appear exactly `iters` times per device, and after a start-sort the two must strictly
    interleave d,c,d,c. Fusion or elision breaks one of those, and then BOTH directions
    publish unavailable with the reason -- never one series wearing two labels, which is what
    the first cut did (identical p50/p90/p95/p99 under `dispatch` and `combine`).
    """
    trace, events = best_trace(trace_dir, chain_marker)
    if trace is None:
        return {"ok": False, "reason": "no readable trace"}

    totals: dict[str, float] = defaultdict(float)
    per_name: dict[str, list] = defaultdict(list)
    for event in events:
        name = str(event.get("name", ""))
        if not any(family in name for family in CHAIN_ANCHOR_FAMILIES):
            continue
        if any(fragment in name for fragment in _MARKER_OP_FRAGMENTS):
            continue
        # No marker filter: best_trace() already returned only the matching events, so one
        # here would be a condition that cannot fail.
        value = duration_us(event)
        if value is None or event.get("ts") is None:
            continue
        totals[name] += value
        per_name[name].append(event)
    # Both anchors must come from the SAME device rows, and choosing by row-set SIZE does
    # not achieve that: measured on run 31180043680, a `...cloned.1.call-done` and a bare
    # `ragged-all-to-all.N` occupy equal-sized but DISJOINT sets -- tpu7x logs several core
    # kinds, and the widest-set filter kept both, exactly as this file's own disjoint test
    # said it would. Floors went to 2 of 10 points, worse than the 5 of 10 before it.
    #
    # So group the candidates BY their row set and take the group carrying the most device
    # time. That is the row set where the chain's collective work actually lives, and the
    # two anchors are then pairable by construction rather than by luck.
    # Only groups that can form a pair AT ALL are eligible -- otherwise a single large op
    # alone on a sparse-core row wins on total and the point fails with one candidate.
    # Kept for the diagnostic: after grouping, the surviving set can be empty, and a
    # reason that says "0 ops" without naming what WAS there is not a diagnosis.
    seen = sorted(totals)
    groups: dict[frozenset, list] = defaultdict(list)
    for name, value in totals.items():
        groups[frozenset(e.get("pid") for e in per_name[name])].append((name, value))
    # Carried into the failure so a rejected point DIAGNOSES itself. Three selection rules
    # were tried against this data and each rejection cost a hardware run to interpret,
    # because the artifact said which ops it saw but never how they were distributed over
    # rows -- the one fact the decision turns on.
    layout = sorted(
        ({"rows": sorted(str(pid) for pid in rows),
          "ops": sorted(name for name, _ in members),
          "total_us": round(sum(value for _, value in members), 3)}
         for rows, members in groups.items()),
        key=lambda group: -group["total_us"])
    eligible = {rows: members for rows, members in groups.items() if len(members) >= 2}
    if eligible:
        dominant = max(eligible, key=lambda rows: sum(v for _, v in eligible[rows]))
        totals = dict(eligible[dominant])
    elif groups:
        totals = {}          # nothing pairable; fall through to the fail-closed below
    if len(totals) < 2:
        return {"ok": False,
                "reason": f"only {len(totals)} distinct collective op(s) on the busiest "
                          f"pairable device row set under {chain_marker!r} "
                          f"(saw {len(seen)}: {', '.join(seen) or 'none'}); "
                          f"cannot tell dispatch from combine",
                "names": seen, "row_groups": layout}

    first, second = sorted(totals, key=totals.get, reverse=True)[:2]
    for name in (first, second):
        counts = defaultdict(int)
        for event in per_name[name]:
            counts[event.get("pid")] += 1
        if set(counts.values()) != {int(iters)}:
            return {"ok": False,
                    "reason": f"{name!r} occurs {sorted(set(counts.values()))} times per "
                              f"device, not {iters}; the compiler transformed the loop",
                    "names": [first, second]}

    # Both names must live on the SAME devices. The occurrence gate above checks each name's
    # own per-device counts INDEPENDENTLY, so two ops logged on disjoint device rows both
    # pass it -- and then the interleave check below degenerates: it merges only one name's
    # events for such a device, gets a single-label sequence, and a single-label sequence
    # trivially satisfies "alternating". Measured on run 31147054940: at 10 of 14 points the
    # two anchors were different op KINDS (a `...cloned.1.call-done` against a bare
    # `ragged-all-to-all.N`), and the pair could not be formed on any device -- which
    # surfaced only because the phase and the gap both went null there while the floors were
    # published as if the two directions had been measured.
    devices_first = {e.get("pid") for e in per_name[first]}

    # Direction by start order, checked on every device rather than assumed from one. The
    # row GROUPING is what makes the alternation check meaningful: both anchors came out of
    # one group keyed by their pid set, so every pid here carries both names and a
    # single-label sequence -- which would satisfy alternation vacuously -- cannot arise.
    # (An explicit `devices_first != devices_second` guard used to state this; grouping
    # makes it unreachable, so it was removed rather than left as a check that can't fire.)
    order = set()
    for pid in devices_first:
        merged = sorted(
            [(float(e["ts"]), first) for e in per_name[first] if e.get("pid") == pid]
            + [(float(e["ts"]), second) for e in per_name[second] if e.get("pid") == pid])
        labels = [label for _, label in merged]
        if labels != [labels[i % 2] for i in range(len(labels))]:
            return {"ok": False,
                    "reason": f"{first!r} and {second!r} do not interleave on device {pid}; "
                              f"start order cannot assign direction",
                    "names": [first, second]}
        order.add(labels[0])
    if len(order) != 1:
        return {"ok": False,
                "reason": f"devices disagree on which of {first!r}/{second!r} runs first",
                "names": [first, second]}
    dispatch = order.pop()
    combine = second if dispatch == first else first

    # Alternation is necessary but NOT sufficient. Each direction emits more than one
    # collective op -- measured, a component's inventory carries both `...call-done` and a
    # `ragged_all_to_all.N` -- so the two highest-total names can both belong to DISPATCH.
    # Their starts would then read d1a, d1b, d2a, d2b: perfectly alternating, and
    # mislabelled. What separates the cases is WHERE in the period the second op starts.
    # Two ops of one direction sit next to each other near one end; a genuine
    # dispatch/combine pair is separated by roughly half the period.
    phases = []
    for pid in {e.get("pid") for e in per_name[dispatch]}:
        starts_d = sorted(float(e["ts"]) for e in per_name[dispatch] if e.get("pid") == pid)
        starts_c = sorted(float(e["ts"]) for e in per_name[combine] if e.get("pid") == pid)
        if len(starts_d) < 2 or len(starts_c) != len(starts_d):
            continue
        periods = [b - a for a, b in zip(starts_d, starts_d[1:])]
        offsets = [c - d for d, c in zip(starts_d, starts_c)]
        usable = [off / per for off, per in zip(offsets, periods) if per > 0]
        if usable:
            phases.append(sum(usable) / len(usable))
    phase = (sum(phases) / len(phases)) if phases else None
    return {"ok": True, "dispatch": dispatch, "combine": combine,
            # Published, not gated: a threshold here would need a value nobody has measured
            # yet. Near 0.5 is a real pair; near 0 or 1 means both names are one direction
            # and the two floors are not the two directions.
            "phase": phase,
            "phase_note": None if phase is None else (
                "consistent with a dispatch/combine pair" if 0.25 <= phase <= 0.75
                else "SUSPECT: both anchors may belong to the same direction")}


def chain_floors(trace_dir: str, chain_marker: str, anchor: str, drop: int = 1,
                 period_us: float | None = None) -> dict:
    """Per-occurrence device time of the anchor op, cross-device MIN -- waiting excluded.

    Fails closed rather than publishing a marker's duration: if the resulting floor is a
    negligible fraction of the period, the anchor is not the collective and the block goes
    unavailable. A floor that is 0.16% of the pair period is not a floor.
    """
    trace, matched = best_trace(trace_dir, chain_marker)
    if trace is None:
        return {"availability": "unavailable", "reason": "no readable trace"}
    scoped = [e for e in matched if anchor in str(e.get("name", ""))]
    grouped: dict[object, list[float]] = defaultdict(list)
    for event in sorted(scoped, key=lambda e: e.get("ts") or 0):
        value = duration_us(event)
        if value is not None:
            grouped[event.get("pid")].append(value)
    series = [v for v in grouped.values() if v]
    if not series:
        return {"availability": "unavailable", "reason": f"no {anchor!r} durations"}
    floors = [min(step) for step in zip(*series)][drop:]
    if not floors:
        return {"availability": "unavailable", "reason": "no occurrences survive drop"}
    pcts = _pcts(floors)
    if period_us and pcts and pcts["p50"] < period_us * _FLOOR_MIN_SHARE_OF_PERIOD:
        return {"availability": "unavailable", "anchor": anchor,
                "reason": f"floor p50 {pcts['p50']:.4f}us is "
                          f"{pcts['p50'] / period_us:.4%} of the {period_us:.1f}us period "
                          f"-- {anchor!r} is a marker op, not the collective"}
    return {"availability": "measured", "origin": "chained-cross-rank-min",
            "anchor": anchor, "percentiles_us": pcts, "sample_count": len(floors)}
