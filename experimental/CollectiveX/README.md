[English](README.md) | [中文](README_zh.md)

# CollectiveX

CollectiveX is an experimental MoE expert-parallel communication benchmark. It measures dispatch,
combine, and paired roundtrip latency across EP libraries and accelerator systems, then uploads
neutral result artifacts.

CollectiveX schedules benchmarks, executes them on real allocations, and uploads the neutral
artifacts each run emits. It does not validate those artifacts, promote, rank, recommend, select, or
decide what a consumer displays. Any downstream display or comparison is the consumer's
responsibility. The full measurement methodology is in [docs/methodology.md](docs/methodology.md).

## Execution Profile

The workload uses packed placement and one pinned `fixed-profile` resource configuration per
backend/topology; there is no tuning sweep. Combine is always BF16; dispatch precision is a swept
dimension — a BF16 control plus, on every backend whose FP8 dispatch is supported upstream
(DeepEP V2, MoRI, UCCL-EP), an FP8 dispatch, caller-prequantized in `normal` mode (in `low-latency`
the DeepEP and UCCL-EP kernels quantize internally from BF16; MoRI stays caller-prequantized). NCCL
EP is BF16-only this release, so it emits the control alone. Coverage is uniform routing only. Cases
run in one of two modes:

- `normal` uses `layout-and-dispatch-v1`, rank-deduplicated token payloads, and activation-only,
  unweighted rank-sum combine. It runs the full decode and prefill ladders.
- `low-latency` uses each backend's decode-optimized kernel family: on DeepEP the legacy
  `deep_ep.Buffer` IBGDA `low_latency_dispatch`/`low_latency_combine` (a per-expert padded receive
  and a source-side gate-weighted combine); on UCCL-EP the same legacy `Buffer` low-latency kernels
  over its CPU-proxy transport; on MoRI the `IntraNodeLL` kernel (single-call,
  pure-intranode, same compact layout and unweighted rank-sum combine as `IntraNode`). It is a
  decode-phase-only, per-SKU-capability-gated addition whose runnable set differs from `normal`'s, so
  it is enabled from each SKU's `ll_backends` registry entry (currently DeepEP V2 EP8 on H100/H200/B200,
  MoRI EP8 on MI300X/MI325X/MI355X, and UCCL-EP EP8 on H100/H200/B200 only — UCCL's low-latency kernel
  trips a warp-group assertion on AMD's CU count, so the AMD SKUs keep UCCL-EP normal mode without LL;
  NCCL EP has no low-latency row on any SKU while its decode kernels carry
  [NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303)).
  Scoped single-node EP8 runs over the intra-node NVLink/XGMI
  low-latency path (no `/dev/gdrdrv` needed — validated on H200 with it absent); NVSHMEM/IBGDA on the
  wire is only a multi-node scale-out (EP16) concern.

Cases use a fixed timing profile from `configs/sweep.json`: 256 trials x 8 timed iterations (2048
samples per component) with 32 synchronized full roundtrip warmups before each measured component at
every trial/point. Component measurement order rotates each trial so every timed component occupies
every position in the sequence; each iteration takes the cross-rank maximum before nearest-rank
p50/p90/p95/p99, and roundtrip p99 is the headline latency. A keyed BLAKE2b counter produces
byte-identical routing and gate weights on every runtime.

Correctness is checked against an implementation-independent oracle that reproduces the backend's
two-level reduction — intra-scale-up-domain FP32, then a BF16 cast of each domain's partial for the
scale-out send. The combine gate is a tight max elementwise relative error below `8 * 2^-8`
(denominator clamped at 0.02), which holds across scale-up and multi-node scale-out topologies
alike. Under FP8 dispatch the oracle applies the same per-token cast round-trip to its semantic
payload, so the dispatched-payload compare stays bit-exact and the combine gate is unchanged — the
quantization is modeled, not tolerated. Any failed rank or point makes the case ineligible in the
result it writes.

The matrix covers H100, H200, B200, B300, GB200, GB300, MI300X, MI325X, and MI355X. `sweep_matrix.py` materializes
the requested SKUs, backends, EP sizes, and token ladders, then extracts strict per-shard controls
and rejects missing, stale, malformed, or altered shard controls. `--only-sku`, `--exclude-skus`,
`--ep-sizes`, and `--precisions` select a subset; the matrix is generated per dispatch, with no
frozen digest or locked case count.

| Systems | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink, scale-up | 2x8 NVLink + RDMA, scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI, scale-up | 2x8 XGMI + RDMA, scale-out |
| GB200/GB300 | 2x4 MNNVL, scale-up | 4x4 MNNVL, scale-up |
| TPU v7 (tpu7x) | 1x8 ICI, scale-up | 2x8 ICI, **scale-up** |

Physical host count does not determine scope: both GB topologies stay inside one 72-GPU MNNVL
scale-up domain.

**TPU v7 EP16 is scale-up, and that is not a labelling convenience.** `tpu7x`'s ICI is a 3D torus
that spans hosts — the [documented](https://docs.cloud.google.com/tpu/docs/tpu7x) topologies go
`2x2x1` (4 chips, 1 host) → `2x2x2` (8 chips, 2 hosts) → up to `8x16x16`, all one ICI fabric — so a
16-device cell crosses a host boundary without leaving the scale-up domain. Nothing touches DCN.
`scale_up_domain` is therefore the literal `"ep"` for this SKU, meaning it follows the EP degree:
the slice is provisioned at the size the case needs, so EP8 reports a domain of 8 and EP16 a domain
of 16, and both stay `scope: scale-up`. A flat 8 would have labelled EP16 scale-out over DCN, which
is false; a flat 16 would have over-claimed the EP8 shard, whose slice really is 8 devices wide.

The consequence for comparison: **TPU EP16 is comparable to GB200/GB300 EP16** — also scale-up,
inside a 72-GPU MNNVL domain — and **not** to b200/h100/mi355x EP16, which genuinely take an RDMA
hop. Same EP degree, different regime; the artifact's `scope` is what disambiguates.
`scale_out_transport: dcn` on the SKU describes the genuinely-scale-out case, cross-*slice* traffic
over the 100 Gbps-per-chip data-center network, which no cell uses today.

Running it needs a multi-host slice, which the obvious command does not create: `--tpu-topology`
alone builds a *placement* policy and `tpu7x` rejects it ("Use workload policy instead"). The pool
behind EP16 was created with an explicit workload policy —
`resource-policies create workload-policy --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2` —
and the recipe is recorded in `launchers/launch_tpu-gke.sh` beside the guard that needs it. The
shard then runs as an Indexed Job with a headless Service so GKE injects `TPU_WORKER_ID` and
`TPU_WORKER_HOSTNAMES` for `jax.distributed.initialize()`. Both precisions share ONE multi-host
shard: a slice is a single indivisible allocation, so two shards each wanting all its nodes get one
pod apiece and both half-formed slices abort.

Two multi-host invariants the probe enforces, because breaking either produces a plausible-looking
number rather than an error. Every process must issue the *same sequence* of slice-wide collectives,
so `reference_timing` runs `lockstep` (fixed iteration counts) under multi-host — a duration-driven
loop makes the count depend on each host's clock and libtpu aborts the slice. And the mesh is built
from `jax.devices()`, never `jax.local_devices()`: the latter would build an EP8 mesh per host and
measure two independent 8-way exchanges under an EP16 label. That the exchange really spans 16 ranks
is checked against physics rather than asserted — dispatch cost tracks the routing fanout
(measured 1.232x from EP8 to EP16 at T=8192 against a fanout ratio of 1.232), where two independent
8-way exchanges would sit at 1.0.

| Backend | Current scope |
|---|---|
| DeepEP V2 | `normal` mode is PR #605 `ElasticBuffer` plus exact upstream #630 and #640 fixes: LSA for scale-up and GIN for x86 EP16 scale-out. FP8 dispatch via `use_fp8_dispatch` (blockwise e4m3fn) alongside BF16. `low-latency` mode is the legacy `deep_ep.Buffer` IBGDA decode kernels (per-expert padded layout, weighted combine, `use_fp8` e4m3fn), decode/EP8 only |
| MoRI | `normal` mode uses the direct `IntraNode` kernel for scale-up EP8 on every CDNA SKU and pins `InterNodeV1` for EP16 over 2x8 XGMI + RDMA. `low-latency` mode selects the `IntraNodeLL` decode kernel (single-call, pure-intranode, same compact layout and unweighted combine as `IntraNode`), decode/EP8 only. FP8 dispatch is caller-prequantized (per-SKU e4m3fnuz on gfx942, e4m3fn on gfx950); combine stays BF16 (`quant_type=none`) alongside BF16 dispatch |
| UCCL-EP | [UCCL](https://github.com/uccl-project/uccl) EP: a drop-in, API-identical DeepEP replacement whose CPU proxies issue GPUDirect RDMA over plain `libibverbs` (no NVSHMEM/IBGDA), with software message ordering, atomics, and flow control; scale-up is single-node `cudaIpc` over NVLink/XGMI (never MNNVL). `normal` mode is the legacy `Buffer` `dispatch`/`combine` (unweighted rank-sum); `low-latency` reuses the legacy `low_latency_dispatch`/`low_latency_combine` decode kernels (weighted combine), decode/EP8 only. FP8 dispatch is caller-prequantized in `normal` mode (blockwise e4m3fn, per-SKU e4m3fnuz on gfx942); in `low-latency` mode the caller sends BF16 and the decode kernel quantizes to e4m3 internally (`use_fp8`). Combine is BF16. Runs on NVIDIA and AMD (H100/H200/B200 + MI300X/MI325X/MI355X), EP8 scale-up. Cross-node EP16 is functional (the internode RDMA path connects and the light case passes correctness) but its CPU-proxy throughput overruns the standardized per-case wall-clock budget on heavy token counts, so EP16 is an unsupported coverage row for now |
| NCCL EP | [NCCL EP](https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep): NVIDIA's native MoE dispatch/combine on the NCCL Device API — LSA (NVLink load/store) intra-node, GIN (GPU-Initiated Networking) inter-node — driven through the `nccl4py` bindings. `normal` mode selects the `HIGH_THROUGHPUT` algorithm (FLAT `[N, hidden]` receive, unweighted rank-sum combine); the `LOW_LATENCY` algorithm is implemented in the adapter but has no enabled cell (see the `ll_backends` note above). BF16 only: NCCL EP's FP8 machinery exists upstream but its RELEASE.md lists it unsupported/untested, so no FP8 case is emitted. NVIDIA-only and CUDA 13 only. EP8 scale-up on H100/H200/B200/B300 plus EP8 and EP16 on GB200/GB300, where EP16 stays inside the MNNVL scale-up domain. x86 EP16 scale-out is an unsupported coverage row: the cross-node GIN path faults inside `nccl_ep.cc` identically on RoCE and IB across four SKUs, a GDAKI limit rather than a fabric-selection one |

| JAX ragged A2A (TPU probe) | `jax.lax.ragged_all_to_all` on a 1-D device mesh over the slice's ICI domain — the primitive JAX MoE stacks use for EP dispatch/combine — at EP8 (one host) and EP16 (two hosts, still one ICI domain). `normal` mode, BF16 and FP8: FP8 dispatch is blockwise e4m3fn with one FP32 scale per 128-element block (DeepEP's `per_token_cast_to_fp8`), values and scales as two ragged exchanges, `fp8_consume: native` so the expert consumes fp8 directly and no standalone conversion sits between the collectives; the conversion is measured separately as `stage`. Combine is BF16 on both precisions, because the expert emits BF16. Dispatch is a permute gather plus the ragged exchange; combine is the reverse exchange plus an fp32 scatter-add (unweighted rank-sum). A JAX build lacking `ragged_all_to_all` **fails closed**: there is no padded fixed-capacity `all_to_all` fallback, because padding would move different bytes and quietly measure something else. See the TPU Probe section for what this measurement is and is not |

DeepEP V2 means the `ElasticBuffer` implementation introduced by
[DeepEP PR #605](https://github.com/deepseek-ai/DeepEP/pull/605), not a newer legacy `Buffer` build.
The pinned source is the [PR #630](https://github.com/deepseek-ai/DeepEP/pull/630) head, whose parent
is the #605 merge tree, plus the exact one-line library matcher from upstream
[PR #640](https://github.com/deepseek-ai/DeepEP/pull/640). The first fixes pure scale-up
initialization when GIN is unavailable; the second prevents NCCL shared-memory mappings from being
misclassified as duplicate NCCL libraries. Scale-up cases request NCCL Device API LSA and fail closed
unless the realized LSA team covers the full EP world. x86 EP16 scale-out cases instead require the
hybrid path with GIN, two logical scale-out domains represented by two physical RDMA ranks, and eight
scale-up ranks per domain; GB EP16 remains MNNVL scale-up and therefore uses LSA. Whether a given
SKU/backend/EP cell is attempted is a capability fact; whether it succeeded is decided by the
benchmark's return code.

## Workflow And Artifacts

`.github/workflows/collectivex-sweep.yml` has two jobs. `setup` generates a public-SKU matrix
(`backend`, `only_sku`, `exclude_skus`, `ep_sizes` inputs) and uploads the matrix.
`sweep` extracts a strict ignored `.shards/<id>.json` control per matrix entry, executes one
allocation per shard, fetches pinned DeepEP source before allocation when required, and uploads the
result artifacts with `always()` so a red or partial run still uploads.

Each shard emits per-case result JSON and a small mechanical summary. A case counts as successful on
the benchmark's own return code; there is no completeness or privacy validation step, and failed or
unsupported cells produce no synthetic record. No step promotes a run,
builds a dataset, or advances a channel; the neutral artifacts are the output. A consumer downloads
them and decides what to display.

No operator credentials are passed to the workflow or uploaded; runner-local overrides and any
selectors stay on the runner. Per-step runner logs are kept on the runner for postmortem, and
result artifacts carry only the fields listed in the methodology.

## Runner Configuration

Each SKU's Slurm and storage values come from its tracked baseline in the registry. An optional
runner-local JSON document at `$XDG_CONFIG_HOME/inferencex/collectivex.json` or
`COLLECTIVEX_OPERATOR_CONFIG` overlays that baseline per field; unknown runners, fields, duplicate
keys, and non-JSON input fail closed, and configuration is never evaluated as shell. GHA passes no
operator secret, so a SKU runs entirely from its tracked baseline unless a runner-local document is
present.

All public per-SKU platform data lives in the tracked `configs/platform_config.json` registry:
architecture/product, vendor, accelerator runtime, container image and platform, fixed placement,
launcher, runnable backend/EP pairs, the scale-out `fabric` identity (NIC and switch — so same-GPU
clusters on different fabrics are distinct entries, e.g. a second b200 cluster), tracked operator
defaults, and scale-out RDMA selectors. `vendor` is arbitrary normalized metadata and is not
restricted to AMD or NVIDIA; `runtime` selects the execution path and is a closed set
(`runtime/config.py`'s `ACCELERATOR_RUNTIMES`) because it decides which benchmark entrypoint and
launcher family a case can use: `cuda`/`hip` run `bench/run_ep.py` (torch, NCCL/RCCL, one process per
rank) and `tpu` runs `bench/run_ep_jax.py` (JAX, XLA collectives, one process per host). Adding
another vendor therefore does not require changing a vendor allowlist, although a new
runtime or collective implementation still requires its own compatible entrypoint and launcher.
An optional `scale_out_transport` field names the SKU's cross-host fabric when it is not RDMA
(TPU hosts use `dcn`), so a scale-out row is never labeled with a fabric the cluster does not have.
Operator documents can override the defaults. Launchers
declare and check the fields they actually require. `sweep_matrix.py` derives EP topology from the
placement fields; the sweep includes every registered SKU by default.

Every selected non-MNNVL EP16 placement additionally requires `socket_ifname` and `rdma_devices` for
its operator-approved fabric; optional `ib_gid_index`, `rdma_service_level`, `rdma_traffic_class`,
and `rail_isolated` are also allowlisted. Service level and traffic class are mapped into MoRI's
RDMA/IO QoS environment.
CollectiveX does not heuristically select a management route or HCA. After allocation, every
non-MNNVL scale-out node must prove that all configured interfaces and active HCA ports exist before
backend setup. Scale-up and MNNVL jobs clear these overrides. Scale-out NCCL/RCCL is pinned to `IB`
with exact-match HCA selectors so a socket fallback fails instead of being mislabeled as RDMA.
Scale-out also disables NCCL dual-port NIC fusion (`NCCL_IB_MERGE_NICS=0`): a fused device disables
NCCL GIN, which the DeepEP V2 EP16 hybrid path requires, and a rail-isolated fabric
(`rail_isolated=1`, e.g. B300's multi-plane RoCE) additionally sets `NCCL_CROSS_NIC=0`.

`ib_gid_index` is applied only when every selected HCA port reports an Ethernet link layer, where it
selects the operator-approved RoCE GID. Native InfiniBand profiles retain explicit HCA and service
level pinning but leave the RoCE-only GID override unset so NVSHMEM/NCCL can use the native LID path.
Mixed Ethernet and InfiniBand HCA lists are rejected.

`stage_dir` is a pre-existing, runner-owned, non-symlinked base outside the checkout and workflow
workspace. It is not group- or world-writable and is visible at the same path on the runner and every
allocated node. Jobs create only a marked mode-0700 execution child, prove cross-node read/write
visibility, and remove that exact child after allocation teardown; they never mount the runner
checkout or create a stage beneath image storage on AMD. When an AMD operator row omits `stage_dir`,
the runner derives a private base beside its standard `_work` directory on the shared runner
filesystem; the root-owned squash cache is never used as a repository stage.

H200, B200, and B300 runners may omit `stage_dir`; their isolated execution child is created under a
runner-owned mode-0700 base in the validated operating-system account home, independent of the
workflow's temporary `HOME`. H100 may also omit `stage_dir`; its private base is created beside, never
beneath, the configured shared container directory so it is compute-visible. Canonical B300 execution
ignores any legacy configured `stage_dir` and always uses the validated compute-visible account-home
base; an execution-ID suffix isolates parallel B300 workers. Canonical GB300 execution likewise
ignores its legacy group-writable `stage_dir` and derives an execution-specific private base beneath the
validated compute-visible account home. Backend preparation runs from that staged tree on every node.

Enroot imports the configured image tag into a per-run-scoped squash keyed by image tag and image
platform, so one run never reuses another run's imported filesystem. The image tag and platform are
per-SKU registry fields; the DeepEP V2 source pin lives in `runtime/common.sh` and its build is
fetched and verified at the pinned commit, checked for `ElasticBuffer`, and cached in a
cluster-local build cache keyed by architecture, image, and commit. Only the fixed `/cx-cache` mount
reaches the container.

## TPU Probe

The TPU path is a probe, not a peer of the GPU backends, and the differences are recorded in the
artifact rather than left for a reader to infer.

It runs on the GKE/ARC `tpuv7` pool, whose runner is a CPU-only coordinator pod. There is no Slurm,
no enroot/pyxis, and no shared filesystem, so `launchers/launch_tpu-gke.sh` mirrors
`runners/launch_tpuv7-gke.sh` instead of `launch_single-slurm.sh`: it packages the coordinator's
already-pinned source subtree into a ConfigMap (~120 KB, no repository credential needed inside the
pod), spawns ONE Kubernetes Job per shard onto `tpu7x` nodes, runs every case in that one Job, and harvests
the result JSONs back through the pod log. At EP16 the Job is **Indexed** with one pod per host of the
slice plus a headless Service, so GKE injects `TPU_WORKER_ID`/`TPU_WORKER_HOSTNAMES` for
`jax.distributed.initialize()`; every pod computes the same rows and the pod at completion index 0
writes the artifact, which is also the pod the harvest reads (`jax.process_index()` does NOT track the
pod index, so keying off it would harvest the wrong pod half the time). A red or partial leg still ships whatever it produced.

Two properties differ from the GPU family's, and each is named in the artifact:

- `measurement.timing_source` is **`xla-device-trace-span`**. There is no public JAX equivalent of
  `torch.cuda.Event`, so the probe reads device time out of the XLA profiler trace instead: SPANS
  (first start to last end of a component's scope, not a sum of op durations), reduced MAX across
  devices per occurrence, RAW percentiles. That is the same *kind* of number as the GPU SKUs' CUDA
  events — chip time, no host dispatch cost.

  `host-wallclock-blocked` remains as a fallback and is **opt-in** (`--allow-host-fallback`),
  deliberately: host wall-clock on tpu7x carries a per-call dispatch floor that is payload-dependent
  and reaches thousands of microseconds, so a host figure published as if it were comparable would
  be badly wrong at the small-token end. Without the flag, a point that fails to yield a device span
  fails the case rather than silently downgrading. Provenance is recorded PER COMPONENT
  (`row.timing_source`), because the profiler can cover some components and not others and a
  whole-case label once claimed device timing for rows that were host-timed.

  An earlier revision amortized instead, chaining N operations inside one program to divide the host
  floor (`--in-program-iters`). Device spans made that unnecessary and the flag is gone; if you find
  a reference to it, the reference is stale.
- `components.stage` is `unavailable` **on BF16 rows**: the permute is fused into dispatch and
  there is no conversion, so there is nothing to time. **FP8 rows publish it** — the fp8→bf16
  conversion, hoisted out of both combine and the chained roundtrip and measured separately, which
  is what lets `roundtrip + stage` reconstruct the mismatched-config cost. Do not add `stage` to
  `roundtrip` when comparing against a GPU `native` row; do add it when comparing against a
  `CX_FP8_CONSUME=dequant` one. The per-destination layout (offsets, sizes, gather indices) is precomputed on
  host and excluded from the timed region, the same treatment DeepEP's layout pass gets; the
  on-device permute gather and the combine scatter-add ARE timed, because production pays them.
- `implementation.oracle` is `probe-source-identity-and-exact-rank-sum`, narrower than the GPU
  harness's full per-expert transform oracle. It still proves real things: every dispatched copy is
  decoded back to the source token it claims to be (the ID rides in the SIGN of the first columns,
  which survives quantization), its landing offset is checked against the exchange plan, and the
  combine is compared against the exact expected rank-sum. The expert is the identity, so no
  per-expert transform is modelled.

  Under BF16 the payload is compared bit-for-bit against the source. Under FP8 it deliberately is
  NOT, and the reason is worth knowing before anyone "fixes" it: two XLA compilations of the same
  quantize recipe round this lattice's e4m3 midpoints in opposite directions (measured — the wire
  carried `161/256` where a standalone program produced `152/256`), so an expectation re-derived by
  quantizing again is a coin flip. The GPU harness compares re-quantized bits only because
  `assert_quantize_identity` establishes that premise on metal first. Instead: the arrived chunk is
  compared byte-for-byte against the STAGED chunk from the same execution; combine is scored against
  the rows dispatch actually delivered, each attributed to the token its payload claims to be; and
  the timed program's own scales and dequantized values are tied back to the oracle program's, since
  the two are separate compilations. Nothing in that chain re-quantizes anything.

### The chained pair period

`pair_period` is a **different quantity** from `roundtrip`: the steady-state rate of one
dispatch→combine pair in a chain that never drains, where `roundtrip` is one pair entered from
idle. Do not sum them and do not substitute one for the other. `--chain-iters` (default 64) pairs
run inside one compiled program, the carry of each feeding the next — a real data dependency, not
an `optimization_barrier`, because a barrier constrains ordering and not liveness and XLA will
delete a computation nothing consumes.

**The chain is renormalised, and it has to be.** Combine is an unweighted rank-sum, so one pair
multiplies token `t` by `d_t`, its number of unique destination ranks — measured 3..7 at EP8 and
4..8 at EP16 for deepseek-v3 top-8. Over 64 iterations that is `d_t**64` against bf16's 3.39e38
ceiling: unrenormalised, the first element saturates at iteration 43 and **100% of EP16 elements
are infinite by 64**. The body divides the fp32 rank-sum by `d_t` before casting back, which makes
one pair the identity BITWISE (the fp32 sum of `d_t ≤ 8` copies of a bf16 value is exact, and IEEE
division is correctly rounded). The count is shipped, never `1/d`: over 200k values
`(d*v)*float32(1/d)` disagrees with `v` for `d == 7` on 58% of them, and seven destinations occurs
in both layouts.

`correctness.chain_regime_passed` is that identity, scored on the timed program and reduced across
hosts. **Tri-state**: `true` passed, `false` means it ran and disagreed (which fails the row),
`null` means it could not be evaluated (which withholds the period but does not condemn the drained
components measured in the same case). The drained oracle cannot cover this regime — it only ever
checks one pair entered from idle, so a transport that corrupts only under free-running pairs would
present as the fastest in the suite.

`chain_floor_us` is per direction, `origin: chained-cross-rank-min`. Candidates are the collectives
that occur exactly `--chain-iters` times per device; they are then grouped by the set of device rows
they appear on, and the anchors are the two highest-total ops in the **busiest row group that can
form a pair at all** (≥2 ops). Row grouping is what makes the two anchors comparable: tpu7x logs
more than one kind of core, so a `sparse-core-…` op can out-total a real anchor, and two ops on
equal-sized but *disjoint* row sets are not a pair however large they are — a rule that selected on
row-set **size** published floors at 2 of 10 points, worse than the 5 of 10 it replaced. The
`≥2` eligibility matters separately: without it a single large op alone on a sparse-core row wins
on total and the point fails holding one candidate. The surviving pair must also interleave
d,c,d,c. Direction comes from per-iteration start order. If any gate fails, **both** directions publish
`unavailable` with the reason; null here means "not measured", never "same as the other one".
`chain_health.anchor.phase` reports where the combine anchor starts within the period — ~0.5 is a
real pair, ~0 or ~1 means both anchors are one direction, which alternation alone cannot detect.

**A cross-check worth running on any new SKU:** if the anchors really are dispatch-then-combine
back to back, `phase` should track `chain_floor_us.dispatch / pair_period`, because combine starts
when dispatch's collective ends. Those come from independent quantities — one from start
timestamps, the other from op durations — so agreement is evidence the direction LABELS are right
and not merely self-consistent. Measured on bf16 (run 31194062843), across a 128x size range and
both EP degrees:

| T | `floor_dispatch / period` EP8 | `phase` EP8 | `floor_dispatch / period` EP16 | `phase` EP16 |
|--:|--:|--:|--:|--:|
| 64 | 0.321 | 0.361 | 0.332 | 0.365 |
| 512 | 0.276 | 0.294 | 0.287 | 0.311 |
| 8192 | 0.265 | 0.280 | 0.271 | 0.286 |

`phase` sits slightly above the ratio on all 28 chained rows (largest gap 0.064), which is the
expected sign: the floor is a cross-rank **min** and the phase is measured on the device that
actually paces the chain. A `phase` that does NOT track that ratio means the two anchors are not
the two directions, however cleanly they interleave.

`chain_health` mirrors the GPU family's block shape (each field a component with `percentiles_us`,
not a bare float):

| field | meaning |
|---|---|
| `interpair_gap_us` | start-to-start minus the chain scope's own extent per iteration. Near zero is free-running. **Measured 0.0–0.2 µs, 0.0–0.1% of the period**, at EP8 and EP16. Not the collective floors and not an op sum — both leave the permute and scatter-add inside the "gap" and read 47% and 20–30% respectively. |
| `settle_drift_us` | per-device late-half minus early-half p50, reduced by signed max-magnitude. Defends or indicts `--chain-drop`. Measured ≤0.6 µs. |
| `pair_spread_us` | cross-device spread per iteration. |
| `devices` / `devices_expected` | at EP16 each process profiles only its own 8 devices, so the per-device period matrix is allgathered before the reduction — the missing half is exactly where inter-host stragglers live. `gathered_across_hosts` says whether that happened; `devices_expected` stays the honest 16. |
| `op_inventory` | every op in the chain body, per iteration. This is what makes the fp8 exclusion checkable rather than argued. |
| `renorm_us` | **`unavailable`, and expected to be**: XLA fuses the divide into the adjacent cast, so no op carries the scope. The cost is bounded by cross-run differencing at ≤0.05% of the period, not measured directly. |
| `capture_s` / `capture_budget_s` | what the chain capture cost against the launcher's own per-case timeout. Measured 13–22 s against 2700. |

**FP8 is deliberately not chained.** Its combine needs BF16, so a chained fp8 body would carry the
fp8→bf16 conversion inside the loop and the period would include work the `native` contract says
production does not do standalone. `implementation.chained_period` is `false` on fp8 rows and the
chained fields are `unavailable` with that reason — not silently absent.

Workload identity IS shared with the GPU family: `bench/routing_np.py` is a parity-tested numpy port
of `bench/routing.py` (the TPU image has no usable torch), so both families benchmark the identical
routing trace and identical activation bytes, and both bill the same deduplicated (token,
destination-rank) payload unit. `tests/test_tpu_probe.py` asserts that parity element-for-element
whenever torch is importable and pins a golden digest when it is not, and it simulates the whole
exchange plan in pure numpy so an offset or transpose error fails locally rather than on a TPU node.

## Local Checks

```bash
python3 -m unittest discover experimental/CollectiveX/tests -p 'test_*.py'
python3 experimental/CollectiveX/sweep_matrix.py --backend all --out /tmp/cx-matrix.json >/dev/null
bash -n experimental/CollectiveX/runtime/*.sh experimental/CollectiveX/launchers/*.sh
```

Core paths are `configs/`, `sweep_matrix.py`, `summarize.py`, `bench/`, `runtime/`, `launchers/`,
and `tests/`.
