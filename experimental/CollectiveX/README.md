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
| TPU v7 (tpu7x) | 1x8 ICI, scale-up | unsupported coverage row — no multi-host slice |

Physical host count does not determine scope: both GB topologies stay inside one 72-GPU MNNVL
scale-up domain.

TPU v7 EP16 is recorded as unsupported rather than attempted, and the reason is how the pool is
PROVISIONED, not a limit of the hardware. `tpu7x`'s ICI is a 3D torus that spans hosts — the
[documented](https://docs.cloud.google.com/tpu/docs/tpu7x) topologies go `2x2x1` (4 chips, 1 host)
→ `2x2x2` (8 chips, 2 hosts) → up to `8x16x16`, all one ICI fabric — so a multi-host TPU EP cell
would be **scale-up over ICI**, not scale-out. What blocks it here is that `tpu-v7x` is provisioned
as six *independent* `2x2x1` slices, so these particular hosts have no shared fabric. The fix is a
larger slice (a JobSet plus `jax.distributed.initialize`, and `scale_up_domain` raised to match),
not a cross-host transport; the `tpu-gke` launcher refuses `nodes > 1` rather than quietly measuring
something else. `scale_out_transport: dcn` on the SKU describes the genuinely-scale-out case,
cross-*slice* traffic over the 100 Gbps-per-chip data-center network, which no cell uses today.

| Backend | Current scope |
|---|---|
| DeepEP V2 | `normal` mode is PR #605 `ElasticBuffer` plus exact upstream #630 and #640 fixes: LSA for scale-up and GIN for x86 EP16 scale-out. FP8 dispatch via `use_fp8_dispatch` (blockwise e4m3fn) alongside BF16. `low-latency` mode is the legacy `deep_ep.Buffer` IBGDA decode kernels (per-expert padded layout, weighted combine, `use_fp8` e4m3fn), decode/EP8 only |
| MoRI | `normal` mode uses the direct `IntraNode` kernel for scale-up EP8 on every CDNA SKU and pins `InterNodeV1` for EP16 over 2x8 XGMI + RDMA. `low-latency` mode selects the `IntraNodeLL` decode kernel (single-call, pure-intranode, same compact layout and unweighted combine as `IntraNode`), decode/EP8 only. FP8 dispatch is caller-prequantized (per-SKU e4m3fnuz on gfx942, e4m3fn on gfx950); combine stays BF16 (`quant_type=none`) alongside BF16 dispatch |
| UCCL-EP | [UCCL](https://github.com/uccl-project/uccl) EP: a drop-in, API-identical DeepEP replacement whose CPU proxies issue GPUDirect RDMA over plain `libibverbs` (no NVSHMEM/IBGDA), with software message ordering, atomics, and flow control; scale-up is single-node `cudaIpc` over NVLink/XGMI (never MNNVL). `normal` mode is the legacy `Buffer` `dispatch`/`combine` (unweighted rank-sum); `low-latency` reuses the legacy `low_latency_dispatch`/`low_latency_combine` decode kernels (weighted combine), decode/EP8 only. FP8 dispatch is caller-prequantized in `normal` mode (blockwise e4m3fn, per-SKU e4m3fnuz on gfx942); in `low-latency` mode the caller sends BF16 and the decode kernel quantizes to e4m3 internally (`use_fp8`). Combine is BF16. Runs on NVIDIA and AMD (H100/H200/B200 + MI300X/MI325X/MI355X), EP8 scale-up. Cross-node EP16 is functional (the internode RDMA path connects and the light case passes correctness) but its CPU-proxy throughput overruns the standardized per-case wall-clock budget on heavy token counts, so EP16 is an unsupported coverage row for now |
| NCCL EP | [NCCL EP](https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep): NVIDIA's native MoE dispatch/combine on the NCCL Device API — LSA (NVLink load/store) intra-node, GIN (GPU-Initiated Networking) inter-node — driven through the `nccl4py` bindings. `normal` mode selects the `HIGH_THROUGHPUT` algorithm (FLAT `[N, hidden]` receive, unweighted rank-sum combine); the `LOW_LATENCY` algorithm is implemented in the adapter but has no enabled cell (see the `ll_backends` note above). BF16 only: NCCL EP's FP8 machinery exists upstream but its RELEASE.md lists it unsupported/untested, so no FP8 case is emitted. NVIDIA-only and CUDA 13 only. EP8 scale-up on H100/H200/B200/B300 plus EP8 and EP16 on GB200/GB300, where EP16 stays inside the MNNVL scale-up domain. x86 EP16 scale-out is an unsupported coverage row: the cross-node GIN path faults inside `nccl_ep.cc` identically on RoCE and IB across four SKUs, a GDAKI limit rather than a fabric-selection one |

| JAX ragged A2A (TPU probe) | `jax.lax.ragged_all_to_all` on a 1-D device mesh over one host's ICI domain — the primitive JAX MoE stacks use for EP dispatch/combine. `normal` mode and BF16 only: the probe measures the interconnect collective itself, so there is no quantized dispatch path to label. Dispatch is a permute gather plus the ragged exchange; combine is the reverse exchange plus an fp32 scatter-add (unweighted rank-sum). `--transport-impl padded` swaps in a fixed-capacity `jax.lax.all_to_all` for JAX builds without the ragged primitive — a strictly worse model of production, never substituted silently, and named in the artifact. See the TPU Probe section for what this measurement is and is not |

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
pod), spawns ONE Kubernetes Job per shard onto a `tpu7x` node, runs every case in that one pod so
each case after the first reuses the node-local XLA compile cache, and harvests the result JSONs back
through the pod log. A red or partial leg still ships whatever it produced.

Three properties are deliberately weaker than the GPU family's, and each is named in the artifact:

- `measurement.timing_source` is `host-wallclock-blocked...`. There is no public JAX equivalent of
  `torch.cuda.Event`, so latencies are host wall-clock around `block_until_ready` on a jitted
  program and include host dispatch overhead the CUDA-event numbers exclude. Measured on tpu7x that
  overhead is a **~500 µs floor per call** — at T=1 it was ~98% of the reading, which made the
  small-token end of the ladder meaningless.

  The probe therefore **amortizes** by default: `--in-program-iters N` (default: the case's own
  `--iters`, so each trial is one timed call) chains N identical operations inside a single jitted
  program, dividing the floor by N. Each iteration's carry passes through
  `jax.lax.optimization_barrier`, which keeps the value unchanged while creating the data dependency
  that stops XLA from common-subexpressioning the N calls into one — without it the chain would
  report a fictitiously fast number, so a build lacking the barrier fails closed instead of
  amortizing. `--in-program-iters 1` opts out (`COLLX_TPU_IN_PROGRAM_ITERS` on the runner).

  The trade is statistical and is recorded, not hidden: with N > 1 each sample is the **mean** of N
  consecutive operations, so `p99` no longer captures a single slow operation. `timing_source`
  becomes `host-wallclock-blocked-amortized-xN` and `sampling.in_program_iterations` carries N, so a
  consumer can tell the two apart without reading the code. Amortization also divides the host-call
  count by N (10,240 → 1,280 calls per component at N=8), which is what keeps the prefill ladder
  inside its wall-clock budget.
- `components.stage` is `unavailable`: the permute is fused into dispatch, so there is no separate
  staging pass to time. The per-destination layout (offsets, sizes, gather indices) is precomputed on
  host and excluded from the timed region, the same treatment DeepEP's layout pass gets; the
  on-device permute gather and the combine scatter-add ARE timed, because production pays them.
- `implementation.oracle` is `probe-source-identity-and-exact-rank-sum`, narrower than the GPU
  harness's full per-expert transform oracle. It still proves real things: every dispatched copy is
  decoded back to the source token it claims to be and compared bit-for-bit, its landing offset is
  checked against the exchange plan, and the combine is compared against the exact expected
  unweighted rank-sum. The expert is the identity, so no per-expert transform is modelled.

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
