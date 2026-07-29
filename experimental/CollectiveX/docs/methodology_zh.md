[English](methodology.md) | [中文](methodology_zh.md)

# CollectiveX EP 基准测试方法

CollectiveX 调度专家并行（EP）通信基准测试，在真实加速器 allocation 上执行，并上传每次
运行生成的中立产物。它不验证、晋级、排名、推荐、筛选或隐藏这些产物，也不决定任何消费者
应展示什么。Frontend 读取中立的 matrix、result 与 summary 产物，自行决定 coverage 和
展示方式。本文说明用例如何被调度、测量、检查和记录，而不是 publication 或 qualification
contract。

## 产品边界

CollectiveX 是通信 microbenchmark，用于：

- 在同一 chip/topology 上比较不同 EP library；
- 在相同工作负载下比较不同系统的 EP latency 与 logical payload bandwidth；以及
- 显式呈现 unsupported、failed、invalid 和 unstable 用例，而不是隐藏它们。

若没有单独的 correlation study，它不预测 serving throughput。

## Matrix

已实现的工作负载为 `deepseek-v3`：hidden 7168、top-k 8、256 个 routed expert、
packed placement，并为每个 backend/topology 使用一个固定资源配置。Combine 始终为
BF16；dispatch precision 是扫描维度，包括 BF16 对照组，以及上游支持 FP8 dispatch 的
backend（DeepEP V2、MoRI、UCCL-EP）上的 FP8 dispatch（`bf16`、`fp8`）。
`normal` 模式由调用方预量化；`low-latency` 模式下 DeepEP 和 UCCL-EP kernel 从 BF16
内部量化，MoRI 仍由调用方预量化。本版本 NCCL EP 仅支持 BF16。每个 backend 的 precision
集合位于 `sweep_matrix.py` 的 `BACKEND_PRECISIONS`，backend 不会为不支持的 precision
生成用例。`normal` 用例使用 `layout-and-dispatch-v1`；`low-latency` 用例使用各
backend 的 decode-kernel semantics。

- `ep-core`：对工作负载 token ladder 进行 uniform routing。对于 `deepseek-v3`，
  decode 为 T=1..512 的二次幂，prefill 为 T=1024..8192 的二次幂。Ladder 与 workload
  一起定义在 `configs/sweep.json`。

`sweep_matrix.py` 将请求的 SKU、backend、EP size 和 token ladder 具体化为 matrix
文档，再提取严格的 per-shard control。`--only-sku`、`--exclude-skus`、`--ep-sizes`
和 `--precisions` 可选择子集；子集只会缩小 matrix，不会改变 contract。Matrix 在每次
dispatch 时生成，不存在冻结的 matrix digest 或锁定 case count。

| 系统 | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink，scale-up | 2x8 NVLink + RDMA，scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI，scale-up | 2x8 XGMI + RDMA，scale-out |
| GB200/GB300 | 2x4 MNNVL，scale-up | 4x4 MNNVL，scale-up |

物理 host 数量不定义 scope。两个 GB cell 都留在一个 72-GPU MNNVL scale-up domain 内。

Unsupported 组合会在 matrix 中显式分类，而不是被静默跳过。DeepEP V2 是 PR #605 引入
的 `ElasticBuffer`，固定包含上游 PR #630 的最小纯 scale-up 修复，以及 PR #640
排除 NCCL shared-memory mapping 的精确 library matcher。Scale-up 用例请求 NCCL
Device API LSA，并在 realized LSA team 未覆盖完整 EP world 时 fail closed。x86 EP16
scale-out 使用 GIN hybrid path，需要两个逻辑 scale-out domain、两个物理 RDMA rank，
每个 domain 含八个 scale-up rank。GB EP16 仍为 MNNVL scale-up，使用 LSA。MoRI EP8
在所有 CDNA SKU 上使用直接 `IntraNode` kernel；EP16 使用固定配置的 `InterNodeV1`
在 2x8 XGMI + RDMA 上运行。UCCL-EP 是 API 完全一致的 DeepEP 替代实现，通过普通
`libibverbs` 的 CPU-proxy GPUDirect RDMA transport 运行，不依赖 NVSHMEM/IBGDA；
其 scale-up 为单节点 `cudaIpc` over NVLink/XGMI，因此不使用 MNNVL。NCCL EP 是
NVIDIA 基于 NCCL Device API 的原生 MoE dispatch/combine，由 `nccl4py` 驱动；
`normal` 使用 `HIGH_THROUGHPUT` algorithm，其 FLAT `[N, hidden]` receive 与无权重
rank-sum combine 精确符合 `layout-and-dispatch-v1`。它仅支持 NVIDIA 和 CUDA 13，
在 H100/H200/B200/B300 上运行 EP8，在 GB200/GB300 上运行 EP8 和 EP16。x86 EP16
scale-out 在 `nccl_ep.cc` 内发生 fault，因此记为 unsupported coverage row。

第二种 `low-latency` 模式加入每个 backend 的 decode 优化 kernel。DeepEP 使用旧版
`deep_ep.Buffer` 低延迟 decode kernel（`low_latency_dispatch`/
`low_latency_combine`），返回 per-expert padded receive buffer，并在 source-side
combine 中应用 top-k gate weight。已纳入范围的单节点 EP8 cell 使用节点内 NVLink
低延迟路径；只有多节点 scale-out EP16 才通过 NVSHMEM/IBGDA 使用 `/dev/gdrdrv`。
MoRI 使用 `IntraNodeLL`，这是 single-call、纯节点内的 decode kernel，保持与 throughput
`IntraNode` 相同的按 rank 去重 compact layout 和无权重 combine。低延迟仅作为
decode-phase addition，其可运行集合由 registry 的 `ll_backends` 逐 cell 控制。当前
启用 DeepEP V2 EP8（H100/H200/B200）、MoRI EP8（MI300X/MI325X/MI355X）和
UCCL-EP EP8（H100/H200/B200）。AMD SKU 不启用 UCCL-EP 低延迟 kernel，因为它会在
AMD CU 数量下触发 warp-group assertion。NCCL EP adapter 虽实现 `LOW_LATENCY`
algorithm，但当前不在任何 SKU 的 `ll_backends` 中：已发布 wheel 的 signal protocol
会消费 stale peer signal 并导致 pipeline wedge，见
[NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303)。某个
SKU/backend/EP/mode cell 是否被尝试是 capability 事实；其成功与否仅由生成的产物决定。

## 工作负载身份

使用 `configs/sweep.json` 中的 workload seed，在 global token batch 上生成一个
deterministic workload，并按 source rank 切片。带 key 的 BLAKE2b counter 对
`(token, slot, attempt, stream)` coordinate 生成逐字节一致的 expert index 和 gate
weight；在用例成功前，harness 会证明所有 rank 的 realized routing trace 完全一致。

Routing traffic 区分：

- token-expert assignment：决定 expert compute load；以及
- 按 rank 去重的 token payload copy：决定 EP activation traffic。

Adapter 不得自行生成 routing，也不得将两种数量互相解释。

## 测量

Normal 模式使用 `layout-and-dispatch-v1`：dispatch 计时包含 layout 与通信，combine
通过无权重 rank-sum path 返回 activation payload。Expert-output staging 位于独立
combine 计时之外，但包含在成对 roundtrip 中。每个 component 声明 availability、
origin 和 sample count。仅支持 paired API 时，isolated component 报告 null；
`isolated_sum` 为派生值。产物记录 mode，使 reader 能分离不同 measurement contract。

所有被测 component 使用 `configs/sweep.json` 中唯一的固定 timing profile：

- 256 个 trial x 8 次计时迭代，共 2048 个 observation；
- 在每个 trial/point 测量每个可用 component 前，执行 32 次同步的完整
  dispatch-stage-combine warmup；
- 每个 trial 轮换 component measurement order，并按 trial 轮换 token ladder，使
  每个 component 均匀出现在所有顺序位置；以及
- 每次迭代先取跨 rank 最大 latency，再计算 nearest-rank p50/p90/p95/p99。

Roundtrip p99 是主要 latency。Decode 和 prefill 只表示一个 MoE-layer collective
对应的 serving regime，不改变相同 shape 的 timed primitive。沿 ladder 升序执行时，
每个 shape 在 correctness check 前会先运行 8 次不计时的完整 roundtrip，使 clock、
fabric 和 buffer state 稳定。所有计时都在每个 shape warmup 并通过检查后开始。
Conditioning round 不被测量或输出。

Logical payload bandwidth 定义为：

`logical_payload_bytes / measured_latency_seconds`

Payload byte 使用按 rank 去重的 token-rank activation，不包含 expert metadata、padding
和 backend buffer capacity。BF16 每个值为 2 byte，不带 scale payload；FP8 dispatch
每个值为 1 byte，DeepEP 和 UCCL-EP 的 blockwise codec 还包含每 128 block 的 FP32
scale，MoRI 的普通 e4m3 cast 不包含；combine 始终为 BF16。因此 dispatch 与 combine
方向可能具有不同 byte count，roundtrip 为对应字段之和。该按 rank 去重计数对 normal
layout 是精确值；low-latency layout 按 `(token, expert)` assignment 发送，因此当同一
token 的多个 expert 位于相同 destination rank 时，logical count 是 kernel 实际移动
byte 的下界。主要指标 latency 直接测量，不受此影响。若没有定义 primitive model 或
transport counter，不输出 algorithm bandwidth、bus bandwidth、wire utilization 或
physical-link utilization。Logical bandwidth 不得标记为 physical bandwidth。
Payload 和 token rate 命名为 `rate_at_latency_percentile`，即 byte 或 token 除以匹配的
latency percentile；它们是 p99 latency 下的 lower-tail service rate，不是倒数分布的
p99 percentile。

## 正确性

独立于实现的 oracle 使用 expert-specific deterministic transform，避免错误 expert
routing 通过 identity roundtrip。它对每个 rank 和 point 检查：

1. destination rank/expert、source token、multiplicity、gate weight 和 receive count；
2. 计时前的 dispatched payload 与 metadata；
3. 计时前的 combined output；
4. 所有 timed sample 中语义 input 未被修改；以及
5. 计时后的 dispatched payload/metadata 与 combined output。

Normal-mode adapter 使用仅 activation、无权重的 rank-sum combine。Oracle 先构建每个
rank 的 gate-weighted expert aggregate，再从实际通信的值推导 expected combine，
重现两级 reduction：destination rank 将 FP32 aggregate 转为 payload dtype（BF16）；
共享一个 scale-up domain 的 rank 在 FP32 中归约；每个 domain 将 aggregate 转为 BF16
后再发送到 scale-out。可完全放入一个 scale-up domain 的 group（`ep_size <=
scale_up_domain`，即所有 EP8 与 MNNVL EP16）只有一个 domain，不产生 scale-out
rounding；多节点 RoCE EP16 每个 node 携带一个 BF16 partial。对 per-domain cast 的建模
使最大逐元素相对误差 gate 能稳定保持低于 `8 * 2^-8`，分母下限为 0.02。

Low-latency adapter 使用 source-side gate-weighted combine：kernel 将每个 expert 返回
message 乘以该 assignment 的 top-k weight，因此 adapter stage 未加权的 per-expert
transform，而专用 per-`(source, expert)`-slot oracle 将预期 combine 推导为 per-expert
BF16 message 的 gate-scaled sum。由于低延迟 kernel 在 source rank 归约，不存在
per-domain intermediate。Delivered assignment multiset 和 per-expert count 会与 routing
trace 对比。在 FP8 dispatch 下，oracle 在 dispatched-payload compare 和 combine
expectation 前对语义 payload 应用 backend 的精确 per-token cast round-trip，因此
payload match 保持 bit-exact，使用同一 combine gate。任何 rank 或 point 失败都会使结果
中的用例不合格。由于 native receive slot 可非确定性分配，physical receive order 不被视为
正确性属性。

## 结果产物

每个 raw case 文档包含 `record_type: "case-attempt"` 和单一 `version`，并包含：

- `identity`：`case_id`、`attempt_ordinal`、`case_factors`（SKU 与调度用例，包括
  backend、EP size、mode、precision、phase、suite、workload 和 topology coordinate）
  及 `allocation_factors`（run id、run attempt、source SHA）；
- `workload`：`cross_rank_consistent`，表示 routing trace 是否已证明跨 rank 一致；
- `measurement`：dispatch/combine dtype、semantics、`sampling` 和 per-point `rows`；
- `implementation`：backend name 与 kernel generation；
- `topology`：请求的 SKU/product、placement、node、scale-up domain、transport 和
  world size；
- `provenance`：mounted image tag 与 source SHA；以及
- `outcome`：`status`（`success` 或 `invalid`）与 `reasons`。

Runtime `vendor` 来自 registry metadata，而不是由 CUDA/HIP 推断。它接受任意规范化的
vendor identifier；独立的 `runtime` registry 字段选择当前已实现的 CUDA 或 HIP 执行
路径。这样 private 的非 AMD/NVIDIA 系统可以保留正确的结果身份，同时不会假装新的
accelerator runtime 能复用不兼容的 backend 或 launcher。

每个 `rows` entry 包含 point latency、byte accounting、token rate、correctness、load
和 fanout；per-point statistic 在原位汇总，不拆成单独文档。每个 dispatched case 精确写入
一个 raw result document；unsupported 或从未运行的 cell 不生成 synthetic record。

## 身份

Identifier 是可读的 factor string：

- `case_id`：`{sku}-{backend}-{workload}-{mode}-{phase}-ep{ep}-{routing}-{precision}`，
  每个 factor 都经过 slug normalization；以及
- `attempt_ordinal`：用于区分同一 `case_id` 重复执行的正整数。

Backend source pin 位于 `runtime/common.sh`，并通过精确的 fetched-commit 比较强制执行；
loaded DeepEP V2 build 还会检查必需的 `ElasticBuffer` API。

这些 ID 允许消费者对匹配配置分组并区分不同配置。Benchmark 本身不计算 cohort、
controlled comparison、sensitivity pair、eligibility 或 recommendation，均由 reader
决定。

## 执行隔离

每个非 MNNVL scale-out 用例都使用 operator 固定的 socket 和 RDMA selector。Launcher
拒绝缺失或不完整的 profile，然后在 backend 初始化前，探测每个 allocated node 上的
configured interface、active HCA port 和 configured GID。它不会替换为 default route、
继承的 runner environment 或 transport fallback。Scale-up 和 MNNVL 用例会清除这些
profile；scale-out CUDA path 强制 `NCCL_NET=IB`，HIP path 让 RCCL 自行选择 plugin，
两者都使用 exact HCA matching。Scale-out 还设置 `NCCL_IB_MERGE_NICS=0`，避免
dual-port NIC fusion 禁用 DeepEP V2 EP16 hybrid path 所需的 NCCL GIN；
`rail_isolated` fabric 还会设置 `NCCL_CROSS_NIC=0`。Selector 来自 tracked platform
registry，可被 operator config 覆盖，并只出现在 mode-0600 private log 中。

Repository staging 使用 checkout 和 workflow workspace 之外、预先存在、由 runner
拥有且不可由 group/world 写入的 shared base。Parent process 在复制前解析精确 execution
child；backend preparation 随后从该 tree 在所有 allocated node 上运行。Cleanup 等待
allocation teardown 确认后，只删除该 child。DeepEP V2 source 在 allocation 前从固定
revision 获取，初始化固定的 `fmt` submodule，并应用所需 local patch。

H200、B200 和 B300 在 operating-system account home 对 compute-visible 时，可从该
位置派生 private base。H100 则从 shared container directory 的 sibling 派生，绝不放在
image storage 下。Canonical B300 execution 忽略旧版 operator `stage_dir`，始终从已验证的
shared account home 派生 base；其 UID-mapped Actions shell 只在 owner 与 private parent
owner 匹配时接受该精确 base。Execution-ID suffix 隔离并行 B300 worker。当前 NFS export
可能将新建 base 实现为 UID 0；只接受该创建路径，预先存在的 root-owned base 会被拒绝。
Canonical GB300 execution 同样忽略旧版 group-writable `stage_dir`，并在已验证的
compute-visible account home 下派生 execution-specific private base。

## Image 固定与 build 隔离

Enroot 将配置的 image tag 导入按 image tag 与 image platform 分键的 per-run-scoped
squash，因此不会跨 run 复用导入的 filesystem。Image 内置 DeepEP 也会检查精确 package
version 与预期 API。Source-built DeepEP V2 使用单独的 mode-0700 cluster-local cache，
且仅 mount 为 `/cx-cache`。其路径绑定 CPU/GPU architecture、image 和 upstream commit。
Cache 不是产物；per-execution source/result stage 保持隔离且可销毁，runtime probe 在
复用前 fail closed。Runner UID 位于可信 cluster boundary 内：该 cache 防止 stale 或
意外修改，而不是防御恶意 same-UID job。只有尚未发布的 partial build 可以自动 reset；
integrity 或 runtime check 失败的 cache 会保留并被拒绝，避免并发 allocation 丢失正在
使用的文件。

## 中立产物交付

不存在 result server、attached store 或 managed object store。每个 shard 运行一次
allocation，生成 per-case result JSON 与小型机械 summary，并使用 `always()` 上传 GitHub
artifact，使红色或部分完成的 run 仍能保留结果。用例是否成功完全由自身返回码决定；上传前
不存在 completeness 或 privacy validation，failed 或 unsupported cell 不生成 synthetic
record。

任何步骤都不会晋级 run、构建 dataset 或推进 channel；artifact 就是输出。所有下游展示和
比较都由消费者负责。

## 旧版数据

历史 numeric schema 3-5 不属于本 benchmark 的 artifact。它们仍是历史诊断证据，但当前
sweep 不生成或消费这些 schema。
