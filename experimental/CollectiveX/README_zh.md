[English](README.md) | [中文](README_zh.md)

# CollectiveX

CollectiveX 是实验性的混合专家（MoE）专家并行（EP）通信基准测试。它测量不同 EP
库和加速器系统上的 dispatch、combine 及成对 roundtrip 延迟，并上传中立的结果产物。

CollectiveX 负责调度基准测试、在真实资源分配上执行测试，并上传每次运行生成的中立产物。
它不验证、晋级、排名、推荐或筛选这些产物，也不决定消费者应展示什么内容。所有下游展示与
比较都由消费者负责。完整测量方法见
[docs/methodology_zh.md](docs/methodology_zh.md)。

## 执行配置

工作负载采用 packed placement，并为每个 backend/topology 使用一个固定的
`fixed-profile` 资源配置，不进行调优扫描。Combine 始终使用 BF16；dispatch precision
是扫描维度：包含一个 BF16 对照组，并在上游支持 FP8 dispatch 的 backend（DeepEP V2、
MoRI、UCCL-EP）上加入 FP8 dispatch。`normal` 模式由调用方预量化；`low-latency`
模式下，DeepEP 和 UCCL-EP kernel 从 BF16 内部量化，而 MoRI 仍由调用方预量化。
本版本 NCCL EP 仅支持 BF16，因此只生成对照组。覆盖范围仅包含 uniform routing。
用例运行在以下两种模式之一：

- `normal` 使用 `layout-and-dispatch-v1`、按 rank 去重的 token payload，以及仅包含
  activation、无权重的 rank-sum combine，并运行完整的 decode 和 prefill ladder。
- `low-latency` 使用各 backend 的 decode 优化 kernel：DeepEP 使用旧版
  `deep_ep.Buffer` IBGDA `low_latency_dispatch`/`low_latency_combine`；UCCL-EP
  通过其 CPU proxy transport 复用相同的旧版低延迟 kernel；MoRI 使用 `IntraNodeLL`
  kernel。该模式仅覆盖 decode/EP8，并按 SKU capability 启用，因此可运行集合与
  `normal` 不同，由每个 SKU 的 `ll_backends` registry 条目控制。当前包括 H100/H200/B200
  上的 DeepEP V2 EP8、MI300X/MI325X/MI355X 上的 MoRI EP8，以及 H100/H200/B200
  上的 UCCL-EP EP8。AMD SKU 保留 UCCL-EP normal 模式，但不启用其会触发
  warp-group assertion 的低延迟 kernel；NCCL EP 暂无低延迟条目，相关 decode kernel
  仍受 [NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303) 影响。
  单节点 EP8 走节点内 NVLink/XGMI 低延迟路径，不需要 `/dev/gdrdrv`；只有多节点
  scale-out EP16 才涉及 NVSHMEM/IBGDA。

固定 timing profile 位于 `configs/sweep.json`：每个 point 执行 256 个 trial，每个
trial 含 8 次计时迭代，共 2048 个 sample；在每个 trial/point 测量每个 component
前，执行 32 次同步的完整 dispatch-stage-combine warmup。每个 trial 会轮换 component
测量顺序，使每个计时 component 均匀出现在各位置；每次迭代先取跨 rank 最大延迟，再计算
nearest-rank p50/p90/p95/p99。Roundtrip p99 是主要延迟指标。基于带 key 的 BLAKE2b
counter 在所有 runtime 上生成逐字节一致的 routing 和 gate weight。

正确性由独立于实现的 oracle 检查。Oracle 重现 backend 的两级 reduction：先在
scale-up domain 内以 FP32 归约，再将每个 domain 的 partial 转为 BF16 后用于
scale-out。Combine gate 要求最大逐元素相对误差小于 `8 * 2^-8`，分母下限为 0.02，
该规则同时适用于 scale-up 和多节点 scale-out topology。在 FP8 dispatch 下，oracle
对语义 payload 应用相同的 per-token cast round-trip，因此 dispatched-payload 比较仍
保持 bit-exact，combine gate 不变。任一 rank 或 point 失败都会使该用例在其结果中
标记为不合格。

Matrix 覆盖 H100、H200、B200、B300、GB200、GB300、MI300X、MI325X 和 MI355X。
`sweep_matrix.py` 生成请求的 SKU、backend、EP size 和 token ladder，然后提取严格的
per-shard control，并拒绝缺失、过期、格式错误或被修改的 shard control。
`--only-sku`、`--exclude-skus`、`--ep-sizes` 和 `--precisions` 可选择子集。
Matrix 在每次 dispatch 时生成，不存在冻结 digest 或锁定的 case count。

| 系统 | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink，scale-up | 2x8 NVLink + RDMA，scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI，scale-up | 2x8 XGMI + RDMA，scale-out |
| GB200/GB300 | 2x4 MNNVL，scale-up | 4x4 MNNVL，scale-up |

物理 host 数量不决定 scope：两个 GB topology 都位于同一个 72-GPU MNNVL scale-up
domain 内。

| Backend | 当前范围 |
|---|---|
| DeepEP V2 | `normal` 模式使用 PR #605 的 `ElasticBuffer`，并包含上游 #630 和 #640 的精确修复：scale-up 使用 LSA，x86 EP16 scale-out 使用 GIN。FP8 dispatch 通过 `use_fp8_dispatch`（blockwise e4m3fn）与 BF16 并列。`low-latency` 模式使用旧版 `deep_ep.Buffer` IBGDA decode kernel（per-expert padded layout、weighted combine、`use_fp8` e4m3fn），仅覆盖 decode/EP8 |
| MoRI | `normal` 模式在所有 CDNA SKU 上以直接 `IntraNode` kernel 执行 scale-up EP8，并为 2x8 XGMI + RDMA 的 EP16 固定使用 `InterNodeV1`。`low-latency` 模式选择 `IntraNodeLL` decode kernel，仅覆盖 decode/EP8。FP8 dispatch 由调用方预量化：gfx942 使用 per-SKU e4m3fnuz，gfx950 使用 e4m3fn；combine 保持 BF16（`quant_type=none`） |
| UCCL-EP | [UCCL](https://github.com/uccl-project/uccl) EP 是 API 完全一致的 DeepEP 替代实现；其 CPU proxy 通过普通 `libibverbs` 发起 GPUDirect RDMA，不使用 NVSHMEM/IBGDA，并通过软件处理消息顺序、atomic 和 flow control。Scale-up 是单节点 `cudaIpc` over NVLink/XGMI，不使用 MNNVL。`normal` 模式使用旧版 `Buffer` `dispatch`/`combine`；`low-latency` 复用旧版低延迟 kernel，仅覆盖 decode/EP8。`normal` 模式的 FP8 dispatch 由调用方预量化；低延迟模式由 decode kernel 内部量化为 e4m3；combine 为 BF16。它在 NVIDIA 和 AMD 的 EP8 scale-up 上运行。EP16 虽可建立连接并通过轻量正确性用例，但重 token 数下超出统一的 per-case wall-clock budget，因此当前记为 unsupported coverage row |
| NCCL EP | [NCCL EP](https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep) 是 NVIDIA 基于 NCCL Device API 的原生 MoE dispatch/combine：节点内使用 LSA，节点间使用 GIN，并通过 `nccl4py` binding 驱动。`normal` 模式选择 `HIGH_THROUGHPUT` algorithm；低延迟 adapter 已实现但当前没有启用条目。此版本仅支持 BF16，且仅适用于 NVIDIA 和 CUDA 13。它在 H100/H200/B200/B300 上运行 EP8，在 GB200/GB300 上运行 EP8 和 EP16；GB EP16 仍处于 MNNVL scale-up domain 内。x86 EP16 scale-out 在 `nccl_ep.cc` 内发生 fault，因此记为 unsupported coverage row |

DeepEP V2 指 [DeepEP PR #605](https://github.com/deepseek-ai/DeepEP/pull/605)
引入的 `ElasticBuffer`，而不是更新版本的旧版 `Buffer` build。固定源代码来自
[PR #630](https://github.com/deepseek-ai/DeepEP/pull/630) 的 head，其 parent 是
#605 merge tree，并应用上游 [PR #640](https://github.com/deepseek-ai/DeepEP/pull/640)
精确的一行 library matcher。前者修复 GIN 不可用时纯 scale-up 初始化；后者避免将 NCCL
shared-memory mapping 误判为重复 NCCL library。Scale-up 用例请求 NCCL Device API
LSA，并在 realized LSA team 未覆盖完整 EP world 时 fail closed。x86 EP16 scale-out
要求 GIN hybrid path、两个逻辑 scale-out domain、两个物理 RDMA rank，以及每个 domain
八个 scale-up rank；GB EP16 保持 MNNVL scale-up，因此使用 LSA。是否尝试某个
SKU/backend/EP cell 是 capability 事实；是否成功由基准测试返回码决定。

## Workflow 与产物

`.github/workflows/collectivex-sweep.yml` 包含两个 job。`setup` 生成 public-SKU matrix
（输入为 `backend`、`only_sku`、`exclude_skus`、`ep_sizes`）并上传 matrix。
`sweep` 为每个 matrix entry 提取严格且被忽略的 `.shards/<id>.json` control，每个
shard 执行一次 allocation，在需要时于 allocation 前获取固定版本的 DeepEP source，
并使用 `always()` 上传结果，使红色或部分完成的运行仍能保留产物。

每个 shard 生成 per-case result JSON 和一个小型机械 summary。用例是否成功完全取决于
自身返回码；不存在 completeness 或 privacy validation 步骤，失败或 unsupported cell
不会生成 synthetic record。任何步骤都不会将 run 晋级、构建 dataset 或推进 channel；
中立产物就是最终输出，消费者自行决定展示方式。

Workflow 不接收或上传 operator credential；runner-local override 和 selector 均留在
runner 上。每个步骤的 runner log 保留于 runner 以便 postmortem，结果产物只包含
methodology 中列出的字段。

## Runner 配置

每个 SKU 的 Slurm 和 storage 值来自 registry 中的 tracked baseline。可选的 runner-local
JSON 文档位于 `$XDG_CONFIG_HOME/inferencex/collectivex.json`，或由
`COLLECTIVEX_OPERATOR_CONFIG` 指定；它可以逐字段覆盖 baseline。未知 runner、未知字段、
重复 key 和非 JSON 输入都会 fail closed，配置不会作为 shell 执行。GHA 不传 operator
secret，因此没有本地文档时，SKU 完全使用 tracked baseline。

所有 per-SKU platform 数据都位于 `configs/platform_config.json` registry，包括
architecture/product、vendor、accelerator runtime、container image 与 platform、
固定 placement、launcher、可运行的 backend/EP pair、scale-out `fabric` identity
（NIC 与 switch）、tracked operator default 和 scale-out RDMA selector。`vendor` 是
任意规范化 metadata，不限于 AMD 或 NVIDIA；`runtime` 选择当前已实现的 CUDA 或 HIP
执行路径。因此新增其他 vendor 不需要修改 vendor allowlist，但新的 runtime 或 collective
实现仍需兼容的 backend 与 launcher。Operator 文档可以覆盖 default。Launcher 只声明并
检查自身真正需要的字段。`sweep_matrix.py` 从 placement 字段推导 EP topology；默认 sweep
包含每个已注册 SKU。

每个被选中的非 MNNVL EP16 placement 还需要 operator 批准的 `socket_ifname` 和
`rdma_devices`；也允许配置 `ib_gid_index`、`rdma_service_level`、
`rdma_traffic_class` 和 `rail_isolated`。Service level 与 traffic class 会映射到
MoRI 的 RDMA/IO QoS 环境。CollectiveX 不会启发式选择 management route 或 HCA。
Allocation 后，每个非 MNNVL scale-out node 必须证明所有已配置 interface 和 active HCA
port 存在，之后才可初始化 backend。Scale-up 和 MNNVL job 会清除这些 override。
Scale-out NCCL/RCCL 固定为 `IB` 并使用 exact-match HCA selector，使 socket fallback
直接失败而不是被误标为 RDMA。Scale-out 还设置 `NCCL_IB_MERGE_NICS=0`，避免 dual-port
NIC fusion 禁用 DeepEP V2 EP16 hybrid path 所需的 NCCL GIN；`rail_isolated=1` 的
multi-plane fabric 还会设置 `NCCL_CROSS_NIC=0`。

仅当所有已选 HCA port 都报告 Ethernet link layer 时才应用 `ib_gid_index`，用于选择
operator 批准的 RoCE GID。原生 InfiniBand profile 保留显式 HCA 与 service level
pinning，但不设置 RoCE-only GID override。混合 Ethernet 与 InfiniBand 的 HCA list
会被拒绝。

`stage_dir` 是 checkout 和 workflow workspace 之外、预先存在且由 runner 拥有的
非 symlink base，不可由 group 或 world 写入，并在 runner 与所有 allocated node 上以
同一路径可见。Job 只创建带 marker 的 mode-0700 execution child，验证跨节点读写可见性，
并在 allocation teardown 后仅删除该 child；不会 mount runner checkout，也不会在 AMD
image storage 下创建 stage。AMD operator row 未提供 `stage_dir` 时，runner 会在其标准
`_work` 目录旁的共享 runner filesystem 上派生 private base；root-owned squash cache
永远不会作为 repository stage。

H200、B200 和 B300 runner 可省略 `stage_dir`；其 isolated execution child 会创建在
已验证的 operating-system account home 下的 mode-0700 base 中，与 workflow 临时
`HOME` 无关。H100 也可省略 `stage_dir`，其 private base 位于 shared container directory
旁而非目录内，以确保 compute-visible。Canonical B300 execution 忽略旧版配置的
`stage_dir`，总是使用已验证且 compute-visible 的 account-home base；execution-ID
suffix 用于隔离并行 B300 worker。Canonical GB300 execution 同样忽略旧版 group-writable
`stage_dir`，并在已验证的 compute-visible account home 下派生 execution-specific
private base。每个 node 的 backend preparation 都从该 staged tree 运行。

Enroot 会按 image tag 和 image platform 将配置的 image 导入 per-run-scoped squash，
因此不会跨 run 复用导入的 filesystem。Image tag 与 platform 是 per-SKU registry 字段。
DeepEP V2 source pin 位于 `runtime/common.sh`，其 build 会获取并验证固定 commit、检查
`ElasticBuffer`，并缓存于按 architecture、image 与 commit 分键的 cluster-local build
cache。只有固定的 `/cx-cache` mount 会进入 container。

## 本地检查

```bash
python3 -m unittest discover experimental/CollectiveX/tests -p 'test_*.py'
python3 experimental/CollectiveX/sweep_matrix.py --backend all --out /tmp/cx-matrix.json >/dev/null
bash -n experimental/CollectiveX/runtime/*.sh experimental/CollectiveX/launchers/*.sh
```

核心路径为 `configs/`、`sweep_matrix.py`、`summarize.py`、`bench/`、`runtime/`、
`launchers/` 和 `tests/`。
