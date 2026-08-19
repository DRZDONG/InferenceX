[English](README.md) | [中文](README_zh.md)

# CollectiveX

CollectiveX 是一个实验性的 MoE 专家并行通信基准测试。它测量不同 EP 库和加速器系统上的 dispatch、
combine 以及配对 roundtrip 延迟，然后上传中立的结果产物。

CollectiveX 负责调度基准测试、在真实分配的资源上执行测试，并上传每次运行生成的中立
产物。它不会验证这些产物，也不会对其进行推广、排名、推荐、选择，或
决定消费者显示什么。任何下游显示或比较均由消费者
负责。完整的测量方法见 [docs/methodology.md](docs/methodology_zh.md)。

## 执行配置

工作负载采用紧凑放置，并为每种 backend/拓扑使用一个固定的 `fixed-profile` 资源配置。
不进行调优扫描。Combine 始终使用 BF16。Dispatch 精度是一个扫描
维度，包括一个 BF16 对照组，以及在所有上游支持 FP8 dispatch 的 backend
（DeepEP V2、MoRI、UCCL-EP、FlashInfer EP）上增加一个 FP8 dispatch；在 `normal` 模式下由调用方预量化（在
`low-latency` 模式下，DeepEP 和 UCCL-EP 内核会在内部从 BF16 量化。MoRI 仍由
调用方预量化，而 FlashInfer 没有 `low-latency` 路径）。调用方的量化开销计入
测得的 dispatch，因为生产环境中的前向传播会在关键路径上承担这项开销。本次发布中的 NCCL EP
仅支持 BF16，因此它只生成对照组。覆盖范围仅限均匀路由。用例在以下两种模式之一运行：

- `normal` 使用 `layout-and-dispatch-v1`、按 rank 去重的 token 载荷，以及仅针对激活值、
  无权重的 rank-sum combine。它运行完整的 decode 和 prefill 阶梯。
- `low-latency` 使用各 backend 针对 decode 优化的内核系列：在 DeepEP 上使用旧版
  `deep_ep.Buffer` IBGDA `low_latency_dispatch`/`low_latency_combine`（按专家填充的接收
  和源端门控加权 combine）。在 UCCL-EP 上使用相同的旧版 `Buffer` 低延迟内核，
  在所限定的 EP8 运行中通过 NVLink 使用 `cudaIpc`，而不是其 CPU 代理传输。在 MoRI 上使用 `IntraNodeLL` 内核（单次调用、
  纯节点内、采用与 `IntraNode` 相同的紧凑布局和无权重 rank-sum combine）。它是一个
  仅限 decode 阶段、受各 SKU 能力约束的附加项，其可运行集合与 `normal` 不同，因此
  由各 SKU 的 `ll_backends` 注册表条目启用（目前包括 DeepEP V2：在
  H100/H200 上为 EP8，在 B200 上为 EP8 *和 EP16*（nscale 裸金属池，其由 gdrdrv 支持、通过原生 IB 运行的 IBGDA
  正是低延迟 scale-out 所需的条件，而虚拟化池均不具备），以及
  GB200/GB300，其 EP16 仍位于 MNNVL scale-up 域内；
  此外还包括 MI300X/MI325X/MI355X 上的 MoRI EP8，以及仅限 H100/H200/B200 的 UCCL-EP EP8（UCCL 的低延迟主机端
  断言 `kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group` 在 AMD 上无法成立，因为
  `kNumMaxWarpGroups` 为 16；上游已将 `kNumMaxTopK` 从 9 提高到 16（uccl#1016，2026-07-13），而
  我们固定的版本晚了六天。对于任何 CU 数量，该乘积均为 16，因此这是一个有明确时间点的回归，
  而不是硬件限制；AMD SKU 保留 UCCL-EP normal 模式，但不启用 LL）；
  还包括全部六种 NVIDIA SKU 上的 NCCL EP EP8；在单句柄修复消除了曾导致其卡死的
  [NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303) 信号别名问题后，这些支持已恢复。
  B300 将 `candidate` NCCL EP 作为其*唯一*的低延迟行，因此它没有生产级
  decode 覆盖。DeepEP V2 在 B300 上完全不生成 LL 行（原因是下方 backend 表中的 IBGDA 地址句柄壁垒），
  而 `_ll_runnable` 仅添加可运行的单元格，因此该壁垒在这里以文字说明，
  而不是分类矩阵行的形式出现）。
  限定范围的单节点 EP8 运行走节点内 NVLink/XGMI
  低延迟路径（不需要 `/dev/gdrdrv`，已在缺少该设备的 H200 上验证）。只有多节点 scale-out（EP16）运行才会
  在线路上使用 NVSHMEM/IBGDA 传输载荷。旧版 Buffer 在 EP8 下仍会
  自动启用 IBGDA，这正是 B300 在此失败的原因。

用例采用 `configs/sweep.json` 中固定的计时配置：256 次 trial x 8 次计时迭代（每个组件 2048 个
样本），并且在每个 trial/点对每个被测组件进行测量前，先完成 32 次同步的完整 roundtrip 预热。
每个 trial 都会轮换组件的测量顺序，使每个计时组件都能占据序列中的每个位置；
每次迭代先取跨 rank 最大值，再计算 nearest-rank p50/p90/p95/p99。一个带键的 BLAKE2b 计数器会在
每个 runtime 上生成逐字节完全相同的路由和门控权重。

这些组件全部测量**全新进入**（每个计时窗口前后都会排空 GPU），即
空闲流水线的延迟，而不是 decode 循环实际承担的延迟。因此，每一行还会包含**链式
配对周期**：4 次 trial x（连续发出 128 个 dispatch→combine 对，中间不做主机同步，丢弃前
16 个作为流水线填充）= 448 个观测值，并通过中位数进行跨 rank 归约。每个 trial 运行
**两条同级链**（先运行仅携带逐操作事件的 floors 链，再运行仅携带外层配对事件的 period 链），
因为第一版在每对中使用六个事件的单链方案；当设备运行速度超过主机时，其中四次内部 `record()` 调用
会被计入 period，从而把一个约为 10–30µs、几乎恒定的主机开销发布成传输开销（T=1 时增加 +20–38%，
且影响整个设备群）。对于每个包含该字段的行，`components.pair_period` 都是
核心延迟指标（在 b200/h200/gb200 手工参考结果通过两遍设备群产物确认后，于 2026-08-06 发布），
而 `summarize.py` 会说明带星号的列在两种情况下分别包含什么。floors 链发布 `chain_floor_us`，即
每个操作窗口的跨 rank 最小值；period 链还会产生 `chain_health.pair_spread_us`
（跨 rank 节拍证明）、`interpair_gap_us`（已发布窗口之外的每对开销；它既是防止检测开销重新混入的
回归保护，也是一项判别指标，可避免把由同步主导的 `period − Σfloors` 差值误读为该缺陷再次出现。有关
该残差每种符号的含义，请参阅方法文档中的 `chain_floor_us` 项）以及
`settle_drift_us`（后半段 period 减去
前半段 period，是 `chain_drop` 仅作假设时所缺少的收敛证明）。链式逐操作
*中位数*从不发布：rank 间等待会落在某个 rank 阻塞所在的操作窗口中，对每个 rank 而言稳定，
但在不同 rank 之间具有任意性，并且只有在配对总计中才守恒。现有内容均未被重命名或重新定义，
扫描的 `version` 仍为 1，因此消费者应以
`components.pair_period` 是否存在作为判断依据。

链式机制会经过两重检查：每个链式 trial 自身最终的 combine 输出，都会通过完全相同的代码路径
与一个已排空的配对进行比较（`correctness.chain_last_output_passed`，
任何差异的大小记录在 `correctness.chain_last_output_error` 中）；同时，完整 oracle 会在每个阶梯点
针对链结束后留下的状态运行一次
（`correctness.post_chain_state_passed`）。第二项始终作为门禁。第一项仅在
链按配对执行 staging 时作为门禁；若 staging 被提升到链外，则该项为 `null`。在这种提升下，
两种机制都不会 combine 与其自身 dispatch 相匹配的输入，因此二者不可比较。该边界是实测得出的，
而非假设：相同的 h100 用例在采用提升时，差异为 combine 容差的 1000×–2966×，不采用提升时则
差异恰好为零。参见方法文档的 Correctness 章节。这里的 null 是一个有意保留且边界明确的缺口。仅发生于 FP8、
仅发生于自由运行且无状态的损坏不会触发任何红色门禁，而 `CX_FP8_CONSUME=dequant` 逃生口
是对此进行探测的常驻手段。

每一行中的 `roundtrip` 都表示先 dispatch 再 combine（即传输）。专家输出 staging 位于
其外部，并作为 `stage` 单独报告。在 FP8 下，该组件是测试框架脚手架，
代替专家 GEMM；而生产环境中的专家 GEMM 会原生使用 FP8 操作数，而不会
物化 BF16 副本，因此不得将 `stage` 累加到总计中，也不得在 backend 之间比较。
在此变更之前测量的行，会将 MoRI BF16 和
FlashInfer BF16 的 staging 拷贝包含在链内；扫描的 `version` 在此变更前后仍保持为 1，因此
只有 `implementation.stage_excluded_from_roundtrip` 以及是否存在 `stage` 组件
能够区分这两代结果。完整契约见 [docs/methodology.md](docs/methodology_zh.md)。

正确性通过一个与具体实现无关的 oracle 检查，该 oracle 会复现 backend 的
两级归约：先在 scale-up 域内使用 FP32，再将每个域的部分结果转换为 BF16，
用于 scale-out 发送。Combine 门禁要求最大逐元素相对误差严格低于 `8 * 2^-8`
（分母下限为 0.02）；这一标准同时适用于 scale-up 和多节点 scale-out 拓扑。
在 FP8 dispatch 下，oracle 会对其语义载荷应用相同的逐 token 转换 round-trip，
因此 dispatch 载荷比较仍然逐位精确，combine 门禁也保持不变。量化是被建模的，
而不是通过容差放行的。任何 rank 或点失败，都会使该用例失去写入结果的资格。

矩阵覆盖 H100、H200、B200、B300、GB200、GB300、MI300X、MI325X、MI355X 和 TPU v7。
`sweep_matrix.py` 会实例化
所请求的 SKU、backend、EP 大小和 token 阶梯，然后提取严格的逐 shard 控制项，
并拒绝缺失、过期、格式错误或被修改的 shard 控制项。`--only-sku`、`--exclude-skus`、
`--ep-sizes` 和 `--precisions` 用于选择子集。矩阵会按每次 dispatch 动态生成，
不存在冻结的摘要或锁定的用例数。

| 系统 | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink，scale-up | 2x8 NVLink + RDMA，scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI，scale-up | 2x8 XGMI + RDMA，scale-out |
| GB200/GB300 | 2x4 MNNVL，scale-up | 4x4 MNNVL，scale-up |
| TPU v7 (tpu7x) | 1x8 ICI，scale-up | 2x8 ICI，**scale-up** |

物理主机数量并不决定范围：两种 GB 拓扑都位于同一个 72-GPU MNNVL
scale-up 域内。

**TPU v7 EP16 是 scale-up，这并非为了方便标注。** `tpu7x` 的 ICI 是一个跨主机的 3D 环面网络，
[文档所列](https://docs.cloud.google.com/tpu/docs/tpu7x)拓扑从
`2x2x1`（4 个芯片，1 台主机）→ `2x2x2`（8 个芯片，2 台主机）→ 一直到 `8x16x16`，全部属于同一个 ICI 互连，因此
一个 16 设备单元会跨越主机边界，但不会离开 scale-up 域。不会有任何流量经过 DCN。
因此，该 SKU 的 `scale_up_domain` 字面值为 `"ep"`，意味着它随 EP 度数变化：
切片会按照用例所需的大小进行配置，因此 EP8 报告的域大小为 8，EP16 报告的域大小
为 16，并且二者都保持 `scope: scale-up`。固定为 8 会把 EP16 错误标注为通过 DCN 的 scale-out；
固定为 16 则会夸大 EP8 shard，因为它的切片实际上只有 8 个设备宽。

这对比较的影响是：**TPU EP16 可与 GB200/GB300 EP16 比较**，后者同样是 scale-up，
位于 72-GPU MNNVL 域内，但**不可**与 b200/h100/mi355x EP16 比较，因为后者确实会经过一次 RDMA
跳转。EP 度数相同，但机制不同；应使用产物中的 `scope` 进行区分。
SKU 上的 `scale_out_transport: dcn` 描述的是真正的 scale-out 情况，即通过每芯片 100 Gbps 的数据中心网络
进行跨*切片*流量传输；当前没有任何单元使用这种方式。

运行它需要一个多主机切片，而看似显然的命令并不会创建这种切片：单独使用 `--tpu-topology`
只会创建一个*放置*策略，而 `tpu7x` 会拒绝该策略（"Use workload policy instead"）。EP16 背后的池
是通过显式工作负载策略创建的，
`resource-policies create workload-policy --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2`。
相应操作步骤记录在 `launchers/launch_tpu-gke.sh` 中，就在依赖该策略的保护逻辑旁边。
随后，shard 以带无头 Service 的 Indexed Job 形式运行，以便 GKE 注入 `TPU_WORKER_ID` 和
`TPU_WORKER_HOSTNAMES`，供 `jax.distributed.initialize()` 使用。两种精度共享同一个多主机
shard：切片是单个不可分割的分配单元，因此如果两个 shard 各自请求其全部节点，它们将各自只得到一个
pod，随后两个仅形成一半的切片都会中止。

探针会强制执行两项多主机不变量，因为违反任意一项都不会报错，而是会产生一个看似合理的
数值。每个进程都必须执行*完全相同的 slice-wide collective 序列*，
因此在多主机环境下，`reference_timing` 以 `lockstep` 模式运行（固定迭代次数）。基于时长驱动的
循环会使计数依赖各主机的时钟，并导致 libtpu 中止整个切片。并且 mesh 必须由
`jax.devices()` 构建，绝不能使用 `jax.local_devices()`：后者会在每台主机上构建一个 EP8 mesh，
并在 EP16 标签下测量两个彼此独立的 8 路交换。交换确实跨越 16 个 rank 这一点，是通过物理规律检查的，
而不是直接断言的。dispatch 成本会随路由扇出变化
（在 T=8192 时测得从 EP8 到 EP16 为 1.232x，与 1.232 的扇出比一致），而两个独立的
8 路交换应保持在 1.0。

| 后端 | 引擎可用性 | 当前范围 |
|---|---|---|
| DeepEP V2 | `production`，vLLM 使用 `--all2all-backend deepep_v2`，SGLang 使用 `--moe-a2a-backend deepep` | `normal` 模式采用 PR #605 的 `ElasticBuffer`，并包含上游 #630 和 #640 的精确修复：scale-up 使用 LSA，x86 EP16 scale-out 使用 GIN。除 BF16 外，还通过 `use_fp8_dispatch` 进行 FP8 dispatch（分块 e4m3fn）。`low-latency` 模式采用旧版 `deep_ep.Buffer` IBGDA decode kernel（按 expert 填充的布局、加权 combine、`use_fp8` e4m3fn），仅用于 decode；凡启用处均支持 EP8，此外还支持 GB200/GB300 上的 EP16（位于 MNNVL 域内），以及 B200 的 nscale 裸金属池上的 EP16（通过原生 IB rail 使用 IBGDA，并带有 `/dev/gdrdrv`，但此前的虚拟化 b200 池始终无法运行它）。B300 在 `low-latency` 中是不受支持的覆盖行：即使是单节点 EP8 运行，旧版 Buffer 也会自行启用 NVSHMEM IBGDA，而在 B300 上创建地址句柄会失败（`ibgda.cpp:2234 Unable to create ah`），所有八个 rank 均返回 rc255。`NVSHMEM_DISABLE_IB=1` 无法解决问题。无论如何，Buffer 都会重新启用 IBGDA，并且无论设置还是不设置该变量，运行都会以相同方式失败（在 b300-002 和 b300-011 上测得） |
| MoRI | `production`，vLLM 使用 `--all2all-backend mori_*`，SGLang 使用 `--moe-a2a-backend mori` | `normal` 模式在每个 CDNA SKU 上都使用直接的 `IntraNode` kernel 实现 scale-up EP8。EP16 在三者上都是不受支持的覆盖行：adapter 将 `InterNodeV1` 固定用于 2x8 XGMI + RDMA，但其 combine 会在传输层损坏数据（ROCm/mori#475），因此 registry 发布的是 `mori: [8]`，不会 dispatch 任何 EP16 case。`low-latency` 模式选择 `IntraNodeLL` decode kernel（单次调用、纯节点内、与 `IntraNode` 相同的紧凑布局和非加权 combine），仅支持 decode/EP8。FP8 dispatch 由调用方预量化（gfx942 上使用各 SKU 对应的 e4m3fnuz，gfx950 上使用 e4m3fn）。除 BF16 dispatch 外，combine 保持 BF16（`quant_type=none`） |
| UCCL-EP | `candidate`（没有引擎公开 UCCL-EP selector） | [UCCL](https://github.com/uccl-project/uccl) EP：可直接替换 DeepEP 且 API 完全相同，其 CPU proxy 通过普通 `libibverbs` 发起 GPUDirect RDMA（不使用 NVSHMEM/IBGDA），并通过软件实现消息排序、原子操作和流量控制。Scale-up 是通过 NVLink/XGMI 使用单节点 `cudaIpc`（绝不使用 MNNVL）。`normal` 模式采用旧版 `Buffer` 的 `dispatch`/`combine`（非加权 rank-sum）。`low-latency` 复用旧版 `low_latency_dispatch`/`low_latency_combine` decode kernel（加权 combine），仅支持 decode/EP8。`normal` 模式下的 FP8 dispatch 由调用方预量化（分块 e4m3fn，gfx942 上使用各 SKU 对应的 e4m3fnuz）。在 `low-latency` 模式下，调用方发送 BF16，decode kernel 在内部将其量化为 e4m3（`use_fp8`）。Combine 为 BF16。可在 NVIDIA 和 AMD 上运行（H100/H200/B200 + MI300X/MI325X/MI355X），支持 EP8 scale-up。跨节点 EP16 在功能上可用（节点间 RDMA 路径能够连接，轻量 case 通过正确性验证），但在 token 数量较大时，其 CPU proxy 吞吐量会超出标准化的逐 case 墙钟时间预算，因此 EP16 目前是不受支持的覆盖行 |
| NCCL EP | `candidate`（NVIDIA 自有库，但没有引擎公开 NCCL-EP selector） | [NCCL EP](https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep)：NVIDIA 基于 NCCL Device API 的原生 MoE dispatch/combine，节点内使用 LSA（NVLink load/store），节点间使用 GIN（GPU-Initiated Networking），并通过 `nccl4py` binding 驱动。`normal` 模式选择 `HIGH_THROUGHPUT` 算法（FLAT `[N, hidden]` 接收，非加权 rank-sum combine）。在单句柄修复消除 NVIDIA/nccl#2303 的 signal aliasing 后，`LOW_LATENCY` 算法恢复了全部六种 NVIDIA SKU 上的 EP8 `ll_backends` 行。该 LL decode 阶梯被限制为 T<=128，低于其 256-slot 接收容量：`nccl_ep` 的 combine recv pipeline 移植自 DeepEP #642 之前的 kernel，并且同样缺少 `mbarrier_arrive` 之前的 shared-memory fence，导致 GB300 上的 T=256 在 5 次执行中有 1 次发生损坏。其结果呈双峰分布，正常行的相对误差为 0.0039，而失败时为 0.4704。NVIDIA/nccl master 中不存在该 fence，因此上游尚未修复。该限制降低了暴露概率，但**并非**安全边界：每个 combine recv 都缺少该 fence，而 T=256 只是 pipeline 迭代次数最多的一级，因此较低各级只是更不容易触发竞态，并非不受影响。待包含修复的 wheel 发布后恢复，仅支持 BF16：`contrib/nccl_ep/RELEASE.md` 写道“不支持 FP8”，因此不会生成 FP8 case。该说明值得重新测试，而不应直接信任，因为我们固定 commit 中的 C 库确实会读取 `inputs->scales` 并根据 e4m3/e5m2 进行切换，文档列出的两个 FP8 排除项都是我们未使用的 expert-major 布局，而且自 2026-06-11 起 `NVIDIA/nccl` 一直未变，而 `NVIDIA/nccl-extensions` 已彻底替换该行。仅支持 NVIDIA 和 CUDA 13。H100/H200/B200/B300 上支持 EP8 scale-up，GB200/GB300 上支持 EP8 和 EP16，其中 EP16 保持在 MNNVL scale-up 域内。x86 EP16 scale-out 是不受支持的覆盖行：跨节点 GIN 路径在四种 SKU 上，无论使用 RoCE 还是 IB，都会在 `nccl_ep.cc` 内以相同方式发生 fault，这是 GDAKI 限制，而非 fabric 选择问题 |
| FlashInfer EP | `production`，vLLM 使用 `--all2all-backend flashinfer_nvlink_one_sided` | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) `MoeAlltoAll`：TensorRT-LLM 的单边 MNNVL all-to-all，其中每个 rank 将 token 直接写入其 peer 的 workspace window，combine 再将其读回，不存在 send/recv 配对，也不使用 NVSHMEM。仅支持 `normal` 模式（只有一个 kernel family，没有独立的 decode 路径），且仅支持 GB200/GB300，因为其传输采用 MNNVL。FP8 dispatch 由调用方预量化为分块 e4m3fn，作为第四个 dispatch payload 与其每个 128 元素块对应的 FP32 scale 一并传输，同时 combine 平面被强制设为 BF16。C++ `toNvDataType` 对 combine 仅接受 fp16/bf16/fp32，因此 FP8 combine buffer 会抛出异常，而不是导致数据损坏。支持 EP8 和 EP16，二者均位于 scale-up 域内。与此处其他所有后端不同，其 combine 使用 PAYLOAD dtype 而非 FP32 进行累加：0.6.16 之前的 wheel 使用成对 BF16 树归约 top-k contribution，并在每一层进行舍入，因此 oracle 直接对该归约建模（`combine_reduction = "topk-slot-tree"`），而不是放宽容差。0.6.16 将 accumulator 改为 FP32，adapter 会根据已安装版本切换模型 |

| JAX ragged A2A（TPU probe） | `candidate`（没有推理引擎公开 JAX ragged A2A selector） | 在 slice 的 ICI 域上，基于一维 device mesh 使用 `jax.lax.ragged_all_to_all`（JAX MoE stack 用于 EP dispatch/combine 的 primitive），支持 EP8（一个 host）和 EP16（两个 host，但仍位于同一个 ICI 域）。支持 `normal` 模式、BF16 和 FP8。FP8 dispatch 使用分块 e4m3fn，每个 128 元素块对应一个 FP32 scale（DeepEP 的 `per_token_cast_to_fp8`），并将 value 和 scale 作为两次 ragged exchange 发送。`fp8_consume: native` 表示 expert 直接消费 fp8，collective 之间不存在独立转换。转换会作为 `stage` 单独测量。两种精度下的 combine 均为 BF16，因为 expert 输出 BF16。Dispatch 是 permute gather 加 ragged exchange。Combine 是反向 exchange 加 fp32 scatter-add（非加权 rank-sum）。缺少 `ragged_all_to_all` 的 JAX build 会**失败关闭**。不存在填充式固定容量 `all_to_all` fallback，因为 padding 会传输不同数量的字节，并悄然测量成其他内容。关于该测量是什么以及不是什么，请参阅 TPU Probe 一节 |

DeepEP V2 指的是由
[DeepEP PR #605](https://github.com/deepseek-ai/DeepEP/pull/605) 引入的 `ElasticBuffer` 实现，而不是更新的旧版 `Buffer` build。
固定的源代码是上游 `main`，其中包含 #605，以及
[PR #630](https://github.com/deepseek-ai/DeepEP/pull/630)（修复 GIN
不可用时的纯 scale-up 初始化）、[PR #640](https://github.com/deepseek-ai/DeepEP/pull/640)（防止 NCCL
shared-memory mapping 被错误分类为重复 NCCL 库），以及
[PR #642](https://github.com/deepseek-ai/DeepEP/pull/642)（low-latency combine fence，用于修复
[issue #700](https://github.com/deepseek-ai/DeepEP/issues/700) 中 Blackwell 最高一级的数据损坏）；此前固定的版本，即合并前 #605 分支上的 #630
head，早于这些内容。Scale-up case 请求 NCCL Device API LSA，除非实际建立的 LSA team 覆盖完整 EP world，否则会失败关闭。x86 EP16 scale-out case 则要求使用
GIN 的混合路径，其中两个逻辑 scale-out 域由两个物理 RDMA rank 表示，每个域包含八个
scale-up rank。GB EP16 仍然是 MNNVL scale-up，因此使用 LSA。是否尝试给定的
SKU/backend/EP cell 属于能力事实。是否成功则由
benchmark 的返回码决定。

## 工作流与产物

`.github/workflows/collectivex-sweep.yml` 包含两个作业。`setup` 生成一个公共 SKU 矩阵
（输入为 `backend`、`only_sku`、`exclude_skus`、`ep_sizes`）并上传该矩阵。
`sweep` 为每个矩阵条目提取一个严格且被忽略的 `.shards/<id>.json` 控制文件，为每个分片执行一次
资源分配，在需要时于资源分配前获取固定版本的 DeepEP 源码，并使用 `always()` 上传
结果产物，因此即使运行失败或仅部分完成，仍会执行上传。

每个分片都会生成逐用例结果 JSON 和一份简短的机械式摘要。用例是否成功以
基准测试自身的返回码为准。不存在完整性或隐私验证步骤，失败或
不受支持的单元格不会生成合成记录。没有任何步骤会提升某次运行、
构建数据集或推进频道。中立产物即为输出。消费者下载
这些产物，并自行决定显示哪些内容。

不会向工作流传递或上传任何运维人员凭据。runner 本地覆盖项及任何
选择器均保留在 runner 上。每个步骤的 runner 日志保留在 runner 上以供事后分析，而
结果产物仅包含方法论中列出的字段。

## Runner 配置

每个 SKU 的 Slurm 和存储值均来自注册表中受跟踪的基线。可选的
runner 本地 JSON 文档可位于 `$XDG_CONFIG_HOME/inferencex/collectivex.json`，或由
`COLLECTIVEX_OPERATOR_CONFIG` 指定，并按字段覆盖该基线。没有注册表条目的 runner、
未知字段以及非 JSON 输入都会以关闭方式失败，并且配置绝不会作为 shell 求值。
不会拒绝重复的 JSON 键。`json.load` 会静默保留最后一个值，而且除当前正在解析的键之外，
其他 runner 键不会被验证，因此 SKU 名称中的拼写错误会被忽略，而不会
被报告。GHA 不会传递任何运维人员密钥，因此除非存在 runner 本地文档，否则 SKU
将完全使用其受跟踪的基线运行。

所有公开的逐 SKU 平台数据都位于受跟踪的 `configs/platform_config.json` 注册表中：
架构/产品、厂商、加速器运行时、容器镜像及平台、固定放置方式、
启动器、可运行的 backend/EP 组合、scale-out `fabric` 标识（NIC 和交换机，因此即使使用相同 GPU，
采用不同 fabric 的集群也是不同条目，例如第二个 b200 集群）、受跟踪的运维人员
默认值，以及 scale-out RDMA 选择器。`vendor` 是任意的规范化元数据，并不
局限于 AMD 或 NVIDIA；`runtime` 用于选择执行路径，并且是一个闭集
（`runtime/config.py` 中的 `ACCELERATOR_RUNTIMES`），因为它决定用例可以使用哪个基准测试入口点和
启动器系列：`cuda`/`hip` 运行 `bench/run_ep.py`（torch、NCCL/RCCL，每个
rank 一个进程），而 `tpu` 运行 `bench/run_ep_jax.py`（JAX、XLA collectives，每台主机一个进程）。因此，
添加其他厂商并不需要修改厂商允许列表，但新的
运行时或 collective 实现仍然需要其自身兼容的入口点和启动器。
可选的 `scale_out_transport` 字段用于在 SKU 的跨主机 fabric 并非 RDMA 时为其命名
（TPU 主机使用 `dcn`），因此 scale-out 行绝不会被标记为集群并不具备的 fabric。
运维人员文档可以覆盖默认值。启动器会
声明并检查其实际需要的字段。`sweep_matrix.py` 根据
放置字段推导 EP 拓扑。默认情况下，sweep 包含每个已注册的 SKU。

每个选定的非 MNNVL EP16 放置还需要为其经运维人员批准的 fabric 配置
`socket_ifname` 和 `rdma_devices`。`ib_gid_index`、`rdma_service_level`、`rdma_traffic_class`
和 `rail_isolated` 也在可选允许列表中。服务级别和流量类别会映射到 MoRI 的
RDMA/IO QoS 环境。
CollectiveX 不会通过启发式方法选择管理路由或 HCA。资源分配后，每个
非 MNNVL scale-out 节点必须在 backend 设置前证明所有已配置的接口和活动 HCA 端口均存在。
Scale-up 和 MNNVL 作业会清除这些覆盖项。Scale-out NCCL/RCCL 被固定为 `IB`，
并使用精确匹配的 HCA 选择器，因此套接字回退会失败，而不会被错误标记为 RDMA。
Scale-out 还会禁用 NCCL 双端口 NIC 融合（`NCCL_IB_MERGE_NICS=0`）：融合设备会禁用
NCCL GIN，而 DeepEP V2 EP16 混合路径需要它；采用 rail 隔离的 fabric
（`rail_isolated=1`，例如 B300 的多平面 RoCE）还会额外设置 `NCCL_CROSS_NIC=0`。

仅当所有选定的 HCA 端口都报告以太网链路层时，才会应用 `ib_gid_index`，此时它
选择经运维人员批准的 RoCE GID。原生 InfiniBand 配置文件会保留显式 HCA 和服务
级别固定，但不会设置仅适用于 RoCE 的 GID 覆盖项，以便 NVSHMEM/NCCL 使用原生 LID 路径。
混合使用以太网和 InfiniBand 的 HCA 列表会被拒绝。

`stage_dir` 是 checkout 和工作流
workspace 之外一个预先存在、由 runner 拥有且非符号链接的基础目录。它不允许组或其他用户写入，并且在 runner 和每个
已分配节点上都能通过同一路径访问。作业只会创建一个带标记的 mode-0700 执行子目录，验证跨节点读写
可见性，并在资源分配拆除后仅删除该子目录。它们绝不会挂载 runner
checkout，也不会在 AMD 的镜像存储下创建 stage。当 AMD 运维人员条目省略 `stage_dir` 时，
runner 会在共享 runner 文件系统上其标准 `_work` 目录旁边派生一个私有基础目录。
绝不会将 root 拥有的 squash 缓存用作仓库 stage。

H200、B200 和 B300 runner 可以省略 `stage_dir`。它们的隔离执行子目录会创建在
经验证的操作系统账户主目录中的 runner 自有 mode-0700 基础目录下，与
工作流的临时 `HOME` 无关。H100 也可以省略 `stage_dir`。它的私有基础目录创建在已配置的共享容器目录旁边，而绝不位于
该目录之下，因此计算节点可见。规范的 B300 执行会
忽略任何旧版配置的 `stage_dir`，并始终使用经验证、计算节点可见的账户主目录
基础目录。执行 ID 后缀用于隔离并行 B300 worker。规范的 GB300 执行同样
会忽略其旧版、组可写的 `stage_dir`，并在经验证、计算节点可见的账户主目录下派生一个
执行专属的私有基础目录。backend 准备工作会在每个节点上从该 staged 目录树运行。

Enroot 会将配置的镜像标签导入到一个按单次运行限定作用域、以镜像标签和镜像
平台为键的 squash 中，因此一次运行绝不会复用另一次运行导入的文件系统。镜像标签和平台是
逐 SKU 注册表字段。DeepEP V2 源码固定版本位于 `runtime/common.sh` 中，其构建会
在固定 commit 处获取并验证，检查是否包含 `ElasticBuffer`，然后缓存在一个
以架构、镜像和 commit 为键的集群本地构建缓存中。只有固定的 `/cx-cache` 挂载
会进入容器。

## TPU 探针

TPU 路径是一个探针，并非 GPU 后端的对等实现；相关差异会记录在 artifact 中，而不是留给读者自行推断。

它运行在 GKE/ARC `tpuv7` 池上，其 runner 是一个仅含 CPU 的协调器 pod。这里没有 Slurm，
没有 enroot/pyxis，也没有共享文件系统，因此 `launchers/launch_tpu-gke.sh` 仿照的是
`runners/launch_tpuv7-gke.sh`，而不是 `launch_single-slurm.sh`：它将协调器中
已固定版本的源代码子树打包到 ConfigMap 中（约 120 KB，pod 内不需要仓库凭据），为每个 shard
在 `tpu7x` 节点上启动一个 Kubernetes Job，在该 Job 中运行所有 case，并通过 pod 日志
回收结果 JSON。在 EP16 下，该 Job 为 **Indexed**，slice 的每台主机各有一个 pod，
并配有一个无头 Service，因此 GKE 会注入 `TPU_WORKER_ID`/`TPU_WORKER_HOSTNAMES`，供
`jax.distributed.initialize()` 使用；每个 pod 都计算相同的行，由完成索引为 0 的 pod
写入 artifact，而回收过程读取的也是这个 pod（`jax.process_index()` 并不跟踪 pod 索引，
因此若以它为依据，有一半时间会从错误的 pod 回收）。标红或部分完成的 leg 仍会提交其已生成的所有内容。

有两个属性与 GPU 系列不同，并且每个属性都会在 artifact 中明确标注：

- `measurement.timing_source` 为 **`xla-device-trace-span`**。JAX 没有与
  `torch.cuda.Event` 对等的公开 API，因此该探针改为从 XLA profiler trace 中读取设备时间：采用 SPANS
  （某个组件作用域从首次开始到最后结束，而非操作时长之和），每次 occurrence 在各设备间取 MAX 进行归约，
  使用 RAW 百分位数。这与 GPU SKU 的 CUDA event 属于同一*类*数值，即芯片时间，不包含主机 dispatch 开销。

  `host-wallclock-blocked` 仍作为回退方案保留，并且是**选择启用**的（`--allow-host-fallback`），
  这是有意为之：tpu7x 上的主机挂钟时间包含一个与 payload 相关的单次调用 dispatch 下限，
  可高达数千微秒，因此如果将主机测量值当作可比较数据发布，在小 token 端会严重失真。若未指定该标志，
  某个 point 无法产生设备 span 时会使该 case 失败，而不是静默降级。来源信息按组件记录
  （`row.timing_source`），因为 profiler 可能覆盖某些组件而未覆盖另一些组件，而此前使用整个 case
  的标签时，曾将实际采用主机计时的行误标为设备计时。

  较早的修订版本改用摊销方式，在一个程序内串联 N 次操作以均摊主机下限
  （`--in-program-iters`）。设备 span 使其不再必要，该标志现已移除；如果你发现对它的引用，
  则该引用已过时。
- 在 **BF16 行上**，`components.stage` 为 `unavailable`：permute 已融合到 dispatch 中，
  且没有转换，因此没有可计时的内容。**FP8 行会发布该项**，即 fp8→bf16
  转换；它被移出 combine 和串联的 roundtrip，并单独测量，因此
  `roundtrip + stage` 能够重建配置不匹配时的成本。与 GPU `native` 行比较时，不要将 `stage`
  加到 `roundtrip`；与 `CX_FP8_CONSUME=dequant` 行比较时则应相加。每个 destination 的布局
  （offset、size、gather index）在主机上预计算，并排除在计时区域之外，这与 DeepEP 的 layout pass
  所采用的处理方式相同；设备上的 permute gather 和 combine scatter-add **会**计时，因为生产环境会承担这些成本。

- `implementation.oracle` 为 `probe-source-identity-and-exact-rank-sum`，其范围比 GPU
  测试框架的完整逐专家变换预言机更窄。它仍然证明了真实有效的性质：每个 dispatch 的副本
  都会被解码回其所声明的源 token（ID 携带在前几列的 SIGN 中，因此可在量化后保留），
  其落点偏移会根据交换计划进行检查，并将 combine 与精确的预期 rank-sum 进行比较。
  专家执行恒等变换，因此不对逐专家变换建模。

  在 BF16 下，payload 会与源数据逐比特比较。在 FP8 下则特意不会这样做，而在有人“修复”
  它之前，有必要了解其中的原因：同一量化方案的两次 XLA 编译会以相反方向舍入该点阵的
  e4m3 中点（实测结果为线上传输了 `161/256`，而独立程序生成了 `152/256`），因此通过
  再次量化重新推导期望值无异于抛硬币。GPU 测试框架之所以比较重新量化后的比特，仅仅是
  因为 `assert_quantize_identity` 首先在实际硬件上确立了这一前提。这里改为：将到达的
  chunk 与同一次执行中 STAGED 的 chunk 逐字节比较；根据 dispatch 实际交付的行对 combine
  进行评分，并依据各行的 payload 所声明的 token 将其归属到相应 token；同时，由于计时程序
  与预言机程序来自两次独立编译，还会将计时程序自身的 scale 和反量化值关联回预言机程序
  中的对应值。该链路中的任何环节都不会重新量化任何内容。

### 链式 pair period

`pair_period` 与 `roundtrip` 是**不同的量**：它表示在一条始终不排空的链中，一个
dispatch→combine 对的稳态周期；而 `roundtrip` 表示从空闲状态进入的一个操作对。不要将二者
相加，也不要相互替代。`--chain-iters`（默认 128）个操作对在同一个已编译程序中运行。
每个操作对的 carry 都会馈入下一个操作对，形成真实的数据依赖，而不是使用
`optimization_barrier`，因为 barrier 只约束顺序、不保证活性，XLA 仍会删除无人消费的计算。

**链必须重新归一化，而且这是必需的。** Combine 是无权重 rank-sum，因此一个操作对会将
令牌 `t` 乘以 `d_t`，即其唯一目标 rank 数，在 deepseek-v3 top-8 上实测 EP8 为 3..7、
EP16 为 4..8。迭代 64 次后得到 `d_t**64`，而 bf16 的上限是 3.39e38：若不重新归一化，
第一个元素会在第 43 次迭代饱和，并且到第 64 次时 **EP16 的元素有 100% 都是无穷大**。
主体在转回 bf16 前将 fp32 rank-sum 除以 `d_t`，使一个操作对在比特级保持恒等（`d_t ≤ 8`
个 bf16 值副本的 fp32 求和是精确的，IEEE 除法也会正确舍入）。传递的是计数值，而不是
`1/d`：在超过 200k 个值中，`(d*v)*float32(1/d)` 对 `d == 7` 的值有 58% 与 `v` 不同，
而两种布局中都会出现七个目标。

`correctness.chain_regime_passed` 用于判定这种恒等性；它在计时程序上评分，并跨主机归约。
它是**三态**的：`true` 表示通过，`false` 表示已运行但不一致（会使该行失败），`null` 表示
无法评估（会隐去 period，但不会否定同一用例中测得的 drained component）。Drained oracle
无法覆盖该 regime。它始终只检查从空闲状态进入的一个操作对，因此仅在自由运行操作对下
发生破坏的传输可能会成为套件中看起来最快的实现。

`chain_floor_us` 按方向给出，`origin: chained-cross-rank-min`。候选项是在每台设备上恰好出现
`--chain-iters` 次的 collective；随后按它们出现的设备行集合分组，anchor 是**能够组成操作对的
最繁忙行组**（至少 2 个 op）中总时长最高的两个 op。按行分组才能保证两个 anchor 可比：
tpu7x 会记录不止一种 core，因此 `sparse-core-…` op 的总时长可能超过真正的 anchor；两个
出现在大小相同但彼此**不相交**的行集合上的 op，无论多大也不能组成操作对。按行集合
**大小**选择的规则只在 10 个 point 中的 2 个发布了 floor，比它替代的 5/10 更差。
`≥2` 的资格条件也同样重要：否则，sparse-core 行上的单个大型 op 会凭总时长获胜，导致该
point 只剩一个候选项而失败。最终留下的操作对还必须按 d,c,d,c 交错。方向由每次迭代的开始
顺序确定。如果任一 gate 失败，**两个**方向都以相应原因为 `unavailable`；这里的 null 表示
“未测量”，绝不表示“与另一个方向相同”。`chain_health.anchor.phase` 报告 combine anchor
在 period 内开始的位置，约 0.5 表示真正的操作对，约 0 或 1 表示两个 anchor 都属于同一
方向，这是仅靠交替无法发现的。

**在任何新 SKU 上都值得执行的交叉检查：**如果 anchor 确实是首尾相接的 dispatch 和 combine，
`phase` 应跟随 `chain_floor_us.dispatch / pair_period`，因为 combine 会在 dispatch collective
结束时开始。二者来自相互独立的量，前者来自开始时间戳，后者来自 op 时长，因此一致性是
方向**标签**正确的证据，而不只是内部自洽。在 bf16 上（运行 31194062843），跨 128x 尺寸
范围和两个 EP degree 的测量如下：

| T | `floor_dispatch / period` EP8 | `phase` EP8 | `floor_dispatch / period` EP16 | `phase` EP16 |
|--:|--:|--:|--:|--:|
| 64 | 0.321 | 0.361 | 0.332 | 0.365 |
| 512 | 0.276 | 0.294 | 0.287 | 0.311 |
| 8192 | 0.265 | 0.280 | 0.271 | 0.286 |

在全部 28 个 chained 行上，`phase` 都略高于该比值（最大差距为 0.064），这正是预期的符号：
floor 是跨 rank 的**最小值**，而 phase 则在真正决定链节奏的设备上测量。无论两个 anchor
交错得多么规整，如果 `phase` 不跟随该比值，就说明它们并非两个方向。

`chain_health` 与 GPU 系列的块形状一致（每个字段都是带 `percentiles_us` 的 component，
而不是裸浮点数）：

| 字段 | 含义 |
|---|---|
| `interpair_gap_us` | 每次迭代的 start-to-start 减去链 scope 自身的 extent。接近零表示自由运行。EP8 和 EP16 上**实测为 0.0–0.2 µs，占 period 的 0.0–0.1%**。它既不是 collective floor，也不是 op 总和，因为这两者都会把 permute 和 scatter-add 留在“gap”内，分别读出 47% 和 20–30%。 |
| `settle_drift_us` | 每台设备后半段减前半段的 p50，并按带符号的最大幅值归约。用于支持或否定 `--chain-drop`。实测 ≤0.6 µs。 |
| `pair_spread_us` | 每次迭代的跨设备 spread。 |
| `devices` / `devices_expected` | 在 EP16 下，每个进程只对本地 8 台设备做 profile，因此每台设备的 period 矩阵会在归约前执行 allgather，缺失的另一半恰恰是可能出现跨主机 straggler 的位置。`gathered_across_hosts` 表示是否完成该步骤；`devices_expected` 始终如实保持为 16。 |
| `op_inventory` | 链主体中的每个 op，按迭代列出。这使 fp8 排除项可检查，而不只是一项论断。 |
| `renorm_us` | **`unavailable`，而且理应如此**：XLA 会把除法融合进相邻的 cast，因此没有任何 op 带有该 scope。其成本通过跨运行差分限定为 period 的 ≤0.05%，而非直接测得。 |
| `capture_s` / `capture_budget_s` | 链 capture 相对于 launcher 自身 per-case timeout 的开销。实测为 13–22 s，而预算为 2700。 |

**FP8 被有意排除在链式测量之外。**它的 combine 需要 BF16，因此 chained fp8 主体会把
fp8→bf16 转换带入循环，使 period 包含 `native` contract 所规定生产环境不会单独执行的工作。
FP8 行上的 `implementation.chained_period` 为 `false`，chained 字段也会以该原因为
`unavailable`，而不是静默缺失。

工作负载标识与 GPU 系列**完全共享**：`bench/routing_np.py` 是经过一致性测试的
`bench/routing.py` NumPy 移植版（TPU 镜像中没有可用的 torch），因此两个系列测量完全相同的
routing trace 和 activation byte，并使用相同的去重后（token、destination-rank）payload
单元计费。只要 torch 可导入，`tests/test_tpu_probe.py` 就会逐元素断言两者一致；无法导入时
则固定一个 golden digest。该测试还会在纯 NumPy 中模拟完整 exchange plan，因此 offset 或
transpose 错误会在本地失败，而不是到 TPU 节点上才失败。

## 本地检查

```bash
python3 -m unittest discover experimental/CollectiveX/tests -p 'test_*.py'
python3 experimental/CollectiveX/sweep_matrix.py --backend all --out /tmp/cx-matrix.json >/dev/null
bash -n experimental/CollectiveX/runtime/*.sh experimental/CollectiveX/launchers/*.sh
```

核心路径包括 `configs/`、`sweep_matrix.py`、`summarize.py`、`bench/`、`runtime/`、`launchers/`
和 `tests/`。
