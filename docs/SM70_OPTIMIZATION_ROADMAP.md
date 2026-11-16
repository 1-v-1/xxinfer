# SM70 / V100 优化队列

> 目标：**2× Tesla V100-16GB** 跑 `unsloth/Qwen3.8-27B-NVFP4`，KV cache 充足，decode 与 prefill 双线提速，对照 1Cat/vLLM。
> 本文件**只排队列与工序**。每项的现象/假设/验收条/覆盖区间/结论在 [specs/](<./specs/>) 对应那份，**不在本表重复**。
> 事实依据见 [SM70_VOLTA_FINDINGS.md](<./SM70_VOLTA_FINDINGS.md>)；逐次尝试不再单列流水，直接落进对应规格的 §5。

## 1. 执行顺序（按 收益/风险 排）
| 序 | 动作 | 收益 | 风险 | 前置 | 规格 |
|---|---|---|---|---|---|
| 1 | 工作流落地：`[patch]` path 依赖 + 独立基准骨架 + 编译成本实测 | 编译 30 min → 分钟级 | 无 | — | 本文件 §2 |
| 2 | prefill 分解（长 prompt 定位） | 0（但是前提） | 无 | 1 | [m1](<./specs/m1-prefill-breakdown.md>) |
| 3 | NVFP4 QPN2 split-K GEMV | +5 tok/s | 低 | 1 | [d1](<./specs/d1-nvfp4-qpn2-splitk.md>) |
| 4 | GDN prefill 上 SM70（chunked 并行） | prefill 大头（48/64 层） | 中 | 1,2 | [p1](<./specs/p1-gdn-prefill-sm70.md>) |
| 5 | FA prefill 走 CUTLASS 2.x | prefill 大头（16 层） | 高 | 1,2,4 | [p2](<./specs/p2-fa-prefill-cutlass2.md>) |
| 6 | 图正确化 | ≈+8 tok/s | 高 | — | [g1](<./specs/g1-nccl-allreduce-in-graph.md>)（已采纳）、[g2](<./specs/g2-flashinfer-plan-handshake.md>) |
| 7 | 图开关合并 + 清理 `XINFER_DIAG_*` | 可维护性 | 低 | 6 | [g3](<./specs/g3-graph-switch-merge.md>) |
| 8 | prefill 图（slab 分配器） | 几个百分点 | 中 | 6 | [p4](<./specs/p4-prefill-graph-slab.md>) |

**1Cat 对照数字**：decode 4 卡 103.87 tok/s（2 卡折半 ≈52）；prefill D256 GQA 62~68 TFLOPS。

## 2. 工序（先做完再动性能）
### 2.1 编译成本模型（实测）
| 操作 | 成本 | 说明 |
|---|---|---|
| 独立 nvcc 基准 | **~1 min** | `xxinfer-dev-env/mmatest/bench_*.cu` ← **主战场** |
| 改 1 个 `.rs`（缓存命中） | **9 s** | 实测：`touch src/core/runner.rs` |
| 什么都不改（缓存命中） | **43 s** | 实测：no-op build |
| 改 1 个 `.cu`（缓存命中，`[patch]` path 依赖下） | **281 s ≈ 4.7 min** | 只重编改动的那一个 TU |
| 改 1 个 `.cu`（缓存失效） | **~30 min** | 90 个 TU 全重编，主导是 `flashinfer_adapter_{decode,prefill}.cu`（各 ~28 min 卡在 `cicc` 的 CUTLASS 模板展开）——**而我们不会改这两个** |
| push attention-rs / 改 feature 组合 | 触发上一行 | git dep ⇒ 新 `OUT_DIR` ⇒ cudaforge 缓存 hash 变 |

⇒ `[patch]` path 依赖把"改一个 kernel"的代价从 ~30 min 压到 ~4.7 min（**6×**）。这是整个工作流的地基。
根因：cudaforge 的缓存 hash 含 fetch 进来的 git 依赖路径，而 cargo 每次给 git dep 都发新 `OUT_DIR`；`build.rs` 已把输出目录钉死在 `<workspace>/target/cuda-kernels/<profile>`（实测 88 MB / 77 个 `.o`），**但 crate 自身路径也在 hash 里**。
**已验证对策**：`[patch]` 指向本地 clone 后缓存仍命中 ⇒ **开发期永远不要 push attention-rs**，交付前删掉 patch 段、切回固定 rev。

### 2.2 三条铁律
1. 任何 CUDA kernel 改动，先在独立 nvcc 基准里做（1 min vs 30 min）：数值对齐 + 带宽达标再搬进 `attention-rs`。
2. 开发期不 push `attention-rs`，用 `[patch]` 指向本地 clone。
3. 一次只改一个东西，改完立刻量。

**坑 A**：cudaforge 按**内容 hash** 判断要不要重编，`touch` 不触发 ⇒ 测编译成本必须真改内容。
**坑 B**：`build.rs` 的 `rerun-if-changed` 手工维护，116 个源文件只登记了 91 个；未登记的 25 个（`src/attention/*`、`src/marlin/*`、`src/moe/*`、`paged_attention_v*.cu`、`dsa_lightning_indexer.cu`、`gptq_cuda_kernel.cu`）**改了不会触发重编 ⇒ 测出来的性能是假的**。

### 2.3 抄 1Cat 代码的 5 步（每步有准入门槛）
| 步 | 动作 | 门槛 |
|---|---|---|
| S1 原样复现 | 抄到独立 nvcc 基准，**剥掉 torch**（host 侧改裸 `cudaMalloc`/`cudaMemcpy`），CUTLASS 头用 cudaforge 已下的那份（v4.x 仍含 2.x legacy API） | 数值对齐参考实现 + 打出 TFLOPS |
| S2 形状对齐 | 换成我们的真实形状（见各规格 §1） | 跑通 + 记录 TFLOPS |
| S3 搬进 attention-rs | 放 `src/kernels/src/`；去 torch；`extern "C"` 暴露 FFI；**加 `rerun-if-changed`**；SM70 需新宏通道 | 单元测试通过 |
| S4 接 Rust 侧 | `ffi.rs` 加绑定 + 模型层 dispatch | 固定 prompt + greedy，**逐 token** 对比 eager 基准 |
| S5 性能验收 | 端到端 tok/s + 单独 kernel 带宽 | 达到该规格 §5 的预期 |

**准入门槛：S1 不过 ⇒ 不许进 S3。**（历史上最大的时间浪费就是把没在基准里验证的 kernel 塞进整条编译链。）

### 2.4 测试分层与远端卫生
| 层 | 工具 | 测什么 | 门槛 |
|---|---|---|---|
| L0 | 独立 nvcc 基准 + CUDA event | kernel 数值、带宽、TFLOPS | 误差达标 + 带宽达标 |
| L1 | attention-rs 单元测试 | FFI 绑定、形状/布局 | 通过 |
| L2 | 端到端（1× 短 prompt，温度 0） | **贪心逐 token** 对比 | token 序列完全一致 |
| L3 | 端到端性能 | tok/s | 落进该规格 §5 |

- `nvprof`/`nsys`/`ncu` 对 xinfer 全部失效（Driver API / 崩溃 / 无计数器权限）⇒ 性能定位只能靠 **L0 基准 + CUDA event**。
- 开工前清僵尸编译（它们会偷 build 锁）：`pkill -9 -x cargo; pkill -9 -x nvcc; pkill -9 -x cicc; pkill -9 -x ptxas; pkill -9 -x xinfer`，再 `nvidia-smi --query-compute-apps=pid --format=csv,noheader` 确认无残留占卡。**必须用 `-x` 精确匹配**，否则杀掉自己的 SSH 通道。
