# SM70 / V100 完整优化路线图（prefill + decode）

> 目标：**2× Tesla V100-16GB** 跑 `unsloth/Qwen3.8-27B-NVFP4`，KV cache 充足，
> decode 与 prefill 双线提速，对照 1Cat/vLLM。
> 结论与事实依据见 [`SM70_VOLTA_FINDINGS.md`](SM70_VOLTA_FINDINGS.md)；逐次流水见仓库根 `PROGRESS_PORT.md`。

---

## Part 0：先把工具链修好（**优先级最高，先做完再动性能**）

### 0.1 编译成本模型（**实测**）

| 操作 | 成本 | 说明 |
|---|---|---|
| 独立 nvcc 基准 | **~1 min** | `repos/xxinfer-dev-env/mmatest/bench_*.cu` ← **主战场** |
| 改 1 个 `.rs`（缓存命中） | **9 s** ⭐ | 实测：`touch src/core/runner.rs` |
| 什么都不改（缓存命中） | **43 s** ⭐ | 实测：no-op build |
| **改 1 个 `.cu`（缓存命中，`[patch]` path 依赖下）** | **281 s ≈ 4.7 min** ⭐⭐ | 实测：往 `gdn.cu` 真实加一行注释 ⇒ `Compiling attention-rs` + `Compiling xinfer` 都执行、只重编改动的那一个 TU |
| 改 1 个 `.cu`（缓存失效） | **~30 min** | 90 个 TU 全重编，主导是 `flashinfer_adapter_decode.cu` / `flashinfer_adapter_prefill.cu`（各 **~28 min** 卡在 `cicc` 的 CUTLASS 模板展开）—— **而我们不会改这两个文件** |
| **push attention-rs** | **触发上一行** | git dep ⇒ 新 checkout ⇒ 新 `OUT_DIR` ⇒ cudaforge 缓存 hash 变 |
| **改 feature 组合** | **触发上一行** | 实测：`--features ...,graph` 与不带 `graph` 的构建**缓存不共享**（nvcc 参数 hash 变了） |

**⇒ 结论：`[patch]` path 依赖把「改一个 kernel」的代价从 ~30 min 压到 ~4.7 min（6×）。**
**这是整个工作流的地基：不 push attention-rs + 固定 feature 组合 + `[patch]` 指向本地 clone。**

**根因（`attention.rs/src/kernels/build.rs:134-173` 的注释已写明）：**
cudaforge 的缓存 hash 包含 fetch 进来的 git 依赖路径，依赖被 fetch 进 `OUT_DIR`，
而 cargo 每次给 git dep 都发新的 `OUT_DIR` ⇒ 缓存每次都失效。
`build.rs` 已经把输出目录钉死在 `<workspace>/target/cuda-kernels/<profile>`（稳定，实测 88 MB / 77 个 `.o` + `.cudaforge_cache.json`），**但 crate 自身路径也在 hash 里**。

**★ 已验证的对策**：`[patch]` 把 attention-rs 指向本地 clone 后，**缓存仍然命中**
（观察到只重编被 touch 的那一个 `.cu`，而不是 90 个 TU）。
⇒ **开发期永远不要 push attention-rs。**

### 0.2 三条铁律

1. **任何 CUDA kernel 改动，先在独立 nvcc 基准里做** —— 1 min 编译 vs 30 min。
   基准里跑通 + 数值对齐 + 带宽达标，再搬进 `attention-rs`。
2. **不 push `attention-rs`，除非是要交付。**
   开发期用 `[patch]` 把 git 依赖指向本地 clone；交付前删掉 patch 段、切回固定 rev。
   ```toml
   # xinfer/Cargo.toml 末尾（交付时删掉）
   [patch."https://github.com/1-v-1/attention.rs.git"]
   attention-rs = { path = "/workspace/attention.rs" }
   ```
3. **一次只改一个东西，改完立刻量。** 任何"我觉得这样更快"的改动，先用基准测出数字再决定留不留。

### 0.2.1 两个会浪费时间的坑

**坑 A：cudaforge 按**内容 hash** 判断要不要重编，`touch` 不触发。**
⇒ 想验证"改一个 .cu 要多久"，必须**真的改内容**（加一行注释也算），
否则会看到"9 秒就编完了"的假象，然后实际改了代码却忘了它没被编进去。

**坑 B：`build.rs` 的 `rerun-if-changed` 是手工维护的，116 个源文件里只登记了 91 个。**
未登记的 25 个文件（`src/attention/*`、`src/marlin/*`、`src/moe/*`、
`src/paged_attention_v*.cu`、`src/dsa_lightning_indexer.cu`、`src/gptq_cuda_kernel.cu` 等）
**改了不会触发重编 ⇒ 测出来的性能是假的。**
⇒ **新加 `.cu`/`.cuh` 必须在 `build.rs` 加一行 `cargo:rerun-if-changed=src/<路径>`。**
（`src/flash/*.cuh`、`gdn.cu`、`gdn_flashinfer_prefill.cu` 都在列表里 ⇒ 当前工作不受影响。）

### 0.2.2 远端卫生（每次开工前）

```bash
pkill -9 -x cargo; pkill -9 -x nvcc; pkill -9 -x cicc; pkill -9 -x ptxas   # 清僵尸编译（它们会偷 build 锁）
pkill -9 -x xinfer                                                          # 清模型进程
nvidia-smi --query-compute-apps=pid --format=csv,noheader                  # 确认没有残留占卡
```
**必须用 `-x` 精确匹配**，否则会杀掉自己的 SSH 通道。

### 0.3 抄 1Cat 代码的标准流程（5 步，每步有准入门槛）

1Cat 的 kernel 是 C++/CUDA + torch + CUTLASS 2.x；我们的落点是 `attention.rs` 的 `kernels` crate（Rust FFI）。

| 步 | 动作 | 验证 / 门槛 | 成本 |
|---|---|---|---|
| **S1 原样复现** | 抄到独立 nvcc 基准（`mmatest/`），**剥掉 torch**（host 侧改裸 `cudaMalloc`/`cudaMemcpy`），CUTLASS 头用 cudaforge 已下的那份（`/root/.cudaforge/git/checkouts/cutlass-b46b16d*/include`，v4.x **仍含 2.x legacy API**） | 数值对齐参考实现；打出 TFLOPS | **~1 min** |
| **S2 形状对齐** | 换成我们的真实形状：**TP=2 下每 rank `num_qo_heads=12` / `num_kv_heads=2` / `head_dim=256` / GQA group=6 / `page_size=64` / chunk=2048 / KV 在 paged 布局** | 跑通 + 记录 TFLOPS | 分钟级 |
| **S3 搬进 attention-rs** | 放 `src/kernels/src/`；去 torch；`extern "C"` 暴露 FFI；**`build.rs` 必须加 `cargo:rerun-if-changed=src/<新文件>`**；**SM70 需要新的宏通道**（现有 `-DUSE_CUTLASS` 被 `compute_cap >= 90` 卡死，见 §3.2） | attention-rs 单元测试 | 首次全量（一次性） |
| **S4 接 Rust 侧** | `ffi.rs` 加绑定 + 模型层 dispatch（参考 `gdn_flashinfer_prefill.cu` 的接法） | 端到端正确性：固定 prompt + greedy，**逐 token 对比** eager 基准 | 编译 |
| **S5 性能验收** | 端到端 tok/s + 单独 kernel 带宽 | 达到预期 | — |

**准入门槛：S0/S1 不过 ⇒ 不许进 S3。**（历史上最大的时间浪费就是把没在基准里验证的 kernel 塞进整条编译链。）

### 0.4 测试分层

| 层 | 工具 | 测什么 | 门槛 |
|---|---|---|---|
| **L0** | 独立 nvcc 基准 + CUDA event | kernel 数值、带宽、TFLOPS | 数值逐位/误差达标 + 带宽达标 |
| **L1** | attention-rs 单元测试 | FFI 绑定、形状/布局正确 | 通过 |
| **L2** | 端到端（远端，1× 短 prompt） | **贪心逐 token 对比** eager 基准 | token 序列完全一致 |
| **L3** | 端到端性能 | tok/s、prefill tok/s | 记录到 `PROGRESS_PORT.md` |

**★ `nvprof` / `nsys` / `ncu` 对 xinfer 全部失效**（Driver API / 崩溃 / 无计数器权限）⇒ 性能定位只能靠 **L0 基准 + CUDA event**。

**★ 跑端到端测试的规矩**：
- 固定 prompt + **greedy**（温度 0），输出**逐 token** 与基准比对，不看"看起来像不像"。
- 一次只改一处；每跑一次都写 `PROGRESS_PORT.md`（含命令、耗时、结论）。
- 跑之前先 `nvidia-smi --query-compute-apps=pid --format=csv,noheader` 确认没有残留进程占卡。

---

## Part 1：定位（先测后动，一次投入）

| 编号 | 测量 | 目的 | 怎么测 |
|---|---|---|---|
| **M1** | **prefill 分解**：长 prompt（≥4096 token），拆 attention / GDN / GEMM / cast / 其他 | 决定 prefill 先投 GDN 还是 FA | L0 基准逐个 kernel 打时间；端到端跑长 prompt 对总时间 |
| **M2** | decode 分解（已有，需随新模型复测） | 复核 GEMV 收益是否保持 | 端到端 + `XINFER_DIAG_*` |
| **M3** | KV cache 实际容量 & 占用 | 确认"充足"且没被 cutlass workspace 挤 | `sm70_diagnostics` 输出 |

**★ M1 必须先做** —— 现在只有 22 token 的样本（0.21 s / 106 tok/s），完全看不出瓶颈。

---

## Part 2：Decode 路线（收益确定、风险低 ⇒ 优先）

> 现状：eager **38.74 tok/s**（26 ms/step）。DRAM 下限 13.7 ms/step（≈73 tok/s）。
> 1Cat 4 卡 103.87 tok/s ⇒ 折合 2 卡 ≈52 tok/s。

| 编号 | 项 | 预估收益 | 风险 | 依据 / 做法 |
|---|---|---|---|---|
| **D1** | **NVFP4 QPN2 split-K GEMV** | **+5 tok/s** | 低 | 468 → 818 GB/s。抄 1Cat `fp8_qpn8_split_cta_m1` 的 CTA SplitK。TurboMind QPN2 布局 `[tile=N/32][group=K/16][lane=32][8B]`，Volta `mma.m8n8k4`，scales 用 `__half`，额外 reduce kernel ≈0.25 ms/step；Phase 1 在 prefill 把 QPN2→u32 转成每层 22 MB scratch（+3.5 ms）。新文件 `nvfp4_qpn2_sm70.cu` + `bench_qpn2.cu`；Rust 侧 F16 在 GPU 上是裸 `u16` |
| **D2** | **CUDA graph 正确化** | 28.2 → ~21.5 ms/step（≈46 tok/s） | **高** | 见 Part 4（三层修复） |
| **D3** | 剩余 launch 间隙清理 | 每次 launch ~7.8 µs × 1305 | 低 | 靠 D2 的图吃掉大部分；剩余的是图外的 kernel |

**★ D1 已有现成落点**：我们已有的 SM70 GEMV（`attention.rs` `ba78b00`）就是同一条路的上一版，
1Cat 的 QPN2 是它的"分块 + split-K"升级版 ⇒ **可以在现有基准上直接加变体，不需要新代码骨架**。

---

## Part 3：Prefill 路线（当前最痛：用户反馈只有 200+ tok/s）

> **实测的三个 `[slow]`（远端真实输出）**：
> ```
> [slow]  flashinfer_prefill — the FlashInfer prefill kernels require SM80+ (cp.async is not available on Volta)
> [slow]  cutlass_kernels  — build lacks the `cutlass` feature
> [slow]  gdn_fast_prefill — the chunked GDN prefill kernel requires SM90
> ```
> 外加 **16 条**（= 16 个 full-attn 层）：
> `flashinfer disabled for this layer: unsupported gqa/head_dim combination, falling back group_size=6 head_size=256 is_prefill=true prefill_sm_supported=false`
> ⇒ **prefill 三条快路径全废**。而 **64 层里 48 层是 GDN** ⇒ GDN 那条影响面最大。

| 编号 | 项 | 影响层数 | 风险 | 做法 |
|---|---|---|---|---|
| **P1** | **GDN prefill 上 SM70（chunked 并行）** | **48 层** | 中 | **现状已查清**（见下）：prefill 时 GDN 走的是 `gdn.cu` 里的**顺序 recurrence**（`gated_delta_rule_recurrence_kernel_tiled` / `_fallback`）—— `gdn.cu` 2841 行里**没有任何 chunked 实现**（grep `chunk` 无结果）；而快的 chunked 路径 `gdn_flashinfer_prefill.cu`（275 行）调的是 **FlashInfer 上游** `flat::launch_delta_rule_prefill_kernel_gbai<..., cutlass::arch::Sm90, ...>` 且带 `major != 9` 运行时硬检查 ⇒ **改 arch tag 不够，要动 FlashInfer**。<br>**⇒ 两条路**：<br>**P1a** 抄 1Cat `flash-attention-v100/kernel/h3/`（`fmha.h` 39 KB + `default_fmha.h` + `prefetch_mma.h` + `forward.cu`，共 ~80 KB）：**独立于 FlashInfer**，落在我们的 attention-rs 里；**但它是 D128 + MHA（"no GQA head packing"）⇒ 要改成 D256 + GQA**。<br>**P1b** 抄 1Cat `flashinfer-sm70/csrc/h3_noncausal_sm70.cu`（在 FlashInfer 内加 SM70 分支）⇒ 要 fork FlashInfer，成本更高。<br>**★ 推荐 P1a**：算法结构（chunked 并行 scan）可复用，CUTLASS 2.x 的 QK/PV GEMM 用我们自己的 SM70 `m8n8k4` 替换即可（我们已有 SM70 GEMV/mma 经验）。 |
| **P2** | **FA prefill 走 CUTLASS 2.x** | 16 层 | 高 | 抄 1Cat `csrc/attention/sm70_v37/`：**形状逐字匹配我们**（"six query heads, one KV head and head dimension 256"；我们 GQA group=24/4=6，head_dim=256）。做法：prefix 用 **M128/N256/K32 SM70 TC GEMM**，QK epilogue 做 FP32 logits→tile-local 概率→FP32 max/sum→舍入 FP16，PV 用 FP32 累加。**1Cat 实测 D256 GQA @KV128K：62 TFLOPS（Q8000）/ 68（Q8192）/ 47（旧 Split-D 路径）** ⇒ 这是 prefill 的对标数字 |
| **P3** | **FlashInfer prefill 的 SM70 移植** | 16 层 | 高 | 1Cat `flashinfer-sm70/` 是 Volta WMMA 兼容层（`nvcuda::wmma` 发 `HMMA.884`）+ `h3_noncausal_sm70.cu`；README 自述 *"does not enable an attention backend … primitive probe only"* ⇒ **参考价值有限**，且它的 KV 变更是 contiguous，我们的是 paged ⇒ 差距大 |
| **P4** | **prefill CUDA graph** | — | 中 | prefill chunk 形状固定（2048）⇒ 形状可捕获 ⇒ 可做图。但 **prefill 是 compute-bound，launch 间隙占比小 ⇒ 收益只有几个百分点**。**前提**是 `CUDA_GRAPH_SLAB_PLAN.md` 的 slab 分配器（解决捕获期分配变成 MEM_ALLOC 节点）⇒ 排在最后 |

**★ 1Cat h3 的技术要点（`flash-attention-v100/kernel/h3/UPSTREAM.md`，抄的时候要照抄的部分）：**
- *"SM70 CUTLASS TensorOp QK/PV with **FP32 accumulation**, online max/sum and output state"*
- QK tile 64×64 或 64×128，PV tile 64×128；`prefetch_mma.h` 适配 SM70 PV 主循环（源自 CUTLASS `gemm/mma_from_smem.h`），
  **接受第一个 softmax 之前加载的 V fragment**，保持 CUTLASS 的软件流水与累加顺序
- *"The wrapper supports **graph replay** and copies an unaligned or strided input only when necessary"* ← 对我们后续做 CUDA graph 很重要
- 自动 dispatch：Q/K 长度都 ≥1024 时用 128 keys，否则 64
- 注册数：`246 registers/thread without spills`、`34,304 shared bytes/block`（K128 变体 255 regs，K64 用 210）
- **没有额外常驻 device 分配** ⇒ 对 16 GB 卡的显存预算友好
- 坑（README 记录）：*"Softmax row distribution uses the actual warp count, rather than inferring eight warps from the tile area; the latter leaves rows 32..63 unnormalized and caused the old length-63 NaNs."* ⇒ **抄的时候别把这个 bug 一起抄进来**

### 3.2 SM70 走 CUTLASS 的落地方式（关键技术点）

**★ CUTLASS 2.x 在 SM70 可用**（Volta `mma.m8n8k4`）；**3.x 不行**（要 TMA / block-scaled / sm90+）。

1Cat 的 SM70 路线全部用 2.x：
`csrc/attention/sm70_79t/prefill.cu` include `cutlass/gemm/kernel/default_gemm.h` +
`cutlass/gemm/kernel/gemm.h`，配**本地补丁** `MmaPipelined79T`
（`csrc/attention/sm70_79t/include/cutlass/gemm/threadblock/mma_pipelined.h`）。

**我们的现状**（`attention.rs/src/kernels/build.rs:292-321`）：
```rust
let cutlass_reachable = compute_cap >= 90;      // ← SM70 上 CUTLASS 被整个关掉
if cutlass_reachable { builder = builder.arg("-DUSE_CUTLASS"); }
else { cargo::warning("CUTLASS disabled: sm{} has no reachable CUTLASS GEMM path", ...); }
builder = builder.with_cutlass(Some("b46b16d003484063bca4ed365e44095c4c6ed633"));
```
⇒ **需要新开一条 `-DUSE_CUTLASS_SM70` 通道**（且 `with_cutlass` 要与 `CARGO_FEATURE_CUTLASS` 解耦，
否则会被迫编 3.x 的 SM90 kernel）。
**★ 利好**：cudaforge 已经把 CUTLASS v4.x 拉到 `/root/.cudaforge/git/checkouts/cutlass-b46b16d*/include`，
而 **v4.x 仍保留 2.x 的 legacy API**（`cutlass/gemm/kernel/default_gemm.h`）⇒ 1Cat 那套 API 大概率能直接用。

### 3.3 我们已有的 prefill 底子（不是从零开始）
- `src/kernels/src/flash/flash_prefill_paged.cuh`（**864 行**）—— native flash prefill，SM70 走 `m8n8k4`
- `src/kernels/src/gdn.cu`（**2841 行**）—— GDN 主 kernel
- `src/kernels/src/gdn_flashinfer_prefill.cu`（**275 行**）—— chunked GDN prefill，被 SM90 卡住

**⇒ P1 的落点就在 `gdn_flashinfer_prefill.cu`（275 行，改 arch 即可试）；P2 的落点在 `flash_prefill_paged.cuh`。**

---

## Part 4：CUDA Graph 路线（收益最大、风险最高 ⇒ 放最后）

> 三个独立问题，必须逐个解决：

| 层 | 问题 | 状态 | 做法 |
|---|---|---|---|
| **G1** | **层间 all-reduce 根本没进图**（图里 0 个 AllReduce 节点）⇒ 重放时没有 TP 归约 ⇒ logits 乱码 | **✅ 已定位并验证修复** | `XINFER_NCCL_GROUP=0`（捕获时不开 NCCL group）⇒ 图节点 2017→2273，`EVENT_RECORD` 1→129，出现 `ncclDevKernel_AllReduce_Sum_f` ⇒ 128 次 all-reduce 进图 |
| **G2** | **FlashInfer decode plan 的跨 rank handshake** ⇒ 图里必 hang（双卡 SM 100%） | ⚠️ 已定位，未解 | 把 plan 移出图（1Cat 就是把 attention/KV 排除在图外）；或用 FlashInfer 自己的图模式（但 KV 长度逐 token 变 ⇒ partition 失配 ⇒ 乱码） |
| **G3** | **两个开关重复**（`--disable-cuda-graph` 只管捕获，`XINFER_ENABLE_DECODE_GRAPH` 只管 replay，且默认状态相反） | ⚠️ 待改 | 图修好后**合并成一个开关**，删掉 env 那个 |

**★ G2 的绕行（`XINFER_SKIP_FLASHINFER=1`）只用于诊断，不是产品路径**：FlashInfer BatchDecode 只占 0.34 ms/step，
eager+flashinfer 35.55 vs eager+skip 36.10（差 1.5%）。

---

## Part 5：执行顺序（按 收益/风险 排）

| 序 | 动作 | 收益 | 风险 | 前置 |
|---|---|---|---|---|
| **1** | **工作流落地**：`[patch]` path 依赖 + 独立基准骨架 + 编译成本实测 | 编译 30 min → 分钟级 | 无 | — |
| **2** | **M1 prefill 分解**（长 prompt 定位） | 0（但是前提） | 无 | 1 |
| **3** | **D1 NVFP4 QPN2 split-K** | +5 tok/s | 低 | 1 |
| **4** | **P1 GDN prefill SM70**（275 行改 arch） | prefill 大头（48/64 层） | 中 | 1,2 |
| **5** | **P2 FA prefill CUTLASS 2.x**（抄 1Cat `sm70_v37`） | prefill 大头（16 层） | 高 | 1,2,4 |
| **6** | **G1+G2 图正确化** | ≈+8 tok/s | 高 | — |
| **7** | **G3 开关合并 + 清理所有 `XINFER_DIAG_*`** | 可维护性 | 低 | 6 |
| **8** | **P4 prefill 图**（slab 分配器） | 几个百分点 | 中 | 6 |

**★ 1Cat 对照数字**：decode 4 卡 103.87 tok/s（2 卡折半 ≈52）；prefill D256 GQA 62~68 TFLOPS。

---

## 附：每条动作都要写进 `PROGRESS_PORT.md`（格式）

```
### 尝试 N：<一句话标题>（日期）
- **假设**：……
- **做法**：命令 / 文件 / 改动点
- **结果**：数字（带宽、TFLOPS、tok/s、token 序列对比）
- **结论**：保留 / 回退 / 下一步
```
