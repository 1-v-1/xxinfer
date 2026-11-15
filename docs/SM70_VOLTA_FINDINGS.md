# SM70 / Volta (V100) 移植与 CUDA Graph 诊断结论

> 本文是 `unsloth/Qwen3.8-27B-NVFP4` 在 **2× Tesla V100-SXM2-16GB** 上的可复用结论。
> 详细的逐次尝试流水在仓库根目录的 `PROGRESS_PORT.md`；本文只保留**结论、命令、开关与坑**。
> 1Cat/vLLM（`/Volumes/SMB重要/1Cat-vLLM`）是本项目的对照实现。

---

## 0. 现状速览

| 项 | 值 |
|---|---|
| 目标硬件 | Tesla V100-SXM2-16GB = SM70 / Volta，80 SM，HBM2 ~900 GB/s |
| 模型 | `unsloth/Qwen3.8-27B-NVFP4`（24.69 GB，mixed-precision） |
| 拓扑 | TP=2（2 个独立进程，每进程 1 卡） |
| decode 速度（eager，当前最佳正确路径） | **38.74 tok/s**（26 ms/step） |
| 每步 DRAM 理论下限 | 13.7 ms/step（≈73 tok/s） |
| 1Cat 对照 | 4×V100 上 103.87 tok/s ⇒ 折合 2 卡 ≈52 tok/s |

**模型结构（关键，踩过坑）**：`Qwen3_5ForConditionalGeneration`，64 层 = **48 层 GDN（线性注意力）+ 16 层全注意力**，
hidden 5120，intermediate 17408，vocab 248320，head_dim 256，num_heads 24，num_kv_heads 4，rope_theta 1e7。
**layer 0 是 GDN 层，不写 paged KV cache。**

---

## 1. 环境与命令

### 安装（用户实测可用的命令）
```bash
ENABLE_FLASHINFER_SOFTWARE_FP8=1 cargo install --git https://github.com/1-v-1/xxinfer.git \
    --branch perf/sm70-volta xinfer --features cuda,nccl,flashinfer
```
> `--features cuda,nccl,flashinfer` 已经等价于加上 `graph`：`cuda = [..., "graph"]`，
> 而 `graph = ["flash", "attention-rs/graph", "candle-core/graph"]`（上游 `b2576c5` 引入）。
> 额外写 `graph` 无害但冗余。

### 运行
```bash
xinfer --m unsloth/Qwen3.8-27B-NVFP4
```
`--m` 传不存在的路径会被当作 HF repo id，**自动下载并缓存**（`src/utils/downloader.rs`）。
模型源务必用 **`unsloth/Qwen3.8-27B-NVFP4`**；结构相同的 `orcarouter/...-Uncensored-NVFP4`
是另一个微调版，权重不同。

### 关键环境变量
```bash
XINFER_NVFP4_SM70_HMMA884=1      # 启用 SM70 的 NVFP4 HMMA.884 kernel（提速主力）
ENABLE_FLASHINFER_SOFTWARE_FP8=1 # 编译期：FlashInfer 在 SM70 用软件 FP8
```

### 远端构建
```bash
export RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo \
       ENABLE_FLASHINFER_SOFTWARE_FP8=1 XINFER_NVCC_THREADS=16
cargo build --release --features cuda,nccl,flashinfer
```
* **只改 Rust** ⇒ 约 **1m40s**
* **改了 candle / cudarc / CUDA kernel** ⇒ 全量重编，**几十分钟** ⇒ 放后台
* 独立 nvcc 基准 ≈1 min 编译，是唯一便宜的迭代手段

---

## 2. 性能：做了什么、还剩什么

### 已经拿到的提速（真实 27B 实测）
| 改动 | 效果 | 位置 |
|---|---|---|
| **重写 SM70 decode GEMV**（"一个 warp 一列"） | **10.16 → 36.14 tok/s** | `attention.rs` `ba78b00` |
| fused SiLU-and-mul | 35.55 → **36.13 tok/s** | `mlp.rs` `6830d86` |

**GEMV 重写要点**（1Cat dense decode 思路的 SM70 化，详见 `PROGRESS_PORT.md` attempt 168）：
1. 先定位：模型是 mixed-precision，**NVFP4 只覆盖 42% 的线性层**，FP8 占 51%、lm_head BF16 占 11%。
   两个 GEMM 原先只跑到带宽上限的 **8~12%**（FP8 94.8 GB/s、NVFP4 105.2 GB/s，上限 ~800 GB/s），
   累加 ≈112 ms/step，与实测 ~98 ms/step 吻合 ⇒ **两个 GEMM 吃掉几乎全部 decode 时间**。
2. 手法：一个 warp 负责一个输出列，32 lane **striped** 切 K；权重 **direct global→register**（`__ldcs`），
   不做 global→shared 中转；反量化用**无分支位技巧**（E4M3→F16: `(sign<<15)|(((code&0x7f)<<7)+0x2000)`）；
   FP32 累加。
3. 收益：FP8 0.3055→**0.0367 ms（8.33×，715 GB/s，数值逐位一致）**；
   NVFP4 0.2614→**0.0595 ms（4.39×，468 GB/s，数值更准）**。

### 剩余空间（28.2 ms/step = kernel 19.2 ms + 间隙 9.2 ms）
| 方向 | 预估 | 状态 |
|---|---|---|
| **CUDA graph 正确化** | 省 ~1305 次 launch 的间隙 ≈6.6 ms/步 ⇒ ~46 tok/s | 根因已定位，见 §3 |
| **NVFP4 QPN2 split-K** | 468 → 818 GB/s（1Cat `fp8_qpn8_split_cta_m1` 手法）⇒ 每步再省 ~2.5 ms ⇒ **+5 tok/s** | 未做，方案见 §7 |
| DRAM 下限 | 13.7 ms/步（≈73 tok/s） | — |

---

## 3. CUDA Graph 路径：三个已定位的根因

> 设计前提（`src/utils/graph.rs` 的注释）：**TP 的两个 rank 是独立进程、各自独立 launch 图，
> 图内 collective 没有跨进程 rendezvous ⇒ collective 不能放在图里。**

### 3.1 ✅ 已修复：层间 all-reduce 根本没有进图（最严重）
* **现象**：捕获的图里 **0 个 AllReduce 节点**，只有 1 个 `EVENT_RECORD` + 1 个 `WAIT_EVENT`。
  ⇒ 重放时**完全没有 TP 归约**，每层输出只是本 rank 的分片 ⇒ **logits 必然乱码**。
* **根因**：`ncclGroupStart/End` 把一步里的 128 次 all-reduce **聚合**，
  NCCL 把聚合后的 collective launch 到**内部流**（mixing 模式）⇒ **不进图**。
  对照：图里那个**单个** all-gather 走的是正常路径 ⇒ **进图了**。
* **修复**：`XINFER_NCCL_GROUP=0`（捕获时不开 NCCL group，每个 collective 单独发到捕获流）
  ⇒ 实测图节点 **2017 → 2273**，`EVENT_RECORD` **1 → 129**，出现 `ncclDevKernel_AllReduce_Sum_f`
  ⇒ **128 次 all-reduce 成功进图**，且不死锁。
* **注意**：`ncclGroupStart/End` 本身实现是正常的（cudarc `group_start()` 就是 `ncclGroupStart()`）。

### 3.2 ⚠️ 已知绕行：FlashInfer decode plan 的跨 rank handshake ⇒ 图里必 hang
* **现象**：prefill 打完（`Prefilling 1 seq(s) [0]: 23 total tokens in 0.22s`）后卡住，
  **双卡 SM 100%**，进程不返回，日志停在 `[flashinfer][decode_plan] pbs=1 split_kv=0 ...`。
* **机制**：FlashInfer 的 decode **plan** kernel 是**跨 rank 的 grid-wide handshake**
  —— 两个 rank 的 plan kernel 必须都到达才能一起完成。放进图后两 rank 各自独立 launch，
  **没有 rendezvous** ⇒ rank0 永远等不到 rank1。
* **证据**：`XINFER_SKIP_ALLREDUCE=1`（collective 变 no-op）**仍 hang** ⇒ 不是 all-reduce 的锅；
  `XINFER_SKIP_FLASHINFER=1`（走 native flash）**完全跑通**（39.84 tok/s @256 tok）⇒ 是 plan 的锅。
* **状态**：**`XINFER_SKIP_FLASHINFER=1` 是诊断旁路，不是提速手段**（FlashInfer BatchDecode 只占
  0.34 ms/step；eager+flashinfer 35.55 vs eager+skip-flashinfer 36.10，差 1.5%）。
  真正的解法是让 FlashInfer 的 plan **不进图**（像 1Cat 那样把 attention/KV 排除在图外），
  或用 `XINFER_FI_GRAPH_MODE=1` 让 FlashInfer 自己管图（但 KV 长度逐 token 增长 ⇒ partition 失配 ⇒ 乱码）。

### 3.3 ℹ️ 设计如此：词表 all-gather 在图外
`gather_logits()` 在 replay **之后**执行。注释写明原因就是 §3 的前提。
⇒ **图只产出本 rank 的 vocab 分片**，采样前再 all-gather。

---

## 4. 诊断开关清单（可复用工具箱）

| 开关 | 作用 | 备注 |
|---|---|---|
| `XINFER_ENABLE_DECODE_GRAPH=1` | **真正启用图 replay** | `--disable-cuda-graph` 只控制**捕获**阶段；不设这个变量 ⇒ 始终走 eager |
| `XINFER_DIAG_COMPARE=1` | **图输出 vs 同输入 eager 输出**逐元素对比（`max_diff` / `argmax`） | 定位图 bug 的**首选工具** |
| `XINFER_DIAG_KV=1` | 重放前后 dump KV cache（**每层**，per-rank 写 `/tmp/kvdiag_<pid>_<tid>.log`） | 多 rank stdout 会逐字符交错，**必须写文件** |
| `XINFER_DIAG_NODES=1` | 捕获图的**节点统计 + kernel 名字直方图 + KV kernel 的参数** | 需与 `XINFER_DIAG_GRAPH=1` 同时设 |
| `XINFER_DIAG_NCCL=1` | 打印每次 all-reduce 的序号与当时流是否在捕获中 | 判断 collective 有没有在捕获期间发出 |
| `XINFER_DIAG_STATE=1` | 打印 GDN / KV 状态摘要 | 判断状态有没有推进 |
| `XINFER_GRAPH_MAX_STEPS=N` | 只让图服务前 N 步，之后回落 eager | 用来区分「图算错」与「图破坏状态」 |
| `XINFER_NCCL_GROUP=0` | 捕获时**不开 NCCL group** ⇒ all-reduce 进图 | 见 §3.1 |
| `XINFER_SKIP_FLASHINFER=1` | 跳过 FlashInfer，走 native flash | **只为绕 hang**，非提速 |

> ⚠️ 上面多数开关在 `perf/sm70-volta` 分支上**尚未提交**（诊断代码在本地 patch 里）。
> 需要时用 `my_diag.patch` 贴回。

---

## 5. 踩过的坑（避免重走）

1. **layer 0 是 GDN 层，不写 KV cache。**
   之前一直 dump layer 0 的 KV 得出「图没写 KV」的结论 —— 错。
   Qwen3.8 是 **48 GDN + 16 full-attn** 混合架构，图里的 16 个 `flash_reshape_and_cache`
   对应的是散布的 full-attn 层。**dump 必须遍历所有层。**
2. **KV cache 的 slot 数是 512（block_size=512），不是 64**（config 里的 `block_size: 64` 是另一个量）。
3. **CUDA stream capture 语义**：捕获期间 kernel **只被记录、不执行**；
   H2D/D2D 拷贝变成 MEMCPY 节点 ⇒ 「捕获时的 KV 写没落盘」是**正常现象**，不是 bug。
4. **捕获时的 kernel 参数值不重要**：kernel 在重放时读的是**刷新后**的 buffer 内容。
   （实测 `graph_vars.slot_mapping` 的 device 指针 == 图里 kernel 的 `a4` 参数 == 刷新后的缓冲区 ⇒ 刷新是生效的。）
5. **旧诊断的 `addr()` 只支持 F32/BF16/F16**，对 I64/U32 返回 0 ⇒ 曾据此误判「指针不匹配」。
6. **`self.device.synchronize()` 不存在**（E0599）⇒ 用 `lib().cuCtxSynchronize()`。
7. **多 rank 的 stdout 会逐字符交错** ⇒ 诊断一律写 per-rank 文件。
8. **性能剖析工具对 xinfer 全部失效**：`nvprof`（只 hook Runtime API，xinfer 走 Driver API）、
   `nsys 2023.4`（`QuadDAnalysis::EventCollection::CheckOrder` 崩）、`ncu`（`ERR_NVGPUCTRPERM` 无计数器权限）。
   ⇒ 只能用**独立 nvcc 基准 + CUDA event 计时**。
9. **不要** `pgrep -f`（会匹配到自己）；用 `nvidia-smi --query-compute-apps=pid --format=csv,noheader`。
   **不要** `pkill -f "target/release/xinfer"`（会杀掉 SSH 通道）⇒ 用 `pkill -9 -x xinfer`。
   **不要** `nvidia-smi -pm 0`；**不要**同时跑两个 xinfer。

---

## 6. CUTLASS 在 V100 上：不是"用不了"，而是"用错了版本"

> ⚠️ 本节在 2026-11-15 被修正过一次。**先前写的「CUTLASS 在 SM70 上用不了」是错的** ——
> 1Cat 的 SM70 注意力内核正是建立在 **CUTLASS 2.x** 之上的。

### 6.1 两代 CUTLASS 的架构支持（关键区分）

| CUTLASS | API 风格 | 支持架构 | 对 V100(SM70) |
|---|---|---|---|
| **2.x** | `cutlass::gemm::kernel::DefaultGemm` + `MmaPipelined` | **sm_70 起**（Volta `mma.m8n8k4` / `HMMA.884`） | ✅ **可用** |
| **3.x** | CuTe / `CollectiveBuilder` / block-scaled / TMA warp-specialized | sm_90+ | ❌ 不可用 |

**1Cat 的 SM70 路线全部走 2.x**：
* `csrc/attention/sm70_79t/prefill.cu` include `cutlass/gemm/kernel/default_gemm.h`、
  `cutlass/gemm/kernel/gemm.h`，并用**本地补丁**的 `MmaPipelined79T`
  （`csrc/attention/sm70_79t/include/cutlass/gemm/threadblock/mma_pipelined.h`）
  组装 QK / PV 两个 GEMM。
* `flash-attention-v100` 的 H3 内核同样基于 CUTLASS —— `default_fmha.h` / `fmha.h`
  派生自 CUTLASS `examples/41_fused_multi_head_attention`。
* 反观 1Cat 的 **3.x** 部分（`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/`）
  只覆盖 **SM90/SM100/SM120**；`CutlassMLABackend` 只支持 SM100。

### 6.2 为什么我们的 `--features cutlass` 对 V100 仍然没用
* xinfer 的 `cutlass` feature = `attention-rs/cutlass`，编出来的是 attention-rs 里**现成的 3.x kernel**：
  `KernelTmaWarpSpecializedCooperativeFP8Blockwise`（SM90）、
  `KernelTmaWarpSpecializedBlockwise1SmSm100`（SM100）、`GemmUniversal+StreamK`（SM80+）。
  **这些在 Volta 上一个都跑不了** ⇒ **保持不开 `--features cutlass` 仍然是对的**
  （只增编译时间，SM70 零收益）。
* 项目对此已有显式防护，开不开 feature 都不影响 SM70：
  * `uses_cutlass_workspace()`（`src/utils/kvcache_allocator.rs`）：只有 `sm >= 90` 才预留 **512 MiB**
    workspace ⇒ **SM70 上不会白占显存**（KV cache 不受影响 ✅）
  * `weight_scale_cutlass`（`src/models/layers/linear.rs`）：sm90/sm100 才非 None ⇒ SM70 恒为 `None`
  * `mxfp4_linear.rs` 的 cutlass 分支要求 **Blackwell SM100+**
* **但**：若要在 V100 上用 CUTLASS，必须**引入 2.x 风格的 kernel**（照 1Cat 的 `sm70_79t` /
  `sm70_v37` 那样自己组装 `DefaultGemm`），而不是打开现有 feature。

### 6.3 1Cat 的 SM70 注意力内核清单（prefill 优化的对照物）

| 目录 | 作用 | 与我们的匹配度 |
|---|---|---|
| `csrc/attention/sm70_v37/` | **prefill-only**，FP32 累加，GQA **Hq6/Hkv1/D256**，prefix 用 **M128/N256/K32 SM70 TC GEMM**；把 6 个 query head 打包进 GEMM 的 M 维；prefix=non-causal + tail=exactly masked | ★★★ **完全匹配**（我们 GQA group=24/4=6，head_dim=256）|
| `csrc/attention/sm70_79t/` | Q8000~Q8192 大 chunk 架构核，`prefill_dense_d256_gqa_79t_fp32`；Q4096 有原生 256-token tail 特化 | ★★★ 形状匹配；我们的 chunk=2048 走 general path |
| `flash-attention-v100/kernel/` | FA2 移植：`fused_mha_forward_paged.cu`（paged prefill）、`flash_decode_paged.cu`、`h3/`（**GDN**）、`fp8_kv_bridge.cu`（E4M3 KV） | ★★ |
| `csrc/attention/sm70_grouped_long/` | `grouped-attention.cu` 多 request 批量 prefill | ★★ |
| `flashinfer-sm70/` | Volta WMMA 兼容层（`nvcuda::wmma` 发 `HMMA.884`）+ `h3_noncausal_sm70.cu`；README 自述 *"primitive probe only"* | 参考 |

**性能对标（1Cat README 实测）**：D256 GQA 在 KV128K/KV248K 下
**62 TFLOPS（Q8000）/ 68（Q8192）/ 47（被替换的旧 Split-D 路径）**。

---

## 7. 下一步提速路线（按性价比）

1. **NVFP4 QPN2 split-K GEMV**（对标 1Cat `fp8_qpn8_split_cta_m1`）
   * 目标：468 → **818 GB/s**，每步再省 ~2.5 ms ⇒ **+5 tok/s**
   * 布局：TurboMind QPN2 `[tile=N/32][group=K/16][lane=32][8B]`，Volta `mma.m8n8k4`，
     CTA SplitK，scales 用 `__half`，额外 reduce kernel ≈0.25 ms/step
   * Phase 1 在 prefill 时把 QPN2→u32 转成每层 22 MB scratch（+3.5 ms）
   * 新文件：`attention.rs/src/kernels/src/nvfp4_qpn2_sm70.cu` + `xxinfer-dev-env/mmatest/bench_qpn2.cu`
   * `build.rs` 需要 `rerun-if-changed`；Rust 侧 F16 在 GPU 上是裸 `u16`
2. **CUDA graph 正确化 + 把 FlashInfer plan 移出图**（§3）⇒ ~46 tok/s
3. 清理所有临时诊断（`XINFER_DIAG_*`、`XINFER_SKIP_FLASHINFER`、
   `XINFER_KEEP_CAPTURE_POOL`、`XINFER_CAPTURE_MEM_POOL`、`XINFER_CAPTURE_IN_WARMUP`、
   `XINFER_CAPTURE_DECODE_SHAPE`、`XINFER_FI_GRAPH_MODE`、`XINFER_GRAPH_NO_AUTO_FREE`、
   `XINFER_SKIP_ALLREDUCE`、`XINFER_CAPTURE_CTX_LEN`、`XINFER_GRAPH_MAX_STEPS`）

---

## 8. 一句话结论

> **TP 的 collective 与 FlashInfer 的 plan 这类「跨 rank handshake」，只要放进 CUDA graph 就会坏：
> 两个 rank 各自独立 launch 图，没有 rendezvous。**
> 要么把 collective 留在图外（像词表 all-gather 那样），要么让 collective 变成普通的图节点
> （不开 NCCL group，让 NCCL 直接 launch 到捕获流）。
