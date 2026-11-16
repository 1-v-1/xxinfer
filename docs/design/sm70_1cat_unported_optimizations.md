# 1Cat 最优 prefill / decode 路径 —— 未移植优化清单

> **性质**：上游逆向产物（不是已否决清单，也不是已拍板决策）。本件只回答一个问题：**上游 1Cat-vLLM 在 SM70/V100 上当前最优的 prefill 与 decode 路径里，哪些优化还没搬进 xinfer。**
> **上游**：`/Volumes/SMB重要/1Cat-vLLM`（只读）。按 [AGENTS.md](../../AGENTS.md) D2 钉在 `bcd581d230`；本机 SMB 挂载上**没有 `.git`**（`git log` 报 "not a git repository"），**commit 无法就地校验**，下述行号取自该工作树当前磁盘内容，commit 由 AGENTS.md 保证。
> **对照对象**：`repos/xxinfer`（主仓）、`repos/attention.rs`（内核 crate）。
> **排除口径**：已被 [specs/](../specs/) 立项的（m1/p1/p2/p4/d1/d2/g1/g2/g3）与已被 [sm70_1cat_rejected_paths.md](<./sm70_1cat_rejected_paths.md>) 标 `否决/证伪/未准入` 的，**不在本清单重复**；只在需要区分易混项时点名。
>
> ⚠️ **证据口径警告（内核行号会失真）**：`xxinfer/Cargo.toml:54` 把 attention-rs 钉在 git rev **`144a07f`**，
> 而本地 `repos/attention.rs` HEAD 是 **`3810dad`**（领先 2 个提交）**且工作树有未提交改动**
> （`src/kernels/src/ffi.rs`、`src/kernels/src/nvfp4_gemm_sm70.cu`、`src/nvfp4_linear.rs` 三个文件 M）。
> ⇒ **本件引用 `repos/attention.rs` 的行号对应的是"本地源码快照"，不等于实际编译产物**。
> 涉及内核文件的行号在动手前必须先 `git -C repos/attention.rs diff 144a07f` 复核。

---

## 0. 一页结论

上游 1Cat 在 SM70 上的"最优路径"是**三层叠加**，不是单个 kernel：

| 层 | 上游实现 | xinfer 现状 |
|---|---|---|
| **L1 策略层** | `csrc/sm70_policy_fields.inc`（**67** 字段）+ `flash-attention-v100/include/flash_v100_policy.h`（**66** 字段）+ `vllm/config/sm70_native.py` / `policy_defaults.py` / `vllm/_sm70/runtime.py`（typed policy + slot 绑定 + token 序列化 ABI） | **完全没有**。只有 `XINFER_DIAG_*` / `XINFER_*_PROBE` 两个族（[DECISIONS.md D10](../DECISIONS.md)） |
| **L2 运行时自动调优层** | TurboMind GEMM `measure → reuse` 派发缓存（`ops/gemm_runtime.{h,cpp}`、`kernels/gemm/dispatch_cache.cu`、`kernels/gemm/gemm.cu` 的 `Measure/Find/Insert`、`kernels/gemm/tuner/*`）+ 导入/导出 + warmup 后冻结 | **没有**。GEMM 全靠固定启发式选择 |
| **L3 算子层** | FA-V100 的 **8 项 default-on prefill 加速** + **~15 项 decode 加速** + **79T 架构内核族** + ~20 个 SM70 专属 GEMV/MoE/采样/all-reduce 内核 | 只搬了**基线**：FA-V100 base prefill/decode、HMMA.884 GEMM、QPN2/QPN4/QPN8 GEMV、GDN verify、one-shot all-reduce；**上游已采纳的加速项一件没有** |

**最关键的一条**：xinfer 的 `flash_prefill_paged.cuh`（864 行）对应上游 `fused_mha_forward_paged.cu`（**4150 行**），
xinfer 的 `flash_decode_paged.cuh`（632 行）对应上游 `flash_decode_paged.cu`（**6443 行**）——
**两边差的都是"上游默认开启、且实测被接受"的那部分**。

---

## 1. 上游最优路径的固定契约与基线数字

上游对**与本仓目标同族**的模型（Qwen3.8 NVFP4）留了一份可复现契约：
`docs/design/sm70_qwen38_reproducible_baseline.md`。

**已采纳组件表**（上游原文 `:13-20`）：

| 组件 | 上游实现 | xinfer |
|---|---|---|
| HC projection/mix + TP4 通信 | `vllm/models/qwen4_exp/nvidia/sm70_fp16_hc.py`、`csrc/custom_all_reduce.cu` | `src/models/layers/qwen4/hyper_connection.rs` ✅；all-reduce 只搬了 one-shot push |
| NVFP4 M1 experts + 融合 W13 + 直连 W2 归约 | `csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu`、`vllm/_sm70_ops.py` | 部分（`nvfp4_qpn2_sm70.cu` / `nvfp4_qpn4_sm70.cu`） |
| 精确 QSA top-k + router | `csrc/qsa_lexicographic_topk.cuh`、`vllm/model_executor/layers/fused_moe/fused_topk_router.py` | `src/models/layers/qwen4/qsa_attention.rs` ✅ |
| Flash-V100 逻辑页序质量修复 | `flash-attention-v100/kernel/fused_mha_forward_paged.cu` 及其 planner 回归 | **未搬**（见 §3 P0-1 / P0-4） |
| GDN + 融合 FP16 投影 | `flash_qla/`、`vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` | GDN 走顺序 recurrence（p1 已立项）；`flash_qla/` 未搬 |
| 混合 PLE + 统一 prefill/decode 配置 | `vllm/v1/ple_offload/`、`vllm/config/vllm.py` | `src/models/layers/qwen4/ple.rs` ✅ |

**固定契约**（上游 `:46-60`）与 xinfer 现状：

| 项 | 上游 | xinfer | 差距 |
|---|---|---|---|
| 精度 | FP16 act/KV + TurboMind W4A16 | 同 | — |
| prefill chunk | **8192** | `DEFAULT_PREFILL_CHUNK_SIZE = 8*1024`（`src/utils/config.rs:406`） | ✅ 一致（但见下方注意） |
| 长 prefill 调度 | `long_prefill_token_threshold`（默认 `max_model_len*0.04`）+ `long_prefill_token_threshold_adaptive`（`vllm/config/scheduler.py:82,89`） | 无 | **缺** |
| decode 图 | Runner V2，**full static decode CUDA Graph**（`FULL_AND_PIECEWISE`） | decode 图存在但 **experimental / opt-in**（`src/core/runner.rs:1289`） | **缺**（g1/g2/g3 在做） |
| recurrent state | `mamba_ssm_cache_dtype=auto` → 原生 FP32 | FP32（D15） | ✅ |
| MTP / prefix cache | 验收基线里**都关** | 开 | ✅ 更宽 |

**上游基线（4×V100-32G，TP4，单请求）**（上游 `:119-124`）：

| 输入 | prefill tok/s | decode tok/s | TPOT ms |
|---:|---:|---:|---:|
| 8192 | 6936.60 | 97.826 | 10.222 |
| 65536 | 6245.95 | 90.729 | 11.022 |
| 131072 | 5830.36 | 83.814 | 11.931 |
| 261631 | 5137.10 | 73.977 | 13.518 |

> xinfer 侧参照数（`docs/specs/d1-nvfp4-qpn2-splitk.md`）：decode eager **38.74 tok/s**；DRAM 下限 ≈73 tok/s；
> 1Cat 4 卡 103.87 tok/s ⇒ 2 卡折半 ≈52。**decode 缺口 ≈ 26%**；prefill 侧 1Cat 对标 **D256 GQA 62~68 TFLOPS**。

> ⚠️ **chunk 口径需在 m1 里先钉死**：代码默认是 8192（`config.rs:406`），但 [p4](../specs/p4-prefill-graph-slab.md) 写"prefill chunk 形状固定（2048）"、
> [p1](../specs/p1-gdn-prefill-sm70.md) 的 TP2 形状描述也写 `chunk=2048`。**79T 路线要求 chunk 落在 Q8000~Q8192 且 KV 对齐 32 token**，
> 若实际 chunk 是 2042/2048，则 79T 直接不适用、只能走 general path。**这是 m1 的第一个必答项。**

---

## 1.5 上游 prefill 的**两条并行默认路线**（先看这个，否则会抄错件）

上游不是"一个 prefill kernel"，而是**一个 dispatcher + 两条默认开启的路线**
（`vllm/v1/attention/backends/flash_v100/prefill_candidates.py:611-617`）：

```
PREPARED 链:  BflaPrefill → Fa2Prefill → ContiguousBhmdPrefill → ContiguousDensePrefill
FALLBACK 链:  Fp8BridgePrefill → SplitKvPrefill → PagedPrefill
```

| 路线 | 组成 | 默认开关（`vllm/envs.py:765-780`） | 准入形状 |
|---|---|---|---|
| **(A) 79T architecture dense** | `csrc/attention/sm70_79t/`（Q8000 / Q8192 / Q4096 三特化）+ cuBLAS FP32-accum QK + CUTLASS FP32-accum PV | `ARCH_128K_EXPERIMENTAL=True` | Q∈[8000,8192] 或 [4000,4096]，Hq6/Hkv1/D256/scale 1/16/KV 32 对齐 |
| **(B) Split-D N32 exact-dense** | `csrc/attention/sm70_v37/tail.cu` 的 `sm70_d256_splitd_n32_dense_fwd` + **长前缀 gather** + **Split-KV3** + 短前缀运行时 split-KV | `FA2_D256_PREFILL=True`、`GATHER_DENSE=True`、`SPLITKV3=True` | SM70/FP16/causal/D256/Q≥1024/长度 64 整除 |
| （C）v37 attention backend | `GQA_V37` | **False**（默认关，README 明写是"回滚/对照开关"） | — |

79T 的编译配方在 `cmake/sm70_79t.cmake:26-45`（README:74-76 明写 *"The CMake recipe is the source of truth"*）：
`PREFIX_PV_FP32_MMA_ACCUMULATE` / `PREFIX_QK_CUBLAS_FP32_ACCUM` / `PREFIX_TORCH_PREFIX_FP32_OUTPUT` /
`PREFIX_TORCH_BLOCK_N=8192` / `PREFIX_BATCHED_TAIL_TILE_TOKENS=320` / `PREFIX_TAIL_FINE_PV_GROUP_TILES=4` /
`PV_TB_M=128` / `PV_TB_N=256` / `PV_WARP_M=64` / `PV_WARP_N=64`；
`VLLM_SM70_79T_PREFILL` 在 SM70 FA2 构建下**默认 ON**。

> ⚠️ **对 p2 spec 的直接影响**：`docs/specs/p2-fa-prefill-cutlass2.md` 的目标件写的是 `csrc/attention/sm70_v37/`，
> 而上游 `GQA_V37` **默认是 False**。p2 抄的那条**不是上游默认路线**。
> 上游默认的是 **(A) 79T** 与 **(B) Split-D N32 + gather + Split-KV3**——**后者上游文档明写
> "The route is enabled by default"**（`docs/design/sm70_fa2_d256_prefill_pipeline.md:12-13`）。
> ⇒ **p2 应拆成两条 spec：p2a = 79T 架构族，p2b = Split-D N32 三件套（gather / N32 / Split-KV3）。**

---

## 2. P0 —— prefill 侧未移植

### P0-1 ⭐ FA-V100 D256 prefill 的 6 个**上游已默认开启**的加速模板开关

上游把 D256 prefill 的优化做成**内核模板参数**，由 `flash_v100_policy.h` 控制，默认全开：

| 上游 env（`flash_v100_policy.h:139-152`） | 默认 | 作用 | 上游实测 |
|---|---|---|---|
| `VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM` | **on** | `BLOCK_M=16 / BLOCK_N=128 / KV_STAGE_N=1 / 16 warps`（`fused_mha_forward_paged.cu:75-81`） | — |
| `VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P` | **on** | 全 P-panel 复用 | — |
| `VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH` | **on** | pair scratch 布局 | — |
| `VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE` | **on** | **BM32 phase-reuse 专用内核**（`fused_mha_forward_paged.cu:2090-2115`） | **-18.99% ~ -20.42%**（`sm70_1cat_rejected_paths.md:1032`） |
| `VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268` | **on** | 输出 stride 4（`fused_mha_forward_paged.cu:142`） | **-8.67%**，但**仅 page 784 生效**（A20） |
| `VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK` / `_PV` | **on** | Q/P swizzle 软件流水线（`fused_mha_forward_paged.cu:191-210, 523, 648-674`） | **-12.326%**，wavefronts -56.8%（`:1033`） |

**xinfer 现状**：`repos/attention.rs/src/kernels/src/flash/flash_prefill_paged.cuh` 共 **864 行**；
对上述 6 组关键词（`BM32` / `PHASE` / `swizzle` / `stride_268` / `ALL_P` / `PAIR_SCRATCH` / `LOW_SMEM` / `contig_fast` / `scalar_qk` / `scalar_pv` / `wmma_qk`）**grep 全部 0 命中**。
即：xinfer 搬的是**基线** kernel，上游默认开的加速**一件没搬**。

> ⚠️ **易混项提醒**：`sm70_1cat_rejected_paths.md:184-192`（A21）否决的是"**单独**把 Q-tile 从 BLOCK_M 16 改成 32"；
> A21 与**被接受**的 BM32 phase-reuse 是两件不同的事（附录 B `:1032`）。**不要因为 A21 就跳过 BM32 phase-reuse。**
> 同理 A20 记录：被接受的是"output-accum pitch 264→268"（仅 page 784），被否的是 pitch 136→144、p_strict 双 16x24、contiguous pitch 40→48、`prefetch.global.L1`(+1.463%)、aligned column-major(+4.157%)。
> 另：D13 记"跨 block 软件流水线"本身也是接受项（-2.377%，与 Q/P swizzle 合并后 -12.326%）——
> 被否的是 runtime final-block 谓词版（A18/A26）、8+8 named-barrier（A7）、4 累加器管道（A8）。

### P0-2 ⭐ 79T 架构内核族（上游 D256 GQA 的**默认路线**）

上游 `csrc/attention/sm70_79t/`（`prefill.cu` / `prefill_q4096.cu` / `prefill_q8192.cu` / `shared_workspace.h` / `stable_rows.cuh` / `legacy_tail_adapter.cu`）
是 **Q8000-core** 架构 kernel，按 [README.md](</Volumes/SMB重要/1Cat-vLLM/csrc/attention/sm70_79t/README.md>)：

- **默认选中**（`VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL=1`，回滚 env 记在 `sm70_1cat_rejected_paths.md:891`）。
- 准入形状：FP16 Q8000~Q8192 / Hq6 / Hkv1 / D256 / causal / scale 1/16 / KV 对齐 32 token。
- **三个特化**：Q8000（320-token tail 特化）、**Q8192（原生 256-token tail 特化）**、**Q4096（半高 score buffer，384 MiB 而非 768 MiB）**。
- **前导 padding 分桶**：Q4000~Q4095 → Q4096；Q8001~Q8191 → Q8192（保 bottom-right 因果对齐，额外算量 <2.4%）。
- **非连续 paged KV 汇聚进可复用 workspace**（多请求批次也支持）。
- 算力：Q8000 **62 TFLOPS** / Q8192 **68 TFLOPS** / 被它替换的旧 Split-D 路线 **47 TFLOPS**；
  **qualified medians > 75 TFLOPS @KV128K / KV256K**。
- 质量取舍：**scores 与 probabilities 存 FP16 + 采样式 score maxima**（近似 attention，需 sampled-FP32-oracle + 冷启动模型双关）。

**xinfer 现状**：`grep -rn "79t|79T|d256_gqa" ` 在 `xxinfer/src/` 与 `attention.rs/src/` **全部 0 命中**。
> **注意与既有 spec 的关系**：`docs/specs/p2-fa-prefill-cutlass2.md` 只规划抄 `csrc/attention/sm70_v37/`，
> 而 v37 正是**被 79T 取代的上一代**（上游 README 明写 "the v37 implementation remains available in the same extension"，v37 是回滚项）。
> **p2 的目标件选错了**：照 p2 抄完只能拿到 62~68 TFLOPS 的上一代，拿不到上游当前默认的 >75 TFLOPS。

### P0-2a ⭐⭐ 79T 的 **FP32 累加契约**（这是**正确性**项，不只是性能项）

上游 79T 的历史配方用 **zero-shift 指数 + FP16 分子**，在真实模型输入下 **PV 分子溢出 65504（实测 75310）→ inf → token 变成 -1**。
上游的修法是把整条链改成 **FP32 累加**：

| 环节 | 上游做法 | 证据 |
|---|---|---|
| QK | cuBLAS **FP32 accum** | `cmake/sm70_79t.cmake:28-30`（`PREFIX_QK_CUBLAS_FP32_ACCUM`） |
| PV | **FP32 MMA accum** | 同上（`PREFIX_PV_FP32_MMA_ACCUMULATE`） |
| prefix block 输出 + online merge | **FP32** | 同上（`PREFIX_TORCH_PREFIX_FP32_OUTPUT`）；`csrc/attention/sm70_79t/prefill.cu:5851-5868` |
| threadblock 拓扑 | **128×256 threadblock / 64×64 warp**，消除寄存器溢出 | `cmake/sm70_79t.cmake:44-45`（`PV_TB_M/N/K`、`PV_WARP_M/N/K`）；`prefill.cu:473-475`；README:48-51 |
| 数值护栏 | ①stride-8 分数最大值采样 + 4.0 margin + 指数输入上限 10.0；②V 中心化（首 4096 token 均值 \|mean\|≥0.05）；③V 按精确 2 的幂缩放（64× headroom）；④24K prefix block 写 FP32 并 FP32 merge | `csrc/attention/sm70_79t/stable_rows.cuh:14-21, 35-38, 55-72, 93-103, 150, 192-203` |
| 越界 tile 修复 | 采样漏检导致个别 tile 分数溢出时，用 64-token query tile **重算**（`repair_prefix_pv_kernel`），无 host 回读 | `stable_rows.cuh:10-13, 105-120, 129-131, 199-203`；`prefill.cu:1812, 6629-6636`；VALIDATION.md:191-194 |

上游 README:56-57 明写 *"build option defaults to ON … runtime route defaults to this FP32-accumulated kernel"*，
VALIDATION.md 达标 **75.4–76.2 TFLOPS**（这正是 §0 的 ">75 TFLOPS"）。
**xinfer 的 `flash_prefill_paged.cuh` 没有这条契约**（它不是 79T）。

> **质量提示**：79T 是**近似 attention**（scores/probabilities 存 FP16 + 采样式 score maxima），
> 上游要求"sampled-FP32-oracle + 冷启动模型"双关才能准入。按 D6（先精度后性能），
> 移植 79T **必须先过精度门禁**，不能只看 TFLOPS。

### P0-2b prefix / tail **双流重叠** + 细粒度 tail PV 分组

| 优化 | 证据 | 说明 |
|---|---|---|
| prefix 高优先级流 + tail 低优先级流 + event 同步，tail 分数与 prefix QK **并发** | `csrc/attention/sm70_79t/prefill.cu:6280-6334, 6607-6653`；`cmake/sm70_79t.cmake:37-39`（`PREFIX_TAIL_IDLE_SM_OVERLAP` / `PREFIX_TAIL_IDLE_SM_FINE_PV` / `PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE` / `PREFIX_TAIL_FINE_PV_GROUP_TILES=4`） | 在空闲 SM 上跑尾部，掩盖 QK barrier。构建宏默认开启 |
| 细粒度 tail PV 分组：Q8000 用 **320-token tile**，Q8192/Q4096 用原生 **256-token tile** | `cmake/sm70_79t.cmake:35`（`PREFIX_BATCHED_TAIL_TILE_TOKENS=320`）；`prefill_q8192.cu:9-10`；`prefill_q4096.cu:9-10`；`prefill.cu:6060-6070` | 尾部 tile 过大导致 wave 不满 |
| cuBLAS FP32-accum QK + **转置输入** + 转置 score workspace | `cmake/sm70_79t.cmake:32, 36`（`PREFIX_QK_PRETRANSPOSE_INPUTS` / `PREFIX_TRANSPOSED_SCORE_WORKSPACE` / `PREFIX_BATCHED_TAIL_QK_ALGO=CUBLAS_GEMM_ALGO11_TENSOR_OP`）；`prefill.cu:195, 2063-2067` | 让 QK 与 PV 读同一转置布局，省一次转置 |

### P0-2c 共享 score workspace + per-device ExecutionGate + CUDA-graph 安全

`csrc/attention/sm70_79t/shared_workspace.h:13-39`、`runtime_owner.h:53-71`、`prefill.cu:6032-6051, 6107-6117, 6318-6329`：
kernel 参数**按值捕获**、tail 元数据**按 KV 长度固化**、用 event 而非 host 回读。
解决两个问题：同一物理 device 上多个 engine 竞争 device-global 指针；graph replay 悬垂指针。
README:35-38 要求**每个 rank 都要出现 route counter**。
→ xinfer 没有这套 per-device 生命周期管理（[p4](../specs/p4-prefill-graph-slab.md) 只解决捕获期 slab 的一部分）。

### P0-3 GDN chunked prefill（h3）
已由 [p1](../specs/p1-gdn-prefill-sm70.md) 立项（目标件 `flash-attention-v100/kernel/h3/`，上游 `kernel/h3/` 实存）。
**不重复列**，但注意上游同目录另有 `flash_qla/`（GDN 的 QLA 内核树）未纳入 p1 范围。

### P0-4 prefill 侧的 paged KV 布局转换与 FP8-KV bridge
上游 `flash-attention-v100/kernel/` 下有 `contiguous_to_paged.cu`(75) / `paged_to_contiguous.cu`(181) /
`paged_to_contiguous_fixed.cu`(88) / `fp8_kv_bridge.cu`(249)。
xinfer `attention.rs` **grep 0 命中**这四件。
影响：非连续 KV 的 prefill 需要额外 gather；FP8 KV 存储需要 bridge。

### P0-4a ⭐⭐ Split-D N32 三件套（路线 B，**上游默认开启**）

上游 `docs/design/sm70_fa2_d256_prefill_pipeline.md` 记录的这一整套，xinfer **一件都没有**：

| # | 件 | 上游证据 | 上游实测 | xinfer |
|---|---|---|---|---|
| b1 | **Split-D N32 exact-dense 算子**：BLOCK_M=64/BLOCK_N=32 + 4×D64 chunk；8 warp 组成 4 个 2-warp 组，每 warp 独占 8 行 QK 去重，warp 对共享 P tile；PV 用 TT 映射 + 交替 PV 寄存器片 + 128-bit V store | `docs/design/sm70_fa2_d256_prefill_pipeline.md:46-72`；`csrc/attention/sm70_v37/tail.cu:28-128`（`kSmemBytes==45568`） | vs 通用 FA2 **1.60× / 1.23× / 1.30×**；vs 1Cat v1.2.2 **1.66×–2.20×** | 无 |
| b2 | **N32 顺序是质量硬要求**（不是形状调优）：N64 变体 rel L2 1.27e-4 累积后**改变官方采样 token** | `sm70_fa2_d256_prefill_pipeline.md:74-78` | — | 无 |
| b3 | **长前缀 gather 到 exact-dense workspace**：`index_select` 解开交错 `[page, K/V, 784, Hkv, D]` 布局，末页切到精确长度 | `vllm/v1/attention/backends/flash_v100/prefill_candidates.py:243-263`；`sm70_fa2_d256_prefill_pipeline.md:146-180`；`vllm/envs.py:770-772`（`PREFILL_GATHER_DENSE=True`, MIN_Q=4096, MIN_KV=8192） | 算子净减 **7.47–10.81%**；全模型 64K **省 458.7 ms** | 无 |
| b4 | **Split-KV3 三路 KV 分区**：每 CTA 可见 KV 切 3 段独立 launch（384→1152 CTA = 16 wave），独立 kernel 合并 FP32 numerator/max/sum | `vllm/v1/attention/backends/flash_v100/dense_prefill.py:276-302`；`sm70_fa2_d256_prefill_pipeline.md:182-219`；`vllm/envs.py:773-774`（`SPLITKV3=True`, MIN_KV=32768） | 64K **+8.78%**；全模型 **省 345.9 ms** | 无 |
| b5 | **短前缀 chunk 的运行时 split-KV**：按 CTA wave + merge 代价选 split 数（24 MiB FP32 partial workspace / 4096 query 行） | `dense_prefill.py:60-70, 220-258`；`sm70_fa2_d256_prefill_pipeline.md:221-245` | 110K keys 时 Q=64 **13.5 → 1.1 ms** | 无 |

准入条件（`dense_prefill.py:648-742`）：`try_sm70_fa2_d256_prefill` 门 → Q<1024 分流 → **必须 SM70 (7,0)** → `_splitd_admission`（`:1571`）。

### P0-4b page-784 的 cross-block **steady/drain pipeline**

上游 `docs/design/sm70_flash_v100_prefill_operator_optimization.md:39, 110, 138-145`：
block n+1 的 QK 与 block n 的首个 32-token PV panel **重叠**；编译期 steady 循环 + 泛型 drain。
状态：*"accepted prerequisite"*；local load **16.04M → 3.58M**，**-2.377%**。
条件：M1024/N65536/Hq6/Hkv1/D256/page784。

> 注意 A13 记"仅 warp-local QK double-B（-0.56%, 40/40）保留为 opt-in"，
> 被否的是 PV B lookahead(+1.26%)、double A+B(+4.51%)、triple-B、runtime final-block 谓词 local loads(16.04M)。
> ⇒ **cross-block 版本是被接受的，runtime 谓词版本是被否的**，别混。

### P0-4c block-FP8 prefill 投影走 CUTLASS GEMM（**投影，不是 attention**）

上游 `csrc/sm70_turbomind/ops/qwen38_prefill_cutlass.cu:25-39, 62-74`：
128×256×32 / 64×64×32 / 8×8×4 三种 tile；准入 M∈[8000,8192]、k/n 128 整除、FP16、非 gated_silu；
policy 默认 1（`:64-66`）。
xinfer 的 `attention.rs/src/kernels/build.rs:353-361` 在 SM70 **不加** `-DUSE_CUTLASS` ⇒ 这条**整条不通**。

### P0-5 ⭐⭐ prefill / decode **混批**（上游无 phase 分离，xinfer 是硬分阶段）

| | 上游 | xinfer |
|---|---|---|
| 分相 | **没有** decoding phase / prefill phase 之分（`vllm/v1/core/sched/scheduler.py:677-680`） | **硬分阶段**：`src/core/scheduler.rs:307-310` 一旦本步有 prefill 就 `return (ids, true)`，decode 段根本不执行 |
| 自适应预算 | `vllm/v1/core/sched/mixed_prefill.py`（`MixedPrefillBudget`，按实测 step 成本**在线学习** token 预算） | 无。chunk 大小是**固定配置**（`src/utils/config.rs:406` = 8192），预算只受 `max_num_batched_tokens` 约束 |
| 分相开关 | `vllm/v1/core/sched/phase_separation.py` | 无 |
| prefill 图 | `vllm/config/compilation.py:60-64` 有 `CUDAGraphMode.PIECEWISE`；`:627` 明写 "piecewise cudagraph for prefill and mixed prefill-decode batches" | **完全没有**：`src/utils/graph.rs:1532-1533` 对 prefill 直接 `bail!("Graph replay is not used for prefill!")`，也无 prefill 捕获函数 |

**为什么这是 decode 侧的第一号缺口**：xinfer 的 decode 步里已量到 **kernel 19.2 ms + 间隙 9.2 ms = 28.2 ms/step**
（DRAM 下限 13.7 ms）。间隙的根因就是**同一步里不能同时装下 prefill 与 decode**——
空出来的 SM 只能等 prefill 走完。上游靠混批把它填掉。
上游还有 `vllm/v1/core/sched/scheduler.py:552-575 _balanced_prefill_chunk`
（`VLLM_BALANCED_PREFILL_CHUNKS`）把长 prompt 切成**近似等长**的 n 块，
避免"n-1 个满块 + 1 个短尾"让尾块独占一步；xinfer `src/core/sequence.rs:223-243` 是**满块 + 短尾**。

> **与既有 spec 的边界**：[p4](../specs/p4-prefill-graph-slab.md) 做的是 prefill 图的**捕获期 slab 分配器**（A′ 方案），
> 解决"图能不能捕"；**"要不要有 prefill 图 / 要不要混批"是上游策略问题，不在 p4 范围内**。
> [g3](../specs/g3-graph-switch-merge.md) 做的是图开关合并，同样不含混批。
> ⇒ **混批 + prefill 图 需要单独立项。**

### P0-6 partial prefix cache（上游有"部分块"概念，xinfer 没有）

| | 上游 | xinfer |
|---|---|---|
| 部分块命中 | `vllm/v1/core/partial_prefix_cache.py` 独立模块 | 无。`src/models/layers/../../core/prefix_cache.rs:69-122` 是单一 hash 链，命中粒度 = **整块**（`block_manager.rs:454` `cached_tokens = matched_blocks * block_size`） |
| 块池 | `vllm/v1/core/block_pool.py` | `src/core/block_manager.rs:394-516` 把"匹配 + 块分配 + 抬 `num_cached_tokens`"揉在一个函数里 |

后果：最后一个块只命中一半时，xinfer 要么整块重算、要么整块不命中。
上游的 partial 命中对本仓目标模型的 512-token block 粒度收益有限（block_size=512），
**优先级低于 P0-5，但迁移成本也低**。

---

## 3. P0 —— decode 侧未移植

### P0-7 ⭐ FA-V100 decode 的 XQA 加速族（上游 ~15 项 default-on）

上游 `flash-attention-v100/kernel/flash_decode_paged.cu`（**6443 行**）在 decode 侧叠了一整套 XQA 路线；
xinfer `flash_decode_paged.cuh`（**632 行**）对全部 15 个标记 **grep 0 命中**。

**与 FP16-KV 基线直接相关（优先）**：

| 上游 env（`flash_v100_policy.h:88-153`） | 默认 | 上游实现位置 |
|---|---|---|
| `VLLM_FLASH_V100_XQA_PADDED_SMEM` | **on** | `flash_decode_paged.cu:148` |
| `VLLM_FLASH_V100_XQA_G6_QK_PIPELINE`（warps 6/8） | **on** | `flash_decode_paged.cu:332-340` |
| `VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING` | **on** | `flash_decode_paged.cu:127` |
| `VLLM_FLASH_V100_XQA_BLOCK784_INDEX` | **on** | `flash_decode_paged.cu:336` |
| `VLLM_FLASH_V100_XQA_G6_P1024_AUTO` | **on** | `flash_decode_paged.cu:229` |
| `VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH` | **on** | `flash_decode_paged.cu:272` |
| `VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS` | **on** | `flash_decode_paged.cu:302` |
| `VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH` | **on** | `flash_decode_paged.cu:307` |
| `VLLM_FLASH_V100_XQA_SPLIT_REDUCE_DIM_TILE` | 8 | `flash_v100_policy.h:352-359` |
| `VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA` | **on** | `flash_decode_paged.cu:247` |
| `VLLM_FLASH_V100_DECODE_PARTITION_SIZE` | 可覆写 | `flash_v100_policy.h:277` |

**依赖 FP8/E4M3/E5M2 KV（次优先，目标模型用 FP16 KV）**：
`XQA_E4M3_BATCH_XQA`(+`_OPTIMIZED`)、`XQA_E4M3_PAGE800_FASTPATH`、`XQA_E5M2_G6_DUAL_CTA`、`XQA_E5M2_G6_SPLIT_REDUCE`、
`XQA_E5M2_PARTITION_PAGE_IDS`、`XQA_E5M2_PAIR_LOAD`、`XQA_E5M2_BATCH_WIDE_LOAD`、`XQA_E5M2_G6_WAVE_PARTITIONS` 等。

### P0-8 ⭐ TurboMind GEMM 运行时自动调优 + 派发缓存（**L2 层**）

上游机制（已由本仓逆向记录在 `sm70_1cat_rejected_paths.md` §3；此处补全上游源码级细节）：

- **决策链** `ops/gemm_runtime.cpp:448-487`：超 `tune_max_m` → `kDefault`；有导入缓存 → `kReuse`；
  该族不参与调优 → `kDefault`；warmup 后已冻结 → `kReuse`；该 shape 已测过 → `kReuse`；
  **图捕获中 → 只复用/默认，绝不测量**（否则递归加锁死锁，`:474-484`）；否则首次 → `kMeasure`（`:485-486`）。
  策略枚举：`DispatchPolicy ∈ {kDefault, kMeasure, kReuse, kAppend}` + flag
  `kPreserveDefaultSplits / kPreserveDefaultSplitCount / kMxfp4MoeGroupedM8Fast / kSm70Fp8PrefillPrescaled /
  kSm70Nvfp4Prescaled / kPreserveDefaultPartition`（`gemm.cu:440-486`）。
- **SM70 专属搜索空间**（`kernels/gemm/gemm.cu:534-546`）：`max_splits=16`、`max_waves=32`、`swizzle={0..4}`、
  `top_k=0`（全量枚举）、`min_iter=2`、`max_iter=20`、`max_time=2s`。
  注释原文：*"V100 decode is dominated by many tiny GEMM/GEMV problems. A broader search space consistently finds better launch specs."*
- **缓存键**：`GemmDesc` 是 **25 元 tuple**（arch/类型/order/striding/pack/量化类型+group_size/batch_dim/group_axis/m/n/k/num，
  `dispatch_cache.cu:117-124`；`d.epilogue` 不参与）；`KernelDesc` 另有 **26 元**比较键（`:36-43`）用于 import 反查。
  两级结构：`cache_` 是 `map<GemmDesc, Flat>`，`Flat.idxs` 按 batch_size 排 `(batch_size, spec_index)`。
  `Find` 精确匹配 batch_size；**`LowerBound` 取 ≤ 请求 batch 的最大已测 batch**（`:151-172`）⇒ 复用较小的已测计划而不是测新形状。
- **测量前 L2 flush**：`tuner/measurer.cu:63-87` 按 `l2CacheSize` 分配 buffer 并 memset ⇒ 冷测量，
  否则 L2 命中会选出"偏快但非最优"的 kernel。停止条件 `tuner/stopping_criterion.cc:10-27`。
- **精确形状 fast selector**（**默认 on**）：`gemm.cu:125-140, 159-179` 直接锁定 `(cta_m, cta_n, cta_k, splits, swizzle)`，
  绕过 tuning —— 避免启动期测量噪声与跨 rank 计划不一致。
  对应策略字段 `nvfp4_qwen38_moe_fast_prefill`（`sm70_policy_fields.inc:54` 默认 true）；
  `gemm_runtime.cpp:546-552`（NVFP4 Qwen3.8 TP4 M=1 精确形状直接 kDefault）、`:633-647`（MXFP4 grouped M8 → `kMxfp4MoeGroupedM8Fast`）、
  `:619-631`（FP8 grouped BMM decode 精确形状 kDefault）。
- **冻结语义**：warmup 未测到的 shape 复用"下一个更大已测 plan"，而不是在首个请求到达时测量
  （上游原话：否则首请求会因 1.4~3.7 s 的测量而变慢，且 kernel 选择依赖请求顺序）。
- **导入导出**：`dispatch_cache.cu:79-113, 212-236`、`gemm_runtime.cpp:454-456, 578-585`；
  env `sm70_gemm_import_cache` / `sm70_gemm_export_cache` / `sm70_gemm_freeze_tuning`。
- **三份 kernel 注册表**（`kernel/sm70_884_{4,8,16}.cu`，由 `ops/tm_registry_sm70.cu:14-16` 依次注册）：
  `sm70_884_4.cu`(558 行，含 GGUF/FP4 batch supply/MXF4/NVF4 与 4 个**解码专用**形状闸门)、
  `sm70_884_8.cu`(167 行，FP8-E4M3 `Config_QuantizedBatch` + 3 个解码形状闸门)、`sm70_884_16.cu`(35 行，纯 FP16 dense 家族)。

**xinfer 现状**：GEMM 全部走固定启发式（Rust 侧 dispatch），**无 per-shape 自动调优、无派发缓存、无导入导出、无 fast selector**。
`d1`（QPN2 split-K）是**手工**做的一次性优化，性质上替代不了按 shape 自适应的派发。

> ⚠️ **路线澄清（重要）**：上游 `docs/design/sm70_qwen38_nvfp4_decode.md:41-49` 的路线表写明
> "NVFP4 gate/up / down → **TurboMind N32**, lookahead 1/2"——即**上游 NVFP4 的默认路线是 TurboMind W4A16**，
> 而自研 QPN（QPN4/QPN8）走 FP8 路线且 **QPN8 仍是 opt-in、未默认 promotion**（`sm70_qwen38_qpn8_decode.md:17-21`）。
> ⇒ 本仓 `d1`（NVFP4 QPN2 split-K）与上游**默认路线不是同一条**，对标数字不要混用。

### P0-9 ⭐ decode 侧专属 SM70 内核（`csrc/sm70_turbomind/ops/`，上游共 77 个文件）

| 上游文件 | 行数 | 作用 | 上游证据 | xinfer |
|---|---:|---|---|---|
| `top1x_sm70.cu` | 99 | TP4 greedy top-1，**不 gather logits**，两跳递归倍增 NVLink 交换 8B (value,id,tag,tag)；归约序是全序 ⇒ 与 all-gather+argmax 等价；device counter 作 epoch tag ⇒ **CUDA-graph 可重放** | `:1-10, 46-75` | 无 |
| `sm70_top20_philox_sampler.cu` | 289 | batch-one top-20 **精确稀疏 Philox 采样**：只对 top-20 取随机数，复刻 native curand 每线程 4 值序列与 `exponential` 舍入 | `:34-52, 107-148` | 无（`gpu_sampling.cu` 是 vLLM 通用版，且**无 argmax kernel**） |
| `sm70_sampler_graph_branch.cu` | 289 | 采样进图分支：CUDA Graph conditional node 在 reference（原生）与 compact（Philox）子图间运行期切换；并摘除 NCCL all-gather 的 event bridge | `:54-69, 209-287, 78-203` | 无（需 CUDA ≥12.8） |
| `mtp_local_top1_sm70.cu` | 123 | MTP draft 本地 top-1 分段 + warp 归约（不 gather 全 vocab） | `:51-88, 108-112` | 无 |
| `compact_half_topk_sm70.cu` | 96 | **无损** FP16 compact key 打包 + 两段 radix-sort merge，k=64（N≤65535, M≤32） | `:13-49, 75-86` | 无 |
| `sm70_router.cu` + `.cuh` | 82 | C++ router：**top-10 选择与 Q8_1 量化融合成一个 kernel**（`:46-70`、`.cuh:106-118`）；`key=(ordered<<9)\|(511-id)` 无分支 warp 级选择（`.cuh:10-15, 56-94`） | — | 无（xinfer 用 `moe/topk_softmax_kernels.cu` 的通用 top-k softmax，**未与量化融合**） |
| `exact_row_reduce.cu` | — | TP4 **bit-exact** all-reduce：peer IPC + release/acquire barrier + **15 种加法树**复刻 NCCL 归约序（要求 **≥80 SM**，V100-SXM2-32GB 满足 80 SM） | `:29-40, 41-77, 78-92, 142-145` | 无 |
| `glm53_fp16_gemv_sm70.cu` | — | GLM53 FP16 GEMV：16 lane/row × 8 chunk，复刻 cuBLAS 的 32 元素升序 FMA 链 | `:24-81, 83-103` | 无 |
| `mtp_moe_fp16_sm70.cu` | — | MTP MoE 无符号 13-bit 打包权重（低字节+高半字节+符号字节）+ M1 专用 staged kernel | `:11-43, 309-311` | 无 |
| `dflash2_fp16_m8_sm70.cu` | — | DFlash2 M8：每 warp 同算 gate/up 两投影、复用 A 寄存器。**上游已默认**（"The normal operator selects this route automatically"，`sm70_m8_exact_decode.md:7`；"production dispatch adds no environment flag"，`:44`） | `:28-82, 33-44` | 无 |

### P0-10 MoE decode 专属 SM70 内核

| 上游文件 | 行数 | 作用 |
|---|---:|---|
| `mxfp4_qpn_m1_sm70.cu` | **1794** | MXFP4 M=1 quadpair-N `m8n8k4`（源自 dnv2003/v100-skinny）。含：FP4/E2M1 反量化**折叠为纯整型位运算**（`:88-95`）、MXFP4 E8M0 scale **纯移位**解码（`skinny_moe_qpn_sm70.cu:35-39`）、NVFP4 scale 高/低分解（`:482-493, 495-536`）、**广播输入** 10 条路由共享 1 行激活（`:268-272, 686, 1195-1201`）、**W13 融合 SwiGLU**（`:646-778, 766-777, 1046-1063`）、**W2 direct reduce** 路由加权在 kernel 内完成（`:389-480, 471-478`）、**split-K 全在 CTA 内 FP32 规约**（无全局 partial、无 FP atomic，`:656`） |
| `skinny_moe_qpn_sm70.cu` | 571 | skinny NVFP4/MXFP4 GEMM + grouped MoE：权重 load 时预包成 mma fragment 序（`:9-13`）；RB 行块共享一次权重读取，decode(RB=1) 直接从 global 取 A；**激活 stage 与 split-K reduce buffer 共享同一块 smem**（`:200-235, 277-300`）；SPLITK/NACC ∈ {(8,1),(8,2),(10,1),(16,1),(16,2)}（`:531-566`） |
| `awq_qpn_m1_sm70.cu` | 354 | Qwen3.8 TP4 native-g32 **AWQ M=1**（逐 route 四元对，`:162`）；CTA-local FP32 归约**数值等价**于 TurboMind 串行 split-K |
| `nvfp4_grouped_decode_sm70.cu` | 406 | 多行 native-NVFP4 decode（上游注释：*Python dispatch defaults off*，`:4`）；`kMaxRoutes=160 / kPack=8 / kExperts=512` |

对应上游策略字段（`csrc/sm70_policy_fields.inc`）：
`MOE_SINGLE_TOKEN_FASTPATH` / `_PERMUTE_` / `_UNPERMUTE_`（默认 on）、`FP8_MOE_PREPARE_VEC`、
`FP8_MOE_SINGLE_TOKEN_PER_EXPERT_DISPATCH`、`NVFP4_MOE_GROUPED_PREFILL`（默认 on）、
`MXFP4_MOE_GROUPED_M8`、`MXFP4_MOE_COMPACT_GROUPED_DECODE`、`MXFP4_MOE_BROADCAST_INPUT_DECODE`、
`AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE`、`AWQ_MTP_M5_FAST_SELECTOR`、`FP8_GROUPED_BMM_DECODE`（默认 on）等。

**xinfer 现状**：`attention.rs` 有 `moe_gemv.cu`（decode，`moe.rs:564`，size_m≤128）/ `moe_gemm_wmma.cu`（prefill，`moe.rs:545`）/ `moe/topk_softmax_kernels.cu`，
但**没有**上述 skinny/QPN-M1/grouped-decode 家族，也没有对应的 per-shape 派发。
另注：`moe_gemm.cu` 的 `moe_gemm` 入口（`ffi.rs:498`）**Rust 侧无调用者，是死代码**。

### P0-11 all-reduce 家族

| 上游 | 行数 | 作用 | xinfer |
|---|---:|---|---|
| `csrc/custom_all_reduce.cu` | 1515 | SM70 **push** all-reduce（IPC 环形 push + sentinel 判空，完全绕开 NCCL，`:127-240, 615-620`）；**all-reduce + Gemma RMSNorm 融合**（`:561-701, 754-829`）；`top1_argmax`（`:1241-1268`） | `attention.rs/src/kernels/src/oneshot_allreduce_sm70.cu` **126 行**——只搬了 `world_size==2` 的 push 协议（`distributed.rs:450-453` 其余 rank 数回退 NCCL） |
| `csrc/exact_row_reduce.cu` | — | TP4 bit-exact all-reduce（15 种加法树） | 无 |
| `csrc/sm70_ring_collectives.cu` | 209 | ring collective | 无 |
| TileRT：GEMM↔all-reduce 融合（`sm70_tile_runtime_signal.cuh`） | — | 无 |

> ⚠️ **修正一条**：我最初把 `csrc/custom_quickreduce.cu` + `csrc/quickreduce/` 列为 SM70 缺口——**这是错的**。
> `csrc/quickreduce/quick_reduce.h:4` 直接 `#include <hip/hip_runtime.h>`，整个目录是 **ROCm/HIP 的 two-shot all-reduce**，
> grep `sm70|major == 7|multimem` **零命中**。V100 走的是 `custom_all_reduce.cu` 的 SM70 push 路径，与 quickreduce 无关。

上游策略字段 `top1_custom_ar` / `awq_tile_ar` / `awq_tile_overlap` / `tp4_push` / `tp8_hierarchical` / `moe_add_allreduce` /
`long_prefill_norm` / `symm_mem` 均与 TP 度数强相关（见 `vllm/config/policy_defaults.py:308-332` 的条件化启用表）。

### P0-12 ⭐ 采样路径（GPU argmax 缺失 ⇒ 每步把整词表拉回 host）

| | 上游 | xinfer |
|---|---|---|
| argmax | `vllm/v1/sample/ops/fast_argmax.py:26-53`（Triton 分块 argmax，`VLLM_FAST_ARGMAX=1` opt-in）+ `topk_topp_sampler.py`（GPU 上 in-place 完成） | **没有 GPU argmax kernel**。`src/utils/logits_processor.rs:67-70` 走 candle `argmax` + D2H；host 回退路径 `logits_processor.rs:90` / `:316` 把**整个词表** `to_vec2()` 拉回 host（vocab **248320**） |
| top-k/top-p | `gpu_sampling.cu` 只实例化 k∈{32,64,128,256}，纯 TopP 强制 k=256（`logits_processor.rs:285`） | 有，但**只在策略是 TopK/TopP/TopKThenTopP 且 k>0 时**才走 GPU（`:295-302`） |
| 拒绝采样 | `vllm/v1/sample/rejection_sampler.py`（概率拒绝采样） | `src/speculative/verify.rs:17-67` **只做 greedy argmax 前缀匹配**，无概率拒绝采样 |

⇒ **贪心 decode（ArgMax）这条路每步都有一次 248320 维的 D2H**，这是 decode 侧除 GEMM 外的头号未移植项。
上游的对应件 `top1x_sm70.cu`（P0-9）连 logits 都不 gather。

### P0-13 decode CUDA graph：**捕获默认开、replay 却 opt-in**

| | 上游 | xinfer |
|---|---|---|
| 捕获 | 默认捕获并**直接使用**（`vllm/v1/worker/gpu_model_runner.py:12582 capture_model()`，除非 `cudagraph_mode == NONE`） | 捕获默认开（`src/core/engine.rs:274-295` + `core/runner.rs:2737-2747 warmup_capture()`） |
| replay | 默认 | **opt-in**：`src/core/runner.rs:1294-1296` 要求 `XINFER_ENABLE_DECODE_GRAPH`；注释明说 *"The decode CUDA graph is still experimental on SM70: it replays, but its numbers do not match the eager path yet … it stays opt-in"*（`:1289-1293`） |

⇒ **当前状态是"白付代价不拿收益"**：捕获在启动时发生，照样吃掉 capture pool（`graph.rs:404-540`）
与 graph workspace（`kvcache_allocator.rs:798-806, 962 cuda_graph_bytes()`），
但 replay 不开 ⇒ 没有任何加速。已量到 CUDA graph 正确化预估省 **~1305 次 launch 间隙 ≈ 6.6 ms/步 ⇒ ~46 tok/s**。

> **边界**：[g1](../specs/g1-nccl-allreduce-in-graph.md)（已采纳冻结）做的是 all-reduce 进图；
> [g2](../specs/g2-flashinfer-plan-handshake.md) 做的是 FlashInfer plan 不进图；
> [g3](../specs/g3-graph-switch-merge.md) 做的是图开关合并与 `XINFER_DIAG_*` 清理。
> **"为什么 replay 不开"（数字与 eager 不一致的根因）没有被任何 spec 承接** —— 这是 P0-13 的真正内容。

### P0-14 策略层（L1）整体缺失

- `csrc/sm70_policy_fields.inc`（**67** 个 calculation/diagnostic 字段，顺序即 ABI；实测 `grep -c '^SM70_POLICY_FIELD('` = 67，Python 侧 `NATIVE_FIELDS` 元组也是 67 条——**此前记的 69 是把该文件行数当成了字段数**）
- `flash-attention-v100/include/flash_v100_policy.h`（**66** 字段 + `PreparedPolicy` + `Scope`）
- Python 侧：`vllm/config/sm70_native.py`(746) / `policy_defaults.py`(523) / `vllm/_sm70/runtime.py`（slot 绑定 + token 序列化）
- C++ 侧：`csrc/sm70_policy.h`(315) / `sm70_runtime_state.h` / `sm70_runtime.h` / `sm70_tile_runtime_signal.cuh`

上游把这层当成"**把每条性能结论固化成三件东西**"的载体（`sm70_1cat_rejected_paths.md` §7）：
①默认保守的策略开关 ②硬回滚 env ③进入编译缓存指纹的 calculation 字段。
xinfer 目前只有 D10 的两个 env 族，**没有对应的"能力门禁 + 硬回滚 + 指纹失效"三元组**。

---

## 4. P1 —— 其它未移植

| # | 项 | 上游证据 | 说明 |
|---|---|---|---|
| P1-1 | **HCX full-mesh 本地调度** | `ops/hcx_sm70.cu`（TP4 HC 一发融合 all-reduce+norm+down+up，M≤40）、`hcx_batch_sm70.cuh`（并发 verify 批次）、`hc_ll_shard_sm70.cu`（MTP verify 的 TP4 分片 HC LL）；上游总控文档**最新条目**（2026-10-09 "Full-mesh HCX local schedule"） | HC 是 xinfer 目标模型的层结构（`hyper_connection.rs`），上游的 HC 调度/通信尚未对齐 |
| P1-2 | **长 prefill 阈值调度** | `vllm/config/scheduler.py:82,89`（`long_prefill_token_threshold` + `_adaptive`）；79T README 建议 `max_num_batched_tokens=16384` + 阈值 8192 ⇒ 一步两个 Q8192 chunk | xinfer chunk 大小已是 8192，但缺"长 prefill 阈值"这一层调度 |
| P1-3 | **Marlin SM70 策略族** | `SM70_MARLIN_{DENSE,MOE}_{CTA_GEOMETRY,SPLIT_K,METADATA_CACHE}`（`sm70_policy_fields.inc:64-69`）+ `csrc/moe/marlin_moe_wna16/sm70_marlin_*.cu`(9 文件) | xinfer 有 `marlin/` 基础件但无 SM70 CTA 几何/split-K 策略 |
| P1-4 | **flash_qla/** | 上游 `flash_qla/`（GDN/linear-attn 的 QLA 内核树） | p1 只覆盖 `flash-attention-v100/kernel/h3/`，未覆盖 `flash_qla/` |
| P1-5 | **79T 的 paged KV gather workspace** | `csrc/attention/sm70_79t/shared_workspace.h`、`stable_rows.cuh` | 与 P0-2 同批才有意义 |
| P1-6 | **GDN prefill 线程数 128→256** | `docs/design/sm70_gdn_prefill_threads.md:1-45`（TileLang 工厂接受线程参数，仅索引/direct-state 子集合格；路由 trace 掉 ~63 ms） | **上游默认仍是 128**，256 是 opt-in（JSON `{"gdn":{"original_prefill_threads":256}}`）。且上游 GDN `warps=8` / `BK` 调度在本仓是 **B15 未准入**（op 级通过、全模型 token 等价失败）——**不要与 B15 混淆** |
| P1-7 | **QSA dense / prep SM70 内核** | `ops/qsa_dense_sm70.cu`（QSA 短上下文 decode split-K）、`ops/qsa_prep_sm70.cu`（decode/verify 的 RMSNorm+RoPE+cache 融合） | xinfer 的 `qsa_attention.rs` 有 QSA 结构，但**稀疏 mask 未真正生效**（见 §6 风险 R3） |
| P1-8 | **概率拒绝采样** | `vllm/v1/sample/rejection_sampler.py` | xinfer `speculative/verify.rs:17-67` 只做 greedy argmax 前缀匹配，无拒绝采样。与 P0-12 同源 |
| P1-9 | **上游十余种 proposer** | `vllm/v1/spec_decode/`：ddtree/dflash/dspark/eagle/medusa/ngram/qwen4_exp/suffix_decoding/gemma4/step3p5 | xinfer `src/speculative/` 只有 mtp/dflash/mamba/verify/metadata。**属功能面扩展**，非本轮性能目标 |

---

## 5. 已在队列 / 已否决 —— 不要重复

**已立项**（`docs/specs/`）：`m1` prefill 分解 · `p1` GDN prefill SM70 · `p2` FA prefill CUTLASS 2.x ·
`p4` prefill graph slab · `d1` NVFP4 QPN2 split-K · `d2` MTP verify · `g1` NCCL 进图（已采纳冻结）·
`g2` flashinfer plan handshake · `g3` 图开关合并 · `nvfp4-hmma884-port`。

**已否决/证伪/未准入**：`sm70_1cat_rejected_paths.md` §A（Flash-V100 D256 prefill 结构族 35 项）、
§B（GEMM/量化 16 项）、§C（MoE/投机/DFlash2 18 项）、§F（V100 结构性不可能 14 项）。
本件 §2/§3 列的项**均不在**上述否决集合内。

**唯一需要交叉确认的易混项**：
`§A21 BM32 Q-tile（BLOCK_M 16→32）` 被否决 ≠ `BM32 phase-reuse` 被接受（附录 B `:1032`）——两者不是同一实验。

**逐条易混对照表**（本清单的项 ↔ 清单里"看起来像但已被否"的项）：

| 本清单的项 | **不要**误当成这些已否决项 |
|---|---|
| P0-1 BM32 phase-reuse | A21 BM32 Q-tile（否决，-10.3%/-10.6%）、A8 8+8 四累加器管道（证伪） |
| P0-1 Q/P swizzle | A17 K/V global-load-first 重排序（-12.244% 但相对 swizzle 无增量）、A11 最小通信 shuffle 重打包（+1.4%） |
| P0-1 OUTPUT_STRIDE_268 | A20 里被否的 pitch 136→144 / p_strict 双 16x24 / aligned column-major(+4.157%) |
| P0-4b cross-block steady/drain | A13 被否的 PV B lookahead / double A+B / triple-B / runtime final-block 谓词版 |
| P0-4a Split-D N32 | A22 BN192/BN256 direct-KV（否决，"最大化 CTA 数不是最优"）、A2 八 warp BM64 staged-K/V（否决 +23.3%~+42.6%） |
| P0-4a gather / Split-KV3 | A23 FlashInfer paged-prefix split-KV 原型（否决，default-off，限 MAX_Q=2048）——**上游自己的 gather/SplitKV3 是另一套**（`prefill_candidates.py` / `dense_prefill.py`，默认开） |
| P0-4a b2 N32 顺序 | 无（这是上游的**质量硬要求**，不是可调项） |
| P0-8 TurboMind L2 派发缓存 | B2 强制 split/swizzle selector + 固定 `8x256x64` registry（否决，只省 0.1–0.2 ms）——**上游缓存是 per-shape 自适应，不是固定 registry** |
| P0-9/P0-10 skinny / QPN-M1 MoE | B11 AWQ MoE grouped/batched GEMM（**证伪**，严格相等失败 max_diff 2.441e-04）、B12 FP8 MoE compact（未准入 paused-unsafe） |
| P0-10 NVFP4 QPN2 路线 | C1 NVFP4 QPN2 **M32**（否决，140.94 vs ≈71.59 us）；**保留的是 QPN2 M16 NATIVE** |
| P0-12 采样 | C7 单遍 top-p kernel（证伪，改变 703 个 mask 元素）、C6 proposal temperature 0.85（否决） |
| P0-11 all-reduce | C17 里的 "TP4 push all-reduce"（否决，指 xinfer 侧那次尝试）——**上游 `custom_all_reduce.cu` 的 push 路径是已采纳件** |
| P0-5 混批 | 无（上游 `mixed_prefill.py` 是默认机制，本仓没有对应物，也没有对应否决记录） |
| P1-6 GDN 线程数 | B15 GDN `warps=8`/`BK` 调度（未准入：op 级通过但全模型 token 等价失败） |

---

## 6. 建议落地顺序（按 收益/风险，供排队列用）

| 序 | 动作 | 对应条目 | 收益 | 风险 | 理由 |
|---|---|---|---|---|---|
| 1 | **修正 p2 目标件**：拆成 p2a（79T 架构族，含 FP32 累加契约）+ p2b（Split-D N32 三件套 gather/N32/SplitKV3） | P0-2 / P0-2a / P0-4a | prefill 大头 | 中高 | 上游 `GQA_V37=False`，p2 现在抄的不是默认路线；上游默认是 79T **和** Split-D N32 两条并行 |
| 2 | **p2 增补 FA-V100 的 6 个 default-on 加速开关** | P0-1 | prefill 大头 | 中 | 上游已实测 -19~-20% / -12.3% / -8.67% / -5.11%，模板参数化，移植面清晰 |
| 3 | **混批 + prefill 图**（拆出独立 spec） | P0-5 | decode 大头 | 中高 | decode 的 9.2 ms/step 间隙根因；上游无 phase 分离 + 有 PIECEWISE prefill graph。**注意 p4/g3 都不覆盖这一项** |
| 4 | **GPU argmax + 采样家族** | P0-12 / P0-9 | decode 中头 | 低 | 贪心路径现在每步把 248320 维词表 D2H；`top1x` 连 logits 都不 gather。独立、好验证 |
| 5 | **d1 之后接 L2 派发缓存**（per-shape measure→reuse + 导入导出 + warmup 冻结 + L2 flush + fast selector） | P0-8 | decode + prefill 双线 | 中 | decode 是 26% 缺口的正解；d1 是手工特例，自适应派发才能覆盖多 shape |
| 6 | **decode 侧补 XQA 的 FP16-KV 子集** | P0-7 | decode | 中 | 上游 6443 行 vs 632 行的差额主体；先做与 FP16-KV 直接相关的子集 |
| 7 | **MoE decode 家族** | P0-10 | decode | 中高 | 1794 行量级；上游 default-on 的那条（`skinny`）是 MIT 来源，可直接抄 |
| 8 | **decode graph：查清"数字不一致"根因并打开 replay** | P0-13 | decode | 中 | 现在是"白付显存不拿收益"；已量到 ~6.6 ms/步 ⇒ ~46 tok/s。**g1/g2/g3 都不承接"根因定位"** |
| 9 | **all-reduce 家族补齐**（TP4 push / bit-exact `exact_row_reduce` / all-reduce+norm 融合） | P0-11 | decode + TP | 中 | xinfer 目前只有 TP2 one-shot |
| 10 | **策略层 L1**（67+66 字段 + typed policy + 编译指纹） | P0-14 | 间接（可维护性 + 防回归） | 中 | 是"把结论固化成开关+回滚+指纹"的载体，是后续所有加速项的落地面 |
| 11 | **partial prefix cache / 均衡 chunk** | P0-6 / P0-5 | prefill 中 | 低 | 迁移成本低，收益中等 |
| 12 | **P0-4c block-FP8 prefill 投影走 CUTLASS 2.x** | P0-4c | prefill | 中 | 需要新开 `-DUSE_CUTLASS_SM70` 通道（与 `CARGO_FEATURE_CUTLASS` 解耦）——与 p2 的开通道工作可合并 |

---

## 7. 顺带发现的**正确性风险**（不是性能项，但优先级更高）

这些是盘点过程中在 xinfer 侧发现的**沉默失败路径**，与上游无关，但按 D6「先精度后性能」必须先处理：

| # | 风险 | 证据 | 后果 |
|---|---|---|---|
| R1 | **`nvfp4_gemm_cutlass.cu` 在 SM70 是空函数体**：`#else // !ENABLE_FP4` 给出**空的** `nvfp4_cutlass_gemm_f16/bf16`（`ENABLE_FP4` 仅 `kernels/build.rs:364-368` 的 `compute_cap>=100` 定义）；调用方 `attention.rs/src/nvfp4_linear.rs:746/760` **不检查返回值** | `nvfp4_gemm_cutlass.cu:699-717` | SM70 上**静默返回未初始化输出** |
| R2 | **`fp8_grouped_gemm_fused` 返回 -1 但调用方不检查，且该路径无 sm 门控** | `fp8_gemm_cutlass.cu:1155-1157`；`attention.rs/src/fp8_linear.rs:1533` | 与 R1 同类。`fp8_matmul_f16_cutlass` 那条有 `fp8_linear.rs:956-959` 的 `bail!("fp8_matmul_cutlass requires sm90+")` 兜住，这条**没有** |
| R3 | **Qwen4 QSA 稀疏 mask 未生效**：indexer 没有自己的 raw-key cache，decode 时只在当前 forward 内打分；`qsa_attention.rs:144` 的 TODO 明写"当前跑的是全注意力" | `src/models/layers/qwen4/qsa_attention.rs:122, 144` | 跑的**不是** QSA 路径，性能与精度都偏 |
| R4 | **decode CUDA graph 白付显存**：捕获默认开（`core/engine.rs:274-295`），replay 却 opt-in（`core/runner.rs:1294-1296`） | 同 P0-13 | 吃掉 capture pool（`graph.rs:404-540`）+ graph workspace（`kvcache_allocator.rs:798-806, 962`）**却零收益** |
| R5 | **死代码占编译与维护成本**：`moe_gemm.cu` 的 `moe_gemm`（`ffi.rs:498`）、`fast_topk.cu` 的 `fast_topk_select`（`ffi.rs:5027`）、`fused_sigmoid_topk`（`ffi.rs:809`）均无 Rust 调用者；`prefill_paged_attn.cu` / `_opt.cu` 在 cuda 构建不可达 | `attention.rs/src/lib.rs:1244, 1847`；`moe.rs:453` vs `:545` | 读者会误以为这些是活跃路径 |
| R6 | **编译期陷阱（会让性能测量是假的）**：`build.rs` 的 `rerun-if-changed` 手工维护，116 个源文件只登记 91 个；未登记的 25 个（`src/attention/*`、`src/marlin/*`、`src/moe/*`、`paged_attention_v*.cu` 等）**改了不会触发重编** | `attention.rs/src/kernels/build.rs` | 改了这 25 个文件里的任何一个，测出来的性能是**假的** |
| R7 | **上游 commit 不可核验**：`/Volumes/SMB重要/1Cat-vLLM` 上 `git log` 报 "not a git repository"（`.git` 存在，但 git 在 `/Volumes` 文件系统边界停止） | 本次盘点实测 | 本件所有上游行号只对"当前磁盘快照"成立；引用上游必须带 commit（D2），而 commit 只能靠 AGENTS.md 钉 |

---

## 8. 复核方法（本件结论可重跑）

```bash
UP="/Volumes/SMB重要/1Cat-vLLM"
XR="/Users/xx/xxinfer-dev/repos/xxinfer"
AR="/Users/xx/xxinfer-dev/repos/attention.rs"

# 0) 先确认内核源码与实际编译产物是否同一份（本件顶部警告）
grep -n "attention-rs" "$XR/Cargo.toml"                        # => rev = "144a07f"
git -C "$AR" log --oneline -1                                  # => 本地 HEAD（若 ≠ 144a07f 则行号只对本地快照成立）
git -C "$AR" status --porcelain                                # => 若非空，行号同样只对工作树成立

# 1) prefill 加速项：xinfer 侧 0 命中
wc -l "$AR/src/kernels/src/flash/flash_prefill_paged.cuh" \
      "$UP/flash-attention-v100/kernel/fused_mha_forward_paged.cu"
grep -c -E "BM32|PHASE|swizzle|stride_268|PAIR_SCRATCH|LOW_SMEM|contig_fast" \
     "$AR/src/kernels/src/flash/flash_prefill_paged.cuh"        # => 0

# 2) decode 加速项：xinfer 侧 0 命中
wc -l "$AR/src/kernels/src/flash/flash_decode_paged.cuh" \
      "$UP/flash-attention-v100/kernel/flash_decode_paged.cu"
grep -c -E "wave_partitions|merged_wave|batch_context_routing|block784|qk_pipeline|sawtooth|padded_smem|split_reduce" \
     "$AR/src/kernels/src/flash/flash_decode_paged.cuh"          # => 0

# 3) 79T / Split-D 在 xinfer 侧完全不存在
grep -rn -E "sm70_79t|d256_gqa|splitd_n32|SPLITKV3|gather_paged_kv|sm70_v37" "$XR/src" "$AR/src"   # => 无命中
#    注意：不要用宽模式 "splitd" —— `src/utils/gguf_helper.rs` 的 `SplitDelimiterBehavior` 会假命中

# 4) 上游策略字段总数
grep -c "^SM70_POLICY_FIELD(" "$UP/csrc/sm70_policy_fields.inc"   # => 67（不是 69；69 是文件行数）
grep -c "VLLM_FLASH_V100_\|VLLM_XQA_" "$UP/flash-attention-v100/include/flash_v100_policy.h"  # => 66

# 5) 上游 SM70 专属 ops 清单
ls "$UP/csrc/sm70_turbomind/ops" | wc -l                     # => 77

# 6) 上游 prefill 两条默认路线的开关默认值
sed -n '765,780p' "$UP/vllm/envs.py"    # FA2_D256_PREFILL / GATHER_DENSE / SPLITKV3 / ARCH_128K_EXPERIMENTAL / GQA_V37

# 7) quickreduce 确与 SM70 无关（澄清 P0-11 的修正）
grep -rn "hip_runtime" "$UP/csrc/quickreduce/"                # => 命中 ⇒ ROCm/HIP，非 SM70

# 8) 上游 commit 不可核验（R7）
git -C "$UP" log --oneline -1                                  # => fatal: not a git repository
```
