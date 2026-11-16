# 规格 p2-fa-prefill-cutlass2 —— 待开工

> 状态：`待开工`（前置 = 工作流落地 + m1 + p1）。影响 16 个 full-attn 层，风险高。

## 1. 现象与已知事实
- 16 条 full-attn 层实测输出：`flashinfer disabled for this layer: unsupported gqa/head_dim combination, falling back group_size=6 head_size=256 is_prefill=true prefill_sm_supported=false`。
- `attention.rs/src/kernels/build.rs:292-321`：`let cutlass_reachable = compute_cap >= 90;` ⇒ **SM70 上 CUTLASS 被整个关掉**，走不到 2.x 路径。
- 我们已有的底子：`src/kernels/src/flash/flash_prefill_paged.cuh`（864 行，native flash prefill，SM70 走 `m8n8k4`）。

## 2. 待验证假设
- A1 抄 1Cat `csrc/attention/sm70_v37/`：**形状逐字匹配我们**（"six query heads, one KV head and head dimension 256"）。prefix 用 **M128/N256/K32 SM70 TC GEMM**，QK epilogue 做 FP32 logits → tile-local 概率 → FP32 max/sum → 舍入 FP16，PV 用 FP32 累加。
- A2 需新开一条 `-DUSE_CUTLASS_SM70` 通道，且 `with_cutlass` 要与 `CARGO_FEATURE_CUTLASS` 解耦（否则被迫编 3.x 的 SM90 kernel）。CUTLASS **2.x 在 SM70 可用**（Volta `m8n8k4`），3.x 不行（要 TMA / sm90+）。
- A3 利好：cudaforge 已把 CUTLASS v4.x 拉到 `/root/.cudaforge/git/checkouts/cutlass-b46b16d*/include`，而 **v4.x 仍保留 2.x 的 legacy API**（`cutlass/gemm/kernel/default_gemm.h`）⇒ 1Cat 那套 API 大概率能直接用。
- 对标数字：1Cat 实测 D256 GQA @KV128K = **62 TFLOPS（Q8000）/ 68（Q8192）/ 47（旧 Split-D 路径）**。

## 3. 验收条
- `p2#1` L0 基准数值对齐（容差声明在比对脚本里）。
- `p2#2` 达到 62~68 TFLOPS 区间的对标表（含测量条件）。
- `p2#3` 端到端 greedy 逐 token 与 eager 基准完全一致。
- `p2#4` 新 `.cu`/`.cuh` 已登记 `build.rs` 的 `rerun-if-changed`。

## 4. 覆盖区间与回退路径
覆盖 16 个 full-attn 层的 prefill。回退路径 = 现有 native flash（864 行）。

## 5. 结论
未开始。
