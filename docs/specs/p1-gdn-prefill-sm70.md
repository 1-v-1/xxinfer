# 规格 p1-gdn-prefill-sm70 —— 待开工

> 状态：`待开工`（前置 = 工作流落地 + m1）。影响 **48/64 层**，影响面最大，风险中。

## 1. 现象与已知事实
- prefill 时 GDN 走的是 `gdn.cu`（2841 行）里的**顺序 recurrence**（`gated_delta_rule_recurrence_kernel_tiled` / `_fallback`）；该文件里**没有任何 chunked 实现**（grep `chunk` 无结果）。
- 快的 chunked 路径 `gdn_flashinfer_prefill.cu`（275 行）调的是 FlashInfer 上游 `flat::launch_delta_rule_prefill_kernel_gbai<..., cutlass::arch::Sm90, ...>`，且带 `major != 9` 运行时硬检查 ⇒ **改 arch tag 不够，要动 FlashInfer**。

## 2. 待验证假设
- **A1（推荐）P1a**：抄 1Cat `flash-attention-v100/kernel/h3/`（`fmha.h` 39 KB + `default_fmha.h` + `prefetch_mma.h` + `forward.cu`，共 ~80 KB），**独立于 FlashInfer**，落在我们的 `attention.rs` 里。但它原是 **D128 + MHA（"no GQA head packing"）⇒ 要改成 D256 + GQA（TP2 下每 rank `num_qo_heads=12`/`num_kv_heads=2`/GQA group=6/`page_size=64`/chunk=2048）**。算法结构（chunked 并行 scan）可复用，CUTLASS 2.x 的 QK/PV GEMM 用我们自己的 SM70 `m8n8k4` 替换。
- **A2 P1b**：抄 1Cat `flashinfer-sm70/csrc/h3_noncausal_sm70.cu`（在 FlashInfer 内加 SM70 分支）⇒ 要 fork FlashInfer，成本更高。
- 抄 h3 时照抄的要点：QK tile 64×64 或 64×128、PV tile 64×128；`prefetch_mma.h` 接受**第一个 softmax 之前加载的 V fragment**，保持 CUTLASS 的软件流水与累加顺序；自动 dispatch（Q/K 长度都 ≥1024 用 128 keys，否则 64）；`246 registers/thread without spills`、`34,304 shared bytes/block`（K128 变体 255 regs，K64 用 210）；**无额外常驻 device 分配**（16 GB 卡友好）；wrapper 支持 graph replay。
- ⚠ 别把它的坑一起抄进来：*"Softmax row distribution uses the actual warp count, rather than inferring eight warps from the tile area; the latter leaves rows 32..63 unnormalized and caused the old length-63 NaNs."*

## 3. 验收条
- `p1#1` L0 基准数值对齐（容差写在比对脚本里）。
- `p1#2` 端到端 greedy 逐 token 与 eager 基准完全一致。
- `p1#3` 48 层 GDN prefill 占比下降的 A/B 表 + wins + 测量条件。
- `p1#4` GDN 循环状态池**恒 F32** 未被改动（D15）。

## 4. 覆盖区间与回退路径
覆盖 48 个 GDN 层的 **prefill**；decode 与 MTP verify 不变。回退路径 = 现有顺序 recurrence。

## 5. 结论
未开始。
