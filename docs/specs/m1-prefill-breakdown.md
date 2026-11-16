# 规格 m1-prefill-breakdown —— 待开工

> 状态：`待开工`（前置 = 工作流落地）。前置齐了才动 P1/P2。

## 1. 现象与已知事实
- prefill 只有 200+ tok/s，但当前只有 22 token 的样本（0.21 s / 106 tok/s），**完全看不出瓶颈在哪**。
- 64 层里 **48 层是 GDN**、16 条 full-attn ⇒ GDN 那条影响面最大。
- 三条 prefill 快路径全废（三个 `[slow]`：FlashInfer prefill 要 SM80+、cutlass feature 缺、chunked GDN 要 SM90）。

## 2. 待验证假设
- A1（最便宜，先做）：用 L0 独立 nvcc 基准逐个 kernel 打时间 + 端到端跑长 prompt（≥4096 token）对总时间，拆出 **attention / GDN / GEMM / cast / 其他** 五项占比 ⇒ 决定 prefill 先投 GDN 还是 FA。

## 3. 验收条
- `m1#1` 长 prompt（≥4096）分段耗时表（五项，含 shape/batch/TP/是否 graph/时钟）。
- `m1#2` 表落进本件 §5，并给出"先投哪条"的判定。

## 4. 覆盖区间与回退路径
纯测量，不改代码。测不出区分假设的数 ⇒ 回到 M2（decode 分解复测）而不是硬推。

## 5. 结论
未开始。
