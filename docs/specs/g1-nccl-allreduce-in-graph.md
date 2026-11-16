# 规格 g1-nccl-allreduce-in-graph —— 已采纳（冻结）

> 状态：`已采纳`。判完生死即冻结，后续工作另开一份。

## 1. 现象与已知事实
- 图里 **0 个 AllReduce 节点** ⇒ 重放时没有 TP 归约 ⇒ logits 乱码。
- 修复实测：`XINFER_NCCL_GROUP=0`（捕获时不开 NCCL group）⇒ 图节点 **2017→2273**、`EVENT_RECORD` **1→129**、出现 `ncclDevKernel_AllReduce_Sum_f` ⇒ **128 次 all-reduce 进图**。

## 2. 待验证假设（已验）
- A1 捕获期不开 NCCL group，all-reduce 即以普通 kernel 节点进图（与 D15 一致：1Cat 删的是 event bridge，不是 collective kernel）。

## 3. 验收条
- `g1#1` 图节点计数表（2017→2273、`EVENT_RECORD` 1→129、出现 `ncclDevKernel_AllReduce_Sum_f`）。
- `g1#2` TP2 下 greedy 逐 token 与单卡基准完全一致。

## 4. 覆盖区间与回退路径
覆盖 TP2 的 decode 图路径。回退路径 = `--disable-cuda-graph`（与 g3 的开关合并待办）。

## 5. 结论
`已采纳`。证据即 §1 的节点计数；剩余缺口不在本件（见 g2、g3）。
