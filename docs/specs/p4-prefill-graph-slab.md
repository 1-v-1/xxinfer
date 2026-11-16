# 规格 p4-prefill-graph-slab —— 待开工

> 状态：`待开工`（前置 = g1 + g2）。收益只有几个百分点，风险中 ⇒ 排在最后。

## 1. 现象与已知事实
- prefill chunk 形状固定（2048）⇒ 形状可捕获 ⇒ 可做图。但 prefill 是 compute-bound，launch 间隙占比小 ⇒ 收益只有几个百分点。
- 捕获期每一次分配走 `cuMemAllocAsync` ⇒ 图里出现 MEM_ALLOC 节点（实测 **1766** 个）⇒ 每次 replay 重分配。

## 2. 待验证假设
- A1 按 [D14](../DECISIONS.md) 方案 A′：`cudarc` 实现捕获期 slab（≈200 行，`RELAXED` capture mode + 普通 `cuMemAlloc`，free 只记账 + 归还 freelist）；`candle` 零改动只 bump rev；`xinfer` ≈120 行在捕获前后开关 slab。
- A2 捕获前须 ≥1 次 NONE 模式 dummy run + `cudaDeviceSynchronize()`，使捕获期分配序列与 replay 期完全一致。

## 3. 验收条
- `p4#1` 用 `cuGraphGetNodes` + `cuGraphNodeGetType` 数出 `CU_GRAPH_NODE_TYPE_MEM_ALLOC` **== 0**（nsys 看不到，这是唯一可靠手段）。
- `p4#2` prefill tok/s 的 A/B 数值表 + wins + 测量条件。
- `p4#3` `AUTO_FREE_ON_LAUNCH` 不再是正确性依赖（D14）。

## 4. 覆盖区间与回退路径
覆盖 prefill 图；decode 图由 g1/g2 覆盖。回退路径 = eager prefill。

## 5. 结论
未开始。
