# 规格 g2-flashinfer-plan-handshake —— 进行中

> 状态：`进行中`（已定位，未解）。收益 ≈+8 tok/s，风险高。

## 1. 现象与已知事实
- FlashInfer decode plan 的**跨 rank handshake** ⇒ 进图必 hang（双卡 SM 100%）。
- 绕行 `XINFER_SKIP_FLASHINFER=1` **只用于诊断，不是产品路径**：FlashInfer BatchDecode 只占 **0.34 ms/step**；eager+flashinfer 35.55 vs eager+skip 36.10（差 1.5%）。
- 目标：28.2 ms/step（kernel 19.2 + 间隙 9.2）→ ~21.5 ms/step（≈46 tok/s）。

## 2. 待验证假设
- A1（先做，便宜）把 plan 移出图 —— 1Cat 就是把 attention/KV 排除在图外（D15）。
- A2 用 FlashInfer 自己的图模式 —— 但 KV 长度逐 token 变 ⇒ partition 失配 ⇒ 乱码。

## 3. 验收条
- `g2#1` TP2 下图 replay 不 hang（SM 占用率证据）。
- `g2#2` greedy 逐 token 与 eager 基准完全一致。
- `g2#3` 28.2 → ~21.5 ms/step 的 A/B 数值表 + wins + 测量条件。

## 4. 覆盖区间与回退路径
覆盖 decode 图 + TP2。回退路径 = eager decode（`--disable-cuda-graph`）。

## 5. 结论
未解。当前只有绕行证据（0.34 ms/step、35.55 vs 36.10），不构成产品路径。
