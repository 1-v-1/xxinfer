# 规格 d1-nvfp4-qpn2-splitk —— 待开工

> 状态：`待开工`（前置 = 工作流落地）。收益 +5 tok/s，风险低。

## 1. 现象与已知事实
- decode eager **38.74 tok/s**（26 ms/step）；DRAM 下限 13.7 ms/step（≈73 tok/s）；1Cat 4 卡 103.87 tok/s ⇒ 2 卡折半 ≈52。
- 现有 SM70 GEMV（`attention.rs` `ba78b00`）就是同一条路的上一版 ⇒ **可在现有基准上加变体，不需要新骨架**。

## 2. 待验证假设
- A1 抄 1Cat `fp8_qpn8_split_cta_m1` 的 CTA SplitK，把 468 → **818 GB/s**。TurboMind QPN2 布局 `[tile=N/32][group=K/16][lane=32][8B]`，Volta `mma.m8n8k4`，scales 用 `__half`。
- A2 额外 reduce kernel ≈0.25 ms/step；Phase 1 在 prefill 把 QPN2→u32 转成每层 22 MB scratch（+3.5 ms）——这两项会吃掉多少收益，须实测。
- A3 新文件 `nvfp4_qpn2_sm70.cu` + `bench_qpn2.cu`；Rust 侧 F16 在 GPU 上是裸 `u16`。

## 3. 验收条
- `d1#1` L0 基准数值对齐（容差按 D6 声明在比对脚本里）。
- `d1#2` 端到端 greedy（温度 0）**逐 token** 与 eager 基准完全一致。
- `d1#3` A/B 数值表 + wins 计数 + 测量条件（D12）。
- `d1#4` 新 `.cu` 已在 `build.rs` 加 `cargo:rerun-if-changed`（否则测出来的是假性能）。

## 4. 覆盖区间与回退路径
只覆盖 decode 的 M=1 路径；`m>1` 走原路。回退路径 = 现有 `ba78b00` GEMV。

## 5. 结论
未开始。
