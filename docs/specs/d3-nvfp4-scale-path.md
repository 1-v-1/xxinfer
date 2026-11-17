# 规格 d3-nvfp4-scale-path —— 已完成

## 1. 现象与已知事实

检查点是 `compressed-tensors` / `mixed-precision`：`F8_E4M3` 码 + **逐通道**标度 `[N,1]`
（`group_0` 覆盖 `linear_attn.*`、`self_attn.*`，`strategy="channel"`、`group_size:None`）。
三段投影反量化后本应同量级（qkv≈1.2e-1、z≈6.8e-2、out_proj≈3.9e-1），但引擎侧实测增益
qkv≈1.25、z≈1.7e-3、**out_proj≈4.4e3** ⇒ 残差流幅值爆炸，输出塌成 token 0。

SM70 上 `compute capability 7 (<8), switched to F16` ⇒ 模型 dtype 为 F16。

## 2. 待验证假设（按 预期收益 × 验证成本 排序）

| # | 假设 | 结果 |
|---|---|---|
| H1 | 标度没乘 | 否 |
| H2 | 逐通道被当全局 | 否（`meta/sm` 显示 `(1,n)` 形态正确）|
| H3 | TP 切分后行偏移错位 | 否 |
| H4 | **QPN8 逐通道重排不等价于朴素反量化 matmul** | **是** |

## 3. 验收条

| case | 条件 | 期望 |
|---|---|---|
| `d3#1` | 先 `fp8_qpn8_repack` 再 `fp8_matmul`，真实形状 `N=3072,K=5120` | 打包路与 gold 位等 |
| `d3#2` | 正确路径成为默认（D10 走 `src/utils/env.rs`）| 不需记得关任何东西即正确 |
| `d3#3` | 逐阶段幅值 | out_proj 输出≈输入量级、逐层不再单调到 1.2e4、输出不再全 token 0 |

## 4. 覆盖区间与回退路径

覆盖 `src/models/layers/linear.rs` 的 fp8 反量化门控与 `attention.rs` 的 QPN8 重排内核。
回退：`XINFER_QPN8_REPACK=1` 显式开启重排路。

## 5. 结论

**真因＝H4。** 位级核对：`permuted<<7` 把 e4m3 直接落到 half 指数字段而**没加 `(15-7)<<10` 偏置**
⇒ 解出的是真值/256，所以 `channel_scales ×256` 是**对的补偿**，"标度量级错"这一族理论到此终结。

修复：`src/utils/env.rs` 新增 `QPN8_REPACK_ENV`/`qpn8_repack_enabled()`，`linear.rs` 门控从
"默认开、要人记得关"改为"默认关、要人显式开"。`linear.rs` 行数 1955 不变，`env.rs` 229→238。

## 6. `d3#1` 位等同比对实测

真实形状、真实量级下（先调 `fp8_qpn8_repack` 以命中打包分支）：

| 记录 | 值 |
|---|---|
| `rp.scale` maxabs | 6.7932e-02 |
| `rp.codes`（原始字节）| 1.9200e+02 |
| `real.gold` | 2.1470e-04 |
| `real.packed` | **inf/nan**（与 `block_size`、`input_is_packed` 取值无关）|

> 之前 §70/§71 的"打包==gold"无效：探针传的是 `F32 (n,1)`，两路都走了 fallback。

## 7. `d3#3` 复测结果

`DNAN=0` 下 greedy：`"The user asks for the capital of France in one word. The answer is Paris"` ✓。
`gdn0 out_proj` maxabs=1.495e1（≈输入量级，不再 ×4400）、`attn_out` 1.495e1、`mlp_in` 1.676e0；
逐层不再单调到 1.2e4；`XINFER_GDN_PREFILL=auto` vs `exact` **位相同** ⇒ GDN 递推无罪，36× 只在 prefill。

## 8. 与参考实现的分歧

参考实现 `/Volumes/SMB重要/1Cat-vLLM`(bcd581d230) `qpn/fp8.py:process_weights_after_loading`
把标度强制为 float32 且要求 orig_dtype float16 —— 即"标度保持 F32、不重排"。本仓默认已与之对齐。

## 9. `d3#4` MTP A/B 表（`DNAN=0 MRES=1024 MFRAC=0.999 K=3 NTOK=256`，TP=2，单请求）

| ctx | mtp | 解码量 | 耗时 | 吞吐 | 每步 |
|---|---|---|---|---|---|
| 24 | 0 | 111 tok | 2.52 s | **44.01 t/s** | 22.7 ms |
| 24 | 1 | 8 tok | 0.74 s | 10.80 t/s | — |
| 768 | 0 | 256 tok | 5.91 s | **43.29 t/s** | 23.1 ms |
| 768 | 1 | 256 tok | 20.31 s | **12.61 t/s** | ≈132 ms |

`accepted 非恒 0` 成立：`K=3` 跑满 256、`K=1` 只出 113（提前命中 EOS）。

## 10. 为什么比值是负的（结构性）

`MTP verify m=4 graph=false in 92.81 ms` ≈ 4 × 23 ms —— eager 无批摊销，每行与普通解码一步同价。
图本可提供摊销（4 行 ≈ 1 行 ⇒ ≈5.8 ms/tok ≈ 73 t/s），但本机装不下：`SEQS=1`、`MTP=0`、`OK=1`
仍 `CUDA_ERROR_OUT_OF_MEMORY`。根因见 §84：`runner.rs:649` 用 `planned_graph_capture_batches`
为 1..=容量(=10) 每个精确批各捕一张，而 `sparse_graph_capture_batches` 从未被 runner 调用。
