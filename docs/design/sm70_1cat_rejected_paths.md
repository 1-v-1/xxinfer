# 1Cat-vLLM：V100/SM70 已否决 / 已证伪优化方案清单 + 运行时策略产品化

仓库：`/Volumes/SMB重要/1Cat-vLLM`，commit `bcd581d230`
主控文档：`docs/design/sm70_v100_migration_control.md`（49,698 行）
本报告所有行号均指向工作区当前文件（最新版）。

---

## 0. 阅读指引

- **否决**= 该方案已被实测证伪并从源码移除/标为 `rejected`，文档明确写 "do not repeat"。
- **证伪**= 机理上被证明不成立（硬件特性导致不可能有效），同样不再重试。
- **未准入**= 数值/质量不达标或未走完门禁，不等于"机理被证伪"，本文分开标注。
- 硬件约束关键词：Volta/SM70 **无 `cp.async`/`ldmatrix`/`wgmma`/warp-specialization**，寄存器 255/thread 上限，shared memory 96KB/SM，**2 CTA/SM** 是绝大多数实验的硬门禁，**64 regs/thread** 是 BM32 路径的硬门禁。

---

# 第一部分：已否决 / 已证伪方案清单

## A. FlashAttention / V100 D256 Prefill —— 结构级方案否决

### A1. FlashAttention-2 风格 M64 CTA（512 线程）—— 全面否决
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:901-940`
- **方案**：FA2 式 M64/BN128 complete-phase micro，16 warp 全用于 QK，每 PV warp 保 4 个 M16 输出 panel，V fragment 跨 4 panel 复用。
- **硬门禁通过**：M64 = 128 regs/thread、83,472 B 动态 shared、0 spill、**1 CTA/SM**；BM32 = 64 regs、41,744 B、**2 CTA/SM**。输出 packed-uint32 全 bitwise 相等。
- **实测回退**（locked 1200 MHz，100 交替 A/B）：
  | KV blocks | 2×BM32 | M64 | speedup | M64 wins |
  |---|---|---|---|---|
  | 1 (random) | 55.808 us | 55.936 us | **-0.229%** | 30/100 |
  | 2 (random) | 85.760 us | 87.296 us | **-1.791%** | 2/100 |
  | 4 (random) | 146.944 us | 149.760 us | **-1.916%** | 0/100 |
  | 64 (random) | 1,976.576 us | 2,016.896 us | **-2.040%** | 0/100 |
- **推理依据**：NCU 显示 M64 把 L1TEX global-load sector 从 56.36M 降到 47.16M (-16.3%)、barrier stall 从 24.82% 降到 8.12%，tensor 指令完全相同 75.50M。但 **active warps/scheduler 7.98 → 4.00，eligible warps 0.905 → 0.514，long-scoreboard stall 14.61% → 24.78%**，NCU duration 1.828 → 1.874 ms (+2.49%)。V100 的 96KB shared 让 M64 只能占 1 个 CTA/SM，失去一半 latency-hiding 池，V 复用省下的不足以抵偿。
- **结论**：不得只改 shared layout 拼写重试 512-thread/1-CTA 的 M64 形式。

### A2. 八 warp BM64 staged-K/V shared stage —— 否决（+23% ~ +43%）
- **文件**：`docs/design/sm70_flashinfer_native_backend.md:224-285`
- **方案**：8 warp / 4 个 M16 query-owner 对 / D128 per warp / 8KB 可复用 K/V stage / 16-GPR 有界 V payload。
- **可行性全过**：128 regs/thread、43,776 B shared、2 CTA/SM、0 stack/spill/`LDL`/`STL`、packed-FP16 bitwise 相等。
- **实测回退**（vs two-BM32 `ALL_P + PAIR_SCRATCH`）：
  | input | KV blocks | two-BM32 | 8-warp BM64 | delta | wins |
  |---|---|---|---|---|---|
  | random | 1 | 56.576 us | 69.760 us | **+23.30%** | 0/100 |
  | alternating | 1 | 56.832 us | 70.144 us | **+23.42%** | 0/100 |
  | random | 2 | 86.144 us | 116.480 us | **+35.22%** | 0/100 |
  | alternating | 2 | 86.656 us | 117.120 us | **+35.16%** | 0/100 |
  | random | 4 | 147.712 us | 210.560 us | **+42.55%** | 0/100 |
  | alternating | 4 | 147.522 us | 210.432 us | **+42.64%** | 0/100 |
  全网格复核：144 groups 时 1/2/4-block 回退 11.3% / 19.7% / 22.9%；288 groups 时 11.9% / 19.3% / 23.6%。回退随 KV block 数增长 → 稳态结构性损失，不是 launch noise。
- **推理依据**：SASS 保留 768 HMMA，LDG 74→21、STS 70→54，但 LDS 160→200、BAR 6→**12**，**resident warps 32→16**。省下的 global operand load 不足以偿还 shared staging + 额外同步 + 一半的延迟隐藏池。
- **结论**：8-warp BM64 shared-stage 结构整体关闭（含 barrier 拼写与 stage layout 变体）。

### A3. 十二 warp direct-K/V owner-local 形式 —— 编译期否决（spill）
- **文件**：`docs/design/sm70_flashinfer_native_backend.md:270-275`
- `__launch_bounds__(384,2)` 迫使 PTXAS 用 80 regs，产出 **816 B stack + 1,512 B spill stores + 1,992 B spill loads per thread**，SASS 含 453 `LDL` / 237 `STL`。无法满足 2-CTA 资源包络。
- 由此 **4 / 8 / 12 / 16-warp 的整个 M64 family 在当前算术契约下全部关闭**。

### A4. FlashInfer "Closed Paths" 汇总（8 条不得在新 backend 名下重试）
- **文件**：`docs/design/sm70_flashinfer_native_backend.md:209-220`
  1. 512-thread M64 + 83 KB shared：barrier 少但 eligible warps 低，long shape **-2%**。
  2. 通用 8-producer/8-consumer shared K/V staging。
  3. 16-warp paired-QK staging：在 64-register 边界 spill。
  4. 两个 D128 CTA 重复 QK 和 softmax。
  5. probability-only pitch / group-rotation 调优。
  6. source-only prefetch fence（不改 SASS）。
  7. raw HMMA.884 分解（把一个原生 WMMA 换成更长的串行指令链）。
  8. 直接改 capability gate 让 SM75+ FlashInfer kernel 跑在 SM70。

### A5. 直接 global Q WMMA 加载（不做 shared staging）—— 否决 +28.09%
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:113`；主控文档 `docs/design/sm70_v100_migration_control.md:42271`
- **方案**：Q 片段不走 shared，直接从 global 载入 WMMA。
- **实测**：`+28.09%` 回退，约 **1.99B** global sector。**理由**：global 读放大远超 shared staging 的成本。

### A6. 8 producer / 8 consumer QK shared double buffer —— 否决 +33.56%
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:97`；`docs/design/sm70_tp4_nomtp_long_context_decode.md:194-200`
- **实测**：`40.221 → 53.720 ms`，**+33.56%**。**理由**：barrier 与 global→shared 往返主导。
- 同族 PV double buffer：`40.212 → 42.750 ms`，**+6.31%**（`:98`）；降低 PV staging pitch 仍 **-3.60%** 回退。

### A7. True 8+8 named-barrier 调度 —— 否决 +4.84%
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:112`
- **实测**：`+4.84%`。**理由**：barrier 占用 **9.31%**、tensor active 仅 **8.89%** —— 用 barrier 换来的"无 stage barrier"并没有减少总同步成本。
- **V100 硬件依据**：Volta 无 `bar.sync` 变体支持 warp-specialization 式的 producer/consumer 分离，named barrier 数量受限于硬件 barrier slot，且 512-thread CTA 下 `bar.sync 0,512` 全 CTA 同步代价高。

### A8. 8+8 producer/consumer 完整 BM32 管道（4 个常驻 PV 累加器）—— 寄存器门禁否决
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:307-348`
- **实测/资源**：四种 build 全部否决：
  | build | regs/thread | stack/thread | shared/CTA | 决策 |
  |---|---|---|---|---|
  | natural, one-CTA launch bound | 128 | 8 B | 45,456 B | reject；只能 1 CTA 常驻 |
  | `__launch_bounds__(512,2)` | 64 | 480 B | 45,456 B | reject；**299 条静态 `LDL/STL`** |
  | inline `bar.sync 0,512` 诊断 | 64 | 480 B | 45,456 B | reject；改 barrier 拼写不消 spill |
  | shared-epoch producer/consumer | 64 | 96 B | 45,600 B | reject；零 barrier 协议仍 spill PV 状态 |
- **推理依据（V100 核心约束）**：硬预算 `65,536 / (512 threads × 2 CTAs) = 64 regs/thread`。4 个常驻 PV 累加器本身已占 32 个 FP32 寄存器，PV 更新峰值还需 matrix-A/B 片段、HMMA rename 寄存器、2 个 P 地址、64-bit V 寻址、row scale 与循环状态。**QK 角色的空闲寄存器无法分配给 PV 线程，因为寄存器分配对整个 kernel 是均匀的。**

### A9. D128 双 CTA 拆分（每 CTA 算 M32 但只算一半 D128 输出）—— 否决 +21.82%
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:342-348`
- **实测**：达到 64 regs、0 spill、37,408 B shared、2 CTA/SM、bitwise 相等，但 `134.912 → 164.352 us`，**+21.82%**，0/100 wins。
- **推理依据**：两个 D-half CTA **重复执行 QK 和 softmax**，重复成本 > V-load 复用节省。

### A10. Raw-HMMA（`m8n8k4`）分解 —— 否决 +18.2% / +27.5%
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:120-122, 259-271`；`docs/design/sm70_hmma_pipeline_search.md:20-35`
- **方案**：把原生 `m16n16k16` 拆成 raw `m8n8k4`。
- **实测**：
  - raw-HMMA QK（最佳 B-reuse 形状）：`212.352 → 251.008 us`，**+18.204%**，0/100 wins。
  - raw-HMMA PV：`34.432 → 43.904 us`，**+27.509%**，0/100 wins。
  - P0 布局求解器穷举 SM70 TN/TT/NT/NN 全布局 + quadpair 8! lane 置换：**不存在 zero-transfer 映射**，最佳映射仍需 32/64 个 FP16 值跨 lane；transposed-P 更差。
- **推理依据（V100 核心约束）**：Volta 原生 HMMA.884 就是 `m16n16k16`。raw `m8n8k4` 把一个原生 M16 操作拆成**额外的串行 HMMA 链**，且多数形状**重读 B**。更多寄存器调度无法消除这一结构性成本。

### A11. P1 shuffle 最小通信 C→A 重打包 —— 动态指令降 1.79% 但 wall 回退 1.4%
- **文件**：`docs/design/sm70_hmma_pipeline_search.md:37-60`
- **实测**：静态 SHFL 33→17、总指令 2,184→2,160、SHFL 52→36、F2F 80→72，但 **HADD2 8→0、SEL 0→20**。动态指令 -1.79%，但 **long-scoreboard 成本从 0.096 涨到 0.311 cycles/issue**，64K 下 wall **回退约 1.4%**。
- **推理依据**："**数学上的最小通信数 ≠ Volta 上的最小关键路径**"。静态指令数下降反而拉长了 issue 依赖链。**Closed for production.**

### A12. P2/P3 剩余 closed forms（HMMA 流水线搜索收尾）
- **文件**：`docs/design/sm70_hmma_pipeline_search.md:157-162`
  - shuffle-based 最小通信 C→A 重打包；
  - 第二个 K lookahead fragment 而第一个发布不变；
  - 围绕既有 FP32 输出的 per-row / warp-uniform 条件 rescale；
  - 在已枚举 ownership set 内做手工 TN/TT/NT layout 猜测。
- **同文档 `:129`**：per-row condition 方案 **+6.1% / +4.7% reject**（即用户提到的 "4.7%" 项，见 §D）。
- 终局结论（`:150-155`）：P0-P3 未达目标，剩余 NCU 证据由 MIO 与 dependency issue 压力主导，**HBM 远未饱和**。

### A13. Warp-local lookahead / 双缓冲 lookahead 组合 —— 多数否决
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:99-107`
  | 候选 | 结果 | 决策 |
  |---|---|---|
  | warp-local QK double-B lookahead | -0.56%, 40/40 | 保留为 opt-in；不与 early-store 叠加 |
  | warp-local PV B lookahead | **+1.26%** | reject；fragment lifetime 更长 |
  | double A+B lookahead | QK 不变；PV **+4.51%** | reject；stack/liveness 上升 |
  | QK triple-B lookahead | -0.59% | reject；相对 double-B 无增益 |
  | coarse PV shared stage pitch 128 | **+10.47%** | reject |
  | coarse PV shared stage pitch 136 | **+3.70%** | reject |
  | coarse PV shared stage pitch 152 | **+3.60%** | reject；冲突降低不足以偿 staging |
  | early `sP` store + QK double-B 组合 | -2.583%（相对 original） | reject as combination；无增益超过 early-store 本身 |
  | 跨 block 循环带 runtime final-block 谓词 | local loads **16.04M** | reject；loop-carried state spill |

### A14. QK-idle-warp V L1 prefetch —— 否决 +0.482%
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:108, 126-131`
- **实测**：`39.219 → 39.408 ms`，**+0.482%**，0/20 wins。
- **反证据（关键）**：L1 global load hit sector **上升 15.67M**，miss 下降 12.89M（缓存确实更热），但 **PV HMMA long-scoreboard samples 403,052 → 462,351** 反而上升；local loads 5.11M → 9.79M；named midpoint barrier 只是把既有 QK 等待切开。
- **结论**：不得通过改变预取量或 panel 顺序重试。

### A15. Q duplicate-lane shuffle / paired QK accumulator reuse —— 否决
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:114-116`
  - paired QK accumulator reuse：仅 **-1.68%**，local loads 105M / stores 25.1M → reject。
  - Q duplicate-lane shuffle：仅 **-3.97%** → reject；**shuffle 成本超过 shared-load 节省**。

### A16. 直接构造 col-major WMMA B 片段 / half-lane col-major B 加载 + 8 次 shuffle —— 否决
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:119-120, 241-257`
  - direct vector construction：`43.584 → 43.868 ms`，**+0.652%**，0/100 wins（`LDG.E.128.STRONG.CTA` 变体 -0.663%，原生 weak/SYS 变体 +0.652%）。
  - half-lane col-major B + 每 K16 8 次 fragment shuffle：QK panel `185.536 → 199.296 us`，**+7.416%**，0/100 wins，SASS 每 warp chain **新增 128 条 SHFL**。
- **推理依据**：减少参与的 load lane 无法偿付 Volta 的 shuffle 依赖链；重复 lane 已在 L1 tag/L2 sector 层面合并（每条 E128 仅 16 个 sector），不是主瓶颈。

### A17. K/V global-load-first 源码重排序 —— 否决（无增量收益）
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:118`
- **实测**：`49.654 → 43.575 ms`，**-12.244%**（locked 1200 MHz）但**相对已接受的 Q/P swizzle 无增量收益** → reject。

### A18. PV second-K16 compiler fence（`asm volatile("" ::: "memory")`）—— 否决
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:872-899`
- **实测**：两侧都 64 regs、41,744 B shared、0 stack/spill、2 CTA/SM，静态指令完全相同（768 HMMA / 74 LDG / 160 LDS / 6 BAR），输出 bitwise 相等。100 交替 A/B：
  | input | KV blocks | baseline | compiler fence | speedup |
  |---|---|---|---|---|
  | random | 1 | 54.144 us | 54.208 us | **-0.118%** |
  | random | 2 | 83.456 us | 83.328 us | +0.153% |
  | random | 4 | 128.128 us | 128.320 us | **-0.150%** |
  | alternating | 1/2/4 | — | — | 0.000%（全部持平） |
- **结论**：**源码级顺序不足以改变 SASS**；后续必须先证明发射的 SASS 顺序真的变了再测。

### A19. Probability-only bank layout（P group pad）—— 否决（未过 2% 门禁）
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:845-857`
- **实测**：random 4 blocks `87.936 us` 级别，改进仅 **1.15%** / **1.43%**（99/100、100/100），**未达强制的 2% 全case 门禁**，跳过 NCU 与生产集成。**关闭 probability-only bank-layout 工作**；不得靠叠加多个 sub-gate layout 凑过阈值。

### A20. Layout 实验账本（p_strict / pitch / prefetch / column-major）—— 多条否决
- **文件**：`docs/design/sm70_tp4_nomtp_long_context_decode.md:147-165`
  | 候选 | 结果 | 决策 |
  |---|---|---|
  | score-buffer pitch 136→144 floats | invalid pilot（模板 flag 未下传到 device `KernelConfig`） | 不得引用此前 timing |
  | output-accumulator pitch 264→268 floats | `44.039 → 40.222 ms` (-8.67%) | **Accept**（`VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268`，page 784 only） |
  | `p_strict` 双 `16x24` half slab | `40.227 → 40.686 ms` **+1.14%**，1/200 | Reject and remove |
  | contiguous `p_strict` pitch 40→48 half | `40.127 → 40.125 ms` -0.006%，103/200 | Reject（噪声级） |
  | full current-V-subtile `prefetch.global.L1` | `40.173 → 40.760 ms` **+1.463%**，0/200 | Reject；V 已 L2 resident，额外 L1TEX/issue 流量超过 latency hiding |
  | aligned compact column-major `p_strict` | `40.285 → 41.960 ms` **+4.157%**，0/40 | Reject；row-owned softmax producer 的散列 half store 成本 > P-load 冲突 |

### A21. BM32 Q-tile（BLOCK_M 16→32）—— 否决 -10.3% / -10.6%
- **文件**：`docs/design/sm70_v100_migration_control.md:36107-36135`
- **实测**（V100 GPU0）：
  | shape | block | BM16 low-smem | BM32 low-smem | 结果 |
  |---|---|---|---|---|
  | `4096x65536,Hq=12,Hkv=2,D=256` | 784 | 312.590 ms | 348.370 ms | **-10.3%** 吞吐 |
  | `16324x65296,Hq=12,Hkv=2,D=256` | 784 | 1098.908 ms | 1228.638 ms | **-10.6%** 吞吐 |
- **推理依据**：BM32 把 Q-tile CTA 数减半（应减少 K/V 重扫），但**per-CTA shared footprint 增长到 V100 失去 occupancy 与调度弹性**。
- **结论**：不默认 BM32，也不再花时间做这一 tile-size 变更。

### A22. BN192 / BN256 direct-KV —— 否决
- **文件**：`docs/design/sm70_v100_migration_control.md:34528-34529, 34681-34690`
- `BM16/BN192` 与 `BM16/BN256` direct-KV 都通过严格相等，但**长上下文点都输给 BN128**，`BN256` 明确回退。
- **推理依据**：**最大化 CTA 数不是最优** —— `BN64` 达到 3 CTA/SM 却不如 `BN128` 的 2 CTA/SM；更大的 BN192/BN256 开始损失长上下文效率。
- **BM256 归属**：本仓库中 `BM=256` 未以该名字出现；等价实验是上面的 `BN256` 与后续的 `BM32 phase` 家族。**"BM=256 变慢" 一项在本 commit 未找到直接对应记录，请以 `BN256` 与 `BM32` 两项为准。**

### A23. FlashInfer-inspired paged-prefix split-KV 原型 —— 大-Q 无效（访存/CTA 已饱和）
- **文件**：`docs/design/sm70_v100_migration_control.md:36046-36106`
- **实测**：
  | shape | baseline | split-KV | split | diff | 结果 |
  |---|---|---|---|---|---|
  | `4096x32768` | 155.441 ms | 150.089 ms | 16K | 3.05e-5 | +3.6% |
  | `4096x65536` | 312.920 ms | 307.963 ms | 16K | 3.05e-5 | +1.6% |
  | `4096x65536` | 312.920 ms | 308.767 ms | 32K | 3.05e-5 | +1.3% |
  | small-Q `512x65536` | 51.066 ms | 42.930 ms | 16K | 1.53e-5 | **+18.9%** |
- **推理依据（"split-K 对访存受限场景无效"的实证）**：**在 `M>=4096` 时 baseline 已有足够 CTA**，额外的 global workspace + merge kernel 基本抵消了 KV-partition 并行度。split-KV 只在 **small-Q / long-KV**（baseline CTA 数不足）有效。
- **裁决**：split-KV **保持 default-off**，启用时默认限制到 small-Q long-KV（`MAX_Q=2048`）；`VLLM_FLASH_V100_PREFILL_SPLIT_KV_MAX_Q=0` 仅用于显式大-Q 实验。数值 diff 归类为 Type-B softmax merge-order 漂移。

### A24. FlashInfer split-KV 的 row-max sentinel / contiguous dense / chunk window —— 多条否决
- **文件**：`docs/design/sm70_flashinfer_native_backend.md:290-420`（Active Prefill Decision / Split-KV 分节）
  - **contiguous dense path** 548 ms vs paged 256 ms → 结构性 ~2× 慢 → 排除。
  - **row-max sentinel**：partial SASS 3,432 → 3,504 指令，`328.876 ms` vs `322.370 ms` → 回退。
  - **split-KV merge 需改 FP32 reduction tree** → 无法 bitwise 相等 → 不能作为默认。
  - **chunk window 筛选**：`N=24288` -0.988% reject；`N=80960` -2.029% 为 promote 边界。

### A25. P3 G6 KV64 三 CTA 变体 —— 否决
- **文件**：`docs/design/sm70_tp4_nomtp_long_context_decode.md:628-654`
- **实测**：确实达到 96 regs/thread、25.86 KB shared、**3 CTA/SM**、27.71% occupancy，但分区只跑到 **216.30 GB/s** DRAM、eligible warps/scheduler **0.28**、**81.22%** 周期无可用 warp，分区耗时 **1.46 ms vs P1 的 702.75 us**。64K 整体算子更慢：`0.53081 ms` vs raw baseline `0.33067 ms`。与 P2 组合还产生 `max_abs_diff=3.814697265625e-05` → 精确门禁也失败。
- **推理依据**：单纯把 K/V shared 减半制造了**更多同步和更少的每-CTA 有用工作**；编译器 local spill 破坏访问合并。

### A26. P4/P5/P6/P7/P8/P9 路线账本 —— 5 条否决
- **文件**：`docs/design/sm70_tp4_nomtp_long_context_decode.md:702-742`
  | 路径 | 结果 | 决策 |
  |---|---|---|
  | P4 staged QK/softmax/PV | 128K 直接精确，但 `0.4004 → 0.4921 ms` **+23% 慢** | Rejected；QK stage 仍 168 regs / 43.26 KB / 18.65% occupancy |
  | P5 one-stage KV prefetch | `0.400256 → 0.400336 ms` | Rejected；NCU 资源与调度计数均未动 |
  | P6 block16 index specialization | 16-token page 测试 `0.3985 → 0.3743 ms`，但线上 Qwen hybrid cache 不是 16-token paged | 非终点结果；仅 opt-in |
  | C1 block784 `ld.global.cg` K/V load | 顺序 128K bitwise 精确且快约 3%，但 reverse 131,071-token paging 输出改变 `max_abs_diff=7.629e-06` | **Rejected before NCU or model execution**；实验 dispatch 已移除 |
  | P8 transposed QK `m32n8k16` | 顺序 128K bitwise 精确 `0.38634 → 0.37504 ms` (-2.93%)，reverse 131,071-token paging 改变 `max_abs_diff=7.629e-06` | **Rejected before NCU or model execution**；dispatch 与 harness toggle 已移除 |
  | P9 register-resident PV accumulator | 去掉 32 B local stack，但 reverse-page 比较不安全（底层 G6 baseline 本身非确定） | Rejected and removed |

### A27. P8 transposed `m32n8k16` —— 数值硬停（非性能否决）
- **文件**：`docs/design/sm70_tp4_nomtp_long_context_decode.md:847-885`
- **方案**：QK Tensor Core 表达式从 `Q[8,256]*K^T[256,32]`（`m8n32k16`）改为转置 `K[32,256]*Q^T[256,8]`（`m32n8k16`）。
- **实测**：顺序 128K bitwise 相等且 `-2.93%`；reverse 131,071-token tail `-3.25%` 但**不 bitwise 相等**。
- **结论**：**数值硬停**，不是性能否决。不得用 cache hint 或 transposed shape 做"纯速度"候选。

### A28. C1 `__ldcg(uint4)` K/V cache policy —— 否决（cache hint 不数值中性）
- **文件**：`docs/design/sm70_tp4_nomtp_long_context_decode.md:832-846`；主控 `docs/design/sm70_v100_migration_control.md:2201-2205`
- **背景**：接受的 P7 K/V 向量读走 Volta read-only 路径 `LDG.E.128.CONSTANT.SYS`（来自 `__ldg`）。q=1 长上下文 K/V 是单遍流、L1/L2 复用极低。
- **实测**：顺序 128K `0.38889 → 0.37739 ms`（快约 3%），但 reverse block-table/tail（131,071 token）**不 bitwise 相等**（`max_abs_diff=7.62939453125e-06`）。
- **结论**：**硬质量否决**。除非先解释并消除可见性/数值差异，否则不得重测 `cg`/`ca`/`cs` 或等价的 cache-hint 变更作为纯速度候选。未做 NCU 与全模型计时。

### A29. G6 baseline 自身非确定性（推翻 P7 endpoint 结论）
- **文件**：`docs/design/sm70_v100_migration_control.md:2206-2214`
- 重复的 131,071-token reverse-page 自比较显示 **G6 192-thread p256 base path 本身间歇性非 bitwise 相等**（无论是否叠加 P2/P7）。**P7 endpoint 仅为性能证据，不能当作质量已验收的默认值。** G6 需要先做 aligned shared-pitch 修复。

### A30. `VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16=1` —— 未准入（仍 default-off）
- **文件**：`docs/design/sm70_v100_migration_control.md:2196-2200`
- 虚拟把每个 784-token manager page 切成 49 个 attention block，**不减少 KV 字节**，且没有已验收的 Qwen3.6 no-MTP endpoint 结果。

### A31. Triton attention 调度：num_warps / tile —— 多条否决
- **文件**：`docs/design/sm70_v100_migration_control.md:4660-4740`
  - `num_warps=8`：decode 快 **1.193×**、prefill 快 **1.349×**，但 **prefill 严格相等失败**（`max_diff=1.52587890625e-05`）→ **不得进入主线路**。
  - `num_warps=2`：严格相等但**远慢于默认** → 否决。
  - `tile16` / `tile64`：改变 softmax tiling，prefill/decode **双双严格相等失败** → 仅实验。
  - `num_warps=4`（全模型端到端）：9.4682 → 9.2949 tok/s，**慢 1.83%**（token hash 不变）→ 只是安全的实验旋钮，**不构成 decode 加速**。

### A32. Triton tile 形状与预填充衰减（已定位为非注意力瓶颈）
- `docs/design/sm70_v100_migration_control.md:44290-44293`：attention 实验在 B8 改善 4.2%、B16 改善 6.2%；剩余 short-context gap B8 8.6% / B16 8.2% —— **不要再重复 attention 实验去关闭它**，下一个目标是 GDN state 流量与 dense projection 调度。

### A33. FP8/FP16/FP4 权重流式预取、额外 M64 tile、强制寄存器上限、paired-half scale、L2-prefetch —— 全部否决
- **文件**：`docs/design/sm70_v100_migration_control.md:48271-48274`
- **实测**：放宽 split-K partition 只让 M64 GEMM 估计改善 **4.1%** 且改变输出 bit → 已回滚，无验收主张。

### A34. Sparse-MLA `BLOCK_K=32/64` —— 每个上下文长度都更慢
- **文件**：`docs/design/sm70_v100_migration_control.md:45106-45113`
- **实测**：当前精确 `BLOCK_K=16`，单 C128 层 `12.43 us`(1K) / `18.37 us`(64K) / `26.98 us`(128K) / `42.40 us`(256K) / `85.63 us`(512K) / `168.74 us`(1M)。**`BLOCK_K=32/64` 在每个测试上下文都更慢，并改变约 36–44% 的 FP16 元素**；`BLOCK_K=8` 不是合法 Triton dot 形状。**动态更大 K 选择被否决。**

### A35. 其它 Flash-V100 attention 结构否决
- **文件**：`docs/design/sm70_flash_v100_prefill_operator_optimization.md:640-700`、`:868-880`
  - `swizzle.global.L1` / 预取 fence 类：见 A14、A18。
  - P0 G6 dual-CTA redesign：被 P3 否决（见 A25）。
  - P4/P5：见 A26。

---

## B. GEMM / 量化算子 —— 方案否决

### B1. CUDA-core GEMV 替代 Tensor Core —— 全族否决
- **文件**：`docs/design/sm70_nvfp4_turbomind_operator_optimization.md:588-596`
- **实测**：
  | 候选 | 结果 | 决策 |
  |---|---|---|
  | raw M=1 NVFP4 CUDA-core GEMV prototype | graph best `19.502 ms` vs graph GEMM `11.884 ms`；non-graph best `22.743 ms` vs `10.881 ms` | **reject raw scalar GEMV** |
  | warp-cooperative raw GEMV | non-graph MLP total **36.020 ms**，比 scalar split-K GEMV 更差 | reject；**没有 Tensor Core 吞吐的 shuffle/K 协作不够** |
  | half2 raw GEMV with fp16 partials | graph MLP total `16.108 ms`，比 scalar raw GEMV 快但比 graph GEMM **慢 1.36×** 且精度更差 | reject as production route；**用它终止纯 CUDA-core GEMV 调优** |
- **推理依据**：V100 的 HMMA.884 是唯一高吞吐路径；标量/half2 FMA 路径没有可用的 Tensor Core 吞吐。

### B2. 强制 split/swizzle selector + `8x256x64` registry 候选 —— 否决
- **文件**：`docs/design/sm70_nvfp4_turbomind_operator_optimization.md:584-587`
  - forced split/swizzle selector probes：最好 gate/up `6.247 ms`、down `4.215 ms`，只有 **0.1–0.2 ms** 量级 → selector forcing alone rejected。
  - `8x256x64` registry candidate：**gate/up 回退**；down 只到 `4.139 ms`；整体在噪声带 → 已回滚。

### B3. Stream-dense epilogue（stream-store）—— 否决
- **文件**：`docs/design/sm70_nvfp4_turbomind_operator_optimization.md:735`（AWQ Experiment Ledger）
- **实测**：`183.10 us` vs 同 fast-selector 默认 `166.46 us` → **回退**，确认 stream-store epilogue 在当前最优 family 上为负。

### B4. `Operand_A_Swizzle_8x64` 复用 —— 否决（性能性否决，数值正确）
- **文件**：`docs/design/sm70_v100_migration_control.md:41705-41722`
- **实测**：冲突确实改善（excessive shared wavefronts `211,200 (47%) → 15,360 (6%)`，通用路由的 9 次 16-byte A load 各有 16-way 冲突），但 **registers 159 → 163，executed instructions 1.769M → 1.860M，no-eligible cycles 仍约 81%，NCU duration 30.56 → 31.90 us 回退**。
- **结论**：临时注册已移除，**不得作为默认 tactic**。后续 TP4 row 工作必须转向更低寄存器的依赖/同步设计（warp-private A staging 或独立 N32 warp group），而不是再来一次整 tile XOR swizzle。

### B5. TP4 row 八 warp 分解（`TG_N=8` 裸变体）—— 否决
- **文件**：`docs/design/sm70_v100_migration_control.md:41726-41735`
- **实测**：局部利用率改善（regs `159 → 115`，occupancy `16.03% → 22.63%`，no-eligible `81.29% → 73.45%`），但 **总指令 1.769M → 2.151M**；交替 CUDA-event 对 **-0.45% 与 +0.44%** → 无默认 selector。
- **理由**：当前 mainloop 在新增 warp group 间**重复 staging/control 工作**。

### B6. TP4 gate/up M1 live-A HMMA mainloop —— 首道微基准即否决 +17.3%
- **文件**：`docs/design/sm70_v100_migration_control.md:41739-41752`
- **实测**：`35.30 → 41.42 us`，**+17.3%**。NCU：regs/thread `120 → 168`，理论 occupancy `25% → 18.75%`，指令 `5.919M → 6.341M`，DRAM 吞吐 `48.67% → 40.75%`，no-eligible `58.45% → 60.55%`。
- **结论**：临时 mainloop/config/registration 已移除；不得重试扩大寄存器 footprint 的复制型 M1 mainloop。
- **附带结构性结论**：接受的 N64 grid 只有 272 个有用 work tile；N32/N64/N128 备选的 warp 供给基本相同，**tile 尺寸本身无法修复 underfill**。

### B7. Direct-W13 split-K 多档（1/4/5/8/10/16/20/32）—— 全部非 bitwise
- **文件**：`docs/design/sm70_v100_migration_control.md:47963-47965`
- 相对通用路径全部非 bitwise → **不得作为 output-parity 修复重复**。

### B8. FP4 multirow 16/32-row reuse —— 否决
- **文件**：`docs/design/sm70_v100_migration_control.md:47852-47855`
- **实测**：保持 bits 但 **M32 gate/up 从 145 回退到 157–161 us**，对 down 无帮助。

### B9. FP8 M32 四相 split-K —— 否决
- **文件**：`docs/design/sm70_v100_migration_control.md:47851-47853`
- **实测**：四相 split-K 回退；减少 unrolling 把 regs `138 → 127`，但只在一个 shape 上省约 **5.36 us**。

### B10. 完整 W13 N64 tile / M16 tactic / 更大 W2 M64-M128 tactic / padded dequantize-to-FP16 bmm —— 全部否决
- **文件**：`docs/design/sm70_v100_migration_control.md:45659-45665`
- **实测**：full-W13 N64 tiles 单 W13 就 `6.098 ms`；padded dequantize-to-FP16 bmm `3.684 + 2.223 ms`，还要付约 **1.26 GiB** 的 dequantized 写入。**W2 direct atomic weighted reduction 也被否决** —— 多 token 累加非确定性。

### B11. AWQ MoE grouped/batched GEMM —— 严格相等失败
- **文件**：`docs/design/sm70_v100_migration_control.md:6524-6535`（条目 34）
- **实测**：grouped/batched AWQ MoE GEMM 严格相等失败，`max_diff` 高达 `0.000244140625`（即使 `VLLM_SM70_AWQ_TUNE_SMALL_SHAPES=0`）。默认 AWQ MoE 路线退回 per-expert dense TurboMind exactness bridge（`max_diff=0.0`）。

### B12. FP8 MoE compact / gated-SiLU epilogue / broadcast-input / router-topk C++ / W2 direct reduce —— 整族 "paused-unsafe"
- **文件**：`docs/design/sm70_v100_migration_control.md:6122-6164`（Unsafe or Non-Mainline Routes）
- **理由**：严格 diff 工作发现 compact/gated 变体有非零输出漂移；用户要求严格 `max_diff == 0`。
- **代码护栏**：实验 MoE 代码保留在 `VLLM_ENABLE_SM70_TURBOMIND_EXPERIMENTAL_MOE` 下，默认 CMake hook 不启用；**没有添加 Python facade 或 Torch binding**。
- 同族还包括：**Greedy-only 快捷方式被应用到官方 sampling**（否决）。

### B13. 旧 SM70 GDN/FLA `chunk_delta_h` autotune-schedule 移植 —— 已完全回滚
- **文件**：`docs/design/sm70_v100_migration_control.md:6134-6141`
- **理由**：**显示了速度但改变了确定性 token**，已被完整 revert。新的 latest-only default-off delta_h gate 是另一回事（op 级严格通过，但全模型 default-off 确定性还不够稳，不能做 clean max-diff 验收）。

### B14. 旧 0.0.3 GDN/FLA 全 op 行为作为 latest-mainline schedule 源 —— 否决
- **文件**：`docs/design/sm70_v100_migration_control.md:6142-6146`
- **理由**：old 与 latest 各自确定，但 old-vs-latest 的 `chunk_gated_delta_rule` 输出/状态在同一 Qwen-like 输入上**不是 bitwise 相同**。

### B15. GDN `warps=8` / `BK=32→64` 调度 —— 未准入
- **文件**：`docs/design/sm70_v100_migration_control.md:6100-6125`、`:6522`、`:26750-26800`
- **实测**：delta_h op fixture 上 `BV=16, warps=8, stages=1` 最快（`1.4366 ms` vs default-off `4.4327 ms`），12 个配置全部 strict-pass `max_diff=0`；**但全模型 token 等价不成立** —— default1 vs default2 在 output index 6 就分叉，`code_2048_o64` 在 index 1 分叉。latest default-off 本身在 3/4 prompt 上跨进程 token 不稳定。
- **裁决**：gate 保持 default-off 且不进主线路；需要先冻结全模型 GDN/FLA autotune 选择或捕获真实模型 GDN/logit 张量。
- **KKT schedule gate**：`VLLM_SM70_GDN_KKT_SCHEDULE=1` 严格安全但**运行时中性**，不是 decode 速度恢复路径（`:6522` 条目 30）。
- **GDN BK env**：`VLLM_SM70_GDN_KKT_BK=64` 等被记录为已测得的调优配置（`csrc` 与 `:26779`），但**没有独立记录 "BK 16→32→64 三档对照回退表"**；已记录的 GDN 调度类回退是 `BV=16/8` 相关（见 §D 表）。

### B16. CUDA core GEMV 替代 HMMA 的 AWQ 变体 —— 同 B1（AWQ 侧）
- **文件**：`docs/design/sm70_awq_exact_m5_batched_gemv.md:235`
- 该候选**不执行任何 HMMA 指令**，是 FP16 管的 uint4 工作。

---

## C. MoE / 投机解码 / DFlash2 —— 方案否决

### C1. NVFP4 QPN2 M32 —— 删除（用户点名的条目，已核实）
- **文件**：`docs/design/sm70_v100_migration_control.md:46319-46323`；`docs/design/sm70_quasar_nvfp4_dflash2_acceptance.md:555-559`
- **原文**：*"NVFP4 QPN2 M32 was removed because the exact candidate regressed down projection to `140.94 us` versus approximately `71.59 us`."*
- **推论**：down projection **回退约 1.97×**（71.59 → 140.94 us）。
- **对比**：被保留的 **QPN2 M16 NATIVE** 反而成功：真实 layer-55 TP4 shard 上 gate/up `72.30 → 66.76 us`、down projection `38.06 → 31.31 us`，每 target round 预计省 `0.786 ms`，M9/M15/M16 输出 bitwise 相等。

### C2. q5 投机长度 —— 否决（-7.4%）
- **文件**：`docs/design/sm70_v100_migration_control.md:46322-46324`；`sm70_quasar_nvfp4_dflash2_acceptance.md:559-561`
- **实测**：q5 TP4/B8 只到 `335.71 token/s`，比 q7 低 **7.4%**；平均发射长度降到 3.93 tokens/round（尽管接受率 58.58%）。

### C3. 单累加器 channel-FP8 QPN8 M16 —— 删除（改变 reduction order）
- **文件**：`docs/design/sm70_v100_migration_control.md:46323-46325`
- **实测**：一个 B2 样本 `274.28 token/s`，但**接受率归一化的 round rate 差约 0.6%**，且改变了 reduction order 削弱质量契约 → 源码已删除。

### C4. q3 投机长度（B8）—— 否决 -31.3%
- **文件**：`docs/design/sm70_v100_migration_control.md:46477-46480`
- **实测**：target shape 从 M64 降到 M32，但只到 `249.14 token/s`，比 q7 稳态 B8 低 **31.3%**；接受率 72.85%，但更短的提议只发 3.19 tokens/round vs q7 的 4.49。

### C5. FP8 QPN8 native M64 —— 删除（B8 回退 21.7%）
- **文件**：`docs/design/sm70_v100_migration_control.md:46454-46457`
- **实测**：算子输出 bitwise 正确、微基准热缓存也赢，但 **B8 回退到 `283.94 token/s`，比稳态 `362.53 token/s` 低 21.7%**。

### C6. Proposal temperature scale 0.85 —— 否决
- **文件**：`docs/design/sm70_v100_migration_control.md:46453-46454`
- B2 `258.38 token/s`，但 B4 降到 `313.46 token/s`（对比 native-M32 默认提议）。

### C7. 单遍 top-p kernel / 非默认 top-p tile 尺寸 —— 否决
- **文件**：`docs/design/sm70_v100_migration_control.md:46082-46088`
- 单遍 top-p kernel 更快，但在随机 GLM 形状精确性测试上**改变了 703 个 mask 元素**；非默认 top-p tile 尺寸同样改变 mask。8-warp rejection block-stat 候选改变了 accepted-token chain（尽管隔离中间 buffer 相同）→ 已移除。

### C8. AOT compile（GLM-5.3 DFlash2）—— 否决（未过 30 ms 门禁）
- **文件**：`docs/design/sm70_v100_migration_control.md:46076-46081`
- AOT compile 测得 `30.98 ms/round`，输出前缀精确，但**未达 30 ms 延迟门禁** → 否决。

### C9. Compact MXFP4 / direct-order 路由（首版）—— 因质量回退被否决
- **文件**：`docs/design/sm70_v100_migration_control.md:44860-44865`
- 首版 `69.949 token/s` 但质量门失败、输出不连贯 —— 原因是 M2–M8 warmup 与 verifier compaction 会覆写 `compact_expert_offsets`，而 direct-order 错误假设该 buffer 仍含 `[0,1,...,6]`。修复后改读 immutable `slot_expert_offsets`。
- 早期模型级质量回退记录见 `docs/design/sm70_v100_migration_control.md:44842-44845`（item 37 方向性差异 `2 → 0`，无候选获胜）→ 候选否决，**PP2 x TP4 QPN8 路线默认关闭**。

### C10. PP2/TP4 shared-gate prescaled 候选 —— 质量否决
- **文件**：`docs/design/sm70_v100_migration_control.md:45138-45105`（区间 `:45138-45152`）
- GSM item 37 与 54 回退且无获胜，HumanEval/16 回退，三个性能流全部改变 → **否决并移除运行时集成**（不是保留为 opt-in）。

### C11. PLE CPU-offload 的本地 staging / 随机 UVA 路径 —— 否决
- **文件**：`docs/design/sm70_v100_migration_control.md:46373-46388`
- 随机 UVA 读路径：即便用精确自然校准 hash 也从 `6986` 回退到 `5522 tok/s`（当前随机 UVA 路径在该形状约耗 0.3 s）。
- **同步本地 staging 候选**：warm prefill 只有 `5355-5399 tok/s`，decode 仍 `85.72 tok/s`；实测 gather `60-76 ms`，无法复现已验收的 offload graph —— **候选已停止，本地 staging 代码已移除**。

### C12. MTP4 fixed-slice GDN 候选 —— 单元等价但全模型门禁失败
- **文件**：`docs/design/sm70_v100_migration_control.md:47943-47983`
- 隔离调用从 `0.189 → 0.027 ms`，单元测试张量相等；**但全模型门禁失败**：同一 direct-copy PLE 路线下 fixed 输出在 token 25 分叉，verification 从 307 涨到 441 rounds；自然 prompt 0 与 2 分别在 token 36 与 208 分叉。**已回滚**。
- 后续修复版本（`index_select` 替代 GPU boolean indexing + 独立拷贝 pure-spec fast path）通过：draft-combine→next-PLE 均值 `4.951 → 3.248 ms`（**1.703 ms / 34.4%** 降幅）。

### C13. 通用 FP16 GEMM 替换（MTP4 batch GDN）—— 否决
- **文件**：`docs/design/sm70_v100_migration_control.md:48385-48387`

### C14. MTP4 batch GDN 的 HC / W13 / shared-gate 方向 —— 多条否决
- **文件**：`docs/design/sm70_v100_migration_control.md:48395-48399`
  - original-layout HC down **回退**；
  - packed HC 约 0.47 ms/96 pairs 的投影收益，但需约 **1.20 GiB/rank** 且 M10 down 回退；
  - shared-gate vector8 dot **改变 FP16 logits**；
  - fused packed output projection：M5 精确但耗 **22–28 us vs 16–17 us**，M10 算术不同 → 未经模型运行即否决。
- 更多见 `:48687-48690`、`:48506-48508`、`:48430-48433`。

### C15. Triton QSA scorer（新）—— 移除（约 5× 组件回退）
- **文件**：`docs/design/sm70_v100_migration_control.md:48687-48692`
- 实际 page204 导致寄存器 spill，**约 5× 组件回退**（尽管输出精确）。`tl.dot` 在此 lower 成标量 FP32 FMA，**不是 Tensor Core MMA**。原生精确 scorer 屏在 M5 中性 → 保留原始 scoring。

### C16. W2 expert-pair N16 / W13 shuffled epilogue-quad-split / HC producer-ready flags / HC 2-4-warp CTAs —— 全部否决
- **文件**：`docs/design/sm70_v100_migration_control.md:48819-48826`
- 全部保 bits 但回退。GDN BV16 分歧被追溯到 LLVM 的 first-product/FMA 收缩（最后四个 V 行）；显式钉住原 BV32 算术可恢复全部输出/state bit，但**精确 BV8/BV16 每 36 层组件链只省约 0.05 ms** → 作为研究保留，不值得为这点收益跑全模型。

### C17. 其它 DFlash2 / TP4 / PP 结构否决
- `docs/design/sm70_v100_migration_control.md:44231-44237`：per-step tail walker（每步做 device→host 同步 + 大量小 zero launch）与 sampled-token vocabulary clamp（把损坏输出变成任意边界 token，掩盖生产者）被拒；GDN PAD replication 与 last-column gather clamp 也被拒（会掩盖损坏的 block-table 契约）。
- `docs/design/sm70_v100_migration_control.md:45643-45650`：**Fusing SiLU into Triton 被拒** —— 近似改变了两个 FP16 值达 `1.22e-4`；**Cooperative HC combine/down、sum2/combine fusion、确定性替换 router、SGLang atomic persistent-HC 设计、细粒度 down-project/push pipelining** 均因更慢、非确定或死锁被拒。
- `docs/design/sm70_v100_migration_control.md:46544-46550`：直接 per-row publication `2.229951 ms`、coalesced revision `2.037357 ms`、exact one/two/four-part down `1.842709/1.850873/1.864315 ms`，全部慢于 `1.743988 ms` 对照；hidden two-row/four-warp、four-row/eight-warp、four-row/sixteen-warp 也都更慢（`1.738779/1.741660/1.719310 ms`）。
- `docs/design/sm70_v100_migration_control.md:46551-46565`：tensor-gather 预取实现因 32 KiB 动态 shared / 255 regs / 480 B stack 在编译期被拒；静态寄存器修订版编译通过但 matched 中位数 `2.109467 → 2.159213 ms`（回退）→ **停止 combine/norm + down 融合方向**。
- `docs/design/sm70_v100_migration_control.md:46566-46576`：packed down（每 rank 多 301.875 MiB）`2.174867 ms`、both `2.159398 ms` → 否决；packed up 单独只省 `0.013476 ms`，证据不足。
- `docs/design/sm70_v100_migration_control.md:46683-46693`：hidden-sharded 80-CTA up/mix/output-gather 原型 → net 更慢（control `2.105945` vs CUDA split `2.265607` vs fused `2.180970`）→ 否决。
- `docs/design/sm70_v100_migration_control.md:47007-47010`：exact down packet screen —— control `1.989379 ms` vs CUDA down+separate gather `2.072549 ms` vs fused `2.218926 ms`，全部 bitwise 但回退 → 否决。
- `docs/design/sm70_v100_migration_control.md:47121-47122`：H8/512 私有调度 `1.953417 ms` 比生产 `1.948767 ms` 慢 0.004649 ms → 不晋升。
- `docs/design/sm70_v100_migration_control.md:47821-47824`：TP4 push all-reduce 屏在 M64 回退 → 不叠加无限定通信开关。
- `docs/design/sm70_v100_migration_control.md:49138-49143`：**scalar-projection cooperative GDN 段使 C16 回退约 113%**；保留原生 batched projection 后损失降到约 7% 但**每个测量宽度都回退** → 两个变体都不准入。

### C18. Wheel RPATH / 依赖契约 —— 交付物否决
- `docs/design/sm70_v100_migration_control.md:45879-45913`：含绝对 RPATH 的诊断 wheel 被拒；RPATH-clean 替换版也因 FastAPI `0.141.1` + `prometheus-fastapi-instrumentator` `7.1.0` 组合导致 Uvicorn 请求 HTTP 500 被拒。

---

## D. 用户点名条目的逐条核实

| # | 用户条目 | 核实结论 | 证据位置 | 实测数字 |
|---|---|---|---|---|
| 1 | split-K 对访存受限场景无效 | **部分成立，需限定**：split-KV 只在 small-Q/long-KV 有效；`M>=4096` 时 baseline CTA 已饱和，额外 workspace + merge kernel 抵消并行度 | `sm70_v100_migration_control.md:36093-36101` | small-Q `+18.9%`；`4096x65536` 仅 `+1.6%`/`+1.3%` |
| 2 | BM=32/BN=128 变慢 | **部分成立**：BN=128 是**被接受的最优**，不是被否；被否的是 **BM32**（BLOCK_M 16→32） | `sm70_v100_migration_control.md:36122-36133`；接受记录 `:34555-34565` | BM32 `312.590 → 348.370 ms`（**-10.3%**）；`1098.908 → 1228.638 ms`（**-10.6%**） |
| 3 | BM=256 变慢 | **名称不对应**：本 commit 无 `BM=256` 记录。等价项是 **BN256**（长上下文点输给 BN128，明确回退） | `sm70_v100_migration_control.md:34528-34529, 34681-34690` | BN64 3 CTA/SM 仍输给 BN128 的 2 CTA/SM |
| 4 | 直接读 global 输入（不做 shared staging） | **成立** | `sm70_flash_v100_prefill_operator_optimization.md:113`；`sm70_flashinfer_native_backend.md:102` | direct global Q **+28.09%**，约 1.99B global sector |
| 5 | 权重改 uint4 加载（变慢 83%） | **未找到该 83% 记录**。仓库中 uint4 相关结论是：(a) AWQ uint4 是**已接受**的权重布局，不是被否项；(b) Marlin 拒绝 `weight_bits=uint4`（`sm70_v100_migration_control.md:2554`）；(c) NVFP4 路线要求保持 TurboMind prepared layout 而非 `[K, N/8]` uint4（`sm70_nvfp4_turbomind_operator_optimization.md:201`）。**"变慢 83%" 数字在本 commit 未核实到** | — | — |
| 6 | 流式 scale（变慢 4.7%） | **数字接近但归属不同**：`4.7%` 出现在 GDN per-row condition 调度屏（reject）；AWQ **stream-store epilogue** 是独立的否决项 | `sm70_hmma_pipeline_search.md:129`（per-row condition `+6.1% / +4.7% reject`）；`sm70_nvfp4_turbomind_operator_optimization.md:735`（stream-store `183.10 vs 166.46 us`） | per-row condition **+4.7%**；stream-store 约 **+10%** |
| 7 | `__ldg` 读激活/读权重 | **成立（作为 cache-policy 变体被否）**：接受的 P7 K/V 读已走 `LDG.E.128.CONSTANT.SYS`（`__ldg`）；改 `__ldcg(uint4)` 顺序 128K 快约 3% 但 reverse 131,071-token paging **不 bitwise 相等**（`max_abs_diff=7.629e-06`）→ 硬质量否决 | `sm70_tp4_nomtp_long_context_decode.md:832-846`；`sm70_v100_migration_control.md:2201-2205` | `0.38889 → 0.37739 ms`（顺序）但 reverse 页 `max_abs_diff=7.629e-06` |
| 8 | barrier 优化（仅 1% 收益） | **成立且更差**："1%" 这条更像是 `probability-only bank-layout` 的 **1.15%/1.43%**；而 barrier 方向本身是 **+4.84% 回退** | `sm70_flash_v100_prefill_operator_optimization.md:112`（+4.84%）、`:845-857`（1.15%/1.43% 未过 2% 门禁）、`:212`（barrier 9.31% / tensor active 8.89%） | barrier 调度 **+4.84%** |
| 9 | BK 16→32→64 | **部分核实**：文档记录 `VLLM_SM70_GDN_KKT_BK=64` 等 env 与 pin（`sm70_v100_migration_control.md:26779`、`:5981`、`:17552`），但**未找到 BK 三档对照回退表**。已记录的 GDN 调度回退是 **BV8 screen 回退 1.06%**（`:48307-48365`）；delta_h 最快配置是 `BV=16, warps=8, stages=1`（warps 2→8 是**接受**的 op 级结果，只是全模型 token 门禁未过） | `sm70_v100_migration_control.md:6100-6125`、`:6522`、`:26779` | warps=8 op 级 `1.4366 ms` vs default `4.4327 ms`（接受但 default-off） |
| 10 | GDN warps/block 2→8 | **部分成立**：op 级是**接受**的（`warps=8` 最快，12 配置全 strict-pass）；**被否的是全模型默认启用**（token 等价失败） | `sm70_v100_migration_control.md:6100-6125`、`:6522-6535` | op 级 `4.4327 → 1.4366 ms`；全模型 token 在 index 0/1/6/40 分叉 |
| 11 | bank conflict 优化 | **成立但收益不足**：probability-only bank layout **+1.15%/+1.43% 未过 2% 门禁**；aligned column-major P 布局 **+4.157% 回退**；被接受的是 **Q/P swizzle**（excessive wavefronts `637.8M → 275.5M`，-56.8%，且 total `39.064 → 34.249 ms`，-12.326%） | `sm70_flash_v100_prefill_operator_optimization.md:115-117, 845-857`；`sm70_tp4_nomtp_long_context_decode.md:157-165` | probability-only 1.15%/1.43%；column-major **+4.157%**；swizzle **-12.326%** |
| 12 | mma 减半 | **成立**：raw `m8n8k4` 分解（把 M16 拆两半） | `sm70_flash_v100_prefill_operator_optimization.md:120-122, 259-271`；`sm70_hmma_pipeline_search.md:20-35` | QK **+18.204%**、PV **+27.509%**；P0 证明 **不存在 zero-transfer 映射** |
| 13 | prefill 软件流水线 | **成立（部分）**：**跨 block 软件流水线（cross-block steady/drain）是接受项**（`39.153 → 38.222 ms`，-2.377%；与 Q/P swizzle 合并后 -12.326%）；**被否的是**：runtime final-block 谓词版本（local loads 16.04M，loop-carried state spill）、8+8 named-barrier 调度（+4.84%）、4 累加器 8+8 完整管道（spill） | `sm70_flash_v100_prefill_operator_optimization.md:109-110, 112, 138-145, 307-332` | 见左 |
| 14 | NVFP4 QPN2 M32 删除 | **完全成立，数字精确** | `sm70_v100_migration_control.md:46319-46321`；`sm70_quasar_nvfp4_dflash2_acceptance.md:555-558` | down projection **140.94 us vs ≈71.59 us** |
| 15 | CUDA core GEMV 替代 Tensor Core | **完全成立** | `sm70_nvfp4_turbomind_operator_optimization.md:588-596` | raw GEMV `19.502` vs graph GEMM `11.884 ms`；warp-coop `36.020 ms`；half2 `16.108 ms`（**1.36× 慢**且精度更差） |
| 16 | W4A4 / 量化激活（SM70 上是 W4A16） | **完全成立** | `sm70_quasar_tp2_tp4_quality.md:25-30, 160-180`；`sm70_quasar_nvfp4_dflash2_acceptance.md:24` | 量化激活使各投影输出变化：GDN in **6.88%**、GDN out **8.90%**、attn qkv **5.96%**、attn out **12.24%**、MLP gate/up **9.11%**、MLP down **9.01%** rel L2；只改最后 MLP down 即产生 **5.042%** sampling TV |

---

## E. 已被"关闭"的路线族 —— 顶层 "Closed Paths" / "Do not repeat"

### E1. FlashInfer native backend 的 Closed Paths（8 条）
见 §A4。出处：`docs/design/sm70_flashinfer_native_backend.md:209-220`。

### E2. 通用 SM70 迁移硬规则中点名的禁则
- **文件**：`docs/design/sm70_v100_migration_control.md:1783-2249`（Hard Rules）、`:6122-6164`（Unsafe or Non-Mainline Routes）
- 核心禁则：
  1. **不允许任何输出质量折衷**。
  2. **Type A（布局等价）路径必须 `torch.equal` 且 `max_diff == 0`**；任何 ≥`1e-6` 的稳定非零 diff 一律视为 indexing/mask/KV-source/block-table/`cu_seqlens` bug。
  3. **Type B（归约顺序不同）** 必须证明输入对齐 + 归约顺序来源 + dtype 阈值（fp32 ≤`1e-5`、fp16 ≈`2e-3`、bf16 ≈`1e-2`）+ 长度不单调发散。
  4. **未分类的非零 diff 路径保持 experimental，不能计入已验收/默认性能**。
  5. **两个对称的禁忌**：不能把 Type A 非零当噪声放过；也不能为了追 bitwise 一致而拖慢或替换已证明的 Type B 快路径。
  6. **Latest-TRITON 调度调优只是安全中间分支**，不替代 FlashAttention2/V100 迁移/修复轨道。
  7. **Flash-V100 是必须迁移的目标**：数值不匹配先分类，再决定修 `A-bug`、保留 `B-pending` 或以完整 `B-accept` 证据接受 —— **不匹配不构成放弃该路线的许可**。

### E3. Flash-V100 迁移验收状态（`B-pending`）
- **文件**：`docs/design/sm70_v100_migration_control.md:6387`（Acceptance Matrix 行）
- 摘要：Flash-V100 是 required migration item；varlen 与布局等价路径已修到 Type A `max_diff=0.0`；长 q=1 标量 paged decode 合成算子路径是 `B-accept`；**但全 Flash-V100 仍需修复/分类与模型级验收才能最终签收**；mixed Triton-prefill + Flash-decode **不再是已验收终点**。

### E4. 通用"不得重复"清单（按主题）
| 主题 | 明确"do not repeat"的方案 | 出处 |
|---|---|---|
| K-shard 接收器 | 集中式 receiver `77–82 µs`、分布式 receiver `44–48 µs`（对照 `23.7 µs`）；last-arrival norm reduction、grouped partial layout、更小 barrier arrival set 均无额外收益 | `sm70_v100_migration_control.md:110-116` |
| M8/M16 词汇投影 / sampling guard | 已证伪的 M8/M16 词汇投影或 sampling-guard 根因假设；M8 树的字面扩展也被拒 | `sm70_v100_migration_control.md:160-166` |
| 量化 batch reuse | 已拒的 prepared-weight、仅寄存器上限、M16-slicing 实验 | `sm70_v100_migration_control.md:312-315` |
| 采样绕过 | Greedy-only 快捷方式应用到官方 sampling | `sm70_v100_migration_control.md:6161` |
| DFlash2 32K 运行时 | auxiliary-stream overlap、可变 K、近似 reranking 均保持否决 | `sm70_v100_migration_control.md:44405-44406` |
| E4M3 FP32-SSM | FP32-SSM 配置不能解决全部失败 → 不得原样重复该长 cohort | `sm70_v100_migration_control.md:47403-47406` |
| 权重预取 | W13 固定几何只省 `0.000714 ms/48 calls`；next-group 权重/scale 预取只省 `0.007588 ms/48 calls` → 无生产 launcher 启用 | `sm70_v100_migration_control.md:47306-47347` |
| 普通 output GEMV | 已用 64-bit 向量化加载 → 不得再做投机性的 scalar→vector 改写 | `sm70_v100_migration_control.md:47351-47352` |
| 启动缓存 | 不得把 graph-node trace TPOT 当性能基线（trace 本身有开销）；不得为凑一个好看的分数重复宽采样扫描 | `sm70_v100_migration_control.md:697-701`、`:564-570` |
| 选择器 env | 不得用 `VLLM_ATTENTION_BACKEND` 等不支持的 selector env 当路线证据 | `sm70_v100_migration_control.md:9783-9786` |
| CUPTI 多进程 | 两次 Nsight 在 `cuptiActivityFlushAll` 崩溃无报告 → 外部运行时变化前不要重试该多进程 CUPTI 路径 | `sm70_v100_migration_control.md:46192-46194`、`:46455-46457` |
| NUMA pinning | 只给约 0.03–0.04 ms，已恢复原 affinity → 不要重复 | `sm70_v100_migration_control.md:46882-46886` |
| 环境表面清理 | 移除 6 个无消费方的 env 不等于"重新准入被拒的 schedule"（`VLLM_SM70_FLASH_V100_0DOT3_BENCHMARK_COMBO_KERNEL` 例子） | `sm70_environment_surface.md:70-87` |

---

## F. 硬件层面的"结构性不可能"总结（V100/SM70 特有）

这些不是某次实验的失败，而是 V100 微架构决定的天花板，任何后续方案都应先对照：

| # | V100 限制 | 后果 | 证据 |
|---|---|---|---|
| F1 | **无 `cp.async` / `LDGSTS`** | 不能用 512-byte stage 模拟 async copy；通用 `uint4 → shared → barrier` 模拟 **不被接受**，先前粗粒度 K/V staging **已使 wall time 上升** | `sm70_flashinfer_native_backend.md:92-104` |
| F2 | **无 `ldmatrix`** | 必须用 conflict-reduced `ld.shared.v4.u32` 手工构造 Volta fragment lane map | `sm70_flashinfer_native_backend.md:92` |
| F3 | **无 `wgmma` / warp-group register control（`setmaxnreg`）** | warp specialization 不可用；**强制寄存器上限被明令禁止**；必须靠 tile/role 分解去满足 Volta 的统一 per-thread 分配 | `sm70_flashinfer_native_backend.md:92-104` |
| F4 | **无 TMA / persistent 调度描述符** | 必须用 FlashInfer host planner、预计算 page-pointer 元数据、graph-stable workspace、SM-count 分区调度 | `sm70_flashinfer_native_backend.md:92` |
| F5 | **96 KB shared/SM，2 CTA/SM 是硬门禁** | M64 (83 KB) 只能 1 CTA/SM → 失去一半延迟隐藏 → M64 全族关闭 | `sm70_flashinfer_native_backend.md:270-275`；`sm70_flash_v100_prefill_operator_optimization.md:910-936` |
| F6 | **255 regs/thread 硬上限 + 512 线程时 64 regs 预算** | 4 个常驻 PV 累加器 + PV 更新峰值超预算 → 8+8 管道必 spill；寄存器在 kernel 内均匀分配，QK 空闲寄存器不能借给 PV | `sm70_flash_v100_prefill_operator_optimization.md:324-340`；`sm70_flashinfer_native_backend.md:275` |
| F7 | **Volta shuffle 依赖链极长** | 减少参与 load lane（half-lane loader，+7.416%）与最小通信 shuffle 映射（P1，-1.4% wall）都因 shuffle/依赖链变长而失败 | `sm70_flash_v100_prefill_operator_optimization.md:249-257`；`sm70_hmma_pipeline_search.md:56-60` |
| F8 | **原生 HMMA.884 = `m16n16k16`** | raw `m8n8k4` 分解必然更长串行链 + 重读 B（+18.2% / +27.5%） | `sm70_flash_v100_prefill_operator_optimization.md:267-271` |
| F9 | **寄存器分配 kernel 内均匀** | QK 角色空闲寄存器不能重分配给 PV 线程（决定 8+8 方案失败的根因） | `sm70_flash_v100_prefill_operator_optimization.md:328-332` |
| F10 | **HBM 未饱和** | 长上下文 prefill 的瓶颈是 **MIO / dependency issue 压力**，不是带宽 → 优化方向必须是 load/instruction 复用，不是加大访存并发 | `sm70_hmma_pipeline_search.md:150-155` |
| F11 | **DRAM 吞吐极低（0.53%–0.56%），L2 hit 99.23%** | prefill 不是 HBM 带宽天花板；增加 shared staging 只会加同步 | `sm70_flash_v100_prefill_operator_optimization.md:54-58`、`:492-501` |
| F12 | **NSight Compute PM Sampling 从 CC 7.5 起**；Nsight Systems GPU Metrics 不支持 SM70；`ERR_NVGPUCTRPERM` | 观测栈被限定为 CUDA-event / NVTX / NCU aggregate+PC Sampling / `clock64()` phase stamps；**不得把 NCU 全kernel 平均当作瞬时曲线** | `sm70_flash_v100_prefill_operator_optimization.md:1118-1130`；`sm70_v100_migration_control.md:46769-46773` |
| F13 | **V100 应用时钟 877/1290 MHz，非支持的 877/1530 MHz** | 所有 A/B 必须在 locked 1200 MHz 下比较，否则结果不可复现 | `sm70_v100_migration_control.md:46770-46773`；`sm70_flash_v100_prefill_operator_optimization.md:191` |
| F14 | **SM70 无 `cp.async`、无 warp specialization → 无法实现 FlashInfer 的 async-group commit/wait** | 必须用 prologue/steady/drain 编译期调度 + participant-counted named barrier 替代 | `sm70_flashinfer_native_backend.md:92`、`:200-207` |

---

# 第二部分：SM70 性能结论如何产品化为运行时策略

## 1. 分层架构总览

```
┌─ Python 侧 ────────────────────────────────────────────────────────────┐
│ vllm/config/sm70_native.py      NATIVE_FIELDS：69 个原生策略字段        │
│   - (field, env_alias, families, is_calculation)                      │
│   - 解析 env → 捕获 values 快照（不改 os.environ）                    │
│ vllm/_sm70/runtime.py           NativeRuntimeOwner：把 values 与 token  │
│   - bind(values, token) → slot 名（"sm70:slot:kernel_config.xxx"）    │
│   - activate() contextmanager → enter/exit 两个 extension 的 runtime   │
│ vllm/_sm70/policy.py            register_policy_op / call_native       │
│   - 100+ CONFIGURED_OPERATORS 显式接收 native_policy 参数            │
└───────────────────────────────────────────────────────────────────────┘
                            ↓ token（"sm70:1:len:val:..." 序列化）
┌─ C++ 侧 ──────────────────────────────────────────────────────────────┐
│ csrc/sm70_policy.h              PolicyField 枚举 + PreparedPolicy      │
│   - policy_key()：只对 calculation 字段做 FNV-1a hash                 │
│   - PolicyScope：TLS 切换 active_policy / active_prepared_policy      │
│   - policy_atoi / policy_exact_one / policy_dispatch_override         │
│ csrc/sm70_runtime_state.h       RuntimeState：policies/resources/ctr  │
│ csrc/sm70_runtime.h             NativeRuntime<Domain> TorchScript 类   │
│ csrc/sm70_tile_runtime_signal.cuh 跨 rank device-visible flag 协议      │
└───────────────────────────────────────────────────────────────────────┘
                            ↓ DispatchPolicy
┌─ Kernel launch ───────────────────────────────────────────────────────┐
│ csrc/sm70_turbomind/ops/gemm_runtime.cpp                             │
│   select_dense_dispatch_policy_impl() → kDefault/kMeasure/kReuse       │
│ csrc/.../kernels/gemm/gemm.cu                                       │
│   Gemm::Impl::Dispatch(ctx, policy, ...) → LaunchSpec                  │
│   Measure() → Find() → cache_.Insert(desc, spec)                     │
│   → spec.kernel->Launch(...)                                        │
└───────────────────────────────────────────────────────────────────────┘
```

## 2. 策略字段与 env 开关（`csrc/sm70_policy_fields.inc`）

**69 个字段**，由 `vllm/config/sm70_native.py::NATIVE_FIELDS` 生成（文件头注释：`// Generated from vllm.config.sm70_native.NATIVE_FIELDS. Order is ABI.`）。顺序即 ABI，**只能追加不能重排**。

字段元组格式：`(field_name, env_alias, families, is_calculation)`
- `families`：适用的量化族（`awq`/`fp8`/`mxfp4`/`nvfp4`/`gguf`/`f16`）
- `is_calculation`（第三列布尔）：**True = 参与 `policy_key()` hash 与编译缓存指纹；False = 仅诊断/追踪，不影响 hash**

### 2.1 计算类字段（参与 hash，40 个）
| env alias | 语义 |
|---|---|
| `TM_GEMM_TUNE` | TurboMind GEMM 调优参数串（解析失败回落默认值） |
| `VLLM_SM70_AWQ_DENSE_TUNE_MAX_M` | AWQ dense 参与调优的最大 M |
| `VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2` | AWQ MoE W2 exact 路径 |
| `VLLM_SM70_AWQ_MOE_DISPATCH_POLICY` | **dispatch override**：`default`/`reuse`/`measure` |
| `VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS` / `VLLM_SM70_FP8_DENSE_TUNE_MAX_M` / `VLLM_SM70_FP8_MOE_TUNE_MAX_TOKENS` / `VLLM_SM70_NVFP4_*_TUNE_MAX_M` 等 | 各族调优形状上限 |
| `VLLM_SM70_AWQ_TP2_FAST_SELECTOR` / `VLLM_SM70_AWQ_TP2_FAST_TARGETS` | TP2 fast-path selector 与目标描述符串 |
| `VLLM_SM70_AWQ_TP4_QKV_CTA64` | TP4 QKV CTA 几何 |
| `VLLM_SM70_AWQ_TUNE_SMALL_SHAPES` / `FP8_...` / `MXFP4_...` / `NVFP4_...` | 小形状是否参与调优（**默认不同，见 §2.3**） |
| `VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS` / `_ONLY` / FP8 同名 | 保留默认 split 数（只调 kernel 不改 reduction tree） |
| `VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE` / `VLLM_SM70_FP8_REUSE_IMPORTED_CACHE` | 复用导入的调优缓存 |
| `VLLM_SM70_FP8_QPN8_M16` / `_M32_CHUNKED` / `_M32_NATIVE` | FP8 QPN8 三条 decode 路线（**均 default-off**） |
| `VLLM_SM70_FP8_GROUPED_BMM_DECODE` | FP8 grouped BMM decode（默认 on，回滚 `=0`） |
| `VLLM_SM70_FP8_PREPARE_VEC` / `VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH` / `_PERMUTE_` / `_UNPERMUTE_` | MoE 单 token 快路径 |
| `VLLM_SM70_MXFP4_MOE_GROUPED_M8` / `_EXPERT_ROWS` / `_FAST_SELECTOR` / `_GROUPED_VERIFIER` / `MXFP4_MOE_COMPACT_GROUPED_DECODE` / `MXFP4_MOE_BROADCAST_INPUT_DECODE` | MXFP4 MoE 族 |
| `VLLM_SM70_NVFP4_QPN2_M16_NATIVE` | **已接受的 NVFP4 QPN2 M16**（M32 已被删除，见 C1） |
| `VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL` / `_GROUPED_EXPERT_ROWS` / `NVFP4_QWEN38_MOE_FAST_PREFILL` / `NVFP4_QWEN38_TP4_M1_FAST_SELECTOR` | NVFP4 族 |
| `VLLM_SM70_F16_DENSE_MAX_M` / `F16_DENSE_TUNE_MAX_M` | FP16 dense 边界 |
| `VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR` / `FP8_SAFE_FAST_SELECTOR` / `FP8_PREFILL_FAST_SELECTOR` / `FP8_PREFILL_CUTLASS` | FP8 selector |
| `VLLM_SM70_DFLASH2_QPN8_RERANK` / `_SHADOW` | DFlash2 rerank（shadow 走 `VLLM_SM70_DEBUG=selector`） |
| `VLLM_SM70_GLM53_MOE_SHUFFLE_SORT_Q8` / `VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS` | GLM5.3 族 |
| `VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE` / `VLLM_SM70_AWQ_MTP_M5_FAST_SELECTOR` | AWQ 族 |
| `VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC` | DFlash2 sharded context FC |
| `SM70_MARLIN_DENSE_CTA_GEOMETRY` / `SM70_MARLIN_DENSE_SPLIT_K` / `SM70_MARLIN_DENSE_METADATA_CACHE` / `SM70_MARLIN_MOE_CTA_GEOMETRY` / `SM70_MARLIN_MOE_SPLIT_K` / `SM70_MARLIN_MOE_METADATA_CACHE` | Marlin 族（保留的 immutable init binding） |

### 2.2 诊断类字段（不进 hash，6 个）
`TM_GEMM_TRACE`、`TM_GEMM_TRACE_FILTER`、`TM_GEMM_TRACE_LIMIT`、`TM_GEMM_WARN_CACHE_MISS`、`VLLM_SM70_PROFILE_TRACE`、`TM_GEMM_CACHE_SUMMARY`。

> **关键设计**：`policy_key()`（`csrc/sm70_policy.h:62-70`）**只对 `calculation_fields[i]==true` 的字段做 FNV-1a hash**。因此"加一条 trace 开关"不会让编译缓存全部失效，而"改一个数值上限"会。

### 2.3 已知默认值（C++ 侧 `policy_atoi(field, fallback)`）
| 字段 | 默认 | 位置 |
|---|---|---|
| `awq_tune_small_shapes` | **1** | `csrc/sm70_turbomind/ops/gemm_runtime.cpp:192-194` |
| `fp8_tune_small_shapes` | **0** | `:201-204` |
| `mxfp4_tune_small_shapes` | **1** | `:206-209` |
| `nvfp4_tune_small_shapes` | **1** | `:242-245` |
| `mxfp4_moe_compact_grouped_decode` | **1** | `:211-214` |
| `mxfp4_moe_broadcast_input_decode` | **1** | `:216-219` |
| `mxfp4_moe_grouped_m8` | **0** | `:221-224` |
| `mxfp4_moe_grouped_verifier` | **0** | `:226-229` |
| `mxfp4_moe_grouped_m8_expert_rows` | **0** | `:231-234` |
| `mxfp4_moe_grouped_m8_fast_selector` | **1** | `:236-240` |
| `nvfp4_moe_grouped_prefill` | **1** | `:247-250` |
| `nvfp4_moe_grouped_expert_rows` | **0** | `:252-255` |
| `nvfp4_qwen38_tp4_m1_fast_selector` | **1** | `:257-261` |
| `fp8_moe_single_token_per_expert_dispatch` | **0** | `:263-267` |
| `fp8_0dot3_dense_selector` | **0** | `:269-272` |
| `fp8_safe_fast_selector` | **0** | `:274-277` |
| `f16_dense_max_m` | **64** | `:420-422` |
| `dflash2_qpn8_rerank` / `_shadow` | **0** | `gemm_runtime.cpp:496-500` |

> 注意 `awq_tune_small_shapes` 默认 **1** 而 `fp8_tune_small_shapes` 默认 **0** —— AWQ 族在小形状上默认参与调优，FP8 族默认不参与。这是 §B12「AWQ MoE grouped/batched 严格相等失败」被隔离的历史原因之一。

### 2.4 Dispatch override（`csrc/sm70_policy.h:72-80`）
`VLLM_SM70_AWQ_MOE_DISPATCH_POLICY` 可取：
| 值 | 语义 |
|---|---|
| 未设置 / 空 | `Unset`（走正常策略） |
| `default` | 强制 `kDefault` |
| `reuse` | 强制 `kReuse`（只复用已测 plan，不测） |
| `measure` | 强制 `kMeasure` |
| 其它 | `Invalid` → 抛异常 |

### 2.5 其它 env 层
- `vllm/config/sm70_runtime.py:48-53`：`runtime_aliases` = `VLLM_SM70_AWQ_WARMUP_MAX_M`、`VLLM_SM70_ASYNC_STAGED_INPUT_PREP`、`VLLM_SM70_AUX_KERNEL_WARMUP`、`VLLM_SM70_MTP_CONCURRENCY_WARMUP`。
- `sm70_runtime.py:69-75`：`freeze_tuning_after_warmup`（bool，默认 **False**）—— 语义见 §4。
- `vllm/config/sm70_environment_surface.md`：**`VLLM_SM70_DEBUG`** 统一调试入口，接受逗号分隔 `trace`/`mtp`/`events`/`routing`/`selector`，**默认空**（观察不添加任何工作）。
- `SM70_QSA_TUNING`（`sm70_environment_surface.md:55-68`）：已测默认 = 64 MiB score tiles、512-row cuBLAS crossover、1048576 score elements、64-row page4 XQA crossover。

---

## 3. 运行时测量（measure）→ 固定（reuse）的机制 ★核心

### 3.1 DispatchPolicy 位标志（`csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/types.h:159-170`）
```cpp
enum class DispatchPolicy : int {
  kDefault = 0,                  // 启发式默认
  kMeasure = 1,                  // 本 shape 首次到达 → 实测 autotune
  kReuse   = 2,                  // 已测过 → 复用缓存 plan
  kAppend  = 3,
  kPreserveDefaultSplits      = 4,
  kPreserveDefaultSplitCount  = 8,
  kMxfp4MoeGroupedM8Fast      = 16,
  kSm70Fp8PrefillPrescaled    = 32,
  kSm70Nvfp4Prescaled         = 64,
  kPreserveDefaultPartition  = 128,
};
```
位标志设计意味着"复用调优结果"与"保留默认 split 数"等策略可以**正交叠加**。

### 3.2 测量维度（选什么 plan）
`GemmDesc` 决定 plan 身份（`csrc/.../gemm/gemm.cu:564`）：**M、N、K、group_size、量化类型、batch dim、epilogue**。
`TuneKey`（`csrc/sm70_turbomind/ops/gemm_runtime.cpp:468, 574`）额外带 **device** 与 **policy key**。
`LaunchSpec` 记录：**kernel 名、splits、swizzle、CTA tile (cta_m/cta_n/cta_k)、warp tile (mma)、stages、smem_size、regs、max_ctas**（`gemm.cu:509-518` 的 trace 输出即完整描述）。

### 3.3 决策链（`csrc/sm70_turbomind/ops/gemm_runtime.cpp:448-487`）
```cpp
select_dense_dispatch_policy_impl(device, m, n, k, group_size, stream,
                                  kind, tune_enabled, reuse_imported_cache, max_m):
  1. if (m > max_m)                       return kDefault;      // 超出调优形状上限
  2. if (reuse_imported_cache && has_imported_cache(device))
                                             return kReuse;    // 导入缓存优先
  3. if (!tune_enabled)                   return kDefault;      // 该族不参与调优
  4. if (is_tuning_frozen(device))        return kReuse;        // ★warmup 后冻结
  5. key = {kind, device, m, n, k, group_size}
     if (dense_tuned_shapes.contains(key)) return kReuse;        // 该 shape 已测过
  6. if (is_stream_capturing(stream)) {                          // ★图捕获中绝不测量
       if (has_imported_cache(device))   return kReuse;
       return kDefault;
     }
  7. dense_tuned_shapes.insert(key);       return kMeasure;      // ★首次 → 实测
```

**三个关键保证**：
1. **图捕获期间绝不测量**（第 6 步）—— 注释明确指出 `tune_mutex` 已持锁，再调 `has_imported_cache()` 会递归加锁死锁。捕获期只用已缓存或默认 plan。
2. **测量只在形状第一次出现时发生**，之后一律 `kReuse`。
3. **冻结后（warmup 结束）未测形状走"下一个更大已测 plan"**，而不在请求到达时测量。理由见 `vllm/config/sm70_runtime.py:69-75`：
   > *"a shape the warmup did not measure reuses the next larger measured plan (else the heuristic) instead of being measured on the first request that brings it, which slowed the first DeepSeek-V4 prefills of some lengths after a boot by 1.4 to 3.7 s and made the kernel choice depend on the request order."*

### 3.4 测量执行与缓存写入（`csrc/.../kernels/gemm/gemm.cu:1080-1143`）
```cpp
const bool measured = operation.dispatch & DispatchPolicy::kMeasure;
{
  lock(dispatch_mutex_);
  if (measured) impl_->Measure(ctx, barriers_size, partials_size, 1, launch, stream,
                                operation.dispatch & kPreserveDefaultPartition);
  spec = impl_->Dispatch(ctx, operation.dispatch, barriers_size, partials_size);
  // kPreserveDefaultSplits / kPreserveDefaultSplitCount：kernel 或 split 数回退到默认，
  //   swizzle 取 min(spec.swizzle, kernel->GetMaxSwizzle(default_desc))
  //   —— 只优化 kernel 选择，不改 reduction tree（保证数值等价）
}
if (spec.kernel && dispatch_context != &context) {
  // 实际 group 数与 override 不同时，重新夹紧 splits 到 workspace 能容纳的上限
  spec.splits = max(1, min(spec.splits, kernel->GetMaxSplits(actual_desc, swizzle, ...)));
}
MaybeTraceGemmDispatch(desc, policy, spec, measured);   // TM_GEMM_TRACE 输出
return launch(spec, stream);                            // ← 真正的 kernel launch
```

`Dispatch()` 内部（`gemm.cu:562-766`）：
- `cache_.Find(desc)` 命中 → 直接返回（**这是"reuse"**）；
- 否则 `Find(ctx, barriers_size, partials_size, top_k, false)` 枚举可行 spec（`top_k=0` 表示全量；`top_k=1` 表示只要默认），
- 选 `specs.front()`，可选 `SelectSm70AwqTp2FastSpec` 覆盖（TP2 fast path），
- `cache_.Insert(desc, selected)` 写入缓存（**这是"measure 后固定"**）。

### 3.5 SM70 专属的调优搜索空间（`gemm.cu:534-546`）
```cpp
if (arch_ == 700) {
  // V100 decode is dominated by many tiny GEMM/GEMV problems.
  // A broader search space consistently finds better launch specs than
  // the generic defaults for these SM70 workloads.
  tuning_.max_splits = 16;
  tuning_.max_waves  = 32;
  tuning_.swizzle    = {0, 1, 2, 3, 4};
  tuning_.top_k      = 0;      // 全量枚举
  tuning_.clusters   = 0;      // Volta 无 cluster
  tuning_.min_iter   = 2;
  tuning_.max_iter   = 20;
  tuning_.max_time   = 2.f;    // 每个候选最多 2 秒
}
```
`TM_GEMM_TUNE` 可覆盖以上参数；解析失败打印 `[Gemm2] Failed to parse 'TM_GEMM_TUNE', default value will be used.` 并回落默认（`gemm.cu:547-556`）。

### 3.6 冻结与导入/导出（`csrc/sm70_turbomind/ops/gemm_runtime.cpp:868-934`）
| op | 作用 |
|---|---|
| `sm70_gemm_import_cache(device_hint, path)` | 从文件导入已测 plan；成功后把 `{device, policy_key}` 记入 `imported_cache_devices`（`:882-886`） |
| `sm70_gemm_export_cache(device_hint, path)` | 导出当前 cache 为文件（`:890-915`） |
| `sm70_gemm_freeze_tuning(device_hint)` | 把 `{device, policy_key}` 记入 `frozen_tuning_devices`，并返回本设备已测 shape 数（`:917-933`） |

对应的 Python 绑定：`vllm/_sm70/linear.py:2060-2075`（`sm70_gemm_import_cache` / `sm70_gemm_export_cache` / `sm70_gemm_freeze_tuning`）。
`vllm/config/sm70_runtime.py:67-68`：`gemm_lut_path` —— "Optional tuning-cache template; device/rank expansion remains worker-local."

### 3.7 策略如何绑定到 kernel launch
1. **Python 初始化期**：把 69 个 env 值解析成 `NATIVE_FIELDS` 顺序的 `values` 元组，**快照**（不改 `os.environ`）。
2. **绑定**：`NativeRuntimeOwner.bind(values, token)`（`vllm/_sm70/runtime.py:84-114`）：
   - 找到与该 `values` 关联的 owner 路径 → slot = `"sm70:slot:" + path`（如 `sm70:slot:kernel_config.sm70_awq.native`）；
   - 找不到则派生 slot = `"sm70:slot:derived:" + sha256(只含 calculation 字段)`（**诊断字段与地址不进 hash**）；
   - `handle.bind(slot, token)` 写入 C++ `RuntimeState::policies`（`csrc/sm70_runtime.h:13-21`，slot 一旦绑定不可改）。
3. **token 序列化**：`sm70:1:<len>:<val>:<len>:<val>...`（`csrc/sm70_policy.h:180-211`），C++ 侧解析成 `PreparedPolicy`，计算 `key` 并 `parse_scalars()`。
4. **每次调用**：`with_policy(op)` 包装器（`csrc/sm70_policy.h:306-313`）在调用前构造 `PolicyScope`（TLS 切换 `active_policy` / `active_prepared_policy` / `active_policy_key`），调用后自动还原。
   ```cpp
   template <typename Result, typename... Args>
   auto with_policy(Result (*operation)(Args...)) {
     return [operation](Args... args, std::optional<std::string> native_policy) -> Result {
       const PolicyScope scope(native_policy);
       return operation(args...);
     };
   }
   ```
5. **kernel 选择**：`select_*_dispatch_policy()` → `DispatchPolicy` → `Gemm::Impl::Dispatch()` → `LaunchSpec` → `spec.kernel->Launch(...)`（`gemm.cu:1139-1143`）。

### 3.8 图捕获安全
- `csrc/sm70_policy.h:304-305` 注释：*"This scope is host-only and ends after launch. Capture records the selected kernels; graph replay needs neither TLS nor policy storage addresses."* —— **CUDA graph 只记录已选定的 kernel，replay 不需要策略存储**。
- `csrc/sm70_runtime_state.h:20-21`：*"Host owners are never serialized into an AOT graph. A reloaded graph resolves its stable policy slot under the current engine's forward context."*
- `RuntimeState` 是 `RuntimeResource` 的宿主（`sm70_runtime_state.h:15-18`），`close()` 会释放所有 native 资源并清空 policies/resources/counters。

---

## 4. 运行时上下文的组织方式

### 4.1 RuntimeState（`csrc/sm70_runtime_state.h:22-40`）
```cpp
struct RuntimeState {
  inline static std::atomic<uint64_t> next_id{1};
  const uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
  bool closed = false;
  std::mutex mutex;
  std::unordered_map<std::string, const PreparedPolicy*> policies;   // slot → policy
  std::unordered_map<std::string, std::unique_ptr<RuntimeResource>> resources;
  std::unordered_map<std::string, std::atomic<unsigned>> counters;    // 诊断计数
  void close() { /* 关资源、清 map、置 closed */ }
};
```
- `active_runtime` / `runtime_stack` 是 **thread_local**（`:42-43`），支持嵌套 enter/exit（`:45-56`）。
- `current_native_runtime()`（`:58-63`）：没有 engine owner 时回落到进程级 `legacy` state（兼容老的无 config 调用）。
- `diagnostic_counter(name)`（`:65-69`）是引擎拥有的 per-runtime 诊断计数（如 `tm_gemm_dispatch`）。

### 4.2 NativeRuntime TorchScript 类（`csrc/sm70_runtime.h:8-34`）
`bind(slot, token)` / `enter()` / `exit()` / `close()` / `resource_count()` / `policy_count()`。
注册了两个 domain（`_C` 与 `_moe_C`），并导出 `sm70_native_runtime_abi() -> 1` 与 `sm70_native_runtime_context_id() -> int`（`:46-52`）。

### 4.3 Python 侧 owner（`vllm/_sm70/runtime.py`）
- `NATIVE_OWNERS`（`:18-30`）列出所有可绑定的 owner 路径：
  ```
  kernel_config.layer_execution.native
  kernel_config.sm70_marlin
  kernel_config.sm70_mxfp4
  kernel_config.sm70_{awq,fp8,nvfp4,gguf}.native
  kernel_config.sm70_moe.{awq,fp8,nvfp4,mxfp4}.native
  ```
- 初始化时探测两个 extension 的 ABI 是否为 1，否则抛 `"Engine-owned SM70 native resources require runtime ABI 1; rebuild both normal _C and _moe_C extensions."`（`:36-41`）。
- 通过 `sm70_native_runtime_context_id()` 探测两个 DSO 的 TLS 是否共享；若共享则只保留一个 handle（`:50-63`）。
- `activate()` 是 `@contextmanager`（`:116-133`），负责 enter/exit + ContextVar 切换。
- `BoundNativeCall`（`:142-155`）：**AOT 记录稳定 slot；eager 初始化借用同一 owner**。若当前已在 owner 上下文内则直接调用，否则临时 activate。
- `bind_native_runtime()`（`:158-166`）：从 `current_runtime_resources()` 取/建 owner，并注册到 `execution_context_owners`。

### 4.4 策略作用域（`csrc/sm70_policy.h:278-302`）
```cpp
class PolicyScope {
  explicit PolicyScope(const std::optional<std::string>& token);  // 设置 active_policy/
  ~PolicyScope();                                                  // 恢复 previous_
  // 复制被禁用（非拷贝语义）
};
```
`thread_local` 变量：`active_policy`（值向量指针）、`active_policy_key`（FNV hash）、`active_prepared_policy`（`PreparedPolicy*`）（`:59-60`）。

### 4.5 兼容路径（`csrc/sm70_policy.h:43-57`）
```cpp
// Legacy direct callers have no policy argument. Capture compatibility inputs
// once, without mutating the process environment. Prepared engines supply all
// values explicitly and never consult this fallback.
inline const std::vector<std::string>& legacy_policy() {
  static const auto values = [] { /* 一次性读 getenv，未设置填 "\x1f" */ }();
  return values;
}
```
即：**兼容路径只读一次 env，不写回 `os.environ`**；引擎路径完全不读 env。

### 4.6 跨 rank device-visible signal（`csrc/sm70_tile_runtime_signal.cuh`）
```cpp
constexpr int kMaxBlocks       = 64;    // TileRT/all-reduce 默认 block 上限（不变）
constexpr int kMaxSignalBlocks = 512;   // 更大的 signal 容量，为长 prefill collective-fusion 预留
constexpr int kMaxRanks        = 8;
using FlagType = uint32_t;

struct Signal {                       // alignas(128)
  FlagType start[kMaxSignalBlocks][kMaxRanks];
  FlagType end[kMaxSignalBlocks][kMaxRanks];
  FlagType _flag[kMaxSignalBlocks];
};
struct __align__(16) RankData  { const void* ptrs[kMaxRanks]; };
struct __align__(16) RankSignals { Signal* signals[kMaxRanks]; };
```
设备端原子可见协议（`:31-46`）：
```cpp
store_flag_sys_visible: asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::: "memory");
load_flag_sys_visible:  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.sys;" ...);
```
- **`membar.sys` + `volatile` 是被否决后的保守替代**：`docs/design/sm70_v100_migration_control.md:32272-32277` 记录直接用 `st.release.sys`/`ld.acquire.sys` 的尝试**在 V100 上编译通过但 graph capture 期间 worker 以 `-11` 崩溃**；"one membar before/after the wait loop" 的低开销尝试也在启动时触发 `Bus error`；最终接受的是保守的 per-flag `membar.sys + volatile`。
- `RankSignals` 是 tile 级 GEMM↔all-reduce 融合的数据面（`docs/design/sm70_tile_runtime_exploration.md`）。

---

## 5. 已否决结论如何"变成"策略

这是本仓库最核心的方法论：**每次否决都被固化为一个永久的策略边界**，而不是只留在文档里。三条落地路径：

### 5.1 变成"策略默认值 = 保守值"
| 被否决的激进方案 | 固化的策略 |
|---|---|
| split-KV 在大-Q 下无效 | `VLLM_FLASH_V100_PREFILL_SPLIT_KV` 保持 default-off，启用时 `MAX_Q=2048`（`sm70_v100_migration_control.md:36099-36105`） |
| BM32 Q-tile 变慢 | `VLLM_FLASH_V100_PREFILL_D256_BM32` 保持 default-off（`:36133`） |
| 8+8 named-barrier / 4 累加器管道 | 只保留"一 CTA/group + 每 PV warp 2 个常驻累加器"的 phase-reuse 结构（`sm70_flash_v100_prefill_operator_optimization.md:350-357`） |
| M32 phase-reuse 之外的所有 BM32 变体 | `VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE` 严格受限：SM70 FP16 KV、`D=256`、page 784、`M>=32` 且 `M%32==0`、无 BFLA mask、无 sliding window；其它 shape 自动回落旧 kernel（`:418-423`） |
| QPN2 M32 慢 | 只保留 `VLLM_SM70_NVFP4_QPN2_M16_NATIVE`（`csrc/sm70_policy_fields.inc:53`） |
| FP8 QPN8 M64 回退 21.7% | 只保留 `M16` / `M32_CHUNKED` / `M32_NATIVE`，且**全部 default-off**（`sm70_v100_migration_control.md:46458-46460`） |
| W4A4 量化激活改变 9–12% 输出 | SM70 路线固定为 **W4A16**（`sm70_quasar_tp2_tp4_quality.md:25-30, 160-166`） |

### 5.2 变成"policy 层的能力门禁 / 硬 rollback env"
每个 default-on 的加速路线都配一个 `=0` 的硬回滚 env，形成"接受 + 可回退"的对：
| 加速路线 | 硬回滚 env | 出处 |
|---|---|---|
| Flash-V100 D256 GQA architecture 128K | `VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL=0` | `sm70_v100_migration_control.md:3071-3072` |
| Q/P swizzle 软件流水线 | `VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK=0` / `_PV=0` | `sm70_flash_v100_prefill_operator_optimization.md:207-210` |
| BM32 phase-reuse | `VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE=0` | `sm70_flash_v100_prefill_operator_optimization.md:420-423` |
| output stride 268 | `VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268=0`（仅 page 784） | `sm70_tp4_nomtp_long_context_decode.md:167-172` |
| FP8 grouped BMM decode | `VLLM_SM70_FP8_GROUPED_BMM_DECODE=0` | `sm70_v100_migration_control.md:44680` |
| NVFP4 QPN2 / QPN2 prefill | `VLLM_SM70_NVFP4_QPN2=0` / `VLLM_SM70_NVFP4_QPN2_PREFILL=0` | `sm70_dflash2_nvfp4_prefill_promotion.md:44-45` |
| MXFP4 compact grouped decode | `VLLM_SM70_MXFP4_MOE_COMPACT_GROUPED_DECODE=0` 等 | `sm70_v100_migration_control.md:44854-44856` |

### 5.3 变成"编译缓存指纹的一部分"（防止被否方案被隐式重新启用）
- `policy_key()` 只对 calculation 字段做 hash（`csrc/sm70_policy.h:62-70`）；Python 侧 `compute_hash()`（`vllm/config/sm70_native.py:736-746`）同样排除 `sm70_profile_trace`。
- `resolve_for_owners(overrides, *, layers=None, trace=None)`（`:698`）按 TP 度数与模型家族**条件性**启用字段，例如 `field.startswith("sm70_tp4_") and tp == 4`（`:722-725`）。
- **Admission 从不读模型名、checkpoint、`model_type` 或架构标识**（`sm70_v100_migration_control.md:44956`）：*"Admission never reads model name, checkpoint, `model_type`, or architecture identity."*
- 结果：任何策略变更都会导致编译缓存指纹变化 → 旧 graph 不会被复用。

### 5.4 变成"route snapshot 回归门"
`docs/design/sm70_dflash2_policy.md:58-70`：
```console
python -m tools.sm70_route_snapshot --category dflash2 \
  --baseline-ref <previous-main-sha> \
  --expected-changes tests/config/data/sm70_dflash2_capability_changes.json
```
- 执行历史 getter/default 与 GDN 构造谓词；
- 标准 324 个配置保持路由不变；
- **只有 1 个显式 TP2/local-layout edge 是有意的准入变更**；
- "Snapshot 描述的是权限与准入，不是观察到的请求 kernel 命中。"

**效果**：把"某条路线曾被否决"变成 CI 里可执行的断言，防止回归。

---

## 6. 策略绑定到 kernel launch 的完整链路（一图流）

```
[进程启动]
  env VLLM_SM70_*/TM_GEMM_*  ──►  vllm/config/sm70_native.py::NATIVE_FIELDS
                                     │ 解析成 values 元组（顺序 = ABI）
                                     │ 快照，不写 os.environ
                                     ▼
                              NativeRuntimeOwner.__init__()
                                     │ 探测 _C / _moe_C 的 runtime ABI == 1
                                     │ 探测两 DSO 的 TLS 是否共享
                                     ▼
                              NativeRuntimeOwner.bind(values, token)
                                     │ slot = "sm70:slot:kernel_config.sm70_awq.native"
                                     │   或 "sm70:slot:derived:<sha256(calculation 字段)>"
                                     ▼
                              C++ RuntimeState::policies[slot] = PreparedPolicy
                                     │ token = "sm70:1:len:val:..."
                                     │ policy_key = FNV-1a(calculation 字段)
                                     ▼
[每次 kernel 调用]
  vllm/_sm70/policy.py::call_native(op, native_policy, *args)
                                     │ native_policy = slot 名
                                     ▼
  csrc/torch_bindings.cpp: ops.impl(name, with_policy(&sm70_xxx))
                                     ▼
  PolicyScope(token)  ──►  active_policy / active_prepared_policy / active_policy_key (TLS)
                                     ▼
  sm70_xxx(...)  ──►  policy_atoi(PolicyField::xxx, fallback)
                                     ▼
  select_dense_dispatch_policy(device, m, n, k, group_size, stream, ...)
                                     │
        m > max_m                        ──► kDefault
        imported_cache && reuse_flag     ──► kReuse
        !tune_enabled                    ──► kDefault
        is_tuning_frozen(device)        ──► kReuse
        dense_tuned_shapes ∋ key        ──► kReuse
        is_stream_capturing(stream)     ──► kReuse（有导入缓存）or kDefault
        否则                             ──► kMeasure   ★首次实测
                                     ▼
  Gemm::Impl::Dispatch(ctx, policy, ...)
        cache_.Find(desc) 命中 ──────────────────────────► 返回缓存 LaunchSpec
        否则 Find() 枚举 → specs.front() → cache_.Insert(desc, selected)
                                     ▼
  Gemm::Impl::Measure(ctx, ...)   （仅 kMeasure 时）
                                     ▼
  launch(spec, stream)  ──►  spec.kernel->Launch(A, B, C, D, swizzle, splits, workspace, stream)
                                     ▼
  MaybeTraceGemmDispatch(desc, policy, spec, measured)   [TM_GEMM_TRACE]
```

---

## 7. 策略层与"被否决结论"的对偶关系（一句话总结）

1Cat 把 SM70 的每一条性能结论固化成 **三件东西**：
1. **一个默认关闭或默认保守的策略开关**（`vllm/_sm70/policy.py` / `vllm/config/sm70_native.py` / `csrc/sm70_policy_fields.inc`）；
2. **一个硬回滚 env**（`=0` 即恢复旧路线）；
3. **一条进入编译缓存指纹的 calculation 字段**（改动即失效旧 graph）。

而每一条被否决的方案，都被固化成 **一条永久的 "do not repeat" 边界**，落在三处：
- 文档层的 **Closed Paths / Do not repeat / Unsafe or Non-Mainline Routes** 章节；
- 代码层的 **dispatch 移除 + env gate 移除**（不是 `default-off 保留`，而是**源码删除**）；
- CI 层的 **route snapshot 回归断言**（`tests/config/data/sm70_dflash2_capability_changes.json`）。

这三层合起来，使得"V100 上已经试过并被否决的优化方案"不会以"换个名字重试"或"默认开启"的形式重新进入产品 —— 这正是 `docs/design/sm70_flashinfer_native_backend.md:28-31` 所说的原则：
> *"A failed candidate is removed from the new route or kept behind an explicit diagnostic switch; it is never allowed to silently fall through and be reported as a FlashInfer-SM70 result."*

---

## 附录 A：文档清单（本次检索覆盖的 sm70/v100/volta 文档）

`docs/design/` 下共 **140+** 个 sm70 命名的设计文档（见 `ls docs/design/ | grep -iE "sm70|v100|volta"`）。本次重点精读：

| 文档 | 行数 | 角色 |
|---|---:|---|
| `sm70_v100_migration_control.md` | 49,698 | **核心控制文档**，work log + Hard Rules + Unsafe Routes |
| `sm70_flashinfer_native_backend.md` | 563 | **Closed Paths** 权威来源 |
| `sm70_flash_v100_prefill_operator_optimization.md` | 1,134 | **Experiment Ledger** 权威来源（prefill 结构实验） |
| `sm70_hmma_pipeline_search.md` | 173 | HMMA 流水线 P0–P3 搜索与 closed forms |
| `sm70_tp4_nomtp_long_context_decode.md` | ~1,400 | 长上下文 decode 的 Layout Ledger + P4–P9 账本 |
| `sm70_nvfp4_turbomind_operator_optimization.md` | 872 | NVFP4 GEMM 路线与 Dead Ends |
| `sm70_nvfp4_bundled_scale.md` | 95 | NVFP4 bundled scale（code+scale 打包） |
| `sm70_quasar_nvfp4_dflash2_acceptance.md` | — | **QPN2 M32 删除的原始证据** |
| `sm70_quasar_tp2_tp4_quality.md` | — | **W4A4 vs W4A16 审计** |
| `sm70_dflash2_policy.md` | 95 | DFlash2 策略层 + route snapshot 门禁 |
| `sm70_tile_runtime_exploration.md` | 511 | TileRT 路线与 all-reduce 分块负结果 |
| `sm70_environment_surface.md` | 92 | env 控制面与 `VLLM_SM70_DEBUG` |

## 附录 B：关键实测数字速查

| 方案 | 回退/收益 | 出处行 |
|---|---|---|
| 8-warp BM64 staged K/V | **+23.3% ~ +42.6%** | flashinfer:237-244 |
| M64 (512-thread) | **-0.23% ~ -2.04%** | prefill_opt:920-927 |
| D128 双 CTA | **+21.82%** | prefill_opt:346-348 |
| 8+8 QK shared double buffer | **+33.56%** | prefill_opt:97 |
| 8+8 PV shared double buffer | **+6.31%** | prefill_opt:98 |
| true 8+8 named barrier | **+4.84%** | prefill_opt:112 |
| direct global Q WMMA | **+28.09%** | prefill_opt:113 |
| raw-HMMA QK / PV | **+18.204% / +27.509%** | prefill_opt:121-122 |
| half-lane col-major B + shuffle | **+7.416%** | prefill_opt:120 |
| BM32 Q-tile (BLOCK_M 16→32) | **-10.3% / -10.6%** | control:36122-36126 |
| QPN2 M32 down projection | **140.94 vs ≈71.59 us** | control:46319-46321 |
| CUDA-core raw GEMV | **19.502 vs 11.884 ms** | nvfp4_opt:592 |
| warp-coop raw GEMV | **36.020 ms** | nvfp4_opt:593 |
| half2 raw GEMV | **16.108 ms (1.36× 慢)** | nvfp4_opt:595 |
| `__ldcg(uint4)` cache hint | 顺序 +3%，reverse **7.629e-06** | tp4_decode:838-841 |
| P8 transposed `m32n8k16` | -2.93% 但 reverse **7.629e-06** | tp4_decode:872-878 |
| P4 staged QK/softmax/PV | **+23%** | tp4_decode:711 |
| P3 G6 KV64 三 CTA | 分区 1.46 ms vs 702.75 us | tp4_decode:639-643 |
| BM32 phase-reuse（接受） | **-18.99% ~ -20.42%** | prefill_opt:437-443 |
| Q/P swizzle（接受） | **-12.326%**，wavefronts -56.8% | prefill_opt:164-176 |
| P1 shuffle 最小通信 | 动态 -1.79%，wall **+1.4%** | hmma:56-59 |
| q5 / q3 投机长度 | **-7.4% / -31.3%** | control:46322 / 46477 |
| FP8 QPN8 native M64 | **-21.7%** | control:46454-46457 |
| W4A4 量化激活 | 投影输出 **+5.96% ~ +12.24%** rel L2 | quasar_tp2:170-176 |
