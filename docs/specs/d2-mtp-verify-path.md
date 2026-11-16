# 规格 d2-mtp-verify-path —— 进行中

> 状态：`进行中`。诊断开关已就位（`XINFER_DIAG_MTP` + `XINFER_LOG_LEVEL`），真模型首轮 A/B 已出数（§5）。
> 五节结构见 CONVENTIONS §1.1；case id = `d2#<n>`。

## 1. 现象与已知事实
- **现象（上游文档口径）**：`docs/moe_offload.md:54` —— "MTP (`--num-speculative-tokens`) works but was slower in our tests."。这是本件要判生死的原始读数。
- **真模型实测：MTP 开比关快 31.3%**（`unsloth/Qwen3.8-27B-NVFP4`，64 层 / hidden 5120 / vocab 248320 / `mtp_num_hidden_layers=1`；§5.1，5/5 wins）⇒ 原始读数在能跑通的配置上**不复现**。
- **要让真模型跑通，先踩过两颗雷**：
  1. `--d 0,1` 加载期 `CUDA_ERROR_ILLEGAL_ADDRESS`（`cudarc …/driver/safe/core.rs:275`），落在 `e7fa54b` 的 world-2 oneshot allreduce 上；`XINFER_ONESHOT_DISABLE=1` 一关就好（本件全部读数都在关掉它的条件下取）。
  2. MTP 开启要从 KV 预算额外扣 `XINFER_MTP_RESERVE_MB`（默认 1024 MB/卡，`src/utils/kvcache_allocator.rs:905-913`）+ 快照下限抬到 4（`src/models/layers/deltanet.rs:1001`）；22 GB 权重 /2 之后剩余不够，报 `Available: 0.00 MB, Required: 32.00 MB, Reserved: 512.00 MB`。本件用 `--max-model-len 1024 XINFER_MTP_RESERVE_MB=128` 才跑通 ⇒ **A6 已证实为"挤掉上下文"而非"变慢"**。
- 一个 MTP 周期跑 **两次主干 forward**：`mtp_decode_step1` 单 token（`src/speculative/mtp.rs:220`）+ `[anchor, draft₀..K₋₁]` 的 prefill 形态 verify（`:288`、`:307-385`）。
- verify 快路径（`attention.rs` 工作区未提交的 +103 行）在 **每次调用里新建 `[n,k]` F16 权重并整权重 dequant**（`attention.rs:src/nvfp4_linear.rs:834-848`、`:1817-1836`）：写回 4× 权重字节，图内还会引入 MEM_ALLOC 节点（D14 根因形态）。
- 批量路径：`batch>1` 时 step1 对 **每条序列单独跑一次** 单 token forward（`src/speculative/mtp.rs:442-451`）。

## 2. 待验证假设（按 预期收益 × 验证成本 排序）
- **A2**：verify 段是周期里最大一段（实测 55%，§5.1），其中"每调用重新 dequant + 新分配 F16 权重"占多少未拆。
- **A3**：`step1` 那次整模型 forward 是纯浪费——anchor 的 hidden 本可由上一周期 verify 的输出直接给出。实测它占周期 36%，且等于 4-token verify 的 65%。
- **A5**：`verify_len` 未命中已捕获图 ⇒ 回退 eager（`src/utils/graph.rs:2118`）。真模型上 MTP + 开图目前直接 OOM（图捕获阶段），挂死/OOM 点未定位。
- **A4**：draft 的 K 次 lm-head GEMV —— 实测只占 9%，**降级**。
- **A6**：预算挤压 ⇒ 已证实为跑不通的原因（§1），不是"变慢"的原因。

## 3. 验收条
- `d2#1` 基线 A/B 表：同模型/同 prompt/贪心，MTP 关 vs 开，decode tok/s，含 shape/batch/TP/是否 graph/时钟（D12）。
- `d2#2` 一个周期的分段耗时表（step1 / draft / verify / 其他，各占比），带测量条件。**要 CUDA event 级读数**（§2.4 L0）；当前探针是 host 侧计时，且周期计数与 256/K 推算的 ≥64 不一致 ⇒ 未达标准。
- `d2#3` 主导项判定 + 修复后 A/B：MTP 开 ≥ MTP 关 × 1.5（K=3、接受率如实测出），数值表 + wins 计数。
- `d2#4` 精度：MTP 开的贪心输出与 MTP 关的贪心输出 **逐 token 完全一致**（状态提交/回退属严格项，D6）。
- `d2#5` 开关合规：诊断开关属 `XINFER_DIAG_*` 族、默认关，读取走 `src/utils/env.rs` 的 `OnceLock`，热路径不反复 `env::var`（D9/D10）。

## 4. 覆盖区间与回退路径
- 生效区间：`batch=1` 的 MTP verify（`src/speculative/mtp.rs:211-415`）；`batch>1` 的批量路径（`:442-451`）在单序列结论出来前不动。
- 失效区间：非 Qwen3.5/Qwen4 混合 Mamba 模型、DFlash2 路径（`uses_dflash`）不受本件影响。
- 回退路径：任一条变红 ⇒ 关掉 `--num-speculative-tokens` 回到单 token decode 基线，并把证伪结论当场进 `docs/design/sm70_1cat_rejected_paths.md`（D4）。

## 5. 结论
### 5.1 `d2#1` / `d2#2` 真模型首轮实测（2026-11-16，`tests/probes/mtp_decode_breakdown_probe.sh`）
测量条件：`unsloth/Qwen3.8-27B-NVFP4`（64 层 / hidden 5120 / vocab 248320 / mtp 1 层）、`--d 0,1`（TP2）、`--dtype f16`、`--max-model-len 1024`、`XINFER_MTP_RESERVE_MB=128`、`XINFER_ONESHOT_DISABLE=1`、`--disable-cuda-graph`、256 tok、`temperature=0`、`--ignore-eos`、5 轮交替、时钟**未锁**（空闲 135 MHz / 上限 1530 MHz，容器内无权限改）。

| 轮 | MTP 关 | MTP 开(K=3) | 差 |
|---|---|---|---|
| 1 | 45.06 | 59.79 | +32.7% |
| 2 | 45.08 | 59.25 | +31.4% |
| 3 | 45.13 | 59.86 | +32.6% |
| 4 | 45.21 | 58.09 | +28.5% |
| 5 | 45.41 | 59.44 | +31.0% |

**wins = 5/5**，中位 45.13 → 59.44（**+31.3%**）。

分段（每周期，host 计时；三段和 62.2 ms × 52 = 3.23 s，实测整步 4.30 s ⇒ 约 25% 未被三段覆盖）：

| 段 | 每周期 | 占三段和 |
|---|---|---|
| step1（整模型 1 token） | 22.36 ms | 36.0% |
| draft（3× head + 3× lm head） | 5.55 ms | 8.9% |
| verify（整模型 4 token） | 34.25 ms | 55.1% |

⚠ 周期数按日志计数为 52（已按 rank 去重），与 256 tok ÷ (K+1)=4 推算的 ≥64 不一致 ⇒ 计数待校准，`d2#2` 未达标准，下一轮换 CUDA event。

### 5.2 作废：8 层假模型的一轮
`/data/qwen3_5_8layer`（TP1/bf16）曾出 +15.1%（4/4 wins）。该 checkpoint 不是本件目标，数字只留作对照，不作结论。
