# 规格 d2-mtp-verify-path —— 进行中

> 状态：`进行中`。诊断开关已就位（`XINFER_DIAG_MTP` + `XINFER_LOG_LEVEL`），首轮 A/B 已出数（§5）。
> 五节结构见 CONVENTIONS §1.1；case id = `d2#<n>`。

## 1. 现象与已知事实
- **现象（上游文档口径）**：`docs/moe_offload.md:54` —— "MTP (`--num-speculative-tokens`) works but was slower in our tests."。这是本件要判生死的原始读数。
- **现象在本机唯一跑得通的 MTP checkpoint 上不复现**：8 层 Qwen3.5（`/data/qwen3_5_8layer`，hidden 5120 / vocab 248320 / 8 层 / `mtp.num_hidden_layers=1`）、TP1、bf16、**关图**，MTP 开比关 **快 15.1%**（§5，4/4 wins）。
- **但开图就跑不完**：同一配置去掉 `--disable-cuda-graph`，MTP 开启的进程在首个 decode 步 100% 占卡空转 17 min 不出结果（`Captured MTP verify graph len=4 (flashinfer=true)` 之后无任何步进日志）。
- **TP2 直接崩**：`--d 0,1` 下模型加载期 `CUDA_ERROR_ILLEGAL_ADDRESS`（`cudarc …/driver/safe/core.rs:275`），落在 `e7fa54b` 的 world-2 oneshot allreduce 上；本件因此只能在 TP1 取数。
- 一个 MTP 周期跑 **两次主干 forward**：`mtp_decode_step1` 单 token（`src/speculative/mtp.rs:220`）+ `[anchor, draft₀..K₋₁]` 的 prefill 形态 verify（`:288`、`:307-385`）。K=3 时主干吞下 `1+4=5` 个 token 的活，换回 4.92 个 token（§5）。
- draft 循环 **每步一次完整 lm head**（vocab 248320 × hidden）再 argmax（`src/models/qwen3_5_mtp.rs:517-540`）——实测是周期里最大的一段（§5）。
- verify 走 prefill 形态：`set_linear_is_prefill(true)`（`src/speculative/mtp.rs:288`）、GDN 一步一快照递推（`src/models/layers/deltanet.rs:1189`）。
- verify 快路径（`attention.rs` 工作区未提交的 +103 行）在 **每次调用里新建 `[n,k]` F16 权重并整权重 dequant**（`attention.rs:src/nvfp4_linear.rs:834-848`、`:1817-1836`）：写回 4× 权重字节，图内还会引入 MEM_ALLOC 节点（D14 根因形态）。
- 批量路径：`batch>1` 时 step1 对 **每条序列单独跑一次** 单 token forward（`src/speculative/mtp.rs:442-451`）。
- 预算侧：MTP 开启即从 KV 预算扣 `XINFER_MTP_RESERVE_MB`（默认 1024MB/卡，`src/utils/kvcache_allocator.rs:905-913`），快照下限抬到 4（`src/models/layers/deltanet.rs:1001`）。
- 前序已生效：`56fb086` QPN8 channel-wise GEMM（单序列 MTP 17.74→30.08 tok/s）、`144a07f` 载入期 repack。

## 2. 待验证假设（按 预期收益 × 验证成本 排序）
- **A3（本件主案）**：`step1` 那次整模型 forward 是纯浪费——anchor 的 hidden/logits 本可由上一周期 verify 的输出直接给出。实测它占周期 30.6%（§5，读数偏低不偏高）。
- **A4**：draft 的 K 次完整 lm-head GEMV 是周期里最大一段（58.9%）。
- **A2**：verify 段被"每调用重新 dequant + 新分配 F16 权重"拖慢（4× 写带宽 + 图内 MEM_ALLOC）⇒ 载入期/首次调用期物化一次并复用。
- **A5**：`verify_len` 未命中已捕获图 ⇒ 回退 eager（`src/utils/graph.rs:2118`）。**已证实到"开图直接挂死"这一步**（§1），挂死点未定位。
- **A6**：预算挤压（1024MB/卡 + 快照下限 4）使 KV 预算变小 → 分片更多，污染端到端读数。
- **A7（新）**：在 MoE + offload 的真目标模型上，批量 verify 一次要流式进 `(K+1)×top_k` 个专家的权重，host→device 流量随 K 放大，而 offload 路径对 `is_mtp_verify` 零感知（`src/models/layers/moe.rs`、`moe_offload/*` 内 mtp 命中 0 处）。

## 3. 验收条
- `d2#1` 基线 A/B 表：同模型/同 prompt/贪心，MTP 关 vs 开，decode tok/s，含 shape/batch/TP/是否 graph/时钟（D12）。
- `d2#2` 一个周期的分段耗时表（step1 / draft / verify / 其他，各占比），带测量条件。**要 CUDA event 级读数**（§2.4 L0），当前探针是 host 侧计时，只作排序用。
- `d2#3` 主导项判定 + 修复后 A/B：MTP 开 ≥ MTP 关 × 1.5（K=3、接受率如实测出），数值表 + wins 计数。
- `d2#4` 精度：MTP 开的贪心输出与 MTP 关的贪心输出 **逐 token 完全一致**（状态提交/回退属严格项，D6）。
- `d2#5` 开关合规：诊断开关属 `XINFER_DIAG_*` 族、默认关，读取走 `src/utils/env.rs` 的 `OnceLock`，热路径不反复 `env::var`（D9/D10）。

## 4. 覆盖区间与回退路径
- 生效区间：`batch=1` 的 MTP verify（`src/speculative/mtp.rs:211-415`）；`batch>1` 的批量路径（`:442-451`）在单序列结论出来前不动。
- 失效区间：非 Qwen3.5/Qwen4 混合 Mamba 模型、DFlash2 路径（`uses_dflash`）不受本件影响。
- 回退路径：任一条变红 ⇒ 关掉 `--num-speculative-tokens` 回到单 token decode 基线，并把证伪结论当场进 `docs/design/sm70_1cat_rejected_paths.md`（D4）。

## 5. 结论
### 5.1 `d2#1` / `d2#2` 首轮实测（2026-11-16，`tests/probes/mtp_decode_breakdown_probe.sh`）
测量条件：`/data/qwen3_5_8layer`（hidden 5120 / vocab 248320 / 8 层 / mtp 1 层）、`--d 0`（TP1）、`--dtype bf16`、`--max-model-len 8192`、`--max-tokens 256`、`temperature=0`、`--ignore-eos`、**`--disable-cuda-graph`**（开图时 MTP 侧挂死，见 §1）、4 轮交替、时钟 **未锁**（1312/1530 MHz，容器内无权限改）。

| 轮 | MTP 关 tok/s | MTP 开(K=3) tok/s | 差 |
|---|---|---|---|
| 1 | 128.71 | 148.23 | +15.2% |
| 2 | 129.10 | 148.75 | +15.2% |
| 3 | 128.97 | 148.32 | +15.0% |
| 4 | 128.58 | 148.23 | +15.3% |

**wins = 4/4**，中位 128.84 → 148.28（**+15.1%**）。⇒ 原始"比未开还慢"在本机唯一可跑配置上 **不复现**；差异必来自 §1 的另两个配置（开图 / TP2 / MoE+offload）。

分段（52 cycles / 256 tok ≈ 4.92 tok/cycle，host 侧计时，只作排序）：

| 段 | 每周期 | 占周期 | 占整步 |
|---|---|---|---|
| step1（整模型 1 token） | 7.97 ms | 30.6% | 23.7% |
| draft（3× head + 3× lm head） | 15.33 ms | 58.9% | 45.5% |
| verify（整模型 4 token） | 2.74 ms | 10.5% | 8.1% |
| 其他（未覆盖部分） | — | — | ≈22% |

⚠ 三段和 = 26.0 ms × 52 = 1.35 s，实测整步 1.73 s ⇒ 还有 ≈0.38 s 未被三段覆盖；且 host 计时只量到"启动后剩余等待"，**不是整段 GPU 时间** ⇒ `d2#2` 未达标准，下一轮改 CUDA event。
