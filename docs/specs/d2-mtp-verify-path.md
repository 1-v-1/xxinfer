# 规格 d2-mtp-verify-path —— 进行中

> 状态：`进行中`。诊断开关已就位（`XINFER_DIAG_MTP` + `XINFER_LOG_LEVEL`），真模型首轮 A/B 已出数（§5）。
> 五节结构见 CONVENTIONS §1.1；case id = `d2#<n>`。

## 1. 现象与已知事实
- **现象（上游文档口径）**：`docs/moe_offload.md:54` —— "MTP (`--num-speculative-tokens`) works but was slower in our tests."。这是本件要判生死的原始读数。
- **真模型实测：现象复现的条件是"上下文长度"，不是"开不开"**：短上下文（24 tok）MTP 开 +34%；768-token 上下文 MTP 开 **−25.3%**（§5.3）。cli 侧 1024 上限下测到的 +31.3%（§5.1）是这两端的混合读数。
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

### 5.3 `d2#1` 走 OpenAI API 的矩阵：单发/并发 × 小/大上下文（2026-11-16，`tests/probes/mtp_api_perf_probe.py`）
测量条件：同一模型、`--d 0,1`、`--dtype f16`、贪心（`temperature 0`/`top_k 1`）、`stream=true`、`XINFER_ONESHOT_DISABLE=1`、`XINFER_MTP_RESERVE_MB=128`、`--mamba-fraction 0.35`、`--prefill-chunk-size=<该格 mml>`、**关图**、时钟未锁。每格一台服务，`--max-num-seqs` = 该格并发，`mml = ctx + 256 + 64`。

| 格 | MTP 关（解码段 tok/s） | MTP 开(K=3) | 差 |
|---|---|---|---|
| 单发 × 小上下文（24 tok） | 45.5 | 61.2–61.6 | **+34%** |
| 单发 × 大上下文（768 tok） | 45.96 | 34.34 | **−25.3%** |
| 并发 4 × 小上下文 | 起不来 | 起不来 | — |
| 并发 4 × 大上下文 | 起不来 | 起不来 | — |

⇒ **现象复现了，但不在"并发"这一维，而在"上下文长度"这一维**：短上下文 MTP +34%，768-token 上下文上 MTP −25.3%。TTFT：小上下文 off 214.9 ms / on 32.7 ms；大上下文 off 455.1 ms / on 402.9 ms。

并发格起不来的两条原因（都未解）：
- 关图时 hybrid 状态槽位容量塌到 1：`MambaCache: no free slots (max_batch_size=1), increase preallocated capacity`（`mamba_cache_capacity` 默认取 `hybrid_mamba_graph_capture_max_batch(max_num_parallel_reqs)`，`src/core/runner.rs:624-632`）。
- 开图时预算不够：`GPU Memory Budget: 1.45 GB available → 0.57 GB workspace reserve (FlashInfer 520M + Transient 68M) → 0.87 GB for caches` ⇒ `GPU memory is insufficient for one prefill chunk after KV-cache, Mamba, attention, and graph buffers`。

⚠ 首轮大上下文格曾因 needle 提示词让模型只答 8 个 token 而失真（`chunks=8`）；改成"再写至少 200 词"后重取才是上表。

## 6. `d2#3` 一步多 token 的哨兵 `-1`（用户实测：并发无 MTP、单发开 MTP 都无限吐 4294967295）

**共同点 = 一个 step 里 token 数 > 1。** 引擎里"每条序列一份"的元数据与"每个 token 一份"的元数据是两条不同长度的轴：

| 轴 | 长度 | 出处 |
|---|---|---|
| 序列轴 | 序列数 | `mamba_slot_mapping`（`attention.rs/src/mamba_cache.rs:475` `get_slots_for_sequences`）、`block_tables`、`context_lens`、`cu_seqlens_*` |
| token 轴 | token 数 | `input_ids`、`positions`、`slot_mapping`、`batch_indices` |

- 并发无 MTP：`prepare_decode` 每条序列 push 一份（`src/core/runner.rs:2238-2242`），N>1 时两轴才不等长。
- MTP 单发：`build_verify_metadata` 里 token 轴长 `q_len = K+1`，槽位只给 1 个（`src/speculative/metadata.rs:49`，且用 `get_`（只读不建）而非 `ensure_`）。
- `batch=1` 且非 MTP 时两轴都是 1，永远重合 ⇒ 从不暴露。

**为什么打印成 4294967295**：贪心（`temperature 0`/`top_k 1`）走的是 CUDA 快路（`sample_with_strategy` → `sample_cuda`，`src/utils/logits_processor.rs:295-301`），核里"没选中过任何有效下标"就写哨兵：stage A 越界下标写 `i[it] = -1`（`gpu_sampling.cu:139`），stage B 以 `topi[t] = -1` 初始化（`:191`），`picked = topi[0]` 兜底（`:269-272`）。`i32 -1` 被 `Vec<u32>` 打成 4294967295。所以打印的不是 token，是"这一行 logits 根本没被认成有效行"。

**修法（按优先级）**
1. 把序列轴→token 轴的展开做成显式一步：产出 `seq_of_token`，让 `slot_mapping[t] = slots[seq_of_token[t]]`，并在两处入口断言 `slot_mapping.len() == input_ids.len()`、`mamba_slot_mapping.len() == seqs.len()`。
2. `build_verify_metadata` 改用 `ensure_mamba_slots_for_sequences`（与 prefill 一致），别在 verify 里 `get_`。
3. 采样侧加硬闸：核写 `valid` 标志或调用方检出哨兵就报错，NaN logits 必须显式失败，不许静默出 token（D9 的日志门控走 `XINFER_DIAG_*`）。

## 7. 复现环境事实：物理卡 4 上挂着 1113 MiB 的主上下文

`start-container.sh` 把宿主机物理卡 **4 和 5** 挂进容器（容器内重编号为 0/1）。2026-11-17 02:2x 实测：

```
4, GPU-f9db7e33…, 1113 MiB, Enabled   ← 无任何 compute-app 持有，是 persistence mode 留住的主上下文
5, GPU-74d2d5c8…,    1 MiB, Enabled
```

后果：模型加载后容器内 `GPU 0`（=物理 4）只剩 `0.66 GB available`，`GPU 1` 剩 `1.73 GB`。`max_model_len` 一旦给出，`kv_fraction` 就被强制成 0.8（`kvcache_allocator.rs:719-724`，且不允许与 `kv_fraction` 同时传），KV 池按预算铺满 → `max_parallel_reqs()`（`:1004-1045`）里 `remaining − fixed_runtime` 归零 → `max_parallel == 0` → 直接 bail（`:1201-1205`）。

⇒ 之前"同样参数跑不通"不是参数问题，是这张卡比别人少 1.1 GB。

## 8. 复位后实测（2026-11-17 02:5x，CLI 模式）—— 本节的"跑通了"是误判，见 §10

**单发 + MTP（`1 1024 --num-speculative-tokens 3`）当时只 grep 了 `dims|sampled`：**

```
[diag] logits dims=[1, 248320] n=248320 min=-11.63 max=29.84 mean=0.357 nan=0 graph=false
[diag][ordinal 0] sampled token ids: [760]    [diag][ordinal 1] sampled token ids: [760]
```

⇒ 这些行只证明 **prefill 那一行 logits 是好的**（第一个 token 是 prefill 出的）。同一份 `/tmp/tiny.log` 里还有 6 次 `k_dim=32 not supported` 和 `flashinfer decode head_dim=32 is not compiled for this device; supported: [128, 256]` —— decode 一上来就炸。当时把错误行过滤掉了，所以误以为"跑通了"。

## 10. `tiny-random/Qwen3.5` 在这台 V100 上走不到 decode

它的 `head_dim: 32`、`num_attention_heads: 8`、`num_key_value_heads: 4`。而这份构建里 flashinfer decode 只编了 `[128, 256]`，`gated_delta_rule_recurrence_varlen_gqa` 也不支持 `k_dim=32`。所以"用同架构小模型跑矩阵"这条路要先解决头维不被支持，否则连一步 decode 都迈不过去。
**并发 2 路在这台机上任何参数组合都起不来**：`max_parallel_reqs()` 归零 → `GPU memory is insufficient for one prefill chunk …`。而且日志里 `slot_bytes: 0`、`mamba_memory_bytes: 0` —— 这个模型实例**没被当成 hybrid mamba**。⇒ §6 第 1/2 条按"槽位轴 vs token 轴"的推断在这条路径上不成立，判据要改。

## 9. 为什么"异常退出后显存不释放"会反复出现

容器里 PID 1 = `tail`，不 `wait()`，所以任何孤儿 runner 永远停在 `Z xinfer <defunct>`（本轮收尾时 42 个），卡 0 停在 363 MiB、卡 1 停在 6 MiB。这不是参数问题，是容器没有 reaper。

## 11. 真实 27B × 两卡 × 满长度生成（2026-11-17 04:2x，`--d 0,1`）

| 格 | MTP 关 | MTP 开 | Δ | TTFT 关→开 |
|---|---|---|---|---|
| ctx=24, c=1 | 43.0 | **59.9** | **+39.3%** | 63.7 → 39.0 ms |
| ctx=24, c=2 | 起不来 | 59.6（聚合 58.5） | — | — → 4587.1 ms |
| ctx=768, c=1 | **43.7** | 32.5 | **−25.6%** | 572.1 → 415.9 ms |
| ctx=768, c=2 | 起不来 | 31.8（聚合 32.4） | — | — → 8469.0 ms |

结论：MTP 的收益随上下文变长而翻转 —— 小上下文 +39%，768 token 时 −26%。`chunks` 都是 257（=MAXT+1），所以是满长度生成，不是提前停。

两格的"关 MTP + 并发"仍然 `server not ready`，所以并发列只有 MTP 开的单边数，不能拿来比 Δ。

## 12. −25.6% 贵在哪一段：不用插桩就能定死

三段计时还没做，但已有数据已经够定死，因为对照组的形状是平的：

| | ctx=24 | ctx=768 | 随上下文 |
|---|---|---|---|
| MTP 关（每步 1 个 query） | 43.0 | 43.7 | **平的** |
| MTP 开（每步 K+1=4 个 query） | 59.9 | 32.4 | **塌** |

⇒ 慢的不是"上下文"，是**verify 那一步走的内核**。

`build_verify_metadata` 把 `is_prefill: true`，并填 `prefill_plan_info`（`metadata.rs:87-106`），所以 verify 吃的是 **prefill** 注意力内核。而加速报告自己写着：

> `[slow] flashinfer_prefill — the FlashInfer prefill kernels require SM80+ (cp.async is not available on Volta)`

1Cat 把两条索引分开：`state_indices_tensor_d` / `query_start_loc_d`（decode）对 `_p`（prefill），所以它的 verify 走 decode 那条；这里只有一条，且被标成 prefill。

修法即 §6 第 2 条的具体形态：verify 不该 `is_prefill=true` 走 prefill 内核，该走 decode 批量内核（`decode_plan_info`）。
## 13. 这条结论有既有事实背书（不是新猜的）

`docs/design/sm70_1cat_rejected_paths.md` 里已经记着：

- **F10**：长上下文 prefill 的瓶颈是 **MIO / dependency issue 压力**，不是带宽 → 加大访存并发没用。
- **F11**：DRAM 吞吐 0.53%–0.56%，L2 hit 99.23% → prefill 不是 HBM 天花板。
- **F12**：观测栈被限定为 **CUDA-event** / NVTX / `clock64()` phase stamps —— 这正是 `d2#2` 用 CUDA event 的依据。
- 第 501 行：`长 q=1 标量 paged decode 合成算子路径是 B-accept`，且 `mixed Triton-prefill + Flash-decode 不再是已验收终点`。

也就是说：verify 现在吃 prefill，等于每一步都付一次"Volta 上没有 cp.async 的 prefill"的 MIO 代价，而它随上下文涨 —— 与 §12 表里"只有 mtp=1 塌"完全对上。

## 14. 三段计时实测（ctx=768, c=1, K=3, 2026-11-17）

探针默认没开 `XINFER_LOG_LEVEL=debug`，所以 §12 之前一直读不到这三行 —— 已补进 `env_for`。

| 段 | 每步耗时 | 说明 |
|---|---|---|
| step1（主模型出 logits+hidden） | **23.39 ms** | 等价于一次普通解码步 |
| draft（MTP 头出 3 tok） | **5.28 ms** | |
| verify（`m=4 graph=false`） | **32.27 ms** | **比一整次普通解码步还贵** |

一步合计 ≈ **60.9 ms**。对照：非 MTP 一个 token = 1/43.7 tok/s = **22.9 ms**。

⇒ MTP 要赢，每步得净接受 ≥ 60.9/22.9 = **2.66** 个 token（上限 4）。实测 32.4 tok/s 反推净接受 ≈ 2.0 → **输**。

而 `graph=true` 计数为 **0**：verify 全程 eager。§12 的判断到这里是**带数的**：verify 走 `is_prefill=true` 的 prefill 内核、且没被图覆盖，一步付掉 32.27 ms，比主模型整步还贵。

## 15. 杠杆 1（verify 走图）在当前配置下不可用

把探针默认从 `--disable-cuda-graph` 翻成开图（`OK=1`）重跑，**8 格全 0.0**，报的是：

```
KVCache allocation failed: Insufficient GPU memory for KVCache allocation.
```

注意这条和 §14 那条**不是同一条**：之前是"一个 prefill chunk 不够"，现在是"KVCache 池本身开不出来"。也就是说开图后图池要的额外预算把 KV 池挤没了，`MFRAC=0.35 / mml=1024` 这组参数下没有既开图又装得下的组合。

⇒ `graph=false` 不是漏配，是当前显存预算下的必然。**杠杆 1 在这台机器上不是"纯赚"**，它要先解决图池预算。

所以 −25.6% 的修法只剩杠杆 2：verify 的内核路径（`is_prefill=true` 的 prefill 内核 → `decode_plan_info`）。

## 16. 杠杆 2（verify 换 decode 内核）—— 证伪

`XINFER_MTP_VERIFY_DECODE_KERNEL=1`（经 `/tmp/mvp` 传给探针，避开命令行被小写化）：

| 段 | 关（prefill 内核） | 开（decode 内核） |
|---|---|---|
| step1 | 23.39 ms | 27.90 ms |
| verify `m=4` | **32.27 ms** | **88.11 ms** |
| 解码段 tok/s | 32.4 | 32.4–32.8 |

**verify 反而贵了 2.7 倍**（32.27 → 88.11）。批量 decode 内核在 `q=4` 上没有 prefill 内核快 —— 它是为 `q=1` 设计的（`rejected_paths:501`：`长 q=1 标量 paged decode 合成算子路径是 B-accept`，即被验收的是 **q=1**）。

⇒ §12 那句"该走 decode 内核"**证伪**：不是内核选错，是 **`q=4` 这件事本身在 Volta 上没有快路径**。开关保留默认关，作为已证伪的 A/B 面。

剩下唯一还没试的：净接受率。§14 的算术是"需净接受 ≥2.66/4 才赢，实测 ≈2.0"。

## 17. 净接受率：反推出来是 ≈0，问题不在内核

`verify.rs:119-126` 的口径是 `avg_tokens/step = accepted/steps + 2`（anchor + bonus 各 1，draft 接受数另计）。

用 §14 的两个实测数反推：一步 `60.9 ms`，吞吐 `32.4 tok/s` ⇒

```
tokens/step = 0.0609 × 32.4 = 1.97
⇒ accepted/steps = 1.97 − 2 ≈ 0
```

**draft 出来的 3 个 token 基本一个都没被接受。** 那 MTP 就纯粹是开销：付了 draft(5.28) + verify(32.27) 的钱，只换回 anchor+bonus 两个 token，和不开 MTP 的 1 个 token 比必输。

所以 −25.6% 的根因不是"verify 走哪个内核"（§16 已证伪两条），是**草稿和主模型对不上**。

下一步该查的是 `verify.rs:46-57` 那条贪心前缀匹配与 draft 的 token 序，不是内核路径。

## 18. 贪心前缀匹配是**对齐的** —— §17 的"形状错位"假设不被代码支持

读 `mtp.rs:278-281` + `verify.rs:41-53`：

```rust
let mut verify_tokens = vec![anchor_token];
verify_tokens.extend_from_slice(&draft_tokens);   // [anchor, d0..d{K-1}]，长 K+1
...
let target_vec = all_logits.argmax(Minus1)?;      // 行 j = 位置 seq.len+j 的预测
compare_len = num_proposed.min(num_positions);    // = K
for i in 0..compare_len { if target_vec[i] == draft_tokens[i] { .. } }
```

行 0 是 anchor 那一行，它预测的正是 `d0`；所以 `target_vec[i]` 对 `draft_tokens[i]` 是**同一个位置**。索引对齐，没有 off-by-one。

⇒ §17 说的"两边的 i 不是同一个 token"这条**不成立**，别再顺着它找。

剩下的唯一解释是草稿本身与主模型不符（MTP 头输出错），而那需要一个直接量：**第一次不匹配时把 draft[i] 与 target_vec[i] 一起打出来**。没有这个数，"接受率≈0"仍然只是从吞吐反推的，不是测出来的。

## 19. 实测：草稿输出的是 **token 0**（不是"略有不符"）

`XINFER_DIAG_MTP=1` 下第一次跑出来的实测行：

```
MTP accept: proposed=3 accepted=0 first_mismatch=Some(0) draft=[0, 0, 0] target=[198, 248046, …]
```

| 量 | 值 | 含义 |
|---|---|---|
| `draft` | **`[0, 0, 0]`** | 草稿头每一步吐的都是 **token id 0** |
| `target` | `[198, 248046, …]` | 主模型的真实预测（`248046` 是 eos 之一）|
| `first_mismatch` | `Some(0)` | 第 0 个就对不上 |
| `accepted` | `0` | 一个都没接受 |

§17 的 `accepted ≈ 0` 从**反推**变成**测出**，而且原因比"对不上"更硬：**`draft_tokens_gpu` 输出全 0**。

所以这条链到此闭合：MTP 在这个移植上**从来没通过** —— 草稿头没产出有效 token，每步纯付 `draft 5.28 + verify 32.27 ms` 换 anchor+bonus。§12/§15/§16 三条内核路径的争论全都是次要的。

下一步是查 `draft_tokens_gpu` 为什么返回 0（MTP 头权重没加载 / 输入 hidden 是 0 / argmax 取到了 pad 行），不是再调内核。

## 20. 为什么 draft 全 0：权重侧已排掉一半，剩一条待测

| 已核实 | 证据 |
|---|---|
| 检查点里**有** mtp 权重 | 索引里 **15** 个 mtp 张量：`mtp.fc.weight`、`mtp.layers.0.input_layernorm.weight` … |
| 头**确实**按前缀取权重 | `qwen3_5_mtp.rs:224 let prefix = "mtp.";` + `:227 vb.pp(&mtp_block_prefix)` |
| 头有"权重在不在"自查 | `:56 keys.iter().any(|key| vb.has_key(key))` |
| 但配置里 `mtp_enabled: false` | 用户那份 TP4 日志的 `Config` 回显：`mtp_enabled: false, mtp_max_verify_tokens: 0` |

`argmax` 对**全相等**（全 0）的 logits 返回下标 **0** —— 与 §19 的 `draft=[0,0,0]` 完全一致。所以最可能是：`has_mtp_weights` 判假 → 头走了未初始化/零权重分支，或 `mtp_enabled=false` 让权重根本没被接上。

下一轮的唯一探针：在 `draft_tokens_gpu` 入口打 `initial_hidden` 的 min/max/mean/nan 与 `has_mtp_weights` 的返回值。若 hidden 非 0 而 logits 全 0，就是权重分支；若 hidden 全 0，问题在 step1 的返回。

## 21. 分支定了：hidden 是好的，问题在权重分支

`draft_tokens_gpu` 入口探针实测：

```
MTP draft in: hidden n=5120 nonzero=5119 nan=0 absmean=1.042e0
```

| 量 | 值 | 判 |
|---|---|---|
| `n` | `5120` | = `hidden_size`，形状对 |
| `nonzero` | `5119/5120` | **输入不是 0** |
| `nan` | `0` | 不是 NaN |
| `absmean` | `1.042` | 量级正常 |

⇒ §20 的两个候选里，**"hidden 全 0"排除**。剩下唯一一条：**`forward_step` 的权重分支没接上**（`mtp_enabled=false` 下 `mtp.*` 那 15 个张量没被真正绑进头）。

`argmax(全相等 logits) = 0` → `draft=[0,0,0]` → `accepted=0` → 每步只换回 anchor+bonus → 一步 60.9 ms 换 1.97 token → **−25.6%**。

下一轮：在 `forward_step` 里打 `hidden_out` 的同一组统计。若 hidden_out 全相等，就坐实权重没绑上。

## 22. 名字对不上：这份检查点走不到 `nextn.*` 那条分支

检查点里 15 个 mtp 张量的**全名**：

```
mtp.fc.weight                      mtp.layers.0.input_layernorm.weight
mtp.pre_fc_norm_hidden.weight      mtp.pre_fc_norm_embedding.weight
mtp.norm.weight                    mtp.layers.0.{mlp.*,self_attn.*}.weight
```

而 `qwen3_5_mtp.rs:240-257` 按分支取两套名字：

| 分支 | 取的键 | 这份检查点里有吗 |
|---|---|---|
| `!is_qvar_builder` | `mtp.pre_fc_norm_hidden` | **有** |
| `is_qvar_builder` | `nextn.hnorm` → `mtp.nextn.hnorm.weight` | **没有** |

模型是 nvfp4（`quant_method: "nvfp4"`），`is_qvar_builder()` 为真 ⇒ 代码去找 `mtp.nextn.hnorm.weight`，而检查点里只有 `mtp.pre_fc_norm_hidden.weight`。`:236-238` 那句注释"MTP 不继承骨干的 NVFP4 设置、`mtp_config.quant = None`"说明 mtp 权重是未量化 F16，但**取名的键仍跟着 qvar 分支走了**。

⇒ 这就是 §21 说的"权重分支没接上"的具体形态：**键名分支选错，norm 权重取不到**。

## 23. §22 那条改动实测无效 —— 目标未达成（round 12 收尾）

按 §22 改了 `qwen3_5_mtp.rs:223`：

```rust
let is_qvar_builder = vb.is_qvar_builder()
    && !has_any_key(vb, &["mtp.pre_fc_norm_hidden.weight"]);
```

重跑矩阵（`--d 0,1`，真实 27B）：

| 格 | 改前 | 改后 |
|---|---|---|
| ctx=24, mtp=1, c=1 | 59.9 | **60.1 / 60.7** |
| **ctx=768, mtp=1, c=1** | 32.4 | **32.6 / 32.6** |
| mtp=0 四格 | — | 仍 `server not ready` |

⇒ 大上下文那格 **32.4 → 32.6**，几乎没动，**没到 ≥43.7**。

所以 §22 的"键名分支选错"即便成立，也不是 `accepted=0` 的成因；或者 `has_any_key` 在这里根本没按那个全名命中（它按 builder 的相对名解析）。

**未证伪的还剩**：`forward_step` 里 `hidden_out` 是否全相等（§21 留的那一步没做）。
**结构性障碍**：`mtp=0` 四格在这台两卡机上始终起不来（`max_parallel_reqs()==0` / KV 池开不出来），所以"单发/并发 × 大小上下文"的完整矩阵从来没有可比的两边。

## 24. 根因测出来了：MTP 头输出的是 **NaN**

`forward_step` 出口探针实测：

```
MTP head step 0: hidden_out n=5120 nonzero=0 absmean=NaN; logits n=248320 nonzero=0 absmean=NaN
```

| 量 | 值 | 读法 |
|---|---|---|
| `nonzero=0` | NaN 过不了 `abs() > 1e-9` | 不是"全 0" |
| `absmean=NaN` | 求和遇 NaN 即 NaN | **整段是 NaN** |
| `n=5120` / `248320` | 形状对 | 不是切片错 |

`argmax(全 NaN)` 返回下标 **0** ⇒ `draft=[0,0,0]`（§19）⇒ `accepted=0` ⇒ 每步只换回 anchor+bonus ⇒ `ctx=768` 32.2 对 43.9。

**修正 §21/§22 的措辞**：不是"权重没绑上所以输出 0"，是**头把一段健康的输入算成了 NaN**（输入侧 §21 已证健康：5119/5120 非零、absmean 1.042）。

## 25. NaN 出在哪一段：`norm_embed`，不是整条链

逐段探针实测（`XINFER_DIAG_MTP=1`）：

```
MTP head norm_hidden: n=5120 nan=0     absmean=5.671e-1   ← 健康
MTP head norm_embed:  n=5120 nan=5120  absmean=NaN        ← 全 NaN
```

| 段 | 结果 |
|---|---|
| `pre_fc_norm_hidden(backbone_hidden)` | **健康**（absmean 0.567，无 NaN）|
| `pre_fc_norm_embedding(token_embedding)` | **5120 个全 NaN** |

⇒ NaN 不是从骨干 hidden 带进来的（§21 已证输入健康），是**第二路 embedding 查表**产生的。

对应 `draft_tokens_gpu` 里这一行：

```rust
let token_embed = embed_weight.index_select(&current_token_t, 0)?;
```

`embed_weight` 在 nvfp4 模型上是**量化张量**，`index_select` 直接按行切片量化块 ≠ 反量化后取值 ⇒ 查出来是 NaN。

`cat([norm_embed, norm_hidden])` 之后整段被污染，`fc` → `layer` → `norm` 全 NaN，`argmax(全 NaN)=0` ⇒ `draft=[0,0,0]`。

## 26. 修点定了：草稿拿的是**量化后的** `embed_weight()`

`mtp.rs:236-240` 把主干的 `embed_weight()` 直接递给 `draft_tokens_gpu`：

```rust
let embed_weight = match self.model() {
    Model::Qwen3_5(m) => m.embed_weight(),
    ...
};
```

而主干自己前向时**不是**这么用的 —— 它经 `qwen3_5.rs:715/732` 的 `index_select`，作用在**已经过 qvar 机制反量化过的** `xs` 上。`embed_weight()` 返回的是原始张量；nvfp4 下它是打包的量化块，按行 `index_select` 切出来的不是那一行的值 ⇒ NaN（§25）。

检查点里 mtp 那 15 个张量**不含** embedding，MTP 必须复用主干嵌入，所以修法只有两条：

1. 让草稿这条路拿到**反量化后**的嵌入行（和主干前向同一条路径）；
2. 或在头内先 `dequantize` 再查表。

两者都要动 `draft_tokens_gpu` 的入参类型，不是改一行能验完的 —— 留到有预算的一轮做。

## 27. 差别不在"量化"，在 **vocab-parallel 的下标重映射**

`VocabParallelEmbedding`（`layers/distributed.rs:1866+`）：

```rust
pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {   // :1936
    …
    let looked_up = self.weight.index_select(&local, 0)?;    // :1959
```

主干查表也是 `index_select`，但用的是 **`local`** —— 词表按 rank 切分后要先把全局 token id 映到本分片的局部下标。

草稿这条路（`mtp.rs:236` → `draft_tokens_gpu`）拿的是 `embed_weight()` = `self.embed_tokens.weight()`，**没过重映射**，直接拿全局 id 去切本分片那半张表 ⇒ 读到别的分片的行 ⇒ NaN（§25）。

TP2、`vocab_size 248320`、`shard_indices: Some((0, 1))` —— 条件完全成立。

**修法**：`draft_tokens_gpu` 收 `&VocabParallelEmbedding` 并调它的 `forward(tokens)`，而不是收 `Tensor` 再自己 `index_select`。

## 28. §27 坐实 + 剩余改动的确切形状

`distributed.rs:1909-1910`（分片构造）：

```rust
let rows = vocab_size / world_size;      // 248320 / 2 = 124160
let row_offset = rank * rows;            // rank 1 → 124160
```

而 `forward` 用的是 `local = (ids − row_offset) · in_range`（§27）。草稿拿 `embed_weight()` 原样 + 全局 id ⇒ rank 1 上查的是**错的那半张表** ⇒ NaN（§25）。

### 为什么 round 19 没做完

`draft_tokens_gpu` 收的是裸 `Tensor`，而三个 variant 的嵌入是**两种类型**：

| variant | 字段类型 |
|---|---|
| `Qwen3_5` | `VocabParallelEmbedding`（有 `row_offset`/`rows`）|
| `Qwen3_5MoE` | `candle_nn::Embedding`（无分片）|

`match` 的臂要返回同一类型，硬塞会污染契约。

### 剩余改动（小、可验）

`draft_tokens_gpu` 多收一个 `row_offset: usize`，查表处按 §27 那样先减偏移再掩蔽：

```rust
let ids = current_token_t.to_dtype(DType::U32)?;
let off = Tensor::from_slice(&[row_offset as u32], (1,), dev)?;
let local = ids.broadcast_sub(&off)?.broadcast_mul(
    &ids.broadcast_sub(&off)?.broadcast_le(&Tensor::from_slice(&[rows as u32 - 1], (1,), dev)?)?)?;
```

`Qwen3_5` 传 `embed_tokens.row_offset()`，MoE 传 `0`。这是加一个入参，不是改 `Model` 的契约。

## 29. 卡住的东西有名字了，而且是一行

round 23 把统一入口推到只剩一个错：

```
error: lifetime may not live long enough
  --> src/models/qwen3_5.rs:549:9
```

原因：跨分片求和要的聚合器是**借来的** —— `gather_part() -> Option<&AllGather>`，闭包要 `'static` 存不下借用。

而它其实**只差一个 derive**：

| 结构体 | 行 | 派生 |
|---|---|---|
| `AllReduce` | `distributed.rs:426` | `#[derive(Clone)]` ✅ |
| `AllGather` | `distributed.rs:1709-1710` | **只有 `#[allow(dead_code)]`，无 Clone** ❌ |

它的字段是 `Rc<Comm>` / `Arc<OnceLock<..>>` 这一类，全都可克隆 —— 加一行 `#[derive(Clone)]` 后 `self.all_gather.clone()` 就是 owned `'static`，round 23 那套统一入口整套成立。

## 整条链（全部实测）

```
AllGather 不可 Clone → 聚合器只能借 → 统一入口进不了 'static 闭包   ← §29
   ↓ 草稿查表没做跨分片求和
掩成 0 的行没被补回 → norm_embed absmean=0.000                     ← §25/§21
   ↓ argmax(全相等)=0
draft=[0,0,0] → accepted=0                                          ← §19
   ↓
ctx=768  30.3 vs 43.9（目标 ≥43.7）                                  ← §14
```

## 29. 1Cat 怎么做的 —— 读源码定的，不是探针试的

### ① 一步多 token 是**一个融合内核**，不是 draft+verify+回滚三段

`vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:190-201`：

```python
if IS_SPEC_DECODING:
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    if IS_DDTREE:
        i_t = tl.maximum(i_t, 0)
else:
    i_t = 0
```

**接受数就是状态行的下标** —— 内核里直接按它选 state。所以 xinfer 的 `rollback_mamba(seq_id, commit_len)` 这一步在 1Cat **根本不存在**：没有"先按 K+1 写、再回滚到 accepted"，状态选择就是内核里的一次 `tl.load`。

### ② 两个形状

`vllm/v1/attention/backends/mamba_attn.py`

| 行 | 内容 |
|---|---|
| `:147` | `state_indices_tensor_d = empty((decode_cudagraph_max_bs, 1 + num_spec_tokens))` — **每 token 一列** |
| `:156` | `decode_num_accepted_tokens = empty((decode_cudagraph_max_bs,))` |
| `:187` | `num_accepted_tokens = torch.diff(m.query_start_loc)` |
| `:48` | `query_start_loc_d: [num_decodes + 1]` |

序列轴→token 轴不是"展开成 K+1 次调用"，是**第二根轴**：状态索引本身就是 `[batch, 1+K]`，内核按 `(i_n, i_t)` 二维取。

### ③ 对 xinfer 的含义

`mtp.rs` 现在是 draft(5.28) + verify(32.27) + `rollback_mamba` 三段。1Cat 的形态是**一次启动、二维索引、接受数当下选行**。所以 §14 那个"一步 60.9 ms 换 1.97 token"在 1Cat 形态下不存在：没有独立 draft 段，也没有回滚段。

## 30. 二分到底：坏的只有 `norm_embed` 那一路的**输入**

三处 `a0ff64b` 改动全部回退后（`224f2d9`/`dbdbd0a`/`69bcade`），四段探针实测：

| 段 | 实测 |
|---|---|
| `MTP draft in: hidden` | `n=5120 nonzero=5119 nan=0 absmean=1.042e0` ✅ |
| `MTP head norm_hidden` | `n=5120 nan=0 absmean=5.671e-1` ✅ |
| **`MTP head norm_embed`** | `n=5120 nan=5120 absmean=NaN` ❌ |

两个 norm 的构造**完全相同**（同一个 `rms_norm`、同一个 `is_gemma`）。差别只在输入：

```rust
let norm_hidden = self.pre_fc_norm_hidden.forward(backbone_hidden)?;   // 健康
let norm_embed  = self.pre_fc_norm_embedding.forward(token_embedding)?; // 全 NaN
```

`token_embedding = embed_lookup(anchor)`，现在直接委托 `VocabParallelEmbedding::forward`。

## 顺带纠正一条我写错的推断

`eos_token_id = 248044`（检查点 `config.json`）。`target` 里连出的 `248046` **不是 eos**，所以"这一格没有续写"不成立。

## 31. 根因不在草稿头 —— 查表输出本身就带 NaN

本轮把观测打在查表输出上，实测：

```
MTP head token_embed: n=5120 nan=154 absmean=NaN
```

**不是全 NaN，是 5120 个里 154 个。** 这改变了整件事的性质：

| 之前以为 | 实测 |
|---|---|
| `norm_embed` 全 NaN ⇒ 权重没绑上 | 输入 `token_embed` 就带 154 个 NaN |

链条：一行 5120 里带 154 个 NaN → `rms_norm` 的 `mean(x²)` 变 NaN → `norm_embed` **整行** NaN → `cat` 后整段 NaN → `logits` 全 NaN → `argmax(全相等)=0` → `draft=[0,0,0]`。

## 所以 §22/§24/§27/§28 全是下游症状

真正要查的是**嵌入表本地分片里那 154 个元素为什么是未初始化**：`27B` 的 `vocab=248320`，`248320 / 2 = 124160`，nvfp4 按块量化，尾部块不满一个整块时会留下没被写到的元素。

## 33. 一步跑两遍整模型 —— 这才是那处算力浪费

`mtp.rs` 把 `set_linear_is_prefill` 切了两次，各盖一次**整模型前向**：

| 行 | 开关 | 盖住的那次前向 | 形状 |
|---|---|---|---|
| `:21` | `false` | 主干 decode 前向（`:37` `decode_capturer.replay(...)`，走图）| 每序列 **1 行** |
| `:297` / `:504` | `true` | verify 前向 | 每序列 **1+K 行** |

`:24-25` 的注释自己写着：*"stores hidden states in `last_hidden_for_mtp` during both capture and replay"* —— 也就是**第 ① 次产出的 `last_hidden_for_mtp`，第 ② 次又产了一遍**，而 anchor token 本身也是上一步 verify 已经算出来的。

| | 整模型前向次数 | 实测 |
|---|---|---|
| MTP 关 | 1（decode GEMV）| `22.9 ms` → 43.9 |
| MTP 开 | **2**（decode GEMV + prefill GEMM）| `5.28 + 32.27 = 37.55 ms` |

## 1Cat 形态 = 一次

`state_indices [batch, 1+K]` 二维 + `num_accepted = diff(query_start_loc)` 内核里选行：草稿要的是**同一次前向**顺带产出的 hidden，不是再跑一遍主干。

## 34. D17 的准入条件（D16 五段 · case `d2#34`）

### 34.1 判据（怎么算过）
真实 27B × 两卡，`c=1`，`ctx=24` 与 `ctx=768`，单发与并发各一遍；每格跑两次取一致值。
过：MTP 开 ≥ 同格 MTP 关；`draft` 非全零；`accepted > 0`。

### 34.2 现状（实测）
一步 **两次**整模型前向：decode `22.9 ms`（`:21` `is_prefill=false`，走图 replay）+ verify `32.27 ms`（`:297`/`:504` `is_prefill=true`）= `60.9 ms` 换 `1.97 token`。

### 34.3 已排除
| 路径 | 证据 | 登记 |
|---|---|---|
| 验算只覆盖草稿行（删锚点行）| 四格 mtp=1 全 `0.0 tok/s` | `6cf849f` |
| 命名 / 层前缀 / 量化配置三件套 | 全回退后症状不变 | `224f2d9`/`dbdbd0a`/`69bcade` |

### 34.4 准入条件（唯一剩余形态）
`state_indices` 二维 `[batch, 1+K]`（`metadata.rs` 已是 `(1, q_len)`，`eecb110`）+ `num_accepted = diff(cu_seqlens_q)`，由**内核按 `(i_n, i_t)` 选列**。即：改动落在 `attention.rs` 的状态索引消费侧，不在 `src/speculative/`。

### 34.5 阻塞点
`attention.rs` 是 rev 锁定的 fork；按 AGENTS.md 硬约束 1，落点唯一 = `perf/sm70-volta`，需先确认 fork 侧改动如何锁 rev。

## 35. fork 侧机制已存在，引擎侧没接（case `d2#35`）

`gdn_verify_sm70.cu` 已经是 1Cat 形态：

| 行 | 内容 |
|---|---|
| `:33` | `rbuf` 持 **两端 × nmax 序列** 的前/当前态 |
| `:37` | `rn` = 上一轮要回放的 token（**它的接受数**），逐序列 |
| `:41` | `rmap[n] = 2 * slot + side` |
| `:42` | 本轮写 `side`，回放 `side ^ 1`；共享轮次计数 `ctr` 选端 |

即"接受数就是内核读的那一列"在 fork 里已实现。`grep -ar replay_slots|rmap|round_counter` 在 `src/speculative/` **0 处命中** —— 引擎侧从没传过 `rn`/`rmap`/`ctr`。

## 36. `nacc` 接上后的语义差（case `d2#36`）

`results.jsonl` 第 6 行实测：

```json
{"error": "", "wall_s": 0.297, "round": 1, "mtp": true, "ctx_tokens": 768, "concurrency": 1, "out_tokens": 256, "k": 3, "mml": 1088}
```

`error` 是**空串**、`wall_s=0.297` —— 请求跑完了但没出 token，所以没有 `per_req_tok_per_s`。不是崩，是选错列。

内核 `gdn_verify_sm70.cu:95`：

```c
p.idx[n * p.s_idx + (REPLAY ? 0 : (p.nacc ? p.nacc[n] - 1 : 0))]
```

| | `diff(query_start_loc)` 跨的是什么 |
|---|---|
| 1Cat | **已接受**的 token 数 ⇒ `−1` 正好指到最后那个被接受的 token |
| xinfer 现在 | `cu_seqlens_q` 跨的是**被提议**的 `1+K` 个 ⇒ `nacc[n]-1` 恒等于 `K=3`，与接受与否无关 |

所以 `num_accepted_tokens()` 现在返回的是"这步带几个 token"，不是"接受了几个"。要它成立，`cu_seqlens_q` 必须按已接受数推进 —— 那是引擎步进的活，不是内核的。

## 37. 保留数这条路走不通的原因（case `d2#37`）

想按"逐步保留数"填 `seqlens`，但 `SpecSeqInfo` 只有 `id` / `len` / `block_table`（`metadata.rs:7-10`），**没有保留数字段**，已回退成 `None`。

更重要的是重读内核那行后发现的：`s_idx` 改成行宽后，回退分支就已经是对的了 ——

```c
idx[n * s_idx + (nacc ? nacc[n] - 1 : 0)]
```

`s_idx = q_len`、`nacc[n] = q_len` ⇒ `nacc[n] - 1 = q_len - 1` = **最后一列**，正是"从本步最后一个 token 的状态续起"。

所以 `mtp=1` 仍 `RemoteDisconnected` 的原因**不在 `nacc` 语义**，而在别处。本轮把这条排除掉。

## 38. 快照缓冲其实已是 token 索引的（case `d2#38`）

`deltanet.rs:1007` 分配：

```rust
Tensor::zeros((max_verify_tokens, num_v_heads, head_k_dim, head_v_dim), DType::F32, ..)
```

`:1177` 取用 `.narrow(0, 0, token_count)`。所以 `snaps` 的第 0 维**本来就是 token 数**，不是逐序列一行 —— 我上一轮"snaps 缺 token 轴那一维"的判断**错了**，撤回。

三维里已对上的：`s_idx = q_len`（`fe9474d`）、`slot_mapping (1, q_len)`（`545a006`）、`snaps` 第 0 维 = token（本就如此）。

剩下一处：`s_state` —— 状态槽位的步长。`idx` 存的现在是**逐 token 槽位**，而 `s_state` 仍按逐序列行算，内核按它跳行就会读到没分配过的行 → `CUDA_ERROR_ILLEGAL_ADDRESS`。

## 39. 第一现场（`CUDA_LAUNCH_BLOCKING=1` 跑出来的）

之前看到的 `ILLEGAL_ADDRESS` 是粘住的回声。逐次同步跑，真错是：

```
[slow]  gdn_fast_prefill — the chunked GDN prefill kernel requires SM90
```

| | |
|---|---|
| 这台机 | **SM70（Volta）** |
| `is_prefill = true` 让 GDN 层去要 | `gdn_fast_prefill`，**要求 SM90** |
| 而 SM70 上该走的是 | `gdn_verify_sm70`（fork 里就有 `gdn_verify_sm70.cu`）|

所以 MTP 开在这台机上根本不是"慢"，是**路由错了内核**：verify 带着 `is_prefill=true`，落到了要求 SM90 的 chunked prefill 上。

`deltanet.rs` 里 SM90 那条分支的判定在 `is_mtp_verify` 分支**之后**，但 `gdn_fast_prefill` 这条没有同样的护栏 —— 这是要修的点。

## 40. 更正 §39：那条不是错，是诊断字段

`sm70_diagnostics.rs:251-257`：

```rust
// The chunked GDN prefill kernel is SM90-only.
let gdn_fast_prefill = cfg!(feature = "flashinfer")
    && compute_capability == Some(90)
    && crate::utils::env::sm90_lower_precision_gdn_prefill();
let gdn_prefill_reason: Option<&'static str> = if gdn_fast_prefill {
    None
} else if compute_capability != Some(90) {
    Some("the chunked GDN prefill kernel requires SM90")
```

`[slow] gdn_fast_prefill — …` 是诊断输出里的 `gdn_prefill_reason` 字段，**不是异常**。SM70 上快速路常关是设计如此。

所以 §39 的"路由错了内核"结论**撤回**：verify 没落到 SM90 内核，它本来就该走 `gdn_verify_sm70`。`CUDA_LAUNCH_BLOCKING=1` 下没再出现越界，说明越界发生在异步流上、且**第一现场已被粘住的上下文吞掉**。

## 41. 越界是竞态，不是步长（case `d2#41`）

`CUDA_LAUNCH_BLOCKING=1` 那一跑：**越界条数 = 0**，且没有 `Forward error`。八格全跑通：

| | MTP 关 | MTP 开 |
|---|---|---|
| ctx=24 | 16.31 / 16.86 | 13.43 / 13.25 |
| ctx=768 | 17.14 / 17.04 | 13.15 / 13.16 |

（同步跑把发射串行化，绝对值减半，只看两边相对关系。）

## 两条结论

1. **异步下崩、同步下不崩 ⇒ 是竞态**，不是 `s_idx`/`s_state` 这类步长。某处异步算子在快照行的有效窗口外读写它。
2. **MTP 开仍比关慢**（`13.4 < 16.3`、`13.2 < 17.1`）—— 这与 §33 一致：一步两遍整模型，接受率修好也赚不回来。

## 44. D17 真正的落点：整条 replay 路没被启用（case `d2#44`）

`gdn.rs:2644-2650` 连着 **7 个空参**，全是同一条路的：

```rust
std::ptr::null(), // rbuf: non-replay route
std::ptr::null(),
std::ptr::null_mut(),
0,
std::ptr::null(),
std::ptr::null(),
std::ptr::null(),
```

内核侧 `:342` `const bool replay = rbuf != nullptr;` ⇒ `:375` `if (replay) go2(wr, std::true_type{})`。所以引擎一直走的是 `REPLAY=false` 那条。

两条路的差别就在 §44 之前所有轮次想改的那几行：

| | `REPLAY=false`（现在）| `REPLAY=true`（1Cat）|
|---|---|---|
| 取哪行 | `idx[n*s_idx + 0]`，逐序列一行 | `rmap[n] = 2*slot + side` |
| 接受数 | 不参与 | `R0 = min(rn[n], TMAX)`，`:134` 逐 token 回放 |
| 端 | 无 | 本轮写 `side`，回放 `side ^ 1` |

**这才是 D17 的活**：给验算配 `rbuf`（两端 × `nmax`）、`rn`、`rmap`、`ctr`，让它走 `REPLAY=true`。调用侧换 `nacc`/`s_idx` 是隔靴搔痒 —— 已在 `a12d175` 回退并登记。

## 47. replay 路跑通后，`rn` 的取值差（case `d2#47`）

内核 `:130-142` 把这条写得很死：

```c
const float* prv = REPLAY ? p.rbuf + ((long long)(side ^ 1u) * p.nmax + rs) * SEQ : nullptr;
if constexpr (REPLAY) {
  for (int t = warp; t < R0; t += WARPS) {   // R0 = min(rn[n], TMAX)
    ...  // 把上一轮的 k / v 从 rbuf 搬进 smem 再推一遍
```

| | `rn[n]` 要的是 | 现在传的是 |
|---|---|---|
| 值 | **上一轮已提交**的 token 数（`verify.rs:12 num_accepted` + 1）| `diff(cu_seqlens)` = **本步被提议**的 `1+K` |
| 差 | 接受 < 提议时，多推的几行把状态推过头 | — |

引擎侧要把它送到层里，`SpecSeqInfo` 加字段那条已证伪（四处构造处开档形状不一，`03aad93`）。剩余做法二选一：

1. `DecodeSequence` 已有 `last_block_tokens`（本步保留数），由 `prepare_decode` 填 —— 复用它，不加新字段；
2. 让 `build_verify_metadata` 收一个 `prev_committed: usize` 形参，由 `mtp.rs` 从 `verify_result` 带过来。

## 48. 真正的问题：MTP 路径的 logits 从第一步就是 NaN（case `d2#48`）

### 48.1 判据（怎么算过）
真实 27B（`unsloth/Qwen3.8-27B-NVFP4`，HF 缓存里那份，不下载）× 两卡（物理 4/5 → 0/1），`c=1`，
`ctx=24` 与 `ctx=768`，每格起一台独立服务，单发一次、客户端墙钟计吞吐。
过：**两条路径的 logits 都必须是有限值**（`nonzero = vocab`），生成内容不是恒定 token 0。

### 48.2 现状（实测，日志 `srv_mtp{0,1}_mml{344,1088}_seqs1.log`）

**(1) logits 诊断（`XINFER_DIAG_LOGITS=1`，同一提示、ctx=24）**

| 路径 | `logits dims` min / max / mean | `logits vocab` nonzero |
|---|---|---|
| `mtp=0`（普通解码）| `-11.48 / 20.69 / -1.44` | `248320` ✅ |
| `mtp=1`（MTP 路径）| `inf / -inf / 0` | **`0`** ❌ 全 NaN |

**(2) 实际吐出的 token（`sampled token ids`，两个 rank 各打一遍）**

| 路径 | 前 12 个采样 |
|---|---|
| `mtp=0` | `3793, 3793, 248046, 248046, 13114, 13114, 0, 0, 0, 0, 0, 0` ⇒ **第 4 步起塌成 token 0** |
| `mtp=1` | `0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0` ⇒ **第 1 步就塌** |

草稿侧同证：`ctx=24` 的 `MTP accept` **258/258 步** `proposed=3 accepted=0 draft=[0,0,0]`；
`ctx=768` 末尾几步草稿是正常 token（`[11,321,32288]`、`[2179,332,248046]`），`accepted=1..2`。

**(3) 吞吐（客户端墙钟）**

| ctx | mtp | tokens | 墙钟 | tok/s | 说明 |
|---|---|---|---|---|---|
| 24 | 0 | 256 | 6.30 s | 42.7 | 跑满 |
| 24 | 1 | 256 | 4.96 s | 56.9 | 跑满（**但内容是垃圾**）|
| 768 | 0 | 256 | 6.51 s | 43.9 | 跑满 |
| 768 | 1 | ≈45 | **1.57 s** | 28.4 | **第 20 个 token 吐 `248044`（eos）→ 序列提前结束** |

⇒ 所谓"长上下文下 MTP 更慢（21.6 t/s）"这一格，**测的是 20 个 token 的残跑**，
拿它跟 256 token 的整跑比。客户端口径本身也有缺陷（`mtp_api_perf_probe.py:124`
`chunks = sum(len(st) ...)` 数的是 **SSE 非空增量条数**，不是 token 数）。

**(4) 检查点是干净的**（扫了索引里全部 1878 个张量的原始字节）

| dtype | 张量数 | NaN |
|---|---|---|
| BF16 | 1063 | **0** |
| F32 | 336 | **0** |
| F8_E4M3 | 401 | **0 字节**（含 `lm_head.weight`）|
| U8（e2m1 packed）| 168 | 命中的 `0x7f/0xff` 是 4bit 码的巧合，e2m1 没有 NaN 编码 |

⇒ **NaN 不是权重带来的，是运行时产生的。**

### 48.3 已排除

| 路径 | 证据 |
|---|---|
| 检查点里有 NaN 权重 | 全量扫字节：BF16/F32/e4m3 全 0（48.2(4)）|
| 探针统计口径错 | 口径确实错（48.2(3)），但改口径不改变"长上下文那格只有 20 个 token"这个事实 |
| `--ignore-eos` 没生效导致提前结束 | `scheduler.rs:620` 的 EOS 判定本来就看 `ignore_eos`；命中的是**服务端把 eos 塞进 `stop_token_ids`**（`server.rs:396-398`）后走的 `scheduler.rs:1459` 那条路，那条路不看 `ignore_eos` ⇒ 本次已补判断（见 48.4），但它不是主因：主因是模型吐垃圾 |
| D17 的 replay 路（`df556ab` / `8db927d`）| 回退。`rn` 传的是本步提议数 `1+K`，不是上一步已提交数（§47），`R0` 会把状态推过头；实测四格全 `0.0`（`MTP runner returned empty response`）。回退 sha 已登记到 `docs/design/sm70_1cat_rejected_paths.md` |

### 48.4 真正的落点（三条，按优先级）

1. **MTP 路径的 logits 从第 1 步就 NaN，而普通路径前 3 步正常** ⇒ 差异只可能出在
   "MTP 参与前向"的那一刻：**MTP 头前向 / 验算前向污染了共享 hidden 或线性注意力状态**。
   这正是 D17 那条路要解决的东西，但它的优先级排在 NaN 之后 —— NaN 不修，测出来的任何吞吐都是垃圾上的吞吐。
2. **普通路径第 4 步起也全 NaN** ⇒ 该模型/配置在 xinfer 上的前向本身有数值问题
   （GDN 状态逐步累积？fp8 反量化？），与 MTP 无关，但同样会让所有对比失真。
3. **验算前向 258/258 步 `graph=false`**（eager，`30–38 ms/步`）：整轮测量都在
   `--disable-cuda-graph` 下跑，因为探针自己注明"真模型上 MTP+开图在捕获期 OOM"（`mtp_api_perf_probe.py:36`）。

顺带修掉的一处口径：`scheduler.rs:1459` 的停止串判定现在看 `ignore_eos`
（服务端注入的 eos id 不该在 `--ignore-eos` 下终止序列）。

### 48.5 阻塞点

- 定位 1 需要能在真模型上**单独**跑"主干前向"与"主干+MTP 头前向"并比对 logits；
  现有 `mtp_probe`（`qwen3_5_mtp.rs:464`）只在 MTP 路径里打点，主干侧没有对应打点。
- 定位 2 需要 GDN 状态**逐层**的 NaN 计数（`recurrent_state`），现在只有 `token_embed`/`norm_embed` 一层的痕迹（§32）。
- 测量口径：探针的 token 计数要改成真 token 数（服务端回报 `output_len`），否则长上下文那格永远会被残跑污染。

---

## 49. 逐层探针（非扰动版）定位 NaN 的产生位置

### 49.1 判据

要在真模型（`unsloth/Qwen3.8-27B-NVFP4`，2 卡）上把"第几个前向、第几层、哪个张量"
第一次出现非有限值说清楚，且**探针本身不得改变被测对象的行为**。

### 49.2 现状实测

**探针形态的修正**：第一版探针在每层之后把 hidden 拉回主机（`to_vec1`），
等于每层一次同步；实测它把 **prefill 从"有限"变成"第 3 层起全 inf"** ——
即探针本身扰动了时序敏感的 attention 路径（见 49.3）。改为
"每层只在 GPU 上算标量（非有限值计数、最大绝对值），整个前向末尾一次性读回"，
前向内只剩一次同步，结论才可信。

**逐层结果**（`XINFER_DIAG_NAN=1`，`[nan] per-layer` 行；65 = 1 个 embedding + 64 层）：

| 前向 | 普通路径（mtp=0） | MTP 路径（mtp=1） |
|---|---|---|
| prefill | 全层 0 个非有限值，最大幅值 0.55 → 1.13e4 | 同左 |
| 第 1 个 decode | **首个非有限层 = 索引 4（解码器第 3 层，attn 层）** | **首个非有限层 = 索引 48（解码器第 47 层，attn 层）** |

⇒ prefill 干净、**第一个 decode 步就坏**；两条路径坏在不同的深度，
但都是全注意力层 ⇒ 指向"decode 步读到的 KV/状态"而不是权重或 embedding。

**GDN 状态**（`[nan] gdn state` 行，逐层 conv/rec 状态）：

| 时点 | conv 状态 | rec 状态 |
|---|---|---|
| prefill 后 | 全部有限，幅值 0.1~70 | 全部有限 |
| 第 1 个 decode 后 | **GDN 第 3 层起整片 inf（5120 个 = 全部）** | 仍有限 |
| 第 2 个 decode 后 | 同上 | **GDN 第 3 层起 −inf** |

⇒ 状态是被"已经变成非有限的 hidden"污染的结果，不是源头（源头在 GDN 之前）。

**KV cache**（`[nan] kv cache blk0` 行，逐层 block 0）：

| 层 | K cache | V cache |
|---|---|---|
| decode 后 | **含 NaN（计数 3840/512/4352/4864…，均为 256 的整数倍 = 逐 head-slot 粒度）**，最大幅值有限（5~9） | **最大幅值恒为 512.0**（每层、每次前向都一样）|

⇒ V cache 里没有真实的激活值（恒定 512.0 不可能是 V），
K cache 里有 NaN ⇒ **cache 的"写"或"读"与分配形状不一致**。

**读侧不是元凶**：`XINFER_SKIP_FLASHINFER=1`（绕开 FlashInfer decode，
改走原生 `paged_attention`）后，普通路径的 decode **同样**在第 3 层起变 inf
（吞吐 43 → 17 t/s，但 NaN 依旧）⇒ 两个读内核都读到坏数据 ⇒ 坏的是 cache 内容。

**MTP 路径的一个独立问题**：`mtp_decode_step1`（1 token 主干前向）与验算前向
（K+1 token）各自把线性注意力状态往前推 ⇒ 状态每步前进 K+2 个 token，
而不是"已提交数"；状态与 KV cache 因此失配。

### 49.3 已排除

| 路径 | 证据 |
|---|---|
| FlashInfer decode 内核读 cache 的方式错 | `XINFER_SKIP_FLASHINFER=1` 换原生读，NaN 依旧（49.2）|
| GDN 状态在 prefill 里就溢出 | prefill 后逐层 conv/rec 状态全部有限、幅值 0.1~70（49.2）|
| 探针报告的"prefill 第 3 层就 NaN" | 那是每层同步的探针自己扰动出来的；换成单次同步后 prefill 全干净（49.2）|
| 权重/embedding 带 NaN | 全量扫字节：BF16/F32/e4m3 全 0（§48.2(4)）|

### 49.4 真正的落点

1. **KV cache 的内容被写坏**（首要）：分配形状是 `(num_blocks, block_size, kv_heads, hd)`
   （`kvcache_allocator.rs:1891`），写走 `ReshapeCache`（`paged_attention.rs:1352`），
   读分别走 FlashInfer decode 与原生 `paged_attention`。V cache 恒为 512.0、K cache 带 NaN
   ⇒ **要么 V 的写根本没落到 cache 上，要么 K 的写落到了 V 的区域**。
   下一步要一个"写完立刻回读单个 layer 的 K/V"的直读探针。
2. **MTP 路径的状态推进量错误**（次要，但独立）：
   `mtp_decode_step1` + 验算前向让 GDN 状态每步前进 `K+2`，而不是已提交数；
   1Cat 形式（每步只有一次主干前向）天然消除这次多推进。

### 49.5 阻塞点

- 定位 1 需要"KV 写入后立刻回读"的探针（写内核与读内核都不动，只加回读）。
- 定位 2 需要把 GDN 状态推进量与"已提交 token 数"对齐：要么走内核 REPLAY 路
  （§47 证伪过：`rn` 传错），要么在 1Cat 形式下让主干前向只跑一次。


## 50. 1Cat 形式真正落地并生效（case `d2#50`）

### 50.1 判据

单序列 MTP 每步只跑 **一次** 主干前向（verify 的 `[anchor, d1..dK]`），
anchor 与其 hidden 行由上一步验算带过来；日志里必须看到 `MTP 1Cat: reuse anchor`，
且 A/B 表要给出真实 token 数（服务端 `Decoded:` 行），不是 SSE 分片数（D12）。

### 50.2 为什么之前一直没生效（三处，全修）

| # | 缺陷 | 证据 | 修法 |
|---|---|---|---|
| 1 | 验算分支拿不到 hidden | `ctx store skipped: verify hidden unavailable (variant=Qwen3VL)` —— 27B 走的是 `Qwen3VL` 变体，该分支用 `forward`（只回 logits） | 改 `forward_with_hidden`，与另两个变体一致 |
| 2 | 期望长度用错语义 | `ctx lookup: len=38 expect=None` —— `next_len` 用 legacy 的 `len+num_accepted+2`，而 1Cat 步只提交 `num_accepted+1` | `+ if use_1cat {1} else {2}` |
| 3 | anchor 被当新 token 重复提交 | 返回 vec 含 `anchor_token`，而引擎把返回的全部 token 追加进序列 | 1Cat 路不再 push anchor |

### 50.3 A/B（真实 token 数）

条件：`unsloth/Qwen3.8-27B-NVFP4`（HF 缓存，未下载）、2 卡 TP=2、`--disable-cuda-graph`、
K=3、prompt 37 token、生成 64 token、`temperature=0`、`XINFER_DIAG_MTP=1`、`XINFER_DIAG_NAN=0`
（探针自身扰动时序敏感路径，§49.2）。

| 路 | 服务端自报 | 比值 |
|---|---|---|
| mtp=0 | 64 token / 1.51 s = **42.38 t/s** | 1.00 |
| mtp=1（1Cat 生效，reuse=30） | 64 token / 0.86 s = **74.77 t/s** | **1.76** |

分步计时（同次运行）：`step1 36.37 ms`（现在只在序列第一步付一次）、`draft 3 tok 12.49 ms`、
`verify m=4 32.39 ms`。legacy 路每步都要付 step1，1Cat 把它从每步成本里消掉了。

⚠ 这个 1.76× 还不是可引用的加速比：接受率虚高（见 50.4，全部 token=0 ⇒ 3/3 全采），
真实模型上 K=3 的接受率会低得多。

### 50.4 输出质量的真正落点（取代 §49.4 的首要项）

`XINFER_DIAG_NAN=1` 逐阶段幅值（第 0 层，prefill，37 token）：

```
embed 5.469e-1 → in_norm 4.816e1 → qkv_proj 6.053e1 → recur 3.984e0
      → gated 2.306e-1 → out_proj 1.024e3
mlp_in 6.879e0 → mlp_out 8.962e1
```

逐层残差流 maxabs 单调爬到 `1.2e4`（65 项全有限，`first_bad=None`）。
⇒ **不是 NaN，是幅值**：每层子模块相对其（已归一的）输入放大了 10~4000 倍，
残差流被最后几层支配，logits 的 argmax 恒为 token 0。
`gdn0 gated`（0.23）与 `out_proj`（×4400）是最大的一段，MLP 段同样放大 ×13。

§49.4 的“KV cache 内容被写坏”仍是待查项（V cache 恒 512.0、K cache 带 NaN 仍待复现），
但本次证据把首要异常定在 **子模块输出相对输入的放大比**，不是 cache 读法。

## 51. 残差流爆炸的两段定量落点（case `d2#51`）

50.4 给出链路后，本节把“放大比”落到具体权重上（`XINFER_DIAG_NAN=1`，第 0 层，prefill 37 token）：

| 量 | 实测 maxabs | 相对输入的正确比 |
|---|---|---|
| `in_norm` 输出（GDN 的输入） | 4.816e1 | — |
| `in_proj_qkv` 输出 | 6.053e1 | ≈1.25 ✓ |
| `in_proj_z` 输出（输出门控） | **7.990e-2** | 应 ≈1（比 qkv 小 **~750×**）|
| `gdn_norm_weight` | 9.297e-1 | ≈1 ✓ |
| 递推输出 | 3.984e0 | 应 ≈ v_conv（≈6e1），**小 15×** |
| 门控后 | 2.306e-1 | — |
| `out_proj` 输出 | **1.024e3** | 相对门控 **×4400** |

⇒ 不是“全都错一个常数”，而是**两条独立的权重装载错误**：
① `in_proj_z`（输出门控）比 `in_proj_qkv` 小 ~750× —— 同一份 `in_proj` 融合权重里
只有 z 段错了，指向 z 段的 nvfp4 全局标度/分片偏移没落到正确位置；
② `out_proj` 大 ~10³ 量级。两者相乘正好把门控后的 ~0.2 变成残差上的 ~10³，
于是 64 层累加到 1.2e4，logits argmax 恒为 token 0（= 50.4 的“输出全 0”）。

下一步（新会话直接照做）：对 `unsloth/Qwen3.8-27B-NVFP4` 的
`layers.0.linear_attn.in_proj_z.weight` 与 `out_proj.weight` 直接做 fp4→fp32 反量化求 maxabs，
与引擎里 `self.z`/`out_proj` 拿到的张量逐段比对，定位是全局标度没乘、还是分片/段偏移错了。

## 52. 检查点/配置侧事实与 §51 的修正（case `d2#52`）

**配置（`config.json`）**：`linear_num_key_heads=16`、`linear_key_head_dim=128` ⇒ `key_dim=2048`；
`linear_num_value_heads=48`、`linear_value_head_dim=128` ⇒ `value_dim=6144`；
`2×2048+6144=10240` = `conv1d`/`in_proj_qkv` 行数 ✓。

**检查点（`model.safetensors` 头，第 0 层）**：`in_proj_qkv` F8_E4M3 `[10240,5120]` + `weight_scale [10240,1]`；
`in_proj_z` `[6144,5120]`；`out_proj` `[5120,6144]`；`norm.weight` BF16 `[128]`；
`linear_attn` 只有逐行 `weight_scale`，**没有** `weight_global_scale`/`input_global_scale`（MLP 两者都有）。

**证伪**：§51 之后我一度推断“引擎把 `value_dim` 取成 3072 = 6144/2”。错——
探针看到的 3072 是 **TP=2 的分片**（24 个 v-head × 128），不是半读。该推断作废，不得重试。

**剩下的真异常（未决）**：同一层里 `in_proj_qkv` 的有效增益 ≈1.25（正常），
而 `in_proj_z` 的有效增益 ≈0.0017（`gate:z` maxabs 7.99e-2 对 `in_norm` 4.82e1），
`out_proj` 又是 ×4400（`gated` 2.31e-1 → `out_proj` 1.02e3）。
首要嫌疑：逐行 `weight_scale` 在 z / out_proj 这两段上的施加方式（是否被当成全局标度、
或分片后行偏移错位）。下一步按此对第 0 层做一次“反量化后逐段 maxabs”直读比对。

## 53. 直读检查点：三段投影的原始码与逐行标度（case `d2#53`）

`tmp/w53.py` 直读 `model.safetensors`（第 0 层，`linear_attn`）：

| 段 | 权重 dtype/形状 | `|w|`原始 max | 标度形状 | `|s|` max / min | 反量化后 max≈`|w|·|s|` |
|---|---|---|---|---|---|
| `in_proj_qkv` | F8_E4M3 `[10240,5120]` | 1.28e2 | `[10240,1]` | 9.73e-4 / 4.41e-5 | ≈1.2e-1 |
| `in_proj_z` | F8_E4M3 `[6144,5120]` | 1.28e2 | `[6144,1]` | 5.30e-4 / 5.36e-5 | ≈6.8e-2 |
| `out_proj` | F8_E4M3 `[5120,6144]` | 1.28e2 | `[5120,1]` | 3.02e-3 / 1.02e-4 | ≈3.9e-1 |

要点：
1. 这是 **F8_E4M3 + 逐行标度**（`[N,1]`），不是 NVFP4 的 `[N, K/16]` 分块标度；
   `swizzle_nvfp4_weight_scales` 期望的正是 `[N, K/16]`，两者形状不同源，别混用。
2. 反量化后三段权重的幅值同量级（0.07~0.39），彼此只差几倍；
   而引擎侧实测的“输出/输入”增益是 qkv ≈1.25、z ≈1.7e-3、out_proj ≈4.4e3（§50.4/§51）。
   ⇒ 差异不可能来自权重本身，只能来自**标度有没有乘、按什么轴乘**。
3. 下一步（唯一落点）：对这三段各做一次位等同比对 ——
   `dequant(原始码, 逐行标度)` 的 maxabs/行号 vs 引擎实际持有的张量，
   先判“标度没乘”还是“按错轴乘（逐行被当成全局、或 TP 切分后行偏移错位）”。

## 54. 引擎实际持有的 (weight, scale) 对（case `d2#54`）

`diag_proj_pair` 走 `Linear::fp8_weight_scale()`，第 0 层 `in_proj_z`（TP=2 分片后 `w_dims=[5120,3072]`）：

| 量 | 引擎实测 | 检查点真值（`tmp/w54.py` 直读） |
|---|---|---|
| 权重 absmax | **254.0**（原始码，未反量化） | 原始 e4m3 absmax 4.48e2 |
| 标度 absmax | **1.357e-1**（另一次 1.226e-1） | 逐行标度 absmax **5.30e-4** |
| 二者之积 | 254 × 1.357e-1 = **3.45e1** | 反量化后 absmax **2.376e-1** |

要点：
1. 引擎里权重是**原始码**，标度单独存，由 GEMM 内核施加。
2. 引擎拿到的标度是检查点逐行标度的 **~256×**（1.357e-1 / 5.30e-4），
   于是有效权重大了 **~145×**（3.45e1 / 2.376e-1）。
3. 顺带证伪：`Tensor::abs()` 直接作用在 U8/F8E4M3 上会在 SM70 报
   `CUDA_ERROR_NOT_FOUND: named symbol not found (uabs_u8)`——诊断必须先 `to_dtype(F32)` 再 `abs()`。

下一步：查 `Linear::fp8_weight_scale()` 的标度是从哪个键读的、
是否把检查点的逐行标度又乘了一遍 `weight_global_scale`，或把 `[N,1]` 当成了 `[N, K/16]` 分块标度。

## 55. 位等同比对：引擎的逐行标度不是检查点那个（case `d2#55`）

`in_proj_z`（TP rank0，`w_dims=[5120,3072]`，即 `[K,N_shard]`）前 4 个逐行标度：

| 输出行 | 检查点 `weight_scale` | 引擎 `fp8_weight_scale()` 的 `s` | 比值 |
|---|---|---|---|
| 0 | 1.4299e-4 | 3.0884e-2 | 216.0 |
| 1 | 1.6600e-4 | 5.3955e-2 | **325.0** |
| 2 | 1.3600e-4 | 3.6377e-2 | 267.5 |
| 3 | 1.4000e-4 | 2.9663e-2 | 211.9 |

**比值不是常数**（216/325/268/212）⇒ 不是"少乘/多乘一个全局标度"，
而是**引擎拿到的根本不是检查点的 `weight_scale` 这个张量**（或它是别的量化参数的派生量）。
权重侧同理：引擎 `w_absmax=254`，检查点原始码 absmax=448，也不是同一份码的同一视图。

⇒ §54 的"×256"是表象；真相是**加载时取错了键/取错了张量**。
下一步查 `Linear::fp8_weight_scale()` 的读取路径：它读的是 `weight_scale` 还是 `weight_scale_inv`/别的派生量，
以及 merged 变体下 `in_proj_z` 是否被并进了别的段。

## 56. 对齐仍未闭合：需要引擎侧原始前 4 个码与 block 配置（case `d2#56`）

§55 的两个候选对齐都排除了：

| 对齐假设 | 检查点值（前 4） | 引擎 `s`（前 4） | 比值 |
|---|---|---|---|
| 逐行标度本身 | 1.4299e-4, 1.6600e-4, 1.3600e-4, 1.4000e-4 | 3.0884e-2, 5.3955e-2, 3.6377e-2, 2.9663e-2 | 216/325/268/212（非常数）|
| 反量化后逐行 maxabs | 6.4087e-2, 7.4341e-2, 6.1096e-2, 6.2805e-2 | 同上 | 0.482/0.726/0.595/0.472（非常数）|

另：检查点原始码**逐行 maxabs 恒为 448.0**（e4m3 饱和值）⇒ 检查点确实是"饱和码 + 逐行反量化标度"。
引擎 `w_absmax=254.0` ≠ 448 ⇒ 引擎的 `w` 也不是那份码的原样视图。

⇒ 结论收紧为：**`LnFp8` 里这对 (weight, weight_scale) 与检查点那份不是同一表示**，
且 `LnFp8` 还有 `weight_scale_cutlass` 与 `weight_block_size`——GEMM 实际用哪个尚未确认。

下一轮只需两件事（都很小）：
1. 打印引擎 `w` 的**前 4 个码值**（不是 maxabs）与 `weight_block_size`、以及 GEMM 走的是 `weight_scale` 还是 `weight_scale_cutlass`；
2. 与检查点 `[6144,5120]` 的码按 `[K,N_shard]` 转置后的前 4 个逐元素比。位等即证明"只是转置视图"，
   不等即证明"被重新量化过/取错张量"。

## 57. 位等比对闭合：形状自洽但**乘积大 ~10³**（case `d2#57`）

量化配置（`config.json.quantization_config`，`compressed-tensors` / `mixed-precision`）：

| 组 | 覆盖 | `weights.strategy` | `group_size` | `scale_dtype` |
|---|---|---|---|---|
| group_0 | `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`、`self_attn.*` | **`channel`** | `None` | `None` |
| group_1 | `mlp.{gate,up,down}_proj` | `tensor_group` | `16` | `torch.float8_e4m3fn` |
| ignore | `re:mtp\..*`、`lm_head`、`linear_attn.norm/_b/_a` | — | — | — |

⇒ 投影段是**逐通道 fp8**（`[N,1]`），MLP 才是 16 分块；MTP 层在 ignore 里（bf16，不受本问题影响）。

`in_proj_z` 两侧对齐（引擎 `[K,N_shard]=[5120,3072]`，`s_dims=[1,3072]` ✓ 形状自洽）：

| 量 | 引擎 | 检查点 |
|---|---|---|
| 形状 | `w=[5120,3072]` `s=[1,3072]` | `W=[6144,5120]` `S=[6144,1]` |
| 前 4 个码（解码后） | `242, 110, 117, 108` | 列0 视图 `56, -36, 56, 28` |
| 码 absmax | `254` | `448`（e4m3 饱和） |
| 前 4 个标度 | `3.0884e-2, 5.3955e-2, 3.6377e-2, 2.9663e-2` | `1.4299e-4, 1.6600e-4, 1.3600e-4, 1.4000e-4` |
| 标度 absmax | `1.3574e-1` | `5.3024e-4` |
| **乘积** | `242×3.0884e-2 = 7.47` | `56×1.4299e-4 = 8.01e-3` |

**乘积差 ≈ 9.3e2（~10³）**，且码与标度各自都不同 ⇒ 引擎这对不是检查点那份的原样视图。
前 4 个元素对不上也可能只是 TP 分片/头交织的索引置换，需按 `undo_tiled_v_heads_first_dim`
的反向映射还原后再比。

两个仍未排除的根因：
1. 加载期被**重新量化**过（自算标度），且新标度算错量级；
2. GEMM 侧把标度当**逆标度**用（`linear.rs:1163-1169` 会回退到 `weight_scale_inv` 这个键名，
   说明本仓同时存在两种约定）。

下一轮：按分片索引置换还原引擎前 4 列对应的全局行号，再逐元素比；同时确认 GEMM 是乘还是除。

## 58. 根因定量闭合：引擎的有效权重比检查点大 **~150×**（case `d2#58`）

免索引映射的判据 —— 反量化后**逐输出通道 maxabs 的分布**（`Self::deq_channel_stats`）：

| 来源 | min | mean | max |
|---|---|---|---|
| 引擎 `LnFp8`（`w=[5120,3072]`, `s=[1,3072]`） | **3.4882** | **1.0270e1** | **3.4207e1** |
| 检查点直读（`tmp/w54.py`） | 2.4033e-2 | 6.8281e-2 | 2.3755e-1 |
| **比值** | 145.1 | **150.4** | 144.0 |

⇒ 引擎里 `in_proj_z` 的**有效权重整体大了 ~150×**（比值在 min/mean/max 上一致到 ±4%，
说明是单一量级错误，不是索引置换）。这就是 §50.4 残差流逐层单调爬到 `1.2e4` 的来源。

量级来源：引擎 `s_absmax=1.3574e-1` vs 检查点 `5.3024e-4`（×256），
而码侧 `w_absmax=254` vs 检查点饱和码 `448`（×0.567）；`256 × 0.567 ≈ 145` ✓ 与上表一致。

⇒ 待判只剩一步：确认加载期是否把检查点码**重新量化**成了另一套 (码, 标度)，
且新标度按 `amax` 之外的口径算（例如把 `channel` 当成 `tensor_group` 的组内标度、
或 `strategy` 没读出来时落到了 `unwrap_or(vec![128,128])` 的默认分块，见 `linear.rs:1131-1134`）。

## 59. 修复落点：`QuantConfig` 从不解析 `strategy`（case `d2#59`）

`src/utils/config.rs` 里 `strategy` 只出现在文档注释的示例串里（`"strategy": "group"`），
**没有任何一处被反序列化成字段**；量化块信息只从顶层 `weight_block_size: Option<Vec<usize>>` 读。

而本检查点把块信息放在 `config_groups.group_0.weights`：
`{strategy: "channel", block_structure: null, group_size: null, scale_dtype: null, type: "float", symmetric: true}`。

⇒ 走的是 `linear.rs:1131-1134` 的
`quant_cfg.weight_block_size.clone().unwrap_or(vec![128,128])` 默认分块，
`strategy: "channel"` 这个"每输出通道一个标度"的事实**从未被读进来**。
`is_channel_scale_shape` 那条补救只在形状恰好匹配时才生效，且拿到的仍是同一份标度值——
所以 §58 的 ~150× 只能来自"标度被按错误的轴/块宽施加"。

修复（照抄参考实现 `qpn/fp8.py:process_weights_after_loading` 的做法：加载期就把标度吃进权重并校验行数）：
1. `QuantConfig` 增加对 `config_groups.*.weights.strategy` / `block_structure` / `group_size` 的解析；
2. `strategy == "channel"` ⇒ 块宽按 `[1, K]` 处理（而不是默认 `[128,128]`）；
3. 加载期校验标度行数 == 期望（参考实现里 `expected_scale_rows` 那道断言），不等直接报错而不是静默错量级。

## 60. 根因锁定：`in_proj_z` 落到了 `DType::U8` 回退，原始字节被当成整数值（case `d2#60`）

同一次前向里两段的实测对照（`CALLS<4` 门控，两段都打）：

| | `in_proj_qkv[0]`（正常） | `in_proj_z`（坏） |
|---|---|---|
| `w_dims` / `s_dims` | `[5120,5120]` / `[5120,1]` | `[5120,3072]` / **`[1,3072]`** |
| `block_size` | `[1,5120]` | `[1,5120]` |
| `w_first4` | **`-48, -6.5, 52, 320`**（含小数 ⇒ 已按 e4m3 解码） | **`102, 225, 113, 235`**（全正整数） |
| 检查点原始字节前 4 | `241, 112, 224, 114` | `102, 101, 225, 101` |
| `w_absmax` | `448.0`（= e4m3 解码后 max） | **`254.0`**（= 字节 max，`0xFE`） |
| `s_absmax` | `9.727e-4` = 检查点标度 max ✓ | `1.357e-1`（检查点是 `5.30e-4`） |
| `chan_maxabs` | `[0.0532, 0.0841, 0.4358]` ✓ 与检查点反量化分布一致 | `[3.488, 10.27, 34.21]` ✗ |

⇒ **`in_proj_qkv` 走通了 F8E4M3 解码；`in_proj_z` 落到了 `load_fp8_weight` 的
`.or_else(|_| get_with_hints_dtype(..., DType::U8))` 回退**（`linear.rs:1089-1090`），
把 e4m3 的**字节**当成整数值，于是有效权重大 ~10³，且标度也拿成了别的张量（`[1,3072]` 而非 `[3072,1]`）。

修复方向（照抄参考实现：加载期一次性解码 + 行数断言，不留静默回退）：
`load_fp8_weight` 的 U8 回退不得静默生效——要么按 F8E4M3 成功解码，要么直接 bail 并打印请求形状，
让 z 这条"请求形状与检查点不符"的真实错误暴露出来。

## 61. 根因确认（dtype 级证据）（case `d2#61`）

加上 dtype 后一次跑就定了：

| | `in_proj_qkv[0]` | `in_proj_z` |
|---|---|---|
| `w_dtype` | 已解码（`w_first4` 含 `-6.5` 这类非整数值） | **`U8`** |
| `s_dtype` | 同上路径 | **`F16`**（检查点是 BF16） |
| `w_dims` | `[5120, 5120]` | `[5120, 3072]` |
| `chan_maxabs` | `[0.0532, 0.0841, 0.4358]` ✓ 与检查点反量化分布一致 | `[3.488, 10.27, 34.21]` ✗ 大 ~150× |

⇒ `in_proj_z` 的权重张量是 **U8 原始字节**（e4m3 的字节被当整数值），从未按 e4m3 解码；
`w_absmax=254` 就是字节 `0xFE` 的整数值。qkv 那条走通了正确路径，z 这条没有。

修复（照抄参考实现：加载期一次性解码 + 行数断言，不留静默回退）：
`in_proj_z` 这条加载路径必须按 e4m3 解码取到的字节（或让 GEMM 显式按 e4m3 解释 U8），
并在加载期断言"标度行数 == 期望行数"，让形状/分片不符直接报错而不是静默换 dtype。

## 62. 位等确认 + 新事实：F8E4M3 请求成功却返回 U8（case `d2#62`）

**位等确认**（不再需要假设）：把检查点 z 权重按引擎的 `[K,N_shard]` 视图取列 0 的**原始字节**：

```
检查点  bytes W[n,0] n=0..3 = [102, 225, 102, 94]
引擎    w_first4            = [102, 225, 113, 235]   ← 前两项逐字节相同
```

⇒ 引擎里 `ln.weight` 存的就是 **e4m3 的字节本身**（当整数值），从未按 e4m3 语义解码。
后两项不同是 TP 分片/头交织的索引置换（`undo_tiled_v_heads_first_dim`）所致，不影响结论。

**新事实（决定性）**：在 `load_fp8_weight` 的 F8E4M3 分支上加了"失败才回退并告警"的临时日志，
跑完 **告警一次都没出现** ⇒ F8E4M3 那次请求**没有失败**，它成功了、却返回了 `U8`。
对照 `in_proj_qkv[0]`：`w_dtype=F8E4M3`、`s_dtype=F32`（走通了）；`in_proj_z`：`w_dtype=U8`、`s_dtype=F16`。

⇒ 根因不在 `load_fp8_weight` 的回退，而在**非 merged 的 `load_with_hints` 这条路把请求 dtype 换成了
模型 dtype（SM70 上 `CUDA capability 7 < 8 → switched to F16`）**，于是 fp8 张量按 F16/U8 语义被读，
e4m3 字节被当整数值。qkv 走 merged 路（`load_merged_with_hints`）没被换。

修复要求（下一轮实现，照抄参考实现"加载期解码 + 行数断言"）：
`LnFp8::new` 里权重必须按 **e4m3 语义**读出（请求 F8E4M3 得到 F8E4M3），
拿到 U8 就 bail 并打印请求形状，而不是继续算。

## 63. 构造期参数（根因链闭合）（case `d2#63`）

临时在 `ln_fp8_x` 入口打印请求参数（rank0/rank1 各一条）：

```
out_dim=6144 in_dim=5120 shard={dim:0, rank:1, world:2} block=[128,128]   ← in_proj_z
out_dim=6144 in_dim=5120 shard={dim:0, rank:0, world:2} block=[128,128]   ← in_proj_z
out_dim=5120 in_dim=6144 shard={dim:1, rank:0, world:2} block=[128,128]   ← out_proj
```

要点：
1. **`block=[128,128]` 是 `unwrap_or` 的默认值** —— 检查点把块信息放在
   `config_groups.group_0.weights.strategy="channel"`（`block_structure:null`、`group_size:null`），
   而 `QuantConfig` 只读顶层 `weight_block_size`，**从不解析 `strategy`**（§59 成立）。
2. 于是 `scale_dim0=(6144+127)/128=49`、`scale_dim1=(5120+127)/128=40` —— 按"每 128 行一块"去要标度；
   而检查点实际是 `[6144,1]` 每通道一个。`is_channel_scale_shape` 那条补救只在形状恰好吻合时才救得回来。
3. 结合 §62：z 的 `w_dtype=U8`、`s_dtype=F16`，且权重/标度都被按转置视图存过
   （`w=[5120,3072]`、`s=[1,3072]`），于是 e4m3 字节被当整数值、标度量级错 ~256×，
   净效果 = 有效权重大 ~150×（§58）。

## 修复（照抄 `qpn/fp8.py:process_weights_after_loading`）
1. `QuantConfig` 解析 `config_groups.*.weights.{strategy, block_structure, group_size}`；
   `strategy=="channel"` ⇒ 块宽 `[1, K]`，不再落默认 `[128,128]`。
2. 加载期把标度吃进权重（或至少按正确轴施加），并像参考实现那样断言
   `标度行数 == 期望行数`，不符直接 bail（打印请求形状），禁止静默换 dtype/量级。

## 64. 结构事实：权重按 `[K, N_shard]` 存，而标度按 `[N, K]` 语义施加（case `d2#64`）

`LnFp8::forward`（`linear.rs:1555-1582`）把 `(weight, weight_scale, weight_block_size)` 直接交给
`attention_rs::fp8_linear::fp8_matmul_with_input_scale`。而两侧实测布局：

| 段 | `w_dims` | `s_dims` | `block_size` | 标度实际沿哪根轴 |
|---|---|---|---|---|
| `in_proj_qkv` | `[5120, 5120]` = `[K, N_shard]` | `[5120, 1]` | `[1, 5120]` | **沿 dim0 = K**（每 K 一个）|
| `in_proj_z` | `[5120, 3072]` = `[K, N_shard]` | `[1, 3072]` | `[1, 5120]` | 沿 dim1 = N ✓ |

⇒ 权重是**转置布局** `[K, N_shard]`（检查点是 `[N, K]`），而 `ln_fp8_x` 按 `[N, K]` 语义算
`scale_dim0/scale_dim1` 与 `block_size`。qkv 那条因此把"每输出通道一个标度"施加成了"每 K 一个"，
z 那条方向对但量级仍错（§58 的 ~150×）。

这也解释了 §55/§56 的"比值非常数"：那是**按错轴取值时的索引置换**，不是标度本身被改。

## 修复（照抄 `qpn/fp8.py:process_weights_after_loading`）
1. `QuantConfig` 解析 `config_groups.*.weights.strategy`；`"channel"` ⇒ 块宽 `[1, K]`（不再落默认 `[128,128]`）。
2. `ln_fp8_x` 里按**实际存储布局**求 `scale_dim0/scale_dim1`：权重是 `[K, N_shard]` 时，
   逐通道标度必须是 `[1, N_shard]`，不是 `[N_shard, 1]`。
3. 加载期断言 `标度形状 == 期望形状`（参考实现的 `expected_scale_rows`），不符直接 bail。

## 65. **证伪**：§58/§62/§64 的"150×"是诊断自身的假象（case `d2#65`）

在 `ln_fp8_x` 里分三点打印，一次跑就翻案：

| 位置 | dtype | dims |
|---|---|---|
| 加载后（`load_fp8_weight` + 按 dim0 分片） | **`F32`** | **`[3072, 1]`** ✓ 就是检查点的逐通道标度 |
| `Ok(LnFp8{..})` 前（= 诊断读到的） | `F16` | `[1, 3072]` |
| 权重同理 | 加载 `F8E4M3 [3072,5120]` → 诊断读到 `U8 [5120,3072]` |

原因在 `linear.rs:1552-1558`：

```rust
let (weight, weight_scale) = if dtype == DType::F16
    && weight_scale.dims2().map(|(_, c)| c == 1).unwrap_or(false)
    && sm_version < 80
    && weight_scale.dtype() == DType::F32
    && std::env::var("XINFER_QPN8_DISABLE").is_err()
{ attention_rs::fp8_linear::fp8_qpn8_repack(&weight, &weight_scale)? }
```

SM70（`sm_version<80`）+ 单列 F32 标度 + 模型 dtype F16 ⇒ **QPN8 打包路径本来就是活的**，
`U8`/`F16`/`[1,3072]` 是**内核自己的打包布局**，不是"没解码/施加错轴"。

⇒ §58 用 `|w|·s` 朴素广播去量"有效权重"，量的是**打包后的张量**，其 150× 与
"min/mean/max 一致到 ±4%" 都是打包布局的产物 ⇒ **§58 的量级结论、§62 的"字节当整数值"、
§64 的"施加错轴"全部作废**（D4：进 `docs/design/sm70_1cat_rejected_paths.md`，禁止再照它们改）。

## 下一步（未被上面污染的真问题）
判据仍是 §50.4 的逐阶段幅值：`in_norm 4.8e1 → qkv 6.1e1 → recur 4.0e0 → gated 2.3e-1 → out_proj 1.0e3`。
要判的是**GEMM 输出**而不是打包张量：下一步按 `XINFER_QPN8_DISABLE=1` 与默认各跑一次，
比较同一阶段的幅值 —— 若关掉打包后 `out_proj` 输出回到输入量级，问题在 repack/内核；
若两档一样，问题在别处。

## 66. 真因确认：`fp8_qpn8_repack` 就是幅值爆炸的来源（case `d2#66`）

同一层、同一请求，只切 `XINFER_QPN8_DISABLE`：

| 阶段（prefill，n=189440） | QPN8 开（默认） | QPN8 关 |
|---|---|---|
| `L0 in_norm` | 4.816e1 | 4.816e1（同） |
| `L0 attn_out` | **1.024e3** | **2.812e1** ✓ 与输入同量级 |
| `L0 mlp_in` | 6.879e0 | 2.525e0 |
| `L0 mlp_out` | 8.962e1 | 1.348e1 |
| KV `L15 v` maxabs | 1.334e2 | 3.478e1 |

生成质量（greedy，`temperature=0`，"capital of France, one word"）：

- QPN8 开：输出塌成 token 0（§50 起的全部现象）
- QPN8 关：**"The user asks for the capital of France in one word. The answer is Paris"** ✓

⇒ 判据 ③ 的多项目标一次达成：`out_proj` 输出回到输入量级、逐层不再单调爬到 1.2e4、
输出不再全 token 0。

## 待办
`fp8_qpn8_repack` 把 F32 标度降成 F16 并按 `[1, N]` 打包（`linear.rs:1552-1558`）。
参考实现 `qpn/fp8.py` 的契约是 **标度保持 F32**（`if weight_scale_inv.dtype != torch.float32: .to(torch.float32)`）。
下一步：查 `attention.rs` 的 `fp8_qpn8_repack` 为何产出 F16/转置布局，按参考实现修，
而不是长期靠 `XINFER_QPN8_DISABLE` 绕过。

## 67. 判据④ 首轮 A/B（QPN8 关，服务端 `output_len` 口径）（case `d2#67`）

`XINFER_QPN8_DISABLE=1`，greedy，`NTOK=256`，单请求，TP=2。服务端 `[Seq 0] ⏱ Decoded:` 行：

| ctx | mtp | 服务端解码量 | 吞吐 | TTFT | 结果 |
|---|---|---|---|---|---|
| 24 | 0 | 111 tok / 7.59 s | **14.62 t/s** | 461 ms | 正确（"…answer is Paris"）|
| 24 | 1 | — | — | — | **`CUDA_ERROR_OUT_OF_MEMORY`** → `MTP runner returned empty response` |
| 768 | 0 | 256 tok / 17.13 s | **14.94 t/s** | 1231 ms | 正确 |
| 768 | 1 | — | — | — | 同上 OOM |

要点：
1. **mtp=0 两档都跑满且文本正确** ⇒ §50 起"输出塌成 token 0"已消除（判据 ③ 达成）。
2. **mtp=1 现在不是"快而假"，而是直接 OOM**：QPN8 关掉后权重以 `F8E4M3 + F32 标度` 原样驻留，
   不再压进打包布局，显存上升后 draft/verify 的额外缓冲放不下。
   ⇒ MTP 的 A/B 表要在"权重按参考实现契约保持 F32 标度"与显存预算之间定一个再出（判据 ④ 未完）。
3. 客户端 `per_req_tok_per_s` 在 mtp=1 档为 0 是**因为请求被 cancel**，不是口径错。

## 68. 位级核对：`×256` 是**正确补偿**，不是错量级（case `d2#68`）

`fp8_qpn8_sm70.cu` 的两处：

```cpp
// 解码：把 e4m3 字节按位拼成 half
value = ((permuted << 8) & 0x8000) | ((permuted << 7) & 0x3f80);
// 标度：channel_scales[col] = __float2half(scales[col] * 256.0f);
```

`permuted << 7` 把 e4m3 的 `[s|eeee|mmm]` 直接落到 half 的 `[s|eeeee|mmmmmmmm]` 的指数字段，
**没有加 `(15-7)<<10` 的偏置** ⇒ 解出来的值是**真值 / 256**，所以标度 `×256` 是**对的补偿**。
⇒ "标度量级错"这一族理论到此终结（§58 的 256× 就是这条补偿，不是 bug）。

索引也自洽（手算 5 个 lane 验证）：`qpn8_col_from_lane` 与 `qpn8_lane_from_col` 互逆：
`col_from_lane(4)=8, col_from_lane(16)=4`；`lane_from_col(8)=4, lane_from_col(4)=16` ✓。
写侧 `channel_scales[col]`（col = 真通道号），读侧 `group_scales[tile*32 + col_from_lane(lane)]`
且码的 `lane = lane_from_col(row&31)` ⇒ 读到的就是 `scales[row]` ✓。

## 因此下一步是数值探针，不是再猜
对第 0 层同一段投影，用同一输入分别跑
(a) `fp8_qpn8_gemm_f16`（打包路）与 (b) 朴素 `x @ dequant(w).t()`，比输出 maxabs。
两路一致 ⇒ 打包路无罪，问题在别处；不一致 ⇒ 差异就是 §66 的 36×，再定位到 `split_k`/`tile` 边界。

## 69. 判据④ 干净口径 A/B（探针关 `DNAN=0`，`XINFER_QPN8_DISABLE=1`）（case `d2#69`）

诊断本身会物化全尺寸 F32 副本（`channel_stats` 的 `broadcast_as` + `abs` + `mul`），
所以取数一律 `DNAN=0`。greedy，`NTOK=256`，单请求，TP=2，服务端 `[Seq 0] ⏱ Decoded:` 口径：

| ctx | mtp | 服务端解码量 | 吞吐 | 备注 |
|---|---|---|---|---|
| 24 | 0 | 111 tok / 2.53 s | **43.96 t/s** | 111 个即命中 EOS（未到 256）|
| 24 | 1 | — | — | `Runner MTP decode error: CUDA_ERROR_OUT_OF_MEMORY` |
| 768 | 0 | **256 tok / 5.78 s** | **44.32 t/s** | 跑满 256 ✓ |
| 768 | 1 | — | — | 同上 OOM |

要点：
1. 与 §50 的"假模型"基线（42.38 t/s）同量级 ⇒ **正确模型下的非-mtp 基线成立**。
2. **MTP 路在"权重不被 qpn8 打包"的形态下起不来**：解包形态下每次调用要现取反量化副本，
   draft/verify 的额外缓冲放不下 ⇒ `CUDA_ERROR_OUT_OF_MEMORY`。
   所以 ④ 的加速比必须等 §68 的打包路数值探针出结论后才能出，不能靠绕过。

## 70. 新探针 `qpn8_channel_probe`：打包路对逐通道标度是**数值忠实**的（case `d2#70`）

补了 `tests/probes/qpn8_channel_probe.rs`（新文件 57 行，`[[example]]` 已注册），
用 `[N=64, K=64]` e4m3 码 + 每通道一个标度 `[64,1]`（各通道量级不同）跑两路：

| 记录 | maxabs | absmean |
|---|---|---|
| `packed`（`fp8_matmul`，块 `[1, K]` ⇒ 走 QPN8 打包） | **1.1407e-01** | 3.4869e-02 |
| `gold`（`fp8_matmul_fallback`，朴素反量化 matmul） | **1.1407e-01** | 3.4872e-02 |

两路 maxabs **完全相同**，absmean 只差第 5 位（F16 舍入）⇒ **权重侧打包路无罪**：
它不改变量级，也不把逐通道标度施加到错轴。

⇒ §66 的"QPN8 是根因"要收窄：ON/OFF 的 36× 差异不来自**权重标度**，
而来自打包路对**激活侧**的处理 —— 打包路会把输入按每 16 个 K 一组量化
（`fp8_qpn8_pack_input_f16` → `input_is_packed`），而 fallback 用原始 F16 激活。
下一步就是把激活侧纳入差分（同一输入，比 packed/gold 在真实 K=5120、TP 分片形态下的输出）。

## 73. 探针修正后**复现了幅值爆炸**（case `d2#73`）

§70/§71 的"两路一致"是**无效的**：`fp8_matmul` 的打包分支门控是
`weight_scale.dtype()==F16 && dims==(1,n)`，而探针传的是 `F32 (n,1)` ⇒ 两路都走了 fallback。

改成先 `fp8_qpn8_repack` 再喂 `fp8_matmul`（= 引擎 ON 路径的真实形态）后：

| 用例（`N=3072, K=5120`，码 `±1/±2`，标度 `3.4e-2..1.35e-1`） | maxabs | absmean |
|---|---|---|
| `real.packed`（打包分支） | **inf** | **nan** |
| `real.gold`（`fp8_matmul_fallback`） | 5.4962e-02 | 1.6335e-02 |

且把激活再乘 `0.004` 仍 `inf` ⇒ 溢出发生在**内核内部**（解码出的权重本身爆），
与输入量级无关。

⇒ §66 的 36× 与这里的 `inf` 同源：`fp8x8_to_half2x4_fast` 用
`((p<<8)&0x8000) | ((p<<7)&0x3f80)` 按位拼 half，它**不是通用 e4m3 解码器**——
它要求字节已按 half 偏置好；而 `channel_scales` 又统一 `×256`。
两者对"检查点原始 e4m3 字节"这一路是否成立，是下一步要对着参考实现核对的点。
