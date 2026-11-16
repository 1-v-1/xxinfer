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
