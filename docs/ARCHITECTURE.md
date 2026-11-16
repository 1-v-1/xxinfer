# 架构：四仓拓扑与代码地图

> 分工：本文件 = 代码在哪、为什么这么分。规则见 [CONVENTIONS.md](<./CONVENTIONS.md>)，决策见 [DECISIONS.md](<./DECISIONS.md>)。
> 四仓全部按 git rev 锁定，改任何一仓的 rev 都是契约级变更（D3）。

## 1. 布局
```
xxinfer-dev/                       ← 开发工作区（本身不是 git 仓库）
└─ repos/
   ├─ xinfer/       ← 主项目：OpenAI 兼容推理服务器（Rust，src/ 127 文件）
   ├─ attention.rs/ ← 注意力/线性层 CUDA 内核 fork（43.4k 行）
   ├─ candle/       ← 张量库 fork（161.5k 行）
   └─ cudarc/       ← CUDA driver 封装 fork（915.7k 行）
```
参考实现（**只读**）：`/Volumes/SMB重要/1Cat-vLLM`，commit `bcd581d230`。本项目是它的**移植目标**，不是它的分支。

## 2. 依赖拓扑（单向无环，每条边锁 rev）
| 仓 | 职责 | 被锁 rev |
|---|---|---|
| `xinfer` | 调度、KV cache、模型层、CUDA graph、MTP | — |
| `attention.rs` | FlashInfer/FlashAttention/FP8/NVFP4/QPN8/GDN 内核 | `144a07f` |
| `candle` | 张量/算子/GGUF 解码 | `8ac335c` |
| `cudarc` | driver/NCCL/graph 封装 | `2e81793` |

`candle` 的 `CudaDevice` 自己没有 `alloc`，所有分配经 `Deref` 落到 `cudarc::driver::CudaDevice::alloc` —— 这是**唯一的分配咽喉点**（D14 的落点）。

## 3. `src/` 模块地图
| 目录 | 文件 | 行数 | 爆炸半径 |
|---|---|---|---|
| `src/models/` | 66 | 39.3k | **最大**，几乎每个模型都过一遍 |
| `src/utils/` | 23 | 19.5k | 大，跨层共享（`graph.rs`、`kvcache_allocator.rs`） |
| `src/server/` | 6 | 13.2k | 中，协议兼容性风险 |
| `src/core/` | 7 | 9.5k | **大**，主循环（`engine`/`runner`/`block_manager`/`scheduler`） |
| `src/speculative/` | 6 | 1.9k | 中，`mtp.rs` 与 `deltanet` 强耦合 |
| `src/tools/`、`src/runner/`、`src/transfer/`、`src/py/`、`src/mcp/` | — | — | 小 |

**三条最热路径**（改它们必须跑端到端 + 精度探针）：
1. **GDN / MTP verify** — `src/models/layers/deltanet.rs`（`conv_snapshots`/`recurrent_snapshots`/`gdn_verify_mtp`）× `src/speculative/mtp.rs`。
2. **量化线性层** — `src/models/layers/linear.rs` × `attention.rs/src/fp8_linear.rs`（QPN2/4/8 三套内核分发，dtype/布局/分块三重敏感）。
3. **CUDA graph** — `src/utils/graph.rs`（捕获、内存池、replay 计划）× `cudarc` 的分配语义（D14）。

## 4. 运行时数据流
```
HTTP → server/parser.rs → core/scheduler.rs → core/runner.rs → core/engine.rs
        ├─ models/layers/deltanet.rs   GDN 状态推进（prefill / chunk / verify 三路）
        ├─ models/layers/linear.rs     量化 GEMM（走 attention.rs 内核）
        ├─ models/layers/distributed.rs TP 集合通信
        ├─ utils/graph.rs              decode 可选 graph 捕获/replay
        └─ utils/kvcache_allocator.rs  KV / 循环状态池预算
```
**三种前向形态分开测，不得互相外推**：prefill（`is_prefill`，chunked GDN / dequant）、decode（M=1..N 的 GEMV，最吃单序列延迟）、MTP verify（M=T 投机步数，`gdn_verify_mtp` 一个内核跑完 T 步递推）。

## 5. 硬件硬约束（SM70 / V100）
- **无** `cp.async` / LDGSTS / TMA / 异步事务屏障；只能软件级 named-barrier `LDG→GPR→STS` 替代。
- occupancy 是最大杠杆：BM32 胜出靠 **64 regs/thread + 41.94 KB shared + 2 CTA/SM**；FA2 风格 M64（128 regs、83.47 KB、1 CTA/SM）虽把 L1 global-load sectors 降 16.3%，墙钟仍回退 2.0%。
- **传输最小 ≠ 关键路径最小**：SHFL 33→17、寄存器 40→32，64K 下墙钟仍回退 1.4%（long-scoreboard 每 issue 成本 0.096→0.311 cycles）。
- chunked all-reduce 差得多：1/2/4/8/16 chunk = 13.21 / 19.97 / 37.37 / 66.55 / 127.68 us，不能靠"多开内核做分块"。
- 原生 `m16n16k16`（展开 16 条 `HMMA.884`）+ top/bottom M16 operand reuse 是唯一赢家；raw `m8n8k4` 结构上更慢（QK 212.352→251.008 us）。
- 性能测量须锁时钟（Volta 1200 MHz），否则 tok/s 无意义。

## 6. 分支拓扑
`perf/sm70-volta` 是唯一工作落点（D7）。历史坑：5 个正式提交曾只活在 `HEAD detached from e83f8c5`，切分支即"消失"；`attention.rs` 曾有 +103 行从未提交只在工作区。
