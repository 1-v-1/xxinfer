# 术语表

> 用途：本项目与上游 1Cat 之间术语大量重名不同义，AI 会话极易混淆。**本表是术语的唯一权威**，冲突以本表为准并回来改文档。

## 硬件与平台
| 术语 | 含义 |
|---|---|
| SM70 / Volta / V100 | 主战场。无 `cp.async`/`ldmatrix`/`wgmma`；255 regs/thread 上限；96 KB shared/SM；2 CTA/SM 与 64 regs 是多数实验的硬门控 |
| SM90 / Blackwell | FlashInfer chunked GDN prefill 仅 SM90；SM70 上必须走 native flash 回退 |
| locked clock | 性能测量须锁时钟（Volta 1200 MHz），否则 tok/s 无意义 |

## 量化与内核
| 术语 | 含义 |
|---|---|
| NVFP4 | e2m1 码 + block scale。⚠ 在本项目里是 **W4A16 模拟**，不是原生 W4A4/激活量化；Nsight 的 "TurboMind NVFP4 GEMM" bucket 名实际指 AWQ W4A16 路径 |
| QPN2 / QPN4 / QPN8 | 1Cat 分块量化内核系列（2/4/8-bit，per-16 一个 scale）；QPN8 = channel-wise FP8 变体，码按 tile-major 排布 |
| RowTiles / M1Only / SplitK / NAcc | 内核模板参数：每 tile 行数 / 单行特化 / K 向切分 / 累加器块数 |
| dequant + cublas | 反量化成 dense F16 再走 cuBLAS。**正确但慢**，是所有"快路"的兜底 |
| HMMA 884 | SM70 的 f16 MMA：`mma.m8n8k4`，fma 做两次得 M8/N32/K8 |
| bit-identity | 与参考实现逐 bit 相同（比"数值接近"严格得多） |

## 模型与运行时
| 术语 | 含义 |
|---|---|
| GDN / GatedDeltaNet | 门控 delta 线性注意层。循环状态池**恒为 F32**（`kvcache_allocator` 硬编码），与权重/激活 dtype 无关 |
| MTP / MTP verify | 多 token 预测 / 投机校验步（`gdn_verify_mtp` 一个内核跑完 T 步递推） |
| conv / recurrent snapshot | verify 用的卷积态 / 递推态快照行；`state_indices` 选 slot，原地推进 |
| family path | 一条内核路径服务多个模型，按 family selector 分发（D5） |
| CUDA graph / capture pool / replay plan | 捕获加速 / 捕获期专用内存池 / replay 执行计划（`XINFER_DIAG_PLAN` 可打印） |
| eager vs graph | 两条执行形态，**可以用不同内核**，但输出须满足同一质量契约 |
| workspace / transient reserve | 按实际启用能力门控 / 随 `max_num_seqs × chunk` 缩放（D13） |
| provenance | 结论的证据链：`文件:行号` + commit hash + 实测数值 |
| 1Cat | 上游参考实现（`bcd581d230`），**只读**；"1Cat 语义"= 以其行为/数值约定为准，而非"通用做法" |

## 仓库与流程
| 术语 | 含义 |
|---|---|
| 四仓 / rev 锁定 | `xinfer`/`attention.rs`/`candle`/`cudarc`；依赖用 `rev = <sha>` 固定，不跟 branch、不用 path |
| 契约级变更 | 改 rev / 改公开 API / 改精度策略 / 改显存预算模型 |
| 已否决清单 + 三态 | `docs/design/sm70_1cat_rejected_paths.md`；`否决`（证伪并移除）/ `证伪`（机理不成立）/ `未准入`（数值或门控未过） |

## 易踩
- ❌ 以为"照抄指令、换布局"即可 → shared 布局必须一起抄（D2）。
- ❌ 把"未准入"当"机理不成立" → 会误杀可重试路径。
- ❌ 把"精度过"当"可以上线" → 还要看覆盖区间与回退路径（TESTING §2）。
- ❌ 把"GDN dtype = F16"当"状态池也是 F16" → 状态池恒 F32。
- ❌ 把 decode tok/s 外推到 prefill/TTFT → 两者不互推（D12）。
