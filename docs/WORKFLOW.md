# 开发流程（怎么干活）

> 分工：[CONVENTIONS.md](<./CONVENTIONS.md>) = 写代码时的规则；本文件 = 推进一项工作的流程。
> 每一步都有落点文件和验收物，目标是让流程可执行而不是"自觉"。

## 1. 一项优化从开工到验收
顺序不可跳（文档先行，CONVENTIONS §1）。

| # | 步骤 | 落点 | 做完的标志 |
|---|---|---|---|
| 1 | 开一份规格（立问题 + 列假设并排序） | `docs/specs/<id>-<subject>.md` | 五节齐全（CONVENTIONS §1.1）；每条事实带 `文件:行号`；验收条已编号成 case id |
| 2 | 选最便宜的测量 | `tests/probes/<name>_probe.rs` | 探针能跑出区分假设的数，且已注册进 `Cargo.toml` |
| 3 | 测基线 | `$XINFER_PROBE_DIR` | 基线数值 + 测量条件（shape/batch/TP/graph/时钟） |
| 4 | 实现 | 源码 | 一条切片，改动范围收敛在一条路径上 |
| 5 | A/B 实测 | 该规格 §5 | 数值表 + wins 计数（不是"快了 12%"）；每条验收条按 `@case` 对账，缺一条即红 |
| 6 | 判生死 | [DECISIONS.md](<./DECISIONS.md>) / 已否决清单 | 通过 → 写 ADR；不通过 → **当场**进已否决清单并标三态 |
| 7 | 提交 + 推送 | git | 在 `perf/sm70-volta`，正文带实测数据，已 push |

第 6 步最容易偷懒。标准参照已否决清单 A1/A2 两条：连 SASS 指令数、regs/thread、CTA/SM 占用、stall 分布都写了原因——这才叫"写到不需要重试"。判完生死这份规格就**冻结**（状态改 `已采纳`/`已否决`），后续工作另开一份，不在冻结件上续写。样板：[specs/nvfp4-hmma884-port.md](<./specs/nvfp4-hmma884-port.md>)。

## 2. 一次跨仓内核移植
上游（1Cat）→ `attention.rs` → `xinfer` 三段式，**每段单独提交、单独推送**。

1. 抄 kernel 到 `attention.rs/src/kernels/src/*.cu`：指令序列 **+ shared 布局字节**（D2）；先只加 `extern "C"` 声明，不接进 xinfer。
2. 包 wrapper（`ffi.rs` + 对应 `src/*.rs`），契约（shape/dtype/布局/返回）写进函数注释。
3. 接进 `src/models/layers/*.rs`：接的是 **family path**（D5）；`Cargo.toml` 的 `attention-rs rev` 指向新 sha。
4. 精度验收（[TESTING.md](<./TESTING.md>) §2）→ 性能验收（D12）。

⚠ 中断点保护：1、2 做完还没接进 xinfer 时，**先把 commit 推上去**。本项目已两次 stash 被丢弃（`781ba04`、`01589ed`），内容后来靠重写找回，那是运气不是流程。

## 3. 决策变更
任何改变架构/契约的决定 → 先写 [DECISIONS.md](<./DECISIONS.md>)（背景/决定/影响）→ 再改代码；引用用 D 编号。

契约级变更的判定：改任一 `rev` / 改公开 API / 改精度策略 / 改显存预算模型。
契约级变更必须在**同一个提交**里带上：ADR 条目 + 精度证据 + 性能证据 + 回归结果。

## 4. 缺陷处理
复现（写进探针/回归脚本）→ 定位 → 修复 → 回归绿 → 提交。

- 不允许无复现修复；回归脚本带 `@repro <缺陷号>`，缺陷关闭后**不删**（永久回归网）。
- 日志不够用时**先补诊断再猜**：`Err(NcclError)` 那个坑的根因是 `cudarc/src/nccl/result.rs:11-15` 的 `Debug` 把 `ncclResult_t` 丢成字面量，先让错误码可见再谈修复。

## 5. 一项工作算完成
精度探针通过（误差在声明的容差内）**且** A/B 数值表齐 **且** 生效区间与失效区间的回退路径都写清 **且** ADR/已否决清单已更新 **且** commit 已 push 且在 `perf/sm70-volta` 上。

## 6. 什么时候必须走 PR
现状（单线程 + 文档先行 + 证据制）下不要求。出现任一信号即要求：并行工作流 ≥2；契约级变更（§3）；第二个开发者加入；跨仓原子变更（kernel + rev + 证据同 PR 评审）。

## 7. 真机冒烟
常规回归全走探针（合成 shape、可重复）；真模型端到端只手动一枪。记录模型、TP 配置、tok/s、错误数 → `docs/SM70_VOLTA_FINDINGS.md`。失败先排环境（时钟、驱动、显存占用、他进程），再疑代码。

## 8. 提交前自检
```
[ ] 在 perf/sm70-volta（不是 detached HEAD），离开前 porcelain 为空
[ ] 精度探针绿，误差数值已记录
[ ] A/B 表已产出（测量条件 + wins 计数）
[ ] 生效边界 + 回退路径已写清
[ ] 结论已进 DECISIONS.md / 已否决清单
[ ] Cargo.toml 的 git rev 与实际使用的 sha 一致
[ ] 提交正文含实测数据，已 push
```
