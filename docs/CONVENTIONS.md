# 工程规范

> 分工：本文件 = 写代码时的硬规则；流程见 [WORKFLOW.md](<./WORKFLOW.md>)，代码在哪见 [ARCHITECTURE.md](<./ARCHITECTURE.md>)，决策依据见 [DECISIONS.md](<./DECISIONS.md>)。
> 适用：`xinfer` 与三个 fork（`attention.rs` / `candle` / `cudarc`）。冲突时**仓库内更具体的优先**。

## 1. 文档先行
- 改架构/契约/内核路径 → 先改 [DECISIONS.md](<./DECISIONS.md>)，再改代码。
- 聊天记录不是真相源，文件才是。口头结论当场落文件。
- 结论落点（唯一去处，不新建"流水"文件）：

| 类型 | 落点 |
|---|---|
| 架构/契约/流程决策 | `docs/DECISIONS.md` |
| 被证伪的方案 | `docs/design/sm70_1cat_rejected_paths.md` |
| 硬件/内核实测结论、开关与坑 | `docs/SM70_VOLTA_FINDINGS.md` |
| 内核移植契约与 fragment 布局 | `docs/design/sm70_nvfp4_hmma884_port.md` |
| 跨仓流程与规范 | `docs/` 对应主题文件 |

- 结论必须带证据：性能带数值表，精度带误差数值 + 探针文件名，行号指向**当前**文件。

## 2. 分支与提交
- 提交前 `git status` 必须显示 `On branch perf/sm70-volta`，不是 `HEAD detached`（D7）。
- 一个切片一提交，**提交后立刻 push**。提交信息：`类型(范围): 做了什么 + 为什么`，类型取自 `feat|fix|perf|build|deps|chore|docs|test|diag|graph|cleanup|runner|distributed`。
- 性能/精度变更的提交，正文必须含实测数值与测量条件，不允许只有结论。
- 离开一个仓前 `git status --porcelain` **必须为空**：要么提交，要么 `git stash` 并在提交信息里写明内容（D7）。
- revert 三件套 + 手工回退自查，见 D8；被回退的方案同时进已否决清单。
- 改任一依赖的 `rev` = 契约级变更（D3），走 WORKFLOW §3。

## 3. 代码
- Rust edition/工具链跟随 `Cargo.toml`；不改 `rust-toolchain` 除非单独决策。
- **不把上游 fork 改脏**：xinfer 专属新增物放本仓，不塞进 fork 制造 diff 噪音。
- 公开 API 变更同步 `docs/` 对应用户文档。
- 性能路径必须在注释里写明**前提**（dtype/layout/shape/TP/是否 graph）。
- 数值安全优先于速度：kernel 改动必须过精度探针（[TESTING.md](<./TESTING.md>) §2）。
- 抄 1Cat kernel 连 shared memory 布局一起抄，不得改用 CUTLASS `ldmatrix` 套路（D2）。

## 4. 日志与开关
- `src/` 内禁止裸 `println!`/`eprintln!`（`tests/probes/`、`example/` 与一次性汇总除外）。诊断用 `tracing::debug!`，错误用 `error!`（D9）。
- 会重复打的日志用 `Once`/`OnceLock` 收敛或加 env 门控。
- 开关全 `XINFER_` 前缀、三族、默认值安全（D10）；新开关在提交信息里写明默认值与影响面。

## 5. 探针
- 命名 `<subject>_<kind>_probe.rs`，`kind ∈ {precision, perf, breakdown, overhead}`，放 `tests/probes/`。
- **必须**在 `Cargo.toml` 注册 `[[example]]`；精度探针配 `compare_*_probe.py`，用 `precision_metrics.py` 的 `report` 统一输出（D11）。
- 产物落 `$XINFER_PROBE_DIR`：产物不入 git，结论入 git。

## 6. 硬性禁止
- ❌ 无 A/B 实测的性能结论（"感觉更快"不算证据，D12）。
- ❌ 只改 shared layout / 换 tile 参数就重试已否决方案（D4）。
- ❌ 把探针/合成 shape 的结果当生产结论（必须标 shape、batch、TP、是否 graph）。
- ❌ 在 detached HEAD 上留正式提交，或把工作留在未提交的工作区（D7）。
- ❌ 用 `git cherry` / `--cherry` 当"没有回退"的证据——手工重写的回退查不出（D8）。
- ❌ 把 1Cat 的 fast path 整条复制进某个模型（走 family path，D5）。
- ❌ 静默改精度：改了就要说、要测、要写进契约（D6）。
