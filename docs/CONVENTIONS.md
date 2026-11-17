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
| 一项工作的规格（含 fragment 布局等契约） | `docs/specs/<id>-<subject>.md` |
| 上游逆向产物（1Cat 行为规格） | `docs/design/` |
| 跨仓流程与规范 | `docs/` 对应主题文件 |

- 结论必须带证据：性能带数值表，精度带误差数值 + 探针文件名，行号指向**当前**文件。

### 1.1 规格模板（`docs/specs/<id>-<subject>.md`）
一项工作一份，开工先写，随做随改；采纳或否决后**冻结**，不再长（后续工作另开一份）。五节固定：

1. **现象与已知事实** —— 每条带 `文件:行号`
2. **待验证假设** —— 按 预期收益 × 验证成本 排序，先做最便宜的
3. **验收条** —— 每条一个 case id = `<spec-id>#<n>`
4. **覆盖区间与回退路径** —— 生效区间 + 失效区间走哪条（D8）
5. **结论** —— A/B 数值表 + wins + 测量条件（D12）

**双向对账**：验收条 = case id = 测试标题。测试标题写 `@case <spec-id>#<n>`；规格有而测试无、或测试有而规格无，都算红——缺一条即红。缺陷回归同网不同表，标 `@repro <缺陷号>`。

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

### 3.1 代码行数预算

> 决策依据见 [D18](<./DECISIONS.md>)；检查器 `scripts/check_line_budget.py`；豁免登记见 [line_budget_exemptions.md](<./line_budget_exemptions.md>)。

**口径**：按**文件总行数**计（含空行与注释；`--count-effective` 切换为近似有效行）。作用域 = `src/**/*.rs`；`tests/`、`example/`、`bench/` 不在此预算内（探针另有上限，见 §5）。

**两档阈值**：

| 档 | 行数 | 含义 |
|---|---|---|
| 软上限 | 300 | 新建文件的目标上限；存量文件超过此线即进入"冻结不增" |
| 硬上限 | 600 | 新建文件不得超过；存量文件超过此线只许拆不许增 |

**判定表**：

| 情形 | 判定 |
|---|---|
| 新建 ≤ 300 行 | OK |
| 新建 301–600 行 | WARN —— 提交信息里必须写明为什么不拆 |
| 新建 > 600 行 | ERROR —— 必须拆 |
| 存量 ≤ 300 行（增减都算） | OK |
| 存量 > 300 行且行数增加 | ERROR —— 超限后冻结不增 |
| 存量 > 300 行但行数减少 | OK —— 降债，方向正确 |

**为什么存量是"冻结不增"而不是一刀切整改**：`src/` 现有 127 个文件 / 92.7k 行，其中 80 个 > 300 行、47 个 > 600 行（占全部行数的 76.6%）。要求一次全量拆分等于一次大重构，与"一个切片一提交"（[WORKFLOW.md](<./WORKFLOW.md>) §2）直接冲突，还会在精度/性能证据链上制造无法复核的大 diff。冻结不增让预算随时间单调收敛：**每个超限文件只能变短，不能变长**。

**拆分方向**（按优先级，先便宜后彻底）：
1. 抽纯数据 / 常量表 → 独立文件（`*_tables.rs`、`*_consts.rs`）。
2. 抽单一职责模块 → 同名目录 + `mod.rs` 只做 re-export（`parser.rs` → `parser/`）。
3. 抽 trait / 泛型参数，消掉"按模型 / 按 dtype / 按 shape 分支"的重复体（与 D5 family path 同向）。
4. 抽跨层复用的算子 / 工具 → `src/utils/`。

**豁免**（不计入预算）：生成代码（`*_generated.rs`）、纯查表/常量数据、探针与测试（探针上限 800 行）。豁免须在文件头写 `// line-budget: exempt <理由>`，**并**登记到 [line_budget_exemptions.md](<./line_budget_exemptions.md>)——无理由的豁免视为违规。

**检查**：
```bash
python3 scripts/check_line_budget.py             # 全仓 src/
python3 scripts/check_line_budget.py --changed   # 提交前：只看相对基线有改动的文件
python3 scripts/check_line_budget.py --debt      # 盘点存量债务（不判失败）
```

**函数体**：单个函数 ≤ 80 行；> 100 行必须拆。此项由 code review 判定，不自动检查（函数边界识别不可靠，宁可漏报不误报）。

## 4. 日志与开关
- `src/` 内禁止裸 `println!`/`eprintln!`（`tests/probes/`、`example/` 与一次性汇总除外）。诊断用 `tracing::debug!`，错误用 `error!`（D9）。
- 会重复打的日志用 `Once`/`OnceLock` 收敛或加 env 门控。
- 开关全 `XINFER_` 前缀、三族、默认值安全（D10）；新开关在提交信息里写明默认值与影响面。

## 5. 探针
- 命名 `<subject>_<kind>_probe.rs`，`kind ∈ {precision, perf, breakdown, overhead}`，放 `tests/probes/`。
- **必须**在 `Cargo.toml` 注册 `[[example]]`；精度探针配 `compare_*_probe.py`，用 `precision_metrics.py` 的 `report` 统一输出（D11）。
- 产物落 `$XINFER_PROBE_DIR`：产物不入 git，结论入 git。
- 单个探针 ≤ 800 行；超了通常说明混了多个 subject，拆成多个探针（每个 subject 一份产物 + 一份对比脚本）。

## 6. 硬性禁止
- ❌ 无 A/B 实测的性能结论（"感觉更快"不算证据，D12）。
- ❌ 只改 shared layout / 换 tile 参数就重试已否决方案（D4）。
- ❌ 把探针/合成 shape 的结果当生产结论（必须标 shape、batch、TP、是否 graph）。
- ❌ 在 detached HEAD 上留正式提交，或把工作留在未提交的工作区（D7）。
- ❌ 用 `git cherry` / `--cherry` 当"没有回退"的证据——手工重写的回退查不出（D8）。
- ❌ 把 1Cat 的 fast path 整条复制进某个模型（走 family path，D5）。
- ❌ 静默改精度：改了就要说、要测、要写进契约（D6）。
- ❌ 让超限文件继续变大（存量冻结不增，D18）；拆文件不许"拆完再塞回去"。
