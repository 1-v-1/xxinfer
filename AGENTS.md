# AGENTS.md — xinfer

SM70/V100 推理引擎移植。本仓是主项目；`../attention.rs`（内核）、`../candle`（张量库）、`../cudarc`（driver 封装）是三个按 rev 锁定的 fork。参考实现 `/Volumes/SMB重要/1Cat-vLLM`（`bcd581d230`）**只读**。

## 动手前必读
- 改架构/契约/内核路径 → [docs/DECISIONS.md](<./docs/DECISIONS.md>)（含已否决方案与边界）。
- 动 `src/models/layers/`、`src/utils/graph.rs`、`src/speculative/` → [docs/ARCHITECTURE.md](<./docs/ARCHITECTURE.md>) §3/§4。
- 不认识的术语 → [docs/GLOSSARY.md](<./docs/GLOSSARY.md>)，不要凭字面猜。

## 硬约束
1. 工作落点唯一 = `perf/sm70-volta`：提交前 `git status` 不得是 `HEAD detached`，提交后立刻 push，离开前 `git status --porcelain` 为空（D7）。
2. revert 带三件套（被回退 sha / 实测证据 / 替代路径与覆盖区间），手工回退额外自查（D8）。
3. 先过精度再谈性能：精度按声明的容差判，性能给 A/B 数值表 + wins + 测量条件（D6/D12）。
4. 被证伪的方案当场进 `docs/design/sm70_1cat_rejected_paths.md`，标 `否决`/`证伪`/`未准入` = 禁止重试（D4）。
5. `src/` 内禁止裸 `println!`/`eprintln!`；重复日志用 `Once`/`OnceLock` 收敛或 `XINFER_DIAG_*` 门控（D9）。

## 常用命令
```bash
cargo build --release
cargo run --release --example nvfp4_precision_probe     # 探针须已注册 [[example]]
python3 tests/probes/compare_nvfp4_probe.py $XINFER_PROBE_DIR/<产物>
cargo run --release -- --model <path> ...
```

## 文档分工
`CONVENTIONS.md` 写代码时的规则 · `WORKFLOW.md` 推进流程 · `ARCHITECTURE.md` 代码在哪 · `TESTING.md` 怎么算验证过 · `DECISIONS.md` 已拍板决策 · `GLOSSARY.md` 术语 · `design/` 内核契约与已否决清单。
**聊天记录不是真相源**：结论、否决、踩坑都要落到文件。
