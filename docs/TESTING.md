# 验证体系

> 分工：CONVENTIONS §5 = 探针规则；WORKFLOW §1/§5 = 流程；本文件 = 分几层、怎么跑、怎么算过。
> 前提：常规回归全走探针（合成 shape、可重复）；真模型端到端只手动冒烟。

**一句话：先过精度再谈性能；数值用容差判，性能用 A/B 表判，边界用回退路径判。**

## 1. 四层
| 层 | 跑什么 | 依赖 | 何时 |
|---|---|---|---|
| L0 契约 | 开关/预算/区间一致性（§2） | 无（读代码） | 每次提交前 |
| L1 精度 | 内核数值正确性 | GPU | 每次内核改动后 |
| L2 端到端 | 真模型 prefill/decode/MTP verify | GPU + 模型 | 每次跨仓升级后 |
| L3 性能 | A/B 数值表 | GPU + 锁定时钟 | 每次性能改动后 |

L0 最便宜、最该先跑：本项目的失败模式大量是"改了开关没改预算""改了上限没改回退路径"。

## 2. L0 契约检查（无需 GPU）
1. 新开关默认值安全（关/保守），影响面写进提交信息。
2. 预算与开关对齐：关掉某能力时其 workspace/KV 预留同步归零（D13）。
3. **覆盖区间与回退路径成对，区间无空洞**——`56fb086` 的新 QPN8 只覆盖 `m<=40`，被回退的 `972355f` 覆盖 `2<=m<256`，`m ∈ [41,255]` 的回退路径在代码里存在但当时没写进任何文档（D8）。
4. `Cargo.toml` 的 `attention-rs rev` == 实际使用的 sha。
5. 新探针已注册 `[[example]]`。⚠ 现存漏项：`prefill_breakdown_probe` 未注册，其文件头注释写的 `cargo run --release --example prefill_breakdown_probe` 当前跑不起来。
6. `src/` 内无裸 `println!`/`eprintln!`，重复日志已 `Once`/`OnceLock` 收敛（D9）。
7. 生效区间 + 失效区间回退路径 + 结论表齐全。

## 3. L1 精度探针
`tests/probes/` 下 13 个 `*_probe.rs`（precision / perf / breakdown / overhead 四类，覆盖 NVFP4、FP8、FlashInfer、GDN、NCCL 错误码）+ 6 个 `compare_*_probe.py`，统一用 `precision_metrics.py` 的 `report` 输出。

```bash
cargo run --release --example nvfp4_precision_probe     # 产物落 $XINFER_PROBE_DIR
python3 tests/probes/compare_nvfp4_probe.py <产物路径>
```

判据（D6）：不默认 bitwise；容差写在比对脚本里、先定容差再跑候选；不放宽阈值凑通过；循环状态（GDN state、长输出漂移）用容差。比对**两条腿**：数值指标 + 任务质量（指令/格式遵循、重复/EOS 健康）。

## 4. L3 性能纪律（D12）
同条件交替 A/B（100 轮），给数值表 + wins 计数，不写百分比口头结论：

| KV blocks | baseline | candidate | delta | wins |
|---|---|---|---|---|
| 1 (random) | 55.808 us | 55.936 us | -0.229% | 30/100 |

测量条件写全：shape、batch、TP/EP、是否 graph、dtype/量化格式、时钟是否锁定。TTFT/prefill 与纯 decode 不互推；量真实路径（含转换、分配、同步、launch）；日志频率不是指标；模型复用类结论需第二个兼容模型的证据。

## 5. 证据
产物（`.bin`、报告）落 `$XINFER_PROBE_DIR`，**不入 git**；数值表、结论、否决理由入 git（`DECISIONS.md` / 已否决清单 / `SM70_VOLTA_FINDINGS.md`）。评审要看的是数字和条件，不是二进制。
