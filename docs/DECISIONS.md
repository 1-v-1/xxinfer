# 决策记录（ADR）

> 分工：本文件 = 已拍板的决策；写代码的规则见 [CONVENTIONS.md](<./CONVENTIONS.md>)，推进流程见 [WORKFLOW.md](<./WORKFLOW.md>)。
> 规则：改变架构 / 契约 / 内核路径的决定必须新增一条。AI 会话丢上下文时**以本文件为准**。
> 状态：✅ 生效 | ⛔ 被取代（注明取代者）。引用用编号，代码注释与提交信息可回链。
> 门槛：没有实测支撑的决策不进本文件。

## D1 ✅ 四仓分离，依赖按 git rev 锁定
- 背景：path 依赖会让 `cargo install --git` 不可用（`b9b6d77` 踩过）。
- 决定：`xinfer` / `attention.rs` / `candle` / `cudarc` 四个独立 fork，依赖单向无环，每条边锁 `rev =`。
- 影响：改一个 rev = 换整条链的二进制行为 → 见 D3。

## D2 ✅ 1Cat 是唯一参考实现，做行为级移植
- 决定：`/Volumes/SMB重要/1Cat-vLLM`（`bcd581d230`）**只读**。抄 kernel 连 **shared memory 布局一起抄**——1Cat 的 fragment 约定不是 PTX 标准约定（B fragment 持同一列的 4 个连续 k，GEMM shared 是 n-major），**不得**改用 CUTLASS 的 `ldmatrix` 套路。
- 影响：引用上游一律带 commit hash + `文件:行号`（行号会漂，commit 不会）。

## D3 ✅ 依赖 rev 变更 = 契约级变更
- 决定：任何 `rev =` 变更走 [WORKFLOW.md](<./WORKFLOW.md>) §3 全流程：精度探针先过 → 性能 A/B 表 → 再进 `perf/sm70-volta`。
- 影响：跨仓升级的 diff = kernel + rev + 证据 三件套，缺一件即未完成。

## D4 ✅ 已否决路径清单化（禁止重试）
- 决定：被实测证伪的方案当场写进 [design/sm70_1cat_rejected_paths.md](<./design/sm70_1cat_rejected_paths.md>)，标注 `否决` / `证伪` / `未准入` 三态，并记机理（SASS 指令数、regs/thread、CTA/SM 占用、stall 分布、A/B 数值表）。
- 影响：条目 = 禁止重试；重试须先在清单里开"重试理由"。

## D5 ✅ 新能力走 family path，不做模型专属 fast path
- 决定：新内核挂在既有 family selector 之后；不新增 dispatcher、不新增 policy store。
- 影响：新模型接入的 diff = 模型映射 + 已有 stage 的参数。

## D6 ✅ 精度不默认 bitwise 相等
- 决定：按算子/dtype 选定误差指标与上限，**先定容差再跑候选**；禁止失败后放宽阈值凑通过。
- 影响：内存/索引/掩码有效性、请求隔离、状态提交回滚、NaN/Inf 检查**必须严格**，不适用容差。

## D7 ✅ 唯一工作分支 + 提交即推送
- 背景：5 个正式提交（`f74d315`…`4588819`）曾只活在 `HEAD detached from e83f8c5`，`git rev-list --branches` 一个都够不到；`attention.rs` 另有从未提交的 +103 行只在工作区。
- 决定：工作落点唯一 = `perf/sm70-volta`；提交后立刻 push；**不许把工作留在工作区**（未提交 = `git checkout` 直接丢，连 reflog 都救不了）。
- 影响：离开一个仓前 `git status --porcelain` 必须为空。

## D8 ✅ revert 带三件套 + 手工回退要额外自查
- 背景：`3e59e24` 干净回退了 `972355f`（blob 互逆）但没留覆盖区间分析 → 新实现只到 `m<=40`，`m ∈ [41,255]` 出现无记录缺口。`d8f0721` 撤销 `a11d4fa` 后最终形态是 env 开关且**默认仍开** = 默认行为回到修复前，当时未记录。
- 决定：revert 必须同时写 ① 被回退的 sha ② 实测证据 ③ 替代路径与覆盖区间；revert 后重跑基线。
- 影响：非 `git revert` 生成的回退 `patch-id` 查不出（`3e59e24`=`d12b6dd0` ≠ `972355f` 逆的 `c62761ee`），须用 `git show <sha>:<file>` 逐行核对被撤销的行是否复活。

## D9 ✅ 诊断日志收敛到 once-per-process + debug 级
- 背景：SM70 上 flashinfer 每层每次打 `tracing::warn!` 刷屏；`57a6138` 改成 `static ONCE` + `debug!`，至今在 `144a07f` 生效。
- 决定：`src/` 内禁止裸 `println!`/`eprintln!`；重复打的日志用 `Once`/`OnceLock` 收敛或加 env 门控。
- 影响：日志不淹没真错误，也不污染测量（→D12）。

## D10 ✅ 开关统一前缀 + 默认值安全
- 决定：三族——`XINFER_DIAG_*`（诊断，默认关）、`XINFER_*_PROBE`（探针参数）、其余功能开关（默认安全值）。
- 影响：读取统一走 `src/utils/env.rs` 的 `OnceLock` 缓存，不在热路径反复 `env::var`。

## D11 ✅ 探针必须注册 + 证据目录统一
- 决定：探针命名 `<subject>_<kind>_probe.rs`，**必须**在 `Cargo.toml` 注册 `[[example]]`；精度探针配 `compare_*_probe.py`；产物落 `$XINFER_PROBE_DIR`。
- 影响：产物不入 git，结论入 git。

## D12 ✅ 性能结论必须有 A/B 数值表 + 测量条件
- 决定：同条件交替采样（100 轮）的数值表 + wins 计数 + 测量条件（shape/batch/TP/是否 graph/时钟是否锁定）。
- 影响：TTFT/prefill 与纯 decode 不互相外推；日志频率不是指标。

## D13 ✅ 显存预算按实际分配门控
- 背景：MTP 权重多占 ~1.3GB/卡（available 1.83→0.55GB），TP2 上 MTP 与 CUDA graph 互斥。
- 决定：预留按实际启用的能力门控（`mtp_enabled` 关则不预留，`577870d`）；transient 随 `max_num_seqs × chunk` 缩放，不写死倍数（`e83f8c5`）。
- 影响：预算由开关与形状推导，不再拍脑袋。

## D14 ✅ 图捕获期用 `cuMemAlloc` + capture-time slab，消灭 MEM_ALLOC 节点
- 背景：捕获期每次分配走 `cuMemAllocAsync` → 图里出现 MEM_ALLOC 节点（实测 1766 个）→ 每次 replay 重分配 → decode 掉到 10 tok/s。PyTorch 默认后端是普通 `cudaMalloc`，图里零 MEM_ALLOC，这才是它不掉速的第一性原因。
- 决定：**方案 A′**——在 `cudarc` 实现捕获期 slab（≈200 行）：捕获期临时切 `RELAXED` capture mode 调普通 `cuMemAlloc`，把指针烧进图节点；slab 生命周期绑定图；捕获期 free **只记账 + 归还 freelist**，绝不 `cuMemFree`；新分配优先取 freelist。`candle` 零代码改动（只 bump rev），`xinfer` ≈120 行并在捕获前后开关 slab。
- 影响：`AUTO_FREE_ON_LAUNCH` 不再是依赖（有它也无东西可 free）；验证手段是 `cuGraphGetNodes` + `cuGraphNodeGetType` 数 `CU_GRAPH_NODE_TYPE_MEM_ALLOC`（nsys 看不到）。捕获前须 ≥1 次 NONE 模式 dummy run + `cudaDeviceSynchronize()`，使捕获期分配序列与 replay 期完全一致。

## D15 ✅ 图内状态与 collective 的形态契约
- 决定：FULL graph 捕获**整个 forward**（全部层 + attention），不是只捕 attention 子图。GDN 循环状态存在 KV cache 的两个 tensor（`conv_state_cache` / `ssm_state_cache`，恒 F32），按 `state_indices` 选 slot **原地推进**；捕获期产出的每个中间 tensor 地址在 replay 期必须有效，且不得被分配给图外新 tensor。
- 影响：vocab all-gather 留在图外；1Cat 删的是 NCCL 的 event bridge（因 conditional node body 不允许 event/host 节点），all-gather kernel 本身保留在图里。
