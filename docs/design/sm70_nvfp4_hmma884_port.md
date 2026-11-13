# 1Cat-vLLM SM70/V100 NVFP4 移植报告（xxinfer）

本文记录 1Cat-vLLM 的 V100（SM70/Volta）NVFP4 优化移植到 xxinfer 的完整过程、
最终实现、验证方法与实测性能。**已否决的方案见同目录
`sm70_1cat_rejected_paths.md`（97 项）。**

---

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 移植内容 | 1Cat 的 SM70 NVFP4 **HMMA.884** tensor-core 路径 |
| 精度 | SM70 路径 vs 软件路径：**mismatched=0**（真实模型全部 GEMM 形状） |
| 真实推理 | 跑通，无崩溃、无 launch 失败 |
| kernel 性能 | 相对本项目最初的 SM70 实现 **快 2–4 倍**；与 1Cat 的软件 SIMT+LUT 路径**持平** |
| 端到端 | 未变快 —— 瓶颈在 `fp8_matmul_kernel`（FP8 量化 GEMM），不属于本次移植范围 |

**⚠️ 精度口径**：SM70（Volta）**没有** FP4 tensor core。NVFP4 在 SM70 上是
**W4A16**（激活保持 FP16，权重为 4-bit + FP8-E4M3 block scale，block size 16）。
**不得**宣称原生 W4A4。

---

## 2. 最终实现

### 2.1 文件

| 文件 | 内容 |
|---|---|
| `repos/attention.rs/src/kernels/src/nvfp4_gemm_sm70.cu` | HMMA.884 kernel + repack kernel + host 入口 |
| `repos/attention.rs/src/kernels/src/ffi.rs` | `nvfp4_repack_weight_sm70` / `nvfp4_matmul_hmma884_f16` 绑定 |
| `repos/attention.rs/src/nvfp4_linear.rs` | Rust 侧封装、权重缓存、SM70 路径选择、数值自检 harness |
| `repos/attention.rs/src/lib.rs` | `XINFER_NVFP4_SM70_CHECK` 开关 |
| `repos/attention.rs/src/kernels/build.rs` | 固定 out_dir，避免全量重编 |

### 2.2 计算原子

`mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` —— M=8, N=8, K=4，
两次链接覆盖 k=0..7。**一次 mma 覆盖整个 8×8 quad**（32 个 lane 一起）。

**权威 fragment 布局**（由穷举求解器 `test_solve.cu` 唯一确定）：

- **A**（8×4 row-major）：`m = (lane&3) + 4*(lane>=16)`；
  `a[0]={A[m][k0],A[m][k1]}`，`a[1]={A[m][k2],A[m][k3]}`
- **B**（4×8 col-major）：`n = (lane&3) + 4*(lane>=16)`；
  `b[0]={B[k0][n],B[k1][n]}`，`b[1]={B[k2][n],B[k3][n]}`
- **D**（8×8 f32，8 元素/lane）：
  `row = (lane&1) + (i&2) + 4*(lane>=16)`；
  `col = (lane&2) + (i&1) + 4*((i>>2)&1)`

**⇒ B fragment 必须持有同一列的 4 个连续 k**
**⇒ decoder 输出 `w0=(k0,k4), w1=(k1,k5), w2=(k2,k6), w3=(k3,k7)` 必须重打包为
`b_lo={pack(w0.lo,w1.lo), pack(w2.lo,w3.lo)}`（k=0..3）与
`b_hi={pack(w0.hi,w1.hi), pack(w2.hi,w3.hi)}`（k=4..7）。**

### 2.3 数据布局

| 张量 | 布局 | 说明 |
|---|---|---|
| 源权重 | `[N, K/2]` U8 | 低 nibble = 偶数 k |
| 源 scale | `[N, K/16]` U8 | FP8 E4M3 |
| repack 后权重 | `[N, K/8]` U32 | 同一 n 的 8 个连续 k 打包进一个字，**元素 j 在 nibble j**（连续，非交错） |
| repack 后 scale | `[N, K/8]` U32 | half 位存两份；16 元素组的 scale 复制到相邻两个 k block |

**⚠️ 踩过的坑**：repack 输出是 `[N, K/8]` 而 kernel 早期按 `[N, K/16]` 索引 ——
scale 布局不匹配是致命 bug。

### 2.4 性能设计

**CTA 形状**：k tile = 64 元素，2 个 warp（decode 路径 grid 够大时 4 warp）。
shared memory 4–8 KiB/CTA，保留高 occupancy。

**搬运**：`uint4`（16 B）向量化 —— tile 在 n-major/k-contiguous 布局下连续 4 个
k block 相邻。

**为什么小 tile 更快**（与直觉相反）：小 tile ⇒ shared memory 占用低 ⇒
occupancy 高 ⇒ 全局内存延迟靠 occupancy 掩盖，而不是靠减少 barrier。
实测见 §4。

---

## 3. 修复的 5 个 bug

| # | Bug | 症状 | 根因 |
|---|---|---|---|
| 1 | B fragment 的 k 序 | 数值全错 | B 必须是同列 4 个连续 k，decoder 输出需重打包 |
| 2 | shared memory 搬运量不足 | 数值全错 | 旧代码只搬了 2048 个字里的 128 个 |
| 3 | cb 循环残留 | 数值全错 | accumulator 被拆成 4 个 quad，实际一次 mma 就覆盖整个 8×8 |
| 4 | scale 布局不匹配 | 数值全错 | repack `[N,K/8]` vs kernel `[N,K/16]` |
| 5 | mma 内联汇编的 C/D 操作数别名 | 未定义行为 | 同一数组既作 C 又作 D |

另外修复：repack 里 `*(const uint32_t *)&h` 的 UB（改用 `memcpy`）；
`kE2M1Scale` 必须是 half2（`0x74007400U`）；scale 已含 global scale 时不得再广播。

---

## 4. 实测性能（真实 V100-SXM2-16GB）

### 4.1 kernel 级（`bench_final.cu`，链到生产 kernel）

| M | N | K | 最初实现 | 优化后 | 加速 |
|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 0.5682 ms | **0.2625 ms** | 2.16× |
| 8 | 4096 | 4096 | 0.5790 ms | **0.2747 ms** | 2.11× |
| 1 | 17408 | 5120 | 2.0079 ms | **0.4792 ms** | 4.19× |
| 8 | 17408 | 5120 | 1.9369 ms | **0.4863 ms** | 3.98× |
| 8 | 5120 | 8704 | 1.1012 ms | **0.6270 ms** | 1.76× |
| 32 | 17408 | 5120 | 2.3457 ms | **1.2404 ms** | 1.89× |

### 4.2 k tile × warp 数扫描（`bench_sweep.cu`）

| 形状 | tileK=512 w1 | 256 w1 | 128 w1 | 64 w1 | 64 w2 | 32 w4 |
|---|---|---|---|---|---|---|
| M=1 N=4096 K=4096 | 0.3087 | 0.3111 | 0.2966 | 0.2697 | 0.2494 | **0.2274** |
| M=8 N=17408 K=5120 | 1.0610 | 0.7357 | 0.4626 | 0.4684 | **0.4448** | 0.4636 |
| M=8 N=5120 K=8704 | 0.5946 | 0.5987 | 0.6167 | 0.5955 | 0.5576 | **0.5504** |
| M=32 N=17408 K=5120 | 3.5478 | 1.9370 | 1.4766 | 1.3096 | **1.2484** | 1.8760 |

### 4.3 nsys 端到端 kernel 占比（48 token 解码）

软件路径：`fp8_matmul_kernel` 35.1% + `nvfp4_matmul_smallm_kernel` 34.7%
SM70 路径：`fp8_matmul_kernel` 31.1% + 本 kernel 33.5%

**⇒ SM70 NVFP4 kernel（117.3 ms）与 1Cat 软件 NVFP4 路径（107.4 ms）持平，
已追平 1Cat 的 SIMT+LUT 实现。**

**⇒ 端到端瓶颈是 `fp8_matmul_kernel`（FP8 量化 GEMM），不属于 1Cat SM70
NVFP4 移植范围。**

---

## 5. 验证方法

### 5.1 数值验证

**独立参考 + 真实形状**（`test_prod_kernel2.cu`，链到生产 kernel）：

| M | N | K | max_err | max_rel | bad |
|---|---|---|---|---|---|
| 8 | 8 | 16 | 0.000000 | 0 | 0/16 |
| 8 | 32 | 512 | 0.054688 | 4.7e-4 | 0/64 |
| 1 | 4096 | 4096 | 0.242188 | 4.9e-4 | 0/64 |
| 8 | 17408 | 5120 | 0.203125 | 4.7e-4 | 0/128 |
| 32 | 4096 | 4096 | 0.242188 | 4.9e-4 | 0/128 |
| 1 | 5120 | 8704 | 0.492188 | 4.4e-4 | 0/64 |

残余误差全部来自 fp16 累加舍入。

### 5.2 进程内交叉验证（`XINFER_NVFP4_SM70_CHECK=1`）

真实模型上 SM70 kernel vs 软件路径：

```
M=7  N=5120  K=8704  max_abs_diff=6.103516e-5  mismatched=0/35840
M=22 N=17408 K=5120  max_abs_diff=9.765625e-4  mismatched=0/382976
M=22 N=5120  K=8704  max_abs_diff=2.441406e-4  mismatched=0/112640
```

### 5.3 方法论教训

1. **验证必须链接生产代码**，不能是测试自带的副本。
2. **fragment 布局必须用满秩矩阵 + 独立参考做穷举求解**。用"自洽"的参考
   （参考里编码了同样的假设）会让 4 重的错误映射也 PASS。
3. **测试的 CPU 参考本身也要审查**。`test_prod_kernel2` 曾把 E2M1 的符号位
   丢掉（`code &= 0x7`），误报成"kernel 全错"。
4. **单 k block 正确 ≠ 整个 GEMM 正确**。必须测多 k block / 多 tile / 真实尺寸。
5. **"kernel 算错时的性能数字"没有意义**。早期记录的 0.2055 ms 是在大部分
   权重没搬进 shared memory 时测的；真实值是 0.26–0.49 ms。
6. **CUDA graph capture 期间不能做 device→host 拷贝**。数值自检 harness
   必须用 `cuStreamIsCapturing` 判定并跳过。

---

## 6. 使用方式

```bash
export ENABLE_FLASHINFER_SOFTWARE_FP8=1
export XINFER_NVFP4_SM70_HMMA884=1   # 默认在 SM70 上自动启用；=0 强制回退软件路径
export XINFER_NVFP4_SM70_CHECK=1     # 可选：数值自检（会拖慢，仅验证用）
```

选择条件（`use_sm70_hmma884`）：`sm_version(dev) == 70` 且 `dtype == F16`
且 `k % 16 == 0` 且 `weight_scale_swizzled.is_none()`。

`input_scale` 在 SM70 路径被有意忽略（激活保持 FP16），与软件路径一致。

---

## 7. 已知限制

- **只覆盖 NVFP4**。FP8 量化的线性层（`LnFp8` → `fp8_matmul_kernel`）仍是
  端到端最大瓶颈，未优化。
- **M > 8 的 tile 粒度**：kernel 每 CTA 覆盖 8 行（M），M 较大时 grid 的
  y 维会放大。
- `input_scale` 不参与 SM70 路径。
- 数值自检 harness 一次运行只检查第一次调用。
