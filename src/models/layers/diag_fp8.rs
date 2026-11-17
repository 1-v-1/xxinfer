//! FP8 加载路径的一次性诊断（`XINFER_DIAG_NAN=1` 门控，D9）。
//!
//! 用途：把「引擎实际持有的 (weight, scale) 对」与检查点直读结果做位等比对，
//! 定位 e4m3 字节是否被当成整数值（见 `docs/specs/d2-mtp-verify-path.md` §54–§61）。
//! 独立成模块是为了满足 D18：`linear.rs` / `deltanet.rs` 均已超软上限且只许变短。

use candle_core::{DType, Result, Tensor};

/// 反量化后「逐输出通道 maxabs」的 min/mean/max。
/// 引擎里 fp8 权重是 `[K, N_shard]`，输出通道在最后一轴，故对倒数第二轴求 max。
pub fn channel_stats(w: &Tensor, s: &Tensor) -> Result<[f32; 3]> {
    let wt = w.to_dtype(DType::F32)?;
    let sb = s.to_dtype(DType::F32)?.broadcast_as(wt.dims())?;
    let r = wt.abs()?.broadcast_mul(&sb)?;
    let rm = r.max(r.dims().len() - 2)?;
    Ok([
        rm.min(0)?.to_scalar::<f32>()?,
        rm.mean(0)?.to_scalar::<f32>()?,
        rm.max(0)?.to_scalar::<f32>()?,
    ])
}

/// 打印某段投影实际持有的 (weight, scale) 对：dtype、形状、反量化分布、前 4 个值。
/// 先转 F32 再 abs：SM70 无 `uabs_u8` 内核，直接 abs 会 `CUDA_ERROR_NOT_FOUND`。
pub fn proj_pair(tag: &str, pair: Option<(&Tensor, &Tensor)>) {
    let (w, s) = match pair {
        Some(p) => p,
        None => {
            crate::log_info!("[nan] {tag}: 无 fp8 pair（走了别的量化路）");
            return;
        }
    };
    let f = |t: &Tensor| {
        t.to_dtype(DType::F32)
            .and_then(|t| t.abs())
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.max(0))
            .and_then(|t| t.to_scalar::<f32>())
    };
    let head = |t: &Tensor| {
        t.to_dtype(DType::F32)
            .and_then(|x| x.flatten_all())
            .and_then(|x| x.narrow(0, 0, 4))
            .and_then(|x| x.to_vec1::<f32>())
    };
    let (wa, sa) = (f(w), f(s));
    let rs = channel_stats(w, s);
    let (w4, s4) = (head(w), head(s));
    crate::log_info!(
        "[nan] {tag} fp8 pair: w_dtype={wdt:?} s_dtype={sdt:?} w_dims={wdim:?} s_dims={sdim:?} chan_maxabs[min/mean/max]={rs:?} w_first4={w4:?} s_first4={s4:?} w_absmax={wa:?} s_absmax={sa:?}",
        wdt = w.dtype(),
        sdt = s.dtype(),
        wdim = w.dims(),
        sdim = s.dims(),
    );
}
