//! Where is the real FlashInfer prefill boundary on SM70 (Volta)?
//!
//! FlashInfer documents "SM75 (Turing) and later", but the decode kernels
//! were measured to work on a V100 while the prefill kernels do not.  The
//! reason is the opt-in shared-memory budget: SM70/SM75 offer 64 KiB against
//! the 164 KiB of SM80+, and at head_dim 256 the q/k/v staging union already
//! fills that 64 KiB, so `cudaFuncSetAttribute()` and the launch itself fail
//! with CUDA_ERROR_LAUNCH_FAILED.
//!
//! That means the real cut-off is a *shared-memory* limit, not an
//! architecture number: a smaller head_dim should still fit on SM70.  This
//! probe measures where the boundary actually is, so the fallback can be
//! based on the real constraint instead of a blanket "sm >= 80" rule.
//!
//! Every head_dim runs in its own process: a failed launch leaves the CUDA
//! context in an error state, which would poison every later case.
//!
//! Run with:
//!   XINFER_FI_SM70_PREFILL_PROBE=1 ./flashinfer_sm70_prefill_probe
//!   XINFER_FI_SM70_PREFILL_PROBE_HEAD_DIM=128 ./flashinfer_sm70_prefill_probe

use attention_rs::flashinfer;
use candle_core::{DType, Device, Tensor};

fn probe(
    device: &Device,
    head_dim: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
) -> anyhow::Result<bool> {
    let (kv_len, append_len, page_size) = (1024usize, 8usize, 64usize);
    let num_pages = kv_len.div_ceil(page_size);
    let prefix_len = kv_len - append_len;
    let dtype = DType::F16;

    let k_all = Tensor::randn(0f32, 1f32, (kv_len, num_kv_heads, head_dim), device)?.to_dtype(dtype)?;
    let v_all = Tensor::randn(0f32, 1f32, (kv_len, num_kv_heads, head_dim), device)?.to_dtype(dtype)?;
    let q = Tensor::randn(0f32, 1f32, (append_len, num_qo_heads, head_dim), device)?.to_dtype(dtype)?;

    let indices_host: Vec<u32> = (0..num_pages).map(|i| (i + 3) as u32).collect();
    let indptr_host = vec![0u32, num_pages as u32];
    let last_len_host = vec![((kv_len - 1) % page_size + 1) as u32];
    let indices = Tensor::from_vec(indices_host.clone(), (num_pages,), device)?;
    let indptr = Tensor::from_vec(indptr_host.clone(), (2,), device)?;
    let last_len = Tensor::from_vec(last_len_host.clone(), (1,), device)?;

    // FlashInfer NHD paged cache: [page, token, kv_head, head_dim].
    let mut k_cache = Tensor::zeros(
        (num_pages + 3, page_size, num_kv_heads, head_dim),
        dtype,
        device,
    )?;
    let mut v_cache = Tensor::zeros(
        (num_pages + 3, page_size, num_kv_heads, head_dim),
        dtype,
        device,
    )?;
    for page in 0..prefix_len.div_ceil(page_size) {
        let start = page * page_size;
        let len = (prefix_len - start).min(page_size);
        let k_page = k_all.narrow(0, start, len)?.unsqueeze(0)?;
        let v_page = v_all.narrow(0, start, len)?.unsqueeze(0)?;
        let physical_page = indices_host[page] as usize;
        k_cache = k_cache.slice_assign(
            &[
                physical_page..physical_page + 1,
                0..len,
                0..num_kv_heads,
                0..head_dim,
            ],
            &k_page,
        )?;
        v_cache = v_cache.slice_assign(
            &[
                physical_page..physical_page + 1,
                0..len,
                0..num_kv_heads,
                0..head_dim,
            ],
            &v_page,
        )?;
    }

    let batch_indices = Tensor::zeros((append_len,), DType::U32, device)?;
    let positions = Tensor::from_vec(
        (prefix_len as u32..kv_len as u32).collect::<Vec<_>>(),
        (append_len,),
        device,
    )?;

    flashinfer::append_kv_cache(
        &k_all.narrow(0, prefix_len, append_len)?,
        &v_all.narrow(0, prefix_len, append_len)?,
        &k_cache,
        &v_cache,
        None,
        None,
        &indices,
        &indptr,
        &last_len,
        Some(&batch_indices),
        Some(&positions),
    )?;

    let q_cu_seqlens_host = vec![0u32, append_len as u32];
    let plan = flashinfer::prefill_plan(
        device,
        &q_cu_seqlens_host,
        &indptr_host,
        &[kv_len as u32],
        append_len as u32,
        1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        None,
        Some(dtype),
        false,
    )?;
    let q_cu_seqlens = Tensor::from_vec(q_cu_seqlens_host, (2,), device)?;
    let out = flashinfer::prefill_with_plan(
        &q,
        &k_cache,
        &v_cache,
        None,
        None,
        &indices,
        &indptr,
        &last_len,
        &q_cu_seqlens,
        append_len as u32,
        page_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1.0 / (head_dim as f32).sqrt(),
        None,
        None,
        &plan,
        false,
    )?;

    // A launch failure can be deferred, so force a synchronization point and
    // make sure the result is actually usable.
    let sum = out.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    if !sum.is_finite() {
        return Ok(false);
    }
    Ok(true)
}

fn main() -> anyhow::Result<()> {
    if std::env::var("XINFER_FI_SM70_PREFILL_PROBE").is_err() {
        println!("SKIP: set XINFER_FI_SM70_PREFILL_PROBE=1 to run");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let sm = attention_rs::cuda_utils::sm_version(device.as_cuda_device()?).unwrap_or(0);
    println!("device: sm{sm}");

    let head_dim: usize = std::env::var("XINFER_FI_SM70_PREFILL_PROBE_HEAD_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let num_qo_heads: usize = std::env::var("XINFER_FI_SM70_PREFILL_PROBE_QO_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let num_kv_heads: usize = std::env::var("XINFER_FI_SM70_PREFILL_PROBE_KV_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    println!(
        "probing head_dim={head_dim} qo_heads={num_qo_heads} kv_heads={num_kv_heads} dtype=f16"
    );
    match probe(&device, head_dim, num_qo_heads, num_kv_heads) {
        Ok(true) => {
            println!("RESULT: OK head_dim={head_dim}");
            Ok(())
        }
        Ok(false) => {
            println!("RESULT: NON_FINITE head_dim={head_dim}");
            std::process::exit(2);
        }
        Err(e) => {
            println!("RESULT: ERROR head_dim={head_dim}: {e:?}");
            std::process::exit(1);
        }
    }
}
