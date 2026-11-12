//! Kernel-level performance probe for the gated delta rule (GDN) decode path.
//!
//! GDN keeps a recurrent state per (batch, v_head) and touches it on every
//! generated token, so the decode kernel runs for as many layers as the
//! model has.  It is a different bottleneck from the NVFP4 GEMM: the state
//! lives in shared memory and is read and written in full on every call.
//!
//! The probe times the decode call for a small model's head layout, so a
//! change to the kernel can be judged on the device rather than on the host.
//!
//! Run with:
//!   XINFER_GDN_PERF_PROBE=1 ./gdn_perf_probe

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

/// (batch, num_k_heads, num_v_heads, k_dim, v_dim) for a Qwen3-Next-style
/// linear-attention layer.
const CONFIGS: &[(usize, usize, usize, usize, usize)] = &[
    (1, 16, 32, 128, 128),
    (4, 16, 32, 128, 128),
    (16, 16, 32, 128, 128),
];

fn timed<F: FnMut() -> anyhow::Result<()>>(
    iters: usize,
    warmup: usize,
    mut f: F,
) -> anyhow::Result<f64> {
    for _ in 0..warmup {
        f()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}

fn main() -> anyhow::Result<()> {
    if std::env::var_os("XINFER_GDN_PERF_PROBE").is_none() {
        println!("SKIP GDN perf probe: set XINFER_GDN_PERF_PROBE=1 to run");
        return Ok(());
    }
    let device = Device::new_cuda(0)?;
    let iters: usize = std::env::var("XINFER_GDN_PERF_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let warmup: usize = std::env::var("XINFER_GDN_PERF_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    println!("# GDN decode performance");
    println!("iters={iters} warmup={warmup} (median of {iters} runs, ms)\n");
    println!("| case | batch | k_heads | v_heads | k_dim | v_dim | ms |");
    println!("|---|---|---|---|---|---|---|");

    for (batch, num_k_heads, num_v_heads, k_dim, v_dim) in CONFIGS {
        let q = Tensor::randn(0f32, 1f32, (*batch, *num_k_heads, *k_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1f32, (*batch, *num_k_heads, *k_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1f32, (*batch, *num_v_heads, *v_dim), &device)?
            .to_dtype(DType::BF16)?;
        let g = Tensor::randn(0f32, 1f32, (*batch, *num_v_heads), &device)?
            .to_dtype(DType::F32)?;
        let beta = Tensor::randn(0f32, 1f32, (*batch, *num_v_heads), &device)?
            .to_dtype(DType::F32)?;
        let mut state = Tensor::zeros(
            (*batch, *num_v_heads, *k_dim, *v_dim),
            DType::F32,
            &device,
        )?;
        let slots = Tensor::from_vec(
            (0..*batch as i64).collect::<Vec<i64>>(),
            (*batch,),
            &device,
        )?;

        let ms = timed(iters, warmup, || {
            let _ = attention_rs::gdn::gated_delta_rule_decode_slots_gqa(
                &q, &k, &v, &g, &beta, &mut state, &slots, 1.0,
            )?;
            Ok(())
        })?;

        println!(
            "| b{batch}_k{num_k_heads}_v{num_v_heads} | {batch} | {num_k_heads} | \
             {num_v_heads} | {k_dim} | {v_dim} | {ms:.4} |"
        );
    }
    Ok(())
}
