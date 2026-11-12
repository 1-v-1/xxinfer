//! Kernel-level performance probe for the NVFP4 GEMM on the current GPU.
//!
//! The accuracy probes answer "is it correct"; this one answers "is it fast",
//! which is what decides whether a decode-path optimization is worth keeping.
//! It measures the shapes that matter for a V100 decode workload:
//!
//!   * decode:  m = 1 (single-token), the memory-bound small-M path
//!   * decode:  m = 8 (speculative / MTP batch)
//!   * prefill: m = 128 (wmma path)
//!
//! Run with:
//!   XINFER_NVFP4_PERF_PROBE=1 ./nvfp4_perf_probe
//!
//! The reported number is the median of several timed iterations after a
//! warm-up, so a noisy machine does not produce a misleading verdict.

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

/// Shapes (m, n, k) for a 27B-class MoE model: the FFN projections dominate
/// the NVFP4 GEMM share of the critical path.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    // Decode: the shapes that decide the per-token latency.
    ("decode_m1", 1, 4096, 4096),
    ("decode_m4", 4, 4096, 4096),
    ("decode_m8", 8, 4096, 4096),
    ("decode_m16", 16, 4096, 4096),
    ("decode_m32", 32, 4096, 4096),
    // Prefill: vary K to see whether the wmma path scales with the work or
    // is dominated by per-k-step overhead, and vary M for the tile shape.
    ("prefill_m128", 128, 4096, 4096),
    ("prefill_m128_k8192", 128, 4096, 8192),
    ("prefill_m512", 512, 4096, 4096),
];

fn timed<F: FnMut() -> Result<(), candle_core::Error>>(
    iters: usize,
    warmup: usize,
    mut f: F,
) -> Result<f64, candle_core::Error> {
    for _ in 0..warmup {
        f()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3); // ms
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}

fn main() -> anyhow::Result<()> {
    if std::env::var_os("XINFER_NVFP4_PERF_PROBE").is_none() {
        println!("SKIP NVFP4 perf probe: set XINFER_NVFP4_PERF_PROBE=1 to run");
        return Ok(());
    }
    let device = Device::new_cuda(0)?;
    let iters: usize = std::env::var("XINFER_NVFP4_PERF_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let warmup: usize = std::env::var("XINFER_NVFP4_PERF_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let dtype = DType::F16;
    println!("# NVFP4 GEMM performance (m, n, k) on this GPU");
    println!("iters={iters} warmup={warmup} (median of {iters} runs, ms)\n");
    // XINFER_NVFP4_PERF_SHAPE="name,m,n,k" runs a single shape, which keeps
    // the kernel-level timings unambiguous when several shapes are profiled
    // together.
    if let Ok(spec) = std::env::var("XINFER_NVFP4_PERF_SHAPE") {
        let parts: Vec<&str> = spec.split(',').collect();
        if parts.len() != 4 {
            println!("bad XINFER_NVFP4_PERF_SHAPE: {spec}");
            return Ok(());
        }
        let one: (&str, usize, usize, usize) = (
            parts[0],
            parts[1].parse().unwrap(),
            parts[2].parse().unwrap(),
            parts[3].parse().unwrap(),
        );
        let all: Vec<(&str, usize, usize, usize)> = vec![one];
        return run_shapes(&all, &device, dtype, iters, warmup);
    }

    run_shapes(SHAPES, &device, dtype, iters, warmup)
}

fn run_shapes(
    shapes: &[(&str, usize, usize, usize)],
    device: &Device,
    dtype: DType,
    iters: usize,
    warmup: usize,
) -> anyhow::Result<()> {
    println!("| case | m | n | k | ms | TFLOP/s | GB/s |");
    println!("|---|---|---|---|---|---|---|");
    for (name, m, n, k) in shapes.iter().copied() {
        let input = Tensor::randn(0f32, 1f32, (m, k), &device)?.to_dtype(dtype)?;
        // NVFP4 weights: [N, K/2] packed bytes plus [N, K/16] fp8 block scales.
        let weight = Tensor::randn(0f32, 1f32, (n, k / 2), &device)?.to_dtype(DType::U8)?;
        let scales = Tensor::randn(0f32, 1f32, (n, k / 16), &device)?.to_dtype(DType::U8)?;

        // Batch the launches and read the result back once at the end: a
        // per-iteration device-to-host copy costs ~100 us on this host and
        // would swamp the kernel time being measured.
        let mut out = None;
        let ms = timed(iters, warmup, || {
            out = Some(attention_rs::nvfp4_linear::nvfp4_matmul(
                &input,
                &weight,
                &scales,
                1.25,
                1.0,
                None,
                m >= 32,
                None,
                None,
            )?);
            Ok(())
        })?;
        if let Some(o) = out {
            let _ = o.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        }

        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let tflops = flops / (ms * 1e-3) / 1e12;
        // The weights are FP4, so a decode step moves K/2 bytes per output
        // row plus one FP8 scale per 16 elements.
        let bytes = (n as f64) * ((k as f64) / 2.0 + (k as f64) / 16.0);
        let gbps = bytes / (ms * 1e-3) / 1e9;
        println!(
            "| {name} | {m} | {n} | {k} | {ms:.4} | {tflops:.3} | {gbps:.1} |"
        );
    }
    Ok(())
}
