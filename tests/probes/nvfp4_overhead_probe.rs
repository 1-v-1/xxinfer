//! Where does the per-call cost of the NVFP4 decode path go?
//!
//! Two lessons are baked into this probe, both learned the hard way on a
//! ppc64le V100 host:
//!
//! 1. A device-to-host copy per iteration costs ~100 us here, which is
//!    larger than the kernel being measured.  The NVFP4 perf probe
//!    therefore batches its launches and reads the result back once.
//! 2. The host-side launch cost is ~16 us, so a single timed call measures
//!    the host, not the device.  Use the kernel timings that nsys reports
//!    for the device-side number.
//!
//! This probe attributes the host-side part, which is what a serving loop
//! pays per projection.
//!
//! Run with:
//!   XINFER_NVFP4_OVERHEAD_PROBE=1 ./nvfp4_overhead_probe

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn bench<F: FnMut()>(label: &str, iters: usize, mut f: F) {
    for _ in 0..10 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("{label:<44} {us:>9.2} us");
}

fn main() -> anyhow::Result<()> {
    if std::env::var_os("XINFER_NVFP4_OVERHEAD_PROBE").is_none() {
        println!("SKIP NVFP4 overhead probe: set XINFER_NVFP4_OVERHEAD_PROBE=1 to run");
        return Ok(());
    }
    let device = Device::new_cuda(0)?;
    let iters = 200;

    let input = Tensor::randn(0f32, 1f32, (1, 4096), &device)?.to_dtype(DType::F16)?;
    let weight = Tensor::randn(0f32, 1f32, (4096, 2048), &device)?.to_dtype(DType::U8)?;
    let scales = Tensor::randn(0f32, 1f32, (4096, 256), &device)?.to_dtype(DType::U8)?;

    bench("Tensor::zeros((1, 4096)) f16", iters, || {
        let _ = Tensor::zeros((1, 4096), DType::F16, &device).unwrap();
    });
    bench("to_dtype(F32) of a 4096-element tensor", iters, || {
        let _ = input.to_dtype(DType::F32).unwrap();
    });
    // Launch only: the host-side cost of the wrapper.
    bench("nvfp4_matmul launch only (no sync)", iters, || {
        let _ = attention_rs::nvfp4_linear::nvfp4_matmul(
            &input,
            &weight,
            &scales,
            1.25,
            1.0,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    });
    // Adding the device sync shows how much of a naive measurement is the
    // wait rather than the work.
    bench("nvfp4_matmul + device sync", iters, || {
        let out = attention_rs::nvfp4_linear::nvfp4_matmul(
            &input,
            &weight,
            &scales,
            1.25,
            1.0,
            None,
            false,
            None,
            None,
        )
        .unwrap();
        let _ = out
            .to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
    });
    Ok(())
}
