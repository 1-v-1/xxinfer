//! 逐通道标度 + QPN8 打包路的精度探针（独立差分产物，与 `XINFER_DIAG_*` 无关）。
//!
//! 现有 `fp8_precision_probe` 只覆盖 `[128, 128]` 分块；本探针覆盖本模型的实际形态：
//! 权重 `[N, K]` e4m3 码 + 每输出通道一个标度 `[N, 1]`，块宽 `[1, K]`。
//! 产物与 `fp8_matmul_fallback`（朴素反量化 matmul）对，判「打包路是否改变量级」。
use anyhow::{Context, Result};
use attention_rs::fp8_linear::{fp8_matmul, fp8_matmul_fallback};
use candle_core::{DType, Device, Tensor};
use std::io::Write;
use std::path::PathBuf;

fn record(file: &mut std::fs::File, name: &str, tensor: &Tensor) -> Result<()> {
    let cpu = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let values = cpu.flatten_all()?.to_vec1::<f32>()?;
    file.write_all(&(name.len() as u64).to_le_bytes())?;
    file.write_all(name.as_bytes())?;
    file.write_all(&(cpu.dims().len() as u64).to_le_bytes())?;
    for dim in cpu.dims() {
        file.write_all(&(*dim as u64).to_le_bytes())?;
    }
    file.write_all(&(values.len() as u64).to_le_bytes())?;
    for value in values.iter() {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0).context("CUDA device 0 is required")?;
    let path = std::env::var_os("XINFER_QPN8_PROBE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/xinfer_qpn8_channel_probe.bin"));
    let mut file = std::fs::File::create(&path)?;
    file.write_all(b"XINFQPN8")?;

    let sm = attention_rs::cuda_utils::sm_version(dev.as_cuda_device()?).unwrap_or(0);
    let dtype = if sm < 80 { DType::F16 } else { DType::BF16 };
    record(&mut file, "meta/sm", &Tensor::new(&[sm as f32], &dev)?)?;

    // 每通道一个标度，且各通道量级不同 —— 若标度被按错轴施加，结果量级会变。
    let (n, k) = (64usize, 64usize);
    let codes: Vec<u8> = (0..(n * k)).map(|i| [0x38u8, 0xb8, 0x40, 0xc0][i % 4]).collect();
    let weight = Tensor::from_vec(codes, (n, k), &dev)?;
    let scales: Vec<f32> = (0..n).map(|r| 1e-4 * (1.0 + r as f32)).collect();
    let scale = Tensor::from_vec(scales.clone(), (n, 1), &dev)?;
    let input = Tensor::randn(0f32, 1f32, (2, k), &dev)?.to_dtype(dtype)?;

    let packed = fp8_matmul(&input, &weight, &scale, None, &[1, k], true)?;
    let gold = fp8_matmul_fallback(&input, &weight, &scale, &[1, k])?;
    for (name, tensor) in [("packed", &packed), ("gold", &gold), ("scale", &scale)] {
        record(&mut file, name, tensor)?;
    }

    // 真实形态：`in_proj_z` 的本地分片是 [3072, 5120]，激活量级 ~4.8e1（见 §50.4）。
    // 小形状下两路一致（§70），本段判真实形状/split-k 分支下是否仍一致。
    let (rn, rk) = (3072usize, 5120usize);
    let rcodes: Vec<u8> = (0..(rn * rk)).map(|i| [0x38u8, 0xb8, 0x40, 0xc0][i % 4]).collect();
    let rweight = Tensor::from_vec(rcodes, (rn, rk), &dev)?;
    let rscales: Vec<f32> = (0..rn).map(|r| 3.4e-2 * (1.0 + (r % 900) as f32 / 900.0)).collect();
    let rscale = Tensor::from_vec(rscales.clone(), (rn, 1), &dev)?;
    // 批大小取 37 = 引擎 prefill 段的 token 数（§72：36× 只在 prefill 出现）。
    let rinput = (Tensor::randn(0f32, 1f32, (37, rk), &dev)? * 0.004)?.to_dtype(dtype)?;
    // 关键：引擎的 ON 路径先做 `fp8_qpn8_repack`（标度变 F16 且形状变 (1,n)），
    // 而 `fp8_matmul` 的打包分支门控正是 `F16 && (1,n)`。传 F32 (n,1) 会让两路都走
    // fallback，测不到打包分支（§70/§71 的"一致"因此无效）。
    let (pw, ps) = attention_rs::fp8_linear::fp8_qpn8_repack(&rweight, &rscale)?;
    let rpacked = fp8_matmul(&rinput, &pw, &ps, None, &[1, rk], true)?;
    let rgold = fp8_matmul_fallback(&rinput, &rweight, &rscale, &[1, rk])?;
    for (name, tensor) in [("real.packed", &rpacked), ("real.gold", &rgold)] {
        record(&mut file, name, tensor)?;
    }
    println!("wrote {}", path.to_string_lossy());
    Ok(())
}
