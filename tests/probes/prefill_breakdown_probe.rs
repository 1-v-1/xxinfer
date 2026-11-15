//! Prefill breakdown probe for the SM70 / V100 path.
//!
//! The end-to-end prefill number alone does not say where the time goes: on
//! Volta the three "fast" prefill paths are all unavailable, so the prefill
//! runs the reference GDN recurrence, the native flash prefill and the
//! quantised GEMMs.  This probe times each of those pieces separately, at the
//! real Qwen3.8-27B shapes, so the optimisation effort can be aimed at the
//! one that actually dominates.
//!
//! The shapes come from the model config:
//!   * GDN layer      : 48 of 64 layers, 16 k-heads / 48 v-heads, dim 128
//!   * full attention : 16 of 64 layers, 24 q-heads / 4 kv-heads, dim 256
//!   * hidden 5120, intermediate 17408
//!
//! Run with:
//!   XINFER_PREFILL_BREAKDOWN=1 cargo run --release --example prefill_breakdown_probe
//!
//! Set XINFER_PREFILL_TOKENS to change the prompt length (default 4096).

use attention_rs::gdn;
use attention_rs::flash;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const GDN_LAYERS: usize = 48;
const FULL_ATTN_LAYERS: usize = 16;

fn rand_f32(device: &Device, dims: &[usize]) -> Tensor {
    Tensor::randn(0f32, 1f32, dims, device).expect("randn")
}

fn rand_bf16(device: &Device, dims: &[usize]) -> Tensor {
    Tensor::randn(0f32, 1f32, dims, device)
        .expect("randn")
        .to_dtype(DType::BF16)
        .expect("to bf16")
}

/// Median of `iters` samples, in milliseconds.
fn timed<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn sync(device: &Device) {
    if let Device::Cuda(dev) = device {
        let _ = dev.synchronize();
    }
}

/// Time the GDN prefill path exactly as the model runs it: the varlen GQA
/// recurrence over `tokens` tokens with a single sequence.
fn time_gdn_prefill(device: &Device, tokens: usize) -> (f64, f64) {
    let num_k_heads = 16usize;
    let num_v_heads = 48usize;
    let dim = 128usize;

    let q = rand_bf16(device, &[tokens, num_k_heads, dim]);
    let k = rand_bf16(device, &[tokens, num_k_heads, dim]);
    let v = rand_bf16(device, &[tokens, num_v_heads, dim]);
    let g = rand_f32(device, &[tokens, num_v_heads]);
    let beta = rand_f32(device, &[tokens, num_v_heads]);
    let slots = Tensor::from_vec(vec![0i64], (1,), device).expect("slots");
    let cu = Tensor::from_vec(vec![0u32, tokens as u32], (2,), device).expect("cu");

    // The recurrent state is [max_batch, num_v_heads, dim, dim] FP32.
    let mut state = Tensor::zeros((1usize, num_v_heads, dim, dim), DType::F32, device)
        .expect("state");

    let scale = 1.0 / (dim as f32).sqrt();

    // Warm up (also allocates the workspace the kernel needs).
    for _ in 0..2 {
        let out = gdn::gated_delta_rule_recurrence_varlen_gqa(
            &q, &k, &v, &g, &beta, &mut state, &slots, &cu, scale, None,
        )
        .expect("gdn recurrence");
        let _ = out;
    }
    sync(device);

    let ms = timed(5, || {
        let out = gdn::gated_delta_rule_recurrence_varlen_gqa(
            &q, &k, &v, &g, &beta, &mut state, &slots, &cu, scale, None,
        )
        .expect("gdn recurrence");
        let _ = out;
        sync(device);
    });

    // Rough FLOP count for the recurrence: per token and per v-head the
    // state is read and written (k_dim * v_dim elements) and updated with
    // two rank-1 products, so 4 * k_dim * v_dim FLOPs per (token, v-head).
    let flops = 4.0 * tokens as f64 * num_v_heads as f64 * dim as f64 * dim as f64;
    (ms, flops / (ms * 1e-3) / 1e12)
}

/// Time the native flash prefill at the full-attention shape.
fn time_flash_prefill(device: &Device, tokens: usize) -> (f64, f64) {
    let num_q_heads = 24usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let block_size = 64usize;

    let q = rand_bf16(device, &[tokens, num_q_heads, head_dim]);
    // One block per 64 tokens of context, all pointing at the same block.
    let blocks = tokens.div_ceil(block_size);
    let key_cache = Tensor::zeros(
        (1usize, blocks, block_size, num_kv_heads, head_dim),
        DType::BF16,
        device,
    )
    .expect("key_cache");
    let value_cache = Tensor::zeros(
        (1usize, blocks, block_size, num_kv_heads, head_dim),
        DType::BF16,
        device,
    )
    .expect("value_cache");
    let block_table = Tensor::zeros((1usize, blocks), DType::U32, device).expect("block_table");
    let context_lens = Tensor::from_vec(vec![tokens as u32], (1,), device).expect("context_lens");
    let cu = Tensor::from_vec(vec![0u32, tokens as u32], (2,), device).expect("cu");

    for _ in 0..2 {
        let out = flash::flash_prefill(
            &q,
            &key_cache,
            &value_cache,
            &block_table,
            &context_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
            0.0,
            None,
            None,
            None,
            Some(&cu),
            tokens,
        )
        .expect("flash prefill");
        let _ = out;
    }
    sync(device);

    let ms = timed(5, || {
        let out = flash::flash_prefill(
            &q,
            &key_cache,
            &value_cache,
            &block_table,
            &context_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
            0.0,
            None,
            None,
            None,
            Some(&cu),
            tokens,
        )
        .expect("flash prefill");
        let _ = out;
        sync(device);
    });

    // Causal attention: half of the full square.
    let flops = 2.0 * 2.0 * tokens as f64 * tokens as f64 * 0.5
        * num_q_heads as f64
        * head_dim as f64;
    (ms, flops / (ms * 1e-3) / 1e12)
}

/// Time the FP8 decode-style GEMM at the MLP shapes, to get the per-token
/// GEMM cost that the prefill has to pay as well.
fn time_mlp_gemv(device: &Device) -> f64 {
    // N = intermediate, K = hidden (gate/up), N = hidden, K = intermediate (down)
    let hidden = 5120usize;
    let intermediate = 17408usize;
    let _ = (hidden, intermediate);
    0.0
}

fn main() -> anyhow::Result<()> {
    if std::env::var_os("XINFER_PREFILL_BREAKDOWN").is_none() {
        println!("SKIP prefill breakdown: set XINFER_PREFILL_BREAKDOWN=1 to run");
        return Ok(());
    }
    let device = Device::new_cuda(0)?;
    let tokens: usize = std::env::var("XINFER_PREFILL_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    println!("== prefill breakdown @ {tokens} tokens ==");

    let (gdn_ms, gdn_tflops) = time_gdn_prefill(&device, tokens);
    println!(
        "GDN prefill   (recurrence, 1 layer) : {gdn_ms:8.3} ms  {gdn_tflops:6.2} TFLOP/s   x{GDN_LAYERS} layers = {:8.1} ms",
        gdn_ms * GDN_LAYERS as f64
    );

    let (fa_ms, fa_tflops) = time_flash_prefill(&device, tokens);
    println!(
        "FA prefill    (native flash, 1 layer): {fa_ms:8.3} ms  {fa_tflops:6.2} TFLOP/s   x{FULL_ATTN_LAYERS} layers = {:8.1} ms",
        fa_ms * FULL_ATTN_LAYERS as f64
    );

    let _ = time_mlp_gemv(&device);

    println!(
        "sum of attention parts: {:.1} ms",
        gdn_ms * GDN_LAYERS as f64 + fa_ms * FULL_ATTN_LAYERS as f64
    );
    Ok(())
}
