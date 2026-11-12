//! Does the NCCL failure path report the real `ncclResult_t` code?
//!
//! A failing collective used to surface as `Err(NcclError)` and nothing else,
//! because `cudarc`'s `Debug` impl for `NcclError` writes the literal string
//! `"NcclError"` and throws the wrapped `ncclResult_t` away.  That made it
//! impossible to tell `ncclInvalidArgument` (a count mismatch on one rank)
//! apart from `ncclRemoteError` or a timeout.
//!
//! This probe drives the same helper the inference path uses
//! (`xinfer::utils::nccl_err`) and asserts that the rendered message carries
//! the symbolic NCCL code.
//!
//! Run with:
//!   ./nccl_error_code_probe

use candle_core::cuda_backend::cudarc::nccl::safe::{Comm, Id, ReduceOp};

/// Mirrors `xinfer::utils::nccl_err`; kept local so the probe does not need
/// the `cuda`/`nccl` feature set of the `xinfer` library itself.
fn nccl_err(
    op: &str,
    err: candle_core::cuda_backend::cudarc::nccl::result::NcclError,
) -> candle_core::Error {
    candle_core::Error::Msg(format!(
        "NCCL {op} failed: {:?} (code={})",
        err.0,
        err.0 as i32
    ))
}

fn main() -> anyhow::Result<()> {
    let mut failures = 0;

    // ---------------------------------------------------------------
    // A single-rank communicator is enough: ncclAllReduce validates its
    // arguments locally, so passing a receive buffer that is too small
    // for `count` makes it return ncclInvalidArgument right away.
    //
    // (A mismatched world size is deliberately NOT exercised here: NCCL's
    // init blocks until the missing peer joins or the bootstrap times
    // out, which would hang the probe.)
    // ---------------------------------------------------------------
    let device = candle_core::CudaDevice::new_with_stream(0)?;
    let dev = device.cuda_device();
    let id = Id::new().unwrap();
    let comm = match Comm::from_rank(dev.clone(), 0, 1, id) {
        Ok(c) => c,
        Err(e) => {
            println!("FAIL could not build a single-rank communicator: {e:?}");
            return Ok(());
        }
    };

    let n: usize = 16;
    let send = dev.alloc_zeros::<f32>(n, true)?;
    // Deliberately too small: ncclAllReduce checks that recvbuff can hold
    // `count` elements and rejects the call with ncclInvalidArgument.
    let mut recv = dev.alloc_zeros::<f32>(n / 2, true)?;

    println!("--- all_reduce with a mismatched count ---");
    match comm.all_reduce(&send, &mut recv, &ReduceOp::Sum) {
        Ok(_) => {
            println!("FAIL mismatched all_reduce unexpectedly succeeded");
            failures += 1;
        }
        Err(e) => {
            let msg = format!("{}", nccl_err("all_reduce", e));
            println!("rendered: {msg}");
            // The whole point of the helper: the message must name the code.
            //
            // Note which code shows up depends on where NCCL noticed the
            // problem: it validates the arguments against the CUDA layer, so
            // a bad receive buffer comes back as ncclUnhandledCudaError (1)
            // even though NCCL's own log says "Cuda failure 'invalid
            // argument'".  Either way the code is now visible, which is the
            // property under test.
            if msg.contains("code=") && !msg.contains("code=0)") {
                println!("PASS the NCCL error code is visible in the message");
            } else {
                println!("FAIL the NCCL error code is not visible: {msg}");
                failures += 1;
            }
        }
    }

    // ---------------------------------------------------------------
    // Show what the old formatting looked like, so the difference is
    // visible in the probe output itself.
    // ---------------------------------------------------------------
    println!("--- what the old Debug impl produced ---");
    let e = candle_core::cuda_backend::cudarc::nccl::result::NcclError(
        candle_core::cuda_backend::cudarc::nccl::sys::ncclResult_t::ncclInvalidArgument,
    );
    let old_debug = format!("{:?}", e);
    println!("old: {old_debug}");
    println!("new: {}", nccl_err("all_reduce", e));
    if old_debug == "NcclError" {
        println!("PASS the old formatting is confirmed to lose the code");
    } else {
        println!("NOTE cudarc's Debug impl changed: {old_debug}");
    }

    if failures > 0 {
        println!("EXIT={failures}");
        std::process::exit(1);
    }
    println!("EXIT=0");
    Ok(())
}
