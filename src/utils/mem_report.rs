//! Reports how much device memory is in use.
//!
//! The numbers that matter when a model does not fit are the peak during
//! loading and the steady state afterwards.  Loading a quantized model
//! allocates the weights and, for the layers that have to be repacked, a
//! second copy of the same size, so the peak can be several times the
//! steady state.
//!
//! The free memory is read through the driver, which is what the allocator
//! sees.  The used memory is the difference against the total, so it also
//! counts the allocations the driver made outside of the pool.

use candle_core::cuda_backend::cudarc::driver::sys::lib;

/// Returns (used, total) device memory in bytes, or `None` when the
/// driver cannot be asked.
pub fn device_memory() -> Option<(usize, usize)> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        lib()
            .cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _)
            .result()
            .ok()?;
    }
    Some((total.saturating_sub(free), total))
}

/// Logs how much device memory is in use, together with a label that says
/// which stage of the run this is.
pub fn log_device_memory(label: &str) {
    #[cfg(feature = "cuda")]
    match device_memory() {
        Some((used, total)) => crate::log_info!(
            "Memory [{}]: {:.2} GB used of {:.2} GB ({:.1}%)",
            label,
            used as f64 / (1024.0 * 1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0 * 1024.0),
            used as f64 * 100.0 / total as f64,
        ),
        None => crate::log_info!("Memory [{}]: the driver did not report it", label),
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = label;
    }
}
