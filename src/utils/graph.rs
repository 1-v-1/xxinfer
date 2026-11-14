#[cfg(feature = "flashinfer")]
use super::FlashInferKvParams;
use crate::models::layers::linear::set_linear_is_prefill;
use attention_rs::InputMetadata;
use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::cuda_backend::cudarc::driver::sys::{
    lib, CUmemAllocationHandleType, CUmemAllocationType, CUmemLocation, CUmemLocationType,
    CUmemPool_attribute, CUmemPoolProps, CUmemoryPool, CUstreamCaptureMode, CUstreamCaptureStatus,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{DType, Device, Result, Tensor};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use tqdm::tqdm;

#[allow(dead_code)]
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: sys::CUstream,
}

impl CudaGraph {
    pub fn begin_capture(stream: sys::CUstream, mode: sys::CUstreamCaptureMode) -> Result<()> {
        unsafe {
            lib()
                .cuStreamBeginCapture_v2(stream, mode)
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("begin_capture failed: {e:?}")))
        }
    }

    pub fn end_capture(stream: sys::CUstream, flags: u64) -> Result<CudaGraph> {
        let mut graph = MaybeUninit::uninit();
        let cu_graph = unsafe {
            lib()
                .cuStreamEndCapture(stream, graph.as_mut_ptr())
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuStreamEndCapture failed: {e:?}"))
                })?;
            graph.assume_init()
        };

        // `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` is required here.
        //
        // The graph contains memory allocation nodes: the model allocates its
        // intermediate tensors while the capture is running, and every one of
        // those allocations becomes a node of the graph.  A graph with memory
        // nodes has to be instantiated with this flag -- on CUDA 12.4 the
        // driver rejects every launch after the first one with
        // `CUDA_ERROR_INVALID_VALUE` otherwise, because the memory the
        // allocation nodes hand out is only valid for the launch it was
        // created for.  The flag makes the driver release the allocations at
        // the end of a launch and hand out fresh ones for the next launch;
        // the driver then rewrites the pointers inside the kernel parameters,
        // so the captured kernels keep working.
        //
        // This is only sound because the allocations come from the driver's
        // own stream-ordered allocator (no custom memory pool is installed
        // during the capture, see `set_capture_mem_pool`), which is what lets
        // the driver take ownership of them in the first place.
        let mut graph_exec = MaybeUninit::uninit();
        let mut params = sys::CUDA_GRAPH_INSTANTIATE_PARAMS {
            flags: flags
                | sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                    as u64,
            hUploadStream: stream,
            ..Default::default()
        };
        let cu_graph_exec = unsafe {
            lib()
                .cuGraphInstantiateWithParams(graph_exec.as_mut_ptr(), cu_graph, &mut params)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuGraphInstantiateWithParams failed: {e:?}"))
                })?;
            graph_exec.assume_init()
        };
        Ok(CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream,
        })
    }

    /// Dumps the node structure of the captured graph.
    ///
    /// Only used by the `XINFER_DIAG_GRAPH` diagnostic: when `cuGraphLaunch`
    /// rejects a graph with `CUDA_ERROR_INVALID_VALUE` the reason is almost
    /// always a node that the driver no longer accepts, and the node types
    /// make that visible.
    pub fn dump_nodes(&self) {
        if std::env::var("XINFER_DIAG_GRAPH").is_err() {
            return;
        }
        unsafe {
            let mut num_nodes: usize = 0;
            if lib()
                .cuGraphGetNodes(self.cu_graph, std::ptr::null_mut(), &mut num_nodes)
                .result()
                .is_err()
            {
                eprintln!("[diag][graph] cuGraphGetNodes(count) failed");
                return;
            }
            let mut nodes: Vec<sys::CUgraphNode> = vec![std::mem::zeroed(); num_nodes];
            if lib()
                .cuGraphGetNodes(self.cu_graph, nodes.as_mut_ptr(), &mut num_nodes)
                .result()
                .is_err()
            {
                eprintln!("[diag][graph] cuGraphGetNodes failed");
                return;
            }
            let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
            for node in nodes.iter().take(num_nodes) {
                let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
                if lib().cuGraphNodeGetType(*node, &mut kind).result().is_err() {
                    *counts.entry("<unknown>".to_string()).or_insert(0) += 1;
                    continue;
                }
                let name = format!("{kind:?}");
                *counts.entry(name).or_insert(0) += 1;
            }
            eprintln!("[diag][graph] captured graph has {num_nodes} nodes: {counts:?}");
        }
    }

    pub fn capture_status(stream: sys::CUstream) -> Result<sys::CUstreamCaptureStatus> {
        let mut status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        unsafe {
            lib()
                .cuStreamIsCapturing(stream, &mut status)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuGraphInstantiateWithFlags failed: {e:?}"))
                })?;
        }
        Ok(status)
    }

    pub fn launch(&self) -> Result<()> {
        unsafe {
            lib()
                .cuGraphLaunch(self.cu_graph_exec, self.stream)
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("cuGraphLaunch failed: {e:?}")))
        }
    }
}

pub trait CudaGraphModule {
    fn start_capture(&mut self, bs: usize) -> Result<()>;
    fn end_capture(&mut self, save: bool) -> Result<()>;
    fn replay(&self, bs: usize) -> Result<()>;
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor>;
    fn report_graph_pool_usage(&self) -> Result<()>;

    /// Detach the capture pool from the device once the capture is over.
    fn restore_default_mem_pool(&self) -> Result<()>;
}

pub struct CudaGraphHandle {
    graph: Arc<CudaGraph>,
}

impl CudaGraphHandle {
    pub fn new(graph: Arc<CudaGraph>) -> Self {
        Self { graph }
    }

    pub fn replay(&self) -> Result<()> {
        // The graph's input buffers are refreshed with `copy_` calls that run
        // on candle's device stream.  The graph itself is launched on the
        // stream that was current during the capture.  If the two differ,
        // the launch can race ahead of the copies and the graph reads stale
        // (or zeroed) input.  Synchronizing the capture stream first makes
        // the dependency explicit.
        unsafe {
            lib()
                .cuStreamSynchronize(self.graph.stream)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuStreamSynchronize before launch failed: {e:?}"))
                })?;
        }
        self.graph
            .launch()
            .map_err(|e| candle_core::Error::Msg(format!("CUDA Graph launch failed: {:?}", e)))?;
        Ok(())
    }
}

pub struct CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
        bool,
    ) -> Result<Tensor>,
{
    module: M,
    captured_graphs: BTreeMap<usize, CudaGraphHandle>,
    capturing: bool,
    current_bs: Option<usize>,
    device: Arc<CudaDevice>,
    pub pool_handle: RwLock<Option<i64>>,
    captured_bs: Vec<usize>,
    /// Number of tensor-parallel ranks; only a single rank needs no NCCL
    /// group around the capture.
    world_size: usize,
}

impl<M> CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
        bool,
    ) -> Result<Tensor>,
{
    pub fn new(module: M, device: Arc<CudaDevice>) -> Self {
        Self::with_world_size(module, device, 1)
    }

    pub fn with_world_size(module: M, device: Arc<CudaDevice>, world_size: usize) -> Self {
        Self {
            module,
            captured_graphs: BTreeMap::new(),
            capturing: false,
            current_bs: None,
            device,
            pool_handle: RwLock::new(None),
            captured_bs: Vec::new(),
            world_size,
        }
    }

    fn sync_stream(&self) -> Result<()> {
        unsafe {
            lib()
                .cuStreamSynchronize(self.device.cu_stream().clone())
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("cuStreamSynchronize failed: {e:?}")))
        }
    }

    /// Create a *private* memory pool for graph capture.
    ///
    /// The previous implementation took the device's default pool
    /// (`cuDeviceGetDefaultMemPool`), pinned its release threshold to
    /// `u64::MAX`, and then set it as the device's default pool again.  The
    /// last step was a no-op: `cuDeviceGetDefaultMemPool` returns the pool
    /// that is already the default, so the "capture pool" and the pool every
    /// other allocation in the process goes through were the same pool.
    ///
    /// The consequence was that the reported "graph pool usage" was really
    /// the whole card: the model weights, the KV cache and the mamba states
    /// all landed in it.  That number was then mistaken for the cost of the
    /// CUDA graphs, which made the graphs look more expensive than the model.
    ///
    /// A private pool fixes the accounting: only what the capture allocates
    /// lands in it, and it is detached from the device again once the
    /// capture is over.
    fn create_capture_pool(&self) -> Result<CUmemoryPool> {
        let mut pool: CUmemoryPool = ptr::null_mut();
        unsafe {
            // `CUmemPoolProps` grew fields across CUDA releases, so start from
            // the default and only set the ones that matter.  The release
            // threshold is applied separately with `cuMemPoolSetAttribute`.
            let props = CUmemPoolProps {
                allocType: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
                handleTypes: CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
                location: CUmemLocation {
                    type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                    id: *self.device.cu_device() as i32,
                },
                ..Default::default()
            };
            let diag = std::env::var("XINFER_DIAG_GRAPH").is_ok();
            if diag { eprintln!("[diag][graph] create_capture_pool: cuMemPoolCreate"); }
            lib()
                .cuMemPoolCreate(&mut pool, &props)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuMemPoolCreate failed: {e:?}"))
                })?;
            if diag { eprintln!("[diag][graph] create_capture_pool: created"); }

            let handle = pool as *mut std::ffi::c_void as usize as i64;
            *self.pool_handle.write() = Some(handle);

            // The release threshold is the amount of *unused* memory the pool
            // keeps before it hands it back to the driver.
            //
            // The threshold is pinned to the maximum on purpose.  This pool
            // holds the allocations of the captured graphs, and those stay
            // live for as long as the graphs exist, so there is nothing to
            // release anyway.  More importantly, the pool must not hand
            // memory back between captures: the graphs refer to the
            // addresses they were captured with, and a released address that
            // gets reused by an unrelated allocation would corrupt a replay.
            //
            // Because the pool is private to the capture, a large threshold
            // only affects the capture itself -- unlike the previous
            // implementation, which pinned the *device default* pool and
            // therefore changed the behaviour of every allocation in the
            // process.
            let threshold: u64 = u64::MAX;
            if diag { eprintln!("[diag][graph] create_capture_pool: set_attribute begin"); }
            lib()
                .cuMemPoolSetAttribute(
                    pool,
                    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const _ as _,
                )
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuMemPoolSetAttribute failed: {e:?}"))
                })?;
            if diag { eprintln!("[diag][graph] create_capture_pool: set_attribute done"); }
        }
        Ok(pool)
    }

    fn set_capture_mem_pool(&self) -> Result<()> {
        // A custom memory pool must NOT be installed as the device default
        // during the capture.
        //
        // Allocations that a CUDA graph captures are turned into *memory
        // nodes* of the graph.  The driver can only take ownership of such an
        // allocation -- and give the graph a private, replayable address for
        // it -- when the allocation came from the driver's own stream-ordered
        // allocator.  With `cuDeviceSetMemPool` the allocation lands in our
        // pool instead, the graph cannot adopt it, and the captured graph
        // keeps a dependency on a pool that no longer matches: every
        // `cuGraphLaunch` is then rejected with `CUDA_ERROR_INVALID_VALUE`.
        //
        // PyTorch's `torch.cuda.graph()` lets the driver own the graph's
        // memory, and so does vLLM; do the same here.
        if std::env::var("XINFER_CAPTURE_MEM_POOL").is_err() {
            return Ok(());
        }
        unsafe {
            let status = CudaGraph::capture_status(self.device.cu_stream().clone())?;
            if status == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
                // A capture is already in progress; the pool it uses is
                // already the device's default pool.
                return Ok(());
            }

            // The pool is created once and then reused for every capture.
            // It has to be re-installed as the device's default pool on each
            // capture, because `restore_default_mem_pool` detaches it when
            // the previous capture finished.  Without this, every capture
            // after the first one would allocate outside the pool, and the
            // pool usage would no longer describe what the graphs hold.
            let diag = std::env::var("XINFER_DIAG_GRAPH").is_ok();
            if diag { eprintln!("[diag][graph] set_capture_mem_pool: capture_status checked"); }
            // The read guard has to be dropped before the pool is created:
            // `create_capture_pool` takes the write lock, and the guard of
            // `self.pool_handle.read()` would still be alive for the whole
            // match expression, so the write would wait for a reader that
            // only this thread can release.  It deadlocks.
            let existing = *self.pool_handle.read();
            let pool = match existing {
                Some(handle) => handle as usize as *mut sys::CUmemPoolHandle_st,
                None => self.create_capture_pool()?,
            };
            if diag { eprintln!("[diag][graph] set_capture_mem_pool: pool={pool:?}"); }

            lib()
                .cuDeviceSetMemPool(*self.device.cu_device(), pool)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuDeviceSetMemPool failed: {e:?}"))
                })?;
            if diag { eprintln!("[diag][graph] set_capture_mem_pool: set_mem_pool done"); }
        }

        Ok(())
    }

    /// Detach the capture pool from the device once the capture is over.
    ///
    /// The graphs keep referring to the pool they were captured in, so the
    /// pool has to stay alive -- but it must not remain the device's *default*
    /// pool, or every later allocation in the process would be served from it
    /// and would keep the pool from ever shrinking.
    fn restore_default_mem_pool(&self) -> Result<()> {
        if std::env::var("XINFER_CAPTURE_MEM_POOL").is_err() {
            // No custom pool was installed, so there is nothing to detach.
            return Ok(());
        }
        let handle = match *self.pool_handle.read() {
            Some(h) => h,
            None => return Ok(()),
        };
        // The read guard is dropped at the end of the statement above.

        unsafe {
            let mut default_pool: CUmemoryPool = ptr::null_mut();
            lib()
                .cuDeviceGetDefaultMemPool(&mut default_pool, *self.device.cu_device())
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuDeviceGetDefaultMemPool failed: {e:?}"))
                })?;

            // The device default pool is never the private capture pool, but
            // guard against restoring the pool we are trying to detach.
            if default_pool as *mut std::ffi::c_void as usize as i64 != handle {
                lib()
                    .cuDeviceSetMemPool(*self.device.cu_device(), default_pool)
                    .result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("cuDeviceSetMemPool failed: {e:?}"))
                    })?;
            }
        }

        Ok(())
    }

    /// Reads a usize attribute from the given CUDA memory pool.
    fn get_mem_pool_attribute(pool: CUmemoryPool, attr: CUmemPool_attribute) -> Result<usize> {
        let mut value: usize = 0;
        unsafe {
            sys::lib()
                .cuMemPoolGetAttribute(pool, attr, &mut value as *mut _ as *mut std::ffi::c_void)
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuMemPoolGetAttribute failed: {e:?}"))
                })?;
        }
        Ok(value)
    }

    /// Returns peak memory used (in bytes) from a given CUDA memory pool.
    pub fn get_peak_memory_usage(pool: CUmemoryPool) -> Result<usize> {
        Self::get_mem_pool_attribute(pool, CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)
    }

    /// Returns current memory usage (in bytes) from a given CUDA memory pool.
    pub fn get_current_memory_usage(pool: CUmemoryPool) -> Result<usize> {
        Self::get_mem_pool_attribute(pool, CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
    }

    /// Retrieves the default CUDA memory pool for a device.
    pub fn get_current_mem_pool(&self) -> Result<CUmemoryPool> {
        // One read, one guard: the first `read()` in the condition would be a
        // separate temporary.
        let handle = *self.pool_handle.read();
        match handle {
            Some(pool_handle) => {
                let pool: CUmemoryPool = pool_handle as usize as *mut sys::CUmemPoolHandle_st;
                Ok(pool)
            }
            None => candle_core::bail!("Memory pool for graph is not init!"),
        }
    }
}

impl<M> CudaGraphModule for CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
        bool,
    ) -> Result<Tensor>,
{
    fn start_capture(&mut self, bs: usize) -> Result<()> {
        let diag = std::env::var("XINFER_DIAG_GRAPH").is_ok();
        self.capturing = true;
        self.current_bs = Some(bs);
        if diag { eprintln!("[diag][graph] start_capture: sync_stream"); }
        self.sync_stream()?;
        if diag { eprintln!("[diag][graph] start_capture: set_capture_mem_pool"); }
        self.set_capture_mem_pool()?;
        if diag { eprintln!("[diag][graph] start_capture: begin_capture"); }
        // Open an NCCL group around the whole capture.
        //
        // Every tensor-parallel layer ends in an all-reduce.  NCCL captures
        // a collective into a CUDA graph by putting it on an internal stream
        // and connecting it to the capturing stream with events, and the
        // event dependencies only come out right when the collectives of one
        // capture belong to a single `ncclGroupStart`/`ncclGroupEnd` region.
        // With one collective per group -- which is what an ungrouped
        // all-reduce gives -- the captured graph ends up with a single wait
        // event instead of one per collective, the all-reduce is left
        // incomplete, and the two ranks end up with different logits.
        #[cfg(feature = "nccl")]
        {
            use candle_core::cuda_backend::cudarc::nccl::safe::{group_end, group_start};
            if self.world_size > 1 {
                group_start().map_err(|e| crate::utils::nccl_err("ncclGroupStart", e))?;
            }
        }
        CudaGraph::begin_capture(
            self.device.cu_stream().clone(),
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )?;
        if diag { eprintln!("[diag][graph] start_capture: done"); }
        Ok(())
    }

    fn end_capture(&mut self, save: bool) -> Result<()> {
        self.capturing = false;
        let bs = self.current_bs.take().unwrap();

        // Close the NCCL group that `start_capture` opened, so that all the
        // all-reduces of this capture become a single collective group.
        #[cfg(feature = "nccl")]
        {
            use candle_core::cuda_backend::cudarc::nccl::safe::group_end;
            if self.world_size > 1 {
                group_end().map_err(|e| crate::utils::nccl_err("ncclGroupEnd", e))?;
            }
        }

        let graph = CudaGraph::end_capture(self.device.cu_stream().clone(), 0)?;
        graph.dump_nodes();
        self.captured_graphs
            .insert(bs, CudaGraphHandle::new(Arc::new(graph)));

        if save {
            self.captured_bs.push(bs);
            self.captured_bs.sort_unstable(); // keep it sorted for binary search
        }
        self.sync_stream()?;
        Ok(())
    }

    fn replay(&self, bs: usize) -> Result<()> {
        if let Some(&next_bs) = self.captured_bs.iter().find(|&&x| x >= bs) {
            if let Some(graph) = self.captured_graphs.get(&next_bs) {
                graph.replay()?;
                self.sync_stream()
            } else {
                candle_core::bail!("No suitable graph is found for batch size {}!", next_bs)
            }
        } else {
            candle_core::bail!("Batch size {} is not captured in graph!", bs)
        }
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        (self.module)(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
        )
    }

    /// Report how much memory the CUDA graphs actually hold.
    ///
    /// The pool is private to the capture, so this is the real cost of the
    /// graphs and nothing else.  The device's total and free memory are
    /// printed next to it, because the previous implementation reported the
    /// *device default* pool, which is every allocation in the process --
    /// that made the graphs look like they held the whole card.
    fn report_graph_pool_usage(&self) -> Result<()> {
        let pool = self.get_current_mem_pool()?;
        let peak = Self::get_peak_memory_usage(pool)?;
        let current = Self::get_current_memory_usage(pool)?;

        let (total, free) = {
            let mut free_bytes: usize = 0;
            let mut total_bytes: usize = 0;
            unsafe {
                let result = lib()
                    .cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes)
                    .result();
                if result.is_err() {
                    (0u64, 0u64)
                } else {
                    (total_bytes as u64, free_bytes as u64)
                }
            }
        };

        println!(
            "CUDA graph capture pool usage: {:.2} MB (current), {:.2} MB (peak)",
            current as f64 / 1e6,
            peak as f64 / 1e6
        );
        if total > 0 {
            println!(
                "  device memory: {:.2} GB total, {:.2} GB free => graphs hold {:.2}% of the card",
                total as f64 / 1e9,
                free as f64 / 1e9,
                current as f64 / total as f64 * 100.0
            );
        }
        Ok(())
    }

    fn restore_default_mem_pool(&self) -> Result<()> {
        CudaGraphWrapper::restore_default_mem_pool(self)
    }
}

#[derive(Clone, Copy)]
#[derive(Debug)]
enum CapturePhase {
    CachePrewarm,
    Warmup,
    Capture,
}

impl CapturePhase {
    const ALL: [Self; 3] = [Self::CachePrewarm, Self::Warmup, Self::Capture];

    fn is_cache_prewarm(self) -> bool {
        matches!(self, Self::CachePrewarm)
    }

    fn is_warmup(self) -> bool {
        !matches!(self, Self::Capture)
    }
}

pub struct GraphCaptureVars {
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub mamba_slot_mapping: Tensor,
    pub slot_mapping: Tensor,
    pub context_lens: Tensor,
    pub block_tables: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indptr: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indices: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_last_len: Tensor,
    pub outputs: BTreeMap<usize, Tensor>,
    /// Logits copied out of the capture memory pool.
    ///
    /// The tensor that the model returns during capture lives in the CUDA
    /// graph memory pool, which the driver may hand back after every launch
    /// (`CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH`).  Reading that
    /// memory after a replay is therefore not reliable, so the logits are
    /// copied into a buffer that is allocated outside the pool while the
    /// capture is still running.  The copy is a device-to-device memcpy that
    /// the capture records, so every replay refreshes the buffer.
    pub stable_outputs: BTreeMap<usize, Tensor>,
}

pub struct MtpGraphCaptureVars {
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub mamba_slot_mapping: Tensor,
    pub slot_mapping: Tensor,
    pub context_lens: Tensor,
    pub block_tables: Tensor,
    pub cu_seqlens_q: Tensor,
    pub cu_seqlens_k: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indptr: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_indices: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_last_len: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_batch_indices: Tensor,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_positions: Tensor,
    pub outputs: BTreeMap<usize, Tensor>,
}

pub struct GraphCapturer<M: CudaGraphModule> {
    pub model: M,
    pub graph_bs: Vec<usize>,
    pub graph_vars: Option<GraphCaptureVars>,
    pub max_num_seqs: usize,
    pub max_model_len: usize,
    pub block_size: usize,
    pub hidden_size: usize,
    pub device: Option<Device>,
    #[cfg(feature = "flashinfer")]
    pub flashinfer_kv_params: Option<FlashInferKvParams>,
    pub is_mla: bool,
    pub mtp_graph_vars: Option<MtpGraphCaptureVars>,
}

pub fn planned_graph_capture_batches(max_num_seqs: usize) -> Vec<usize> {
    // Capture every exact batch up to 32. This is especially important for
    // hybrid GDN/Mamba models, whose graph replay requires an exact batch size
    // because the state-slot mapping cannot be padded safely.
    (1..=max_num_seqs.clamp(1, 32)).collect()
}

/// Batch sizes to capture when the device cannot afford one graph per
/// integer batch size.
///
/// Each captured graph holds its own set of intermediates in the capture
/// memory pool, and none of them are released, so the pool grows with the
/// number of captured sizes.  On a 16 GB card that is the difference between
/// a few hundred megabytes and several gigabytes.
///
/// Replay picks the smallest captured size that is at least the live batch
/// size (`captured_bs.iter().find(|&&x| x >= bs)`), so a geometric ladder
/// covers every batch size with a handful of graphs.  The only cost is that
/// a batch of 3 replays the graph captured for 4, which wastes a little
/// compute.
pub fn sparse_graph_capture_batches(max_num_seqs: usize) -> Vec<usize> {
    let cap = max_num_seqs.clamp(1, 32);
    let mut sizes = Vec::new();
    let mut s = 1usize;
    while s <= cap {
        sizes.push(s);
        if s == cap {
            break;
        }
        s = s.saturating_mul(2);
    }
    // Always include the exact maximum: replay needs a size >= the live
    // batch, and the largest live batch is the maximum.
    if *sizes.last().unwrap_or(&0) != cap {
        sizes.push(cap);
    }
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// Pick the capture batch sizes for a device.
///
/// Volta (SM70) and Turing (SM75) have 16 GB or less of memory per card and
/// no FP4 tensor cores, so the decode path is already close to the memory
/// limit.  Capturing one graph per integer batch size there costs gigabytes
/// of capture-pool memory that the KV cache needs far more.
///
/// Turing is excluded on purpose: its FP8 tensor cores change the shape of
/// the decode path enough that the Volta measurements do not carry over.
#[cfg(feature = "cuda")]
pub fn graph_capture_batches_for_device(max_num_seqs: usize, device: &Device) -> Vec<usize> {
    let sm = match device {
        Device::Cuda(cuda_device) => {
            let inner = cuda_device.cuda_device();
            attention_rs::cuda_utils::sm_version(&inner)
        }
        _ => None,
    };

    if sm == Some(70) {
        let sizes = sparse_graph_capture_batches(max_num_seqs);
        crate::log_warn!(
            "SM70 detected: capturing {} batch sizes {:?} instead of every size up to {} \
             to keep the CUDA graph memory pool small",
            sizes.len(),
            sizes,
            max_num_seqs
        );
        sizes
    } else {
        planned_graph_capture_batches(max_num_seqs)
    }
}

#[cfg(feature = "flashinfer")]
fn graph_decode_plan(
    device: &Device,
    params: &FlashInferKvParams,
    indptr_host: &[u32],
    last_len_host: &[u32],
    kv_len_arr_host: &[u32],
    batch_size: usize,
    is_mla: bool,
    enable_cuda_graph: bool,
) -> Result<(Option<Vec<i64>>, Option<Vec<i64>>)> {
    if is_mla {
        let plan = attention_rs::mla::mla_decode_plan(
            device,
            params.kv_dtype,
            indptr_host,
            batch_size,
            params.num_qo_heads,
            params.page_size,
            enable_cuda_graph,
        )?;
        Ok((None, Some(plan)))
    } else {
        let plan = attention_rs::flashinfer::decode_plan(
            device,
            params.kv_dtype,
            params.out_dtype,
            indptr_host,
            Some(last_len_host),
            Some(kv_len_arr_host),
            batch_size,
            params.num_qo_heads,
            params.num_kv_heads,
            params.head_dim,
            params.page_size,
            enable_cuda_graph,
        )?;
        Ok((Some(plan), None))
    }
}

impl<M: CudaGraphModule> GraphCapturer<M> {
    pub fn new(
        model: M,
        max_num_seqs: usize,
        max_model_len: usize,
        block_size: usize,
        hidden_size: usize,
        #[cfg(feature = "flashinfer")] flashinfer_kv_params: &Option<FlashInferKvParams>,
        is_mla: bool,
    ) -> Self {
        let graph_bs = planned_graph_capture_batches(max_num_seqs);
        println!("The following batches for capture: {:?}", graph_bs);

        Self {
            model,
            graph_bs,
            graph_vars: None,
            max_num_seqs,
            max_model_len,
            block_size,
            hidden_size,
            device: None,
            #[cfg(feature = "flashinfer")]
            flashinfer_kv_params: flashinfer_kv_params.clone(),
            is_mla,
            mtp_graph_vars: None,
        }
    }

    pub fn capture(
        &mut self,
        device: &Device,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
    ) -> Result<()> {
        let _graph_workspace_guard = attention_rs::set_graph_workspace_domain(
            attention_rs::GraphWorkspaceDomain::DecodeGraph,
        );
        let _prefill_guard = set_linear_is_prefill(false);
        self.device = Some(device.clone());

        // The device is only known here, so this is where the capture sizes
        // are chosen: on SM70 a geometric ladder instead of every integer
        // batch size, to keep the capture memory pool small.
        #[cfg(feature = "cuda")]
        {
            let for_device = graph_capture_batches_for_device(self.max_num_seqs, device);
            if for_device != self.graph_bs {
                self.graph_bs = for_device;
            }
        }

        let max_bs = self.graph_bs[self.graph_bs.len() - 1];
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;

        // The decode graph needs its own plan workspace, because the
        // addresses it hands out have to stay stable across replays.  If it
        // were created on first use it would be created *inside* the capture
        // and would come out of the capture's memory pool, where it could
        // never be released again.  Create it here, before the capture
        // starts, so it comes from the device's default pool.
        #[cfg(feature = "flashinfer")]
        if let Device::Cuda(cuda_device) = device {
            attention_rs::workspace::preallocate_graph_workspace(cuda_device)?;
        }

        let input_ids = Tensor::zeros((max_bs,), DType::U32, device)?;
        let positions = Tensor::zeros((max_bs,), DType::I64, device)?;
        let mamba_slot_mapping = Tensor::from_vec(
            (0..max_bs).map(|i| i as i64).collect::<Vec<_>>(),
            (max_bs,),
            device,
        )?;
        let slot_mapping = Tensor::zeros((max_bs,), DType::I64, device)?;
        let context_lens = Tensor::zeros((max_bs,), DType::U32, device)?;
        let block_tables = Tensor::zeros((max_bs, max_num_blocks), DType::U32, device)?;
        #[cfg(feature = "flashinfer")]
        let (flashinfer_indptr, flashinfer_indices, flashinfer_last_len, last_len_host) = {
            let mut indptr = Vec::with_capacity(max_bs + 1);
            indptr.push(0u32);
            let mut indices = Vec::with_capacity(max_bs * max_num_blocks);
            for _ in 0..max_bs {
                for i in 0..max_num_blocks {
                    indices.push(i as u32);
                }
                indptr.push(indices.len() as u32);
            }
            let last = if self.max_model_len == 0 {
                0u32
            } else {
                ((self.max_model_len - 1) % self.block_size + 1) as u32
            };
            let last_len = vec![last; max_bs];

            (
                Tensor::from_vec(indptr, (max_bs + 1,), device)?,
                Tensor::from_vec(indices, (max_bs * max_num_blocks,), device)?,
                Tensor::from_vec(last_len.clone(), (max_bs,), device)?,
                last_len,
            )
        };
        // Capturing a second graph during the warm-up phase is only needed
        // when the warm-up phase itself has to be replayed.  Every capture
        // costs a full set of memory-allocation nodes, and with
        // AUTO_FREE_ON_LAUNCH every launch of that extra graph frees and
        // re-allocates its buffers, which can hand out the very addresses
        // the real capture is using.
        #[cfg(feature = "flashinfer")]
        let capture_in_warmup = self.flashinfer_kv_params.is_some()
            && std::env::var("XINFER_CAPTURE_IN_WARMUP").is_ok();
        #[cfg(not(feature = "flashinfer"))]
        let capture_in_warmup = false;

        let mut outputs = BTreeMap::<usize, Tensor>::new();
        let mut stable_outputs = BTreeMap::<usize, Tensor>::new();
        // Stable output buffers, allocated during the cache-prewarm phase.
        // At that point the capture memory pool is not installed yet, so the
        // allocation comes from the regular heap and survives every launch.
        let mut stable_buffers: BTreeMap<usize, Tensor> = BTreeMap::new();
        let _guard = candle_core::cuda_backend::cuda_param_cache_scope(true);
        for phase in CapturePhase::ALL {
            let iter: Box<dyn Iterator<Item = usize>> = if phase.is_warmup() {
                Box::new(0..self.graph_bs.len())
            } else {
                Box::new(tqdm(0..self.graph_bs.len()).desc(Some("Graph capturing")))
            };
            for i in iter {
                let bs = self.graph_bs[self.graph_bs.len() - i - 1];
                let input_ids_bs = input_ids.narrow(0, 0, bs)?;
                let positions_bs = positions.narrow(0, 0, bs)?;
                #[cfg(feature = "flashinfer")]
                let flashinfer_metadata = if self.flashinfer_kv_params.is_none() {
                    None
                } else {
                    let mut indptr_host = Vec::with_capacity(bs + 1);
                    indptr_host.push(0u32);
                    for i in 0..bs {
                        indptr_host.push(((i + 1) * max_num_blocks) as u32);
                    }
                    if std::env::var("XINFER_DIAG_GRAPH").is_ok() {
                        eprintln!(
                            "[diag][graph] capture bs={bs} max_num_blocks={max_num_blocks} \
                             max_model_len={} block_size={} indptr_host={:?}",
                            self.max_model_len, self.block_size, indptr_host
                        );
                    }

                    let (decode_plan_info, mla_decode_plan_info, kv_len_arr_host) =
                        if let Some(params) = self.flashinfer_kv_params {
                            let mut kv_len_arr_host_bs = Vec::with_capacity(bs);
                            for i in 0..bs {
                                let num_pages = indptr_host[i + 1] - indptr_host[i];
                                if num_pages == 0 {
                                    kv_len_arr_host_bs.push(0);
                                } else {
                                    let full = (num_pages - 1) * params.page_size as u32;
                                    kv_len_arr_host_bs.push(full + last_len_host[i]);
                                }
                            }
                            let (dp, mdp) = graph_decode_plan(
                                device,
                                &params,
                                &indptr_host,
                                &last_len_host[..bs],
                                &kv_len_arr_host_bs,
                                bs,
                                self.is_mla,
                                true, //must be true for graph capture
                            )?;
                            (dp, mdp, Some(kv_len_arr_host_bs))
                        } else {
                            (None, None, None)
                        };

                    Some(attention_rs::FlashInferMetadata {
                        indptr: flashinfer_indptr.narrow(0, 0, bs + 1)?,
                        indptr_host,
                        indices: flashinfer_indices.narrow(0, 0, bs * max_num_blocks)?,
                        last_len: flashinfer_last_len.narrow(0, 0, bs)?,
                        last_len_host: Some(last_len_host[..bs].to_vec()),
                        kv_len_arr_host,
                        total_num_rows: None,
                        batch_indices: None,
                        positions: None,
                        use_cuda_graph: true,
                        decode_plan_info,
                        prefill_plan_info: None,
                        mla_decode_plan_info,
                        mla_prefill_plan_info: None,
                    })
                };
                #[cfg(not(feature = "flashinfer"))]
                let flashinfer_metadata = None;

                let input_metadata = InputMetadata {
                    is_prefill: false,
                    is_mla: self.is_mla,
                    sequence_ids: None,
                    mamba_slot_mapping: Some(mamba_slot_mapping.narrow(0, 0, bs)?),
                    slot_mapping: slot_mapping.narrow(0, 0, bs)?,
                    // TEST: an out-of-range slot makes the KV write kernel
                    // fault, which proves whether the captured kernel runs
                    // during a replay.
                    block_tables: Some(block_tables.narrow(0, 0, bs)?),
                    block_tables_host: None,
                    context_lens_host: None,
                    context_lens: Some(context_lens.narrow(0, 0, bs)?),
                    cu_seqlens_q: None,
                    cu_seqlens_k: None,
                    max_seqlen_q: 0,
                    max_seqlen_k: 0,
                    max_context_len: self.max_model_len,
                    seqlens: None,
                    flashinfer_metadata,
                    is_mtp_verify: false,
                };

                let should_capture =
                    !phase.is_cache_prewarm() && (!phase.is_warmup() || capture_in_warmup);
                if std::env::var("XINFER_DIAG_GRAPH").is_ok() {
                    eprintln!(
                        "[diag][graph] phase={:?} bs={} should_capture={}",
                        phase, bs, should_capture
                    );
                }
                if should_capture {
                    if let Ok(v) = std::env::var("XINFER_DIAG_CAPSLO") {
                        // Write to a distinctive slot at capture time, so that a
                        // replay can be told apart from the prefill: if the
                        // kernel ends up writing to this slot during the
                        // replay, the slot mapping baked into the graph is
                        // stale.
                        let slot: i64 = v.parse().unwrap_or(777);
                        let one = Tensor::from_vec(vec![slot; bs], (bs,), slot_mapping.device())?;
                        slot_mapping.copy_(&one, 0)?;
                    }
                    if std::env::var("XINFER_DIAG_BADSLOT").is_ok() {
                        // Poison the slot mapping right before the capture so
                        // that the captured KV write kernel writes out of
                        // bounds.  A CUDA error during the replay then proves
                        // that the kernel really runs.
                        let poison = Tensor::from_vec(
                            vec![999_999_999i64; bs],
                            (bs,),
                            slot_mapping.device(),
                        )?;
                        slot_mapping.copy_(&poison, 0)?;
                    }
                    self.model.start_capture(bs)?;
                }
                if phase.is_warmup() {
                    let out = self.model.forward(
                        &input_ids_bs,
                        &positions_bs,
                        kv_caches,
                        &input_metadata,
                        false,
                    )?;
                    // The cache-prewarm phase runs before the capture memory
                    // pool is installed, so this allocation is a normal one
                    // that outlives the capture.  Its shape is the logits
                    // shape, which is what the stable buffer needs.
                    if phase.is_cache_prewarm() && !stable_buffers.contains_key(&bs) {
                        stable_buffers.insert(
                            bs,
                            Tensor::zeros(out.shape(), out.dtype(), out.device())?,
                        );
                    }
                    #[cfg(feature = "cuda")]
                    if !should_capture {
                        device.synchronize()?;
                    }
                } else {
                    let out = self.model.forward(
                        &input_ids_bs,
                        &positions_bs,
                        kv_caches,
                        &input_metadata,
                        false,
                    )?;
                    // The logits live in the capture memory pool.  Copy them
                    // into a buffer that was allocated before the capture
                    // started, so that reading them after a replay does not
                    // depend on the pool keeping the allocation alive.
                    let stable = stable_buffers
                        .get(&bs)
                        .expect("stable output buffer must be pre-allocated")
                        .clone();
                    stable.copy_(&out, 0)?;
                    outputs.insert(bs, out);
                    stable_outputs.insert(bs, stable);
                }
                if std::env::var("XINFER_DIAG_GRAPH").is_ok() {
                    eprintln!("[diag][graph] phase={:?} bs={} forward done", phase, bs);
                }
                if should_capture {
                    self.model.end_capture(!phase.is_warmup())?;
                    if std::env::var("XINFER_DIAG_CAPSLO").is_ok() {
                        // Read the KV cache right after the capture: the
                        // capture wrote to a distinctive slot, so this tells
                        // whether the captured KV write took effect at all.
                        device.synchronize()?;
                        if let Some((k, _v)) = kv_caches.and_then(|p| p.first()) {
                            let nb = k.dim(0)?;
                            let mut hits = Vec::new();
                            for b in 0..nb.min(24) {
                                let blk = k.narrow(0, b, 1)?;
                                let s = blk.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
                                if s.abs() > 1e-6 {
                                    hits.push((b, s));
                                }
                            }
                            eprintln!("[diag][graph] after-capture KV blocks: {hits:?}");
                        }
                    }
                }
                if std::env::var("XINFER_DIAG_GRAPH").is_ok() {
                    eprintln!("[diag][graph] phase={:?} bs={} capture done", phase, bs);
                }
            }
            #[cfg(feature = "cuda")]
            device.synchronize()?;
        }

        // The captured graphs keep referring to the private capture pool, so
        // that pool must stay alive for as long as the graphs exist -- the
        // addresses baked into the captured kernels live in it.
        //
        // It must NOT stay the device's *default* pool though, or every later
        // allocation in the process would be served from it and would never be
        // returned to the driver.  `restore_default_mem_pool` swaps the
        // default pool back while leaving the graph's private pool intact.
        self.model.restore_default_mem_pool()?;
        let _ = self.model.report_graph_pool_usage();
        crate::log_warn!("Captured batches {:?}", outputs.keys());
        self.graph_vars = Some(GraphCaptureVars {
            input_ids,
            positions,
            mamba_slot_mapping,
            slot_mapping,
            context_lens,
            block_tables,
            #[cfg(feature = "flashinfer")]
            flashinfer_indptr,
            #[cfg(feature = "flashinfer")]
            flashinfer_indices,
            #[cfg(feature = "flashinfer")]
            flashinfer_last_len,
            outputs,
            stable_outputs,
        });

        Ok(())
    }

    pub fn is_captured(&self, batch: usize) -> bool {
        self.graph_vars.is_some()
            && self
                .graph_vars
                .as_ref()
                .unwrap()
                .outputs
                .keys()
                .find(|&&x| x >= batch)
                .is_some()
    }

    pub fn is_exact_captured(&self, batch: usize) -> bool {
        self.graph_vars.is_some()
            && self
                .graph_vars
                .as_ref()
                .unwrap()
                .outputs
                .contains_key(&batch)
    }

    pub fn replay(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let _graph_workspace_guard = attention_rs::set_graph_workspace_domain(
            attention_rs::GraphWorkspaceDomain::DecodeGraph,
        );
        if input_metadata.is_prefill {
            candle_core::bail!("Graph replay is not used for prefill!")
        }
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;
        let input_batch = input_ids.dim(0)?;
        let require_exact_batch = input_metadata.mamba_slot_mapping.is_some();
        if let Some(graph_vars) = &self.graph_vars {
            let selected_batch = if require_exact_batch {
                graph_vars
                    .outputs
                    .keys()
                    .find(|&&x| x == input_batch)
                    .copied()
            } else {
                graph_vars
                    .outputs
                    .keys()
                    .find(|&&x| x >= input_batch)
                    .copied()
            };
            if let Some(batch) = selected_batch {
                graph_vars.input_ids.zero_()?;
                graph_vars.input_ids.copy_(&input_ids, 0)?;
                graph_vars.positions.zero_()?;
                graph_vars.positions.copy_(&positions, 0)?;

                if let Some(ms_mapping) = input_metadata.mamba_slot_mapping.as_ref() {
                    graph_vars.mamba_slot_mapping.zero_()?;
                    graph_vars.mamba_slot_mapping.copy_(&ms_mapping, 0)?;
                } else {
                    graph_vars.mamba_slot_mapping.zero_()?;
                }

                let s_mapping = input_metadata.slot_mapping.as_ref();
                graph_vars.slot_mapping.zero_()?;
                graph_vars.slot_mapping.copy_(&s_mapping, 0)?;

                let c_lens = input_metadata.context_lens.as_ref().unwrap();
                graph_vars.context_lens.zero_()?;
                graph_vars.context_lens.copy_(&c_lens, 0)?;

                let b_tables = input_metadata.block_tables.as_ref().unwrap();
                let padded_table = b_tables
                    .pad_with_zeros(1, 0, max_num_blocks - b_tables.dim(1)?)?
                    .contiguous()?;

                graph_vars.block_tables.zero_()?;
                graph_vars.block_tables.copy_(&padded_table, 0)?;

                #[cfg(feature = "flashinfer")]
                if let Some(fm) = &input_metadata.flashinfer_metadata {
                    let mut indptr_host = fm.indptr_host.clone();
                    if input_batch == batch {
                        graph_vars.flashinfer_indptr.zero_()?;
                        graph_vars.flashinfer_indptr.copy_(&fm.indptr, 0)?;
                    } else {
                        // Pad indptr to the captured batch size so graph replay sees valid lengths.
                        let last = *indptr_host.last().unwrap_or(&0);
                        for _ in (input_batch + 1)..=batch {
                            indptr_host.push(last);
                        }

                        let indptr_padded = Tensor::from_vec(
                            indptr_host.clone(),
                            (batch + 1,),
                            graph_vars.input_ids.device(),
                        )?;
                        graph_vars.flashinfer_indptr.copy_(&indptr_padded, 0)?;
                    }

                    graph_vars.flashinfer_last_len.zero_()?;
                    graph_vars.flashinfer_last_len.copy_(&fm.last_len, 0)?;

                    graph_vars.flashinfer_indices.zero_()?;
                    graph_vars.flashinfer_indices.copy_(&fm.indices, 0)?;

                    if let Some(params) = self.flashinfer_kv_params {
                        let dev = self
                            .device
                            .as_ref()
                            .ok_or_else(|| candle_core::Error::msg("graph device is missing"))?;
                        let last_len_host = fm.last_len_host.as_deref().ok_or_else(|| {
                            candle_core::Error::msg("graph replay requires last_len_host")
                        })?;
                        let kv_len_arr_host = fm.kv_len_arr_host.as_deref().ok_or_else(|| {
                            candle_core::Error::msg("graph replay requires kv_len_arr_host")
                        })?;
                        let plan = graph_decode_plan(
                            dev,
                            &params,
                            &indptr_host,
                            last_len_host,
                            kv_len_arr_host,
                            batch,
                            self.is_mla,
                            fm.use_cuda_graph,
                        )?;
                        if std::env::var("XINFER_DIAG_PLAN").is_ok() {
                            eprintln!(
                                "[diag][graph] replay plan bs={batch} plan_info={:?}",
                                plan.0.as_ref().map(|v| v.as_slice())
                            );
                        }
                    }
                }

                if std::env::var("XINFER_DIAG_ADDR").is_ok() {
                    let addr = |t: &Tensor| -> u64 {
                        use candle_core::Storage;
                        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
                        let (st, l) = t.storage_and_layout();
                        match &*st {
                            Storage::Cuda(c) => {
                                let base = match t.dtype() {
                                    DType::F32 => c.as_cuda_slice::<f32>().map(|sl| *sl.device_ptr() as u64).unwrap_or(0),
                                    DType::BF16 => c.as_cuda_slice::<half::bf16>().map(|sl| *sl.device_ptr() as u64).unwrap_or(0),
                                    DType::F16 => c.as_cuda_slice::<half::f16>().map(|sl| *sl.device_ptr() as u64).unwrap_or(0),
                                    _ => 0,
                                };
                                base.wrapping_add(l.start_offset() as u64 * t.dtype().size_in_bytes() as u64)
                            }
                            _ => 0,
                        }
                    };
                    let stable = graph_vars.stable_outputs.get(&batch);
                    let out = graph_vars.outputs.get(&batch);
                    let ordinal = self.device.as_ref().map(|d| d.ordinal()).unwrap_or(99);
                    eprintln!(
                        "[diag][addr][o{ordinal}] bs={batch} stable={:?} out={:?} input_ids={:#x} slot_mapping={:#x}",
                        stable.map(|t| format!("{:#x}", addr(t))),
                        out.map(|t| format!("{:#x}", addr(t))),
                        addr(&graph_vars.input_ids),
                        addr(&graph_vars.slot_mapping),
                    );
                }

                let result = self.model.replay(batch);
                if result.is_err() {
                    eprintln!("Error when replaying graph {:?}", result);
                }
                if std::env::var("XINFER_DIAG_KV").is_ok() {
                    #[cfg(feature = "flashinfer")]
                    {
                        let indptr = graph_vars
                            .flashinfer_indptr
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?;
                        let idx = graph_vars
                            .flashinfer_indices
                            .narrow(0, 0, 4)?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?;
                        let llen = graph_vars
                            .flashinfer_last_len
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?;
                        eprintln!(
                            "[diag][graph] fi indptr={indptr:?} indices[0..4]={idx:?} last_len={llen:?}"
                        );
                    }
                    let slot = graph_vars
                        .slot_mapping
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?;
                    let pos = graph_vars
                        .positions
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?;
                    let clen = graph_vars
                        .context_lens
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?;
                    let input_slot = input_metadata
                        .slot_mapping
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?;
                    eprintln!(
                        "[diag][graph] replay bs={batch} slot={slot:?} pos={pos:?} clen={clen:?} \
                         input_slot={input_slot:?}"
                    );
                }
                if std::env::var("XINFER_DIAG_GRAPH").is_ok() {
                    eprintln!("[diag][graph] replay bs={batch} input_batch={input_batch}");
                }

                // Read from the stable buffer: the tensor that was produced
                // during the capture lives in the graph memory pool, which the
                // driver may release after every launch.
                graph_vars
                    .stable_outputs
                    .get(&batch)
                    .unwrap_or(&graph_vars.outputs[&batch])
                    .narrow(0, 0, input_batch)?
                    .contiguous()
            } else {
                candle_core::bail!("Input batch {} is not captured!", input_batch)
            }
        } else {
            candle_core::bail!("Graph is not captured!")
        }
    }

    pub fn capture_mtp(
        &mut self,
        device: &Device,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        mtp_num_speculative: usize,
    ) -> Result<()> {
        if mtp_num_speculative == 0 {
            return Ok(());
        }

        let _graph_workspace_guard =
            attention_rs::set_graph_workspace_domain(attention_rs::GraphWorkspaceDomain::MtpGraph);
        let _prefill_guard = set_linear_is_prefill(true);
        self.device = Some(device.clone());
        let verify_len = mtp_num_speculative + 1;
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;

        // See `capture`: the graph plan workspace must not be created inside
        // the capture, or it would be served from the capture memory pool and
        // could never be released.
        #[cfg(feature = "flashinfer")]
        if let Device::Cuda(cuda_device) = device {
            attention_rs::workspace::preallocate_graph_workspace(cuda_device)?;
        }

        // Capture must use in-bounds, decode-consistent page metadata. Zeroed
        // cu_seqlens / last_len=max_model_len previously caused FlashInfer/GDN
        // OOB during capture; multirank NCCL sync surfaces that as
        // CUDA_ERROR_ILLEGAL_ADDRESS on InputMetadata drop.
        let input_ids = Tensor::zeros((verify_len,), DType::U32, device)?;
        let positions = Tensor::zeros((verify_len,), DType::I64, device)?;
        let mamba_slot_mapping = Tensor::zeros((1,), DType::I64, device)?;
        let slot_mapping = Tensor::zeros((verify_len,), DType::I64, device)?;
        let context_lens = Tensor::from_vec(vec![self.max_model_len as u32], (1,), device)?;
        let block_tables = Tensor::zeros((1, max_num_blocks), DType::U32, device)?;
        let cu_seqlens_q = Tensor::from_vec(vec![0u32, verify_len as u32], (2,), device)?;
        let cu_seqlens_k = Tensor::from_vec(vec![0u32, self.max_model_len as u32], (2,), device)?;

        #[cfg(feature = "flashinfer")]
        let last_page_len = if self.max_model_len == 0 {
            0u32
        } else {
            ((self.max_model_len - 1) % self.block_size + 1) as u32
        };

        #[cfg(feature = "flashinfer")]
        let (flashinfer_indptr, flashinfer_indices, flashinfer_last_len) = {
            let indices: Vec<u32> = (0..max_num_blocks as u32).collect();
            (
                Tensor::from_vec(vec![0u32, max_num_blocks as u32], (2,), device)?,
                Tensor::from_vec(indices, (max_num_blocks,), device)?,
                Tensor::from_vec(vec![last_page_len], (1,), device)?,
            )
        };
        #[cfg(feature = "flashinfer")]
        let flashinfer_batch_indices = Tensor::zeros((verify_len,), DType::U32, device)?;
        #[cfg(feature = "flashinfer")]
        let flashinfer_positions = {
            // Mirror runtime MTP verify append positions at max context:
            // [max_model_len - verify_len, ..., max_model_len - 1].
            let start = self.max_model_len.saturating_sub(verify_len) as u32;
            let pos: Vec<u32> = (0..verify_len as u32).map(|i| start + i).collect();
            Tensor::from_vec(pos, (verify_len,), device)?
        };

        #[cfg(feature = "flashinfer")]
        let use_flashinfer = self.flashinfer_kv_params.is_some();
        #[cfg(not(feature = "flashinfer"))]
        let use_flashinfer = false;

        let capture_in_warmup = use_flashinfer;

        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if let Some(params) = self.flashinfer_kv_params {
            let indptr_host = vec![0u32, max_num_blocks as u32];
            let kv_len_arr_host = vec![self.max_model_len as u32];
            let q_cu_seqlens_host = vec![0u32, verify_len as u32];
            let last_len_host = vec![last_page_len];

            let prefill_plan_info = attention_rs::flashinfer::graph_prefill_plan(
                device,
                &q_cu_seqlens_host,
                &indptr_host,
                &kv_len_arr_host,
                verify_len as u32,
                1,
                params.num_qo_heads,
                params.num_kv_heads,
                params.head_dim,
                params.page_size,
                params.out_dtype,
                None,
                Some(params.kv_dtype),
            )?;

            Some(attention_rs::FlashInferMetadata {
                indptr: flashinfer_indptr.clone(),
                indptr_host,
                indices: flashinfer_indices.clone(),
                last_len: flashinfer_last_len.clone(),
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                total_num_rows: Some(verify_len as u32),
                batch_indices: Some(flashinfer_batch_indices.clone()),
                positions: Some(flashinfer_positions.clone()),
                use_cuda_graph: true,
                decode_plan_info: None,
                prefill_plan_info: Some(prefill_plan_info),
                mla_decode_plan_info: None,
                mla_prefill_plan_info: None,
            })
        } else {
            None
        };
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;

        let input_metadata = InputMetadata {
            is_prefill: true,
            is_mla: self.is_mla,
            sequence_ids: Some(vec![0]),
            mamba_slot_mapping: Some(mamba_slot_mapping.clone()),
            slot_mapping: slot_mapping.clone(),
            block_tables: Some(block_tables.clone()),
            block_tables_host: None,
            context_lens_host: None,
            context_lens: Some(context_lens.clone()),
            cu_seqlens_q: Some(cu_seqlens_q.clone()),
            cu_seqlens_k: Some(cu_seqlens_k.clone()),
            max_seqlen_q: verify_len,
            max_seqlen_k: self.max_model_len,
            max_context_len: self.max_model_len,
            seqlens: None,
            flashinfer_metadata,
            is_mtp_verify: true,
        };

        let mut outputs = BTreeMap::<usize, Tensor>::new();
        let _guard = candle_core::cuda_backend::cuda_param_cache_scope(true);

        for phase in CapturePhase::ALL {
            let should_capture =
                !phase.is_cache_prewarm() && (!phase.is_warmup() || capture_in_warmup);
            if should_capture {
                self.model.start_capture(verify_len)?;
            }
            if phase.is_warmup() {
                let _ = self.model.forward(
                    &input_ids,
                    &positions,
                    kv_caches,
                    &input_metadata,
                    false,
                )?;
                #[cfg(feature = "cuda")]
                if !should_capture {
                    device.synchronize()?;
                }
            } else {
                let out = self.model.forward(
                    &input_ids,
                    &positions,
                    kv_caches,
                    &input_metadata,
                    false,
                )?;
                outputs.insert(verify_len, out);
            }
            if should_capture {
                self.model.end_capture(!phase.is_warmup())?;
            }
            #[cfg(feature = "cuda")]
            device.synchronize()?;
        }

        crate::log_warn!(
            "Captured MTP verify graph len={} (flashinfer={})",
            verify_len,
            use_flashinfer
        );

        self.mtp_graph_vars = Some(MtpGraphCaptureVars {
            input_ids,
            positions,
            mamba_slot_mapping,
            slot_mapping,
            context_lens,
            block_tables,
            cu_seqlens_q,
            cu_seqlens_k,
            #[cfg(feature = "flashinfer")]
            flashinfer_indptr,
            #[cfg(feature = "flashinfer")]
            flashinfer_indices,
            #[cfg(feature = "flashinfer")]
            flashinfer_last_len,
            #[cfg(feature = "flashinfer")]
            flashinfer_batch_indices,
            #[cfg(feature = "flashinfer")]
            flashinfer_positions,
            outputs,
        });
        Ok(())
    }

    pub fn is_mtp_captured(&self, verify_len: usize) -> bool {
        self.mtp_graph_vars
            .as_ref()
            .map_or(false, |v| v.outputs.contains_key(&verify_len))
    }

    pub fn replay_mtp(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let _graph_workspace_guard =
            attention_rs::set_graph_workspace_domain(attention_rs::GraphWorkspaceDomain::MtpGraph);
        let verify_len = input_ids.dim(0)?;
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;

        let mtp_vars = self
            .mtp_graph_vars
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("MTP graphs not captured"))?;

        if !mtp_vars.outputs.contains_key(&verify_len) {
            candle_core::bail!("MTP verify graph for len {} is not captured!", verify_len);
        }

        mtp_vars.input_ids.zero_()?;
        mtp_vars.input_ids.copy_(input_ids, 0)?;
        mtp_vars.positions.zero_()?;
        mtp_vars.positions.copy_(positions, 0)?;

        if let Some(ms_mapping) = input_metadata.mamba_slot_mapping.as_ref() {
            mtp_vars.mamba_slot_mapping.zero_()?;
            mtp_vars.mamba_slot_mapping.copy_(ms_mapping, 0)?;
        }

        mtp_vars.slot_mapping.zero_()?;
        mtp_vars
            .slot_mapping
            .copy_(&input_metadata.slot_mapping, 0)?;

        if let Some(c_lens) = input_metadata.context_lens.as_ref() {
            mtp_vars.context_lens.zero_()?;
            mtp_vars.context_lens.copy_(c_lens, 0)?;
        }

        if let Some(b_tables) = input_metadata.block_tables.as_ref() {
            let padded_table = b_tables
                .pad_with_zeros(1, 0, max_num_blocks - b_tables.dim(1)?)?
                .contiguous()?;
            mtp_vars.block_tables.zero_()?;
            mtp_vars.block_tables.copy_(&padded_table, 0)?;
        }

        if let Some(cu_q) = input_metadata.cu_seqlens_q.as_ref() {
            mtp_vars.cu_seqlens_q.copy_(cu_q, 0)?;
        }
        if let Some(cu_k) = input_metadata.cu_seqlens_k.as_ref() {
            mtp_vars.cu_seqlens_k.copy_(cu_k, 0)?;
        }

        #[cfg(feature = "flashinfer")]
        if let Some(fm) = input_metadata.flashinfer_metadata.as_ref() {
            mtp_vars.flashinfer_indptr.zero_()?;
            mtp_vars.flashinfer_indptr.copy_(&fm.indptr, 0)?;
            mtp_vars.flashinfer_indices.zero_()?;
            mtp_vars.flashinfer_indices.copy_(&fm.indices, 0)?;
            mtp_vars.flashinfer_last_len.zero_()?;
            mtp_vars.flashinfer_last_len.copy_(&fm.last_len, 0)?;
            let batch_indices = fm.batch_indices.as_ref().ok_or_else(|| {
                candle_core::Error::msg("mtp replay requires flashinfer batch_indices")
            })?;
            let positions = fm.positions.as_ref().ok_or_else(|| {
                candle_core::Error::msg("mtp replay requires flashinfer positions")
            })?;
            mtp_vars.flashinfer_batch_indices.zero_()?;
            mtp_vars.flashinfer_batch_indices.copy_(batch_indices, 0)?;
            mtp_vars.flashinfer_positions.zero_()?;
            mtp_vars.flashinfer_positions.copy_(positions, 0)?;

            if let Some(params) = self.flashinfer_kv_params {
                let dev = self
                    .device
                    .as_ref()
                    .ok_or_else(|| candle_core::Error::msg("graph device is missing"))?;
                let kv_len_arr_host = fm.kv_len_arr_host.as_deref().ok_or_else(|| {
                    candle_core::Error::msg("mtp replay requires kv_len_arr_host")
                })?;
                let q_cu_seqlens_host = vec![0u32, verify_len as u32];
                let _ = attention_rs::flashinfer::graph_prefill_plan(
                    dev,
                    &q_cu_seqlens_host,
                    &fm.indptr_host,
                    kv_len_arr_host,
                    verify_len as u32,
                    1,
                    params.num_qo_heads,
                    params.num_kv_heads,
                    params.head_dim,
                    params.page_size,
                    params.out_dtype,
                    None,
                    Some(params.kv_dtype),
                )?;
            }
        }

        self.model.replay(verify_len)?;

        mtp_vars.outputs[&verify_len].contiguous()
    }
}

unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

pub type ModelFn = dyn for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
        bool,
    ) -> Result<Tensor>
    + Send
    + Sync;

pub type CudaGraphFn = Box<
    dyn for<'a> Fn(
            &'a Tensor,
            &'a Tensor,
            Option<&'a Vec<(Tensor, Tensor)>>,
            &'a InputMetadata,
            bool,
        ) -> Result<Tensor>
        + Send
        + Sync,
>;
