use super::device::kernels;

use cuda_core::CudaContext;
use std::error::Error;
use std::sync::{Arc, OnceLock};

pub struct Gpu {
  pub ctx: Arc<CudaContext>,
  pub module: kernels::LoadedModule,
}

/// The CUDA context and loaded kernels module, made on first use and kept for the life of the process.
/// Creating a context and JIT-compiling the embedded PTX to the GPU's SASS takes hundreds of milliseconds,
/// which would otherwise swamp every call.
pub fn gpu() -> Result<&'static Gpu, Box<dyn Error>> {
  // A failed setup is kept as its message, so every later call reports the same error.
  static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new();
  GPU
    .get_or_init(|| {
      let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
      // SAFETY: the bundle embedded in this extension module is the one cargo oxide compiled from `kernels`.
      let module = unsafe { kernels::load(&ctx) }.map_err(|e| e.to_string())?;
      Ok(Gpu { ctx, module })
    })
    .as_ref()
    .map_err(|e| e.clone().into())
}
