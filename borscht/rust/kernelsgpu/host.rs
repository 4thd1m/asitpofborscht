use super::utils::{self, Gpu};

use cuda_core::{DeviceBuffer, DriverError, LaunchConfig, Stream};
use std::error::Error;

const BLOCK: u32 = 32; // sgemm_naive's 32x32 thread blocks

/// Multiplies the row-major (n,k) matrix `a` with the (k,m) matrix `b` into a new (n,m) matrix, on the GPU.
/// Expects a.len() == n*k, b.len() == k*m, and n, k, m > 0, which kernelsgpu.rs checks before calling.
pub fn _sgemm(a: Vec<f32>, b: Vec<f32>, n: usize, k: usize, m: usize) -> Result<Vec<f32>, Box<dyn Error>> {
  let gpu = utils::gpu()?;
  let stream = gpu.ctx.default_stream();
  let (a, b, mut c) = (DeviceBuffer::from_host(&stream, &a)?, DeviceBuffer::from_host(&stream, &b)?, DeviceBuffer::<f32>::zeroed(&stream, n * m)?);
  launch_sgemm_naive(gpu, &stream, (n, k, m), &a, &b, &mut c)?;
  Ok(c.to_host_vec(&stream)?)
}

/// Enqueues C = A@B for the (n,k) matrix `a`, (k,m) matrix `b`, and (n,m) matrix `c` already on the device.
/// borscht names C's shape (n,m) while the article names it (M,N), hence the swap of letters below.
pub fn launch_sgemm_naive(gpu: &Gpu, stream: &Stream, (n, k, m): (usize, usize, usize), a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>, c: &mut DeviceBuffer<f32>) -> Result<(), DriverError> {
  let config = LaunchConfig {
    grid_dim: (n.div_ceil(BLOCK as usize) as u32, m.div_ceil(BLOCK as usize) as u32, 1), // create as many blocks as necessary to map all of C
    block_dim: (BLOCK, BLOCK, 1),                                                        // 32 * 32 = 1024 thread per block
    shared_mem_bytes: 0,
  };
  let (alpha, beta) = (1.0, 0.0);
  // SAFETY: the grid covers C with 2D blocks of at most launch_bounds' 1024 threads, and the callers allocate
  //         a, b, and c with exactly n*k, k*m, and n*m elements, the bound sgemm_naive's unchecked write relies on.
  unsafe { gpu.module.sgemm_naive(stream, config, n, m, k, alpha, a, b, beta, c) }
}

/// Seconds the GPU spends in one sgemm_naive launch on an (n,k) @ (k,m) problem, averaged over `reps`
/// back-to-back launches and timed with CUDA events. Expects n, k, m, and reps > 0.
pub fn _sgemm_bench(n: usize, k: usize, m: usize, reps: usize) -> Result<f64, Box<dyn Error>> {
  let gpu = utils::gpu()?;
  let stream = gpu.ctx.default_stream();
  let a = DeviceBuffer::from_host(&stream, &vec![1.0f32; n * k])?;
  let b = DeviceBuffer::from_host(&stream, &vec![1.0f32; k * m])?;
  let mut c = DeviceBuffer::<f32>::zeroed(&stream, n * m)?;

  launch_sgemm_naive(gpu, &stream, (n, k, m), &a, &b, &mut c)?; // warm up
  let (start, end) = (gpu.ctx.new_event()?, gpu.ctx.new_event()?);
  start.record(&stream)?;
  for _ in 0..reps {
    launch_sgemm_naive(gpu, &stream, (n, k, m), &a, &b, &mut c)?;
  }
  end.record(&stream)?;
  end.synchronize()?;
  let ms = start.elapsed_time(&end)?;
  Ok(f64::from(ms) / 1e3 / reps as f64)
}
