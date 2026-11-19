use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};

mod kernelsgpu;
use kernelsgpu::kernels;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let ctx = CudaContext::new(0)?;
  let stream = ctx.default_stream();

  const N: usize = 1024;
  let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
  let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

  let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
  let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
  let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

  // SAFETY: this package owns the embedded device bundle produced for the
  // kernels module above.
  let module = unsafe { kernels::load(&ctx)? };
  let prepared = module.prepare_vecadd(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;
  module.vecadd(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;

  let c_host = c_dev.to_host_vec(&stream)?;
  let errors = (0..N)
    .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
    .count();

  if errors == 0 {
    println!("PASSED: all {} elements correct", N);
  } else {
    eprintln!("FAILED: {} errors", errors);
    std::process::exit(1);
  }
  Ok(())
}
