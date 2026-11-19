use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
  use super::*;

  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(c_elem) = c.get_mut(idx) {
      *c_elem = a[idx_raw] + b[idx_raw];
    }
  }

  /// Naive C = α*(A@B) + β*C
  /// for the row-major (M,K) matrix `a`, (K,N) matrix `b`, and (M,N) matrix `c`.
  /// Each thread computes one element of C, launched as a grid of 32x32 thread blocks.
  #[kernel]
  #[launch_bounds(1024)]
  #[allow(clippy::too_many_arguments)] // the signature of sgemm_naive in the article
  pub fn sgemm_naive(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32], beta: f32, mut c: DisjointSlice<f32>) {
    // compute position in C that this thread is responsible for
    let x = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    let y = (thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y()) as usize;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if x < m && y < n {
      let mut tmp = 0.0;
      for i in 0..k {
        tmp += a[x * k + i] * b[i * n + y];
      }
      // C = α*(A@B)+β*C
      // Like the article, x walks down C's rows: threadIdx.x is the fast index within a warp, so a warp's 32
      // threads read A and write C 32 rows apart (uncoalesced) — the inefficiency kernel 2 fixes by swapping
      // x and y. cuda-oxide's safe 2D API (thread::index_2d_runtime, and the 2D slices a
      // #[launch_contract(domain = 2)] requires) always maps x to columns, i.e. it *is* kernel 2, so to stay
      // faithful to kernel 1 we write C through a raw pointer and launch without a contract instead.
      // (Not c.get_unchecked_mut: in debug builds, such as `cargo oxide test`'s, its debug_assert! on
      // the slice's length fails cuda-oxide's device codegen.)
      // SAFETY: x < m and y < n, so x*n + y < m*n == c.len() (the host allocates c with m*n elements),
      //         and every thread of the launch has a distinct (x, y), so no two threads alias an element.
      let c_elem = unsafe { &mut *c.as_mut_ptr().add(x * n + y) };
      *c_elem = alpha * tmp + beta * *c_elem;
    }
  }
}

// Run with `cargo oxide test`: plain `cargo test` compiles the #[kernel]s for the CPU only, so the test binary
// has no PTX for kernels::load to find and fails to link.
#[cfg(test)]
mod tests {
  use super::kernels;
  use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};

  #[test]
  fn vecadd() -> Result<(), Box<dyn std::error::Error>> {
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
    let errors = (0..N).filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5).count();
    assert_eq!(errors, 0, "{errors} of {N} elements wrong");
    Ok(())
  }
}
