mod host;
pub mod device;
mod utils;

use pyo3::{
  exceptions::{PyRuntimeError, PyValueError},
  prelude::*,
};

// The Python interop layer: the #[pyfunction]s lib.rs exports, which check their arguments and raise Python
// exceptions, around host.rs, which knows nothing of Python and reports failures as Rust errors.

fn py_err(e: impl std::fmt::Display) -> PyErr {
  PyRuntimeError::new_err(e.to_string())
}

/// Multiplies the row-major (n,k) matrix `a` with the (k,m) matrix `b` into a new (n,m) matrix, on the GPU.
#[pyfunction]
pub fn _sgemm(a: Vec<f32>, b: Vec<f32>, n: usize, k: usize, m: usize) -> PyResult<Vec<f32>> {
  // Checked here because an out-of-bounds read on the device traps, and a trap poisons the CUDA context
  // for every later call in the process.
  if a.len() != n * k || b.len() != k * m {
    return Err(PyValueError::new_err(format!("expected {} and {} elements for ({n}x{k}) @ ({k}x{m}), got {} and {}", n * k, k * m, a.len(), b.len())));
  }
  if n * k * m == 0 {
    return Ok(vec![0.0; n * m]); // nothing to launch, and the driver rejects zero-sized buffers and grids
  }

  host::_sgemm(a, b, n, k, m).map_err(py_err)
}

/// Seconds the GPU spends in one sgemm_naive launch on an (n,k) @ (k,m) problem, averaged over `reps`
/// back-to-back launches and timed with CUDA events. Unlike timing `_sgemm` from Python, this leaves out
/// converting Python lists and copying them between host and device, so it measures the kernel alone.
#[pyfunction]
pub fn _sgemm_bench(n: usize, k: usize, m: usize, reps: usize) -> PyResult<f64> {
  if n * k * m == 0 || reps == 0 {
    return Err(PyValueError::new_err("n, k, m, and reps must all be positive"));
  }

  host::_sgemm_bench(n, k, m, reps).map_err(py_err)
}
