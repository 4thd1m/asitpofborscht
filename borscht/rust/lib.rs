use pyo3::prelude::*;
// mod tensor;
// mod engine;
mod kernelscpu;
mod kernelsgpu;

/// A Python module implemented in Rust.
#[pymodule]
mod _borscht {
  use pyo3::prelude::*;

  #[pymodule_export]
  use crate::kernelscpu::_gemm;
  #[pymodule_export]
  use crate::kernelsgpu::{_sgemm, _sgemm_bench};

  /// Formats the sum of two numbers as string.
  #[pyfunction]
  fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
  }
}
