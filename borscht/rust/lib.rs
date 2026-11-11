// **********************************************************************************************************************
//                                               borscht (Literary Source)
// <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->
//     tensor_managed.py                               3A. From Python to Rust
//                                                         - tensor_nativecpu.py
//                                                         - pyproject.toml
//                                                         - Cargo.toml
//                                                         - lib.rs              <-- HERE
//                                                     3B. From Rust to RISC-V
//                                                         - kernelscpu.rs
// **********************************************************************************************************************

use pyo3::prelude::*;
mod kernelscpu;
mod kernelsgpu;

// mod tensor;
// mod engine;

/// A Python module implemented in Rust.
#[pymodule]
mod _borscht {
  use pyo3::prelude::*;

  #[pymodule_export]
  use crate::kernelscpu::_gemm;

  /// Formats the sum of two numbers as string.
  #[pyfunction]
  fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
  }
}
