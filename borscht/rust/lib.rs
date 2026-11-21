//                                                          ,,
//                       db                               `7MM                          mm               MMP""MM""YMM
//                      ;MM:                                MM                          MM               P'   MM   `7
//  pd""b.             ,V^MM.    ,p6"bo   ,p6"bo   .gP"Ya   MM  .gP"Ya `7Mb,od8 ,6"Yb.mmMMmm .gP"Ya           MM  .gP"Ya `7MMpMMMb.  ,pP"Ybd  ,pW"Wq.`7Mb,od8
// (O)  `8b           ,M  `MM   6M'  OO  6M'  OO  ,M'   Yb  MM ,M'   Yb  MM' "'8)   MM  MM  ,M'   Yb          MM ,M'   Yb  MM    MM  8I   `" 6W'   `Wb MM' "'
//      ,89           AbmmmqMA  8M       8M       8M""""""  MM 8M""""""  MM     ,pm9MM  MM  8M""""""          MM 8M""""""  MM    MM  `YMMMa. 8M     M8 MM
//    ""Yb. ,,       A'     VML YM.    , YM.    , YM.    ,  MM YM.    ,  MM    8M   MM  MM  YM.    ,          MM YM.    ,  MM    MM  L.   I8 YA.   ,A9 MM
//       88 db     .AMA.   .AMMA.YMbmd'   YMbmd'   `Mbmmd'.JMML.`Mbmmd'.JMML.  `Moo9^Yo.`Mbmo`Mbmmd'        .JMML.`Mbmmd'.JMML  JMML.M9mmmP'  `Ybmd9'.JMML.
// (O)  .M'
//  bmmmd'
// **********************************************************************************************************************
//                                               borscht (Literary Source)
// <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->
//             tensor_managed.py                          3A. tensor_nativecpu.py                 4A. tensor_nativegpu.py
//                                                        3B. pyproject.toml/Cargo.toml           4B. Cargo.toml/lib.rs
//                                                        3C. >lib.rs</kernelscpu.rs              4C. kernelsgpu.rs
//                                                                                                      utils.rs
//                                                                                                      host.rs
//                                                                                                      device.rs
// **********************************************************************************************************************

use pyo3::prelude::*;
// mod tensor;
// mod engine;
mod kernelscpu;

/// A Python module implemented in Rust.
#[pymodule]
mod _borscht {
  use pyo3::prelude::*;

  #[pymodule_export]
  use crate::kernelscpu::_gemm;
  #[cfg(feature = "cuda")]
  #[pymodule_export]
  use crate::kernelsgpu::{_sgemm, _sgemm_bench};

  /// Formats the sum of two numbers as string.
  #[pyfunction]
  fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
  }
}

#[cfg(feature = "cuda")] // see [features] in Cargo.toml
mod kernelsgpu;
