//                                                          ,,
//                       db                               `7MM                          mm               MMP""MM""YMM
//                      ;MM:                                MM                          MM               P'   MM   `7
//  pd""b.             ,V^MM.    ,p6"bo   ,p6"bo   .gP"Ya   MM  .gP"Ya `7Mb,od8 ,6"Yb.mmMMmm .gP"Ya           MM  .gP"Ya `7MMpMMMb.  ,pP"Ybd  ,pW"Wq.`7Mb,od8
// (O)  `8b           ,M  `MM   6M'  OO  6M'  OO  ,M'   Yb  MM ,M'   Yb  MM' "'8)   MM  MM  ,M'   Yb          MM ,M'   Yb  MM    MM  8I   `" 6W'   `Wb MM' "'
//      ,89           AbmmmqMA  8M       8M       8M""""""  MM 8M""""""  MM     ,pm9MM  MM  8M""""""          MM 8M""""""  MM    MM  `YMMMa. 8M     M8 MM
//    ""Yb. ,,       A'     VML YM.    , YM.    , YM.    ,  MM YM.    ,  MM    8M   MM  MM  YM.    ,          MM YM.    ,  MM    MM  L.   I8 YA.   ,A9 MM
//       88 db     .AMA.   .AMMA.YMbmd'   YMbmd'   `Mbmmd'.JMML.`Mbmmd'.JMML.  `Moo9^Yo.`Mbmo`Mbmmd'        .JMML.`Mbmmd'.JMML  JMML.M9mmmP'  `Ybmd9'.JMML.
// (O)  .M'  After reading File 1. Hello Borscht (__init__.py), you have a 50,000 ft view of borscht's internals.
//  bmmmd'   After reading File 2. Hello Tensor (tensor_managed.py), you understand the core essence of the Tensor concept and it's primary algorithm of matrix multiplication
//           After reading File 3A. Accelerate Tensor (tensor_nativecpu.py), you understand why deep learning frameworks accelerate their algorithms
//           After reading File 3B. Accelerate Tensor (pyproject.toml), you understand how the Python package can implement extension modules via CPython API
//           After reading File 3B. Accelerate Tensor (Cargo.toml), you understand how the Rust package implements an extension module with Rust FFI
//           With          File 3C. lib.rs and kernelscpu.rs, we will finally transliterate our matrix multiplication algorithm from Python to Rust with _gemm() in the _borscht extension module
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

// For CPU accelerated code, borscht uses the Rust programming language rather than C/C++,
// while still maintaining the Python veneer for user productivity.
// This is possible *because* the Rust language has a feature that supports Foreign Function Interface (FFI) with the keyword extern
// — In short, FFI is a way for a programming language A to define functions and enable programming language B to call them
//   by specifying a specific Application Binary Interface (ABI) which affects the format of the compiled assembly code.
//   In particular, we would specifcy the "C" ABI chosen by the dominant compiler *for* the current target

// [unsafe(no_mangle)]
// pub extern "C" fn call_from_c() {
//     println!("Just called a Rust function from C!");
// }
// book: https://rust-book.cs.brown.edu/ch20-01-unsafe-rust.html#calling-rust-functions-from-other-languages
// rustonomicon: https://doc.rust-lang.org/nomicon/ffi.html#ffi-and-unwinding
// reference: https://doc.rust-lang.org/reference/items/external-blocks.html#abi

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
