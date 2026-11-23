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
//                                             Part 1: From Arrays to NDArrays
// <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->
//             tensor_managed.py                          tensor_nativecpu.py                     tensor_nativegpu.py
//                                                        pyproject.toml/Cargo.toml/lib.rs        Cargo.toml/lib.rs
//                                                      > kernelscpu.rs <                         kernelsgpu.rs
//                                                                                                  utils.rs
//                                                                                                  host.rs
//                                                                                                  device.rs
// **********************************************************************************************************************

// After reading File 2. tensor_managed.py, you now understand the core essence of the Tensor concept and it's primary algorithm of matrix multiplication
// After reading File 3A. From Python to Rust pyproject.toml/Cargo.toml, we understand how Python and Rust interop together.

use pyo3::prelude::*;

fn _axpy(_x: &[u8], _y: &[u8]) -> () {
  todo!()
}

fn _gemv(_x: &[u8], _y: &[u8]) -> () {
  todo!()
}

#[pyfunction]
pub fn _gemm(a: Vec<f64>, b: Vec<f64>, n: usize, k: usize, m: usize) -> Vec<f64> {
  let mut c = vec![0.0; n * m];
  for i in 0..n {
    for j in 0..m {
      for p in 0..k {
        c[i * m + j] += a[i * k + p] * b[p * m + j];
      }
    }
  }
  c
}
