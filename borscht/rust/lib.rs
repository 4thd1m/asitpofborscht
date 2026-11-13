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
//
//                  `7MM"""Yp,        `7MM"""YMM                                     `7MM"""Mq.                       mm         mm                `7MM"""Mq.  `7MMF' .M"""bgd   .g8"""bgd `7MMF'   `7MF'
//                    MM    Yb          MM    `7                                       MM   `MM.                      MM         MM                  MM   `MM.   MM  ,MI    "Y .dP'     `M   `MA     ,V
//          pd""b.    MM    dP          MM   d `7Mb,od8 ,pW"Wq.`7MMpMMMb.pMMMb.        MM   ,M9 `7MM  `7MM  ,pP"Ybd mmMMmm     mmMMmm ,pW"Wq.        MM   ,M9    MM  `MMb.     dM'       `    VM:   ,V
//         (O)  `8b   MM"""bg.          MM""MM   MM' "'6W'   `Wb MM    MM    MM        MMmmdM9    MM    MM  8I   `"   MM         MM  6W'   `Wb       MMmmdM9     MM    `YMMNq. MM              MM.  M'
//              ,89   MM    `Y          MM   Y   MM    8M     M8 MM    MM    MM        MM  YM.    MM    MM  `YMMMa.   MM         MM  8M     M8       MM  YM.     MM  .     `MM MM.mmmmm        `MM A'
//            ""Yb.   MM    ,9 ,,       MM       MM    YA.   ,A9 MM    MM    MM        MM   `Mb.  MM    MM  L.   I8   MM         MM  YA.   ,A9       MM   `Mb.   MM  Mb     dM `Mb.     ,'      :MM;
//               88 .JMMmmmd9  db     .JMML.   .JMML.   `Ybmd9'.JMML  JMML  JMML.    .JMML. .JMM. `Mbod"YML.M9mmmP'   `Mbmo      `Mbmo`Ybmd9'      .JMML. .JMM..JMML.P"Ybmmd"    `"bmmmd'        VF
//         (O)  .M'
//          bmmmd'
// **********************************************************************************************************************
//                                               borscht (Literary Source)
// <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->
//     tensor_managed.py                               3A. From Python to Rust
//                                                         - tensor_nativecpu.py
//                                                           0. From Mathematics to Programming
//                                                           1. From Lambda Calculus to Turing Machines
//                                                           2. From Virtual Machines to Physical Machines
//                                                         - pyproject.toml/Cargo.toml
//                                                           3. From Python to C
//                                                           4. From Python Rust
//                                                     3B. From Rust to RISC-V
//                                                         - lib.rs/kernelscpu.rs
//                                                           5. From Big-O to Rooflines
//                                                           6. von Neumann Architecture = Computation + Communication
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
