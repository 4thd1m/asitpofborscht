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

// von Neumann Architecture: computation, communication

// roofline
// benchmark, profile
// memory hierarchy

// flynn's taxonomy
