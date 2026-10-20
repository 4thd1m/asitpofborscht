struct Tensor {
  shape: (u8, u8),
  stride: (u8, u8),
  dtype: DType
}

enum DType {
  Fp64, Fp32, Fp16
}

impl Tensor {
  fn add(x: &Self) -> Self {
    todo!()
  }

  fn gemv(x: &Self) -> Self {
    todo!()
  }

  fn gemm(x: &Self) -> Self {
    todo!()
  }
}
