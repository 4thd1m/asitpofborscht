def axpy(x: bytes, b: bytes) -> bytes:
  raise NotImplementedError("")

def gemv(x: bytes, b: bytes) -> bytes:
  raise NotImplementedError("")

def gemm(a: bytes, b: bytes, m: int, n: int, k: int) -> bytes:
  # C[m,n] = A[m,k] @ B[k,n], row-major f32
  A = memoryview(a).cast("f")
  B = memoryview(b).cast("f")
  C = memoryview(bytearray(m * n * 4)).cast("f")
  for i in range(m):
    for j in range(n):
      acc = 0.0
      for p in range(k):
        acc += A[i * k + p] * B[p * n + j]
      C[i * n + j] = acc
  return C.tobytes()