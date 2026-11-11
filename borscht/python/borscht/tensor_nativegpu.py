# **********************************************************************************************************************
#                                               borscht (Literary Source)
# <-- File 3. Accelerate Tensor               -- File 4. Accelerate++ Tensor --                     File 4. Abstract Accelerate -->
#     3A. From Python to Rust
#         - tensor_nativecpu.py
#         - pyproject.toml
#         - Cargo.toml
#         - rust/lib.rs
#     3B. From Rust to RISC-V
#         - kernelscpu.rs
# **********************************************************************************************************************

from borscht.tensor_managed import Tensor                 # Let us import Tensor to define our GPU accelerated mm routine
from borscht.tensor_managed import full, mm as mm_managed # Let us import the full constructor and managed matrix multiply routine
from borscht.tensor_nativecpu import mm as mm_native      # Let us import                          native  matrix multiply routine
__all__ = ["Tensor", "full", "mm"]             # Let us re-export the full constructor and Tensor concept as-is from tensor_managed.py as well as our natively GPU accelerated mm


def _main():
    import doctest
    doctest.testmod()

    # === File 2. Hello Tensor
    A = full((2,3), 3.141592) # some pie ...
    B = full((3,2), 3.141592)
    C = mm_managed(A,B) # .. and some borscht!
    print(C)

    n = 256
    A, B = full((n, n), 1.0), full((n, n), 1.0)
    import time    

    # matmul implemented by a managed language running on a virtual machine
    t0 = time.perf_counter()
    mm_managed(A, B)
    t = time.perf_counter() - t0
    print(f"{n}x{n} mm_managed: {t*1e3:.1f} ms, {2*n**3 / t / 1e9:.4f} GFLOP/s")

    # === File 3. Hello Tensor
    # matmul implemented by a native language running on a physical latency-oriented machine
    t0 = time.perf_counter()
    mm_native(A, B)
    t = time.perf_counter() - t0
    print(f"{n}x{n} mm_native: {t*1e3:.1f} ms, {2*n**3 / t / 1e9:.4f} GFLOP/s")

    # matmul implemented  by a native language running on a physical throughput-oriented machine

import math

def mm(A: Tensor, B: Tensor) -> Tensor:
    return mm_native(A,B)

from borscht._borscht import _gemm   # at the top of the file
def mm_native(A: Tensor, B: Tensor) -> Tensor:
    """Multiply the (n,k) matrix `A` with the (k,m) matrix `B` into a new (n,m) matrix.

    Examples:
        >>> A = full((2, 3), 3.141592)
        >>> B = full((3, 2), 3.141592)
        >>> mm(A, B)
        tensor([[29.6088, 29.6088],
                [29.6088, 29.6088]])
        >>> mm(A, A)
        Traceback (most recent call last):
        ...
        RuntimeError: A and B shapes cannot be multiplied (2x3 and 2x3)
    """
    # preconditions for well-defined matrix multiplication
    (nrows, p), (p2, mcols) = A.shape, B.shape
    if len(A.shape) != 2 or len(B.shape) != 2: raise RuntimeError("self must be a matrix" if len(A.shape) != 2 else "B must be a matrix")
    if p != p2: raise RuntimeError(f"A and B shapes cannot be multiplied ({nrows}x{p} and {p2}x{mcols})")
    output = full((nrows, mcols), 0.0)
    output.storage = _gemm(A.storage, B.storage, nrows, p, mcols)
    return output

if __name__ == "__main__": _main()