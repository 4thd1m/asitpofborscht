# *********************************************************************************************************************
#                                                borscht (Literary Source)
# <-- File 2. tensor_managed.py              -- File 3. tensor_nativecpu.py --                 File 4. kernelscpu.rs -->
#                                                       pyproject.toml
#                                                       Cargo.toml
#                                                       rust/lib.rs
# *********************************************************************************************************************
# After reading File 1. __init__.py, you have a 50,000 ft view of borscht's internals.
# After reading File 2. tensor_managed.py, you now understand the core essence of the Tensor concept and the primary operation of matrix multiplication
#            In File 3. let us analyze and benchmark tensor_managed.py's matrix multiplication routine mm
#                       and subsequently set up a mixed Python and Rust source for the borscht Python package
#                       so that we can accelerate our matrix multiply with a native implementation on a RISC-V machine in File 4. kernelscpu.rs

from borscht.tensor_managed import full, mm as mm_managed # Let us import the full constructor and mm routine as mm_managed from tensor_managed.py to benchmark against our accelerated version
from borscht.tensor_managed import Tensor                 # Let us import Tensor to define our accelerated mm routine
__all__ = ["Tensor", "full", "mm"]             # Let us re-export the full constructor and Tensor concept as-is from tensor_managed.py
                                                          #   as well as the mm


def _main():
    import doctest
    doctest.testmod()



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


    # matmul implemented by a native language running on a physical machine
    t0 = time.perf_counter()
    mm_native(A, B)
    t = time.perf_counter() - t0
    print(f"{n}x{n} mm_native: {t*1e3:.1f} ms, {2*n**3 / t / 1e9:.4f} GFLOP/s")

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