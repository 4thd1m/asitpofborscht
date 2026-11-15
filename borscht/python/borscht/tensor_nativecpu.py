#                                                          ,,                                                                                                                                                        
#                       db                               `7MM                          mm               MMP""MM""YMM                                                                                                 
#                      ;MM:                                MM                          MM               P'   MM   `7                                                                                                 
#  pd""b.             ,V^MM.    ,p6"bo   ,p6"bo   .gP"Ya   MM  .gP"Ya `7Mb,od8 ,6"Yb.mmMMmm .gP"Ya           MM  .gP"Ya `7MMpMMMb.  ,pP"Ybd  ,pW"Wq.`7Mb,od8                                                         
# (O)  `8b           ,M  `MM   6M'  OO  6M'  OO  ,M'   Yb  MM ,M'   Yb  MM' "'8)   MM  MM  ,M'   Yb          MM ,M'   Yb  MM    MM  8I   `" 6W'   `Wb MM' "'                                                         
#      ,89           AbmmmqMA  8M       8M       8M""""""  MM 8M""""""  MM     ,pm9MM  MM  8M""""""          MM 8M""""""  MM    MM  `YMMMa. 8M     M8 MM                                                             
#    ""Yb. ,,       A'     VML YM.    , YM.    , YM.    ,  MM YM.    ,  MM    8M   MM  MM  YM.    ,          MM YM.    ,  MM    MM  L.   I8 YA.   ,A9 MM                                                             
#       88 db     .AMA.   .AMMA.YMbmd'   YMbmd'   `Mbmmd'.JMML.`Mbmmd'.JMML.  `Moo9^Yo.`Mbmo`Mbmmd'        .JMML.`Mbmmd'.JMML  JMML.M9mmmP'  `Ybmd9'.JMML.                                                           
# (O)  .M'
#  bmmmd'                                                                                                                                                                                                            
# **********************************************************************************************************************
#                                               borscht (Literary Source)
# <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->
#             tensor_managed.py                          tensor_nativecpu.py
#                                                        pyproject.toml/Cargo.toml
#                                                        lib.rs/kernelscpu.rs
# **********************************************************************************************************************
# After reading File 1. __init__.py, you have a 50,000 ft view of borscht's internals.
# After reading File 2. tensor_managed.py, you now understand the core essence of the Tensor concept and it's primary algorithm of matrix multiplication
#            In File 3. let us analyze the *performance* tensor_managed.py's matrix multiplication routine mm

from borscht.tensor_managed import Tensor                 # Let us import Tensor to define our accelerated mm routine
from borscht.tensor_managed import full, mm as mm_managed # Let us import the full constructor and managed matrix multiply routine
__all__ = ["Tensor", "full", "mm"]             # Let us re-export the full constructor and Tensor concept as-is from tensor_managed.py as well as our natively accelerated mm

# Recall in File 2 we defined
# - the Tensor data structure...           aspirationally representing...           the set R^{d1 \times \cdots, \times dn} known as a field
# - the matrix multiplication algorithm... aspirationally representing...           the function L1: R^k -> R^n and L2: R^m -> R^k known as a composition of linear transformation

def _main():
    import doctest
    doctest.testmod()

    A = full((2,3), 3.141592) # some pie instantiated by the Tensor data structure..
    B = full((3,2), 3.141592)
    C = mm_managed(A,B) # ..evaluating the matrix multiplication algorithm
    print(C)

    # Is this fast?
    # According to Big-O, the time complexity is n -> n^3.
    # The transition from mathematics to programming is the transition from lambda calculus' substitution model
    # of computation to turing machines von neumann model of computation

    n = 256
    A, B = full((n, n), 1.0), full((n, n), 1.0)
    import time

    # Continuing down the route of classical algorithm analysis,
    # we would be led to an algorithm known as Strassen's Algorithm, with time complexity n -> n^2.7.,. lg()

    # Instead of analytically analyzing the software algorithm's asymptotics to predict worst-case time *complexity*,
    #     we can empirically        use the hardware performance counters    to measure actual time



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


# <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->
#     tensor_managed.py                               3A. From Big-O of Virtual Machines to Rooflines of Physical Machines
#                                                         - tensor_nativecpu.py
#                                                         - pyproject.toml/Cargo.toml
#                                                     3B. von Neumann Architecture = Computation + Communication
#                                                         - lib.rs/kernelscpu.rs
