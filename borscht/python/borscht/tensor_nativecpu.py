# **********************************************************************************************************************
#                                               borscht (Literary Source)
# <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->
#     tensor_managed.py                               3A. From Python to Rust
#                                                         - tensor_nativecpu.py
#                                                           0. Mathematicians vs Programmers           <-- HERE
#                                                           1. From Lambda Calculus to Turing Machines
#                                                           2. From Virtual Machines to Physical Machines
#                                                         - pyproject.toml
#                                                         - Cargo.toml
#                                                         - lib.rs
#                                                     3B. From Rust to RISC-V
#                                                         - kernelscpu.rs
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

# If a mathematician or physicist didn't know about the internals of computers started using borscht.Tensors,
# they could construct and read Tensors...
"""
>>> import borscht
>>> A = borscht.full((2,3), 3.141592)               # 1. creating tensors + ...
>>> B = borscht.full((3,2), 3.141592)

>>> A                                               # 2. reading tensors + ...
tensor([[3.1416, 3.1416, 3.1416],
        [3.1416, 3.1416, 3.1416]])
>>> B
tensor([[3.1416, 3.1416],
        [3.1416, 3.1416],
        [3.1416, 3.1416]])
"""
# treated them *as if* they were
# coordinates of some basis in
#                      a field R^{d1 \times \cdots, \times dn},
#                                 \_____________ _____________/
#                                               |                         +----------------------------+
#                                               |                         | Tensor                     |
#                                               | is transliterated to    +----------------------------+
#                                               +------------------------>| + shape:   tuple[int, ...] |
#                                                                         | + ndim:    int             | We use + to denote user-visible state
#                                                                         | - stride:  tuple[int, ...] |    and - to denote user-hidden state (abstracted away)
#                                                                         | - storage: list[float]     | Hidden from them is stride/storage
#                                                                         +----------------------------+ Even though they are technically not private Python fields (prefixed with _ in python)

# And moreover, multiplying two matrices...
"""
>>> import borscht
>>> A = borscht.full((2,3), 3.141592)               # 1. creating tensors + ...
>>> B = borscht.full((3,2), 3.141592)

>>> A                                               # 2. reading tensors + ...
tensor([[3.1416, 3.1416, 3.1416],
        [3.1416, 3.1416, 3.1416]])
>>> B
tensor([[3.1416, 3.1416],
        [3.1416, 3.1416],
        [3.1416, 3.1416]])

>>> borscht.mm(A,B)                                 # 3. updating tensors + ...
tensor([[29.6088, 29.6088],
        [29.6088, 29.6088]])
"""

# is simply treated as evaluating the definition of matrix multiplication
#                                                   mm: R^{m x p} x R^{p x n} -> R^{n x m}
#                                                       (aij      , bij  )   |-> Σ_{k=1}^{p} a_il*b_lj
#                                                                                \_________ _________/
#                                                                                          |
#                                                                                          |
#                                                                                          |                  A_NP, B_PM, C_NM = A, B, C                                                        \
#                                                                                          |                  (ijmp, kjmp), (k2jmp, jjmp), (istridec, jstridec) = A.stride, B.stride, C.stride   |
#                                                                                          |                  A_buf, B_buf, C_buf = A_NP.storage, B_PM.storage, C_NM.storage                      > Hidden from them is
#                                                                                          |                  for i in range(nrows):                                                             |  - the mapping of logical indices to physical ones
#                                                                                          |                      for j in range(mcols):                                                        /   - the procedural control flow to traverse output matrix C
#                                                                                          |                          dot_product = 0.0
#                                                                                          | is transliterated to     for k in range(p):
#                                                                                          +----------------------------> dot_product += A_buf[i*ijmp + k*kjmp] * B_buf[k*k2jmp + j*jjmp]
#                                                                                                                     C_buf[i*istridec + j*jstridec] = dot_product


#   While the mathematicians have to come to terms with their ideals of  real numbers and real analysis not actually being so *real*
#   and join the physicists and computer scientists in reality with floating point numerics, and numerical analysis (which we'll learn more about soon),
#   the current Tensor data structure and matrix multiplication algorithm is satisfactory — They don't even need to worry about any striding nonsense!
#                                                                                           Striding memory is so below the working mathematician!

# That is, according to the *definition* of
# - the set R^{d1 \times \cdots, \times dn} known as a field and
# - the function L1: R^k -> R^n and L2: R^m -> R^k known as a composition of linear transformation,
# borscht.mm() evaluates the correct answer (*with high confidence).

# While mathematics and computation have a close relationship where they concern themselves with
# the same objects or the same mathema (μάθημα), which in greek literally means “that which is learnt”, “what one gets to know” and “these things.”
# In the case of the Tensor data structure and it's matrix multiplication algorithm, mathematicians prioritize the *definition* and properties.
# They are not so much interested in their *evaluation* (except for perhaps, numerical analysts. Numerical analysts are more like our first cousins, while mathematicians second cousins),
# **This is where programmers differ from our mathematical cousins, because we care about the evaluation of these objects.**
# That is, in addition to caring about *correctness*,
#                   we also care about *performance*.                                                                                                                                                                                         
#                                                                                                                                                                         ,,                                                                    
# `7MM"""Mq.                                                                                                                                                             *MM                             mm                                     
#   MM   `MM.                                                                                                                                                             MM                             MM                                     
#   MM   ,M9 `7Mb,od8 ,pW"Wq.   .P"Ybmmm `7Mb,od8 ,6"Yb.  `7MMpMMMb.pMMMb.  `7MMpMMMb.pMMMb.  .gP"Ya `7Mb,od8 ,pP"Ybd      ,p6"bo   ,6"Yb.  `7Mb,od8 .gP"Ya       ,6"Yb.  MM,dMMb.   ,pW"Wq.`7MM  `7MM mmMMmm                                   
#   MMmmdM9    MM' "'6W'   `Wb :MI  I8     MM' "'8)   MM    MM    MM    MM    MM    MM    MM ,M'   Yb  MM' "' 8I   `"     6M'  OO  8)   MM    MM' "',M'   Yb     8)   MM  MM    `Mb 6W'   `Wb MM    MM   MM                                     
#   MM         MM    8M     M8  WmmmP"     MM     ,pm9MM    MM    MM    MM    MM    MM    MM 8M""""""  MM     `YMMMa.     8M        ,pm9MM    MM    8M""""""      ,pm9MM  MM     M8 8M     M8 MM    MM   MM                                     
#   MM         MM    YA.   ,A9 8M          MM    8M   MM    MM    MM    MM    MM    MM    MM YM.    ,  MM     L.   I8     YM.    , 8M   MM    MM    YM.    ,     8M   MM  MM.   ,M9 YA.   ,A9 MM    MM   MM                                     
# .JMML.     .JMML.   `Ybmd9'   YMMMMMb  .JMML.  `Moo9^Yo..JMML  JMML  JMML..JMML  JMML  JMML.`Mbmmd'.JMML.   M9mmmP'      YMbmd'  `Moo9^Yo..JMML.   `Mbmmd'     `Moo9^Yo.P^YbmdP'   `Ybmd9'  `Mbod"YML. `Mbmo                                  
#                              6'     dP                                                                                                                                                                                                        
#                              Ybmmmd'                                                                                                                                                                                                          
                                                                                                                                                                                                                                              
#                                                                                                                              ,,                                     ,...                                                                      
#                                                      mm                                                                    `7MM                                   .d' ""                                                                      
#                                                      MM                                                                      MM                                   dM`                                                                         
#  ,p6"bo   ,pW"Wq.`7Mb,od8 `7Mb,od8 .gP"Ya   ,p6"bo mmMMmm `7MMpMMMb.  .gP"Ya  ,pP"Ybd ,pP"Ybd      ,6"Yb.  `7MMpMMMb.   ,M""bMM      `7MMpdMAo.  .gP"Ya `7Mb,od8 mMMmm,pW"Wq.`7Mb,od8 `7MMpMMMb.pMMMb.   ,6"Yb.  `7MMpMMMb.  ,p6"bo   .gP"Ya  
# 6M'  OO  6W'   `Wb MM' "'   MM' "',M'   Yb 6M'  OO   MM     MM    MM ,M'   Yb 8I   `" 8I   `"     8)   MM    MM    MM ,AP    MM        MM   `Wb ,M'   Yb  MM' "'  MM 6W'   `Wb MM' "'   MM    MM    MM  8)   MM    MM    MM 6M'  OO  ,M'   Yb 
# 8M       8M     M8 MM       MM    8M"""""" 8M        MM     MM    MM 8M"""""" `YMMMa. `YMMMa.      ,pm9MM    MM    MM 8MI    MM        MM    M8 8M""""""  MM      MM 8M     M8 MM       MM    MM    MM   ,pm9MM    MM    MM 8M       8M"""""" 
# YM.    , YA.   ,A9 MM       MM    YM.    , YM.    ,  MM     MM    MM YM.    , L.   I8 L.   I8     8M   MM    MM    MM `Mb    MM        MM   ,AP YM.    ,  MM      MM YA.   ,A9 MM       MM    MM    MM  8M   MM    MM    MM YM.    , YM.    , 
#  YMbmd'   `Ybmd9'.JMML.   .JMML.   `Mbmmd'  YMbmd'   `Mbmo.JMML  JMML.`Mbmmd' M9mmmP' M9mmmP'     `Moo9^Yo..JMML  JMML.`Wbmd"MML.      MMbmmd'   `Mbmmd'.JMML.  .JMML.`Ybmd9'.JMML.   .JMML  JMML  JMML.`Moo9^Yo..JMML  JMML.YMbmd'   `Mbmmd' 
#                                                                                                                                        MM                                                                                                     
#                                                                                                                                      .JMML.





# **********************************************************************************************************************
#                                               borscht (Literary Source)
# <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->
#     tensor_managed.py                               3A. From Python to Rust
#                                                         - tensor_nativecpu.py
#                                                           0. Mathematicians vs Programmers
#                                                           1. From Lambda Calculus to Turing Machines <-- HERE
#                                                           2. From Virtual Machines to Physical Machines
#                                                         - pyproject.toml
#                                                         - Cargo.toml
#                                                         - lib.rs
#                                                     3B. From Rust to RISC-V
#                                                         - kernelscpu.rs
# **********************************************************************************************************************






def _main():
    import doctest
    doctest.testmod()

    # After File 2. Hello Tensor (tensor_managed.py),
    # we now understand that a Tensor is an arbitrarily shaped logical array virtualized to physical storage with a stride
    # Recall our previous program which evaluates the matrix multiplicaion of
    A = full((2,3), 3.141592) # some pie ...
    B = full((3,2), 3.141592)
    C = mm_managed(A,B) # .. with borscht!
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