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
# After reading File 1. Hello Borscht __init__.py, you have a 50,000 ft view of borscht's internals.
# After reading File 2. Hello Tensor tensor_managed.py, you now understand the core essence of the Tensor concept and it's primary algorithm of matrix multiplication
                                                          # In fact, we will be reusing such data structure and algorithm:
from borscht.tensor_managed import Tensor                 # Let us import Tensor defined from File 2 to define our accelerated mm routine
                                                          #               aspirationally representing the set R^{d1 \times \cdots, \times dn} known as a field
from borscht.tensor_managed import full, mm as mm_managed # Let us import the full constructor and managed matrix multiply routine also defined in FIle 2
                                                          #               aspirationally representing the function L1: R^k -> R^n and L2: R^m -> R^k known as a composition of linear transformation                                                                                             

# In Files 3 and 4, we will be natively* accelerating the matrix multiplication algorithm
#                   on a RISC-V CPU and NVIDIA (PTX/SASS) GPU using Rust and CUDA Rust respectively
#                              * where natively means the algorithm's program is executed directly by the physical processor

#                   Both Files are split across three groups of files --------------------------------------------------------------+
# **********************************************************************************************************************            |
#                                               borscht (Literary Source)                                                           |
# <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->            v
#             tensor_managed.py                        > 3A. tensor_nativecpu.py <               4A. tensor_nativegpu.py <--------- A. Python Source
#                                                        3B. pyproject.toml/Cargo.toml/lib.rs    4B. Cargo.toml/lib.rs   <--------- B. Python/Rust interop
#                                                        3C. kernelscpu.rs                       4C. kernelsgpu.rs       <--------- C. Rust/CUDA Rust source
#                                                                                                      utils.rs
#                                                                                                      host.rs
#                                                                                                      device.rs
# **********************************************************************************************************************
#                                              > 3A. tensor_nativecpu.py <
#                                                 0. From Mathematics to Programming          <-- HERE
#                                                 1. From Analytical Big-O to Empirical Rooflines

# If a mathematician or physicist didn't know about the internals of computers started using borscht.Tensors,
# they could construct and read Tensors...
# reminder: enter the expressions in a Python interpreter or
#           run the doctest directly with uv run tensor_nativecpu.py
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
#                                               |
#                                               | is transliterated to
#    ______________________________________________________
#   |                                                      |
#   |  R    - 1/2 R g    + /\ g    =  8 pi G  T            |
#   |   uv           uv        uv     ------   uv          |         \ \ | / /
#   |                                  c^4                 |    \ \\ \\\|||/// // /
#   |                                                      |  ~~\\\ .-'''''''-. ///~~
#   |  C   = A   B         <-- repeated k? sum it.         | ~~=   / ~~~   ~~~ \   =~~
#   |   ij    ik  kj           "a great discovery!"        |  ~=  | (o )   (o ) |  =~
#   |                                                   _  |   ~ (|      /      |) ~
#   |  C = einsum("ik,kj->ij", A, B)  # look, no sigma =(_)\      |     /__     |
#   |                                                     \ \     | ,\\\\|////, |
#   |    u     u   b   a                                   \ \     \   '---'   /
#   |  T'  =  L   L   T                                    |\ \     '._______.'
#   |    v     a   v   b                                   | \ \_______|   |_______           +----------------------------+
#   |                                                      |  \         \ /        \          | Tensor                     |
#   |  E = mc^2        (old news)                          |   \         V          \         +----------------------------+
#   |                                                      |----\|      [=]      |   |------->| + shape:   tuple[int, ...] |
#   |______________________________________________________|     |       o       |   |        | + ndim:    int             | We use + to denote user-visible state
#    \__[####]___=__-___o_________________________________/      |       o       |   |        | - stride:  tuple[int, ...] |    and - to denote user-hidden state (abstracted away)
#                                                                |               |   |        | - storage: list[float]     | Hidden from them is stride/storage
#                                                                |_______________|___|        +----------------------------+ Even though they are technically not private Python fields (prefixed with _ in python)
#                                                                  |     |     | (___)
#                                                                  |     |     |
#                                                                  |     |     |
#                                                                 _|     |     |_
#                                                                (_______|_______)

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
# That is,
# mm: R^{m x p} x R^{p x n} -> R^{n x m}
#     (aij      , bij  )   |-> Σ_{k=1}^{p} a_il*b_lj
#                              \_________ _________/
#                                        |
#                                        |
#                                        |                  A_NP, B_PM, C_NM = A, B, C                                                        \
#                                        |                  (ijmp, kjmp), (k2jmp, jjmp), (istridec, jstridec) = A.stride, B.stride, C.stride   |
#                                        |                  A_buf, B_buf, C_buf = A_NP.storage, B_PM.storage, C_NM.storage                      > Hidden from them is
#                                        |                  for i in range(nrows):                                                             |  - the mapping of logical indices to physical ones
#                                        |                      for j in range(mcols):                                                        /   - the procedural control flow to traverse output matrix C
#                                        |                          dot_product = 0.0
#                                        | is transliterated to     for k in range(p):
#                                        +----------------------------> dot_product += A_buf[i*ijmp + k*kjmp] * B_buf[k*k2jmp + j*jjmp]
#                                                                   C_buf[i*istridec + j*jstridec] = dot_product


#   While the mathematicians have to come to terms with their ideals of  real numbers and real analysis not actually being so *real*
#   and join the physicists and computer scientists in reality with floating point numerics, and numerical analysis (which we'll learn more about soon),
#   the current Tensor data structure and matrix multiplication algorithm is satisfactory — They don't even need to worry about any striding nonsense!
#                                                                                           Striding memory is so below the working mathematician!

# While mathematics and computation have a close relationship where they concern themselves with
# the same objects or the same mathema (μάθημα), which in greek literally means “that which is learnt”, “what one gets to know” and “these things.”
# In the case of the Tensor data structure and it's matrix multiplication algorithm, mathematicians prioritize the *definitions* and the *proofs* of their properties.
# They are not so much interested in their *evaluation* (except for perhaps, numerical analysts. Numerical analysts are more like our first cousins, while mathematicians second cousins),
# **This is where programmers differ from our mathematical cousins, because we care about the evaluation of these objects.**
# That is, in addition to caring about *correctness*  i.e did I get the right answer?
#                   we also care about *performance*. i.e did I get the right answer cheaply*?

#                   * we use cheaply here because performance is about the *cost* of inputs, to receive an output
#                     where the cost can be running time, memory space, energy, bandwidth.
#                     that is, speed is just one way to measure cost.
#                     — in this way, a programmer in the valley is similar to the investor on wall street.
#                       at what price is the output *worth* it?
#
#                                                               /\  /\
#                                                          ____/  \/  \_
#                                                      _.-'     .-.     `.
#                                                    .'  ~~     \@/       \       /\  /\  /\  /\  /\
#                                                   (__..                  \   /\/  \/  \/  \/  \/  \___
#                                         ,~~~,        \VVVVVV.             \_/                         `.
#                                      ,~~~~~~~\               )               __.-----------._           \
#                                     (~~~~~~~~~)      /^^^^^^'               (_/ / / / / / / /            |
#                                      `~~~' _|_      (_________________.        `'-'-'-'-'-'-'            |
#                                         .-'   `-.                      \                                /
#                                        /         \                      `-._/  /_____________________.-'\  \
#                                       |     .-----|               .---------|  |----------------------.  \  \
#                                       |     |=====|>              | .-------(/(/--------------------. |   )  )
#                                       |     '-----|               | | # C = A @ B                   | |  /  /
#                                        \         /                | | for i in range(n):            | | (  (
#                                         `-.___.-'                 | |   for j in range(m):          | |  \  \
#                                         __|___|__                 | |     for k in range(p):        | |   )  )
#                                       /'  .---.  `\               | |       C[i,j] += A[i,k]*B[k,j] | |  /  /
#                                      /   (     )   \              | '-------------------------------' | (  (
#                                     |     \   /     )             |  ::::::                   (o) []  |  \  `.
#                                     |      | |     /              '-----------------------------------'   )   `-._
#                                     |      | |====/=======._______               )      (                /        `.
#                                     |      |_|___________(##):::::\         ____/________\____          (____       \
#                                      \___________/     ======================================================`-.    |
#                                     [===]_____________        ||                                _______  ||     \  /
#                                      | |`----------. |        ||                               |  { }  | ||     /\/\
#                                      | |           | |        ||                               |       | ||     \  /
#                                      | |           | |__      ||                                \     /  ||      \/
#                                     _|_|_          |____)    _||_    o==[]::::::::::::::>        `---'  _||_                                                             ,,                                                                    
#    `7MM"""Mq.                                                                                                                                                             *MM                             mm                                     
#      MM   `MM.                                         ~ slaying the dragon, one token at a time ~                                                                         MM                             MM                                     
#      MM   ,M9 `7Mb,od8 ,pW"Wq.   .P"Ybmmm `7Mb,od8 ,6"Yb.  `7MMpMMMb.pMMMb.  `7MMpMMMb.pMMMb.  .gP"Ya `7Mb,od8 ,pP"Ybd      ,p6"bo   ,6"Yb.  `7Mb,od8 .gP"Ya       ,6"Yb.  MM,dMMb.   ,pW"Wq.`7MM  `7MM mmMMmm                                   
#      MMmmdM9    MM' "'6W'   `Wb :MI  I8     MM' "'8)   MM    MM    MM    MM    MM    MM    MM ,M'   Yb  MM' "' 8I   `"     6M'  OO  8)   MM    MM' "',M'   Yb     8)   MM  MM    `Mb 6W'   `Wb MM    MM   MM                                     
#      MM         MM    8M     M8  WmmmP"     MM     ,pm9MM    MM    MM    MM    MM    MM    MM 8M""""""  MM     `YMMMa.     8M        ,pm9MM    MM    8M""""""      ,pm9MM  MM     M8 8M     M8 MM    MM   MM                                     
#      MM         MM    YA.   ,A9 8M          MM    8M   MM    MM    MM    MM    MM    MM    MM YM.    ,  MM     L.   I8     YM.    , 8M   MM    MM    YM.    ,     8M   MM  MM.   ,M9 YA.   ,A9 MM    MM   MM                                     
#    .JMML.     .JMML.   `Ybmd9'   YMMMMMb  .JMML.  `Moo9^Yo..JMML  JMML  JMML..JMML  JMML  JMML.`Mbmmd'.JMML.   M9mmmP'      YMbmd'  `Moo9^Yo..JMML.   `Mbmmd'     `Moo9^Yo.P^YbmdP'   `Ybmd9'  `Mbod"YML. `Mbmo                                  
#                                 6'     dP                                                                                                                                                                                                        
#                                 Ybmmmd'                                                                                                                                                                                                          
#                                                                                                                                                               
#                                                                                                                                 ,,                                     ,...                                                                      
#                                                         mm                                                                    `7MM                                   .d' ""                                                                      
#                                                         MM                                                                      MM                                   dM`                                                                         
#     ,p6"bo   ,pW"Wq.`7Mb,od8 `7Mb,od8 .gP"Ya   ,p6"bo mmMMmm `7MMpMMMb.  .gP"Ya  ,pP"Ybd ,pP"Ybd      ,6"Yb.  `7MMpMMMb.   ,M""bMM      `7MMpdMAo.  .gP"Ya `7Mb,od8 mMMmm,pW"Wq.`7Mb,od8 `7MMpMMMb.pMMMb.   ,6"Yb.  `7MMpMMMb.  ,p6"bo   .gP"Ya  
#    6M'  OO  6W'   `Wb MM' "'   MM' "',M'   Yb 6M'  OO   MM     MM    MM ,M'   Yb 8I   `" 8I   `"     8)   MM    MM    MM ,AP    MM        MM   `Wb ,M'   Yb  MM' "'  MM 6W'   `Wb MM' "'   MM    MM    MM  8)   MM    MM    MM 6M'  OO  ,M'   Yb 
#    8M       8M     M8 MM       MM    8M"""""" 8M        MM     MM    MM 8M"""""" `YMMMa. `YMMMa.      ,pm9MM    MM    MM 8MI    MM        MM    M8 8M""""""  MM      MM 8M     M8 MM       MM    MM    MM   ,pm9MM    MM    MM 8M       8M"""""" 
#    YM.    , YA.   ,A9 MM       MM    YM.    , YM.    ,  MM     MM    MM YM.    , L.   I8 L.   I8     8M   MM    MM    MM `Mb    MM        MM   ,AP YM.    ,  MM      MM YA.   ,A9 MM       MM    MM    MM  8M   MM    MM    MM YM.    , YM.    , 
#     YMbmd'   `Ybmd9'.JMML.   .JMML.   `Mbmmd'  YMbmd'   `Mbmo.JMML  JMML.`Mbmmd' M9mmmP' M9mmmP'     `Moo9^Yo..JMML  JMML.`Wbmd"MML.      MMbmmd'   `Mbmmd'.JMML.  .JMML.`Ybmd9'.JMML.   .JMML  JMML  JMML.`Moo9^Yo..JMML  JMML.YMbmd'   `Mbmmd' 
#                                                                                                                                           MM                                                                                                     
#                                                                                                                                         .JMML.

# **********************************************************************************************************************
#                                               borscht (Literary Source)
# <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 4. Accelerate++ Tensor -->
#             tensor_managed.py                        > 3A. tensor_nativecpu.py <               4A. tensor_nativegpu.py
#                                                        3B. pyproject.toml/Cargo.toml/lib.rs    4B. Cargo.toml/lib.rs
#                                                        3C. kernelscpu.rs                       4C. kernelsgpu.rs
#                                                                                                      utils.rs
#                                                                                                      host.rs
#                                                                                                      device.rs
# **********************************************************************************************************************
#                                              > 3A. tensor_nativecpu.py <
#                                                 0. From Mathematics to Programming
#                                                 1. From Virtual Big-O to Physical Rooflines <-- HERE
# Since we programmers care about performance, and in this case, speed, we need a way of modeling the cost of a program with respect to time.
# In the natural sciences, in order to model a physical process, experimental data must be collected and the underlying analytical formula (if there is one) must be inferred.
# — In fact, that is the essence of machine learning and deep learning: they mechanize induction.
# In contrast, with the artifical sciences like computation, we have *access* to the source code of our computational processes.
# We can model the cost of our program (in this case the cost we care about is *time*)
# by assigning an *idealized* cost to each operation, and simply adding up the costs of all operations in our program.
    # we need to use a *cost model*.
    # The first cost model we use 

    # Is this fast?
    # According to Big-O, the time complexity is n -> n^3.
    # The transition from mathematics to programming is the transition from lambda calculus' substitution model
    # of computation to turing machines von neumann model of computation

def _main():
    # Recall the running example from File 2. Hello Tensor (tensor_managed.py)..
    A = full((2,3), 3.141592)
    B = full((3,2), 3.141592)
    C = mm_managed(A,B)
    print(C)

    n = 256
    A, B = full((n, n), 1.0), full((n, n), 1.0)

    # Continuing down the route of classical algorithm analysis,
    # we would be led to an algorithm known as Strassen's Algorithm, with time complexity n -> n^2.7.,. lg()
    # Instead of analytically analyzing the software algorithm's asymptotics to predict worst-case time *complexity*,
    #     we can empirically        use the hardware performance counters    to measure actual time

    time_virtually_managed_mm(A,B,n)
    time_physically_native_mm(A,B,n)

import time # We want empirically measure actual wall-clock time, *not* analytically analyzing an upper bound
def time_virtually_managed_mm(A: Tensor, B: Tensor, n: int):
    # matmul implemented by a managed language running on a virtual machine
    t0 = time.perf_counter()
    mm_managed(A, B)
    t = time.perf_counter() - t0
    print(f"{n}x{n} mm_managed: {t*1e3:.1f} ms, {2*n**3 / t / 1e9:.4f} GFLOP/s")

def time_physically_native_mm(A: Tensor, B: Tensor, n: int):
    # matmul implemented by a native language running on a physical machine
    t0 = time.perf_counter()
    mm_native(A, B)
    t = time.perf_counter() - t0
    print(f"{n}x{n} mm_native: {t*1e3:.1f} ms, {2*n**3 / t / 1e9:.4f} GFLOP/s")





# Here, we re-define the matrix multiplication algorithm from File 2. Hello Tensor (tensor_managed.py)
#       with a native programming language so that the program can be executed directly on a physical processor
# The only two lines that are noteworthy of your attention is L285 and L294-----------------------------------------------------

def mm(A: Tensor, B: Tensor) -> Tensor:
    # mm has the same definition from File 2..
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
    C = mm_native(A,B) # <--- mm is just a facade (i.e a wrapper) for mm_native
    return C

from borscht._borscht import _gemm   # The _gemm(a,b,n,k,m) routine is implemented via Rust
                                     # which we will implement for ourselves in File 3C
                                     
def mm_native(A: Tensor, B: Tensor) -> Tensor:
    (nrows, p), (p2, mcols) = A.shape, B.shape          # realias..
    C = full((nrows, mcols), 0.0)                 # construct output
    C.storage = _gemm(A.storage, B.storage, nrows, p, mcols) # <-- and in turn, mm_native is also a facade for _gemm
    return C

if __name__ == "__main__": _main()
__all__ = ["Tensor", "full", "mm"] # Let us export the full() constructor and Tensor concept as-is from File 2. Hello Tensor
                                              # as well as our natively accelerated mm(), a facade for mm_native() which in turn, is a facade for _gemm()