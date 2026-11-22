#                                                          ,,                                                                                                                                                        
#                       db                               `7MM                          mm               MMP""MM""YMM                                                                                                 
#                      ;MM:                                MM                          MM               P'   MM   `7                                                                                                 
#  pd""b.             ,V^MM.    ,p6"bo   ,p6"bo   .gP"Ya   MM  .gP"Ya `7Mb,od8 ,6"Yb.mmMMmm .gP"Ya           MM  .gP"Ya `7MMpMMMb.  ,pP"Ybd  ,pW"Wq.`7Mb,od8                                                         
# (O)  `8b           ,M  `MM   6M'  OO  6M'  OO  ,M'   Yb  MM ,M'   Yb  MM' "'8)   MM  MM  ,M'   Yb          MM ,M'   Yb  MM    MM  8I   `" 6W'   `Wb MM' "'                                                         
#      ,89           AbmmmqMA  8M       8M       8M""""""  MM 8M""""""  MM     ,pm9MM  MM  8M""""""          MM 8M""""""  MM    MM  `YMMMa. 8M     M8 MM                                                             
#    ""Yb. ,,       A'     VML YM.    , YM.    , YM.    ,  MM YM.    ,  MM    8M   MM  MM  YM.    ,          MM YM.    ,  MM    MM  L.   I8 YA.   ,A9 MM                                                             
#       88 db     .AMA.   .AMMA.YMbmd'   YMbmd'   `Mbmmd'.JMML.`Mbmmd'.JMML.  `Moo9^Yo.`Mbmo`Mbmmd'        .JMML.`Mbmmd'.JMML  JMML.M9mmmP'  `Ybmd9'.JMML.                                                           
# (O)  .M' After reading File 1. Hello Borscht (__init__.py), you have a 50,000 ft view of borscht's internals.
#  bmmmd'  After reading File 2. Hello Tensor (tensor_managed.py), you understand the core essence of the Tensor concept and it's primary algorithm of matrix multiplication
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
# <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 5. Accelerate++ Tensor -->            v
#             tensor_managed.py                        > 3A. tensor_nativecpu.py <               4A. tensor_nativegpu.py <--------- A. Python source
#                                                        3B. pyproject.toml/Cargo.toml           4B. Cargo.toml/lib.rs   <--------- B. Python/Rust interop
#                                                        3C. lib.rs/kernelscpu.rs                4C. kernelsgpu.rs       <--------- C. Rust/CUDA Rust source
#                                                                                                      utils.rs
#                                                                                                      host.rs
#                                                                                                      device.rs
#
#                                                File 4. Accelerate+ Tensor

# **********************************************************************************************************************
#                                              > 3A. tensor_nativecpu.py <
#                                                 0. From Stateless Substitution to Stateful Environments
#                                                 1. Analysis 1: From Asymptotic Analysis of Growth to Empirical Measurement of Time <-- HERE
#                                                 2. Three Language Problem: From Virtual Machines to Physical Machines

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


# While the mathematicians have to come to terms with their ideals of real numbers and real analysis not actually being so *real*
# and join the physicists, computer scientists, and numerical analysts in reality with floating point numerics
# the current Tensor data structure and matrix multiplication algorithm is satisfactory — They don't even need to worry about any striding nonsense!
#                                                                                           Striding memory is so beneath the working mathematician!
    # When we first learned the art of designing programs, we learned that the rules how procedure application became evaluated were similar to that of middle school algebra.
    # That is, *substitution*.
    # This substitution model of computation is similar to the one that mathematicians and physicists have when an interpreter evaluates their program.
    # see Structure and Interpretation of Computer Programs Chapter 1.1.5 The Substitution Model for Procedure Application
    #     https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/full-text/book/book-Z-H-10.html#%_sec_1.1.5

    # As programmers, we know there is stateful memory. And so once our programs started using mutation,
    # we refined our model of evaluation to that of the *environment* model where there was an environment of bindings which associated aliases to values.
    # see Structure and Interpretation of Computer Programs Chapter 3.2 The Environment Model of Evaluation
    #     https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/full-text/book/book-Z-H-21.html#%_sec_3.2

    # The two models are similar to the two models of computation in theoretical computer science of Lambda Calculus and Turing Machines. Go take a theory of computation course!

# While mathematics and computation have a close relationship where they concern themselves with
# the same objects or the same mathema (μάθημα), which in greek literally means “that which is learnt”, “what one gets to know” and “these things.”
# In the case of the Tensor data structure and it's matrix multiplication algorithm, mathematicians prioritize the *definitions* and the *proofs* of their properties.


    # They are not so much interested in their *evaluation*.

    # **This is where programmers differ from our mathematical cousins, because we care about the evaluation of these objects.**
    #   meaning, in addition to caring about *correctness*  i.e did I get the right answer?
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
#                                                                                                     q   p                            ,,   q   p                                      ,...                                                                      
#                                                      mm                                              \ /                           `7MM    \ /                                     .d' ""                                                                      
#                                                      MM                                           o=--*--=o                          MM o=--*--=o                                  dM`                                                                         
#  ,p6"bo   ,pW"Wq.`7Mb,od8 `7Mb,od8 .gP"Ya   ,p6"bo mmMMmm `7MMpMMMb.  .gP"Ya  ,pP"Ybd ,pP"Ybd        / \   ,6"Yb.  `7MMpMMMb.   ,M""bMM    / \        `7MMpdMAo.  .gP"Ya `7Mb,od8 mMMmm,pW"Wq.`7Mb,od8 `7MMpMMMb.pMMMb.   ,6"Yb.  `7MMpMMMb.  ,p6"bo   .gP"Ya  
# 6M'  OO  6W'   `Wb MM' "'   MM' "',M'   Yb 6M'  OO   MM     MM    MM ,M'   Yb 8I   `" 8I   `"       d   b 8)   MM    MM    MM ,AP    MM   d   b         MM   `Wb ,M'   Yb  MM' "'  MM 6W'   `Wb MM' "'   MM    MM    MM  8)   MM    MM    MM 6M'  OO  ,M'   Yb 
# 8M       8M     M8 MM       MM    8M"""""" 8M        MM     MM    MM 8M"""""" `YMMMa. `YMMMa.              ,pm9MM    MM    MM 8MI    MM                 MM    M8 8M""""""  MM      MM 8M     M8 MM       MM    MM    MM   ,pm9MM    MM    MM 8M       8M"""""" 
# YM.    , YA.   ,A9 MM       MM    YM.    , YM.    ,  MM     MM    MM YM.    , L.   I8 L.   I8             8M   MM    MM    MM `Mb    MM                 MM   ,AP YM.    ,  MM      MM YA.   ,A9 MM       MM    MM    MM  8M   MM    MM    MM YM.    , YM.    , 
#  YMbmd'   `Ybmd9'.JMML.   .JMML.   `Mbmmd'  YMbmd'   `Mbmo.JMML  JMML.`Mbmmd' M9mmmP' M9mmmP'             `Moo9^Yo..JMML  JMML.`Wbmd"MML.               MMbmmd'   `Mbmmd'.JMML.  .JMML.`Ybmd9'.JMML.   .JMML  JMML  JMML.`Moo9^Yo..JMML  JMML.YMbmd'   `Mbmmd' 
#                                                                                                                                                         MM                                                                                                     
#                                                                                                                                                       .JMML.                                                                                                   





# **********************************************************************************************************************
#                                               borscht (Literary Source)
# <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 5. Accelerate++ Tensor -->
#             tensor_managed.py                        > 3A. tensor_nativecpu.py <               4A. tensor_nativegpu.py
#                                                        3B. pyproject.toml/Cargo.toml           4B. Cargo.toml/lib.rs
#                                                        3C. lib.rs/kernelscpu.rs                4C. kernelsgpu.rs
#                                                                                                      utils.rs
#                                                                                                      host.rs
#                                                                                                      device.rs
#
#                                                File 4. Accelerate+ Tensor

# **********************************************************************************************************************
#                                              > 3A. tensor_nativecpu.py <
#                                                 0. From Stateless Substitution to Stateful Environments
#                                                 1. From Analytically Analyzing Growth to Empirically Measuring Time <-- HERE
#                                                 2. From Virtual RAM to Physical Machine Memory Hierarchies


def _main(): # run the file as a script (i.e uv run tensor_nativecpu.py) on your machine
    n = 256
    A, B = full((n, n), 1.0), full((n, n), 1.0)
    C = mm(A, B)
    # Since we programmers care about performance, and in this case, speed, how much time did our matrix multiplication algorithm take?
    # One way is to measure 




    # Since we programmers care about performance, and in this case, speed, we need a way of modeling the cost of a program with respect to time.
    # In the natural sciences, in order to model a physical process, experimental data must be collected and the underlying analytical formula (if there is one) must be inferred.
    # — In fact, that is the essence of machine learning and deep learning: they mechanize induction.
    # In contrast, with the artifical sciences like computer science, we have *access* to the source code of our computational processes.

    # Given that processes in the artificial sciences are not black box like the ones of the natural sciences
    # (we have access to the source!) we can *model* and *predict* the cost of our program (in this case the cost we care about is *time*)
    # Consider the code of the inlined def mm(A: Tensor, B: Tensor) -> Tensor:
    (nrows, p), (p2, mcols) = A.shape, B.shape
    if len(A.shape) != 2 or len(B.shape) != 2: raise RuntimeError("self must be a matrix" if len(A.shape) != 2 else "B must be a matrix")
    if p != p2: raise RuntimeError(f"A and B shapes cannot be multiplied ({nrows}x{p} and {p2}x{mcols})")
    C = full((nrows, mcols), 0.0) # empty C
    A_NP, B_PM, C_NM = A, B, C # alias input's A, B and output C with hungarian notation shape suffixes
    (ijmp, kjmp), (k2jmp, jjmp), (istridec, jstridec) = A.stride, B.stride, C.stride
    A_buf, B_buf, C_buf = A_NP.storage, B_PM.storage, C_NM.storage # alias their storage buffers
    for i in range(nrows): # for each of A's n rows..
        for j in range(mcols): # ..loop through of B's m cols..
            dot_product = 0.0
            for k in range(p):   # ..take their k-dimensional dot product A[(i,k)] + B[(k,j)]..
                dot_product += A_buf[i*ijmp + k*kjmp] * B_buf[k*k2jmp + j*jjmp] # ..we map logical indices (i,k) and (k,j) to physical ones with strides
            C_buf[i*istridec + j*jstridec] = dot_product # update C[(i,j)] (strided)..by flushing dot_product

            # repeat for the rest of B's columns
        # repeat for the rest of A's rows

    # TODO 1: (Mon. Nov 23): BIG-O
    # Before we can analyze the algorithm, we need a model of the underlying machine that will execute the program.
    # Most algorithm analysis follows a model similar to the environment model of computation known as the RAM model.
    # 

    # Is this fast?
    # According to Big-O, the time complexity is n -> n^3.
    
    # TODO 2: (Mon. Nov 23): Divide and Conquer (CLRS) Strassen
    # Continuing down the route of classical algorithm analysis,
    # we would be led to an algorithm known as Strassen's Algorithm, with time complexity n -> n^2.7.,. lg()





    # Instead of analytically analyzing the software algorithm's asymptotics to predict worst-case time *complexity*,
    #     we can empirically        use the hardware performance counters    to measure actual time
    #                        T(n)
    #                         ^               /                             .-'''''-.
    #                         |  O(n^3)      /       / O(n^2.81)          .'    12   '.
    #                         |             /      .'                    /      |      \ python: 400ms
    #                         |           .'    .-'      ---------->    | 9     o--   3 |
    #                         |        .-'   .-'                         \             / rust: 40ms
    #                         |___..--'__..-'                             '.    6    .'
    #                         +------------------------> n                  '-.....-'
    #                                                                                     ,,                     ,,                     ,,                                           ,,                      ,,                    
    # `7MM"""YMM                                           db                           `7MM              mm     db                   `7MM            db                           `7MM                      db                    
    #   MM    `7                                          ;MM:                            MM              MM                            MM           ;MM:                            MM                                            
    #   MM   d `7Mb,od8 ,pW"Wq.`7MMpMMMb.pMMMb.          ,V^MM.    `7MMpMMMb.   ,6"Yb.    MM `7M'   `MF'mmMMmm `7MM  ,p6"bo   ,6"Yb.    MM          ,V^MM.    `7MMpMMMb.   ,6"Yb.    MM `7M'   `MF',pP"Ybd `7MM  ,pP"Ybd           
    #   MM""MM   MM' "'6W'   `Wb MM    MM    MM         ,M  `MM      MM    MM  8)   MM    MM   VA   ,V    MM     MM 6M'  OO  8)   MM    MM         ,M  `MM      MM    MM  8)   MM    MM   VA   ,V  8I   `"   MM  8I   `"           
    #   MM   Y   MM    8M     M8 MM    MM    MM         AbmmmqMA     MM    MM   ,pm9MM    MM    VA ,V     MM     MM 8M        ,pm9MM    MM         AbmmmqMA     MM    MM   ,pm9MM    MM    VA ,V   `YMMMa.   MM  `YMMMa.           
    #   MM       MM    YA.   ,A9 MM    MM    MM        A'     VML    MM    MM  8M   MM    MM     VVV      MM     MM YM.    , 8M   MM    MM        A'     VML    MM    MM  8M   MM    MM     VVV    L.   I8   MM  L.   I8           
    # .JMML.   .JMML.   `Ybmd9'.JMML  JMML  JMML.    .AMA.   .AMMA..JMML  JMML.`Moo9^Yo..JMML.   ,V       `Mbmo.JMML.YMbmd'  `Moo9^Yo..JMML.    .AMA.   .AMMA..JMML  JMML.`Moo9^Yo..JMML.   ,V     M9mmmP' .JMML.M9mmmP'           
    #                                                                                           ,V                                                                                         ,V                                      
    #                                                                                        OOb"                                                                                       OOb"                                       
                                                                                                                                                                                                                                
    #                                                                ,,             ,,                     ,,                                                                                                                      
    # MMP""MM""YMM          `7MM"""YMM                               db             db                   `7MM      `7MMM.     ,MMF'                                                                                          mm    
    # P'   MM   `7            MM    `7                                                                     MM        MMMb    dPMM                                                                                            MM    
    #      MM  ,pW"Wq.        MM   d    `7MMpMMMb.pMMMb. `7MMpdMAo.`7MM  `7Mb,od8 `7MM  ,p6"bo   ,6"Yb.    MM        M YM   ,M MM  .gP"Ya   ,6"Yb.  ,pP"Ybd `7MM  `7MM  `7Mb,od8 .gP"Ya `7MMpMMMb.pMMMb.  .gP"Ya `7MMpMMMb.mmMMmm  
    #      MM 6W'   `Wb       MMmmMM      MM    MM    MM   MM   `Wb  MM    MM' "'   MM 6M'  OO  8)   MM    MM        M  Mb  M' MM ,M'   Yb 8)   MM  8I   `"   MM    MM    MM' "',M'   Yb  MM    MM    MM ,M'   Yb  MM    MM  MM    
    #      MM 8M     M8       MM   Y  ,   MM    MM    MM   MM    M8  MM    MM       MM 8M        ,pm9MM    MM        M  YM.P'  MM 8M""""""  ,pm9MM  `YMMMa.   MM    MM    MM    8M""""""  MM    MM    MM 8M""""""  MM    MM  MM    
    #      MM YA.   ,A9       MM     ,M   MM    MM    MM   MM   ,AP  MM    MM       MM YM.    , 8M   MM    MM        M  `YM'   MM YM.    , 8M   MM  L.   I8   MM    MM    MM    YM.    ,  MM    MM    MM YM.    ,  MM    MM  MM    
    #    .JMML.`Ybmd9'      .JMMmmmmMMM .JMML  JMML  JMML. MMbmmd' .JMML..JMML.   .JMML.YMbmd'  `Moo9^Yo..JMML.    .JML. `'  .JMML.`Mbmmd' `Moo9^Yo.M9mmmP'   `Mbod"YML..JMML.   `Mbmmd'.JMML  JMML  JMML.`Mbmmd'.JMML  JMML.`Mbmo 
    #                                                      MM                                                                                                                                                                      
    #                                                    .JMML.                                                                                                                                                                         
    import time # We want to empirically measure actual wall-clock time, rather than analytically analyzing an upper bound
    def time_virtual_mm(A: Tensor, B: Tensor, n: int):
        t0 = time.perf_counter()
        _ = mm_managed(A, B)  # ignore the output. we care about performance right now, not correctness
        t = time.perf_counter() - t0 # mm_managed is the matrix multiplication routine implemented from File 2. tensor_managed.py
        return t

    t_virtual = time_virtual_mm(A,B,n) # baseline
    print(f"{n}x{n} mm_managed: {t_virtual*1e3:.1f} ms")

    # Moreover, the first optimization we make to the standard matrix multiplication triple loop algorithm is
    # to simply change the implementation language from Python to Rust so that the code gets executed *directly* by a physical processor
    # That is, to follow the advice of the crowd and "rewrite it in Rust".
    def time_native_mm(A: Tensor, B: Tensor, n: int):
        t0 = time.perf_counter()
        _ = mm_native(A, B) # mm_native is implemented with a native language
        t = time.perf_counter() - t0
        return t
        
    # While the *absolute* speed will vary depending on your hardware,
    t_physical = time_native_mm(A,B,n) # What is the *relative speedup* of the Rust implementation with respect to the Python one?
    print(f"{n}x{n} mm_managed: {t_physical*1e3:.1f} ms, speedup: {t_virtual / t_physical:.2f}x")
                                                                  # 2x, 5x, 10x, 20x, 50x?
    # What explains this?





# **********************************************************************************************************************
#                                               borscht (Literary Source)
# <-- File 2. Hello Tensor                    -- File 3. Accelerate Tensor --            File 5. Accelerate++ Tensor -->
#             tensor_managed.py                        > 3A. tensor_nativecpu.py <               4A. tensor_nativegpu.py
#                                                        3B. pyproject.toml/Cargo.toml           4B. Cargo.toml/lib.rs
#                                                        3C. lib.rs/kernelscpu.rs                4C. kernelsgpu.rs
#                                                                                                      utils.rs
#                                                                                                      host.rs
#                                                                                                      device.rs
#
#                                                File 4. Accelerate+ Tensor

# **********************************************************************************************************************
#                                              > 3A. tensor_nativecpu.py <
#                                                 0. From Stateless Substitution to Stateful Environments
#                                                 1. From Analytically Analyzing Growth to Empirically Measuring Time <-- HERE
#                                                 2. From Virtual RAM to Physical Machine Memory Hierarchies

# As already hinted by the different names of mm_managed and mm_native, besides Python and Rust offering different language *features* to programmers
# (for instance, they provide different features with respect to mutation and asynchrony), they are also differentiated by their *evaluation*,
# meaning, how do the expressions and statements in source programs written in a .py file and a .rs get run?

# The physical processors that are implemented with silicon and electricity only understand a more primitive machine language referred to as *instruction sets*
# — the primary ones being x86, ARM, RISC-V — so how is this gap of semantics (semantic gap) bridged?
# It can be bridged by *interpretation* or *compilation*.
    # We will explain the difference from the programmers perspective, but in order to understand from the implementor's perspective,
    # see Structure and Interpretation of Computer Programs Chapter 4. Metalinguistic Abstraction https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/full-text/book/book-Z-H-25.html#%_chap_4
    #                                                   and Chapter 5. Computing with Register Machines https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/full-text/book/book-Z-H-30.html#%_chap_5
    # If you prefer to implement a more Pythonic language rather a Lispy one,
    # see Programming and Programming Languages Chapter 25. Processing Programs: A First Look at Interpretation https://papl.cs.brown.edu/2020/Processing_Programs__Parsing.html
    # We can't explain everything in this textbook! We are implementing an interpreter for Tensor Programs, not Computer Programs!

# An interpreter *simulates* the source program with a virtual machine implemented with software (another programming language)
#    i.e The primary Python interpreter CPython is implemented with C/C++.
# A compiler *translates* the source program to a target program.
#   i.e The primary Rust compiler rustc translates Rust programs to x86/ARM/RISC-V programs
#                                             The target program is then *executed* with a physical machine implemented with hardware (silicon)

#                                                                                                                                In both scenarios, there is of course underlying hardware running.
#          (1) Interpretation                         (2) Compilation                                                            The question is *which* evaluator your program is evaluated by: a *virtual machine* or a *physical machine*?
#                                                                                                                                    If your program is being interpreted, it's *indirectly* running on hardware, because there is a *layer* of indirection (i.e the interpeter).
#          source program   input                     source program                            input                                If your program is first compiled, then executed, it's *directly* running on hardware.
#                 |           |                              |                                    |
#                 v           v                              v                                    v                                  This is *why* mm_native() has a speedup of 2x, 5x, 10x, 20x, depending on the underlying machine.
#             +------------------+                       +------------------+            +------------------+                                      mm_native() runs directly on the hardware.
#             |   interpreter    |                       |     compiler     |   emits    |  target program  |
#             |    (software)    | ---> output           |    (software)    | ---------> |  (machine code)  | ---> output        We now understand that the semantic gap is bridged either with an intepreter *or compiler,
#             +------------------+                       +------------------+            +------------------+                               and in turn whether the program is evaluated by a virtual machine *or physical machine.
#             |     hardware     |                       |     hardware     |            |     hardware     |                    and that mm_native() has a speedup over mm_managed() because it's written in Rust.
#             +------------------+                       +------------------+            +------------------+
#             <---- run time ---->                       <-- compile time -->            <---- run time ---->                    * Interpretation vs compilation and virtual vs physical machine is a false dichotomy and actually a spectrum
#                                                                                                                                  for there is JIT compilation which starts off by interpreting code and then switched to compilation for performance-sensitive spots.
#                                                                                                                                  This is what the Java Virtual Machine (JVM) does.
#                                                                                                    ,,                           ,,                         ,,                                ,,                  ,,                        
#            `7MMF'            mm                                               mm            mm     db                           db               .M"""bgd  db                              `7MM           mm     db                        
#              MM              MM                                               MM            MM                                                  ,MI    "Y                                    MM           MM                               
#              MM  `7MMpMMMb.mmMMmm .gP"Ya `7Mb,od8 `7MMpdMAo.`7Mb,od8 .gP"Ya mmMMmm  ,6"Yb.mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.      `7MM  ,pP"Ybd     `MMb.    `7MM  `7MMpMMMb.pMMMb.`7MM  `7MM    MM   ,6"Yb.mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.     
#              MM    MM    MM  MM  ,M'   Yb  MM' "'   MM   `Wb  MM' "',M'   Yb  MM   8)   MM  MM     MM 6W'   `Wb MM    MM        MM  8I   `"       `YMMNq.  MM    MM    MM    MM  MM    MM    MM  8)   MM  MM     MM 6W'   `Wb MM    MM     
#              MM    MM    MM  MM  8M""""""  MM       MM    M8  MM    8M""""""  MM    ,pm9MM  MM     MM 8M     M8 MM    MM        MM  `YMMMa.     .     `MM  MM    MM    MM    MM  MM    MM    MM   ,pm9MM  MM     MM 8M     M8 MM    MM     
#              MM    MM    MM  MM  YM.    ,  MM       MM   ,AP  MM    YM.    ,  MM   8M   MM  MM     MM YA.   ,A9 MM    MM        MM  L.   I8     Mb     dM  MM    MM    MM    MM  MM    MM    MM  8M   MM  MM     MM YA.   ,A9 MM    MM  ,, 
#            .JMML..JMML  JMML.`Mbmo`Mbmmd'.JMML.     MMbmmd' .JMML.   `Mbmmd'  `Mbmo`Moo9^Yo.`Mbmo.JMML.`Ybmd9'.JMML  JMML.    .JMML.M9mmmP'     P"Ybmmd" .JMML..JMML  JMML  JMML.`Mbod"YML..JMML.`Moo9^Yo.`Mbmo.JMML.`Ybmd9'.JMML  JMML.db 
#                                                     MM                                                                                                                                                                                     
#                                                   .JMML.                                                                                                                                                                                   
                                                                                                                                                                                                                                
#                                                            ,,    ,,                  ,,                           ,,                                                             ,,                  ,,                                    
#              .g8"""bgd                                     db  `7MM           mm     db                           db              MMP""MM""YMM                                 `7MM           mm     db                                    
#            .dP'     `M                                           MM           MM                                                  P'   MM   `7                                   MM           MM                                           
#            dM'       ` ,pW"Wq.`7MMpMMMb.pMMMb. `7MMpdMAo.`7MM    MM   ,6"Yb.mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.      `7MM  ,pP"Ybd          MM  `7Mb,od8 ,6"Yb.  `7MMpMMMb.  ,pP"Ybd  MM   ,6"Yb.mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.                 
#            MM         6W'   `Wb MM    MM    MM   MM   `Wb  MM    MM  8)   MM  MM     MM 6W'   `Wb MM    MM        MM  8I   `"          MM    MM' "'8)   MM    MM    MM  8I   `"  MM  8)   MM  MM     MM 6W'   `Wb MM    MM                 
#            MM.        8M     M8 MM    MM    MM   MM    M8  MM    MM   ,pm9MM  MM     MM 8M     M8 MM    MM        MM  `YMMMa.          MM    MM     ,pm9MM    MM    MM  `YMMMa.  MM   ,pm9MM  MM     MM 8M     M8 MM    MM                 
#            `Mb.     ,'YA.   ,A9 MM    MM    MM   MM   ,AP  MM    MM  8M   MM  MM     MM YA.   ,A9 MM    MM        MM  L.   I8          MM    MM    8M   MM    MM    MM  L.   I8  MM  8M   MM  MM     MM YA.   ,A9 MM    MM  ,,             
#              `"bmmmd'  `Ybmd9'.JMML  JMML  JMML. MMbmmd' .JMML..JMML.`Moo9^Yo.`Mbmo.JMML.`Ybmd9'.JMML  JMML.    .JMML.M9mmmP'        .JMML..JMML.  `Moo9^Yo..JMML  JMML.M9mmmP'.JMML.`Moo9^Yo.`Mbmo.JMML.`Ybmd9'.JMML  JMML.db             
#                                                  MM                                                                                                                                                                                        
#                                                .JMML.                                                                                                                                                                                      


# Here, we re-define the matrix multiplication algorithm from File 2. Hello Tensor (tensor_managed.py)
#       with a native programming language so that the program can be executed directly on a physical processor
# The only *two* lines that are noteworthy of your attention are L282 and L290.
# After skimming how we call the natively accelerated _gemm() routine, head on over to File 3B (pyproject.toml/Cargo.toml)
# ---------------------------------------------------------------------------------------------------------------------

def mm(A: Tensor, B: Tensor) -> Tensor: # mm has the same definition from File 2..
    # the same doctest..
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
    # the same preconditions for well-defined matrix multiplication..
    (nrows, p), (p2, mcols) = A.shape, B.shape
    if len(A.shape) != 2 or len(B.shape) != 2: raise RuntimeError("self must be a matrix" if len(A.shape) != 2 else "B must be a matrix")
    if p != p2: raise RuntimeError(f"A and B shapes cannot be multiplied ({nrows}x{p} and {p2}x{mcols})")
    C = mm_native(A,B) # <--- but now we call mm_native! that is, mm is just a facade (i.e a wrapper) for mm_native
    return C

from borscht._borscht import _gemm # in turn, mm_native will also be a facade for the _gemm(a,b,n,k,m) routine imported from the _borscht module implemented in Rust
                                   # the module name is prepended with a leading underscore following PEP 8 see: https://peps.python.org/pep-0008/#package-and-module-names
def mm_native(A: Tensor, B: Tensor) -> Tensor: # we now define mm_native
    (nrows, p), (p2, mcols) = A.shape, B.shape          # realias..
    C = full((nrows, mcols), 0.0)                 # construct output
    C.storage = _gemm(A.storage, B.storage, nrows, p, mcols) # <-- and in turn, mm_native is also a facade for _gemm
    return C                                                                #     where _gemm() takes the flat storages of A and B and produces the flat storage for C

if __name__ == "__main__": _main()            # Just running the file is used as a script
__all__ = ["Tensor", "full", "mm"] # Exports for when the file is used as a module
                                              # Here we maintain the same API as File 2. Hello Tensor (tensor_managed.py)
                                              # and export mm rather than mm_native
                                              # Now, before we implement _gemm(a,b,n,km) File 3C kernelcpu.rs to implement the matrix multiplication algorithm in Rust,
                                              # we set up the mixed Python and Rust source in pyproject.toml, Cargo.toml