#                                                            ,,                                                                                                                    
#                         db                               `7MM                          mm                                   MMP""MM""YMM                                         
#                        ;MM:                                MM                          MM                                   P'   MM   `7                                         
#      ,AM              ,V^MM.    ,p6"bo   ,p6"bo   .gP"Ya   MM  .gP"Ya `7Mb,od8 ,6"Yb.mmMMmm .gP"Ya      M         M              MM  .gP"Ya `7MMpMMMb.  ,pP"Ybd  ,pW"Wq.`7Mb,od8 
#     AVMM             ,M  `MM   6M'  OO  6M'  OO  ,M'   Yb  MM ,M'   Yb  MM' "'8)   MM  MM  ,M'   Yb     M         M              MM ,M'   Yb  MM    MM  8I   `" 6W'   `Wb MM' "' 
#   ,W' MM             AbmmmqMA  8M       8M       8M""""""  MM 8M""""""  MM     ,pm9MM  MM  8M"""""" mmmmMmmmm mmmmMmmmm          MM 8M""""""  MM    MM  `YMMMa. 8M     M8 MM     
# ,W'   MM   ,,       A'     VML YM.    , YM.    , YM.    ,  MM YM.    ,  MM    8M   MM  MM  YM.    ,     M         M              MM YM.    ,  MM    MM  L.   I8 YA.   ,A9 MM     
# AmmmmmMMmm db     .AMA.   .AMMA.YMbmd'   YMbmd'   `Mbmmd'.JMML.`Mbmmd'.JMML.  `Moo9^Yo.`Mbmo`Mbmmd'     M         M            .JMML.`Mbmmd'.JMML  JMML.M9mmmP'  `Ybmd9'.JMML.   
#       MM
# **********************************************************************************************************************
#                                               borscht (Literary Source)
#                                             Part 1: From Arrays to NDArrays
# <-- File 3. Accelerate Tensor               -- File 4. Accelerate++ Tensor --          File 5. Abstract Accelerate -->
#             tensor_nativecpu.py                        tensor_nativegpu.py
#             pyproject.toml/Cargo.toml                  Cargo.toml
#             lib.rs/kernelscpu.rs                       kernelsgpu/
#                                                          utils.rs
#                                                          host.rs
#                                                          device.rs
# **********************************************************************************************************************
#                                              > 3A. tensor_nativecpu.py <
#                                                 0. From Latency-Oriented to Throughput-Oriented Processors

from borscht.tensor_managed import Tensor1                # Let us import Tensor1 to define our GPU accelerated mm routine
from borscht.tensor_managed import full, mm as mm_managed # Let us import the full constructor and managed matrix multiply routine
from borscht.tensor_nativecpu import mm as mm_native      # Let us import                          native  matrix multiply routine
__all__ = ["Tensor1", "full", "mm"]            # Let us re-export the full constructor and Tensor1 concept as-is from tensor_managed.py as well as our natively GPU accelerated mm

# +=============================================================================+
# |                HOST INTERFACE  ::  PCIe Gen5 x16  /  NVLink 4               |
# +=============================================================================+
# |          GigaThread Engine  --  deals thread blocks out to idle SMs         |
# +-----------------------------------------------------------------------------+
# |                                                                             |
# |  +--------- GPC 0 ---------+  +--------- GPC 1 ---------+  +-- GPC 2..7 --+ |
# |  | +-TPC-+ +-TPC-+ +-TPC-+ |  | +-TPC-+ +-TPC-+ +-TPC-+ |  |              | |
# |  | |SM|SM| |SM|SM| |SM|SM| |  | |SM|SM| |SM|SM| |SM|SM| |  |   [ ... ]    | |
# |  | +-----+ +-----+ +-----+ |  | +-----+ +-----+ +-----+ |  |              | |
# |  | +-TPC-+ +-TPC-+ +-TPC-+ |  | +-TPC-+ +-TPC-+ +-TPC-+ |  |   same as    | |
# |  | |SM|SM| |SM|SM| |SM|SM| |  | |SM|SM| |SM|SM| |SM|SM| |  |   GPC 0,     | |
# |  | +-----+ +-----+ +-----+ |  | +-----+ +-----+ +-----+ |  |   x6 more    | |
# |  |  . . .  (9 TPC / GPC)   |  |  . . .  (9 TPC / GPC)   |  |              | |
# |  +------------+------------+  +------------+------------+  +------+-------+ |
# |               |                            |                      |         |
# |  =============v============================v======================v=======  |
# |  ||              L2 CACHE  ::  50 MB  ::  shared by all SMs             ||  |
# |  ========^=============^=============^=============^=============^========  |
# |          |             |             |             |             |          |
# |     +----+----+   +----+----+   +----+----+   +----+----+   +----+----+     |
# |     | MC 0,1  |   | MC 2,3  |   | MC 4,5  |   | MC 6,7  |   | MC 8,9  |     |
# |     +----+----+   +----+----+   +----+----+   +----+----+   +----+----+     |
# +==========|=============|=============|=============|=============|==========+
#            |             |             |             |             |
#       .----+----.   .----+----.   .----+----.   .----+----.   .----+----.
#       |  HBM3   |   |  HBM3   |   |  HBM3   |   |  HBM3   |   |  HBM3   |
#       |=========|   |=========|   |=========|   |=========|   |=========|
#       |=========|   |=========|   |=========|   |=========|   |=========|
#       |=========|   |=========|   |=========|   |=========|   |=========|
#       '---------'   '---------'   '---------'   '---------'   '---------'
#          16 GB         16 GB         16 GB         16 GB         16 GB



def _main():
    import doctest
    doctest.testmod()

    n = 256
    A, B = full((n, n), 1.0), full((n, n), 1.0)
    import time    

    # File 2: matmul implemented by a managed language running on a virtual machine
    t0 = time.perf_counter()
    mm_managed(A, B)
    t = time.perf_counter() - t0
    print(f"{n}x{n} mm_managed: {t*1e3:.1f} ms, {2*n**3 / t / 1e9:.4f} GFLOP/s")

    # File 3: matmul implemented by a native language running on a physical latency-oriented machine
    t0 = time.perf_counter()
    mm_native(A, B)
    t = time.perf_counter() - t0
    print(f"{n}x{n} mm_native: {t*1e3:.1f} ms, {2*n**3 / t / 1e9:.4f} GFLOP/s")

    # File 4: matmul implemented by a native language running on a physical throughput-oriented machine
    mm_nativegpu(A, B) # warm up, since the first call also creates the CUDA context and JIT compiles the kernel's PTX
    t0 = time.perf_counter()
    mm_nativegpu(A, B)
    t = time.perf_counter() - t0
    print(f"{n}x{n} mm_nativegpu: {t*1e3:.1f} ms, {2*n**3 / t / 1e9:.4f} GFLOP/s")

    # timed on device with CUDA events, leaving out the Python list conversions and host<->device copies mm_nativegpu also pays for
    for n in (256, 4096):
        t = _sgemm_bench(n, n, n, reps=10)
        print(f"{n}x{n} sgemm_naive kernel: {t*1e3:.3f} ms, {2*n**3 / t / 1e9:.1f} GFLOP/s")

def mm(A: Tensor1, B: Tensor1) -> Tensor1:
    return mm_nativegpu(A,B)

from borscht._borscht import _sgemm, _sgemm_bench
def mm_nativegpu(A: Tensor1, B: Tensor1) -> Tensor1:
    """Multiply the (n,k) matrix `A` with the (k,m) matrix `B` into a new (n,m) matrix on the GPU.

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
    output.storage = _sgemm(A.storage, B.storage, nrows, p, mcols)
    return output

if __name__ == "__main__": _main()