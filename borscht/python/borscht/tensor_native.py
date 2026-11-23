                                                                                                                                                                      
#                                ,,                                                                     ,,                                                              
#    .6*"                db     *MM                  mm                             mm                `7MM                                                              
#  ,M'                  ;MM:     MM                  MM                             MM                  MM                                                              
# ,Mbmmm.              ,V^MM.    MM,dMMb.  ,pP"Ybd mmMMmm `7Mb,od8 ,6"Yb.  ,p6"bo mmMMmm .gP"Ya    ,M""bMM                                                              
# 6M'  `Mb.           ,M  `MM    MM    `Mb 8I   `"   MM     MM' "'8)   MM 6M'  OO   MM  ,M'   Yb ,AP    MM                                                              
# MI     M8           AbmmmqMA   MM     M8 `YMMMa.   MM     MM     ,pm9MM 8M        MM  8M"""""" 8MI    MM                                                              
# WM.   ,M9 ,,       A'     VML  MM.   ,M9 L.   I8   MM     MM    8M   MM YM.    ,  MM  YM.    , `Mb    MM                                                              
#  WMbmmd9  db     .AMA.   .AMMA.P^YbmdP'  M9mmmP'   `Mbmo.JMML.  `Moo9^Yo.YMbmd'   `Mbmo`Mbmmd'  `Wbmd"MML.                                                            
#                                                                                                                                                                      
#                                                                                                                                                                     
#                                                          ,,                                              ,,                                                           
#                       db                               `7MM                          mm                `7MM      MMP""MM""YMM                                         
#                      ;MM:                                MM                          MM                  MM      P'   MM   `7                                         
#                     ,V^MM.    ,p6"bo   ,p6"bo   .gP"Ya   MM  .gP"Ya `7Mb,od8 ,6"Yb.mmMMmm .gP"Ya    ,M""bMM           MM  .gP"Ya `7MMpMMMb.  ,pP"Ybd  ,pW"Wq.`7Mb,od8 
#                    ,M  `MM   6M'  OO  6M'  OO  ,M'   Yb  MM ,M'   Yb  MM' "'8)   MM  MM  ,M'   Yb ,AP    MM           MM ,M'   Yb  MM    MM  8I   `" 6W'   `Wb MM' "' 
#                    AbmmmqMA  8M       8M       8M""""""  MM 8M""""""  MM     ,pm9MM  MM  8M"""""" 8MI    MM           MM 8M""""""  MM    MM  `YMMMa. 8M     M8 MM     
#                   A'     VML YM.    , YM.    , YM.    ,  MM YM.    ,  MM    8M   MM  MM  YM.    , `Mb    MM           MM YM.    ,  MM    MM  L.   I8 YA.   ,A9 MM     
#                 .AMA.   .AMMA.YMbmd'   YMbmd'   `Mbmmd'.JMML.`Mbmmd'.JMML.  `Moo9^Yo.`Mbmo`Mbmmd'  `Wbmd"MML.       .JMML.`Mbmmd'.JMML  JMML.M9mmmP'  `Ybmd9'.JMML.   
# **********************************************************************************************************************
#                                               borscht (Literary Source)
#                                              Part 1: From Arrays to NDArrays                Part 1 END
# <-- File 5. Accelerate++ Tensor              -- File 6. Abstracted Accelerated Tensor --    File 7. Hello Autograd -->
#             5A. tensor_nativegpu.py                     6A. Allocator, Storage              Part 2 START
#             5B. Cargo.toml/lib.rs                       6B. Dispatcher, Iterator
#             5C. kernelsgpu.rs
#                   utils.rs
#                   host.rs
#                   device.rs
#
#

# **********************************************************************************************************************

from borscht.tensor_managed import Tensor1

class Tensor2(Tensor1):
    def __init__(self):
        return                                                                                                                      
                                                                                                                                                                      

# Storage
# Allocator
# Device
# Kernel

# def full(size, fill_value):
#     raise NotImplementedError("borscht.tensor.full")

# class Tensor:
#     def __init__(self, shape: tuple[int, ...], fill_value):
#         raise NotImplementedError("borscht.tensor.Tensor")

# def mm(A: Tensor, B: Tensor) -> Tensor:
#     raise NotImplementedError("borscht.tensor.mm")

# One question you may be asking is how does one decide to write framework code in Python vs in Rust?
# When we say "framework code", we are referring to all code that runs on the CPU unrelated to the code that runs on the GPU
# To answer that question, head on over to tensor_managed.py, the primary module of borscht

# 🎨 DESIGN CHOICE (DEEP vs SHALLOW embedding)
# in borscht/pt1/hips-autograd, AD is implemented via operator overloading and tape
# in jax/pt2, AD is implemented via source transformation
# see: Uwe's `dco` vs `dcr`

# scaffold ch.4.1 automatic difefrentiation
# extend Lab 7: Custom Autograd Backwards for VLA
# def backwards(self):
    # self.grad = Tensor()
    # raise NotImplementedError("")