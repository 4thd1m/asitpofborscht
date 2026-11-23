#                                     ,,    ,,                                                                                       ,,  
#              `7MMF'  `7MMF'       `7MM  `7MM                     db                 mm                                           `7MM  
#                MM      MM           MM    MM                    ;MM:                MM                                             MM  
# M******A'      MM      MM  .gP"Ya   MM    MM  ,pW"Wq.          ,V^MM.  `7MM  `7MM mmMMmm ,pW"Wq.   .P"Ybmmm `7Mb,od8 ,6"Yb.   ,M""bMM  
# Y     A'       MMmmmmmmMM ,M'   Yb  MM    MM 6W'   `Wb        ,M  `MM    MM    MM   MM  6W'   `Wb :MI  I8     MM' "'8)   MM ,AP    MM  
#      A'        MM      MM 8M""""""  MM    MM 8M     M8        AbmmmqMA   MM    MM   MM  8M     M8  WmmmP"     MM     ,pm9MM 8MI    MM  
#     A',,       MM      MM YM.    ,  MM    MM YA.   ,A9       A'     VML  MM    MM   MM  YA.   ,A9 8M          MM    8M   MM `Mb    MM  
#    A' db     .JMML.  .JMML.`Mbmmd'.JMML..JMML.`Ybmd9'      .AMA.   .AMMA.`Mbod"YML. `Mbmo`Ybmd9'   YMMMMMb  .JMML.  `Moo9^Yo.`Wbmd"MML.
#   A'                                                                                              6'     dP                            
#  A'                                                                                               Ybmmmd'
# **********************************************************************************************************************
#                                               borscht (Literary Source)
#                                              Part 2: From Numerical Linear Algebra Library
# Part 1                                                 to Deep Learning Framework
# <-- File 6. Abstracted Accelerated Tensor    -- File 7. Hello Autograd --               File 8. Adding Forward Ops -->
#             6A. Allocator, Storage                      
#             6B. Dispatcher, Iterator
#
#

# **********************************************************************************************************************

from borscht.tensor_native import Tensor2

class Tensor3(Tensor2):
    def __init__(self):
        super().__init__()