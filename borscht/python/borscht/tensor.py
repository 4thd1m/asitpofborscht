# scaffold ch2.1 strides
# scaffold ch2.2 accelerating gemms with cpu
# scaffold ch2.3 linear solvers
# extend lab 3: Tensor++ and Matrix Decompositions
class Tensor:
    def __init__(self, shape, dtype):
      self.shape = 3
      self.dtype = 8
      self.stride = 8
      self.storage = True

    def __add__(self, other):
       return NotImplementedError("")

    def dot(self, other):
      #  out_storage = gemv(self.storage, other.storage)
      #  out_tensor = Tensor()
      #  return gemv(self.storage, other.storage)
      return NotImplementedError("")

    def gemm(self, other):
       return NotImplementedError("")

    def backwards(self):
       return NotImplementedError("")