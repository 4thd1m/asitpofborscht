class Tensor:
    def __init__(self, shape, dtype):
        # scaffold ch2.1 strides
        # scaffold ch2.2 accelerating gemms with cpu
        # scaffold ch2.3 linear solvers
        # extend lab 3: Tensor++ and Matrix Decompositions
        self.shape = ()
        self.stride = ()
        self.dtype = True
        self.storage = bytearray() # 🎨 DESIGN CHOICE ALIASING/BINDING: MUTABLE vs IMMUTABLE)
                                   # in borscht/torch views share storage (MUTATION)

    # 🎨 DESIGN CHOICE (DEEP vs SHALLOW embedding)
    # in borscht/pt1/hips-autograd, AD is implemented via operator overloading and tape
    # in jax/pt2, AD is implemented via source transformation
    # see: Uwe's `dco` vs `dcr`
    def __add__(self, other):
        return NotImplementedError("")

    def dot(self, other):
        #  out_storage = gemv(self.storage, other.storage)
        #  out_tensor = Tensor()
        #  return gemv(self.storage, other.storage)
        return NotImplementedError("")

    def gemm(self, other):
       return NotImplementedError("")

    # scaffold ch.4.1 automatic difefrentiation
    # extend Lab 7: Custom Autograd Backwards for VLA
    def backwards(self):
        # 🎨 DESIGN CHOICE (DEEP vs SHALLOW embedding)
        # in borscht/pt1/hips-autograd, AD is implemented via operator overloading and tape
        # in jax/pt2, AD is implemented via source transformation
        # see: Uwe's `dco` vs `dcr`
        self.grad = Tensor()
        return NotImplementedError("")

# scaffold ch2.1 strides
# scaffold ch2.2 accelerating gemms with cpu
# scaffold ch2.3 linear solvers
# extend lab 3: Tensor++ and Matrix Decompositions
class Dispatcher():
   def __init__(self):
      return

class TensorIterator():
   def __init__(self):
      return