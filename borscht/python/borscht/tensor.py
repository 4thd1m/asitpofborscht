class dtype:
    def __init__(self):
        raise NotImplementedError("")

class layout:
    def __init__(self):
        raise NotImplementedError("")

class device:
    def __init__(self):
        raise NotImplementedError("")

class Generator:
    def __init__(self, device: device|None=None):
        raise NotImplementedError("")

    def manual_seed(self, seed: int) -> "Generator":
        raise NotImplementedError("")

    def seed(self) -> int:
        raise NotImplementedError("")

    def initial_seed(self) -> int:
        raise NotImplementedError("")

    def get_state(self):
        raise NotImplementedError("")

    def set_state(self, new_state) -> "Generator":
        raise NotImplementedError("")

class Tensor: # scaffold ch2.1 strides, # extend lab 3: Tensor++ and Matrix Decompositions
    def __init__(self, shape: tuple[int, ...], dtype):
        self.shape: tuple[int, ...] = shape
        self.shape: tuple[int, ...] = ()
        self.dtype: dtype | None = None
        self.storage = None # 🎨 DESIGN CHOICE ALIASING/BINDING: MUTABLE vs IMMUTABLE)
                            # in borscht/torch views share storage (MUTATION)
                            # TODO: no Storage concept, direct Tensor to Tensor aliasing????

    # 🎨 DESIGN CHOICE (DEEP vs SHALLOW embedding)
    # in borscht/pt1/hips-autograd, AD is implemented via operator overloading and tape
    # in jax/pt2, AD is implemented via source transformation
    # see: Uwe's `dco` vs `dcr`

    # scaffold ch.4.1 automatic difefrentiation
    # extend Lab 7: Custom Autograd Backwards for VLA
    def backwards(self):
        # self.grad = Tensor()
        raise NotImplementedError("")

# ops legend
# - creation ops
# - random sampling ops
# - indexing/slicing/joining/mutating ops
# - math ops

# creation ops
def from_numpy(ndarray):
    raise NotImplementedError("")

def zeros(size, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def zeros_like(input, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def ones(size, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def ones_like(input, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def arange(start, end=None, step=1, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def range(start, end=None, step=1, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def linspace(start, end, steps, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def eye(n, m=None, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def empty(size, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def empty_like(input, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def full(size, fill_value, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def full_like(input, fill_value, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

# random sampling ops
def rand(*size, generator: Generator|None=None, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def randn(*size, generator: Generator|None=None, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def randint(low, high, size, generator: Generator|None=None, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def normal(mean, std, size=None, generator: Generator|None=None, out=None):
    raise NotImplementedError("")

def multinomial(input, num_samples, replacement=False, generator: Generator|None=None, out=None):
    raise NotImplementedError("")

# indexing, slicing, joining, mutating ops

# math ops

# - constants
import math
from typing import Final
e: Final = math.e
inf: Final = math.inf
nan: Final = math.nan
pi: Final = math.pi

# - pointwise ops
def exp(input, out=None):
    raise NotImplementedError("")

def log(input, out=None):
    raise NotImplementedError("")

def pow(input, exponent, out=None):
    raise NotImplementedError("")

def tanh(input, out=None):
    raise NotImplementedError("")
# - reduciton ops
def prod(input, dim=None, keepdim=False, dtype: dtype|None=None, out=None):
    raise NotImplementedError("")

def sum(input, dim=None, keepdim=False, dtype: dtype|None=None, out=None):
    raise NotImplementedError("")

def logsumexp(input, dim, keepdim=False, out=None):
    raise NotImplementedError("")

def min(input, dim=None, keepdim=False, out=None):
    raise NotImplementedError("")

def max(input, dim=None, keepdim=False, out=None):
    raise NotImplementedError("")

# - comparison ops
# - blas/lapack ops
def dot(input, tensor, out=None):
    raise NotImplementedError("")

def det(input):
    raise NotImplementedError("")

def matmul(input, other, out=None):
    raise NotImplementedError("")

def mm(input, mat2, out=None):
    raise NotImplementedError("")

def mv(input, vec, out=None):
    raise NotImplementedError("")

def svd(input, some=True, compute_uv=True, out=None):
    raise NotImplementedError("")
