from borscht.tensor import dtype, layout, device, Generator

# ops legend
# - creation ops
# - random sampling ops
# - indexing/slicing/joining/mutating ops
# - math ops

# // ---------------------------------------------------------------------------
# // op surface required by 3.1_fnn.ipynb and 3.5_gpt2.ipynb
# //
# // creation:    tensor, zeros, ones, arange, tril, randn, randint
# // movement:    view, transpose, unsqueeze, contiguous, split, cat, repeat, t
# // unary:       neg, exp, log, log10, sqrt, tanh, abs
# // binary:      add, sub, mul, div, pow, cmp   (all broadcasting)
# // reduce:      sum, max, mean, var            (over dim, keepdim)
# // matmul:      batched @
# // indexing:    index_select, gather, scatter_add, masked_fill
# // autograd:    backward, grad, no_grad, retain_grad
# // ---------------------------------------------------------------------------

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
