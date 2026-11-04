from . import _borscht
from ._borscht import *

from . import nn
from . import optim        # ch4.2/
from . import linalg       # ch2.3/lab4      🎯 Nov
from . import compile      # ch6/lab 10
from .tensor import Tensor # ch2.1/lab 3     🎯 Nov
from . import func         # ch4.1/lab 7
from . import engine       # ch4.1/lab 7
from . import kernels      # ch2.2,2.4/lab 4 🎯 Nov


__doc__ = _borscht.__doc__
if hasattr(_borscht, "__all__"):
    __all__ = _borscht.__all__

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