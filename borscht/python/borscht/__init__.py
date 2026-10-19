from .borscht import *

# mod nn (level 2)
# mod optim (level 2)
# mod linalg (level 1)

# mod kernels (level 0) quack,fa(CuTe)
# mod tensor (level -1) CuTe?

from . import nn           # level 2
from . import optim        # level 2
from . import linalg       # level 1
from .tensor import Tensor # level 0
from . import engine       # level 0
from . import kernels      # level -1


__doc__ = borscht.__doc__
if hasattr(borscht, "__all__"):
    __all__ = borscht.__all__

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