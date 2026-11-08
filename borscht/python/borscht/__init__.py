# After reading pyproject.toml and Cargo.toml, we now understand how borscht's mixed Python and Rust source works.
# — borscht uses CPython's C API combined with Rust's FFI feature targeting the C ABI.
#   But, rather than manually write Rust code with externs, we use the pyo3 crate so that we can simply decorate data and function with macros.
#   When we drive rustc with maturin rather than cargo via `maturin develop`, a shared object file will be produced,
#   called _borscht.cpython-315-<TARGET>.so which we can import as a Python submodule with the following two lines:
from . import _borscht  # recall that the name was configured in pyproject.toml with [tool.maturin].module-name
from ._borscht import * # this underscore follows the naming convention of natively accelerated Python modules, to discourage use by end-users

# One question you may be asking is how does one decide to write framework code in Python vs in Rust?
# When we say "framework code", we are referring to all code that runs on the CPU unrelated to the code that runs on the GPU
# To answer that question, head on over to tensor_managed.py, the primary module of borscht
# — After we define the borscht.Tensor, we can
#     1. deepen it's semantics "downwards" by accelerating the evaluation of it's operations referred to as "forwards"
#                                          and adding automatic differentiation referred to as "backwards"
#     2. abstract it's semantics "upwards" by adding neural network primitives, optimizers, and linear algebra routines

from . import nn
from . import optim
from . import linalg





# You can see how other module builds on or below the Tensor the Tensor plays a somewhat similar role that
# the IP protocol does in the networking stack or the ISA does in the hwardware-software interface
# That is to say, the Tensor is *the* major interface

# SITP and borscht is inspired by SICP's dual, HtDP[0], which has various
# teachign language levels[1] with Dr. Racket's (originally PLT Scheme) #lang feature
# Such languages are
import os
_SHELANG = os.environ.get("SHELANG") or "BSL"
if   _SHELANG == "BSL":   from .tensor_managed   import Tensor, full, mm  # BSL: beginning student language implements the Tensor concept with matrix multiplication
elif _SHELANG == "ISL":   from .tensor_nativecpu import Tensor, full, mm  # ISL: intermediate student language accelerates the Tensor with the Rust programming language
elif _SHELANG == "ISL+":  from .tensor_nativegpu import Tensor, full, mm  # ISL+: intermediate student language plus accelerates the Tensor with the CUDA Rust programming language
elif _SHELANG == "ASL":   from .tensor import Tensor, full, mm            # ASL: advanced student language abstracts device acceleration with Dispatcher, Storage, and Allocator concepts
else: raise ValueError(f"SHELANG must be one of 'BSL', 'ISL', 'ISL+', 'ASL' but got {_SHELANG!r}")

# [0]: https://cs.brown.edu/people/sk/Publications/Papers/Published/fffk-htdp-vs-sicp-journal/paper.pdf
# [1]: https://docs.racket-lang.org/drracket/htdp-langs.html





from . import compile                                                       
from . import ops
from . import kernels

from . import func
from . import engine


__doc__ = _borscht.__doc__
if hasattr(_borscht, "__all__"):
    __all__ = _borscht.__all__