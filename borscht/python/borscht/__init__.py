# After reading pyproject.toml and Cargo.toml, we now understand how borscht's mixed Python and Rust source works.
# — borscht uses CPython's C API combined with Rust's FFI feature targeting the C ABI.
#   But, rather than manually write Rust code with externs, we use the pyo3 crate so that we can simply decorate data and function with macros.
#   When we drive rustc with maturin rather than cargo via `maturin develop`, a shared object file will be produced,
#   called _borscht.cpython-315-<TARGET>.so which we can import as a Python submodule with the following two lines:
from . import _borscht  # recall that the name was configured in pyproject.toml with [tool.maturin].module-name
from ._borscht import * # this underscore follows the naming convention of natively accelerated Python modules, to discourage use by end-users

# One question you may be asking is how does one decide to write framework code in Python vs in Rust?
# When we say "framework code", we are referring to all code that runs on the CPU unrelated to the code that runs on the GPU
# To answer that question, head on over to tensor.py, the primary module of borscht
# — After we define the borscht.Tensor, we can
#     1. deepen it's semantics "downwards" by accelerating the evaluation of it's operations referred to as "forwards"
#                                          and adding automatic differentiation referred to as "backwards"
#     2. abstract it's semantics "upwards" by adding neural network primitives, optimizers, and linear algebra routines

from . import nn
from . import optim
from . import linalg
from .tensor import Tensor, dtype, layout, device, Generator # <--- You can see how other module builds on or below the Tensor
from . import ops                                            #      the Tensor plays a somewhat similar role that
from . import func                                           #      the IP protocol does in the networking stack or the ISA does in the hwardware-software interface
from . import engine                                         #      That is to say, the Tensor is *the* major interface
from . import kernels
from . import compile


# note: this rebinds the attribute borscht.tensor from the submodule to the tensor() constructor, like torch.tensor
from .tensor import tensor, empty, empty_like, zeros, zeros_like, ones, ones_like, full, full_like, arange, linspace, eye
from .tensor import mm


__doc__ = _borscht.__doc__
if hasattr(_borscht, "__all__"):
    __all__ = _borscht.__all__