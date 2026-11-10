#                          +==========================+
#                          |      THE DATA CYCLE*     |  * original: https://phrack.org/issues/71/calling-all-hackers_md
#                          +==========================+  * adaptaion: https://x.com/fleetwood___/status/2037117778503626937


#                                                FRONTIER LABS
#                    _______________      ,.-^=^=^=^=^=^=^=^=^=^;,
#                  ,;===============>>   E^ OpenAI   Anthropic   '^3.
#                .;^                    E^     Google      xAI     '^3
#               //  labeled data       .E   Meta                    3^
#              /|^                     ^E DeepMind    Mistral       :^
#              ||                       ^;:           Amazon       ;3'
#    ,.^-^-^-^-^-^-^-^-^-^-^;,            ^.=.=_=_=_=_=_=_=_=_=_=_=^
#   E^ Scale AI    Mercor   '^:.              \\\\\\\\\\\\\\\\\\\\
#  E^                Surge    '^3              \\\\\\\\\\\\\\\\\\\\
# E'  Appen                    ^3. DATA         \\\ distillation  \\
# ^;   Remotasks    Prolific   3.' ANNOTATORS    \\\\\\\\\\\\\\\\\\\\
#  E;:           Labelbox     ,3^                 \\\\\\\\\\\\\\\\\\\\
#   ^;._.._._._._._._._._._._,^                    \\\\\\\\\\\\\\\\\\\\
#                                                                /\
#     ^ ^ ^ ^ ^ ^ ^ ^                      DeepSeek   /\  /\  Qwen
#     | | | | | | | |                          Kimi  /::\/::\  /::::\   /\
#     | data extraction |                           / MiniMax /^^^^^^\ /^^\
#     | | | | | | | |          ____________       /      \  /      Yi     \
#                              (AGI 2027 :O)     /____ Baichuan __ GLM ____\
#   \o/ \oXo/\oXoXo/  o        '==========='       CHINESE OPEN-SOURCE LABS
#    |   | |  | | |  /|\          Oo._ /\_/\                  ,///
# __/_\_/_X_\/_X_X_\_/_\__   /_________(@'w'@)_____________.,://'
#         SOCIETY            \''  Cursor-Windsurf ''           MOE
#                                   THE HUMBLE                 runoff
#                                  POST TRAINER

# Welcome to the teaching deep learning framework borscht, a hackable subset of pytorch.
# We continue the hacker's spirit that lives in teaching systems such as
# MIT's teaching operating system xv6[0], Cornell's educational compiler IR bril[1], and Berkeley's educational core sodor[2]

# borscht's source is heavily commented following Knuth's literate programming[3] philosophy
# "Let us change our traditional attitude to the construction of programs:
#  Instead of imagining that our main task is to instruct a computer what to do,
#  let us concentrate rather on explaining to human beings what we want a computer to do."
# Given how comment-heavy borscht's source is, we recommend you to choose your editor's theme accordingly,
# so that you don't strain your eyes to read the comments.
# For VSCode, we like the default light theme "Light+"
#                          and the dark theme "CommentsAreImportant" see: https://marketplace.visualstudio.com/items?itemName=pluveto.comments-are-important

# You are highly encouraged to modify, extend, and hack on borscht.
# Once you feel comfortable, we highly encourage you to graduate onto industry-level deep learning systems such as torch and vllm.
# See you at work.

# 4thd1m
# Waterloo, ON
# November 2026
#
# [0]: https://mit-pdos.github.io/xv6-riscv-book/
# [1]: https://capra.cs.cornell.edu/bril/
# [2]: https://github.com/ucb-bar/riscv-sodor
# [3]: http://www.literateprogramming.com/#
# [4]: https://sitp.ai/










# *********************************************************************************************************************
#                                                borscht (Literary Source)
# <-- File 0. README.md                          -- File 1. __init__.py --                 File 2. tensor_managed.py -->
# *********************************************************************************************************************

# With the introduction aside, welcome to borscht proper — if this __init__.py is the *first* file you are reading, then you are in the right place!
# With the first file of __init__.py, we will understand two ideas:
#   1. How Python's module and packaging system works, since up ahead we will be implementing
#      borscht with a mixed source of Python, Rust, CUDA Rust, and CuTeDSL (Python again!)
#   2. How borscht's internals are organized from a 50,000ft view 





# + ==================================================================================================================+
#                                           1. Code Organization/Distribution
# + ==================================================================================================================+
# Starting with (1), roughly* speaking a programming language's module system is a collection of features to manage code organization.
#                                                  whereas a packaging system is a collection of features to manage code distribution.
# We say roughly speaking because sometimes people call these collections under names,
# but there will always be these two collections of features: code organization and distribution.
# With Python's primary implementation of the CPython interpreter, you can of course evaluate expressions and enter statements.
# But of course upon exiting the interpreter none our program persists. This is problematic for programmers who want to share, and distribute code.
# We are writing a deep learning framework, so this is something we'd like to do.

  # + ------------------------------------------- Module/Script ≜ File -----------------------------------------------
  # The first solution is to create a file with .py suffix appended containing such expressions and statements via text editor.
  # The way to interpret the file is with `python file.py`. Coloquially, this file is known as a *script*
  # If we put all of our deep learning framework in a *single* file and *only* depended on Python's standard library,
  # we could actually distribute such a single file.

  # This becomes an important distinction which we'll see with Rust's module system later.

  # + ------------------------------------------- Package ≜ N Modules = N Files ----------------------------------------
  # However, this method doesn't scale when projects naturally consists of multiple files, which you can see we have below.
  # 


  # + ------------------------------------------- Project ≜  ----------------------------------------
  # 

  # + ------------------------------------------- Wheel ≜  ----------------------------------------
  # 

  # + ------------------------------------------- Package Registry ≜  ----------------------------------------
  # 

  # In asitpofborscht/notebooks
  # uv run python
  # >>> import borscht





# + ==================================================================================================================+
#                                          2. borscht internals, the 50,000ft view
# + ==================================================================================================================+
# For (2), the 50,000 ft view of borscht's internals is simply the following submodules of borscht.
# Once you have familiarized yourself with the three primary "areas", head on over to the second file of our adventure: tensor_managed.py
# There, we will implement the primary abstraction of all deep learning frameworks, the Tensor.
# This is the same borscht.Tensor we used to implement the bigram and perceptron language models.


#-----------------------#
from . import nn        #
from . import optim     #
from . import linalg    #
#-----------^-----------#
#           |
#           |
#           |
#           |
#           |
#           |    After we define the borscht.Tensor, we can
#           |    abstract it's semantics "upwards"
#           |    by adding neural network primitives,
#           |    optimizers, and linear algebra routines
#           |
#           |
# + ================================================================================================================== +

import os
_SHELANG = os.environ.get("SHELANG") or "BSL"


if   _SHELANG == "BSL":   from .tensor_managed   import Tensor, full, mm  # BSL: beginning student language implements the Tensor concept with matrix multiplication
elif _SHELANG == "ISL":   from .tensor_nativecpu import Tensor, full, mm  # ISL: intermediate student language accelerates the Tensor with the Rust programming language
# elif _SHELANG == "ISL+":  from .tensor_nativegpu import Tensor, full, mm  # ISL+: intermediate student language plus accelerates the Tensor with the CUDA Rust programming language
# elif _SHELANG == "ASL":   from .tensor           import Tensor, full, mm  # ASL: advanced student language abstracts device acceleration with Dispatcher, Storage, and Allocator concepts
else: raise ValueError(f"SHELANG must be one of 'BSL', 'ISL', 'ISL+', 'ASL' but got {_SHELANG!r}")

# SITP and borscht is inspired by SICP's dual, HtDP[0], which has various
# teachign language levels[1] with Dr. Racket's (originally PLT Scheme) #lang feature
# [0]: https://cs.brown.edu/people/sk/Publications/Papers/Published/fffk-htdp-vs-sicp-journal/paper.pdf
# [1]: https://docs.racket-lang.org/drracket/htdp-langs.html

# + ================================================================================================================== +
#           |
#           |
#           |    and deepen it's semantics "downwards"
#           |    by accelerating the evaluation of it's ops
#           |    operations referred to as the "forward pass"
#           |    and adding automatic differentiation
#           |    referred to as the "backward pass"
#           |
#           |
#-----------v-----------#
from . import compile   #
from . import ops       #
from . import kernels   #
from . import func      #
from . import engine    #
#-----------------------#












# After reading pyproject.toml, we now understand how borscht's mixed Python and Rust source works.
# — borscht uses CPython's C API combined with Rust's FFI feature targeting the C ABI.
#   But, rather than manually write Rust code with externs, we use the pyo3 crate so that we can simply decorate data and function with macros.
#   When we drive rustc with maturin rather than cargo via `maturin develop`, a shared object file will be produced,
#   called _borscht.cpython-315-<TARGET>.so which we can import as a Python submodule with the following two lines:
from . import _borscht  # recall that the name was configured in pyproject.toml with [tool.maturin].module-name
from ._borscht import * # this underscore follows the naming convention of natively accelerated Python modules, to discourage use by end-users

__doc__ = _borscht.__doc__
if hasattr(_borscht, "__all__"):
    __all__ = _borscht.__all__

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