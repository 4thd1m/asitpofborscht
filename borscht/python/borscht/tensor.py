# After reading pyproject.toml and Cargo.toml, you understand how borscht's mixed source of Python and Rust interop with one another.
# After reading __init__.py, you have a 50,000 ft view of borscht's internals.
# We will now delve deeper into the primary module that everything else hangs from: the Tensor!
#   First, the naming of Tensor instead of NDArray is unfortunate, in the same way the naming of Vector instead of Array in Rust is.
#   Rust's std inherited the naming from C++'s STL, and while the author of C++'s STL (Stepanov) admitted his mistake, he actually propagated the naming from Scheme.
#   The root is rotten /s, dating all the back to Guy Steele and Gerald Sussman when they created Scheme and even before that with Lisp's vetor.
#
#   Similarly, the naming of Tensor with in LuaTorch, Tensorflow, PyTorch is historical
#   dating back to Yann Lecun and Leon Bottou's neural network simulator for AmigaOS.
#   See: https://x.com/norpadon/status/2045515535643758596
#
#   Just understanding this namespace collision already dispels a lot of misconception around deep learning systems.
#   In the same way that auto-regressive large language models are "just" next-token prediction,
#                                     deep learning frameworks are "just" n-rank arrays.

# So, given that the primary abstraction of deep learning frameworks is an array (albeit n-rank),
# there is a lot of transfer from our traditional training in designing data structures for software 1.0
# Every data structure needs a way of creating, reading, updating, and deleting elements — it's CRUD all the way down!

# Although we will eventually build high-level deep learning abstractions
# such as borscht.nn and borscht.optim (as seen from the 50,000 ft view in __init__.py),
# we start our deep learning system adventure with borscht by getting a matrix multiplication to run on the GPU.
#   This is roughly following what is known as the incremental or Ghuloum approach to compiler construction (http://scheme2006.cs.uchicago.edu/11-ghuloum.pdf)
#   where you implement a vertical slice of the entire compiler pipeline for a single language feature i.e arithmetic
#   While we will literally be applying such form in Week 10 with a Tensor compiler following PyTorch 2,
#                                                   we start with a Tensor interpreter following PyTorch 1.

#   However, the one thing to realize, as you now understand from Chapter 1 is that the comparison of software 2.0's
#   matrix multiplication to software 1.0's arithmetic could *not* be more different, for matrix multiplication is
#   *the* core operation of all deep learning. There is nothing more universal than a linear operation.
#   — "Mathematics is the art of reducing any problem to linear algebra", William Stein (of SageMath!)

# With that said, we will implement all the abstractions necessary for the following forward pass to run
# Down the rabbit hole we go!

# >>> import borscht
# >>> A = torch.full((2,3), 3.141592)
# >>> B = torch.full((3,2), 3.141592)
# >>> torch.mm(A,B)
# tensor([[29.6088, 29.6088],
#         [29.6088, 29.6088]])

#              "Oh dear! Oh dear! I shall be too late!"
#              "I do not want to miss tea time with borscht and pie"

#  \\|//  ,,, \|/                                        \|/  \\|//
# ^^^^^^^^^^^^.                                        .^^^^^^^^^^^^
#           ':|             :  :  :                    |  .    :
#       .     |             '  :  '              o     |  ':
#         '. :| _ _ __         '                / \    | . :
#      :   :':||=|H|  |\   (\   /)           .-------. |' :
#         .   ||_|_|__|_\   \\ //            | ^   ^ | |.
#        . :: |==========   (o o)            | \___/ | |'.     .
#      .      |            (  v  )    .--.   '-------' |
#            ,|            /|:::|\___( 12 )            |  .
#         .  :|           ( |:::|     `--'             |   .  .
#         .   |            (_) (_)                     |'  :
#            :|         '  .                           |'
#            ,|       c[_]                       _     |
#       ,  :  |    o                            [_]    |:
#            :|   / \            ______         | |    |  :. .
#        .    | .-----.         [______]       /   \   |.  , :
#           : | |~ X ~|         |ORANGE|      |DRINK|  | :
#        ,   .| | ~ ^~|         |MARMA-|      | ME  |  | :
#        :   .| '-----'         | LADE |      '-----'  |.
#       .     |                 '------'               |
#      ' '  ' |                                        |
#        '    |   .----.                      _______  |  . :
#             |   |Q   |                     |   |   | | ,.   '
#             |   | <3 |      (O)===,_,      |  o|o  | |: ,:'
#           . |   |   Q|                     |___|___| | ' :
#             |   '----'                               |,.
#             |                                        |   '  '
#             |                   .                    |. ' .
#      ::    ::                                        :      .
#             .                   .                    .
#            ''                                        '      .
#                                 .


#      "Down, down, down. Would the fall never come to an end?"








import builtins
import math

class Tensor: # The more accurate name for the Tensor data structure we all use is an NDArray like numpy, and jax.
    def __init__(self, shape: tuple[int, ...], dtype):
        # A tensor is nothing more than a multidimensional (that is, multi rank) array that approximates some element in R^{d0,...,dn}
        self.shape: tuple[int, ...] = shape # The shape field is a tuple which stores the (d0,...,dn)

        # However, because our machines don't have buffers of arbitrary rank, we have to *virtualize* physical storage into a logical shape
        # This is no different than when an operating system *virtualizes* physical storage into logical directories, or physical memory, into, well.. virtual memory
        # How do we do this?

        # Let's first consider a nxm matrix:

        #                 m columns
        #               <----------->
        #         ^   [ a11  a12  a13 ]
        # n rows  v   [ a21  a22  a23 ]

        # To state the naive solution, we *could* literally design the Tensor data structure with n nested arrays.
        # One could argue that even if a single machine ran out of memory and storage, you could offload onto distributed machines.
        # — The primary reason why this is not a desirable design given that we have to *allocate* an array per additional rank.
        #   In systems programming, allocation is something that should be avoided if possible,
        #   the same way that in "non"-systems programming, a non-polynomial algorithm should be avoided.
        #   This is because moving data from memory to processor is extremely expensive, as we'll understand more later.
        # Can we be more clever?

        
        # Let's consider again the case of an array with rank 2 — that is the nxm matrix, and see if we can generalize it for some arbitrary (d0,...,dn) shape.
        # To derive a second solution, let's use *wishful thinking*, a powerful strategy for synthesis
        # So, what *if* we had a single allocation of a flat array for the nxm matrix?
        #   Clearly, it would have to have nxm elements. For the 2x3 matrix above, the flat array would have to be 6 elements total.
        #   Let's do the stupidly obvious thing which is to store all elements row by row in this single array. i.e "flatening" the matrix.
        # 

        #                 m columns
        #               <----------->
        #         ^   [ a11  a12  a13 ]                       [a11, a12, a13, a21, a22, a23] <- Nice! A *single allocation*
        # n rows  v   [ a21  a22  a23 ]
        #
        #              Logical concept                               Physical reality

        
            # But if we wanted to access a23 by indexing A[(1,2)]?
            # How do we traverse the flat array? How we we map the logical index (1,2) to the physical index of 5?
            # Pause and think!
            #   The key observation to make is that we want to *skip* the first 3 elements in the physical storage,
            #   which is equivalent to skipping the first row in the logical picture.
            #   We can skip the first three elements by multiplying the logical index 1 by 3.
            #   This makes us jump from index 0 to index 0+(1*3)=3.

            #   Then, once we've skipped the first three elements, we can *use* the logical index 2 by adding it to the current index of 3.
            #   This gives us 3+2=5.
            #   **This works more generally for any nxm matrix, where instead of 3 we use n and instead of 5 we use m.**

        # Although more unwieldly, how does this work for indexing a rank-3 array with shape (p,n,m) with index (i,j,k)?
        # Consider a concrete logical shape of (4,2,3)

        #                         +-----+-----+-----+
        #                        /     /     /     /|
        #                       /     /     /     / |
        #                      +-----+-----+-----+  +
        #                     /     /     /     /| /|
        #                    /     /     /     / |/ |
        #                   +-----+-----+-----+  +  +
        #                  /     /     /     /| /| /
        #                 /     /     /     / |/ |/   /
        #                +-----+-----+-----+  +  +   /
        #               /     /     /     /| /| /   /
        #              /     /     /     / |/ |/   /
        #             +-----+-----+-----+  +  +   /
        #         ^   |     |     |     | /| /   /  4 deep
        #         |   |     |     |     |/ |/   /
        # 2 rows  |   +-----+-----+-----+  +   /
        #         |   |     |     |     | /   /
        #         v   |     |     |     |/   /
        #             +-----+-----+-----+   /
        #             <----------------->
        #                  3 columns

        # Then the physical array with a single allocation has to have length 4*2*3=24
        # But, how would we lay out the elements in physical memory?
        #   Since we've conveniently considered a rank-3 array where the previous matrix shape of (2x3) is a sub-problem,
        #   we can simply flatten this rank-3 array by flattening each of the four matrices, one by one.
        #   Denoting each matrix by A, B, C, D, we would have
        #   [a11, a12, a13, a21, a22, a23, b11, b12, b13, b21, b22, b23, c11, c12, c13, c21, c22, c23, d11, d12, d13, d21, d22, d23]

        # And how do we recover the logical semantics of indexing with (i,j,k)?
        # Say, returning d23 when a user indexes with [(3,1,2)]?

        #  We continue the same pattern from the previous case, except we *skip* twice-over with the first two indices.

            # First index
            #   The first index is 3, which means we want to skip over the first 3 matrices.
            #   We can do this by multiplying the logical index 3 by the shape of the matrix, which is 3*(2*3)=18
            #   This lets us jump from the index 0 to the index 0+18=18

            # Notice how the second and third index of 1 and 2 along with the shape of the matrix D being 2x3 is the same as the previous case.
            # The only thing that is different is that we are starting at index 18, which selected *which* matrix to tunnel into.

            # Second index
            #   So since the second index is 1, this means we want to skip over the first row of matrix D.
            #   We can do this by multiplying the logical index 1 by the shape of the row vector, which is 1*3=3
            #   This lets us jump from the index 18 to the index 18+3=21.

            # Third index
            #   And finally, in one step of reasoning, with a third index of 2, we jump from index 21 to 21+2=23.
            #   This is similar to the previous example, where we are no longer skipping anything, just *using* the last index directly.
            #   Moreover, as a sanity check, this is in fact the last index of the physical array with length 24.

            # Taking a step back,
            #   we took the first index of 3 (but more generally i) and multiplied it by the sub-shape of the remaining sub arrays.
            #   In this case it was a 2*3=6-sized matrix, but more generally n*m.
            #   That is, multiplying i by n*m.

            #   Then, we took the second index (but more generally j) and multipled it by the sub-shape of the reamining sub array.
            #   In this case it was 3, but more generally it's m.
            #   That is, multiplying j with m.

            #   Once we reached to the last index (more generally k), we could simply just add that last index to the current index.
            #   That is multiplying k with 1.

        # Do you notice the pattern?
        # Can we generalize such a pattern for an arbirtarily ndarray with shape (d0,...,dn)?

        # After making the key idea of skipping and jumping indices in the physical storage,
        # the key observation to make here is that said skipping is done by
        # correspondingly multiplying the index (i,j,k) with another tuple known as the *stride* (n*m,m,1).
        # A stride is the technical jargon for the numbers you subequently multiply your indices to skip and jump over the relevant elements.

        # The stride field is a tuple which stores stride ((s0 *...*s_{n-1}*1), ... , (s_{n-1}*1), 1)
        # We can optimize the following expression multiplying from right to left with a running product, but we'll keep it as is for now.
        self.stride: tuple[int, ...] = tuple(math.prod(self.shape[d+1:]) for d in builtins.range(len(self.shape)))
        # TODO/TEST:  zero-size dims
        
        self.storage = None # explain storage concept

        # one thread of exposition for general allocation on *various* targets (cpu, nvidia, amd)
        # one thread of exposition for avoiding allocation for ops that just change logical picture


        # explain (device,dtype,layout) concepts below
        self.device = None
        self.dtype: dtype | None = None


    # 🎨 DESIGN CHOICE (DEEP vs SHALLOW embedding)
    # in borscht/pt1/hips-autograd, AD is implemented via operator overloading and tape
    # in jax/pt2, AD is implemented via source transformation
    # see: Uwe's `dco` vs `dcr`

    # scaffold ch.4.1 automatic difefrentiation
    # extend Lab 7: Custom Autograd Backwards for VLA
    def backwards(self):
        # self.grad = Tensor()
        raise NotImplementedError("")

# creation ops

def tensor(data, *, dtype: dtype|None=None, device: device|None=None, requires_grad=False, pin_memory=False) -> "Tensor":
    raise NotImplementedError("")

# annotate .... explaining empty...
# borscht follows the same API as torch, which in turn follows the same API as numpy
def empty(size, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def empty_like(input, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def zeros(size, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def zeros_like(input, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def ones(size, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def ones_like(input, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def full(size, fill_value, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def full_like(input, fill_value, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def arange(start, end=None, step=1, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def range(start, end=None, step=1, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def linspace(start, end, steps, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

def eye(n, m=None, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    raise NotImplementedError("")

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