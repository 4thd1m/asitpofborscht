# After reading pyproject.toml and Cargo.toml, you understand how borscht's mixed source of Python and Rust interop with one another.
# After reading __init__.py, you have a 50,000 ft view of borscht's internals.
# We will now delve deeper into the primary module that everything else hangs from: the Tensor!

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

# """
# >>> import borscht
# >>> A = torch.full((2,3), 3.141592)
# >>> B = torch.full((3,2), 3.141592)
# >>> torch.mm(A,B)
# tensor([[29.6088, 29.6088],
#         [29.6088, 29.6088]])
# """

#                                      "Oh dear! Oh dear! I shall be too late!"

#                          \\|//  ,,, \|/                                        \|/  \\|//
#                         ^^^^^^^^^^^^.                                        .^^^^^^^^^^^^
#                                   ':|             :  :  :                    |  .    :
#                               .     |             '  :  '              o     |  ':
#                                 '. :| _ _ __         '                / \    | . :
#                              :   :':||=|H|  |\   (\   /)           .-------. |' :
#                                 .   ||_|_|__|_\   \\ //            | ^   ^ | |.
#                                . :: |==========   (o o)            | \___/ | |'.     .
#                              .      |            (  v  )    .--.   '-------' |
#                                    ,|            /|:::|\___( 12 )            |  .
#                                 .  :|           ( |:::|     `--'             |   .  .
#                                 .   |            (_) (_)                     |'  :
#                                    :|         '  .                           |'
#                                    ,|       c[_]                       _     |
#                               ,  :  |    o                            [_]    |:
#                                    :|   / \            ______         | |    |  :. .
#                                .    | .-----.         [______]       /   \   |.  , :
#                                   : | |~ X ~|         |ORANGE|      |DRINK|  | :
#                                ,   .| | ~ ^~|         |MARMA-|      | ME  |  | :
#                                :   .| '-----'         | LADE |      '-----'  |.
#                               .     |                 '------'               |
#                              ' '  ' |                                        |
#                                '    |   .----.                      _______  |  . :
#                                     |   |Q   |                     |   |   | | ,.   '
#                                     |   | <3 |      (O)===,_,      |  o|o  | |: ,:'
#                                   . |   |   Q|                     |___|___| | ' :
#                                     |   '----'                               |,.
#                                     |                                        |   '  '
#                                     |                   .                    |. ' .
#                              ::    ::                                        :      .
#                                     .                   .                    .
#                                    ''                                        '      .
#                                                         .

#                               "Down, down, down. Would the fall never come to an end?"





# + ==================================================================================================================+
#                                                 What's in a name?
# + ==================================================================================================================+

#                                                          |>
#                                                         /^\
#                                                        /   \
#                                                       /_____\
#        *                 .                            | .-. |
#   .            ,--.             *                     | |_| |
#               / ,-'                    .              |     |              'Tis but thy name that is my enemy;
#       .      | (               .            __________|     |__________    Thou art thyself, though not a Montague.
#               \ `-.      *                  |                         |    What's Montague? It is nor hand, nor foot,
#    *           `--'                   *     |    .---------------.    |    Nor arm, nor face, nor any other part
#                          .                  |   /                 \   |    Belonging to a man. O, be some other name!
#             .                      .        |  |       .@@@.       |  |    What's in a name? That which we call a rose
#   *                                         |  |      @(-.-)@      |  |    By any other name would smell as sweet;
#                  .                      .   |  |      @/) (\@      |  |    So Romeo would, were he not Romeo call'd,
#                                             | _|______/     \______|_ |    Retain that dear perfection which he owes
#        .                     *              | |_|_|_|_|_|_|_|_|_|_|_| |    Without that title. Romeo, doff thy name,
#          .-~~-.~~-.                         | \_____________________/ |    And for that name which is no part of thee
#        .(  o    o  ).                       |  ;                   ;  |    Take all myself.
#       (  o   o    o  )            _\_       | ;%                   %; |    — Juliet, Act II Scene II
#        '-.o  .  o .-'            (^ ^)  /   |  %;                 ;%  |    *see: https://en.wikipedia.org/wiki/A_rose_by_any_other_name_would_smell_as_sweet
#           '-.| |.-'             __) (__/    | ;%                   %; |
#  _______     | |               /<3|  |      |  ;       .-----.     ;  |
# |__|__|_|    | |                  |__|      |          |     |        |
# |_|__|__|    | |                  /  \      |          |   o |        |
# |__|__|_|  \\| |//   @}-,-'--   _/    \_    |__________|_____|________|
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#   First, in the realm of numerical linear algebra (which forms the basis of all deep learning frameworks...)
#   we say these mathematical words like "..a vector is an element of a vector space over the field of real numbers,
#                                         which in turn has a basis that are encoded by matrices.."
#   even though what we really mean are their computational representations

#   For instance, computer arithmetic is done with what is known as a "floating point" representation (TODO, square with dtypes?)
#   which, besides the fact of not being able to represent every real number (actual infinity is very different from potential infinity)
#   it is not even associative!
"""
>>> type(0.2)
<class 'float'>
>>> (0.1 + 0.2) + 0.3
0.6000000000000001
>>> 0.1 + (0.2 + 0.3)
0.6
"""

#   And, as we now understand from Intermezzo One, if you don't have associativity then you sure as don't have distributivity:

__test__ = {"distributivity": """
>>> 0.9 * (0.2 + 0.3)
0.45
>>> 0.9 * 0.2 + 0.9 * 0.3
0.45000000000000007
"""}

#   Since a matrix over a field uses that field's definition of vector addition and scalar multiplication,
#   the matrices we use for computation are no longer linear as well.

#   Second, the naming of Tensor instead of NDArray is unfortunate, in the same way the naming of Vector instead of Array in Rust is.
#   In software 1.0, many languages use the name "Vector" when what they really mean is a List.
#   — That is, the semantics of the data structure is "list-able" or "list-like" i.e the concept of an Iterator
#     Rust (the language we will be using in addition to Python) carries such historical naming,
#     which in turn inherited it from C++'s STL (Stepanov), which in turn inherited it from Scheme (Steele/Sussman).
#     Although this naming collision is something you got used to, it is *extremely* confusing for learners.
#     To people first learning, they are confused whether the semantics is literally like a vector from linear algebra, or something else.
#     This phenomenon known as the "expert blindspot" is well-known problem in the cognitive science literature.
#     see: (Nathan et al., 2001) https://pact.cs.cmu.edu/koedinger/pubs/2001_NathanEtAl_ICCS_EBS.pdf
#     Even Stepanov himself considered it a mistake the name the STL's list a vector instead of an array.

#   Similarly, the naming of Tensor with in LuaTorch, Tensorflow, PyTorch is historical
#   dating back to Yann Lecun and Leon Bottou's neural network simulator for AmigaOS. see: https://x.com/norpadon/status/2045515535643758596
#   While the name Tensor sounds l33t (yeah! Einstein summation), it too, is *extremely* confusing for learners.
#   Moving forwards in the borscht source, in the same way that a "matrix" does *not* refer to a linear mapping,
#                                                               a "tensor" does *not* refer to a bilinear mapping (TODO).

#   Rather, the semantics of the Tensor data structure is "nd-list-able" or "ndlist-like".
#   Just understanding this namespace collision already dispels a lot of misconception around deep learning systems.
#   In the same way that auto-regressive large language models are "just" next-token prediction,
#                                 deep learning frameworks are "just" n-rank arrays.





# + ==================================================================================================================+
#                                              Some pie with your borscht!
# + ==================================================================================================================+
# With those qualifications aside, let us consider the previously mentioned matrix multiplication
# —pirozhi (пирожо́к) goes well with  borscht (борщ)!

#                     (     )     (     )        _
#                      )   (       )   (        //
#                     (     )     (     )      //
#                      )   (       )   (      //
#              .-----------------------------//.
#            .'   ~   ~   ~   .--.   ~   ~   ~   '.
#           (   ~   ~   ~   ~( ** )~   ~   ~   ~   )
#            '.   ~   ~   ~   '--'   ~   ~   ~   .'
#            |'---------------------------------'|
#            |  <><><><><><><><><><><><><><><><>  |         .--------------------------------------.
#            \                                   /        .'   _.-""-._     _.-""-._     _.-""-._   '.
#             \                                 /        /    ( 3.1416 )   ( 3.1416 )   ( 3.1416 )    \
#              '.                             .'         |     `-....-'     `-....-'     `-....-'     |
#                '-._                     _.-'           |     _.-""-._     _.-""-._     _.-""-._     |
#                    '-------------------'               \    ( 3.1416 )   ( 3.1416 )   ( 3.1416 )    /
#                 ________\___________/________           '.   `-....-'     `-....-'     `-....-'   .'
#                '-----------------------------'            '--------------------------------------'

# """
# >>> import borscht
# >>> A = borscht.full((2,3), 3.141592)               # 1. creating tensors + ...
# >>> B = borscht.full((3,2), 3.141592)

# >>> print(A)                                        # 2. reading tensors + ...
# tensor([[3.1416, 3.1416, 3.1416],
#         [3.1416, 3.1416, 3.1416]])
# >>> print(B)
# tensor([[3.1416, 3.1416],
#         [3.1416, 3.1416],
#         [3.1416, 3.1416]])

# >>> borscht.mm(A,B)                                 # 3. updating tensors + ...
# tensor([[29.6088, 29.6088],
#         [29.6088, 29.6088]])
# """                                                 # 4. = agi.

# Our task is now to let the user construct (1) and multiply (3) two matrices A and B with shapes (2,3) and and (3,2) respectively
# We will construct such Tensors in the remainder of tensor.py,
#    and  multiply the two Tensors in ops.py

# Note: We will interchangeably describe a Tensor's with "a matrix of shape (2,3)" and "a 2x3 matrix".
# Both have the same meaning of *shape*.

import builtins
import math

# Although Python does indeed have @staticmethod and @classmethod decorator's,
# borscht's Tensor constructors/factory methods are free functions,
# following torch and numpy, which in turn, were inspired by MatLab and APL
# Many design decisions of systems with history are implicitly inherited rather than explicitly designed.

def full(size, fill_value, out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
    """Create a Tensor of shape `size` filled with `fill_value`.

    Args:
        size: Shape of the output tensor.
        fill_value: Value to fill with. Determines the dtype.

    Keyword Args:
        out (Tensor, optional): Tensor to write the result into.
        dtype (dtype, optional): Data type of the result. Defaults to the type
            inferred from `fill_value`.
        layout (layout, optional): Memory layout of the result.
        device (device, optional): Device to allocate the result on.
        requires_grad (bool, optional): Whether autograd records operations on
            the result. Defaults to False.

    Returns:
        A new Tensor.

    Examples:
        >>> A = full((2, 3), 3.141592)
        >>> A
        tensor([[3.1416, 3.1416, 3.1416],
                [3.1416, 3.1416, 3.1416]])
        >>> A.shape
        (2, 3)
    """

    return Tensor(size, fill_value) # for now, .full(size, fill_value) is simply a wrapper around Tensor.__init__()

class Tensor: # The more accurate name for the Tensor data structure we all use is an NDArray like numpy, and jax.
    def __init__(self, shape: tuple[int, ...], fill_value):
        # A tensor is nothing more than a multidimensional (that is, multi rank) array that approximates some element in R^{d0,...,dn}
        self.shape: tuple[int, ...] = shape # The shape field is a tuple which stores the (d0,...,dn)

        # However, because our machines don't have buffers of arbitrary rank, we have to *virtualize* physical storage into a logical shape
        # This is no different than when an operating system *virtualizes* physical storage into logical directories, or physical memory, into, well.. virtual memory
        # How do we do this?

        # Let's first consider a nxm matrix.
        # To be concrete, the 2x3 matrix A, whose elements (while all 3.141592) are denoted by aij

        #                 m columns
        #               <----------->
        #         ^   [ a11  a12  a13 ]
        # n rows  v   [ a21  a22  a23 ]

        # + ============================================================================================================
        #                                              Naive
        # + ============================================================================================================
        # To state the naive solution, we *could* literally design the Tensor data structure with n nested arrays.
        # One could argue that even if a single machine ran out of memory and storage, you could offload onto distributed machines.
        # — The primary reason why this is not a desirable design given that we have to *allocate* an array per additional rank.
        #   In systems programming, allocation is something that should be avoided if possible,
        #   the same way that in "non"-systems programming, a non-polynomial algorithm should be avoided.
        #   This is because moving data from memory to processor is extremely expensive, as we'll understand more later.
        # Can we be more clever?

        
        # + ============================================================================================================
        #                                  Being more clever, for an nxm matrix
        # + ============================================================================================================
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

        # + ============================================================================================================
        #                                 Does it generalize for a pxnxm tensor?
        # + ============================================================================================================
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
        # We say a "stride" is the tuple of numbers you subequently multiply your indices to skip and jump over the relevant elements.

        # The stride field is a tuple which stores stride ((s0 *...*s_{n-1}*1), ... , (s_{n-1}*1), 1) TODO/TEST:  zero-size dims
        # We can optimize the following expression multiplying from right to left with a running product, but we'll keep it as is for now.
        self.stride: tuple[int, ...] = tuple(math.prod(self.shape[d+1:]) for d in builtins.range(len(self.shape)))
        self.storage: list = [fill_value] * math.prod(self.shape) # The storage field is a flat list, only one allocation!
        self.dtype: dtype | None = None

        # TODO: device, offset, layout??
        self.device = None


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

if __name__ == "__main__":
    import doctest
    doctest.testmod()