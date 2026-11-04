import math

# After reading pyproject.toml and Cargo.toml, you understand how borscht's mixed source of Python and Rust interop with one another.
# After reading __init__.py, you have a 50,000 ft view of how borscht's internals.
# We will will now delve deeper into the primary module that everything else hangs from, the Tensor

#   First, the naming of Tensor instead of NDArray is unfortunate, in the same way the naming of Vector instead of Array in Rust is.
#   Rust's std inherited the naming from C++'s STL. While Stepanov of C++ STL admitted his mistake, he actually propagted the naming from Scheme.
#   The root is rotten, dating all the back to Guy Steele and Gerald Sussman when they created Scheme /s
#
#   Similarly, the naming of Tensor from tensorflow and later pytorch is also historical,
#   dating back to Yann Lecun and Leon Bottou's neural network simulator for AmigaOS.
#   See: https://x.com/norpadon/status/2045515535643758596
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
        #   the same way that a non-polynomial algorithm should be avoided.
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

            # How does this work for indexing a rank-3 array with shape (p,n,m) with index (i,j,k)?
            # Although more unwieldly, consider a concrete logical shape of (4,2,3)

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

            # First index
            #   We continue the same pattern from the previous case, except we *skip* twice-over.
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

        # The shape field is a tuple which stores stride ((s0 *...*s_{n-1}*1), ... , (s_{n-1}*1), 1)
        self.stride: tuple[int, ...] = tuple(math.prod(self.shape[d+1:]) for d in range(len(self.shape)))
        self.storage = None # The storage field is..
        # 🎨 DESIGN CHOICE ALIASING/BINDING: MUTABLE vs IMMUTABLE), in borscht/torch views share storage (MUTATION). TODO: no Storage concept, direct Tensor to Tensor aliasing????


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