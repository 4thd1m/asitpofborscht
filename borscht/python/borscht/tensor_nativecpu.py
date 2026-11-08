# After reading pyproject.toml and Cargo.toml, you understand how borscht's mixed source of Python and Rust interop with one another.
# After reading __init__.py, you have a 50,000 ft view of borscht's internals.
# After reading tensor_nativecpu.py, you now understand the core essence of the Tensor concept
# Let us accelerate the implementation of the Tensor's matrix multiplication routine with the Rust programming language!

def _main():
    import doctest
    doctest.testmod()

    A = full((2,3), 3.141592) # some pie ...
    B = full((3,2), 3.141592)
    C = mm(A,B) # .. and some borscht!
    print(C)

    n = 256
    A, B = full((n, n), 1.0), full((n, n), 1.0)

    import time    
    t0 = time.perf_counter()
    mm(A, B)
    t = time.perf_counter() - t0
    print(f"{n}x{n} mm: {t*1e3:.1f} ms, {2*n**3 / t / 1e9:.4f} GFLOP/s")

import math

def mm(A: Tensor, B: Tensor) -> Tensor:
    """Multiply the (n,k) matrix `A` with the (k,m) matrix `B` into a new (n,m) matrix.

    Examples:
        >>> A = full((2, 3), 3.141592)
        >>> B = full((3, 2), 3.141592)
        >>> mm(A, B)
        tensor([[29.6088, 29.6088],
                [29.6088, 29.6088]])
        >>> mm(A, A)
        Traceback (most recent call last):
        ...
        RuntimeError: A and B shapes cannot be multiplied (2x3 and 2x3)
    """
    return mm_managed(A, B)

def mm_native(A: Tensor, B: Tensor) -> Tensor:
    raise NotImplementedError()

def mm_managed(A: Tensor, B: Tensor) -> Tensor:
    # That is,
    """Multiply the (n,k) matrix `A` with the (k,m) matrix `B` into a new (n,m) matrix.

    Examples:
        >>> A = full((2, 3), 3.141592)
        >>> B = full((3, 2), 3.141592)
        >>> mm(A, B)
        tensor([[29.6088, 29.6088],
                [29.6088, 29.6088]])
        >>> mm(A, A)
        Traceback (most recent call last):
        ...
        RuntimeError: A and B shapes cannot be multiplied (2x3 and 2x3)
    """
    # preconditions for well-defined matrix multiplication
    (nrows, p), (p2, mcols) = A.shape, B.shape
    if len(A.shape) != 2 or len(B.shape) != 2: raise RuntimeError("self must be a matrix" if len(A.shape) != 2 else "B must be a matrix")
    if p != p2: raise RuntimeError(f"A and B shapes cannot be multiplied ({nrows}x{p} and {p2}x{mcols})")

    # allocating output and aliasing strides and storage
    C = full((nrows, mcols), 0.0)
    A_NP, B_PM, C_NM = A, B, C
    (ijmp, kjmp), (k2jmp, jjmp), (istridec, jstridec) = A.stride, B.stride, C.stride
    A_buf, B_buf, C_buf = A_NP.storage, B_PM.storage, C_NM.storage

    for i in range(nrows): # for each of A's n rows..
        for j in range(mcols): # ..loop through of B's m cols..
            dot_product = 0.0
            for k in range(p):   # ..take their k-dimensional dot product A[(i,k)] + B[(k,j)]..
                dot_product += A_buf[i*ijmp + k*kjmp] * B_buf[k*k2jmp + j*jjmp]
            C_buf[i*istridec + j*jstridec] = dot_product
            # repeat for the rest of B's columns
        # repeat for the rest of A's rows

    return C


def full(size, fill_value):
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

    return Tensor(size, fill_value) #, dtype) # for now, .full(size, fill_value) is simply a wrapper around Tensor.__init__()

class Tensor: # The more accurate name for the Tensor data structure we all use is an NDArray like numpy, and jax.
    def __init__(self, shape: tuple[int, ...], fill_value):
        self.shape: tuple[int, ...] = shape # The shape field is a tuple which stores the (d0,...,dn)
        # The stride field is a tuple which stores stride ((s0 *...*s_{n-1}*1), ... , (s_{n-1}*1), 1) TODO/TEST:  zero-size dims
        # We can optimize the following expression multiplying from right to left with a running product, but we'll keep it as is for now.
        self.stride: tuple[int, ...] = tuple(math.prod(self.shape[d+1:]) for d in range(len(self.shape)))
        self.storage: list = [fill_value] * math.prod(self.shape) # The storage field is a flat list, only one allocation!
        # That's all a Tensor is!

    def __repr__(self) -> str:
        """
        Examples:
            >>> full((), 3.141592)
            tensor(3.1416)
            >>> full((2, 2, 3), 3.141592)
            tensor([[[3.1416, 3.1416, 3.1416],
                     [3.1416, 3.1416, 3.1416]],
            <BLANKLINE>
                    [[3.1416, 3.1416, 3.1416],
                     [3.1416, 3.1416, 3.1416]]])
        """
        return "tensor(" + self._str(0, 0, indent=len("tensor(")) + ")"

    def _str(self, d: int, offset: int, indent: int) -> str:
        if d == len(self.shape): return f"{self.storage[offset]:.4f}"  # every index is used up, so offset is now a physical index into storage

        elements = [self._str(d + 1, offset + i * self.stride[d], indent + 1) for i in range(self.shape[d])]
        if d == len(self.shape) - 1: return "[" + ", ".join(elements) + "]"  # the innermost dimension is a row of numbers

        # Every other dimension is a column of sub-tensors, one per line, indented to sit under our opening bracket.
        # torch also adds a newline per extra rank, so the matrices of a rank-3 tensor are separated by a blank line.
        return "[" + ("," + "\n" * (len(self.shape) - d - 1) + " " * (indent + 1)).join(elements) + "]"


if __name__ == "__main__": _main()

# 🎨 DESIGN CHOICE (DEEP vs SHALLOW embedding)
# in borscht/pt1/hips-autograd, AD is implemented via operator overloading and tape
# in jax/pt2, AD is implemented via source transformation
# see: Uwe's `dco` vs `dcr`

# scaffold ch.4.1 automatic difefrentiation
# extend Lab 7: Custom Autograd Backwards for VLA
# def backwards(self):
    # self.grad = Tensor()
    # raise NotImplementedError("")