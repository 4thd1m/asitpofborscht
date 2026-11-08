import math

def full(size, fill_value): #out=None, dtype: dtype|None=None, layout: layout|None=None, device: device|None=None, requires_grad=None):
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

def mm(input: Tensor, mat2: Tensor) -> Tensor: #, out=None):
    """Multiply the (n,k) matrix `input` with the (k,m) matrix `mat2` into a new (n,m) matrix.

    Examples:
        >>> A = full((2, 3), 3.141592)
        >>> B = full((3, 2), 3.141592)
        >>> mm(A, B)
        tensor([[29.6088, 29.6088],
                [29.6088, 29.6088]])
        >>> mm(A, A)
        Traceback (most recent call last):
        ...
        RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
    """
    # mm_naive(input, mat2)
    # _gemm()
    raise NotImplementedError("")

def mm_native(input: Tensor, mat2: Tensor) -> Tensor:
    raise NotImplementedError()

def mm_python(input: Tensor, mat2: Tensor) -> Tensor: #, out=None):
    """Multiply the (n,k) matrix `input` with the (k,m) matrix `mat2` into a new (n,m) matrix.

    Examples:
        >>> A = full((2, 3), 3.141592)
        >>> B = full((3, 2), 3.141592)
        >>> mm(A, B)
        tensor([[29.6088, 29.6088],
                [29.6088, 29.6088]])
        >>> mm(A, A)
        Traceback (most recent call last):
        ...
        RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
    """
    (nrows, k), (k2, mcols) = input.shape, mat2.shape
    if len(input.shape) != 2 or len(mat2.shape) != 2: raise RuntimeError("self must be a matrix" if len(input.shape) != 2 else "mat2 must be a matrix")
    if k != k2: raise RuntimeError(f"mat1 and mat2 shapes cannot be multiplied ({nrows}x{k} and {k2}x{mcols})")

    output = full((nrows, mcols), 0.0)
    A_NK, B_KM, C_NM = input, mat2, output
    A_buf, B_buf, C_buf = A_NK.storage, B_KM.storage, C_NM.storage

    (ijmp, pjmpa), (pjmpb, jjmp), (istridec, jstridec) = input.stride, mat2.stride, output.stride
    for i in range(nrows):
        for j in range(mcols):
            dot_product = 0.0
            for p in range(k):
                dot_product += A_buf[i*ijmp + p*pjmpa] * B_buf[p*pjmpb + j*jjmp]
            C_buf[i*istridec + j*jstridec] = dot_product

    return output


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# 🎨 DESIGN CHOICE (DEEP vs SHALLOW embedding)
# in borscht/pt1/hips-autograd, AD is implemented via operator overloading and tape
# in jax/pt2, AD is implemented via source transformation
# see: Uwe's `dco` vs `dcr`

# scaffold ch.4.1 automatic difefrentiation
# extend Lab 7: Custom Autograd Backwards for VLA
# def backwards(self):
    # self.grad = Tensor()
    # raise NotImplementedError("")