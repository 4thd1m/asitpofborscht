import borscht                          # we look up borscht.mm at *call* time rather than `from borscht import mm` at import time,
                                        # since borscht/__init__.py imports nn *before* it binds mm according to SHELANG (BSL's managed mm, ISL's native mm)
from ..tensor_managed import Tensor1


def linear(x: Tensor1, weight: Tensor1, bias: Tensor1 | None = None) -> Tensor1:
    """Apply the affine map y = xWᵀ + b to a batch of row vectors x.

    Args:
        x: (N, in_features) batch of N row vectors.
        weight: (out_features, in_features) matrix W. Row j is output j's weight vector w_j.
        bias: Optional (out_features,) vector b, added to every row of the output.

    Returns:
        (N, out_features) Tensor where y[i, j] = x[i] · w_j + b[j].

    Examples:
        >>> from borscht.tensor_managed import full
        >>> W = full((2, 3), 0.0); W.storage = [1.0, 2.0, 3.0,
        ...                                     4.0, 5.0, 6.0]
        >>> linear(full((1, 3), 1.0), W, full((2,), 0.5))
        tensor([[6.5000, 15.5000]])
    """
    y = borscht.mm(x, _t(weight))  # (N, in) @ (in, out) -> (N, out): one matmul computes all N*out dot products x[i] · w_j
    if bias is not None:           # then shift every row by b, i.e. broadcast the (out,) vector across the N rows
        (N, out) = y.shape
        for i in range(N):
            for j in range(out):
                y.storage[i*y.stride[0] + j*y.stride[1]] += bias.storage[j*bias.stride[0]]
    return y

# Why is W stored as (out, in) and transposed, rather than stored as (in, out)? Convention inherited from torch:
# in math the layer is y = Wx + b on a single *column* vector x, so W's rows are the out_features "neurons".
# Deep learning frameworks instead batch inputs as *rows* of x (N, in), and transposing both sides gives yᵀ = xᵀWᵀ + bᵀ.
# see: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py

def _t(A: Tensor1) -> Tensor1:
    """Return the (n, m) transpose of the (m, n) matrix A, copied into fresh row-major storage.

    We copy rather than return a view with swapped strides, since File 3's native _gemm reads
    storage as contiguous row-major and ignores strides.
    """
    (m, n) = A.shape
    At = borscht.full((n, m), 0.0)
    for i in range(m):
        for j in range(n):
            At.storage[j*At.stride[0] + i*At.stride[1]] = A.storage[i*A.stride[0] + j*A.stride[1]]
    return At
