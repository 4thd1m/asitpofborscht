import math
import random
from typing import override

import borscht
from ..tensor_managed import Tensor1
from . import functional as F


class Module:
    """The base class of every layer: a callable bag of parameters.

    torch's Module also tracks buffers, hooks, train/eval mode, devices, and state dicts.
    Its essence is just two things: calling it runs forward(), and parameters() finds every
    Tensor it owns (recursing into submodules) so that an optimizer can update them.
    """
    def __call__(self, x: Tensor1) -> Tensor1:
        return self.forward(x)  # layer(x) is sugar for layer.forward(x)

    def forward(self, x: Tensor1) -> Tensor1:
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def parameters(self) -> list[Tensor1]:
        params: list[Tensor1] = []
        for v in vars(self).values():                                  # torch registers parameters in __setattr__ and marks them with nn.Parameter
            if isinstance(v, Tensor1): params.append(v)                # until autograd (File 7) gives us requires_grad,
            elif isinstance(v, Module): params.extend(v.parameters())  # we treat every Tensor attribute as a parameter
        return params


class Linear(Module):
    """Apply the affine map y = xWᵀ + b, i.e. out_features neurons, each taking a dot product with x.

    Args:
        in_features: Size of each input row.
        out_features: Size of each output row.
        bias: If False, the layer learns no additive bias b. Defaults to True.

    Attributes:
        weight: (out_features, in_features) Tensor W, initialized from U(-1/√in_features, 1/√in_features).
        bias: (out_features,) Tensor b, initialized from the same distribution, or None.

    Examples:
        >>> from borscht.tensor_managed import full
        >>> layer = Linear(3, 2)
        >>> layer
        Linear(in_features=3, out_features=2, bias=True)
        >>> [p.shape for p in layer.parameters()]
        [(2, 3), (2,)]
        >>> layer.weight.storage = [1.0, 2.0, 3.0,
        ...                         4.0, 5.0, 6.0]
        >>> layer.bias = full((2,), 0.5)
        >>> layer(full((4, 3), 1.0))
        tensor([[6.5000, 15.5000],
                [6.5000, 15.5000],
                [6.5000, 15.5000],
                [6.5000, 15.5000]])
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        k = 1 / math.sqrt(in_features)                                                  # torch initializes with kaiming_uniform_(weight, a=√5), whose bound
        self.weight: Tensor1 = _uniform((out_features, in_features), -k, k)             # gain·√(3/in) = √(2/(1+5))·√(3/in) simplifies to 1/√in. scaling by 1/√in
        self.bias: Tensor1 | None = _uniform((out_features,), -k, k) if bias else None  # keeps each output's variance from growing with the number of inputs summed

    @override
    def forward(self, x: Tensor1) -> Tensor1:
        return F.linear(x, self.weight, self.bias)  # the module only *holds* W and b, while the math lives in the stateless F.linear

    @override
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


def _uniform(shape: tuple[int, ...], low: float, high: float) -> Tensor1:
    A = borscht.full(shape, 0.0)
    A.storage = [random.uniform(low, high) for _ in A.storage]
    return A
