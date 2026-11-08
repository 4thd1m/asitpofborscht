# Storage
# Allocator
# Device
# Kernel

def full(size, fill_value):
    raise NotImplementedError("borscht.tensor.full")

class Tensor:
    def __init__(self, shape: tuple[int, ...], fill_value):
        raise NotImplementedError("borscht.tensor.Tensor")

def mm(A: Tensor, B: Tensor) -> Tensor:
    raise NotImplementedError("borscht.tensor.mm")
