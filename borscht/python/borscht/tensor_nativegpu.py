import math

# alternative: webgpu

# Storage
# Allocator
# Device
# Kernel

def full(size, fill_value):
    raise NotImplementedError("borscht.tensor_nativegpu.full")

class Tensor:
    def __init__(self, shape: tuple[int, ...], fill_value):
        raise NotImplementedError("borscht.tensor_nativegpu.Tensor")

def mm(A: Tensor, B: Tensor) -> Tensor:
    raise NotImplementedError("borscht.tensor_nativegpu.mm")
