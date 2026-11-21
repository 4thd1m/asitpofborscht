# Storage
# Allocator
# Device
# Kernel

# def full(size, fill_value):
#     raise NotImplementedError("borscht.tensor.full")

# class Tensor:
#     def __init__(self, shape: tuple[int, ...], fill_value):
#         raise NotImplementedError("borscht.tensor.Tensor")

# def mm(A: Tensor, B: Tensor) -> Tensor:
#     raise NotImplementedError("borscht.tensor.mm")

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