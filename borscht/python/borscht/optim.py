from typing import overload

class Optimizer():
    def __init__(self, params):
      raise NotImplementedError("")

    @overload
    def step(self): ...

class SGD(Optimizer):
    def __init__(self, params):
      raise NotImplementedError("")

    def step(self):
      raise NotImplementedError("")