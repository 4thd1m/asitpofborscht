from .borscht import *

# mod nn (level 2)
# mod optim (level 2)
# mod linalg (level 1)

# mod kernels (level 0) quack,fa(CuTe)
# mod tensor (level -1) CuTe?

from . import nn        # level 2
from . import optim     # level 2
from . import linalg    # level 1
from .tensor import Tensor   # level 0
from . import kernels   # level -1


__doc__ = borscht.__doc__
if hasattr(borscht, "__all__"):
    __all__ = borscht.__all__
