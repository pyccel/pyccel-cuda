# pylint: disable=missing-function-docstring, missing-module-docstring
#==============================================================================

from pyccel.decorators import kernel
from pyccel import cuda

# This kernel function increments the value of a in-place
@kernel
def increment_value_inplace(a : int):
    a += 1

# ...
@kernel
def say_hello():
    print("Hello")

# ...
def f():
    say_hello[1,1]()
    cuda.synchronize()



