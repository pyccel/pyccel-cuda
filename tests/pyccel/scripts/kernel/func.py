# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import kernel
# from pyccel import cuda

@kernel
def say_hello():
    print("Hello")

def f():
    say_hello[1,1]()
    # cuda.synchronize()

