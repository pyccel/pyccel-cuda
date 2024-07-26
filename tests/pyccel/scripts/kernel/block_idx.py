# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import kernel
from pyccel            import cuda

@kernel
def print_block():
    print(cuda.blockIdx(0))

def f():
    print_block[5,5]()
    cuda.synchronize()

if __name__ == '__main__':
    f()

