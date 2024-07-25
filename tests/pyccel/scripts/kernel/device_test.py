# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import device, kernel

@device
def device_call():
    print("Hello from device")

@kernel
def kernel_call():
    device_call()

def f():
    kernel_call[1,1]()
    from pyccel import cuda
    cuda.synchronize()

if __name__ == '__main__':
    f()
