from pyccel import cuda
from pyccel.decorators import kernel

@kernel
def kernel_call(a : 'int[:]', size : 'int'):
    i =  cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
    if(i < size):
        a[i] = 1
        print(a[i])

def f():
    size = 10
    x = cuda.device_empty(10)
    kernel_call[1,10](x , size)

if __name__ == "__main__":
    f()