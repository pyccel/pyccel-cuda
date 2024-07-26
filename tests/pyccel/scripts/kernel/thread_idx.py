from pyccel.decorators import kernel
from pyccel            import cuda

@kernel
def print_block():
    print(cuda.threadIdx(0))

def f():
    print_block[5,5]()
    cuda.synchronize()

if __name__ == '__main__':
    f()