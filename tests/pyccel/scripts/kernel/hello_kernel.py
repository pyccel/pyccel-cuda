# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import kernel

@kernel
def say_hello(its_morning : bool):
    if(its_morning):
        print("Hello and Good morning")
    else:
        print("Hello and Good afternoon")

def f():
    its_morning = True
    say_hello[5,5](its_morning)
    from pyccel import cuda
    cuda.synchronize()

if __name__ == '__main__':
    f()

