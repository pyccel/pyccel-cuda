# pylint: disable=missing-function-docstring, missing-module-docstring

def array_int32_1d_scalar_add( x:'int32[:]', a:'int32' ):
    x[:] += a

def array_int32_2d_C_scalar_add( x:'int32[:,:]', a:'int32' ):
    x[:,:] += a

def array_int32_2d_F_add( x:'int32[:,:](order=F)', y:'int32[:,:](order=F)' ):
    x[:,:] += y

def array_int_1d_scalar_add( x:'int[:]', a:'int' ):
    x[:] += a

def array_float_1d_scalar_add( x:'float[:]', a:'float' ):
    x[:] += a

def array_float_2d_F_scalar_add( x:'float[:,:](order=F)', a:'float' ):
    x[:,:] += a

def array_float_2d_F_add( x:'float[:,:](order=F)', y:'float[:,:](order=F)'  ):
    x[:,:] += y

def array_int32_2d_F_complex_3d_expr( x:'int32[:,:](order=F)', y:'int32[:,:](order=F)' ):
    from numpy import full, int32
    z = full((2,3),5,order='F', dtype=int32)
    x[:] = (x // y) * x + z

def array_float_1d_complex_3d_expr( x:'float[:]', y:'float[:]' ):
    from numpy import full
    z = full(3,5)
    x[:] = (x // y) * x + z

def fib(n: int) -> int:
    if n<=1:
        return 0
    elif n==2:
        return 1
    else:
        return fib(n-1) + fib(n-2)

