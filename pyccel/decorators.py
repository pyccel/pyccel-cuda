#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
This module contains all the provided decorator methods.
"""
from pyccel.ast.cudaext import cuda_mod
import warnings

__all__ = (
    'allow_negative_index',
    'bypass',
    'device',
    'elemental',
    'inline',
    'private',
    'pure',
    'stack_array',
    'sympy',
    'template',
    'types',
    'kernel'
)


def sympy(f):
    return f

def bypass(f):
    return f

def types(*args, results = None):
    """
    Specify the types passed to the function.

    Specify the types passed to the function.

    Parameters
    ----------
    *args : tuple of str or types
        The types of the arguments of the function.

    results : str or type, optional
        The return type of the function.

    Returns
    -------
    decorator
        The identity decorator which will not modify the function.
    """
    warnings.warn("The @types decorator will be removed in a future version of " +
                  "Pyccel. Please use type hints. The @template decorator can be " +
                  "used to specify multiple types. See the documentation at " +
                  "https://github.com/pyccel/pyccel/blob/devel/docs/quickstart.md#type-annotations"
                  "for examples.", FutureWarning)
    def identity(f):
        return f
    return identity

def template(name, types=()):
    """template decorator."""
    def identity(f):
        return f
    return identity

def pure(f):
    return f

def private(f):
    return f

def elemental(f):
    return f

def inline(f):
    """Indicates that function calls to this function should
    print the function body directly"""
    return f

def stack_array(f, *args):
    """
    Decorator indicates that all arrays mentioned as args should be stored
    on the stack.

    Parameters
    ----------
    f : Function
        The function to which the decorator is applied
    args : list of str
        A list containing the names of all arrays which should be stored on the stack
    """
    def identity(f):
        return f
    return identity

def allow_negative_index(f,*args):
    """
    Decorator indicates that all arrays mentioned as args can be accessed with
    negative indexes. As a result all non-constant indexing uses a modulo
    function. This can have negative results on the performance

    Parameters
    ----------
    f : Function
        The function to which the decorator is applied
    args : list of str
        A list containing the names of all arrays which can be accessed
        with non-constant negative indexes
    """
    def identity(f):
        return f
    return identity

def kernel(f):
    """
    Decorator for marking a Python function as a kernel.

    This class serves as a decorator to mark a Python function
    as a kernel function, typically used for GPU computations.
    This allows the function to be indexed with the number of blocks and threads.

    Parameters
    ----------
    f : function
        The function to which the decorator is applied.

    Returns
    -------
    KernelAccessor
        A class representing the kernel function.
    """
    class CudaThreadIndexing:
        """
        Class representing the CUDA thread indexing.

        Class representing the CUDA thread indexing.
        """
        def __init__(self, block_idx, thread_idx):
            self._block_idx = block_idx
            self._thread_idx = thread_idx

        def threadIdx(self, dim):
            """
            Get the thread index.

            Get the thread index.
            """
            return self._thread_idx

        def blockIdx(self, dim):
            """
            Get the block index.

            Get the block index.
            """
            return self._block_idx

        def blockDim(self, dim):
            """
            Get the block dimension.

            Get the block dimension.
            """
            return 0

    class KernelAccessor:
        """
        Class representing the kernel function.

        Class representing the kernel function.
        """
        def __init__(self, f):
            self._f = f
        def __getitem__(self, args):
            num_blocks, num_threads = args
            def internal_loop(*args, **kwargs):
                """
                The internal loop for kernel execution.

                The internal loop for kernel execution.
                """
                for b in range(num_blocks):
                    for t in range(num_threads):
                        self._f.__globals__['cuda'].CudaThreadIndexing = CudaThreadIndexing(b, t)
                        self._f(*args, **kwargs)
            return internal_loop

    return KernelAccessor(f)

def device(f):
    """
    Decorator for marking a function as a GPU device function.

    This decorator is used to mark a Python function as a GPU device function.

    Parameters
    ----------
    f : Function
        The function to be marked as a device.

    Returns
    -------
    f
        The function marked as a device.
    """
    return f
