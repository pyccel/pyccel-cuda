#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
CUDA Extension Module
Provides CUDA functionality for code generation.
"""
from .internals      import PyccelFunction
from .literals       import Nil

from .datatypes      import VoidType
from .core           import Module, PyccelFunctionDef
from .numpyext       import process_dtype, process_shape
from .cudatypes      import CudaArrayType
from .numpytypes     import NumpyInt32Type



__all__ = (
    'CudaSynchronize',
    'CudaNewarray',
    'CudaFull',
    'CudaHostEmpty'
)

class CudaNewarray(PyccelFunction):
    """
    Superclass for nodes representing Cuda array allocation functions.

    Class from which all nodes representing a Cuda function which implies a call
    to `Allocate` should inherit.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments of the superclass PyccelFunction.

    class_type : NumpyNDArrayType
        The type of the new array.

    init_dtype : PythonType, PyccelFunctionDef, LiteralString, str
        The actual dtype passed to the Cuda function.

    memory_location : str
        The memory location of the new array ('host' or 'device').
    """
    __slots__ = ('_class_type', '_init_dtype', '_memory_location')
    name = 'newarray'

    @property
    def init_dtype(self):
        """
        The dtype provided to the function when it was initialised in Python.

        The dtype provided to the function when it was initialised in Python.
        If no dtype was provided then this should equal `None`.
        """
        return self._init_dtype

    def __init__(self, *args ,class_type, init_dtype, memory_location):
        self._class_type = class_type
        self._init_dtype = init_dtype
        self._memory_location = memory_location

        super().__init__(*args)

class CudaFull(CudaNewarray):
    """
    Represents a call to `cuda.full` for code generation.

    Represents a call to the Cuda function `full` which creates an array
    of a specified size and shape filled with a specified value.

    Parameters
    ----------
    shape : TypedAstNode
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        For a 1D array this is either a `LiteralInteger` or an expression.
        For a ND array this is a `TypedAstNode` with the class type HomogeneousTupleType.

    fill_value : TypedAstNode
        Fill value.

    dtype : PythonType, PyccelFunctionDef, LiteralString, str, optional
        Datatype for the constructed array.
        If `None` the dtype of the fill value is used.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

    memory_location : str
        The memory location of the new array ('host' or 'device').
    """
    __slots__ = ('_fill_value','_shape')
    name = 'full'

    def __init__(self, shape, fill_value, dtype, order, memory_location):
        shape = process_shape(False, shape)
        init_dtype = dtype
        if(dtype is None):
            dtype = fill_value.dtype

        dtype = process_dtype(dtype)

        self._shape = shape
        rank = len(self._shape)
        class_type = CudaArrayType(dtype, rank, order, memory_location)
        super().__init__(fill_value, class_type = class_type, init_dtype = init_dtype, memory_location = memory_location)


class CudaHostEmpty(CudaFull):
    """
    Represents a call to  Cuda.host_empty for code generation.

    A class representing a call to the Cuda `host_empty` function.

    Parameters
    ----------
    shape : tuple of int , int
        The shape of the new array.

    dtype : PythonType, LiteralString, str
        The actual dtype passed to the NumPy function.

    order : str , LiteralString
        The order passed to the function defoulting to 'C'.
    """
    __slots__ = ()
    name = 'empty'
    def __init__(self, shape, dtype='float', order='C'):
        memory_location = 'host'
        super().__init__(shape, Nil(), dtype, order , memory_location)
    @property
    def fill_value(self):
        """
        The value with which the array will be filled on initialisation.

        The value with which the array will be filled on initialisation.
        """
        return None
class CudaDeviceEmpty(CudaFull):
    """
    Represents a call to  Cuda.device_empty for code generation.

    A class representing a call to the Cuda `device_empty` function.

    Parameters
    ----------
    shape : tuple of int , int
        The shape of the new array.

    dtype : PythonType, LiteralString, str
        The actual dtype passed to the NumPy function.

    order : str , LiteralString
        The order passed to the function defoulting to 'C'.
    """
    __slots__ = ()
    name = 'empty'
    def __init__(self, shape, dtype='float', order='C'):
        memory_location = 'device'
        super().__init__(shape, Nil(), dtype, order , memory_location)
    @property
    def fill_value(self):
        """
        The value with which the array will be filled on initialisation.

        The value with which the array will be filled on initialisation.
        """
        return None
class CudaDimFunction(PyccelFunction):
    """
    Represents a call to a CUDA dimension-related function for code generation.

    This class serves as a representation of a CUDA dimension-related function call.
    """
    __slots__ = ('_dim',)
    _attribute_nodes = ('_dim',)
    _shape = None
    _class_type = NumpyInt32Type()

    def __init__(self, dim=0):
        self._dim = dim
        super().__init__()

    @property
    def dim(self):
        return self._dim

class threadIdx(CudaDimFunction):
    """
    Represents a call to Cuda.threadIdx for code generation.

    This class serves as a representation of a thread call to the CUDA.
    """
    def __init__(self, dim=0):
        super().__init__(dim)

class blockIdx(CudaDimFunction):
    """
    Represents a call to Cuda.blockIdx for code generation.

    This class serves as a representation of a block call to the CUDA.
    """
    def __init__(self, dim=0):
        super().__init__(dim)

class blockDim(CudaDimFunction):
    """
    Represents a call to Cuda.blockDim for code generation.

    This class serves as a representation of a block dimension call to the CUDA.
    """
    def __init__(self, dim=0):
        super().__init__(dim)

class CudaSynchronize(PyccelFunction):
    """
    Represents a call to Cuda.synchronize for code generation.

    This class serves as a representation of the Cuda.synchronize method.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _shape     = None
    _class_type = VoidType()
    def __init__(self):
        super().__init__()

cuda_funcs = {
    'synchronize'       : PyccelFunctionDef('synchronize' , CudaSynchronize),
    'full'              : PyccelFunctionDef('full' , CudaFull),
    'host_empty'        : PyccelFunctionDef('host_empty' , CudaHostEmpty),
    'device_empty'      : PyccelFunctionDef('device_empty' , CudaDeviceEmpty),
    'threadIdx'         : PyccelFunctionDef('threadIdx'   , threadIdx),
    'blockIdx'          : PyccelFunctionDef('blockIdx'    , blockIdx),
    'blockDim'          : PyccelFunctionDef('blockDim'    , blockDim)
}

cuda_mod = Module('cuda',
    variables=[],
    funcs=cuda_funcs.values(),
    imports=[]
)

