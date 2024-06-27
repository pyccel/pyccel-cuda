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

from .datatypes      import VoidType
from .core           import Module, PyccelFunctionDef
from .numpytypes     import NumpyInt32Type

__all__ = (
    'CudaSynchronize',
)

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

cuda_funcs = {
    'synchronize'       : PyccelFunctionDef('synchronize' , CudaSynchronize),
    'threadIdx'         : PyccelFunctionDef('threadIdx'   , threadIdx),
    'blockIdx'          : PyccelFunctionDef('blockIdx'    , blockIdx),
    'blockDim'          : PyccelFunctionDef('blockDim'    , blockDim)
}

cuda_mod = Module('cuda',
    variables=[],
    funcs=cuda_funcs.values(),
    imports=[]
)

