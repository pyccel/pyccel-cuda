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

from .datatypes import (VoidType)
from .core           import Module, PyccelFunctionDef

__all__ = (
    'CudaSynchronize',
)

class CudaSynchronize(PyccelFunction):
    """
    Represents a call to  Cuda.deviceSynchronize for code generation.
    
    This class serves as a representation of a synchronization call to the CUDA.   
    """
    __slots__ = ()
    _attribute_nodes = ()
    _shape     = None
    _rank      = 0
    _class_type = VoidType()
    _precision = None
    _order     = None
    def __init__(self):
        super().__init__()

cuda_funcs = {
    'synchronize'       : PyccelFunctionDef('synchronize' , CudaSynchronize),
}

cuda_mod = Module('cuda',
    variables=[],
    funcs=cuda_funcs.values(),
    imports=[]
)

