#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
CUDA Module
This module provides a collection of classes and utilities for CUDA programming.
"""
from pyccel.ast.core import FunctionCall

__all__ = (
    'KernelCall',
)

class KernelCall(FunctionCall):
    """
    Represents a kernel function call in the code.
    
     The class serves as a representation of a kernel
     function call within the codebase.

    Parameters
    ----------
    func : FunctionDef
        The definition of the function being called.

    args : tuple
        The arguments being passed to the function.

    numBlocks : NativeInteger
        The number of blocks.

    tpblock : NativeInteger
        The number of threads per block.

    current_function : FunctionDef, default: None
        The function where the call takes place.
    """
    __slots__ = ('_numBlocks','_tpblock','_func', '_args')
    _attribute_nodes = (*FunctionCall._attribute_nodes, '_numBlocks', '_tpblock')
    def __init__(self, func, args, numBlocks, tpblock,current_function=None):
        self._numBlocks = numBlocks
        self._tpblock = tpblock
        super().__init__(func, args, current_function)

    @property
    def numBlocks(self):
        """
        The number of blocks in the kernel being called.
    
        The number of blocks in the kernel being called.
        """
        return self._numBlocks

    @property
    def tpblock(self):
        """
        The number of threads per block.

        Launch configuration of kernel call.
        """
        return self._tpblock

