# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Handles name clash problems in Cuda
"""
from .languagenameclashchecker import LanguageNameClashChecker

class CudaNameClashChecker(LanguageNameClashChecker):
    """
    Class containing functions to help avoid problematic names in Cuda.

    A class which provides functionalities to check or propose variable names and
    verify that they do not cause name clashes. Name clashes may be due to
    new variables, or due to the use of reserved keywords.
    """

    # Keywords as mentioned on https://en.cppreference.com/w/c/keyword
    keywords = set(['isign', 'fsign', 'csign', 'auto', 'break', 'case', 'char', 'const',
        'continue', 'default', 'do', 'double', 'else', 'enum',
        'extern', 'float', 'for', 'goto', 'if', 'inline', 'int',
        'long', 'register', 'restrict', 'return', 'short', 'signed',
        'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
        'unsigned', 'void', 'volatile', 'whie', '_Alignas',
        '_Alignof', '_Atomic', '_Bool', '_Complex', 'Decimal128',
        '_Decimal32', '_Decimal64', '_Generic', '_Imaginary',
        '_Noreturn', '_Static_assert', '_Thread_local', 't_ndarray',
        'array_create', 'new_slice', 'array_slicing', 'alias_assign',
        'transpose_alias_assign', 'array_fill', 't_slice',
        'GET_INDEX_EXP1', 'GET_INDEX_EXP2', 'GET_INDEX_EXP2',
        'GET_INDEX_EXP3', 'GET_INDEX_EXP4', 'GET_INDEX_EXP5',
        'GET_INDEX_EXP6', 'GET_INDEX_EXP7', 'GET_INDEX_EXP8',
        'GET_INDEX_EXP9', 'GET_INDEX_EXP10', 'GET_INDEX_EXP11',
        'GET_INDEX_EXP12', 'GET_INDEX_EXP13', 'GET_INDEX_EXP14',
        'GET_INDEX_EXP15', 'NUM_ARGS_H1', 'NUM_ARGS',
        'GET_INDEX_FUNC_H2', 'GET_INDEX_FUNC', 'GET_INDEX',
        'INDEX', 'GET_ELEMENT', 'free_array', 'free_pointer',
        'get_index', 'numpy_to_ndarray_strides',
        'numpy_to_ndarray_shape', 'get_size', 'order_f', 'order_c', 'array_copy_data'
        '__global__', '__device__', '__host__','__constant__', '__shared__',
        '__managed__','threadIdx', 'blockIdx', 'blockDim', 'gridDim',
        'warpSize', 'cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaMemset',
        'cudaMallocHost', 'cudaFreeHost', 'cudaMallocPitch',
        'cudaMallocArray', 'cudaFreeArray', 'cudaHostAlloc',
        'cudaHostRegister', 'cudaHostUnregister', 'cudaHostGetDevicePointer',
        'cudaHostGetFlags', 'cudaDeviceSynchronize', 'cudaDeviceReset',
        'cudaSetDevice', 'cudaGetDeviceCount', 'cudaGetDeviceProperties',
        'cudaChooseDevice', 'cudaSetDeviceFlags', 'cudaGetDevice',
        'cudaStreamCreate', 'cudaStreamDestroy', 'cudaStreamSynchronize',
        'cudaStreamWaitEvent', 'cudaEventCreate', 'cudaEventDestroy', 'cudaEventRecord',
        'cudaEventSynchronize', 'cudaEventElapsedTime', 'cuInit', 'cuDeviceGet',
        'cuDeviceGetCount', 'cuDeviceGetName',
        'cuDeviceComputeCapability', 'cuCtxCreate', 'cuCtxDestroy',
        'cuCtxSynchronize', 'cuModuleLoad', 'cuModuleUnload',
        'cuModuleGetFunction', 'cuModuleGetGlobal', 'cuModuleGetTexRef',
        'cuMemAlloc', 'cuMemFree', 'cuMemcpyHtoD', 'cuMemcpyDtoH',
        'cuMemcpyDtoD', 'cuMemcpyHtoDAsync', 'cuMemcpyDtoHAsync',
        'cuMemcpyDtoDAsync', 'cuMemsetD8', 'cuMemsetD16', 'cuMemsetD32',
        'cuMemsetD2D8', 'cuMemsetD2D16', 'cuMemsetD2D32', 'cuParamSetSize',
        'cuParamSeti', 'cuParamSetf', 'cuParamSetv', 'cuLaunch', 'cuLaunchGrid',
        'cuLaunchGridAsync', 'cuEventCreate', 'cuEventRecord', 'cuEventQuery',
        'cuEventSynchronize', 'cuEventDestroy', 'cuEventElapsedTime',
        'cuStreamCreate', 'cuStreamQuery', 'cuStreamSynchronize',
        'cuStreamDestroy', 'cuFuncSetBlockShape', 'cuFuncSetSharedSize',
        'cuFuncGetAttribute', 'cuTexRefCreate', 'cuTexRefDestroy',
        'cuTexRefSetArray', 'cuTexRefSetAddress', 'cuTexRefSetAddress2D',
        'cuTexRefSetFormat', 'cuTexRefSetAddressMode', 'cuTexRefSetFilterMode',
        'cuTexRefSetFlags', 'cuTexRefGetAddress', 'cuTexRefGetArray',
        'cuTexRefGetAddressMode', 'cuTexRefGetFilterMode', 'cuTexRefGetFormat',
        'cuTexRefGetFlags', 'cuLaunchKernel', 'cuOccupancyMaxActiveBlocksPerMultiprocessor',
        'cuOccupancyMaxPotentialBlockSize', 'cuOccupancyMaxPotentialBlockSizeWithFlags'
    ])

    def has_clash(self, name, symbols):
        """
        Indicate whether the proposed name causes any clashes.

        Checks if a suggested name conflicts with predefined
        keywords or specified symbols,returning true for a clash.
        This method is crucial for maintaining namespace integrity and
        preventing naming conflicts in code generation processes.

        Parameters
        ----------
        name : str
            The suggested name.
        symbols : set
            Symbols which should be considered as collisions.

        Returns
        -------
        bool
            True if the name is a collision.
            False if the name is collision free.
        """
        return any(name == k for k in self.keywords) or \
               any(name == s for s in symbols)

    def get_collisionless_name(self, name, symbols):
        """
        Get a valid name which doesn't collision with symbols or Cuda keywords.

        Find a new name based on the suggested name which will not cause
        conflicts with Cuda keywords, does not appear in the provided symbols,
        and is a valid name in Cuda code.

        Parameters
        ----------
        name : str
            The suggested name.
        symbols : set
            Symbols which should be considered as collisions.

        Returns
        -------
        str
            A new name which is collision free.
        """
        if len(name)>4 and all(name[i] == '_' for i in (0,1,-1,-2)):
            # Ignore magic methods
            return name
        if name[0] == '_':
            name = 'private'+name
        return self._get_collisionless_name(name, symbols)
