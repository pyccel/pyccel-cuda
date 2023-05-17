from .basic          import PyccelAstNode
from .builtins       import (PythonTuple,PythonList)

from .core           import Module, PyccelFunctionDef, Import

from .datatypes      import NativeInteger, NativeVoid

from .internals      import PyccelInternalFunction, get_final_precision

from .literals       import LiteralInteger
from .literals       import LiteralTrue, LiteralFalse
from .operators      import PyccelAdd, PyccelMul
from .variable       import (Variable, HomogeneousTupleVariable)

from .numpyext       import process_dtype, process_shape, NumpyNewArray

#==============================================================================
__all__ = (
    'CudaArray',
    'CudaBlockDim',
    'CudaBlockIdx',
    'CudaCopy',
    'CudaGrid',
    'CudaGridDim',
    'CudaInternalVar',
    'CudaMemCopy',
    'CudaNewArray',
    'CudaSynchronize',
    'CudaThreadIdx'
)

#==============================================================================
class CudaNewArray(NumpyNewArray):
    """ Class from which all Cuda functions which imply a call to Allocate
    inherit
    """
    __slots__ = ()

#==============================================================================

#==============================================================================
class CudaArray(CudaNewArray):
    """
    Represents a call to  cuda.array for code generation.

    arg : list, tuple, PythonList

    """
    __slots__ = ('_arg','_dtype','_precision','_shape','_rank','_order','_memory_location')
    _attribute_nodes = ('_arg',)
    name = 'array'

    def __init__(self, arg, dtype=None, order='C', memory_location='managed'):

        if not isinstance(arg, (PythonTuple, PythonList, Variable)):
            raise TypeError(f"Unknown type of  {type(arg)}.")

        is_homogeneous_tuple = isinstance(arg, (PythonTuple, PythonList, HomogeneousTupleVariable)) and arg.is_homogeneous
        is_array = isinstance(arg, Variable) and arg.is_ndarray

        # TODO: treat inhomogenous lists and tuples when they have mixed ordering
        if not (is_homogeneous_tuple or is_array):
            raise TypeError('we only accept homogeneous arguments')

        # Verify dtype and get precision
        if dtype is None:
            dtype = arg.dtype
            prec = get_final_precision(arg)
        else:
            dtype, prec = process_dtype(dtype)
        # ... Determine ordering
        order = str(order).strip("\'")

        shape = process_shape(False, arg.shape)
        rank  = len(shape)

        if rank < 2:
            order = None
        else:
            # ... Determine ordering
            order = str(order).strip("\'")

            if order not in ('K', 'A', 'C', 'F'):
                raise ValueError(f"Cannot recognize '{order}' order")

            # TODO [YG, 18.02.2020]: set correct order based on input array
            if order in ('K', 'A'):
                order = 'C'
            # ...
        #Verify memory location
        if memory_location not in ('host', 'device', 'managed'):
            raise ValueError("memory_location must be 'host', 'device' or 'managed'")
        self._arg   = arg
        self._shape = shape
        self._rank  = rank
        self._dtype = dtype
        self._order = order
        self._precision = prec
        self._memory_location = memory_location
        super().__init__()

    def __str__(self):
        return str(self.arg)

    @property
    def arg(self):
        return self._arg
    @property
    def memory_location(self):
        return self._memory_location

class CudaSharedArray(CudaNewArray):
    """
    Represents a call to  cuda.shared.array for code generation.

    arg : list, tuple, PythonList

    """

    __slots__ = ('_dtype','_precision','_shape','_rank','_order', '_memory_location')
    name = 'array'

    def __init__(self, shape, dtype, order='C'):

        # Convert shape to PythonTuple
        self._shape = process_shape(False, shape)

        # Verify dtype and get precision
        self._dtype, self._precision = process_dtype(dtype)

        self._rank  = len(self._shape)
        self._order = self._order = NumpyNewArray._process_order(self._rank, order)
        self._memory_location = 'shared'
        super().__init__()

    @property
    def memory_location(self):
        return self._memory_location

class CudaSynchronize(PyccelInternalFunction):
    "Represents a call to  Cuda.deviceSynchronize for code generation."

    __slots__ = ()
    _attribute_nodes = ()
    _shape     = None
    _rank      = 0
    _dtype     = NativeVoid()
    _precision = None
    _order     = None
    def __init__(self):
        super().__init__()

class CudaSyncthreads(PyccelInternalFunction):
    "Represents a call to  __syncthreads for code generation."

    __slots__ = ()
    _attribute_nodes = ()
    _shape     = None
    _rank      = 0
    _dtype     = NativeVoid()
    _precision = None
    _order     = None
    def __init__(self):
        super().__init__()

class CudaInternalVar(PyccelAstNode):
    """
    Represents a General Class For Cuda internal Variables Used To locate Thread In the GPU architecture"

    Parameters
    ----------
    dim : NativeInteger
        Represent the dimension where we want to locate our thread.

    """
    __slots__ = ('_dim','_dtype', '_precision')
    _attribute_nodes = ('_dim',)
    _shape     = None
    _rank      = 0
    _order     = None

    def __init__(self, dim=None):
        
        if isinstance(dim, int):
            dim = LiteralInteger(dim)
        if not isinstance(dim, LiteralInteger):
            raise TypeError("dimension need to be an integer")
        if dim not in (0, 1, 2):
            raise ValueError("dimension need to be 0, 1 or 2")
        #...
        self._dim       = dim
        self._dtype     = dim.dtype
        self._precision = dim.precision
        super().__init__()

    @property
    def dim(self):
        return self._dim


class CudaCopy(CudaNewArray):
    """
    Represents a call to  cuda.copy for code generation.

    Parameters
    ----------
    arg : Variable

    memory_location : str
        'host'   the newly created array is allocated on host.
        'device' the newly created array is allocated on device.
    
    is_async: bool
        Indicates whether the copy is asynchronous or not [Default value: False]

    """
    __slots__ = ('_arg','_dtype','_precision','_shape','_rank','_order','_memory_location', '_is_async')

    def __init__(self, arg, memory_location, is_async=False):
        
        if not isinstance(arg, Variable):
            raise TypeError(f"unknown type of  {type(arg)}.")
        
        # Verify the memory_location of src
        if arg.memory_location not in ('device', 'host', 'managed'):
            raise ValueError("The direction of the copy should be from 'host' or 'device'")

        # Verify the memory_location of dst
        if memory_location not in ('device', 'host', 'managed'):
            raise ValueError("The direction of the copy should be to 'host' or 'device'")
        
        # verify the type of is_async
        if not isinstance(is_async, (LiteralTrue, LiteralFalse, bool)):
            raise TypeError('is_async must be boolean')
        
        self._arg             = arg
        self._shape           = arg.shape
        self._rank            = arg.rank
        self._dtype           = arg.dtype
        self._order           = arg.order
        self._precision       = arg.precision
        self._memory_location = memory_location
        self._is_async        = is_async
        super().__init__()
    
    @property
    def arg(self):
        return self._arg

    @property
    def memory_location(self):
        return self._memory_location

    @property
    def is_async(self):
        return self._is_async

class CudaThreadIdx(CudaInternalVar):
    __slots__ = ()
    pass
class CudaBlockDim(CudaInternalVar):
    __slots__ = ()
    pass
class CudaBlockIdx(CudaInternalVar):
    __slots__ = ()
    pass
class CudaGridDim(CudaInternalVar):
    __slots__ = ()
    pass

class CudaGrid(PyccelAstNode)               :
    """
    CudaGrid locates a thread in the GPU architecture using `CudaThreadIdx`, `CudaBlockDim`, `CudaBlockIdx`
    to calculate the exact index of the thread automatically.

    Parameters
    ----------
    dim : NativeInteger
        Represent the dimension where we want to locate our thread.

    """
    __slots__ = ()
    _attribute_nodes = ()
    def __new__(cls, dim=0):
        if not isinstance(dim, LiteralInteger):
            raise TypeError("dimension need to be an integer")
        if dim not in (0, 1, 2):
            raise ValueError("dimension need to be 0, 1 or 2")
        expr = [PyccelAdd(PyccelMul(CudaBlockIdx(d), CudaBlockDim(d)), CudaThreadIdx(d))\
                for d in range(dim.python_value + 1)]
        if dim == 0:
            return expr[0]
        return PythonTuple(*expr)

cuda_funcs = {
    'array'             : PyccelFunctionDef('array'             , CudaArray),
    'copy'              : PyccelFunctionDef('copy'              , CudaCopy),
    'synchronize'       : PyccelFunctionDef('synchronize'       , CudaSynchronize),
    'syncthreads'       : PyccelFunctionDef('syncthreads'       , CudaSyncthreads),
    'threadIdx'         : PyccelFunctionDef('threadIdx'         , CudaThreadIdx),
    'blockDim'          : PyccelFunctionDef('blockDim'          , CudaBlockDim),
    'blockIdx'          : PyccelFunctionDef('blockIdx'          , CudaBlockIdx),
    'gridDim'           : PyccelFunctionDef('gridDim'           , CudaGridDim),
    'grid'              : PyccelFunctionDef('grid'              , CudaGrid)
}

cuda_Internal_Var = {
    'CudaThreadIdx' : 'threadIdx',
    'CudaBlockDim'  : 'blockDim',
    'CudaBlockIdx'  : 'blockIdx',
    'CudaGridDim'   : 'gridDim'
}

# cuda_sharedmemory = {
#     'array'             : PyccelFunctionDef('array'             , CudaSharedArray),
# }

cuda_sharedmemory = Module('shared', (),
    [ PyccelFunctionDef('array' , CudaSharedArray)])

cuda_mod = Module('cuda',
    variables = [],
    funcs = cuda_funcs.values(),
    imports = [
        Import('shared', cuda_sharedmemory),
        ])