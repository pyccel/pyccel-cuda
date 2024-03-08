
from .internals      import PyccelInternalFunction

from .datatypes import (NativeVoid)
from .core           import Module, PyccelFunctionDef
__all__ = (
    'CudaSynchronize',
)

class CudaSynchronize(PyccelInternalFunction):
    """
    Represents a call to  Cuda.deviceSynchronize for code generation
    
    This class serves as a representation of a synchronization call to the CUDA    
    """
    __slots__ = ()
    _attribute_nodes = ()
    _shape     = None
    _rank      = 0
    _dtype     = NativeVoid()
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