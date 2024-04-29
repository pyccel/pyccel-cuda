# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CudaToPythonWrapper
which creates an interface exposing Cuda code to C.
"""

from pyccel.codegen.wrapper.c_to_python_wrapper import CToPythonWrapper
from pyccel.parser.scope import Scope
from pyccel.ast.core import Module
from .wrapper import Wrapper
from pyccel.ast.bind_c import BindCModule
from pyccel.ast.core import Import
from pyccel.ast.core import Module
from pyccel.parser.scope import Scope
from .wrapper import Wrapper
cwrapper_ndarray_imports = [Import('cwrapper_ndarrays', Module('cwrapper_ndarrays', (), ()))]

class CudaToCWrapper(Wrapper):
    """
    Class for creating a wrapper exposing Fortran code to C.

    A class which provides all necessary functions for wrapping different AST
    objects such that the resulting AST is C-compatible. This new AST is
    printed as an intermediary layer.
    """
    def __init__(self):
        self._wrapper_names_dict = {}
        super().__init__()


    def _wrap_Module(self, expr):
        """
        Create a Module which is compatible with C.

        Create a Module which provides an interface between C and the
        Module described by expr

        Parameters
        ----------
        expr : pyccel.ast.core.Module
            The module to be wrapped.

        Returns
        -------
        pyccel.ast.core.Module
            The C-compatible module.
        """
        # Define scope
        scope = expr.scope
        mod_scope = Scope(used_symbols = scope.local_used_symbols.copy(), original_symbols = scope.python_names.copy())
        self.scope = mod_scope

        name = mod_scope.get_new_name(f'bind_c_{expr.name.target}')
        self.exit_scope()
        return BindCModule(name, expr.variables, expr.funcs,
                scope = mod_scope,
                original_module=expr)
    def _wrap_FunctionDef(self, expr):
        return expr
