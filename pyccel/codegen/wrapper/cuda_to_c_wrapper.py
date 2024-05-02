# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CudaToPythonWrapper
which creates an interface exposing Cuda code to C.
"""

from pyccel.ast.bind_c import BindCModule
from pyccel.errors.errors import Errors
from pyccel.ast.bind_c import BindCVariable
from .wrapper import Wrapper

errors = Errors()
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
        if expr.interfaces:
            errors.report("Interface wrapping is not yet supported for Cuda",
                      severity='warning', symbol=expr)
        if expr.classes:
            errors.report("Class wrapping is not yet supported for Cuda",
                      severity='warning', symbol=expr)

        variables = [self._wrap(v) for v in expr.variables]

        return BindCModule(expr.name, variables, expr.funcs,
                scope = expr.scope,
                original_module=expr)

    def _wrap_Variable(self, expr):
        return expr.clone(expr.name, new_class = BindCVariable)