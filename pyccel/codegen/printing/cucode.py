# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Provide tools for generating and handling CUDA code.
This module is designed to interface Pyccel's Abstract Syntax Tree (AST) with CUDA,
enabling the direct translation of high-level Pyccel expressions into CUDA code.
"""
from pyccel.codegen.printing.ccode import CCodePrinter, c_library_headers

from pyccel.ast.core        import Import, Module
from pyccel.ast.core      import FunctionAddress

from pyccel.ast.datatypes import VoidType, PythonNativeInt

from pyccel.errors.errors   import Errors

from pyccel.ast.literals  import Nil

errors = Errors()

__all__ = ["CudaCodePrinter"]

class CudaCodePrinter(CCodePrinter):
    """
    Print code in CUDA format.

    This printer converts Pyccel's Abstract Syntax Tree (AST) into strings of CUDA code.
    Navigation through this file utilizes _print_X functions,
    as is common with all printers.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    prefix_module : str
            A prefix to be added to the name of the module.
    """
    language = "cuda"

    def __init__(self, filename, prefix_module = None):

        errors.set_target(filename)

        super().__init__(filename)

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        self._current_module = expr.name
        body = ''.join(self._print(i) for i in expr.body)

        global_variables = ''.join(self._print(d) for d in expr.declarations)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [Import(expr.name, Module(expr.name,(),())), *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        code = f'{imports}\n\
                 {global_variables}\n\
                 {body}\n'

        self.exit_scope()
        return code

    def function_signature(self, expr, print_arg_names = True):
        """
        Get the Cuda representation of the function signature.

        Extract from the function definition `expr` all the
        information (name, input, output) needed to create the
        function signature and return a string describing the
        function.
        This is not a declaration as the signature does not end
        with a semi-colon.

        Parameters
        ----------
        expr : FunctionDef
            The function definition for which a signature is needed.

        print_arg_names : bool, default : True
            Indicates whether argument names should be printed.

        Returns
        -------
        str
            Signature of the function.
        """
        arg_vars = [a.var for a in expr.arguments]
        result_vars = [r.var for r in expr.results if not r.is_argument]
        n_results = len(result_vars)

        if n_results == 1:
            ret_type = self.get_declare_type(result_vars[0])
        elif n_results > 1:
            ret_type = self.find_in_dtype_registry(PythonNativeInt())
            arg_vars.extend(result_vars)
            self._additional_args.append(result_vars) # Ensure correct result for is_c_pointer
        else:
            ret_type = self.find_in_dtype_registry(VoidType())

        name = expr.name
        if not arg_vars:
            arg_code = 'void'
        else:
            def get_arg_declaration(var):
                """ Get the code which declares the argument variable.
                """
                code = "const " * var.is_const
                code += self.get_declare_type(var)
                if print_arg_names:
                    code += ' ' + var.name
                return code

            arg_code_list = [self.function_signature(var, False) if isinstance(var, FunctionAddress)
                                else get_arg_declaration(var) for var in arg_vars]
            arg_code = ', '.join(arg_code_list)

        if self._additional_args :
            self._additional_args.pop()

        static = 'static ' if expr.is_static else ''
        cuda_decorater = ""
        if('kernel' in expr.decorators):
            cuda_decorater = "__global__"
        return f'{static} {cuda_decorater} {ret_type} {name}({arg_code})'

    def _print_KernelCall(self, expr):
        func = expr.funcdef
        args = []
        for a in expr.args:
            arg_val = a.value or Nil()
            args.append(arg_val)

        args += self._temporary_args
        self._temporary_args = []
        args = ', '.join([f'{self._print(a)}' for a in args])
        return f"{func.name}<<<{expr.numBlocks}, {expr.tpblock}>>>({args});\n"

    def _print_CudaSynchronize(self, expr):
        return 'cudaDeviceSynchronize();\n'
    def _print_ModuleHeader(self, expr):
        self.set_scope(expr.module.scope)
        self._in_header = True
        name = expr.module.name

        funcs = ""
        cuda_headers = ""
        for f in expr.module.funcs:
            if not f.is_inline:
                if 'kernel' in f.decorators:  # Checking for 'kernel' decorator
                    cuda_headers += self.function_signature(f) + ';\n'
                else:
                    funcs += self.function_signature(f) + ';\n'
        global_variables = ''.join('extern '+self._print(d) for d in expr.module.declarations if not d.variable.is_private)
        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        self._in_header = False
        self.exit_scope()
        function_declaration = f'{cuda_headers}\n\
                    extern "C"{{\n\
                    {funcs}\
                    }}\n'
        return '\n'.join((f"#ifndef {name.upper()}_H",
                          f"#define {name.upper()}_H",
                          imports,
                          global_variables,
                          function_declaration,
                          "#endif // {name.upper()}_H\n"))

