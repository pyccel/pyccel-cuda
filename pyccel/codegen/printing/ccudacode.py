# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=missing-function-docstring


from functools import reduce
from pyccel.ast.builtins  import PythonTuple

from pyccel.ast.core      import (FunctionCall, Deallocate, FunctionAddress,
                                  FunctionDefArgument, Assign, Import,
                                  AliasAssign, Module, Declare, AsName)

from pyccel.ast.datatypes import NativeInteger
from pyccel.ast.datatypes import NativeTuple, datatype
from pyccel.ast.literals  import LiteralTrue, Literal, Nil

from pyccel.ast.numpyext import NumpyFull, NumpyArray, NumpyArange

from pyccel.ast.cupyext import CupyFull, CupyArray, CupyArange

from pyccel.ast.cudaext import CudaCopy, cuda_Internal_Var, CudaArray, CudaSharedArray

from pyccel.ast.operators import PyccelMul, PyccelUnarySub

from pyccel.ast.variable import Variable, PyccelArraySize
from pyccel.ast.variable import InhomogeneousTupleVariable, DottedName

from pyccel.ast.internals import Slice
from pyccel.ast.c_concepts import ObjectAddress

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.errors.errors   import Errors


errors = Errors()

# TODO: add examples

__all__ = ["CCudaCodePrinter", "ccudacode"]

# dictionary mapping numpy function to (argument_conditions, C_function).
# Used in CCodePrinter._print_NumpyUfuncBase(self, expr)
numpy_ufunc_to_c_float = {
    'NumpyAbs'  : 'fabs',
    'NumpyFabs'  : 'fabs',
    'NumpyMin'  : 'minval',
    'NumpyMax'  : 'maxval',
    'NumpyFloor': 'floor',  # TODO: might require special treatment with casting
    # ---
    'NumpyExp' : 'exp',
    'NumpyLog' : 'log',
    'NumpySqrt': 'sqrt',
    # ---
    'NumpySin'    : 'sin',
    'NumpyCos'    : 'cos',
    'NumpyTan'    : 'tan',
    'NumpyArcsin' : 'asin',
    'NumpyArccos' : 'acos',
    'NumpyArctan' : 'atan',
    'NumpyArctan2': 'atan2',
    'NumpySinh'   : 'sinh',
    'NumpyCosh'   : 'cosh',
    'NumpyTanh'   : 'tanh',
    'NumpyArcsinh': 'asinh',
    'NumpyArccosh': 'acosh',
    'NumpyArctanh': 'atanh',
}

numpy_ufunc_to_c_complex = {
    'NumpyAbs'  : 'cabs',
    'NumpyMin'  : 'minval',
    'NumpyMax'  : 'maxval',
    # ---
    'NumpyExp' : 'cexp',
    'NumpyLog' : 'clog',
    'NumpySqrt': 'csqrt',
    # ---
    'NumpySin'    : 'csin',
    'NumpyCos'    : 'ccos',
    'NumpyTan'    : 'ctan',
    'NumpyArcsin' : 'casin',
    'NumpyArccos' : 'cacos',
    'NumpyArctan' : 'catan',
    'NumpySinh'   : 'csinh',
    'NumpyCosh'   : 'ccosh',
    'NumpyTanh'   : 'ctanh',
    'NumpyArcsinh': 'casinh',
    'NumpyArccosh': 'cacosh',
    'NumpyArctanh': 'catanh',
}

# dictionary mapping Math function to (argument_conditions, C_function).
# Used in CCodePrinter._print_MathFunctionBase(self, expr)
# Math function ref https://docs.python.org/3/library/math.html
math_function_to_c = {
    # ---------- Number-theoretic and representation functions ------------
    'MathCeil'     : 'ceil',
    # 'MathComb'   : 'com' # TODO
    'MathCopysign': 'copysign',
    'MathFabs'   : 'fabs',
    'MathFloor'    : 'floor',
    # 'MathFmod'   : '???',  # TODO
    # 'MathRexp'   : '???'   TODO requires two output
    # 'MathFsum'   : '???',  # TODO
    # 'MathIsclose' : '???',  # TODO
    'MathIsfinite': 'isfinite', # int isfinite(real-floating x);
    'MathIsinf'   : 'isinf', # int isinf(real-floating x);
    'MathIsnan'   : 'isnan', # int isnan(real-floating x);
    # 'MathIsqrt'  : '???' TODO
    'MathLdexp'  : 'ldexp',
    # 'MathModf'  : '???' TODO return two value
    # 'MathPerm'  : '???' TODO
    # 'MathProd'  : '???' TODO
    'MathRemainder'  : 'remainder',
    'MathTrunc'  : 'trunc',

    # ----------------- Power and logarithmic functions -----------------------

    'MathExp'    : 'exp',
    'MathExpm1'  : 'expm1',
    'MathLog'    : 'log',      # take also an option arg [base]
    'MathLog1p'  : 'log1p',
    'MathLog2'  : 'log2',
    'MathLog10'  : 'log10',
    'MathPow'    : 'pow',
    'MathSqrt'   : 'sqrt',

    # --------------------- Trigonometric functions ---------------------------

    'MathAcos'   : 'acos',
    'MathAsin'   : 'asin',
    'MathAtan'   : 'atan',
    'MathAtan2'  : 'atan2',
    'MathCos'    : 'cos',
    # 'MathDist'  : '???', TODO
    'MathHypot'  : 'hypot',
    'MathSin'    : 'sin',
    'MathTan'    : 'tan',


    # -------------------------- Hyperbolic functions -------------------------

    'MathAcosh'  : 'acosh',
    'MathAsinh'  : 'asinh',
    'MathAtanh'  : 'atanh',
    'MathCosh'   : 'cosh',
    'MathSinh'   : 'sinh',
    'MathTanh'   : 'tanh',

    # --------------------------- Special functions ---------------------------

    'MathErf'    : 'erf',
    'MathErfc'   : 'erfc',
    'MathGamma'  : 'tgamma',
    'MathLgamma' : 'lgamma',

    # --------------------------- internal functions --------------------------
    'MathFactorial' : 'pyc_factorial',
    'MathGcd'       : 'pyc_gcd',
    'MathDegrees'   : 'pyc_degrees',
    'MathRadians'   : 'pyc_radians',
    'MathLcm'       : 'pyc_lcm',
}

c_library_headers = (
    "complex",
    "ctype",
    "float",
    "math",
    "stdarg",
    "stdbool",
    "stddef",
    "stdint",
    "stdio",
    "stdlib",
    "string",
    "tgmath",
)

dtype_registry = {('float',8)   : 'double',
                  ('float',4)   : 'float',
                  ('complex',8) : 'double complex',
                  ('complex',4) : 'float complex',
                  ('int',4)     : 'int32_t',
                  ('int',8)     : 'int64_t',
                  ('int',2)     : 'int16_t',
                  ('int',1)     : 'int8_t',
                  ('bool',4)    : 'bool'}

ndarray_type_registry = {
                  ('float',8)   : 'nd_double',
                  ('float',4)   : 'nd_float',
                  ('complex',8) : 'nd_cdouble',
                  ('complex',4) : 'nd_cfloat',
                  ('int',8)     : 'nd_int64',
                  ('int',4)     : 'nd_int32',
                  ('int',2)     : 'nd_int16',
                  ('int',1)     : 'nd_int8',
                  ('bool',4)    : 'nd_bool'}

import_dict = {'omp_lib' : 'omp' }

c_imports = {n : Import(n, Module(n, (), ())) for n in
                ['stdlib',
                 'math',
                 'string',
                 'ndarrays',
                 'cuda_ndarrays',
                 'ho_cuda_ndarrays',
                 'math',
                 'complex',
                 'stdint',
                 'pyc_math_c',
                 'stdio',
                 'stdbool',
                 'assert']}

class CCudaCodePrinter(CCodePrinter):
    """A printer to convert python expressions to strings of ccuda code"""
    printmethod = "_ccudacode"
    language = "ccuda"

    _default_settings = {
        'tabwidth': 4,
    }

    def __init__(self, filename, prefix_module = None):

        errors.set_target(filename, 'file')

        super().__init__(filename)
        self.prefix_module = prefix_module
        self._additional_imports = {'stdlib':c_imports['stdlib']}
        self._additional_code = ''
        self._additional_args = []
        self._temporary_args = []
        self._current_module = None
        self._in_header = False
        # Dictionary linking optional variables to their
        # temporary counterparts which provide allocated
        # memory
        # Key is optional variable
        self._optional_partners = {}

    def function_signature(self, expr, print_arg_names = True):
        """
        Get the Ccuda representation of the function signature.
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
        if len(expr.results) > 1:
            self._additional_args.append(expr.results)
        args = list(expr.arguments)
        if len(expr.results) == 1:
            ret_type = self.get_declare_type(expr.results[0])
        elif len(expr.results) > 1:
            ret_type = self._print(datatype('int'))
            args += [FunctionDefArgument(a) for a in expr.results]
        else:
            ret_type = self._print(datatype('void'))
        name = expr.name
        if not args:
            arg_code = 'void'
        else:
            def get_var_arg(arg, var):
                code = "const " * var.is_const
                code += self.get_declare_type(var) + ' '
                code += arg.name * print_arg_names
                return code

            var_list = [a.var for a in args]
            arg_code_list = [self.function_signature(var, False) if isinstance(var, FunctionAddress)
                                else get_var_arg(arg, var) for arg, var in zip(args, var_list)]
            arg_code = ', '.join(arg_code_list)

        if self._additional_args :
            self._additional_args.pop()

        #TODO: need to check if "extern C" is necessary.
        extern_word = 'extern "C"'
        # extern_word = ''

        cuda_deco = ''
        if 'kernel' in expr.decorators:
            cuda_deco = "__global__"
        elif 'device' in expr.decorators:
            cuda_deco = "__device__"

        if isinstance(expr, FunctionAddress):
            return f'{extern_word} {ret_type} (*{name})({arg_code})'
        else:
            return f'{extern_word} {cuda_deco} {ret_type} {name}({arg_code})'

    def _print_Import(self, expr):
        if expr.ignore:
            return ''
        if isinstance(expr.source, AsName):
            source = expr.source.name
        else:
            source = expr.source
        if isinstance(source, DottedName):
            source = source.name[-1]
        else:
            source = self._print(source)

        # Get with a default value is not used here as it is
        # slower and on most occasions the import will not be in the
        # dictionary
        if source in import_dict: # pylint: disable=consider-using-get
            source = import_dict[source]

        if source is None:
            return ''
        if expr.source in c_library_headers:
            return '#include <{0}.h>\n'.format(source)
        else:
            if len(source) > 3 and source[:2] == 'ho':
                # self._additional_imports.pop(source)
                return f'#define HO_CUDA_PYCCEL\n#include "{source}.h"\n'                

            return f'#include "{source}.h"\n'

    def _print_Declare(self, expr):
        if isinstance(expr.variable, InhomogeneousTupleVariable):
            return ''.join(self._print_Declare(Declare(v.dtype,v,intent=expr.intent, static=expr.static)) for v in expr.variable)

        declaration_type = self.get_declare_type(expr.variable)
        variable = self._print(expr.variable.name)

        if expr.variable.memory_location == 'shared':
            preface, init = self._init_shared_array(expr.variable)
        elif expr.variable.is_stack_array:
            preface, init = self._init_stack_array(expr.variable,)
        elif declaration_type == 't_ndarray' and not self._in_header:
            preface = ''
            init    = ' = {.shape = NULL}'
        else:
            preface = ''
            init    = ''

        declaration = f'{declaration_type} {variable}{init};\n'

        return preface + declaration

    def _init_shared_array(self, expr):
        """ return a string which handles the assignment of a shared ndarray

        Parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Returns
        -------
            buffer_array : str
                String initialising the shared (C) array which stores the data
            array_init   : str
                String containing the rhs of the initialization of a stack array
        """
        var = expr
        dtype_str = self._print(var.dtype)
        dtype = self.find_in_dtype_registry(dtype_str, var.precision)
        np_dtype = self.find_in_ndarray_type_registry(dtype_str, var.precision)
        shape = ", ".join(self._print(i) for i in var.alloc_shape)
        tot_shape = self._print(reduce(
            lambda x,y: PyccelMul(x,y,simplify=True), var.alloc_shape))
        declare_dtype = self.find_in_dtype_registry('int', 8)

        dummy_array_name = self.scope.get_new_name('array_dummy')
        is_shared = '__shared__' if expr.memory_location == 'shared' else ''
        buffer_array = f'{is_shared} {dtype} {dummy_array_name}[{tot_shape}];\n'
        shape_init = "({declare_dtype}[]){{{shape}}}".format(declare_dtype=declare_dtype, shape=shape)
        strides_init = "({declare_dtype}[{length}]){{0}}".format(declare_dtype=declare_dtype, length=len(var.shape))
        array_init = ' = (t_ndarray){{\n.{0}={1},\n .nd={4},\n .shape={2},\n'
        array_init += '.strides={3},\n .type={0},\n .is_view={5}\n}};\n'
        array_init = array_init.format(np_dtype, dummy_array_name,
                    shape_init, strides_init, len(var.shape), 'false')
        # TODO: call this only one time per block (need to check threadIdx.xyz == 0 then and only then run the code)
        array_init += 'shared_array_init(&{})'.format(self._print(var))
        self.add_import(c_imports['ho_cuda_ndarrays'])
        return buffer_array, array_init


    def _init_stack_array(self, expr):
        """ return a string which handles the assignment of a stack ndarray

        Parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        -------
            buffer_array : str
                String initialising the stack (C) array which stores the data
            array_init   : str
                String containing the rhs of the initialization of a stack array
        """
        var = expr
        dtype_str = self._print(var.dtype)
        dtype = self.find_in_dtype_registry(dtype_str, var.precision)
        np_dtype = self.find_in_ndarray_type_registry(dtype_str, var.precision)
        shape = ", ".join(self._print(i) for i in var.alloc_shape)
        tot_shape = self._print(reduce(
            lambda x,y: PyccelMul(x,y,simplify=True), var.alloc_shape))
        declare_dtype = self.find_in_dtype_registry('int', 8)

        dummy_array_name = self.scope.get_new_name('array_dummy')
        buffer_array = "{dtype} {name}[{size}];\n".format(
                dtype = dtype,
                name  = dummy_array_name,
                size  = tot_shape)
        shape_init = "({declare_dtype}[]){{{shape}}}".format(declare_dtype=declare_dtype, shape=shape)
        strides_init = "({declare_dtype}[{length}]){{0}}".format(declare_dtype=declare_dtype, length=len(var.shape))
        array_init = ' = (t_ndarray){{\n.{0}={1},\n .shape={2},\n .strides={3},\n '
        array_init += '.nd={4},\n .type={0},\n .is_view={5}\n}};\n'
        array_init = array_init.format(np_dtype, dummy_array_name,
                    shape_init, strides_init, len(var.shape), 'false')
        array_init += 'stack_array_init(&{})'.format(self._print(var))
        self.add_import(c_imports['ndarrays'])
        return buffer_array, array_init


    def _print_Allocate(self, expr):
        free_code = ''
        #free the array if its already allocated and checking if its not null if the status is unknown
        if  (expr.status == 'unknown'):
            free_code = 'if (%s.shape != NULL)\n' % self._print(expr.variable.name)
            free_code += "{{\n{}}}\n".format(self._print(Deallocate(expr.variable)))
        elif  (expr.status == 'allocated'):
            free_code += self._print(Deallocate(expr.variable))
        shape = ", ".join(self._print(i) for i in expr.shape)
        shape_dtype = self.find_in_dtype_registry('int', 8)
        tmp_shape = self.scope.get_new_name('tmp_shape')
        dtype = self._print(expr.variable.dtype)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.variable.precision)
        shape_Assign = "{} {}[] = {{{}}};".format(shape_dtype, tmp_shape, shape)
        is_view = 'false' if expr.variable.on_heap else 'true'
        self.add_import(c_imports['cuda_ndarrays'])
        # define the memory location for the created cuda array
        memory_location = expr.variable.memory_location
        if memory_location in ('device', 'host'):
            memory_location = 'allocateMemoryOn' + str(memory_location).capitalize()
        else:
            memory_location = 'managedMemory'
        alloc_code = f"{expr.variable} = \
            cuda_array_create({len(expr.shape)}, {tmp_shape}, {dtype}, {is_view}, {memory_location});"
        return f"{free_code}\n{shape_Assign}\n{alloc_code}\n"

    def _print_Deallocate(self, expr):
        var_code = self._print(expr.variable)
        if expr.variable.is_alias:
            return f"cuda_free_pointer({var_code});\n"
        else:
            if expr.variable.memory_location == 'host':
                return f"cuda_free_host({var_code});\n"
            else:
                return f"cuda_free({var_code});\n"

    def _print_IndexedElement(self, expr):
        base = expr.base
        inds = list(expr.indices)
        base_shape = base.shape
        allow_negative_indexes = True if isinstance(base, PythonTuple) else base.allows_negative_indexes
        for i, ind in enumerate(inds):
            if isinstance(ind, PyccelUnarySub) and isinstance(ind.args[0], LiteralInteger):
                inds[i] = PyccelMinus(base_shape[i], ind.args[0], simplify = True)
            else:
                #indices of indexedElement of len==1 shouldn't be a tuple
                if isinstance(ind, tuple) and len(ind) == 1:
                    inds[i].args = ind[0]
                if allow_negative_indexes and \
                        not isinstance(ind, LiteralInteger) and not isinstance(ind, Slice):
                    inds[i] = IfTernaryOperator(PyccelLt(ind, LiteralInteger(0)),
                        PyccelAdd(base_shape[i], ind, simplify = True), ind)
        #set dtype to the C struct types
        dtype = self._print(expr.dtype)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.precision)
        base_name = self._print(base)
        if getattr(base, 'is_ndarray', False) or isinstance(base, HomogeneousTupleVariable):
            if expr.rank > 0:
                #managing the Slice input
                for i , ind in enumerate(inds):
                    if isinstance(ind, Slice):
                        inds[i] = self._new_slice_with_processed_arguments(ind, PyccelArraySize(base, i),
                            allow_negative_indexes)
                    else:
                        inds[i] = Slice(ind, PyccelAdd(ind, LiteralInteger(1), simplify = True), LiteralInteger(1),
                            Slice.Element)
                inds = [self._print(i) for i in inds]
                return "cuda_array_slicing(%s, %s, (t_slice []){%s})" % (base_name, expr.rank, ", ".join(inds))
            inds = [self._cast_to(i, NativeInteger(), 8).format(self._print(i)) for i in inds]
        else:
            raise NotImplementedError(expr)
        return "GET_ELEMENT(%s, %s, %s)" % (base_name, dtype, ", ".join(inds))


    def _print_Slice(self, expr):
        start = self._print(expr.start)
        stop = self._print(expr.stop)
        step = self._print(expr.step)
        slice_type = 'RANGE' if expr.slice_type == Slice.Range else 'ELEMENT'
        return f'cuda_new_slice({start}, {stop}, {step}, {slice_type})'


    def _print_KernelCall(self, expr):
        func = expr.funcdef
        if func.is_inline:
            return self._handle_inline_func_call(expr)
         # Ensure the correct syntax is used for pointers
        args = []
        for a, f in zip(expr.args, func.arguments):
            a = a.value if a else Nil()
            f = f.var
            if self.is_c_pointer(f):
                if isinstance(a, Variable):
                    args.append(ObjectAddress(a))
                elif not self.is_c_pointer(a):
                    tmp_var = self.scope.get_temporary_variable(f.dtype)
                    assign = Assign(tmp_var, a)
                    self._additional_code += self._print(assign)
                    args.append(ObjectAddress(tmp_var))
                else:
                    args.append(a)
            else :
                args.append(a)

        args += self._temporary_args
        self._temporary_args = []
        args = ', '.join(['{}'.format(self._print(a)) for a in args])
        # TODO: need to raise error in semantic if we have result , kernel can't return
        if not func.results:
            return '{}<<<dim3{},dim3{}>>>({});\n'.format(func.name, expr.numBlocks, expr.tpblock,args)

    def _print_Assign(self, expr):
        prefix_code = ''
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(lhs, Variable) and lhs.is_optional:
            if lhs in self._optional_partners:
                # Collect temporary variable which provides
                # allocated memory space for this optional variable
                tmp_var = self._optional_partners[lhs]
            else:
                # Create temporary variable to provide allocated
                # memory space before assigning to the pointer value
                # (may be NULL)
                tmp_var = self.scope.get_temporary_variable(lhs,
                        is_optional = False)
                self._optional_partners[lhs] = tmp_var
            # Point optional variable at an allocated memory space
            prefix_code = self._print(AliasAssign(lhs, tmp_var))
        if isinstance(rhs, FunctionCall) and isinstance(rhs.dtype, NativeTuple):
            self._temporary_args = [ObjectAddress(a) for a in lhs]
            return prefix_code+'{};\n'.format(self._print(rhs))
        # Inhomogenous tuples are unravelled and therefore do not exist in the c printer

        if isinstance(rhs, (CupyFull)):
            return prefix_code+self.cuda_arrayFill(expr)
        if isinstance(rhs, CupyArange):
            return prefix_code+self.cuda_Arange(expr)
        if isinstance(rhs, (CudaArray, CupyArray)):
            return prefix_code+self.copy_CudaArray_Data(expr)
        if isinstance(rhs, (NumpyArray, PythonTuple)):
            return prefix_code+self.copy_NumpyArray_Data(expr)
        if isinstance(rhs, (NumpyFull)):
            return prefix_code+self.arrayFill(expr)
        if isinstance(rhs, NumpyArange):
            return prefix_code+self.fill_NumpyArange(rhs, lhs)
        if isinstance(rhs, CudaCopy):
            return prefix_code+self.cudaCopy(lhs, rhs)
        if isinstance(rhs, CudaSharedArray):
            return '\n'
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return prefix_code+'{} = {};\n'.format(lhs, rhs)

    def arrayFill(self, expr):
        """ print the assignment of a NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains a call to the C function array_fill using Cuda api,
        """
        rhs = expr.rhs
        lhs = expr.lhs
        code_init = ''
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
        dtype = self.find_in_ndarray_type_registry(self._print(rhs.dtype), rhs.precision)
        dtype = dtype[3:]

        if rhs.fill_value is not None:
            if isinstance(rhs.fill_value, Literal):
                code_init += 'array_fill_{0}(({1}){2}, {3});\n'.format(dtype, declare_dtype, self._print(rhs.fill_value), self._print(lhs))
            else:
                code_init += 'array_fill_{0}({1}, {2});\n'.format(dtype, self._print(rhs.fill_value), self._print(lhs))
        return code_init

    def cuda_Arange(self, expr):
        """ print the assignment of a NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains a call to the C function array_arange using Cuda api,
        """
        rhs = expr.rhs
        lhs = expr.lhs
        code_init = ''
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
        dtype = self.find_in_ndarray_type_registry(self._print(rhs.dtype), rhs.precision)
        dtype = dtype[3:]

        #TODO: calculate best thread number to run the kernel
        code_init += 'cuda_array_arange_{0}<<<1,32>>>({1}, {2});\n'.format(dtype, self._print(lhs), self._print(rhs.start))
        return code_init

    def cuda_arrayFill(self, expr):
        """ print the assignment of a NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains a call to the C function array_fill using Cuda api,
        """
        rhs = expr.rhs
        lhs = expr.lhs
        code_init = ''
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
        dtype = self.find_in_ndarray_type_registry(self._print(rhs.dtype), rhs.precision)
        dtype = dtype[3:]

        if rhs.fill_value is not None:
            if isinstance(rhs.fill_value, Literal):
                code_init += 'cuda_array_fill_{0}<<<1,1>>>(({1}){2}, {3});\n'.format(dtype, declare_dtype, self._print(rhs.fill_value), self._print(lhs))
            else:
                code_init += 'cuda_array_fill_{0}<<<1,1>>>({1}, {2});\n'.format(dtype, self._print(rhs.fill_value), self._print(lhs))
        return code_init

    def copy_CudaArray_Data(self, expr):
        """ print the assignment of a Cuda NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains the declaration of a dummy data_buffer
                       and a call to an operator which copies it to a Cuda NdArray struct
                if the ndarray is a stack_array the str will contain the initialization
        """
        rhs = expr.rhs
        lhs = expr.lhs
        if rhs.rank == 0:
            raise NotImplementedError(str(expr))
        dummy_array_name = self.scope.get_new_name('cuda_array_dummy')
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
        dtype = self.find_in_ndarray_type_registry(self._print(rhs.dtype), rhs.precision)
        arg = rhs.arg if isinstance(rhs, (CudaArray, CupyArray)) else rhs
        if rhs.rank > 1:
            # flattening the args to use them in C initialization.
            arg = self._flatten_list(arg)

        self.add_import(c_imports['string'])
        if isinstance(arg, Variable):
            arg = self._print(arg)
            cpy_data = "cudaMemcpy({0}.raw_data, {1}.{2}, {0}.buffer_size, cudaMemcpyHostToDevice);".format(lhs, arg, dtype)
            return '%s\n' % (cpy_data)
        else :
            arg = ', '.join(self._print(i) for i in arg)
            dummy_array = "%s %s[] = {%s};\n" % (declare_dtype, dummy_array_name, arg)
            cpy_data = "cudaMemcpy({0}.raw_data, {1}, {0}.buffer_size, cudaMemcpyHostToDevice);".format(self._print(lhs), dummy_array_name, dtype)
            return  '%s%s\n' % (dummy_array, cpy_data)

    def _print_CudaSynchronize(self, expr):
        return 'cudaDeviceSynchronize();\n'

    def _print_CudaSyncthreads(self, expr):
        return '__syncthreads();\n'

    def _print_CudaSharedArray(self, expr):
        return 'TODO'

    def _print_CudaInternalVar(self, expr):
        var_name = type(expr).__name__
        var_name = cuda_Internal_Var[var_name]
        dim_c = ('x', 'y', 'z')[expr.dim]
        return '{}.{}'.format(var_name, dim_c)
    
    def cudaCopy(self, lhs, rhs):
        from_location = 'Host'
        to_location = 'Host'
        if rhs.arg.memory_location in ('device', 'managed'):
            from_location = 'Device'
        if rhs.memory_location in ('device', 'managed'):
            to_location = 'Device'
        transfer_type = 'cudaMemcpy{0}To{1}'.format(from_location, to_location)
        if isinstance(rhs.is_async, LiteralTrue):
            cpy_data = "cudaMemcpyAsync({0}.raw_data, {1}.raw_data, {0}.buffer_size, {2}, 0);".format(lhs, rhs.arg, transfer_type)
        else:
            cpy_data = "cudaMemcpy({0}.raw_data, {1}.raw_data, {0}.buffer_size, {2});".format(lhs, rhs.arg, transfer_type)
        return '%s\n' % (cpy_data)

def ccudacode(expr, filename, assign_to=None, **settings):
    """Converts an expr to a string of ccuda code

    expr : Expr
        A pyccel expression to be converted.
    filename : str
        The name of the file being translated. Used in error printing
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    """
    return CCudaCodePrinter(filename, **settings).doprint(expr, assign_to)
