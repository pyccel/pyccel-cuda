# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : FortranToCWrapper
which creates an interface exposing Fortran code to C.
"""
import warnings
from pyccel.ast.bind_c import BindCFunctionDefArgument, BindCFunctionDefResult
from pyccel.ast.bind_c import BindCPointer, BindCFunctionDef, C_F_Pointer
from pyccel.ast.bind_c import CLocFunc, BindCModule, BindCVariable
from pyccel.ast.bind_c import BindCArrayVariable, BindCClassDef, DeallocatePointer
from pyccel.ast.bind_c import BindCClassProperty
from pyccel.ast.builtins import VariableIterator
from pyccel.ast.core import Assign, FunctionCall, FunctionCallArgument
from pyccel.ast.core import Allocate, EmptyNode, FunctionAddress
from pyccel.ast.core import If, IfSection, Import, Interface, FunctionDefArgument
from pyccel.ast.core import AsName, Module, AliasAssign, FunctionDefResult
from pyccel.ast.core import For
from pyccel.ast.datatypes import CustomDataType, FixedSizeNumericType
from pyccel.ast.datatypes import HomogeneousTupleType, TupleType
from pyccel.ast.datatypes import HomogeneousSetType, PythonNativeInt
from pyccel.ast.datatypes import HomogeneousListType
from pyccel.ast.internals import Slice
from pyccel.ast.literals import LiteralInteger, Nil, LiteralTrue
from pyccel.ast.numpytypes import NumpyNDArrayType
from pyccel.ast.operators import PyccelIsNot, PyccelMul, PyccelAdd
from pyccel.ast.variable import Variable, IndexedElement, DottedVariable
from pyccel.ast.numpyext import NumpyNDArrayType
from pyccel.errors.errors import Errors
from pyccel.parser.scope import Scope
from .wrapper import Wrapper

errors = Errors()

class FortranToCWrapper(Wrapper):
    """
    Class for creating a wrapper exposing Fortran code to C.

    A class which provides all necessary functions for wrapping different AST
    objects such that the resulting AST is C-compatible. This new AST is
    printed as an intermediary layer.
    """
    def __init__(self):
        self._additional_exprs = []
        self._wrapper_names_dict = {}
        super().__init__()

    def _get_function_def_body(self, func, func_def_args, func_arg_to_call_arg, results, handled = ()):
        """
        Get the body of the bind c function definition.

        Get the body of the bind c function definition by inserting if blocks
        to check the presence of optional variables. Once we have ascertained
        the presence of the variables the original function is called. This
        code slices array variables to ensure the correct step.

        Parameters
        ----------
        func : FunctionDef
            The function which should be called.

        func_def_args : list of FunctionDefArguments
            The arguments received by the function.

        func_arg_to_call_arg : dict
            A dictionary mapping the arguments received by the function to the arguments
            to be passed to the function call.

        results : list of Variables
            The Variables where the result of the function call will be saved.

        handled : tuple
            A list of all variables which have been handled (checked to see if they
            are present).

        Returns
        -------
        list
            A list of Basic nodes describing the body of the function.
        """
        optional = next((a for a in func_def_args if a.original_function_argument_variable.is_optional and a not in handled), None)
        if optional:
            args = func_def_args.copy()
            optional_var = optional.var
            handled += (optional, )
            true_section = IfSection(PyccelIsNot(optional_var, Nil()),
                                    self._get_function_def_body(func, args, func_arg_to_call_arg, results, handled))
            args.remove(optional)
            false_section = IfSection(LiteralTrue(),
                                    self._get_function_def_body(func, args, func_arg_to_call_arg, results, handled))
            return [If(true_section, false_section)]
        else:
            args = [FunctionCallArgument(func_arg_to_call_arg[fa],
                                         keyword = fa.original_function_argument_variable.name)
                    for fa in func_def_args]
            size = [fa.shape[::-1] if fa.original_function_argument_variable.order == 'C' else
                    fa.shape for fa in func_def_args]
            stride = [fa.strides[::-1] if fa.original_function_argument_variable.order == 'C' else
                      fa.strides for fa in func_def_args]
            orig_size = [[PyccelMul(l,s) for l,s in zip(sz, st)] for sz,st in zip(size,stride)]
            body = [C_F_Pointer(fa.var, func_arg_to_call_arg[fa].base, s)
                    for fa,s in zip(func_def_args, orig_size)
                    if isinstance(func_arg_to_call_arg[fa], IndexedElement)]
            body += [C_F_Pointer(fa.var, func_arg_to_call_arg[fa], [fa.shape[0]])
                    for fa in func_def_args
                    if isinstance(fa.original_function_argument_variable.class_type, HomogeneousTupleType)]
            body += [C_F_Pointer(fa.var, func_arg_to_call_arg[fa])
                     for fa in func_def_args
                     if not isinstance(func_arg_to_call_arg[fa], IndexedElement) \
                        and fa.original_function_argument_variable.is_optional]
            body += [C_F_Pointer(fa.var, func_arg_to_call_arg[fa]) for fa in func_def_args
                    if isinstance(func_arg_to_call_arg[fa].dtype, CustomDataType)]

            # If the function is inlined and takes an array argument create a pointer to ensure that the bounds
            # are respected
            if getattr(func, 'is_inline', False) and any(isinstance(a.value, IndexedElement) for a in args):
                array_args = {a: self.scope.get_temporary_variable(a.value.base, a.keyword, memory_handling = 'alias') for a in args if isinstance(a.value, IndexedElement)}
                body += [AliasAssign(v, k.value) for k,v in array_args.items()]
                args = [FunctionCallArgument(array_args[a], keyword=a.keyword) if a in array_args else a for a in args]

            if len(results) == 1:
                res = results[0]
                func_call = AliasAssign(res, func(*args)) if res.is_alias else \
                            Assign(res, func(*args))
            else:
                func_call = Assign(results, func(*args))
            return body + [func_call]

    def _get_call_argument(self, bind_c_arg):
        """
        Get the argument which should be passed to the function call.

        The FunctionDefArgument passed to the function may contain additional
        information which should not be passed to the function being wrapped
        (e.g. an array with strides should not pass the strides explicitly to
        the function call, nor should it pass the entire contiguous array).
        This function extracts the necessary information and returns the object
        which can be passed to the function call.

        Parameters
        ----------
        bind_c_arg : BindCFunctionDefArgument
            The argument to the wrapped bind_c_X function.

        Returns
        -------
        TypedAstNode
            An object which can be passed to a function call of the function
            being wrapped.
        """
        original_arg = bind_c_arg.original_function_argument_variable
        arg_var = self.scope.find(self.scope.get_expected_name(original_arg.name), category='variables')
        if original_arg.is_ndarray:
            start = LiteralInteger(1) # C_F_Pointer leads to default Fortran lbound
            stop = None
            indexes = [Slice(start, stop, step) for step in bind_c_arg.strides]
            return IndexedElement(arg_var, *indexes)
        else:
            return arg_var

    def _wrap_Module(self, expr):
        """
        Create a Module which is compatible with C.

        Create a Module which provides an interface between C and the
        Module described by expr. This includes wrapping functions,
        interfaces, classes and module variables.

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

        # Wrap contents
        # We only wrap the non inlined functions
        funcs_to_wrap = [f for f in expr.funcs if f.is_semantic and not f.is_inline]

        funcs = [self._wrap(f) for f in funcs_to_wrap]
        if expr.init_func:
            init_func = funcs[next(i for i,f in enumerate(funcs_to_wrap) if f == expr.init_func)]
        else:
            init_func = None
        if expr.free_func:
            free_func = funcs[next(i for i,f in enumerate(funcs_to_wrap) if f == expr.free_func)]
        else:
            free_func = None
        removed_functions = [f for f,w in zip(funcs_to_wrap, funcs) if isinstance(w, EmptyNode)]
        funcs = [f for f in funcs if not isinstance(f, EmptyNode)]
        interfaces = [self._wrap(f) for f in expr.interfaces if not f.is_inline]
        classes = [self._wrap(f) for f in expr.classes]
        variables = [self._wrap(v) for v in expr.variables if not v.is_private]
        variable_getters = [v for v in variables if isinstance(v, BindCArrayVariable)]
        imports = [Import(self.scope.get_python_name(expr.name), target = expr, mod=expr)]

        name = mod_scope.get_new_name(f'bind_c_{expr.name}')
        self._wrapper_names_dict[expr.name] = name

        self.exit_scope()

        return BindCModule(name, variables, funcs, variable_wrappers = variable_getters,
                init_func = init_func, free_func = free_func,
                interfaces = interfaces, classes = classes,
                imports = imports, original_module = expr,
                scope = mod_scope, removed_functions = removed_functions)

    def _wrap_FunctionDef(self, expr):
        """
        Create a C-compatible function which executes the original function.

        Create a function which can be called from C which internally calls the original
        function. It does this by wrapping the arguments and the results and unrolling
        the body using self._get_function_def_body to ensure optional arguments are
        present before accessing them. With all this information a BindCFunctionDef is
        created which is C-compatible.

        Functions which cannot be wrapped raise a warning and return an EmptyNode. This
        is the case for functions with functions as arguments.

        Parameters
        ----------
        expr : FunctionDef
            The function to wrap.

        Returns
        -------
        BindCFunctionDef
            The C-compatible function.
        """
        if expr.is_private:
            return EmptyNode()

        orig_name = expr.cls_name or expr.name
        name = self.scope.get_new_name(f'bind_c_{orig_name.lower()}')
        self._wrapper_names_dict[expr.name] = name
        in_cls = expr.arguments and expr.arguments[0].bound_argument

        # Create the scope
        func_scope = self.scope.new_child_scope(name)
        self.scope = func_scope

        self._additional_exprs = []

        if any(isinstance(a.var, FunctionAddress) for a in expr.arguments):
            warnings.warn("Functions with functions as arguments cannot be wrapped by pyccel")
            return EmptyNode()

        # Wrap the arguments and collect the expressions passed as the call argument.
        func_arguments = [self._wrap(a) for a in expr.arguments]
        call_arguments = [self._get_call_argument(fa) for fa in func_arguments]
        func_to_call = {fa : ca for ca, fa in zip(call_arguments, func_arguments)}

        func_results = [self._wrap_FunctionDefResult(r) for r in expr.results]

        func_call_results = [r.var.clone(self.scope.get_expected_name(r.var.name)) for r in expr.results]

        interface = expr.get_direct_user_nodes(lambda u: isinstance(u, Interface))

        if in_cls and interface:
            body = self._get_function_def_body(interface[0], func_arguments, func_to_call, func_call_results)
        else:
            body = self._get_function_def_body(expr, func_arguments, func_to_call, func_call_results)

        body.extend(self._additional_exprs)
        self._additional_exprs.clear()

        if expr.scope.get_python_name(expr.name) == '__del__':
            body.append(DeallocatePointer(call_arguments[0]))

        self.exit_scope()

        func = BindCFunctionDef(name, func_arguments, body, func_results, scope=func_scope, original_function = expr,
                docstring = expr.docstring, result_pointer_map = expr.result_pointer_map)

        self.scope.functions[name] = func

        return func

    def _wrap_Interface(self, expr):
        """
        Create an interface containing only C-compatible functions.

        Create an interface containing only functions which can be called from C
        from an interface which is not necessarily C-compatible.

        Parameters
        ----------
        expr : pyccel.ast.core.Interface
            The interface to be wrapped.

        Returns
        -------
        pyccel.ast.core.Interface
            The C-compatible interface.
        """
        functions = [self.scope.functions[self._wrapper_names_dict[f.name]] for f in expr.functions]
        functions = [f for f in functions if not isinstance(f, EmptyNode)]
        return Interface(expr.name, functions, expr.is_argument)

    def _wrap_FunctionDefArgument(self, expr):
        """
        Create the equivalent BindCFunctionDefArgument for a C-compatible function.

        Take a FunctionDefArgument and create a BindCFunctionDefArgument describing
        all the information that should be passed to the C-compatible function in order
        to be able to create the argument described by `expr`.

        In the case of a scalar numerical the function simply creates a copy of the
        variable described by the function argument in the local scope.

        In the case of an array, C cannot represent the array natively. Rather it is
        stored in a pointer. This function therefore creates a variable to represent
        that pointer. Additionally information about the shape and strides of the array
        are necessary, however these objects are created by the `BindCFunctionDefArgument`
        class.

        The objects which describe the argument passed to the `expr` argument of the
        original function are also created here. However the expressions necessary to
        collect the information from the BindCFunctionDefArgument in order to create
        these objects are left for later. This is done to ensure that optionals are
        handled locally to the function call. This ensures that we do not duplicate if
        conditions.

        Parameters
        ----------
        expr : FunctionDefArgument
            The argument to be wrapped.

        Returns
        -------
        BindCFunctionDefArgument
            The C-compatible argument.
        """
        var = expr.var
        name = var.name
        self.scope.insert_symbol(name)
        collisionless_name = self.scope.get_expected_name(var.name)
        if isinstance(var.class_type, (NumpyNDArrayType, HomogeneousTupleType)) or \
                var.is_optional or isinstance(var.dtype, CustomDataType):
            new_var = Variable(BindCPointer(), self.scope.get_new_name(f'bound_{name}'),
                                is_argument = True, is_optional = False, memory_handling='alias')
            arg_var = var.clone(collisionless_name, is_argument = False, is_optional = False,
                                memory_handling = 'alias', allows_negative_indexes=False,
                                new_class = Variable)
            self.scope.insert_variable(arg_var)
        else:
            new_var = var.clone(collisionless_name, new_class = Variable)
        self.scope.insert_variable(new_var)

        return BindCFunctionDefArgument(new_var, value = expr.value, original_arg_var = expr.var,
                kwonly = expr.is_kwonly, annotation = expr.annotation, scope=self.scope,
                wrapping_bound_argument = expr.bound_argument, persistent_target = expr.persistent_target)

    def _wrap_FunctionDefResult(self, expr):
        """
        Create the equivalent BindCFunctionDefResult for a C-compatible function.

        Take a FunctionDefResult and create a BindCFunctionDefResult describing
        all the information that should be returned from the C-compatible function
        in order to fully describe the result `expr`. This function also adds any
        expressions necessary to build the C-compatible return value to
        `self._additional_exprs`.

        In the case of a scalar numerical the function simply creates a local version
        of the variable described by the function result and returns the
        BindCFunctionDefResult.

        In the case of an array, C cannot represent the array natively. Rather it is
        stored in a pointer. This function therefore creates a variable to represent
        that pointer. Additionally information about the shape and strides of the array
        are necessary. These objects are created by the `BindCFunctionDefResult`
        class. The assignment expressions which define the shapes and strides are
        then stored in `self._additional_exprs` along with the allocation of the
        pointer.

        Parameters
        ----------
        expr : FunctionDefResult
            The result to be wrapped.

        Returns
        -------
        BindCFunctionDefResult
            The C-compatible result.
        """
        var = expr.var
        name = var.name
        scope = self.scope
        # Make name available for later
        scope.insert_symbol(name)
        wrap_dotted = isinstance(var, DottedVariable)
        stored_in_c_ptr = var.rank or isinstance(var.dtype, CustomDataType)
        memory_handling = 'alias' if wrap_dotted and stored_in_c_ptr else var.memory_handling
        local_var = var.clone(scope.get_expected_name(name), new_class = Variable,
                            memory_handling = memory_handling)

        if stored_in_c_ptr:
            # Allocatable is not returned so it must appear in local scope
            scope.insert_variable(local_var, name)

            # Create the C-compatible data pointer
            bind_var = Variable(BindCPointer(),
                                scope.get_new_name('bound_'+name),
                                is_const=False, memory_handling='alias')
            scope.insert_variable(bind_var)

            result = BindCFunctionDefResult(bind_var, var, scope)

            # Save the shapes of the array
            self._additional_exprs.extend([Assign(result.shape[i], local_var.shape[i]) for i in range(var.rank)])

            if not (var.is_alias or wrap_dotted):
                # Create an array variable which can be passed to CLocFunc
                ptr_var = Variable(NumpyNDArrayType(var.dtype, var.rank, var.order), scope.get_new_name(name+'_ptr'),
                                    memory_handling='alias')
                scope.insert_variable(ptr_var)

                # Define the additional steps necessary to define and fill ptr_var
                alloc = Allocate(ptr_var, shape=result.shape, status='unallocated')
                if isinstance(local_var.class_type, (NumpyNDArrayType, HomogeneousTupleType, CustomDataType)):
                    copy = Assign(ptr_var, local_var)
                    self._additional_exprs.extend([alloc, copy])
                elif isinstance(local_var.class_type, (HomogeneousSetType, HomogeneousListType)):
                    iterator = VariableIterator(local_var)
                    elem = Variable(var.class_type.element_type, self.scope.get_new_name())
                    idx = Variable(PythonNativeInt(), self.scope.get_new_name())
                    self.scope.insert_variable(elem)
                    assign = Assign(idx, LiteralInteger(0))
                    for_scope = self.scope.create_new_loop_scope()
                    for_body = [Assign(IndexedElement(ptr_var, idx), elem)]
                    if isinstance(local_var.class_type, HomogeneousSetType):
                        self.scope.insert_variable(idx)
                        for_body.append(Assign(idx, PyccelAdd(idx, LiteralInteger(1))))
                    else:
                        iterator.set_loop_counter(idx)
                    fill_for = For((elem,), iterator, for_body, scope = for_scope)
                    self._additional_exprs.extend([alloc, assign, fill_for])
                else:
                    raise errors.report(f"Don't know how to return an object of type {local_var.class_type} to C code.",
                            severity='fatal', symbol = var)
            else:
                ptr_var = var

            self._additional_exprs.append(CLocFunc(ptr_var, bind_var))

            return result
        else:
            return BindCFunctionDefResult(local_var, var, scope)

    def _wrap_Variable(self, expr):
        """
        Create all objects necessary to expose a module variable to C.

        Create and return the objects which must be printed in the wrapping
        module in order to expose the variable to C. In the case of scalar
        numerical values nothing needs to be done so an EmptyNode is returned.
        In the case of numerical arrays a C-compatible function must be created
        which returns the array. This is necessary because built-in Fortran
        arrays are not C-compatible. In the case of classes a C-compatible
        function is also created which returns a pointer to the class object.

        Parameters
        ----------
        expr : pyccel.ast.variables.Variable
            The module variable.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The AST object describing the code which must be printed in
            the wrapping module to expose the variable.
        """
        if isinstance(expr.class_type, FixedSizeNumericType):
            return expr.clone(expr.name, new_class = BindCVariable)
        elif isinstance(expr.class_type, NumpyNDArrayType):
            scope = self.scope
            func_name = scope.get_new_name('bind_c_'+expr.name.lower())
            func_scope = scope.new_child_scope(func_name)
            mod = expr.get_user_nodes(Module)[0]
            import_mod = Import(mod.name, AsName(expr,expr.name), mod=mod)
            func_scope.imports['variables'][expr.name] = expr

            # Create the data pointer
            bind_var = Variable(BindCPointer(),
                                scope.get_new_name('bound_'+expr.name),
                                is_const=True, memory_handling='alias')
            func_scope.insert_variable(bind_var)

            result = BindCFunctionDefResult(bind_var, expr, func_scope)
            if expr.rank == 0:
                #assigns = []
                #c_loc = CLocFunc(expr, bind_var)
                raise NotImplementedError("Classes cannot be wrapped")
            else:
                assigns = [Assign(result.shape[i], expr.shape[i]) for i in range(expr.rank)]
                c_loc = CLocFunc(expr, bind_var)
            body = [*assigns, c_loc]
            func = BindCFunctionDef(name = func_name,
                          body      = body,
                          arguments = [],
                          results   = [result],
                          imports   = [import_mod],
                          scope = func_scope,
                          original_function = expr)
            return expr.clone(expr.name, new_class = BindCArrayVariable, wrapper_function = func,
                                original_variable = expr)
        else:
            raise NotImplementedError(f"Objects of type {expr.class_type} cannot be wrapped yet")

    def _wrap_DottedVariable(self, expr):
        """
        Create all objects necessary to expose a class attribute to C.

        Create the getter and setter functions which expose the class attribute
        to C. Return these objects in a BindCClassProperty.

        Parameters
        ----------
        expr : DottedVariable
            The class attribute.

        Returns
        -------
        BindCClassProperty
            An object containing the getter and setter functions which expose
            the class attribute to C.
        """
        lhs = expr.lhs
        class_dtype = lhs.dtype
        # ----------------------------------------------------------------------------------
        #                        Create getter
        # ----------------------------------------------------------------------------------
        getter_name = self.scope.get_new_name(f'{class_dtype.name}_{expr.name}_getter'.lower())
        getter_scope = self.scope.new_child_scope(getter_name)
        self.scope = getter_scope
        self.scope.insert_symbol(expr.name)
        getter_result = self._wrap(FunctionDefResult(expr))

        getter_arg = self._wrap(FunctionDefArgument(lhs, bound_argument = True))
        self_obj = self._get_call_argument(getter_arg)

        getter_body = [C_F_Pointer(getter_arg.var, self_obj)]

        attrib = expr.clone(expr.name, lhs = self_obj)
        wrapped_obj = self.scope.find(expr.name)
        # Cast the C variable into a Python variable
        if expr.rank > 0 or isinstance(expr.dtype, CustomDataType):
            getter_body.append(AliasAssign(wrapped_obj, attrib))
        else:
            getter_body.append(Assign(getter_result.var, attrib))
        getter_body.extend(self._additional_exprs)
        self._additional_exprs.clear()
        self.exit_scope()

        getter = BindCFunctionDef(getter_name, (getter_arg,), getter_body, (getter_result,),
                                original_function = expr, scope = getter_scope)

        # ----------------------------------------------------------------------------------
        #                        Create setter
        # ----------------------------------------------------------------------------------
        setter_name = self.scope.get_new_name(f'{class_dtype.name}_{expr.name}_setter'.lower())
        setter_scope = self.scope.new_child_scope(setter_name)
        self.scope = setter_scope
        self.scope.insert_symbol(expr.name)

        setter_args = (self._wrap(FunctionDefArgument(lhs, bound_argument = True)),
                       self._wrap(FunctionDefArgument(expr)))
        if expr.is_alias:
            setter_args[1].persistent_target = True

        self_obj = self._get_call_argument(setter_args[0])
        set_val = self._get_call_argument(setter_args[1])

        setter_body = [C_F_Pointer(setter_args[0].var, self_obj)]

        if isinstance(set_val.dtype, CustomDataType):
            setter_body.append(C_F_Pointer(setter_args[1].var, set_val))
        elif isinstance(set_val, IndexedElement):
            func_arg = setter_args[1]
            size = func_arg.shape[::-1] if expr.order == 'C' else func_arg.shape
            stride = func_arg.strides[::-1] if expr.order == 'C' else func_arg.strides
            orig_size = [PyccelMul(sz,st) for sz,st in zip(size, stride)]
            setter_body.append(C_F_Pointer(func_arg.var, set_val.base, orig_size))

        attrib = expr.clone(expr.name, lhs = self_obj)
        # Cast the C variable into a Python variable
        if expr.memory_handling == 'alias':
            setter_body.append(AliasAssign(attrib, set_val))
        else:
            setter_body.append(Assign(attrib, set_val))
        self.exit_scope()

        setter = BindCFunctionDef(setter_name, setter_args, setter_body,
                                original_function = expr, scope = setter_scope)
        return BindCClassProperty(lhs.cls_base.scope.get_python_name(expr.name),
                                  getter, setter, lhs.dtype)

    def _wrap_ClassDef(self, expr):
        """
        Create all objects necessary to expose a class definition to C.

        Create all objects necessary to expose a class definition to C.

        Parameters
        ----------
        expr : ClassDef
            The class to be wrapped.

        Returns
        -------
        BindCClassDef
            The wrapped class.
        """
        name = expr.name
        func_name = self.scope.get_new_name(f'{name}_bind_c_alloc'.lower())
        func_scope = self.scope.new_child_scope(func_name)

        local_var = Variable(expr.class_type, func_scope.get_new_name(f'{name}_obj'),
                             cls_base = expr, memory_handling='alias')

        # Allocatable is not returned so it must appear in local scope
        func_scope.insert_variable(local_var)

        # Create the C-compatible data pointer
        bind_var = Variable(BindCPointer(),
                            func_scope.get_new_name('bound_'+name),
                            is_const=False, memory_handling='alias')
        func_scope.insert_variable(bind_var)

        result = BindCFunctionDefResult(bind_var, local_var, func_scope)

        # Define the additional steps necessary to define and fill ptr_var
        alloc = Allocate(local_var, shape=(), status='unallocated')
        c_loc = CLocFunc(local_var, bind_var)
        body = [alloc, c_loc]

        new_method = BindCFunctionDef(func_name, [], body, [result], original_function = None, scope = func_scope)

        methods = [self._wrap(m) for m in expr.methods]
        methods = [m for m in methods if not isinstance(m, EmptyNode)]
        for i in expr.interfaces:
            for f in i.functions:
                self._wrap(f)
        interfaces = [self._wrap(i) for i in expr.interfaces]

        if any(isinstance(v.class_type, TupleType) for v in expr.attributes):
            errors.report("Tuples cannot yet be exposed to Python.",
                    severity='warning',
                    symbol=expr)

        properties_getters = [BindCClassProperty(expr.scope.get_python_name(m.original_function.name),
                                                 m, None, expr.class_type, m.original_function.docstring)
                                for m in methods if 'property' in m.original_function.decorators]
        methods = [m for m in methods if m not in properties_getters
                        if 'property' not in m.original_function.decorators]

        # Pseudo-self variable is useful for pre-defined attributes which are not DottedVariables
        pseudo_self = Variable(expr.class_type, 'self', cls_base = expr)
        properties = [self._wrap(v if isinstance(v, DottedVariable) else v.clone(v.name, new_class = DottedVariable, lhs=pseudo_self)) \
                        for v in expr.attributes if not v.is_private and not isinstance(v.class_type, TupleType)]
        return BindCClassDef(expr, new_func = new_method, methods = methods,
                             interfaces = interfaces, attributes = properties_getters + properties,
                             docstring = expr.docstring, class_type = expr.class_type)
