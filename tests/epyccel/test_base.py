# pylint: disable=missing-function-docstring, missing-module-docstring
import os

import pytest

from modules import base
from utilities import epyccel_test


def test_is_false(language_with_cuda):
    test = epyccel_test(base.is_false, lang=language_with_cuda)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_is_true(language_with_cuda):
    test = epyccel_test(base.is_true, lang=language_with_cuda)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_compare_is(language_with_cuda):
    test = epyccel_test(base.compare_is, lang=language_with_cuda)
    test.compare_epyccel( True, True )
    test.compare_epyccel( True, False )
    test.compare_epyccel( False, True )
    test.compare_epyccel( False, False )

def test_compare_is_not(language_with_cuda):
    test = epyccel_test(base.compare_is_not, lang=language_with_cuda)
    test.compare_epyccel( True, True )
    test.compare_epyccel( True, False )
    test.compare_epyccel( False, True )
    test.compare_epyccel( False, False )

def test_compare_is_int(language_with_cuda):
    test = epyccel_test(base.compare_is_int, lang=language_with_cuda)
    test.compare_epyccel( True, 1 )
    test.compare_epyccel( True, 0 )
    test.compare_epyccel( False, 1 )
    test.compare_epyccel( False, 0 )

def test_compare_is_not_int(language_with_cuda):
    test = epyccel_test(base.compare_is_not_int, lang=language_with_cuda)
    test.compare_epyccel( True, 1 )
    test.compare_epyccel( True, 0 )
    test.compare_epyccel( False, 1 )
    test.compare_epyccel( False, 0 )

def test_not_false(language_with_cuda):
    test = epyccel_test(base.not_false, lang=language_with_cuda)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_not_true(language_with_cuda):
    test = epyccel_test(base.not_true, lang=language_with_cuda)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_eq_false(language_with_cuda):
    test = epyccel_test(base.eq_false, lang=language_with_cuda)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_eq_true(language_with_cuda):
    test = epyccel_test(base.eq_true, lang=language_with_cuda)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_neq_false(language_with_cuda):
    test = epyccel_test(base.eq_false, lang=language_with_cuda)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_neq_true(language_with_cuda):
    test = epyccel_test(base.eq_true, lang=language_with_cuda)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_not(language_with_cuda):
    test = epyccel_test(base.not_val, lang=language_with_cuda)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_not_int(language_with_cuda):
    test = epyccel_test(base.not_int, lang=language_with_cuda)
    test.compare_epyccel( 0 )
    test.compare_epyccel( 4 )

def test_compare_is_nil(language_with_cuda):
    test = epyccel_test(base.is_nil, lang=language_with_cuda)
    test.compare_epyccel( None )

def test_compare_is_not_nil(language_with_cuda):
    test = epyccel_test(base.is_not_nil, lang=language_with_cuda)
    test.compare_epyccel( None )

def test_cast_int(language_with_cuda):
    test = epyccel_test(base.cast_int, lang=language_with_cuda)
    test.compare_epyccel( 4 )
    test = epyccel_test(base.cast_float_to_int, lang=language_with_cuda)
    test.compare_epyccel( 4.5 )

def test_cast_bool(language_with_cuda):
    test = epyccel_test(base.cast_bool, lang=language_with_cuda)
    test.compare_epyccel( True )

def test_cast_float(language_with_cuda):
    test = epyccel_test(base.cast_float, lang=language_with_cuda)
    test.compare_epyccel( 4.5 )
    test = epyccel_test(base.cast_int_to_float, lang=language_with_cuda)
    test.compare_epyccel( 4 )

def test_if_0_int(language_with_cuda):
    test = epyccel_test(base.if_0_int, lang=language_with_cuda)
    test.compare_epyccel( 22 )
    test.compare_epyccel( 0 )

def test_if_0_real(language_with_cuda):
    test = epyccel_test(base.if_0_real, lang=language_with_cuda)
    test.compare_epyccel( 22.3 )
    test.compare_epyccel( 0.0 )

def test_same_int(language_with_cuda):
    test = epyccel_test(base.is_same_int, lang=language_with_cuda)
    test.compare_epyccel( 22 )
    test = epyccel_test(base.isnot_same_int, lang=language_with_cuda)
    test.compare_epyccel( 22 )

def test_same_float(language_with_cuda):
    test = epyccel_test(base.is_same_float, lang=language_with_cuda)
    test.compare_epyccel( 22.2 )
    test = epyccel_test(base.isnot_same_float, lang=language_with_cuda)
    test.compare_epyccel( 22.2 )

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Strings are not yet implemented for C language"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_same_string(language):
    test = epyccel_test(base.is_same_string, lang=language)
    test.compare_epyccel()
    test = epyccel_test(base.isnot_same_string, lang=language)
    test.compare_epyccel()

def test_same_complex(language):
    test = epyccel_test(base.is_same_complex, lang=language)
    test.compare_epyccel( complex(2,3) )
    test = epyccel_test(base.isnot_same_complex, lang=language)
    test.compare_epyccel( complex(2,3) )

def test_is_types(language_with_cuda):
    test = epyccel_test(base.is_types, lang=language_with_cuda)
    test.compare_epyccel( 1, 1.0 )

def test_isnot_types(language_with_cuda):
    test = epyccel_test(base.isnot_types, lang=language_with_cuda)
    test.compare_epyccel( 1, 1.0 )

def test_none_is_none(language_with_cuda):
    test = epyccel_test(base.none_is_none, lang=language_with_cuda)
    test.compare_epyccel()

def test_none_isnot_none(language_with_cuda):
    test = epyccel_test(base.none_isnot_none, lang=language_with_cuda)
    test.compare_epyccel()

def test_pass_if(language_with_cuda):
    test = epyccel_test(base.pass_if, lang=language_with_cuda)
    test.compare_epyccel(2)

def test_pass2_if(language_with_cuda):
    test = epyccel_test(base.pass2_if, lang=language_with_cuda)
    test.compare_epyccel(0.2)
    test.compare_epyccel(0.0)

optional_marks = [
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python),
    ]
if os.environ.get('PYCCEL_DEFAULT_COMPILER', None) == 'intel':
    optional_marks.append(pytest.param("fortran", marks = [
            pytest.mark.skip(reason="Intel does not use lazy evaluation. See #1668"),
            pytest.mark.fortran]))
else:
    optional_marks.append(pytest.param("fortran", marks = pytest.mark.fortran))

@pytest.mark.parametrize( 'language', optional_marks)
def test_use_optional(language):
    test = epyccel_test(base.use_optional, lang=language)
    test.compare_epyccel()
    test.compare_epyccel(6)

def test_none_equality(language_with_cuda):
    test = epyccel_test(base.none_equality, lang=language_with_cuda)
    test.compare_epyccel()
    test.compare_epyccel(6)

def test_none_none_equality(language_with_cuda):
    test = epyccel_test(base.none_none_equality, lang=language_with_cuda)
    test.compare_epyccel()

def test_none_literal_equality(language_with_cuda):
    test = epyccel_test(base.none_literal_equality, lang=language_with_cuda)
    test.compare_epyccel()
