import pytest
import numpy as np
from pyccel.epyccel import epyccel
from pyccel.decorators import kernel

#------------------------------------------------------------------------------
@pytest.fixture(params=[
        pytest.param("cuda", marks=pytest.mark.cuda),
    ]
)
def language(request):
    return request.param

#==============================================================================

@pytest.mark.gpu
def test_kernel(language):
    @kernel
    def add_one_kernel(a: int):
        print(a)

    def f():
        add_one_kernel[1, 1](1)
        return 1

    epyc_f = epyccel(f, language=language)
    assert epyc_f() == 1
