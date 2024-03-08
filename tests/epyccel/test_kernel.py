# pylint: disable=missing-function-docstring, missing-module-docstring

import pytest
from pyccel.epyccel import epyccel
from pyccel.decorators import kernel
from pyccel import cuda

#------------------------------------------------------------------------------
@pytest.fixture(params=[
        pytest.param("cuda", marks=pytest.mark.cuda),
    ]
)
def language(request):
    return request.param

#==============================================================================

@pytest.mark.gpu
def test_kernel(language, capsys):
    @kernel
    def hello_from_kernel():
        print("Hello from GPU !")

    def f():
        hello_from_kernel[1, 1]()
        cuda.synchronize()

    epyc_f = epyccel(f, language=language)
    epyc_f()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello from GPU !"
