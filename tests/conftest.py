# pylint: disable=missing-function-docstring, missing-module-docstring
import logging
import os
import shutil
import pytest
from mpi4py import MPI
from pyccel.commands.pyccel_clean import pyccel_clean

github_debugging = 'DEBUG' in os.environ
if github_debugging:
    import sys
    sys.stdout = sys.stderr

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    ],
    scope = "session"
)
def language(request):
    return request.param

@pytest.fixture( params=[
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python),
    ],
    scope = "session"
)
def stc_language(request):
    return request.param

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("cuda", marks = pytest.mark.cuda)
    ],
    scope = "session"
)
def language_with_cuda(request):
    return request.param

def move_coverage(path_dir):
    for root, _, files in os.walk(path_dir):
        for name in files:
            if name.startswith(".coverage"):
                shutil.copyfile(os.path.join(root,name),os.path.join(os.getcwd(),name))

def pytest_runtest_teardown(item, nextitem):
    path_dir = os.path.dirname(os.path.realpath(item.fspath))
    move_coverage(path_dir)

    config = item.config
    xdist_plugin = config.pluginmanager.getplugin("xdist")
    if xdist_plugin is None or "PYTEST_XDIST_WORKER_COUNT" not in os.environ \
            or os.getenv('PYTEST_XDIST_WORKER_COUNT') == 1:
        print("Tearing down!")
        marks = [m.name for m in item.own_markers ]
        if 'parallel' not in marks:
            pyccel_clean(path_dir, remove_shared_libs = True)
        else:
            comm = MPI.COMM_WORLD
            comm.Barrier()
            if comm.rank == 0:
                pyccel_clean(path_dir, remove_shared_libs = True)
            comm.Barrier()

def pytest_addoption(parser):
    parser.addoption("--developer-mode", action="store_true", default=github_debugging, help="Show tracebacks when pyccel errors are raised")
    parser.addoption("--gpu_available", action="store_true",
                default=False, help="enable GPU tests")

def pytest_generate_tests(metafunc):
    if "gpu_available" in metafunc.fixturenames:
        if metafunc.config.getoption("gpu_available"):
            metafunc.parametrize("gpu_available", [True])
        else:
            metafunc.parametrize("gpu_available", [False])

def pytest_sessionstart(session):
    # setup_stuff
    if session.config.option.developer_mode:
        from pyccel.errors.errors import ErrorsMode
        ErrorsMode().set_mode('developer')

    if github_debugging:
        logging.basicConfig()
        logging.getLogger("filelock").setLevel(logging.DEBUG)

    # Clean path before beginning but never delete anything in parallel mode
    path_dir = os.path.dirname(os.path.realpath(__file__))

    config = session.config
    xdist_plugin = config.pluginmanager.getplugin("xdist")
    if xdist_plugin is None:
        marks = [m.name for m in session.own_markers ]
        if 'parallel' not in marks:
            pyccel_clean(path_dir)
