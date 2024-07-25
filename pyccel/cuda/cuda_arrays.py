#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
This submodule contains cuda_arrays methods for Pyccel.
"""

def host_empty(shape, dtype = 'float', order = 'C'):
    """
    Create an empty array on the host.

    Create an empty array on the host.

    Parameters
    ----------
    shape : tuple of int or int
        The shape of the array.

    dtype : str, optional
        The data type of the array. The default is 'float'.

    order : str, optional
        The order of the array. The default is 'C'.

    Returns
    -------
    array
        The empty array on the host.
    """
    import numpy as np
    a = np.empty(shape, dtype = dtype, order = order)
    return a
def device_empty(shape):
    """
    Create an empty array on the device.

    Create an empty array on the device.

    Parameters
    ----------
    shape : tuple of int or int
        The shape of the array.

    Returns
    -------
    array
        The empty array on the device.
    """
    import numpy as np
    a = np.empty(shape)
    return a

