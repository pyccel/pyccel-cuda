#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
This module contains all the CUDA thread indexing methods
"""
class CudaThreadIndexing:
    """
    Class representing the CUDA thread indexing.

    Class representing the CUDA thread indexing.
    """
    def __init__(self, block_idx, thread_idx):
        self._block_idx = block_idx
        self._thread_idx = thread_idx

    def threadIdx(self, dim):
        """
        Get the thread index.

        Get the thread index.

        Parameters
        -----------
        dim : int
            The dimension of the indexing. It can be:
            - 0 for the x-dimension
            - 1 for the y-dimension
            - 2 for the z-dimension

        Returns
        -----------------
        int
            The index of the thread in the specified dimension of its block.
        """
        return self._thread_idx

    def blockIdx(self, dim):
        """
        Get the block index.

        Get the block index.

        Parameters
        -----------
        dim : int
            The dimension of the indexing. It can be:
            - 0 for the x-dimension
            - 1 for the y-dimension
            - 2 for the z-dimension

        Returns
        -----------------
        int
            The index of the block in the specified dimension.
        """
        return self._block_idx

    def blockDim(self, dim):
        """
        Get the block dimension.

        Get the block dimension.

        Parameters
        -----------
        dim : int
            The dimension of the indexing. It can be:
            - 0 for the x-dimension
            - 1 for the y-dimension
            - 2 for the z-dimension

        Returns
        -----------------
        int
            The size of the block in the specified dimension.
        """
        return 0