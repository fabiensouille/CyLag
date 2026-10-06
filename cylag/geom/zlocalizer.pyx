# cython: profile=False
#
# -------------------------------------------------------------------------------------------------
#  Project name: CyLag
#  Copyright (C) 2023 Fabien Souille
#
#  This program is free: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License published by the Free Software Foundation,
#  either version 3 of the license, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License for details.
#
#  You should have received a copy of the GNU General Public License
#  with this program. If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------------------------------
#
import numpy as np

# Search lower lawer index of z in zlayers from nlayers with constant dz
def compute_lower_layer_index(nlayers, zs, zb, z):
    """ 
    Search greatest layer such as z(layer) < z 
    
    Parameters
    ----------
    nlayers : int
        Number of layers.
    zs : float
        Surface elevation.
    zb : float
        Bottom elevation.
    z : float
        Elevation to locate in layers.

    Returns
    -------
    index : int
        Index of the lower layer such that z(layer) < z.
    """
    hl = (zs-zb)/max((nlayers-1), 1)
    index = (z-zb)//hl
    if isinstance(index, float):
        return int(index)
    if isinstance(index, np.ndarray):
        return index.astype(int)

cpdef int c_compute_lower_layer_index(int nlayers, double zs, double zb, double z):
    cdef:
        double hl
        double findex
    if nlayers == 1:
        return 0
    hl = (zs-zb)/(nlayers-1)
    # degenerate/dry water column (zs <= zb)
    if not (hl > 0.):
        return 0
    findex = (z-zb)//hl
    # clamp in double space before casting to int: casting an out-of-range
    # or huge double to int is undefined behaviour (can silently yield
    # garbage such as INT_MIN), which later causes out-of-bounds memory
    # access when used to index field arrays. Checks use "not(valid)" form
    # so that a NaN findex (e.g. from a NaN z) also falls back safely.
    if not (findex >= 0.):
        return 0
    if not (findex <= <double>(nlayers-2)):
        return nlayers-2
    return int(findex)

cpdef int[:] c_compute_lower_layer_index_array(int nlayers, double[:] zs, double[:] zb, double[:] z):
    """ 
    Search greatest layer such as z(layer) < z 
    
    Parameters
    ----------
    nlayers : int
        Number of layers.
    zs : array_like, shape (n,)
        Surface elevations.
    zb : array_like, shape (n,)
        Bottom elevations.
    z : array_like, shape (n,)
        Elevations to locate in layers.
    
    Returns
    -------
    index : ndarray, shape (n,)
        Indices of the lower layers such that z(layer) < z.
    """
    cdef:
        int i
        int ni = len(z)
    res = np.zeros((ni), dtype='int32')
    for i in range(ni):
        res[i] = c_compute_lower_layer_index(nlayers, zs[i], zb[i], z[i])
    return res

# Search lower index of value in array (non-constant spacing of values in array)
def compute_lower_index(array, value, last_idx=0):
    """ Search indices i such as array(i) <= value 
    
    Parameters
    ----------
    array : array_like, shape (n,)
        Array of values to search in.
    value : float
        Value to locate in the array.
    last_idx : int, optional
        Last index to start the search from (default is 0).

    Returns
    -------
    index : int
        Index i such that array[i] <= value < array[i+1].
    """
    cdef int i
    cdef int ni = len(array)
    for i in range(last_idx, ni-1):
        if value >= array[i] and value < array[i+1]:
            return i

cpdef int c_compute_lower_index(double[:] array, double value, int last_idx):
    """ 
    Search indices i such as array(i) <= value 
    
    Parameters
    ----------
    array : array_like, shape (n,)
        Array of values to search in.
    value : float
        Value to locate in the array.
    last_idx : int, optional
        Last index to start the search from (default is 0).

    Returns
    -------
    index : int
        Index i such that array[i] <= value < array[i+1].
    """
    cdef int i
    cdef int ni = len(array)
    for i in range(last_idx, ni-1):
        if value >= array[i] and value < array[i+1]:
            return i
