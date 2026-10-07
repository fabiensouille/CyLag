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
import matplotlib.patches as patches
from ..geom.utils import polygon_area

cpdef init_positions_2d(poly, poly_area=None,
        xy_method=0, grid_res=np.array([5, 5]), xy_npart=25, xy_density=25.):
    """
    Initialize particles positions in 2D inside a 2D polygon shape.

    Parameters
    ----------
    poly : array_like, shape (n, 2)
        Coordinates of the polygon vertices.
    poly_area : float, optional
        Area of the polygon (default is computed if not provided).
    xy_method : int, optional
        Initialization method.
        0: use ``grid_res`` for number of particles along x and y axes.
        1: use ``xy_npart`` for total number of particles.
        2: use ``xy_density`` for number of particles per unit area.
    grid_res : array_like, shape (3,) or (2,), optional
        Number of points on the x and y axis.
    xy_npart : int, optional
        Number of particles in the 2D polygon.
    xy_density : float, optional
        Particle density per unit area.

    Returns
    -------
    ndarray, shape (m, 2)
        Positions of particles inside the polygon.
    """
    cdef:
        int i, j, ni, nj
        double x, y
        double dxy
        double[:] grid_xlim = np.array([np.min(poly[:, 0]), np.max(poly[:, 0])])
        double[:] grid_ylim = np.array([np.min(poly[:, 1]), np.max(poly[:, 1])])

    # Exceptions
    if poly.shape[0]<3:
        raise ValueError("Invalid polygon")

    # compute polygon area if not provided
    if poly_area==None:
        poly_area = polygon_area(poly[:, 0], poly[:, 1])

    # define grid res
    if xy_method==0:
        ni = grid_res[0] + 1
        nj = grid_res[1] + 1
    elif xy_method==1:
        dxy = 1./np.sqrt(xy_npart/poly_area)
        ni = int((grid_xlim[1] - grid_xlim[0])/dxy) + 1
        nj = int((grid_ylim[1] - grid_ylim[0])/dxy) + 1
    elif xy_method==2:
        dxy = 1./np.sqrt(xy_density)
        ni = int((grid_xlim[1] - grid_xlim[0])/dxy) + 1
        nj = int((grid_ylim[1] - grid_ylim[0])/dxy) + 1
    else:
        raise ValueError("Invalid initialization method")

    xl = np.linspace(grid_xlim[0], grid_xlim[1], ni)
    yl = np.linspace(grid_ylim[0], grid_ylim[1], nj)

    # define particle positions in each cell of the grid
    npart = (ni-1)*(nj-1)
    positions = np.zeros((npart, 2), dtype='d')

    for i in range(ni-1):
        for j in range(nj-1):
            k = i + j*(ni-1)
            positions[k, 0] = 0.5*(xl[i] + xl[i+1])
            positions[k, 1] = 0.5*(yl[j] + yl[j+1])

    # keep only particules inside polygon
    polygon = patches.Polygon(poly)
    idx_to_delete = []
    for i in range(npart):
        inside = polygon.contains_point((positions[i, 0], positions[i, 1]))
        if not inside:
            idx_to_delete.append(i)
    positions = np.delete(positions, idx_to_delete, 0)

    del polygon
    return positions

cpdef init_positions_3d(poly, poly_area=None,
        xy_method=0, grid_res=np.array([5, 5, 1]), xy_npart=25, xy_density=25.,
        z_method=0, z_npart=1, z_density=1., zmin=0., zmax=1.,):
    """
    Initialize particles positions in 3D inside a 2D polygon shape.
    
    Parameters
    ----------
    poly : array_like, shape (n, 2)
        Coordinates of the polygon vertices.
    poly_area : float, optional
        Area of the polygon (default is computed if not provided).
    xy_method : int, optional
        Initialization method in the polygon plane.
        0: use ``grid_res`` for number of particles along x and y axes.
        1: use ``xy_npart`` for total number of particles.
        2: use ``xy_density`` for number of particles per unit area.
    grid_res : array_like, shape (3,), optional
        Number of points on the x, y, and z axis.
    xy_npart : int, optional
        Number of particles in the polygon.
    xy_density : float, optional
        Particle density per unit area.
    z_method : int, optional
        Initialization method in the z direction.
        0: use ``grid_res`` for number of particles along z.
        1: use ``z_npart`` for number of particle layers.
        2: use ``z_density`` for number of particles per unit length.
    z_npart : int, optional
        Number of particle layers in the z direction.
    z_density : float, optional
        Particle density per unit length in z.
    zmin : float, optional
        Minimum z coordinate.
    zmax : float, optional
        Maximum z coordinate.

    Returns
    -------
    ndarray, shape (m, 3)
        Positions of particles inside the polygon and z range.
    """
    cdef:
        int i, j, l, k
        int ni, nj, nl
        double x, y, z
        double dxy
        double[:] grid_xlim = np.array([np.min(poly[:, 0]), np.max(poly[:, 0])])
        double[:] grid_ylim = np.array([np.min(poly[:, 1]), np.max(poly[:, 1])])

    # Exceptions
    if poly.shape[0]<3:
        raise ValueError("Invalid polygon")

    # compute polygon area if not provided
    if poly_area==None:
        poly_area = polygon_area(poly[:, 0], poly[:, 1])

    # define grid res
    if xy_method==0:
        ni = grid_res[0]+1
        nj = grid_res[1]+1
    elif xy_method==1:
        dxy = 1./np.sqrt(xy_npart/poly_area)
        ni = int((grid_xlim[1] - grid_xlim[0])/dxy) + 1
        nj = int((grid_ylim[1] - grid_ylim[0])/dxy) + 1
    elif xy_method==2:
        dxy = 1./np.sqrt(xy_density)
        ni = int((grid_xlim[1] - grid_xlim[0])/dxy) + 1
        nj = int((grid_ylim[1] - grid_ylim[0])/dxy) + 1
    else:
        raise ValueError("Invalid initialization method")

    if z_method==0:
        nl = grid_res[2]+1
    elif z_method==1:
        nl = z_npart + 1
    elif z_method==2:
        nl = int(z_density*(zmax - zmin)) + 1
    else:
        raise ValueError("Invalid initialization method")

    # define particle positions on grid
    npart = (ni-1)*(nj-1)*(nl-1)
    positions = np.zeros((npart, 3), dtype='d')

    xl = np.linspace(grid_xlim[0], grid_xlim[1], ni) 
    yl = np.linspace(grid_ylim[0], grid_ylim[1], nj)
    zl = np.linspace(zmin, zmax, nl)

    for i in range(ni-1):
        for j in range(nj-1):
            for l in range(nl-1):
                k = i + j*(ni-1) + l*(ni-1)*(nj-1)
                positions[k, 0] = 0.5*(xl[i] + xl[i+1])
                positions[k, 1] = 0.5*(yl[j] + yl[j+1])
                positions[k, 2] = 0.5*(zl[l] + zl[l+1])

    # keep only particules inside polygon
    polygon = patches.Polygon(poly)
    idx_to_delete = []
    for i in range(npart):
        inside = polygon.contains_point((positions[i, 0], positions[i, 1]))
        if not inside:
            idx_to_delete.append(i)
    positions = np.delete(positions, idx_to_delete, 0)

    del polygon
    return positions

cpdef double[:,:] circle(double x0, double y0, double r, int n):
    cdef double[:,:] poly
    cdef double dtheta
    cdef int i
    
    poly = np.zeros((n, 2), dtype='d')
    dtheta = 2.*np.pi/n
    
    for i in range(n):
        theta = i*dtheta
        poly[i, 0] = x0 + r*np.cos(theta)
        poly[i, 1] = y0 + r*np.sin(theta)

    return poly
