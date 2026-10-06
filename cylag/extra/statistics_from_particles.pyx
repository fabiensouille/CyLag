# -*- coding: utf-8 -*-
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
from scipy.interpolate import interp1d
from ..extra.ghost_meshgrid cimport GhostMeshGrid1D
from ..extra.ghost_meshgrid cimport GhostMeshGrid2D

# KERNEL SMOOTHING FUNCTIONS
cpdef double sph_smoothing_kernel(double h, double d, int function, int dim):
    """
    Smoothing kernel.

    Parameters
    ----------
    h : float
        Smoothing length.
    d : float
        Distance.
    function : int
        Type of smoothing kernel.
        0: constant, 1_{d<r} (order 0)
        1: linear, max(0, r-d) (order 1)
        2: quadratic, max(0, r-d)**2 (order 1)
        3: cubic spline, max(0, r**2-d**2)**3 (high order)
        4: cubic spline, Monaghan (1992) (high order)
        5: Gaussian, Liu (2010) (high order)
    dim : int
        Spatial dimension (1, 2, or 3).

    Returns
    -------
    float
        Kernel weight.

    References
    ----------
    J. Monaghan, Smoothed Particle Hydrodynamics,
        Annual Review of Astronomy and Astrophysics, 30 (1992), pp. 543-574.
    M. Liu and G. Liu, Smoothed particle hydrodynamics (SPH): an overview and recent developments,
        Archives of Computational Methods in Engineering, 17.1 (2010), pp. 25-76.
    """
    cdef:
        double value, coef

    if dim==3:
        assert function>=4
    
    # Constant (order 0)
    # ~~~~~~~~~~~~~~~~~
    if function==0:
        if d>h:
            value = 0.
        else:
            value = h
        if dim==2:
            coef = np.pi*2.*h**2
        else:
            coef = 2.*h**2
        wq = value/coef
    # Linear (order 1)
    # ~~~~~~~~~~~~~~~~
    elif function==1:
        value = max(0., h - d)
        if dim==2:
            coef = np.pi*h**2
        else:
            coef = h**2
        wq = value/coef
    # Quadratic (order 2)
    # ~~~~~~~~~~~~~~~~~~~
    elif function==2:
        value = max(0., h - d)**2
        if dim==2:
            coef = np.pi*(2./3.)*h**3
        else:
            coef = (2./3.)*h**3
        wq = value/coef
    # Cubic spline (5th order)
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    elif function==3:
        value = max(0., (h**2 - d**2))**3
        if dim==2:
            coef = np.pi*2.*((16./35.)*h**7)
        else:
            coef = 2.*(16./35.)*h**7
        wq = value/coef
    # Cubic spline (Monaghan)
    # ~~~~~~~~~~~~~~~~~~~~~~~
    elif function==4:
        h = 0.5*h
        q = d/h
        if dim==3:
            coef = 1./(np.pi*h**3)
        elif dim==2:
            coef = 10./(7.*np.pi*h**2)
        else:
            coef = 2./(3.*h)
        if 0.<=q and q<=1.:
            wq = coef*(1.-(3./2.)*(q**2)*(1-0.5*q))
        elif 1.<q and q<=2.:
            wq = coef*0.25*(2-q)**3
        else:
            wq = 0.
    # Gaussian (Liu)
    # ~~~~~~~~~~~~~~
    elif function==5:
        h = 0.5*h
        q = d/h
        if dim==3:
            coef = 1./((np.pi**(3./2.))*(h**3))
        elif dim==2:
            coef = 1./(np.pi*(h**2))
        else:
            coef = 1./(np.sqrt(np.pi)*h)
        if 0.<=q and q<=3.:
            wq = coef*np.exp(-q**2)
        else:
            wq = 0.
    else:
        raise ValueError('Unknown smoothing function')

    return wq

def initialize_particles_from_pdf_1D(x, f, npart, epsilon=0., sampling=1):
    """
    Initialize particle positions from a density function using the CDF (1D).

    Parameters
    ----------
    x : array_like
        Abscissa values.
    f : array_like
        Density function values ``f(x)``.
    npart : int
        Number of particles.
    epsilon : float, optional
        Margin to avoid particles at the extrema of ``f``.
    sampling : int, optional
        Sampling method: 0 for linear, 1 for random.

    Returns
    -------
    xp : ndarray
        Particle positions.
    massp : float
        Pseudo mass associated with particles.
    cdf : ndarray
        CDF of the initializing function.
    """
    cdef int nf
    cdef double integral_f
    cdef double massp

    nf = len(f)

    # computing cdf and integral of f function
    cdf = np.cumsum(f)
    cdf = cdf/cdf[nf-1]
    integral_f = np.trapz(f, x)
    massp = integral_f/npart
    # sampling cdf between min and max
    if sampling==0:
        # (epsilon to avoid point at cdf = 0 or 1)
        cdf_max = np.max(cdf) - epsilon 
        cdf_min = np.min(cdf) + epsilon
        proba = np.linspace(cdf_min, cdf_max, npart)
    else:
        proba = np.random.uniform(np.min(cdf), np.max(cdf), npart)
    # interpolation from cdf
    interp = interp1d(cdf, x)
    xp = interp(proba)

    return xp, massp, cdf

# 1D STATISTICS
cpdef double[:] compute_density_1D_NGP(
        double[:] x, double[:] xp, double massp):
    """
    Compute density from particle positions using the Nearest Grid Point method (NGP).

    Parameters
    ----------
    x : array_like
        Mesh points.
    xp : array_like
        Particle positions.
    massp : float
        Pseudo mass of particles for normalization.

    Returns
    -------
    ndarray
        Density at mesh points.
    """
    cdef:
        int i, k
        int npart = len(xp)
        int nx = len(x)
        double influence, xa, xb
        double[:] density = np.zeros(nx)

    # loop over cells
    for i in range(nx):
        xa = x[max(i-1, 0)]
        xb = x[min(i+1, len(x)-1)]
        for k in range(npart):
            if 0.5*(xa+x[i])<=xp[k] and xp[k]<=0.5*(x[i]+xb):
                influence = 2/(xb-xa)
                density[i] += massp*influence
    return density

cpdef double[:] compute_density_1D_CIC(
        double[:] x, double[:] xp, double massp):
    """
    Compute density from particle positions using the Cloud In Cell method (CIC).

    Parameters
    ----------
    x : array_like
        Mesh points.
    xp : array_like
        Particle positions.
    massp : float
        Pseudo mass of particles for normalization.

    Returns
    -------
    ndarray
        Density at mesh points.
    """
    cdef:
        int i, k
        int npart = len(xp)
        int nx = len(x)
        double sigma, xa, xb
        double influence
        double[:] density = np.zeros(nx)

    # loop over cells
    for i in range(nx):
        xa = x[max(i-1, 0)]
        xb = x[min(i+1, len(x)-1)]
        for k in range(npart):
            if 0.5*(xa+x[i])<=xp[k] and xp[k]<=0.5*(x[i]+xb):
                dist = abs(xp[k]-x[i])
                if xp[k]>=x[i]:
                    sigma = 4./((xb-xa)*(x[i]+xb))
                    influence = sigma*(0.5*(x[i]+xb)-dist)
                else:
                    sigma = 4./((xb-xa)*(xa+x[i]))
                    influence = sigma*(0.5*(xa+x[i])-dist)
                density[i] += massp*influence
    return density

cpdef double[:] compute_density_1D_KS(
        double[:] x, double[:] xp, 
        double massp, double h, int kernel):
    """
    Compute density from particle positions using the Kernel Smoothing method (KS).

    Parameters
    ----------
    x : array_like
        Mesh points.
    xp : array_like
        Particle positions.
    massp : float
        Pseudo mass of particles for normalization.
    h : float
        Smoothing length.
    kernel : int
        Smoothing kernel identifier.

    Returns
    -------
    ndarray
        Density at mesh points.
    """
    cdef:
        int i, k
        int npart = len(xp)
        int nx = len(x)
        double influence
        double dist
        double[:] density = np.zeros(nx)

    # loop over cells
    for i in range(nx):
        # loop over particles
        for k in range(npart):
            dist = abs(xp[k]-x[i])
            if dist<h:
                influence = sph_smoothing_kernel(h, dist, kernel, 1)
                density[i] += massp*influence
    return density

# 1D STATISTICS - GHOST MESH GRID
cpdef double[:] compute_density_1D_KS_GM(
        double[:] x, double[:] xp, 
        double massp, double h, int kernel):
    """
    Compute density from particle positions using the Kernel Smoothing method (KS) with ghost mesh grid.

    Parameters
    ----------
    x : array_like
        Mesh points.
    xp : array_like
        Particle positions.
    massp : float
        Pseudo mass of particles for normalization.
    h : float
        Smoothing length.
    kernel : int
        Smoothing kernel identifier.

    Returns
    -------
    ndarray
        Density at mesh points.
    """
    cdef:
        int i, j, k
        int gcx, gc, ngcf
        int npart = len(xp)
        int nx = len(x)
        double dist, influence
        double[:] density = np.zeros(nx)

    # initialize lookup tables
    ghost_mesh = GhostMeshGrid1D(np.min(x), np.max(x), h)
    xp = ghost_mesh.set_lookup_tables(xp)
    ngcf = len(ghost_mesh.cellid_filled)

    # loop over nodes on which to compute density
    for i in range(nx):
        # get ghost cell of x
        gcx = ghost_mesh.get_ghostcell_id(x[i])

        # loop over ghost cells containing particles
        for j in range(ngcf):
            # ghost cell id
            gc = ghost_mesh.cellid_filled[j]

            # check if gc must contribute to gcx (in neighborhood)
            if gc==gcx-1 or gc==gcx or gc==gcx+1:

                # loop over contributing particles
                for k in range(ghost_mesh.cellid_jump[j],\
                               ghost_mesh.cellid_jump[j+1]):
                    dist = abs(xp[k]-x[i])
                    influence = sph_smoothing_kernel(h, dist, kernel, 1)
                    density[i] += massp*influence
    return density

# 2D STATISTICS
cpdef double[:] compute_density_2D_KS(
        double[:] x, double[:] y, double[:,:] positions, 
        double massp, double h, int kernel):
    """
    Compute density from particles at positions (x, y) using the Kernel Smoothing method (KS).

    Parameters
    ----------
    x : array_like
        Mesh points x coordinates.
    y : array_like
        Mesh points y coordinates.
    positions : array_like, shape (n, 2)
        Particle positions.
    massp : float
        Pseudo mass of particles for normalization.
    h : float
        Smoothing length.
    kernel : int
        Smoothing kernel identifier.

    Returns
    -------
    ndarray
        Density at mesh points.
    """
    cdef:
        int i, k
        int npart = len(positions)
        int npoints = len(x)
        double dist, influence
        double[:] density = np.zeros(npoints)

    # loop over cells
    for i in range(npoints):
        for k in range(npart):
            dist = np.sqrt((positions[k, 0]-x[i])**2 \
                         + (positions[k, 1]-y[i])**2)
            if dist<h:
                influence = sph_smoothing_kernel(h, dist, kernel, 2)
                density[i] += massp*influence
    return density

# 2D STATISTICS - GHOSTMESH
cpdef double[:] compute_density_2D_KS_GM(
        double[:] x, double[:] y, double[:,:] positions, 
        double massp, double h, int kernel):
    """
    Compute density from particles at positions (x, y) using the Kernel Smoothing method (KS) with ghost mesh handling.

    Parameters
    ----------
    x : array_like
        Mesh point x coordinates.
    y : array_like
        Mesh point y coordinates.
    positions : array_like, shape (n, 2)
        Particle positions.
    massp : float
        Pseudo mass of particles for normalization.
    h : float
        Smoothing length.
    kernel : int
        Smoothing kernel identifier.

    Returns
    -------
    ndarray
        Density at mesh points.
    """
    cdef:
        int i, j, k
        int gcx, gc, ngcf
        int npart = len(positions)
        int npoints = len(x)
        double dist, influence
        long[:] gcidf, gcidj
        double[:,:] pos
        double[:] density = np.zeros(npoints)

    # initialize lookup tables
    ghost_mesh = GhostMeshGrid2D(np.min(x), np.max(x),\
                                 np.min(y), np.max(y), h)
    pos = ghost_mesh.set_lookup_tables(positions)
    ngcf = len(ghost_mesh.cellid_filled)

    # loop over nodes on which to compute density
    for i in range(npoints):
        # get ghost cell of x
        gcx = ghost_mesh.get_ghostcell_id(np.asarray([x[i], y[i]]))

        # neighbors of point i (x9 cells)
        neigh = np.array([gcx-1, gcx, gcx+1])
        neigh = np.concatenate((\
            neigh,\
            neigh+ghost_mesh.ngx,\
            neigh-ghost_mesh.ngx), axis=0)

        # loop over ghost cells containing particles
        for j in range(ngcf):
            # ghost cell id
            gc = ghost_mesh.cellid_filled[j]

            # check if gc must contribute to gcx (in neighborhood)
            if gc in neigh:

                for k in range(ghost_mesh.cellid_jump[j],\
                               ghost_mesh.cellid_jump[j+1]):
                    dist = np.sqrt((pos[k, 0]-x[i])**2 +\
                                   (pos[k, 1]-y[i])**2)
                    influence = sph_smoothing_kernel(h, dist, kernel, 2)
                    density[i] += massp*influence

    return density

# 3D STATISTICS
cpdef double[:] compute_density_3D_KS(
        double[:] x, double[:] y, double[:] z, 
        double[:,:] positions, 
        double massp, double h, int kernel):
    """
    Compute density from particles at positions (x, y, z) using the Kernel Smoothing method (KS).

    Parameters
    ----------
    x : array_like
        Mesh points x coordinates.
    y : array_like
        Mesh points y coordinates.
    z : array_like
        Mesh points z coordinates.
    positions : array_like, shape (n, 3)
        Particle positions.
    massp : float
        Pseudo mass of particles for normalization.
    h : float
        Smoothing length.
    kernel : int
        Smoothing kernel identifier.

    Returns
    -------
    ndarray
        Density at mesh points.
    """
    cdef:
        int i, k
        int npart = len(positions)
        int npoints = len(x)
        double dist, influence
        double[:] density = np.zeros(npoints)

    # loop over cells
    for i in range(npoints):
        for k in range(npart):
            dist = np.sqrt((positions[k, 0]-x[i])**2 \
                         + (positions[k, 1]-y[i])**2 \
                         + (positions[k, 2]-z[i])**2)
            if dist<h:
                influence = sph_smoothing_kernel(h, dist, kernel, 3)
                density[i] += massp*influence
    return density
    
# 3D STATISTICS - GHOSTMESH
cpdef double[:] compute_density_3D_KS_GM(
        double[:] x, double[:] y, double[:] z, 
        double[:,:] positions, 
        double massp, double h, int kernel):
    """
    Compute density from particles at positions (x, y, z) using the Kernel Smoothing method (KS) with ghost mesh handling.

    Parameters
    ----------
    x : array_like
        Mesh point x coordinates.
    y : array_like
        Mesh point y coordinates.
    z : array_like
        Mesh point z coordinates.
    positions : array_like, shape (n, 3)
        Particle positions.
    massp : float
        Pseudo mass of particles for normalization.
    h : float
        Smoothing length.
    kernel : int
        Smoothing kernel identifier.

    Returns
    -------
    ndarray
        Density at mesh points.
    """
    cdef:
        int i, j, k
        int gcx, gc, ngcf
        int npart = len(positions)
        int npoints = len(x)
        double dist, influence
        long[:] gcidf, gcidj
        double[:,:] pos_xy = np.empty((npart, 2), dtype='d')
        double[:] density = np.zeros(npoints)

    # initialize lookup tables
    ghost_mesh = GhostMeshGrid2D(np.min(x), np.max(x),\
                                 np.min(y), np.max(y), h)
    pos_xy = ghost_mesh.set_lookup_tables(positions[:, 0:2])
    pos_z = ghost_mesh.sort_table(positions[:, 2])
    ngcf = len(ghost_mesh.cellid_filled)

    # loop over nodes on which to compute density
    for i in range(npoints):
        # get ghost cell of x
        gcx = ghost_mesh.get_ghostcell_id(np.asarray([x[i], y[i]]))

        # neighbors of point i (x9 cells)
        neigh = np.array([gcx-1, gcx, gcx+1])
        neigh = np.concatenate((\
            neigh,\
            neigh+ghost_mesh.ngx,\
            neigh-ghost_mesh.ngx), axis=0)

        # loop over ghost cells containing particles
        for j in range(ngcf):
            # ghost cell id
            gc = ghost_mesh.cellid_filled[j]

            # check if gc must contribute to gcx (in neighborhood)
            if gc in neigh:

                for k in range(ghost_mesh.cellid_jump[j],\
                               ghost_mesh.cellid_jump[j+1]):
                    dist = np.sqrt((pos_xy[k,0]-x[i])**2 +\
                                   (pos_xy[k,1]-y[i])**2 +\
                                   (pos_z[k]-z[i])**2)
                    influence = sph_smoothing_kernel(h, dist, kernel, 3)
                    density[i] += massp*influence

    return density
