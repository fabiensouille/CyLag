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
from ..geom.triangular_mesh cimport TriangularMesh
from ..geom.intersections cimport intersec2d_segment_segment
from ..geom.utils cimport point_in_triangle_dotp
from ..core.constants cimport MAX_TRIANGLE_ADJACENCY

cpdef (int, int) xy_localize(TriangularMesh triangular_mesh, int k,\
        double p0x, double p0y, double x, double y, \
        int method, bint debug):
    """
    Locate point in triangulation

    Parameters
    ----------
    triangular_mesh : cylag.TriangularMesh
    k : int
        index of previous triangle containing point
    p0x : float
        x coordinate of the previous position
    p0y : float
        y coordinate of the previous position
    x : float
        x coordinate of point to locate
    y : float
        y coordinate of point to locate
    method : int, search method
        if 0: search in all triangles (using KDTree)
        if 1: search in neighbors 
        if 2: search from path
        if 3: search from path, then fallback to KDTree (optimized)
    debug : bint, debug printouts

    Returns
    -------
    new_k : int
        index of triangle containing point x,y
    bnd_j : int
        index of boundary edge crossed if particle is out
    """
    cdef:
        int new_k = -1
        int bnd_j = -1

    # First check if point has changed triangle
    new_k = point_in_tri(\
        triangular_mesh.x,
        triangular_mesh.y,
        triangular_mesh.triangles, k, x, y)
    if new_k != -1:
        return new_k, bnd_j

    # Check method conditions
    if method==1:
        # * requires previous k
        # * requires triangle_adjacency tables
        if k==-1:
            method = 0

    if method==2:
        # * requires previous k and previous point
        # * requires edges to traingles connectivity tables
        if k==-1 and (x==p0x and y==p0y):
            method = 0
        if k!=-1 and (x==p0x and y==p0y):
            method = 1

    # Search method 0: search in all triangles (using KDTree)
    # ~~~~~~~~~~~~~~~
    if method==0:
        if debug:
            print(" ~~~> Search method 0: search in all triangles (using KDTree) ")
        new_k = xy_localize_point_kdtree(triangular_mesh, x, y, 10)

    # Search method 1: search in neighbors
    # ~~~~~~~~~~~~~~~
    # WARNING : REQUIRES triangle_adjacency tables
    elif method==1:
        if debug:
            print(" ~~~> Search method 1: search in neighbors ")
        new_k = xy_localize_point_in_neighbourhood(triangular_mesh, k, x, y)

    # Search method 2: search from path
    # ~~~~~~~~~~~~~~~
    elif method==2:
        if debug:
            print(" ~~~> Search method 2: search from path ")
        new_k, bnd_j = xy_localize_point_from_path(triangular_mesh, k, p0x, p0y, x, y)

    # Search method 3: search from path, then fallback to KDTree
    # ~~~~~~~~~~~~~~~
    elif method==3:
        if debug:
            print(" ~~~> Search method 3: search from path")
        new_k, bnd_j = xy_localize_point_from_path(triangular_mesh, k, p0x, p0y, x, y)

        # if path search failed, fallback directly to KDTree global search
        if new_k==-1 and bnd_j==-1:
            if debug:
                print(" ~~~> Path search failed: fallback to KDTree global search")
            new_k = xy_localize_point_kdtree(triangular_mesh, x, y, 10)
    else:
        raise ValueError("Unknown xy localizer method")
                
    return new_k, bnd_j

cdef int point_in_tri(double[:] meshx, double[:] meshy,\
        int[:,:] triangles, int k, double x, double y):
    """
    Check if point x,y is in triangle k of triangles

    Parameters
    ----------
    meshx : double[:]
        x coordinates of mesh vertices
    meshy : double[:]
        y coordinates of mesh vertices
    triangles : int[:,:]
        triangle vertex indices
    k : int
        index of triangle to check
    x : double
        x coordinate of point to check
    y : double
        y coordinate of point to check

    Returns
    -------
    int
        index of triangle containing point x,y if found, else -1
    """
    cdef:
        int v0, v1, v2

    # Triangle vertex indices.
    v0 = triangles[k, 0]
    v1 = triangles[k, 1]
    v2 = triangles[k, 2]

    # Triangle vertex coordinates.
    x0, y0 = meshx[v0], meshy[v0]
    x1, y1 = meshx[v1], meshy[v1]
    x2, y2 = meshx[v2], meshy[v2]

    if point_in_triangle_dotp(x, y, x0, y0, x1, y1, x2, y2):
        return k
    else:
        return -1

cpdef int xy_localize_point(TriangularMesh triangular_mesh, double x, double y):
    """
    Locate point in triangulation - loop over all mesh triangles

    Parameters
    ----------
        triangular_mesh : cylag.TriangularMesh
        x : float, x coordinate of point to locate
        y : float, y coordinate of point to locate

    Returns
    -------
        new_k : int
            index of triangle containing point x,y if found, else -1
    """
    cdef:
        int k
        int new_k = -1
        int nt = triangular_mesh.nt
        double[:] meshx = triangular_mesh.x
        double[:] meshy = triangular_mesh.y
        int[:,:] triangles = triangular_mesh.triangles

    # Loop on all mesh triangles to find the one containing (x,y).
    for k in range(nt):
        new_k = point_in_tri(meshx, meshy, triangles, k, x, y)
        if new_k != -1:
            break

    return new_k

cpdef int xy_localize_point_kdtree(TriangularMesh triangular_mesh, double x, double y, int k_nearest):
    """
    Locate point in triangulation using KDTree for efficient search.
    
    This is an optimized version of xy_localize_point that uses a KDTree
    to quickly find candidate triangles near the query point. This reduces
    the search from O(n) to O(log n + k) where k is the number of nearest
    triangles to check.
    
    The triangular_mesh must have a built KDTree (call build_kdtree() first).

    Parameters
    ----------
        triangular_mesh : cylag.TriangularMesh
            Must have kdtree attribute initialized (call build_kdtree() first)
        x : float, x coordinate of point to locate
        y : float, y coordinate of point to locate
        k_nearest : int, number of nearest triangle centroids to check
    
    Returns
    -------
        new_k : int, index of triangle containing point x,y (-1 if not found)
    
    Notes
    -----
    This method works as follows:
    1. Query KDTree for k_nearest triangle centroids closest to (x,y)
    2. Check if point is in any of those triangles
    3. If not found in k_nearest, progressively expand search
    4. Fall back to exhaustive search if still not found
    
    Performance: For meshes with thousands of triangles, this is typically
    10-100x faster than the exhaustive search in xy_localize_point.
    """
    cdef:
        int i, k, new_k = -1
        int[:] candidates
        double[:] meshx = triangular_mesh.x
        double[:] meshy = triangular_mesh.y
        int[:,:] triangles = triangular_mesh.triangles
        int nt = triangular_mesh.nt
    
    # Check if KDTree is built
    if triangular_mesh.kdtree is None:
        # Fall back to exhaustive search
        return xy_localize_point(triangular_mesh, x, y)
    
    # Query KDTree for k_nearest triangles
    query_point = np.array([x, y])
    
    # Try with k_nearest candidates first
    k_query = min(k_nearest, nt)
    distances, indices = triangular_mesh.kdtree.query(query_point, k=k_query)
    
    # Handle single result (when k=1, query returns scalars not arrays)
    if k_query == 1:
        indices = np.array([indices])
    
    candidates = indices.astype(np.int32)
    
    # Check candidates
    for i in range(candidates.shape[0]):
        k = candidates[i]
        new_k = point_in_tri(meshx, meshy, triangles, k, x, y)
        if new_k != -1:
            return new_k
    
    # If not found in k_nearest, try expanding search
    if k_query < nt:
        # Try with more candidates (2x, 5x, 10x)
        for multiplier in [2, 5, 10]:
            k_expanded = min(k_query * multiplier, nt)
            if k_expanded > k_query:
                distances, indices = triangular_mesh.kdtree.query(query_point, k=k_expanded)
                candidates = indices.astype(np.int32)
                
                # Only check the new candidates (skip already checked ones)
                for i in range(k_query, candidates.shape[0]):
                    k = candidates[i]
                    new_k = point_in_tri(meshx, meshy, triangles, k, x, y)
                    if new_k != -1:
                        return new_k
                
                k_query = k_expanded
    
    # Last resort: exhaustive search (point may be outside mesh or KDTree failed)
    return xy_localize_point(triangular_mesh, x, y)

cpdef int xy_localize_point_in_neighbourhood(\
    TriangularMesh triangular_mesh, int k, double x, double y):
    """
    Locate point in triangulation - looking in triangle neighbourhood

    Parameters
    ----------
        triangular_mesh : cylag.TriangularMesh
        k : int, index previous triangle containing point
        x : float, x coordinate of point to locate
        y : float, y coordinate of point to locate

    Returns
    -------
        new_k : int
            index of triangle containing point x,y if found, else -1
    """
    cdef:
        int max_nn = MAX_TRIANGLE_ADJACENCY
        int i, ii, j, jj
        int new_k = -1
        double[:] meshx = triangular_mesh.x
        double[:] meshy = triangular_mesh.y
        int[:,:] triangles = triangular_mesh.triangles
        int[:,:] neighbors = triangular_mesh.triangle_adjacency

    if k != -1:
        # loop over neighbors of k:
        for i in range(max_nn):
            j = neighbors[k, i]
            if j != -1:
                new_k = point_in_tri(\
                    meshx, meshy, triangles, j, x, y)
                if new_k != -1:
                    return new_k

        # if not found: loop over second neighbors of k
        if new_k == -1:
            for i in range(max_nn):
                j = neighbors[k, i]
                if j != -1:
                    for ii in range(max_nn):
                        jj = neighbors[j, ii]
                        if jj != -1:
                            new_k = point_in_tri(\
                                meshx, meshy, triangles, jj, x, y)
                            if new_k != -1:
                                return new_k
    return new_k

cpdef (int, int) xy_localize_point_from_path(\
        TriangularMesh triangular_mesh,\
        int k, \
        double p0x, double p0y,
        double x, double y):
    """
    Locate point in triangulation - from particule path

    Parameters
    ----------
        triangular_mesh : cylag.TriangularMesh
        triangle_index : int, index previous triangle containing point
        p0x : float, x coordinate of the previous position
        p0y : float, y coordinate of the previous position
        x : float, x coordinate of point to locate
        y : float, y coordinate of point to locate

    Returns
    -------
        new_k : int, index of triangle containing point x,y
    """
    cdef:
        int new_k = - 1
        int niter = 0
        int niter_max = 50
        int intersected_edge = -1
        bint intersects = 0
        int i, j, k0, k1, k_niegh
        int i0, i1
        double[:] P0 = np.array([p0x, p0y])
        double[:] P1 = np.array([x, y])
        double[:] EP0 = np.empty((2), dtype='d')
        double[:] EP1 = np.empty((2), dtype='d')
        double[:] meshx = triangular_mesh.x
        double[:] meshy = triangular_mesh.y
        int[:,:] triangles = triangular_mesh.triangles
        int[:,:] edgevtx = triangular_mesh.edges
        int[:,:] edgetri_map = triangular_mesh.edgetri_map
        int[:,:] triedge_map = triangular_mesh.triedge_map

    # iterative intersection search
    while new_k==-1 and niter<=niter_max:

        # loop on edges of triangle k
        intersects = 0
        for i in range(3):
            # global index of the edge from triangle and vertex ids
            j = triedge_map[k, i]

            # skip last intersected edge 
            # (to avoid comming back in prevous tri)
            if j==intersected_edge:
                continue

            # edge nodes
            i0 = edgevtx[j, 0]
            i1 = edgevtx[j, 1]

            # check if segment intersects edges of triangle k
            EP0[0] = meshx[i0]
            EP0[1] = meshy[i0]
            EP1[0] = meshx[i1]
            EP1[1] = meshy[i1]

            if intersec2d_segment_segment(P0, P1, EP0, EP1):
                intersects = 1
                # indices of triangles that contains edge j
                k0 = edgetri_map[j, 0]
                k1 = edgetri_map[j, 1]
                # index of neighbor triangle
                if k0==k:
                    k_niegh = k1
                else:
                    k_niegh = k0
                # check if not boundary edge
                if k_niegh==-1:
                    return -1, j
                # search in neighbor triangle
                new_k = point_in_tri(\
                    meshx, meshy, triangles, k_niegh, x, y)
                if new_k != -1:
                    return new_k, -1
                # not found: continue the path from the neighbor triangle
                k = k_niegh
                intersected_edge = j
                break

        # if no intersections, check if still in initial triangle
        if intersects==0:
            new_k = point_in_tri(meshx, meshy, triangles, k, x, y)
            if new_k == -1:
                return -1, -1
            else:
                return k, -1

        niter += 1

    return new_k, -1
