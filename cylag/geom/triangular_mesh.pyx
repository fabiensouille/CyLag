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
import h5py
import numpy as np
from scipy.spatial import cKDTree
from ..core.constants cimport INTERN_REF, BND_WALL_REF,\
    MAX_VERTEX_ADJACENCY, MAX_TRIANGLE_ADJACENCY, EPSILON
from ..geom.utils cimport compute_triangle_area

cdef class TriangularMesh:
    """
    Parameters
    ----------
    x, y : (nv,) array of float
        Coordinates of grid points.
    triangles : (nt, 3) array of int
        For each triangle, the indices of the three points that make
        up the triangle, ordered in an anticlockwise manner.
    nopen : int
        number of open boundaries (default: 0)

    Attributes
    ----------
    nv, nt, ne, nvb : int
        Number of mesh vertices, triangles, edges, boundary points
    x, y : (nv,) array of float
        Coordinates of grid points.
    trivtx : (nt, 3) array of int
        For each triangle, the indices of the three points that make
        up the triangle, ordered in an anticlockwise manner.
    edges : (ne, 2) array of int (default: None)
        For each edge, the indices of the two points that make
        up the edge, ordered in from min id to max id.  
    vertex_adjacency : (nv, MAX_VERTEX_ADJACENCY) int (default: None)
        contains indices of neighboring triangles
    triangle_adjacency : (nt, MAX_TRIANGLE_ADJACENCY) int (default: None)
        DEPRECATED - not built by default (only for legacy HDF5 compatibility)
    triangle_neighbors : (nt, 3) int (default: None)
        DEPRECATED - not built by default (only for legacy HDF5 compatibility)
    boundary_points : (nvb, 2) int (default: None)
        contains (vertex id, boundary label) pairs
    boundary_edges : (nvb, 3) int (default: None)
        contains (vertex id 0, vertex id 1, boundary label) triplets
    vertex_labels : (nv,) int (default: None)
        contains boundary labels for each vertex (INTERN_REF for internal vertices)
    edges_labels : (ne,) int (default: None)
        contains boundary labels for each edge (INTERN_REF for internal edges)
    signed_area : (nt,) float (default: None)
        contains signed area of each triangle
    gradx, grady : (nt, 3) float (default: None)
        contains gradients of the linear basis functions for each triangle
    det : (nt, 3) float (default: None)
        contains determinants of the linear basis functions for each triangle
    triangle_centroids : (nt, 2) float (default: None)
        contains centroids of each triangle (used for KDTree)
    """
    def __init__(self, x, y, triangles, \
            edges=None, vertex_adjacency=None, \
            triangle_adjacency=None, triangle_neighbors=None, \
            edgetri_map=None, triedge_map=None, \
            boundary_points=None, boundary_edges=None, \
            vertex_labels=None, edges_labels=None, \
            signed_area=None, gradx=None, grady=None, det=None, \
            triangle_centroids=None, \
            nopen=0):

        # Triangulation
        # ~~~~~~~~~~~~~
        self.x = x
        self.y = y
        self.triangles = triangles
        self.nv = x.shape[0]              # Number of vertices
        self.nt = self.triangles.shape[0] # Number of triangles

        # Edges
        # ~~~~~
        if edges is None:
            self.edges = self._init_edges()
        else:
            self.edges = edges
        self.ne = self.edges.shape[0]  # Number of edges

        # Connectivity
        # ~~~~~~~~~~~~
        if vertex_adjacency is None:
            # Building connectivity from scratch (optimized: skip triangle_adjacency)
            self.vertex_adjacency = self._init_vertex_adjacency()
            self.triangle_adjacency = None
            self.triangle_neighbors = None
            # self._init_triangles_adjacency() - not needed without neighbor search
            self.edgetri_map = None
            self.triedge_map = None
            self._init_triangles_edges_connectivity()
        else:
            # Read connectivity from parameters
            self.vertex_adjacency = vertex_adjacency
            # triangle_adjacency and triangle_neighbors are now optional (deprecated)
            if triangle_adjacency is not None:
                self.triangle_adjacency = triangle_adjacency
            else:
                self.triangle_adjacency = None
            if triangle_neighbors is not None:
                self.triangle_neighbors = triangle_neighbors
            else:
                self.triangle_neighbors = None 
            assert edgetri_map is not None
            assert triedge_map is not None
            self.edgetri_map = edgetri_map
            self.triedge_map = triedge_map

        # Boundaries
        # ~~~~~~~~~~
        self.nopen = nopen
        self.boundary_points = boundary_points
        self.boundary_edges = boundary_edges

        if vertex_labels is None or edges_labels is None:
            # Building boundaries from scratch
            self.nvb = 0
            self.vertex_labels = None
            self.edges_labels = None
            self._init_boundaries()
        else:
            # Read boundaries from parameters
            assert vertex_labels is not None
            assert edges_labels is not None
            self.vertex_labels = vertex_labels
            self.edges_labels = edges_labels
            self.nvb = self.boundary_points.shape[0]

        # Mesh properties (areas, gradients, det)
        # ~~~~~~~~~~~~~~~
        if signed_area is None:
            # Building mesh properties from scratch
            self.signed_area = np.empty(self.nt, dtype='d')
            self.gradx = np.empty((self.nt, 3), dtype='d')
            self.grady = np.empty((self.nt, 3), dtype='d')
            self.det = np.empty((self.nt, 3), dtype='d')
            self._init_mesh_properties()
        else:
            # Read mesh properties from parameters
            assert signed_area is not None
            assert gradx is not None
            assert grady is not None
            assert det is not None
            self.signed_area = signed_area
            self.gradx = gradx
            self.grady = grady
            self.det = det

        # Spatial indexing (KDTree)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        if triangle_centroids is not None:
            # Centroids provided (e.g., from HDF5), rebuild the tree
            self.triangle_centroids = triangle_centroids
            self.kdtree = cKDTree(np.asarray(triangle_centroids))
        else:
            # No centroids, build from scratch
            self.kdtree = None
            self.triangle_centroids = None
            self._build_kdtree()

    @staticmethod
    def from_matplotlib(tri,\
            boundary_points=None, boundary_edges=None, nopen=0):
        """
        Create mesh from matplotlib triangulation

        Parameters
        ----------
        tri : matplotlib.tri.Triangulation
            Matplotlib triangulation object.
        boundary_points : (nvb, 2) int, optional
            Contains (vertex id, boundary label) pairs. If not provided, will be built from edges.
        boundary_edges : (nvb, 3) int, optional
            Contains (vertex id 0, vertex id 1, boundary label) triplets. 
            If not provided, will be built from edges.
        nopen : int, optional
            Number of open boundaries (default: 0).
        """
        # Pass edges to avoid rebuilding them
        return TriangularMesh(tri.x, tri.y, tri.triangles, edges=tri.edges,\
            boundary_points=boundary_points, boundary_edges=boundary_edges,\
            nopen=nopen)

    @staticmethod
    def from_hdf5(file_name="cylag_mesh_file.h5"):
        """
        Create mesh from hdf5 cylag mesh file

        Parameters
        ----------
        file_name : str, optional
            Name of the hdf5 file containing the mesh data (default: "cylag_mesh_file.h5").
        """
        with h5py.File(file_name, 'r') as f:
            # Triangulation
            x = f['x'][:]
            y = f['y'][:]
            triangles = f['triangles'][:]
            # Edges
            edges = f['edges'][:]
            # Connectivity
            vertex_adjacency = f['vertex_adjacency'][:]
            # triangle_adjacency and triangle_neighbors are optional (legacy/deprecated)
            triangle_adjacency = f['triangle_adjacency'][:] if 'triangle_adjacency' in f else None
            triangle_neighbors = f['triangle_neighbors'][:] if 'triangle_neighbors' in f else None
            edgetri_map = f['edgetri_map'][:]
            triedge_map = f['triedge_map'][:]
            # Boundaries
            boundary_points = f['boundary_points'][:]
            boundary_edges = f['boundary_edges'][:]
            vertex_labels = f['vertex_labels'][:]
            edges_labels = f['edges_labels'][:]
            nopen = f['nopen'][:]
            # Mesh properties
            signed_area = f['signed_area'][:]
            gradx = f['gradx'][:]
            grady = f['grady'][:]
            det = f['det'][:]
            # KDTree - load centroids only, tree will be rebuilt
            triangle_centroids = f['triangle_centroids'][:]
        f.close()
        return TriangularMesh(x, y, triangles, edges,\
            vertex_adjacency,  triangle_adjacency, triangle_neighbors, \
            edgetri_map, triedge_map, \
            boundary_points, boundary_edges, vertex_labels, edges_labels, \
            signed_area, gradx, grady, det, triangle_centroids, nopen[0])

    def save_mesh_hdf5(TriangularMesh self, file_name="cylag_mesh_file.h5"):
        """
        Save mesh in hdf5

        Parameters
        ----------
        file_name : str, optional
            Name of the hdf5 file to save the mesh data (default: "cylag_mesh_file.h5").
        """
        with h5py.File(file_name, 'w') as f:
            # Triangulation
            f.create_dataset('x', data=np.array(self.x))
            f.create_dataset('y', data=np.array(self.y))
            f.create_dataset('triangles', data=np.array(self.triangles))
            # Edges
            f.create_dataset('edges', data=np.array(self.edges))
            # Connectivity
            f.create_dataset('vertex_adjacency', data=np.array(self.vertex_adjacency))
            # Only save triangle_adjacency/neighbors if they exist (legacy support)
            if self.triangle_adjacency is not None:
                f.create_dataset('triangle_adjacency', data=np.array(self.triangle_adjacency))
            if self.triangle_neighbors is not None:
                f.create_dataset('triangle_neighbors', data=np.array(self.triangle_neighbors))
            f.create_dataset('edgetri_map', data=np.array(self.edgetri_map))
            f.create_dataset('triedge_map', data=np.array(self.triedge_map))
            # Boundaries
            f.create_dataset('boundary_points', data=np.array(self.boundary_points))
            f.create_dataset('boundary_edges', data=np.array(self.boundary_edges))
            f.create_dataset('vertex_labels', data=np.array(self.vertex_labels))
            f.create_dataset('edges_labels', data=np.array(self.edges_labels))
            f.create_dataset('nopen', data=np.array([self.nopen]))
            # Mesh properties
            f.create_dataset('signed_area', data=np.array(self.signed_area))
            f.create_dataset('gradx', data=np.array(self.gradx))
            f.create_dataset('grady', data=np.array(self.grady))
            f.create_dataset('det', data=np.array(self.det))
            # KDTree - only save centroids, tree will be rebuilt on load
            f.create_dataset('triangle_centroids', data=np.array(self.triangle_centroids))
        f.close()

    cpdef int[:,:] _init_edges(TriangularMesh self):
        """
        Builds edges

        Returns
        -------
        edges : (ne, 2) int
            Array of edges, each defined by a pair of vertex indices.
        """
        cdef:
            int p, i0, i1, j, k
            int[:,:] edges_tmp = -1*np.ones((3*self.nt, 2), dtype='int32')

        for p in range(3*self.nt):
            k = p//3
            j = p%3
            i0 = min(self.triangles[k, j], self.triangles[k, (j+1)%3])
            i1 = max(self.triangles[k, j], self.triangles[k, (j+1)%3])
            edges_tmp[p, 0] = i0
            edges_tmp[p, 1] = i1

        return np.unique(edges_tmp, axis=0)

    cpdef int[:,:] _init_vertex_adjacency(TriangularMesh self):
        """
        Builds vertex adjacency matrix

        Returns
        -------
        vertex_adjacency : (nv, MAX_VERTEX_ADJACENCY) int
            For each vertex, the indices of neighboring triangles.
        """
        cdef:
            int i, j, k
            int[:] dynindex = np.zeros((self.nv), dtype='int32')
            int[:,:] vertex_adjacency = -1*np.ones((self.nv, MAX_VERTEX_ADJACENCY), dtype='int32')

        # First pass: build adjacency and detect overflows
        overflow_detected = False
        for k in range(self.nt):
            for j in range(3):
                i = self.triangles[k, j]
                if dynindex[i] < MAX_VERTEX_ADJACENCY:
                    vertex_adjacency[i, dynindex[i]] = k
                dynindex[i] += 1
                if dynindex[i] >= MAX_VERTEX_ADJACENCY:
                    overflow_detected = True

        # If overflow detected, provide detailed diagnostics
        if overflow_detected:
            dynindex_np = np.asarray(dynindex)
            max_connectivity = int(np.max(dynindex_np))
            overflow_vertices = np.where(dynindex_np >= MAX_VERTEX_ADJACENCY)[0]
            n_overflow = len(overflow_vertices)
            
            # Build detailed error message
            error_msg = (
                f"MAX_VERTEX_ADJACENCY overflow: {n_overflow} vertices exceed limit.\n"
                f"Current MAX_VERTEX_ADJACENCY: {MAX_VERTEX_ADJACENCY}\n"
                f"Required MAX_VERTEX_ADJACENCY: {max_connectivity}\n"
                f"Affected vertices ({min(5, n_overflow)} of {n_overflow} shown): "
                f"{overflow_vertices[:5].tolist()}\n"
            )
            
            # Add connectivity statistics
            if n_overflow <= 10:
                error_msg += "Vertex connectivity:\n"
                for v in overflow_vertices[:10]:
                    error_msg += f"  vertex {v}: {dynindex_np[v]} triangles\n"
            
            # Add recommendation
            error_msg += (
                f"\nRECOMMENDATION: Update MAX_VERTEX_ADJACENCY in "
                f"'cylag/core/constants.pyx' to at least {max_connectivity}.\n"
                f"Typical values: 15 (default), 20 (refined meshes), 30 (highly refined)."
            )
            
            raise ValueError(error_msg)

        return vertex_adjacency

    cpdef void _init_triangles_adjacency(TriangularMesh self):
        """
        Builds matrix containing neighbours triangle indexes

        Returns
        -------
        triangle_adjacency : (nt, MAX_TRIANGLE_ADJACENCY) int
            For each triangle, the indices of neighboring triangles.
        """
        cdef:
            int i, j, k, ej
            int v0, v1, v2
            int neigh_size
            int vertex_nn = np.shape(self.vertex_adjacency)[1]
            int[:,:] tri = self.triangles

        self.triangle_adjacency = -1*np.ones((self.nt, MAX_TRIANGLE_ADJACENCY), dtype='int32')
        self.triangle_neighbors = -1*np.ones((self.nt, 3), dtype='int32')

        # loop on triangles
        for k in range(self.nt):
            v0 = tri[k, 0]
            v1 = tri[k, 1]
            v2 = tri[k, 2]

            # vertices neighbors
            neigh0 = np.asarray(self.vertex_adjacency[v0, :])
            neigh1 = np.asarray(self.vertex_adjacency[v1, :])
            neigh2 = np.asarray(self.vertex_adjacency[v2, :])

            # merge vertices neighbors and remove duplicates
            neigh = np.unique(np.concatenate((neigh0, neigh1, neigh2), 0))[1::]
            neighd = np.delete(neigh, np.where(neigh==k))
            neigh_size = np.shape(neighd)[0] - 1

            if neigh_size >= MAX_TRIANGLE_ADJACENCY:
                raise ValueError("Found more than {} neighboors".format(MAX_TRIANGLE_ADJACENCY))

            ej = 0
            for j in range(MAX_TRIANGLE_ADJACENCY):
                # fill triangle_adjacency
                if j <= neigh_size:
                    i = neighd[j]
                    self.triangle_adjacency[k, j] = i

                    # fill triangle_neighbors
                    if ((v0==tri[i, 0] or v0==tri[i, 1] or v0==tri[i, 2]) and \
                        (v1==tri[i, 0] or v1==tri[i, 1] or v1==tri[i, 2])) or \
                       ((v1==tri[i, 0] or v1==tri[i, 1] or v1==tri[i, 2]) and \
                        (v2==tri[i, 0] or v2==tri[i, 1] or v2==tri[i, 2])) or \
                       ((v2==tri[i, 0] or v2==tri[i, 1] or v2==tri[i, 2]) and \
                        (v0==tri[i, 0] or v0==tri[i, 1] or v0==tri[i, 2])):
                        self.triangle_neighbors[k, ej] = i
                        ej += 1

            if ej > 3: 
                raise ValueError("Found more than 3 neighboors")

    cpdef void _init_triangles_edges_connectivity(TriangularMesh self):
        """
        Builds triangles - edges connectivity

        Returns
        -------
        triedge_map : (nt, 3) int
            For each triangle, the indices of the three edges that make
            up the triangle.
        """
        cdef:
            int nn = MAX_VERTEX_ADJACENCY
            int i, i0, i1, j, k, m, n, temp, trilist_len
            int[:] trilist = -1*np.ones(2*nn, dtype='int32')

        self.triedge_map = -1*np.ones((self.nt, 3), dtype='int32')
        self.edgetri_map = -1*np.ones((self.ne, 2), dtype='int32')

        # loop on edges
        for i in range(self.ne):
            i0 = self.edges[i, 0]
            i1 = self.edges[i, 1]

            # merge vertex adjacency lists
            for j in range(nn):
                trilist[j] = self.vertex_adjacency[i0, j]
                trilist[nn + j] = self.vertex_adjacency[i1, j]
            
            # inline bubble sort (fast for small arrays, no allocation)
            for m in range(2*nn - 1):
                for n in range(2*nn - 1 - m):
                    if trilist[n] > trilist[n + 1]:
                        temp = trilist[n]
                        trilist[n] = trilist[n + 1]
                        trilist[n + 1] = temp
            
            # find duplicates (adjacent triangles sharing this edge)
            # skip -1 values at the start
            j = 0
            while j < 2*nn and trilist[j] == -1:
                j += 1
            
            # scan for consecutive duplicates
            while j < 2*nn - 1:
                if trilist[j] != -1 and trilist[j] == trilist[j + 1]:
                    k = trilist[j]
                    # fill edgetri map
                    if self.edgetri_map[i, 0] == -1:
                        self.edgetri_map[i, 0] = k
                    else:
                        self.edgetri_map[i, 1] = k
                    # fill triedge map
                    if self.triedge_map[k, 0] == -1:
                        self.triedge_map[k, 0] = i
                    else:
                        if self.triedge_map[k, 1] == -1:
                            self.triedge_map[k, 1] = i
                        else:
                            self.triedge_map[k, 2] = i
                    # skip duplicate
                    j += 1
                j += 1

    cpdef void _init_boundaries(TriangularMesh self):
        """ 
        This method initializes the boundary points and edges of the mesh.

        It first checks if the boundary points are provided. 
        If not, it attempts to build them from the edge-triangle connectivity. 
        If the boundary points are provided, it constructs the boundary edges accordingly.
        It also initializes the vertex and edge labels based on the boundary information.
        """
        cdef:
            int i, j, i0, i1

        # Define boundary points and edges from scratch
        if self.boundary_points is None:

            # build from edgetri_map
            if self.edgetri_map is not None:
                bnd_points = []
                bnd_edges = []
                for i in range(self.ne):
                    if self.edgetri_map[i, 1]==-1:
                        i0 = self.edges[i, 0]
                        i1 = self.edges[i, 1]
                        bnd_points.append([i0, BND_WALL_REF])
                        bnd_points.append([i1, BND_WALL_REF])
                        bnd_edges.append([i0, i1, BND_WALL_REF])

                bnd_points = np.unique(bnd_points, axis=0)
                self.boundary_points = np.asarray(bnd_points, dtype='int32')
                self.boundary_edges = np.asarray(bnd_edges, dtype='int32')
                self.nvb = self.boundary_points.shape[0]
                
            # build from triangle neighbors
            elif self.edgetri_map is not None:
                raise ValueError("Failed to build boundaries")
            else:
                raise ValueError("Failed to build boundaries")

        # Define boundary edges from boundary points
        else:
            self.nvb = self.boundary_points.shape[0]
            self.boundary_edges = np.zeros((self.nvb, 3), dtype=np.int32)

            for i in range(self.nvb):
                # boundary edges
                self.boundary_edges[i, 0] = self.boundary_points[i, 0]
                self.boundary_edges[i, 1] = self.boundary_points[(i+1)%self.nvb, 0]

                # boundary edges labels
                if self.boundary_points[i, 1] == self.boundary_points[(i+1)%self.nvb, 1]:
                    self.boundary_edges[i, 2] = self.boundary_points[i, 1]
                else:
                    self.boundary_edges[i, 2] = BND_WALL_REF

        # Define vertex labels
        self.vertex_labels = INTERN_REF*np.ones(self.nv, dtype=np.int32)
        for i in range(self.nvb):
            j = self.boundary_points[i, 0]
            self.vertex_labels[j] = self.boundary_points[i, 1]

        # Define edges labels
        self.edges_labels = INTERN_REF*np.ones((self.ne), dtype=np.int32)
        for i in range(self.ne):
            if self.edgetri_map[i, 1]==-1:
                i0 = self.edges[i, 0]
                i1 = self.edges[i, 1]
                self.edges_labels[i] = BND_WALL_REF
                if self.vertex_labels[i0] == self.vertex_labels[i1]:
                    self.edges_labels[i] = self.vertex_labels[i0]

    cpdef void _init_mesh_properties(TriangularMesh self):
        """ 
        Initialize mesh properties (areas, gradients, det) 
        """
        cdef:
            int k
            int v0, v1, v2
            int[:,:] tri = self.triangles
            double[:] meshx = self.x
            double[:] meshy = self.y
            double xa, ya, xb, yb, xc, yc

        for k in range(self.nt):
            v0 = tri[k, 0]
            v1 = tri[k, 1]
            v2 = tri[k, 2]
            xa, ya = meshx[v0], meshy[v0]
            xb, yb = meshx[v1], meshy[v1]
            xc, yc = meshx[v2], meshy[v2]

            # Compute triangle area.
            self.signed_area[k] = compute_triangle_area(xa, ya, xb, yb, xc, yc)
            
            # Check for degenerate triangles (prevent NaN propagation)
            if abs(self.signed_area[k]) < EPSILON:
                raise ValueError(
                    f"Degenerate triangle detected at index {k}: "
                    f"vertices ({v0}, {v1}, {v2}), area = {self.signed_area[k]:.2e}. "
                    f"Check for duplicate or collinear vertices.")
            
            # Compute gradx, grady, det (for interpolations).
            alpha = 0.5 / self.signed_area[k]
            self.gradx[k, 0] = (yb - yc) * alpha
            self.gradx[k, 1] = (yc - ya) * alpha
            self.gradx[k, 2] = (ya - yb) * alpha
            self.grady[k, 0] = (xc - xb) * alpha
            self.grady[k, 1] = (xa - xc) * alpha
            self.grady[k, 2] = (xb - xa) * alpha
            self.det[k, 0] = (xb*yc - xc*yb) * alpha
            self.det[k, 1] = (xc*ya - xa*yc) * alpha
            self.det[k, 2] = (xa*yb - xb*ya) * alpha

    cpdef void _build_kdtree(TriangularMesh self):
        """
        Build a KDTree for fast spatial queries using triangle centroids.
        
        This method computes the centroid of each triangle and builds a 
        scipy.spatial.cKDTree for efficient nearest-neighbor queries.
        Call this method before using the KDTree-based localization.
        """
        cdef:
            int k, v0, v1, v2
            int[:,:] tri = self.triangles
            double[:] meshx = self.x
            double[:] meshy = self.y
            double[:,:] centroids
        
        # Compute triangle centroids
        centroids_np = np.empty((self.nt, 2), dtype='d')
        centroids = centroids_np
        
        for k in range(self.nt):
            v0 = tri[k, 0]
            v1 = tri[k, 1]
            v2 = tri[k, 2]
            centroids[k, 0] = (meshx[v0] + meshx[v1] + meshx[v2]) / 3.0
            centroids[k, 1] = (meshy[v0] + meshy[v1] + meshy[v2]) / 3.0
        
        self.triangle_centroids = centroids
        self.kdtree = cKDTree(centroids_np)
