cdef class TriangularMesh:

    # Triangulation
    cdef public double[:] x
    cdef public double[:] y
    cdef public int[:,:] triangles
    cdef public int nv
    cdef public int nt

    # Edges
    cdef public int[:,:] edges
    cdef public int ne
        
    # Connectivity
    cdef public int[:,:] vertex_adjacency
    cdef public int[:,:] triangle_adjacency
    cdef public int[:,:] triangle_neighbors
    cdef public int[:,:] edgetri_map
    cdef public int[:,:] triedge_map

    # Boundaries
    cdef public int nopen
    cdef public int nvb
    cdef public int[:,:] boundary_points
    cdef public int[:,:] boundary_edges
    cdef public int[:] vertex_labels
    cdef public int[:] edges_labels

    # Mesh properties
    cdef public double[:] signed_area 
    cdef public double[:,:] gradx
    cdef public double[:,:] grady
    cdef public double[:,:] det
    
    # Spatial indexing
    cdef public object kdtree
    cdef public double[:,:] triangle_centroids
        
    cpdef int[:,:] _init_edges(TriangularMesh self)
    cpdef int[:,:] _init_vertex_adjacency(TriangularMesh self)
    cpdef void _init_triangles_adjacency(TriangularMesh self)
    cpdef void _init_triangles_edges_connectivity(TriangularMesh self)
    cpdef void _init_boundaries(TriangularMesh self)
    cpdef void _init_mesh_properties(TriangularMesh self)
    cpdef void _build_kdtree(TriangularMesh self)
