cdef class GhostMeshGrid1D:

    cdef public int ng
    cdef public double[:] transform
    cdef public long[:] sorting_idx
    cdef public long[:] cellid_filled
    cdef public long[:] cellid_jump

    cpdef double primal_to_ghost(GhostMeshGrid1D self, double x)

    cpdef double ghost_to_primal(GhostMeshGrid1D self, double x)

    cpdef int get_ghostcell_id(GhostMeshGrid1D self, double x)

    cpdef double[:] set_lookup_tables(self, double[:] nodes)

    cpdef double[:] sort_table(self, double[:] table)

cdef class GhostMeshGrid2D:

    cdef public int ngx
    cdef public int ngy
    cdef public int ng
    cdef public double[:] transform
    cdef public long[:] sorting_idx
    cdef public long[:] cellid_filled
    cdef public long[:] cellid_jump

    cpdef double[:] primal_to_ghost(GhostMeshGrid2D self, double[:] P)

    cpdef double[:] ghost_to_primal(GhostMeshGrid2D self, double[:] P)

    cpdef int get_ghostcell_idx(GhostMeshGrid2D self, double[:] P)
    
    cpdef int get_ghostcell_idy(GhostMeshGrid2D self, double[:] P)
    
    cpdef int get_ghostcell_id(GhostMeshGrid2D self, double[:] P)

    cpdef int[:] grid_ij_from_id(GhostMeshGrid2D self, int k)

    cpdef int id_from_grid_ij(GhostMeshGrid2D self, int[:] I)
    
    cpdef double[:,:] set_lookup_tables(self, double[:,:] nodes)
    
    cpdef double[:] sort_table(self, double[:] table)
