from ..geom.triangular_mesh cimport TriangularMesh

cpdef (int, int) xy_localize(\
    TriangularMesh triangular_mesh, int k,\
    double p0x, double p0y,\
    double x, double y,\
    int method, bint debug)

cdef int point_in_tri(\
    double[:] meshx, double[:] meshy,\
    int[:,:] triangles, int k, double x, double y)

cpdef int xy_localize_point(\
    TriangularMesh triangular_mesh,\
    double x, double y)

cpdef int xy_localize_point_kdtree(
    TriangularMesh triangular_mesh, \
    double x, double y, int k_nearest)

cpdef int xy_localize_point_in_neighbourhood(\
    TriangularMesh triangular_mesh,\
    int triangle_index,\
    double x, double y)

cpdef (int, int) xy_localize_point_from_path(\
    TriangularMesh triangular_mesh,\
    int k, \
    double p0x, double p0y,
    double x, double y)

