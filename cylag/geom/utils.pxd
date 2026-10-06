cdef double det22(double[:] v1, double[:] v2)  noexcept nogil
cdef double dot2(double[:] V1, double[:] V2)  noexcept nogil
cdef void sub2(double[:] V1, double[:] V2, double[:] dest) noexcept nogil
cdef void normal2(double[:] V0, double[:] V1, double[:] n) noexcept nogil
cdef void normal2u(double[:] v0, double[:] v1, double[:] n) noexcept nogil

cdef void cross(double[:] V1, double[:] V2, double[:] dest) noexcept nogil
cdef double dot(double[:] V1, double[:] V2) noexcept nogil
cdef void sub(double[:] V1, double[:] V2, double[:] dest) noexcept nogil
cdef void normal(double[:] V0, double[:] V1, double[:] V2, double[:] dest) noexcept nogil

cdef double compute_triangle_area(\
    double xa, double ya,\
    double xb, double yb,\
    double xc, double yc) noexcept nogil

cpdef double polygon_area(\
    double[:] x, 
    double[:] y) noexcept nogil

cdef int orientation(double[:] V0, double[:] V1, double[:] V2) noexcept nogil
cdef bint on_segment(double[:] V0, double[:] V1, double[:] V2) noexcept nogil

cdef int _orientation2d(\
        double v0x, double v0y,\
        double v1x, double v1y,\
        double v2x, double v2y) noexcept nogil

cdef bint _on_segment2d(\
        double v0x, double v0y,\
        double v1x, double v1y,\
    double v2x, double v2y) noexcept nogil

cdef bint point_in_triangle_dotp(\
    double x , double y ,\
    double x0, double y0,\
    double x1, double y1,\
    double x2, double y2) noexcept nogil

cdef bint point_in_triangle_bbox(\
    double x , double y ,\
    double x0, double y0,\
    double x1, double y1,\
    double x2, double y2) noexcept nogil
