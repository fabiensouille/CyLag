cpdef double[:] compute_symetric_point2d(
    double[:] point0, double[:] point1,
    double[:] point, double damp)

cpdef double[:] compute_symetric_point3d(
    double[:] point0, double[:] point1, double[:] point2,
    double[:] point, double damp)

cpdef int intersec2d_segment_segment(
    double[:] point1, double[:] point2, double[:] seg_point0, double[:] seg_point1)

cdef int c_intersec2d_segment_segment(
    double p0x, double p0y, double p1x, double p1y,
    double s0x, double s0y, double s1x, double s1y)

cpdef (double, double) intersec2d_get_intersection_point(\
        double x1, double y1,
        double x2, double y2,
        double x3, double y3,
        double x4,double y4)

cpdef int intersec3d_triangle_segment(
    double[:] point0, double[:] point1, double[:] point2,
    double[:] seg_point0, double[:] seg_point1, double[:] intersection_point)

cpdef int MT_intersec3d_triangle_ray(
    double[:] point0, double[:] point1, double[:] point2,
    double[:] seg_point0, double[:] seg_point1, double[:] intersection_point)
