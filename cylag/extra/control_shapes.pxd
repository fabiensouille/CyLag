cdef class ControlSection:

    cdef public int nseg
    cdef public double[:,:] poly
    cdef public str file_name

    cpdef bint cross(ControlSection self, double[:] pos_0, double[:] pos_1)

    cdef bint c_cross(ControlSection self, double x0, double y0, double x1, double y1)
