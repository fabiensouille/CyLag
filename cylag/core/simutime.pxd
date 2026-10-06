cdef class SimuTime:

    cdef public double time
    cdef public double time_step
    cdef public double initial_time
    cdef public double final_time
    cdef public double[:] times
    cdef public int nt
    cdef public int iteration
    cdef public int maxiteration
    cdef public bint is_finished
    cdef public bint print_progress

    cpdef void increment(SimuTime self)

    cpdef void progress(SimuTime self)

    cpdef tuple convert_time_to_jjhhmmss(SimuTime self, double time_in_seconds)