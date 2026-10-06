# cython: profile=False

from libc.math cimport ceil, log10

cpdef int set_parallel_tag(int size, int rank, int ptag):
    """
    Modify particle tags for parallel IO.

    The parallel tag encodes the original ID and the processor rank.

    Parameters
    ----------
    size : int
        Number of ranks.
    rank : int
        Rank number.
    ptag : int
        Original particle tag.

    Returns
    -------
    int
        Encoded parallel tag.
    """
    cdef int nd = int(ceil(log10(size)))
    cdef int base = <int>(10**nd)
    return ptag*base + rank

cpdef int get_seq_from_par_tag(int size, int rank, int tag):
    """
    Get sequential tag from parallel tag
    """
    cdef int nd = int(ceil(log10(size)))
    cdef int base = <int>(10**nd)
    return (tag - rank)/base