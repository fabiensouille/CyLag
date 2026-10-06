from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet

cdef class Solver:

    cdef public Parameters parameters
    cdef public EulerianFieldSet field_set
    cdef public LagrangianParticleSet particle_set
    cdef public SimuTime simutime
    cdef public object sources
    cdef public object control_sections
    cdef public double[:] _bnd_stats_times
    cdef public int[:,:] _bnd_stats_data
    cdef public int _bnd_stats_count
    cdef public double[:] _csec_stats_times
    cdef public int[:,:] _csec_stats_data
    cdef public int _csec_stats_count

    cpdef void initialize_particle_state(Solver self, int i)

    cpdef void solve(Solver self)

    cpdef void forward(Solver self, double time_delta)

    cpdef void flush_statistics(Solver self)
