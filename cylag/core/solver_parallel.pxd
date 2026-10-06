from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet

cdef class SolverParallel:

    cdef public Parameters parameters
    cdef public EulerianFieldSet field_set
    cdef public LagrangianParticleSet particle_set
    cdef public SimuTime simutime
    cdef public object sources
    cdef public object control_sections
    cdef public object comm
    cdef public int rank
    cdef public int size
    #cdef public int _mpi_owner
    cdef public double[:] _bnd_stats_times
    cdef public int[:,:] _bnd_stats_data
    cdef public int _bnd_stats_count
    cdef public double[:] _csec_stats_times
    cdef public int[:,:] _csec_stats_data
    cdef public int _csec_stats_count

    cpdef void initialize_particle_state(SolverParallel self, int i)

    cpdef void solve(SolverParallel self)

    cpdef void forward(SolverParallel self, double time_delta)
    
    cpdef void merge_results(SolverParallel self)

    cpdef void flush_statistics(SolverParallel self)
