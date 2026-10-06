from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet

cdef class CustomParticleSet(LagrangianParticleSet):

    cdef public double[:,:] addvelocity
    cdef public double[:,:] addforce

    cdef void _increase_pool_size(CustomParticleSet self, int size)

    cpdef void _sort_active_particles(CustomParticleSet self)

    cdef void _parallel_split(CustomParticleSet self, int offset, int count, int pool_size)

    cdef void add_velocity_i(CustomParticleSet self, EulerianFieldSet fset, int i, double dt)

    cdef void add_force_i(CustomParticleSet self, EulerianFieldSet fset, int i, double dt)

