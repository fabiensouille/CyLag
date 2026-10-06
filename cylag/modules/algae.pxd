from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cdef class AlgaeParticleSet(LagrangianParticleSet):

    cdef public double Cd_a
    cdef public double Cd_b

    cpdef double drag_coefficient(AlgaeParticleSet self, double U, double dp)
