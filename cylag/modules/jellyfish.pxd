from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cdef class JellyfishParticleSet(LagrangianParticleSet):

    cdef public double Cd_a
    cdef public double Cd_b
    cdef public double Cd_c1
    cdef public double Cd_d1
    cdef public double Cd_c2
    cdef public double Cd_d2
    cdef public double bdr
    cdef public double tlr

    cpdef double drag_coefficient(JellyfishParticleSet self, double U, double dp)
