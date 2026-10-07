from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cdef class MacrophyteParticleSet(LagrangianParticleSet):

    cdef public double biomass
    cdef public double Cd_c
    cdef public double Cd_d
    cdef public double Cd_e
    cdef public double Cd_f
    
    cpdef double drag_coefficient(MacrophyteParticleSet self, double U, double dp)
