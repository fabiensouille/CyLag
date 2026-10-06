from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet

cdef class OilslickParticleSet(LagrangianParticleSet):

    cdef public double[:,:] addvelocity
    cdef public double[:,:] addforce
    cdef public double[:] dp
    cdef public double wind_coefficient
    cdef public double wind_attenuation_coef
    cdef public double wind_dev_coef
    cdef public bint stokes_drift
    cdef public bint wave_induced_vertical_dispersion

    cdef void _increase_pool_size(OilslickParticleSet self, int size)

    cdef void _parallel_split(OilslickParticleSet self, int offset, int count, int pool_size)

    cpdef void _sort_active_particles(OilslickParticleSet self)

    cdef void add_velocity_i(OilslickParticleSet self, 
            EulerianFieldSet fset, int i, double dt)

    cdef void add_force_i(OilslickParticleSet self,
            EulerianFieldSet fset, int i, double dt)

