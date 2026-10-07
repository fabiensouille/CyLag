from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet

cdef class FishParticleSet(LagrangianParticleSet):

    cdef public double[:,:] addvelocity
    cdef public double[:,:] addforce
    cdef public int swimming_model

    cdef public double[:] swimming_threshold_fluid_velocity
    cdef public double[:] swimming_velocity_states
    cdef public double[:] swimming_thrust_forces
    cdef public double[:] swimming_dispersion

    cdef public double attraction_radius
    cdef public double orientation_radius
    cdef public double repulsion_radius
    cdef public double blind_angle

    cdef void add_velocity_i(FishParticleSet self, EulerianFieldSet fset, int i, double dt)
    cdef void add_velocity_behavior_i_ibm1(FishParticleSet self, int i, double dt)
    cdef void add_velocity_behavior_i_cbm1(FishParticleSet self, int i, double dt)

    cdef void _increase_pool_size(FishParticleSet self, int size)

    cpdef void _sort_active_particles(FishParticleSet self)

    cdef void _parallel_split(FishParticleSet self, int offset, int count, int pool_size)

    cdef void add_force_i(FishParticleSet self, EulerianFieldSet fset, int i, double dt)
    cdef void add_force_behavior_i_ibm1(FishParticleSet self, int i, double dt)
