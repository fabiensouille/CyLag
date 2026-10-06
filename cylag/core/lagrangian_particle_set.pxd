from ..core.eulerian_field_set cimport EulerianFieldSet

cdef class LagrangianParticleSet:

    # dimension
    cdef public int npart
    cdef public int dim
    cdef public int min_pool_size_update

    # pool of particles
    cdef public int npart_active
    cdef public int[:] inactive

    # Particle state vector
    cdef public double[:,:] position
    cdef public double[:,:] velocity
    cdef public double[:,:] fluid_velocity_seen
    #double[:,:] mean_fluid_velocity

    # Particle depth to surface
    cdef public bint compute_depth
    cdef public double[:] depth

    # id of particles for i/o
    cdef public int[:] tag
    cdef public int maxtag

    # stranded particle buffer
    cdef public int max_stranded
    cdef public int stranded_count
    cdef public double[:,:] stranded_positions

    # localization in eulerian mesh
    cdef public int[:] tri
    cdef public int[:] lowerlayer
    cdef public int[:] last_tri
    cdef public double[:,:] last_position
    cdef public double[:,:] last_velocity

    # Particle properties
    cdef public double rho0
    cdef public double density
    cdef public double particle_diameter
    cdef public double particle_volume
    cdef public str diameter_distribution
    cdef public double diameter_distribution_param
    cdef public double[:] diameter
    cdef public double[:] volume

    cdef public int drag_coef_model
    cdef public double drag_coef
    cdef public bint added_mass_force
    cdef public double added_mass_coef

    cdef public int buoyancy_velocity_model
    cdef public double buoyancy_velocity_value

    cdef public bint additional_velocity
    cdef public bint additional_force

    cdef public double a0c
    cdef public double a1c
    cdef public double a2c
    cdef public double a3c

    cdef void _initialize_particle_distribution(LagrangianParticleSet self)

    cdef void _initialize_model_coefficients(LagrangianParticleSet self)

    cdef void _increase_pool_size(LagrangianParticleSet self, int size)

    cdef void _parallel_split(LagrangianParticleSet self, int offset, int count, int pool_size)

    cpdef void _sort_active_particles(LagrangianParticleSet self)

    cpdef void add(LagrangianParticleSet self, double[:,:] position)

    cpdef void delete(LagrangianParticleSet self, int[:] particles_id)
    
    cpdef void delete_single(LagrangianParticleSet self, int i)
    
    cpdef double drag_coefficient(LagrangianParticleSet self, double U, double dp)

    cpdef double buoyancy_velocity(LagrangianParticleSet self, double dp, int model_hr, double rho0)
    
    cdef void add_velocity_i(LagrangianParticleSet self, EulerianFieldSet fset, int i, double dt)

    cdef void add_force_i(LagrangianParticleSet self, EulerianFieldSet fset, int i, double dt)