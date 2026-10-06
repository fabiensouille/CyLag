from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cpdef void write_vtk_2d(\
    LagrangianParticleSet pset,
    double time, int iteration, int size, int rank,
    str outputdir, str file_name, int model)

cpdef void write_vtk_3d(\
    LagrangianParticleSet pset,
    double time, int iteration, int size, int rank, 
    str outputdir, str file_name, int model)
