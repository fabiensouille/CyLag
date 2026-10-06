from ..core.parameters cimport Parameters
from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cpdef void write_particles(\
        LagrangianParticleSet particle_set,\
        double time, int iteration, int size, int rank,
        Parameters params)

cpdef void write_stranded_particles_txt_2d(\
        int size, int rank,
        str outputdir, str file_name,
        double[:,:] stranded_positions,
        int stranded_count)