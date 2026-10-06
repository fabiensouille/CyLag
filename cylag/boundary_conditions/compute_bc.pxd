from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cpdef void compute_boundary_condition(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i, int bnd_j,
        int[:] bnd_stats)
