from ..core.parameters cimport Parameters
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cpdef (double,double,double) update_lsm3_fluctuations(\
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i)
