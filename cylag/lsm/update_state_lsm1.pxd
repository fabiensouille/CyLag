from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cdef int update_state_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i)

cdef void euler_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i)
        
cdef int rk2_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i)

cdef int rk3_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i)

cdef int rk4_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i)
        
cdef void diffusion_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i, int diffmod,
        double nux, double nuz,
        double sigc,
        int rng_method, int tri0, int ll0)