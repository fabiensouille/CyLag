from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cdef int update_state_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i)

cdef void euler_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i)

cdef int rk2_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i)

cdef int rk3_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i)

cdef int rk4_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i)

cdef void diffusion_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i, int diffmod,
        double nux, double nuz,
        double sigc,
        int rng_method, int tri0, int ll0)

cdef void euler_maruyama_lsm2_u(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i, int diffmod,
        double nux, double nuz,
        int nuopt, double sigc,
        int rng_method)

cdef void di_ode_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i)

cdef void di_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i, int diffmod,
        double nux, double nuz,
        int nuopt, double sigc,
        int rng_method)
