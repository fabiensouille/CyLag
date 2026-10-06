from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cdef int update_state_lsm3(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i)

cdef void euler_maruyama_lsm3(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i, int diffmod,
        double nux, double nuz,
        int nuopt, double sigc, double tml_horizontal, double tml_vertical,
        int rng_method)

cdef void di_lsm3(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i, int diffmod,
        double nux, double nuz,
        int nuopt, double sigc, double tml_horizontal, double tml_vertical,
        int rng_method, int edge_opt)
