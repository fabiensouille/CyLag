from ..core.simutime cimport SimuTime
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..extra.scheduler cimport Scheduler

cdef class ParticleSource:

    cdef public Scheduler scheduler
    cdef public double[:,:] poly
    cdef public double poly_area
    cdef public int[:] grid_res
    cdef public int xy_method
    cdef public int xy_npart
    cdef public double xy_density
    cdef public int z_method
    cdef public int z_npart
    cdef public double z_density
    cdef public double zmin
    cdef public double zmax

    cpdef int release_condition(ParticleSource self,\
        SimuTime simutime, EulerianFieldSet fset)

    cpdef init_positions(ParticleSource self, EulerianFieldSet fset, int dim=*)


cdef class ParticleSourceFromField:
    
    cdef public Scheduler scheduler
    cdef public int field_id_criteria
    cdef public int field_id_density
    cdef public str criteria
    cdef public double threshold
    cdef public double virtual_mass
    cdef public long[:] loc
    cdef public bint single_release
    cdef public int release_rec

    cpdef int release_condition(ParticleSourceFromField self, \
        SimuTime simutime, EulerianFieldSet fset)

    cpdef init_positions(ParticleSourceFromField self, EulerianFieldSet fset, int dim=*)
