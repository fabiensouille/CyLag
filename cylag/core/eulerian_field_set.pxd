from ..geom.triangular_mesh cimport TriangularMesh

cdef class EulerianFieldSet:

    cdef public int npoints
    cdef public int nlayers
    cdef public int dim
    cdef public int ntimes
    cdef public bint frozen_hydro
    cdef public double time_interp_factor
    cdef public int record

    cdef public TriangularMesh triangular_mesh
    cdef public object trifinder
    
    cdef public double[:] times
    cdef public double[:,:] elevation
    cdef public double[:,:] velocity_x
    cdef public double[:,:] velocity_y
    cdef public double[:,:] velocity_z
        
    cdef public int nopt
    cdef public double[:,:,:] opt_fields
    cdef public object field_names
    cdef public object opt_fields_names

    # Named field IDs (set from field_names and opt_fields_names at init)
    # Main fields: always at fixed positions
    cdef public int fid_zs
    cdef public int fid_ux
    cdef public int fid_uy
    cdef public int fid_uz
    # Optional fields: -1 if not loaded
    cdef public int fid_turbeng
    cdef public int fid_dissip
    cdef public int fid_zb
    cdef public int fid_windx
    cdef public int fid_windy

    # Friction parameters
    cdef public int friction_model
    cdef public double friction_coefficient

    # Temporary arrays 
    cdef public double[:] _tmp_elevation
    cdef public double[:] _tmp_velocity_x
    cdef public double[:] _tmp_velocity_y
    cdef public double[:] _tmp_velocity_z
    cdef public double[:] _tmp_z0
    cdef public double[:] _tmp_u0
    cdef public double[:] _tmp_v0
    cdef public double[:] _tmp_w0
    cdef public double[:,:] _tmp_opt_fields

    # Cached interpolation factors for performance
    cdef public int _cached_triangle
    cdef public double _cached_x
    cdef public double _cached_y
    cdef public double _cached_f0
    cdef public double _cached_f1
    cdef public double _cached_f2
    cdef public bint _cache_valid

    # Previous-step fields (tmp_*0) are valid for finite-difference time derivatives
    cdef public bint _interp_done
    cdef public bint _prev_valid

    cpdef void _init_tmp(EulerianFieldSet self)
    cpdef void _cache_interp_factors(EulerianFieldSet self, int k, double x, double y)
    cpdef bint _use_cached_factors(EulerianFieldSet self, int k, double x, double y)

    cpdef double[:] _get_field_slice(EulerianFieldSet self, double[:,:] field, int k, int l)
    cpdef double[:,:] _get_field_from_id(EulerianFieldSet self, int field_id)

    cpdef double[:] _get_tmp_slice(EulerianFieldSet self, double[:] field, int l)
    cpdef double[:] _get_tmp_from_id(EulerianFieldSet self, int field_id)
    cpdef double[:] _get_tmp(EulerianFieldSet self, int field_id, int l)

    cpdef void _set_record(EulerianFieldSet self, double time, bint follow)
    cpdef void _time_interpolation(EulerianFieldSet self, double time)
    cpdef void _time_interpolation_field(EulerianFieldSet self, int field_id, double time)

    cpdef double _interpolate_field_2d(\
        EulerianFieldSet self,
        int field_id, 
        int k, 
        int l, 
        double x, double y)
            
    cpdef (double, double) _interpolate_velocity_2d(\
        EulerianFieldSet self,
        int k, 
        double x, double y)

    cpdef int z_localize(\
        EulerianFieldSet self,
        int k,
        double x, double y, double z)

    cpdef (double, double) _interpolate_acceleration_2d(\
        EulerianFieldSet self, int k,\
        double x, double y, double dt,
        double ua, double va)

    cpdef double _interpolate_field_3d(\
        EulerianFieldSet self,
        int field_id,
        int k,
        int lowerlayer,
        bint zlocalize,
        double x, double y, double z)

    cpdef (double, double, double) _interpolate_velocity_3d(\
        EulerianFieldSet self,
        int k,
        int lowerlayer,
        bint zlocalize,
        double x, double y, double z)

    cpdef (double, double, double) _interpolate_acceleration_3d(\
        EulerianFieldSet self,\
        int k, int lowerlayer, \
        double x, double y, double z, double dt,
        double ua, double va, double wa)

    cpdef double _interpolate_surface_velocity(\
        EulerianFieldSet self, int k,\
        double x, double y, double dt,
        double zsp)

    cdef (double, double, double, double) _interpolate_diffusivity_gradient_c(\
        EulerianFieldSet self,
        int k, int lowerlayer,
        double x, double y, double z, double sigc)

    cpdef (double, double) pseudo_3d_velocity_corrections(\
        EulerianFieldSet self,
        int k, double x, double y, double z, double dt,
        double ux, double uy)

    cpdef double friction_calc(EulerianFieldSet self, double h)