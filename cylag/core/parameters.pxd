cdef class Parameters:

        # TIME PARAMETERS:
        cdef public double initial_time
        cdef public double final_time
        cdef public double time_step
        cdef public bint frozen_eulerian_fields

        # LSM PARAMETERS:
        cdef public int model
        cdef public int time_scheme
        cdef public int particle_velocity_init
        cdef public int rng_method

        # PHYSICAL PARAMETERS:
        cdef public double water_density
        cdef public int diffusion_model
        cdef public double horizontal_diffusivity
        cdef public double vertical_diffusivity
        cdef public double schmidt_number
        cdef public int diffusion_lsm2_option
        cdef public double diffusion_lsm3_tl_horizontal
        cdef public double diffusion_lsm3_tl_vertical
        cdef public int lsm3_edge_opt

        # BOUNDARY_CONDITIONS PARAMETERS:
        cdef public bint boundary_conditions
        cdef public int boundary_conditions_type
        cdef public bint boundary_conditions_debug
        cdef public int max_wall_rebounds
        cdef public double rebound_damping_coef
        cdef public double wall_roughness_angle
        cdef public double stranding_probability

        # IO PARAMETERS:
        cdef public bint listing
        cdef public int listing_printout_period
        cdef public str output_rep
        cdef public bint output_file
        cdef public int output_printout_period
        cdef public str output_file_format
        cdef public str output_file_name
        cdef public bint bnd_statistics
        cdef public str bnd_statistics_file_name
        cdef public bint control_sections
        cdef public int ncsec
        cdef public str control_sections_file_name
        cdef public bint stranding_output
        cdef public str stranding_output_file_name