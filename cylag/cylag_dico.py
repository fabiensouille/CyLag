# -*- coding: utf-8 -*-
"""
List of all parameters and options for CyLag and default values.

The user can modify the default values by providing a dictionary with the same keys and new values to the Solver.
See the documentation for more details on each parameter and option.
"""
# List of all parameters and options
CYLAG_DEFAULT_PARAMETERS = {
    # NUMERICAL PARAMETERS:
    'initial_time': 0.,
    'final_time': 3600.,
    'time_step': 1.,
    'frozen_eulerian_fields': False,
    # LSM  PARAMETERS:
    'model': 1,
    'time_scheme': 1,
    'particle_velocity_init': 1,
    'rng_method': 2,
    'lsm3_edge_opt': 2,
    # PHYSICAL PARAMETERS:
    'water_density': 1000.,
    'diffusion_model': 0,
    'horizontal_diffusivity': 0.,
    'vertical_diffusivity': 0.,
    'schmidt_number': 0.72,
    'diffusion_lsm2_option': 1,
    'diffusion_lsm3_tl_horizontal': 1.,
    'diffusion_lsm3_tl_vertical': 1.,
    'pseudo_3d_velocity_profile': 1,
    'pseudo_3d_friction_model': 1,
    'pseudo_3d_friction_coefficient': 40.,
    # BOUNDARY_CONDITIONS:
    'boundary_conditions': True,
    'boundary_conditions_type': 1,
    'boundary_conditions_debug': False,
    'max_wall_rebounds': 4,
    'rebound_damping_coef': 0.5,
    'wall_roughness_angle': 15.0,
    'stranding_probability': 0.,
    # IO PARAMETERS:
    'listing': False,
    'listing_printout_period': 10,
    'output_rep': 'particles',
    'output_file': False,
    'output_printout_period': 10,
    'output_file_format': 'txt',
    'output_file_name': 'particles',
    'bnd_statistics': False,
    'bnd_statistics_file_name': 'bnd_stats',
    'control_sections_file_name': 'cs_stats',
    'stranding_output': False,
    'stranding_output_file_name': 'stranded_particles',
    }
