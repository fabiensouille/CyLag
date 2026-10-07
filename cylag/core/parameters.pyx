# cython: profile=False
#
# -------------------------------------------------------------------------------------------------
#  Project name: CyLag
#  Copyright (C) 2023 Fabien Souille
#
#  This program is free: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License published by the Free Software Foundation,
#  either version 3 of the license, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License for details.
#
#  You should have received a copy of the GNU General Public License
#  with this program. If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------------------------------
#
from ..cylag_dico import CYLAG_DEFAULT_PARAMETERS

cdef class Parameters:
    """
    Parameters class

    This class encapsulates all the parameters and options for the CyLag solver. 
    It provides methods to initialize parameters, check their values, and ensure type consistency.

    Parameters
    ----------
    parameters : dict, optional
        Dictionary containing parameter values.

    Attributes
    ----------
    initial_time : float
        Initial time of the simulation.
    final_time : float
        Final time of the simulation.
    time_step : float
        Time step for the simulation.
    frozen_eulerian_fields : bool
        Whether the Eulerian fields are frozen during the simulation.
    model : int
        Model type (1, 2, or 3).
    time_scheme : int
        Time integration scheme.
    particle_velocity_init : int
        Particle velocity initialization method.
    rng_method : int
        Method for generating random numbers 
        (1 for rectangular approximation, 2 for Ziggurat (default), 3 for Marsaglia polar).
    water_density : float
        Density of water.
    diffusion_model : int
        Diffusion model type.
    horizontal_diffusivity : float
        Horizontal diffusivity coefficient.
    vertical_diffusivity : float
        Vertical diffusivity coefficient.
    schmidt_number : float
        Schmidt number for the simulation.
    diffusion_lsm2_option : int
        Option for diffusion in LSM2.
    diffusion_lsm3_tl_horizontal : float
        Lagrangian time scale for horizontal LSM3 constant diffusion.
    diffusion_lsm3_tl_vertical : float
        Lagrangian time scale for vertical LSM3 constant diffusion.
    lsm3_edge_opt : int
        Edge-case treatment of the LSM3 direct integrator: 
        1 for normalized Taylor series,
        2 for closed-form coefficients with clipping (default).
    boundary_conditions : bool
        Whether boundary conditions are applied.
    boundary_conditions_type : int
        Type of wall rebound: 1 for specular (mirror) reflection,
        2 for diffuse (rough wall) reflection.
    boundary_conditions_debug : bool
        Whether to enable debugging for boundary conditions.
    max_wall_rebounds : int
        Maximum number of wall rebounds allowed.
    rebound_damping_coef : float
        Damping coefficient for rebounds.
    wall_roughness_angle : float
        Standard deviation (degrees) of the random perturbation applied to the
        local wall normal to model diffuse (rough wall) rebounds. Only used
        when boundary_conditions_type == 2.
    stranding_probability : float
        Probability of stranding.
    listing : bool
        Whether to enable listing of simulation progress.
    listing_printout_period : int
        Period for printing listing information.
    output_rep : str
        Output directory for simulation results.
    output_file : bool
        Whether to output results to a file.
    output_printout_period : int
        Period for printing output information.
    output_file_format : str
        Format of the output file ('txt', 'vtk', or 'all').
    output_file_name : str
        Name of the output file.
    bnd_statistics : bool
        Whether to compute boundary statistics.
    bnd_statistics_file_name : str
        Name of the boundary statistics file.
    control_sections : bool
        Whether to enable control sections.
    ncsec : int
        Number of control sections.
    control_sections_file_name : str
        Name of the control sections file.
    """
    def __init__(self, parameters={}):

        # load default parameters from cylag dico (copy to avoid mutating global defaults)
        params = dict(CYLAG_DEFAULT_PARAMETERS)

        # update default parameters with provided parameters
        params.update(parameters) 

        # check types and keys:
        self.check_types(CYLAG_DEFAULT_PARAMETERS, params)
    
        # TIME PARAMETERS:
        self.initial_time = params['initial_time']
        self.final_time = params['final_time']
        self.time_step = params['time_step']
        self.frozen_eulerian_fields = params['frozen_eulerian_fields']

        # override initial_time to 0 if frozen_eulerian_fields is True
        if self.frozen_eulerian_fields==1:
            self.initial_time = 0.

        # LSM PARAMETERS:
        self.model = params['model']
        self.time_scheme = params['time_scheme']
        self.particle_velocity_init = params['particle_velocity_init']
        self.rng_method = params['rng_method']

        # PHYSICAL PARAMETERS:
        self.water_density = params['water_density']
        self.diffusion_model = params['diffusion_model']
        self.horizontal_diffusivity = params['horizontal_diffusivity']
        self.vertical_diffusivity = params['vertical_diffusivity']
        self.schmidt_number = params['schmidt_number']
        self.diffusion_lsm2_option = params['diffusion_lsm2_option']
        self.diffusion_lsm3_tl_horizontal = params['diffusion_lsm3_tl_horizontal']
        self.diffusion_lsm3_tl_vertical = params['diffusion_lsm3_tl_vertical']
        self.lsm3_edge_opt = params['lsm3_edge_opt']

        # BOUNDARY_CONDITIONS PARAMETERS:
        self.boundary_conditions = params['boundary_conditions']
        self.boundary_conditions_type = params['boundary_conditions_type']
        self.boundary_conditions_debug = params['boundary_conditions_debug']
        self.max_wall_rebounds = params['max_wall_rebounds']
        self.rebound_damping_coef = params['rebound_damping_coef']
        self.wall_roughness_angle = params['wall_roughness_angle']
        self.stranding_probability = params['stranding_probability']

        # IO PARAMETERS:
        self.listing = params['listing']
        self.listing_printout_period = params['listing_printout_period']
        self.output_rep = params['output_rep']
        self.output_file = params['output_file']
        self.output_printout_period = params['output_printout_period']
        self.output_file_format = params['output_file_format']
        self.output_file_name = params['output_file_name']
        self.bnd_statistics = params['bnd_statistics']
        self.bnd_statistics_file_name = params['bnd_statistics_file_name']
        self.control_sections_file_name = params['control_sections_file_name']
        self.control_sections = False
        self.ncsec = 0
        self.stranding_output = params['stranding_output']
        self.stranding_output_file_name = params['stranding_output_file_name']

        # set default time scheme depending on model and diffusion
        if not 'time_scheme' in parameters:
            # LSM1 without diffusion: Fourth order Runge-Kutta
            if self.model==1 and self.diffusion_model==0:
                 self.time_scheme = 4
            # LSM1 with diffusion: Euler-Maruyama
            elif self.model==1 and self.diffusion_model!=0:
                 self.time_scheme = 1
            # LSM2 without diffusion: Direct Integrator (DI1)
            elif self.model==2 and self.diffusion_model==0:
                 self.time_scheme = 5
            # LSM2 without diffusion: Direct Integrator (DI2)
            elif self.model==2 and self.diffusion_model!=0:
                 self.time_scheme = 5
            # LSM3: Euler-Maruyama
            elif self.model==3:
                 self.time_scheme = 5

    def check_values(Parameters self):
        """ 
        Check values of parameters and raise ValueError if any parameter is invalid.
        """
        cdef bint failed_initialize = 0
        
        # TIME PARAMETERS:
        if self.time_step < 0.:
            failed_initialize = 1

        # LSM PARAMETERS:
        if self.model < 0:
            failed_initialize = 1
        if self.model > 3:
            failed_initialize = 1

        # LSM1
        if self.model == 1:
            if self.time_scheme < 1\
            or self.time_scheme > 4:
                failed_initialize = 1

        # LSM2
        elif self.model == 2:
            if self.diffusion_model == 0:
                if self.time_scheme < 1\
                or self.time_scheme > 5:
                    failed_initialize = 1
            else:
                if self.time_scheme != 1\
                and self.time_scheme != 5:
                    failed_initialize = 1
            if self.diffusion_lsm2_option < 1\
            or self.diffusion_lsm2_option > 3:
                failed_initialize = 1

        # LSM3
        elif self.model == 3:
            if self.diffusion_model == 0:
                failed_initialize = 1
            if self.time_scheme != 1\
            and self.time_scheme != 5:
                failed_initialize = 1

        if self.particle_velocity_init < 0:
            failed_initialize = 1
        if self.particle_velocity_init > 1:
            failed_initialize = 1

        if self.rng_method < 1:
            failed_initialize = 1
        if self.rng_method > 3:
            failed_initialize = 1

        # PHYSICAL PARAMETERS:
        if self.water_density < 0.:
            failed_initialize = 1

        if self.diffusion_model < 0:
            failed_initialize = 1
        if self.diffusion_model > 3:
            failed_initialize = 1

        if self.horizontal_diffusivity < 0.:
            failed_initialize = 1
        if self.vertical_diffusivity < 0.:
            failed_initialize = 1
        if self.schmidt_number <= 0.:
            failed_initialize = 1

        if self.diffusion_lsm2_option < 1:
            failed_initialize = 1
        if self.diffusion_lsm2_option > 3:
            failed_initialize = 1

        if self.diffusion_lsm3_tl_horizontal <= 0.:
            failed_initialize = 1
        if self.diffusion_lsm3_tl_vertical <= 0.:
            failed_initialize = 1

        if self.lsm3_edge_opt < 1:
            failed_initialize = 1
        if self.lsm3_edge_opt > 2:
            failed_initialize = 1

        # BOUNDARY_CONDITIONS:
        if self.max_wall_rebounds < 0:
            failed_initialize = 1

        if self.boundary_conditions_type < 1:
            failed_initialize = 1
        if self.boundary_conditions_type > 2:
            failed_initialize = 1

        if self.rebound_damping_coef < 0.:
            failed_initialize = 1
        if self.rebound_damping_coef > 1.:
            failed_initialize = 1

        if self.wall_roughness_angle < 0.:
            failed_initialize = 1

        if self.stranding_probability < 0.:
            failed_initialize = 1
        if self.stranding_probability > 1.:
            failed_initialize = 1

        # IO PARAMETERS:
        if self.listing_printout_period < 0:
            failed_initialize = 1

        if self.output_printout_period < 0:
            failed_initialize = 1

        if self.output_file_format not in ['txt', 'vtk', 'all']:
            failed_initialize = 1

        # raise error if failed_initialize is True
        if failed_initialize:
            raise ValueError("Failed initialization, check parameters")

    def check_types(Parameters self, reference, data, allow_missing_keys=True):
        """ 
        Check types of parameters 

        Parameters
        ----------
        reference : dict
            Reference dictionary with expected types.
        data : dict
            Dictionary to check against the reference.
        allow_missing_keys : bool, optional
            If True, missing keys in `data` that are present in `reference` will not raise an error. 
            Default is True.
        """
        errors = {}

        for key, value in data.items():

            if key not in reference:
                errors[key] = "unexpected key"

            elif not isinstance(value, type(reference[key])):
                errors[key] = (
                    f"expected {type(reference[key]).__name__}, "
                    f"got {type(value).__name__}"
                )

        if not allow_missing_keys:
            for key in reference:
                if key not in data:
                    errors[key] = "missing key"

        if errors is not None and len(errors) > 0:
            print(errors)
            raise TypeError(f"Failed initialization, type check failed with errors: {errors}")