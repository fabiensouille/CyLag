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
import numpy as np
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.constants cimport NU0, EPSILON
from libc.math cimport fmax

cdef class MacrophyteParticleSet(LagrangianParticleSet):
    """
    Macrophyte particle class
    
    Macrophyte specific parameters
    ------------------------------
    drag_coefficient_model : int, 
        model for the drag coefficient (default: 1)
        - 0: Constant drag model
        - 1: Schiller et Nauman, (1935)
        - 2: Almedeij (2008)
        - 3: Sand-Jensen (2008) (use Cd_c, Cd_d, Cd_e, Cd_f coefficients depending on species)
    drag_coefficient : float, 
        constant drag coefficient for drag_coefficient_model=0 (default: 70.)
    biomass : float, 
        particle biomass used in the Sand-Jensen formula (default: 2.3)
    Cd_c : float, 
        coef in Sand-Jensen Cd formula (default: Renoncule, c=np.power(10, -0.75225))
    Cd_d : float, 
        coef in Sand-Jensen Cd formula (default: Renoncule, d=0.80969)
    Cd_e : float, 
        coef in Sand-Jensen Cd formula (default: Renoncule, e=0.)
    Cd_f : float, 
        coef in Sand-Jensen Cd formula (default: Renoncule, f=1.37888)

    """
    def __init__(self,\
            position=None, velocity=None, fluid_velocity_seen=None,\
            dim=2, initial_pool_size=10000, min_pool_size_update=1000,
            particle_density=1000., particle_diameter=0.245,
            particle_diameter_distribution=None,
            drag_coefficient_model=3, drag_coefficient=70.,
            added_mass_force=True, added_mass_coef=0.5,
            buoyancy_velocity_model=0, buoyancy_velocity=0.0,
            compute_depth=False, max_stranded=None,
            biomass=2.3, 
            Cd_c=np.power(10, -0.75225), 
            Cd_d=0.80969, 
            Cd_e=0., 
            Cd_f=1.37888):

        # intialize parent class
        LagrangianParticleSet.__init__(self,\
            position, velocity, fluid_velocity_seen,\
            dim, initial_pool_size, min_pool_size_update,
            particle_density, particle_diameter,
            particle_diameter_distribution,
            drag_coefficient_model, drag_coefficient,
            added_mass_force, added_mass_coef,
            buoyancy_velocity_model, buoyancy_velocity,
            compute_depth, max_stranded)

        # Macrophyte parameters
        self.biomass = biomass
        self.Cd_c = Cd_c
        self.Cd_d = Cd_d
        self.Cd_e = Cd_e
        self.Cd_f = Cd_f
        
    cpdef double drag_coefficient(MacrophyteParticleSet self, double U, double dp):
        """ 
        Computes the drag coefficient of a Macrophyte particle.

        Fd=(1/2)*rhof*Sp*Cd*|U|*U
        
        drag_coefficient_model :
            - 0: Constant drag model, Cd=cst
            - 1: Schiller et Nauman, (1935)
            - 2: Almedeij (2008)
            - 3: Sand-Jensen (2008) , Cd=(2./(rhof*Sp))*c*(B**d)*(U**(elog(B)+f-2))

        Coefficients depending on the species :
            - Renoncule :     c=np.power(10, -0.75225), d=0.80969, e=0.   , f=1.37888
            - Elodea/Egerie : c=np.power(10, -0.46)   , d=0.73   , e=-0.28, f=2.15

        Parameters
        ----------
        U   : float, relative velocity
        dp  : float, particle diameter

        Returns
        -------
        Cd : float, drag coefficient
        """
        cdef:
            double Re = fmax(EPSILON, U*dp/NU0)
            double Cd
            double surface = (np.pi*dp**2)/4.

        # Constant drag
        if self.drag_coef_model==0:
            Cd = self.drag_coef

       # Schiller et Nauman (1935)
        elif self.drag_coef_model==1:
            if Re <= 1000.:
                Cd = (24./Re)*(1. + 0.15*Re**0.687)
            else:
                Cd = 0.44

        # Almedeij (2008)
        elif self.drag_coef_model==2:
            phi1 = (24./Re)**10. + (21.*Re**-0.67)**10. + (4.*Re**-0.33)**10. + 0.4**10
            phi2 = 1./((0.148*Re**0.11)**-10. + (0.5)**-10.)
            phi3 = (1.57*(Re**-1.625)*1.e8 )**10.
            phi4 = 1./((6e-17*Re**2.63)**-10.  + 0.2**-10.)
            Cd = (1./( (phi1 + phi2)**-1. + phi3**-1.) + phi4)**(1/10.)

        # Sand-Jensen (2008) 
        elif self.drag_coef_model==3:
            Cd = 2.*self.Cd_c*(self.biomass**self.Cd_d)\
                 *(U**(self.Cd_e*np.log10(self.biomass)+self.Cd_f-2.))\
                 /(surface*self.rho0)
        else:
            Cd = 0.44

        return Cd
