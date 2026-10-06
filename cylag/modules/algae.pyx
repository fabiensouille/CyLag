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
from libc.math cimport fmax, exp, log
from ..core.constants cimport NU0, EPSILON
from ..core.lagrangian_particle_set cimport LagrangianParticleSet

cdef class AlgaeParticleSet(LagrangianParticleSet):
    """
    Algae particle class
    
    Parameters
    ----------
    drag_coefficient_model : int, 
        model for the drag coefficient (default: 1)
        - 0: Constant drag model, Cd=cst
        - 1: Schiller et Nauman, (1935)
        - 2: Almedeij (2008)
        - 3: Algae formula: Cd = exp(self.Cd_a - self.Cd_b*log(Re)) 
    Cd_a : float, 
        coef in algae Cd formula (default: Iridaea Flaccida, a=6.822121)
    Cd_b : float, 
        coef in algae Cd formula (default: Iridaea Flaccida, b=0.800627)

    """
    def __init__(self,\
            position=None, velocity=None, fluid_velocity_seen=None,\
            dim=2, initial_pool_size=10000, min_pool_size_update=1000,
            particle_density=1000., particle_diameter=0.245,
            drag_coefficient_model=1, drag_coefficient=70.,
            particle_diameter_distribution=None,
            added_mass_force=True, added_mass_coef=0.5,
            Cd_a=6.822121, Cd_b=0.800627):

        # intialize parent class
        LagrangianParticleSet.__init__(self,\
            position, velocity, fluid_velocity_seen,\
            dim, initial_pool_size, min_pool_size_update,
            particle_density, particle_diameter,
            particle_diameter_distribution,
            drag_coefficient_model, drag_coefficient,
            added_mass_force, added_mass_coef)

        # Algae parameters
        self.Cd_a = Cd_a
        self.Cd_b = Cd_b

    cpdef double drag_coefficient(AlgaeParticleSet self, double U, double dp):
        """ 
        Computes the drag coefficient of a Algae particle.

        Fd=(1/2)*rhof*Sp*Cd*|U|*U
        
        drag_coefficient_model :
            - 0: Constant drag model, Cd=cst
            - 1: Schiller et Nauman, (1935),
            - 2: Almedeij (2008),
            - 3: Algae formula: Cd = exp(self.Cd_a - self.Cd_b*log(Re)) 

        Coefficients depending on the species : 
            - 1: Iridaea Flaccida : Cd_a=6.822121, Cd_b=0.800627, 
            - 2: Pelvetiopsis Limitata : Cd_a=8.214783, Cd_b=0.877036,
            - 3: Gigartina Leptorhynchos : Cd_a=6.773712, Cd_b=0.774252.

        Parameters
        ----------
        U   : float, relative velocity

        Returns
        -------
        Cd : float, drag coefficient
        """
        cdef:
            double Re = fmax(EPSILON, U*dp/NU0)
            double Cd

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

        # Algae formula (with fallback to Schiller et Naumann):
        elif self.drag_coef_model==3:
            if Re >= 14073.:
                Cd = exp(self.Cd_a - self.Cd_b*log(Re))
            elif Re > 1000000.:
                Cd = 0.2
            elif Re <= 0.4:
                Cd = 24./Re
            else:
                Cd = (24./Re)*(1. + 0.15*Re**0.687)
        else:
            Cd = 0.44

        return Cd
