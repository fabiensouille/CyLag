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

cdef class JellyfishParticleSet(LagrangianParticleSet):
    """
    Jellyfish particle class

    Drag formula from : Wang, K. et al. (2026), 
    An improved Lagrangian particle tracking method with tentacle length correction for jellyfish,
    Ocean Engineering, 364, 127154.
    
    Parameters
    ----------
    drag_coefficient_model : int, 
        model for the drag coefficient (default: 1)
        - 0: Constant drag model, Cd=cst
        - 1: Schiller et Nauman, (1935)
        - 2: Almedeij (2008)
        - 3: Jellyfish formula from Daniel (1983) and Wang & al. (2026)
    Cd_a : float, 
        coef in Jellyfish Cd formula (default: 0.0285)
    Cd_b : float, 
        coef in Jellyfish Cd formula (default: -0.0002)
    Cd_c1 : float, 
        coef in Jellyfish Cd formula (default: 381.4771)
    Cd_d1 : float, 
        coef in Jellyfish Cd formula (default: 33.1604)
    Cd_c2 : float, 
        coef in Jellyfish Cd formula (default: 125.8822)
    Cd_d2 : float, 
        coef in Jellyfish Cd formula (default: 1.153211)
    bell_diameter_ratio : float
        Bell length ratio (bell_height/diameter) (default: 1.)
    tentacle_length_ratio : float
        Tentacle length ratio (length/bell_height) (default: 4.)
    """
    def __init__(self,\
            position=None, velocity=None, fluid_velocity_seen=None,\
            dim=2, initial_pool_size=10000, min_pool_size_update=1000,
            particle_density=1000., particle_diameter=0.25,
            particle_diameter_distribution=None,
            drag_coefficient_model=1, drag_coefficient=70.,
            added_mass_force=True, added_mass_coef=0.5,
            buoyancy_velocity_model=0, buoyancy_velocity=0.0,
            compute_depth=False, max_stranded=None,
            Cd_a=0.0285, Cd_b=-0.0002,
            Cd_c1=381.4771, Cd_d1=33.1604,
            Cd_c2=125.8822, Cd_d2=1.153211,
            bell_diameter_ratio=0.5, 
            tentacle_length_ratio=4.):

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

        # Jellyfish parameters
        self.Cd_a = Cd_a
        self.Cd_b = Cd_b
        self.Cd_c1 = Cd_c1
        self.Cd_d1 = Cd_d1
        self.Cd_c2 = Cd_c2
        self.Cd_d2 = Cd_d2
        self.bdr = bell_diameter_ratio
        self.tlr = tentacle_length_ratio

        # Added mass correction
        self.added_mass_coef = (2.0*self.bdr)**1.4

    cpdef double drag_coefficient(JellyfishParticleSet self, double U, double dp):
        """ 
        Computes the drag coefficient of a Jellyfish particle.

        Fd=(1/2)*rhof*Sp*Cd*|U|*U
        
        drag_coefficient_model :
            - 0: Constant drag model, Cd=cst
            - 1: Schiller et Nauman, (1935),
            - 2: Almedeij (2008),
            - 3: Jellyfish formula from Daniel (1983) and Wang & al. (2026)

        Parameters
        ----------
        U   : float, relative velocity

        Returns
        -------
        Cd : float, drag coefficient
        """
        cdef:
            double Re = fmax(EPSILON, U*dp/NU0)
            double Cd, c, d
            double bell_height = self.bdr*dp

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

        # Jellyfish formula from Daniel (1983) and Wang & al. (2026)
        elif self.drag_coef_model==3:
            # Daniel (1983)
            Cf = 1.328/np.sqrt(Re)
            Cd0 = 0.33*(dp/bell_height) + Cf*(3.*bell_height/dp + 3.*np.sqrt(dp/bell_height))
            # Wang et al. (2026)
            if Re < 220.:
                c = self.Cd_c1
                d = self.Cd_d1
            else:
                c = self.Cd_c2
                d = self.Cd_d2
            Cd = Cd0*(1. + self.Cd_a*self.tlr + self.Cd_b*self.tlr**2)\
               * (1. + c/Re + d/np.sqrt(Re))
        else:
            Cd = 0.44

        return Cd
