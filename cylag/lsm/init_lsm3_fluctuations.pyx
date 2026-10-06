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
from ..core.parameters cimport Parameters
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet
from .random_utils cimport generate_random_pair, generate_random_single
from libc.math cimport fmax, sqrt
from ..core.constants cimport EPSILON, EPSILON_LSM3, C0KOLM

cpdef (double,double,double) update_lsm3_fluctuations(\
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i):
    """
    Draw initial Us fluctuations from its Ornstein-Uhlenbeck stationary law.

    Constant diffusivity K gives variance K/TL. Model 2 uses k-epsilon
    horizontally and constant K vertically; model 3 uses isotropic k-epsilon.
    Up and Xp are not initialized here.

    Parameters
    ----------
        field_set : cylag.EulerianFieldSet
        particle_set : cylag.LagrangianParticleSet
        parameters : cylag.Parameters
    """
    cdef:
        double[:] xp = pset.position[i,:]
        int diffmod = parameters.diffusion_model
        double fluct1 = 0., fluct2 = 0., fluct3 = 0.
        double xix, xiy
        double turbeng, epsilon, tml, sigma_h, sigma_v

    # Only k-epsilon models require the optional turbulence fields.
    if diffmod == 2 or diffmod == 3:
        if fset.dim == 2:
            turbeng = fset._interpolate_field_2d(\
                fset.fid_turbeng, pset.tri[i],
                -1, xp[0], xp[1])
            epsilon = fset._interpolate_field_2d(\
                fset.fid_dissip, pset.tri[i],
                -1, xp[0], xp[1])
        else:
            turbeng = fset._interpolate_field_3d(\
                fset.fid_turbeng, pset.tri[i],
                pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])
            epsilon = fset._interpolate_field_3d(\
                fset.fid_dissip, pset.tri[i],
                pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])
        epsilon = fmax(epsilon, EPSILON)
        tml = fmax(EPSILON_LSM3,
                   (turbeng/epsilon)/(0.5 + (3./4.)*C0KOLM))
        sigma_h = sqrt(0.5*C0KOLM*epsilon*tml)
    else:
        sigma_h = sqrt(parameters.horizontal_diffusivity
                       /parameters.diffusion_lsm3_tl_horizontal)

    if sigma_h > 0.:
        xix, xiy = generate_random_pair(parameters.rng_method)
        fluct1 = sigma_h*xix
        fluct2 = sigma_h*xiy

    if pset.dim == 3:
        if diffmod == 3:
            sigma_v = sigma_h
        else:
            sigma_v = sqrt(parameters.vertical_diffusivity
                           /parameters.diffusion_lsm3_tl_vertical)
        if sigma_v > 0.:
            fluct3 = sigma_v*generate_random_single(parameters.rng_method)

    return fluct1, fluct2, fluct3