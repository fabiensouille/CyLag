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
from ..core.constants cimport EPSILON, EPSILON_LSM2, C0KOLM

cpdef (double,double,double) update_lsm2_fluctuations(\
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i):
    """
    Draw an initial fluctuation for Up from its Ornstein-Uhlenbeck stationary distribution.
    Designed for LSM2 diffusion-on-velocity variant.

    Parameters
    ----------
        simutime : cylag.SimuTime
        field_set : cylag.EulerianFieldSet
        particle_set : cylag.LagrangianParticleSet
        parameters : cylag.Parameters
    """
    cdef:
        double[:] xp = pset.position[i,:]
        double fluct1 = 0., fluct2 = 0., fluct3 = 0.
        double unorm, Cd, a1
        double xix, xiy, xiz
        double turbeng, epsilon, tml
        double b2tau0, b2tau1, b2tau2

    if parameters.diffusion_model == 0:
        return 0., 0., 0.

    # relative velocity ~ 0 since Up = mean flow at this point
    unorm = EPSILON_LSM2
    Cd = pset.drag_coefficient(unorm, pset.diameter[i])
    a1 = pset.a1c*(1.5/pset.diameter[i])*Cd*unorm

    # With B != 0 an undamped velocity is Brownian, not stationary.
    # Option-3 and k-epsilon amplitudes are proportional to a1 and vanish
    # here. Only explicit option-2 amplitudes can remain nonzero.
    if a1 == 0.:
        if parameters.diffusion_lsm2_option == 2:
            if ((parameters.diffusion_model not in (2, 3) and
                 parameters.horizontal_diffusivity != 0.) or
                (pset.dim == 3 and parameters.diffusion_model != 3 and
                 parameters.vertical_diffusivity != 0.)):
                raise ValueError(
                    "Stationary LSM2 initialization requires positive damping "
                    "when the velocity noise amplitude B is nonzero")
        return 0., 0., 0.

    # Draw random fluctuations for Up from its stationary distribution
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # If k-eps model:
    if parameters.diffusion_model == 2 or \
        parameters.diffusion_model == 3:

        # 2D case:
        if fset.dim == 2:
            turbeng = fset._interpolate_field_2d(\
                fset.fid_turbeng, pset.tri[i],
                -1, xp[0], xp[1])
            epsilon = fset._interpolate_field_2d(\
                fset.fid_dissip, pset.tri[i],
                -1, xp[0], xp[1])
        # 3D case:
        else:
            turbeng = fset._interpolate_field_3d(\
                fset.fid_turbeng, pset.tri[i],
                pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])
            epsilon = fset._interpolate_field_3d(\
                fset.fid_dissip, pset.tri[i],
                pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

        epsilon = fmax(epsilon, EPSILON)
        tml = (turbeng/epsilon)/(0.5 + (3./4.)*C0KOLM)
        b2tau0 = a1*C0KOLM*epsilon*tml**2
        b2tau1 = b2tau0
        if parameters.diffusion_model == 2:
            if parameters.diffusion_lsm2_option == 2:
                b2tau2 = parameters.vertical_diffusivity**2/a1
            else:
                b2tau2 = 2.*parameters.vertical_diffusivity*a1
        else:
            b2tau2 = b2tau0

    # constant diffusivity
    else:
        if parameters.diffusion_lsm2_option == 2:
            b2tau0 = parameters.horizontal_diffusivity**2/a1
            b2tau1 = b2tau0
            b2tau2 = parameters.vertical_diffusivity**2/a1
        else:
            b2tau0 = 2.*parameters.horizontal_diffusivity*a1
            b2tau1 = b2tau0
            b2tau2 = 2.*parameters.vertical_diffusivity*a1

    # Zero fluctuations must not advance the shared random stream.
    if b2tau0 > 0.:
        xix, xiy = generate_random_pair(parameters.rng_method)
        fluct1 = sqrt(0.5*b2tau0)*xix
        fluct2 = sqrt(0.5*b2tau1)*xiy

    if pset.dim == 3 and b2tau2 > 0.:
        xiz = generate_random_single(parameters.rng_method)
        fluct3 = sqrt(0.5*b2tau2)*xiz

    return fluct1, fluct2, fluct3