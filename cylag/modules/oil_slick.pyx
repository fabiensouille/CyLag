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
from libc.math cimport sqrt, fmin, fmax, cos, sin, exp, tanh, pi, atan2, fabs
from ..core.constants cimport EPSILON, GRAV, KAPPA, EPSILON_LSM2
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..lsm.random_utils cimport random_gaussian_ziggurat

cdef class OilslickParticleSet(LagrangianParticleSet):
    """
    Oil slick particle class
    
    Requirements
    ------------
    Requires the following optional fields to be loaded in the EulerianFieldSet:
    opt_fields = ['BOTTOM', 'WIND X', 'WIND Y']

    Parameters
    ----------
    additional_velocity : bool, activate oilslick model based on added velocity
        WARNING: compatible with LSM-1 formulation only
    additional_force : bool, activate oilslick model based on added force
        WARNING: compatible with LSM-2 and 3 formulations only
    stokes_drift : bool, 
        activate oilslick stokes' drift (default: T)
    wind_coefficient : float,
        wind effect coefficient (default: 0.036)
    wind_attenuation_coef : float, 
        wind effect attenuation depth (default: 3.4657)
        Only active if dim=3 (pseudo-3D)
    wind_deviation_calibration_coef: float, 
        calibration coefficient for wind deviation angle (default: 1.)
        set to 0. to disable wind deviation
    wave_induced_vertical_dispersion : bool, 
        activate wave-induced vertical dispersion (default: T)
        Only active if dim=3 (pseudo-3D)
    """
    def __init__(self,\
            position=None, velocity=None, fluid_velocity_seen=None,\
            dim=3, initial_pool_size=10000, min_pool_size_update=1000,
            particle_density=980., particle_diameter=5.e-3,
            particle_diameter_distribution=['weibull', 1.8],
            drag_coefficient_model=1, drag_coefficient=0.44,
            added_mass_force=False, added_mass_coef=0.5,
            compute_depth=False,
            max_stranded=None,
            additional_velocity=True,
            additional_force=False,
            wind_coefficient=0.025,
            wind_attenuation_coef=0.69,
            wind_deviation_calibration_coef=1.,
            stokes_drift=True,
            wave_induced_vertical_dispersion=True):

        # intialize parent class
        LagrangianParticleSet.__init__(self,\
            position, velocity, fluid_velocity_seen,\
            dim, initial_pool_size, min_pool_size_update,
            particle_density, particle_diameter,
            particle_diameter_distribution,
            drag_coefficient_model, drag_coefficient,
            added_mass_force, added_mass_coef,
            buoyancy_velocity_model=1,
            buoyancy_velocity=0.0,
            compute_depth=compute_depth,
            max_stranded=max_stranded)

        # Mutually exclusive options
        if additional_velocity and additional_force:
            raise ValueError("additional_velocity (LSM-1) and additional_force (LSM-2/3) are mutually exclusive")

        # WARNING:
        if initial_pool_size < self.npart:
            raise ValueError("INITIAL POOL SIZE MUST BE > NPART")

        # additional velocity and force
        self.addvelocity = np.zeros((self.npart, self.dim), dtype='d')
        self.addforce = np.zeros((self.npart, self.dim), dtype='d')
        self.additional_velocity = additional_velocity
        self.additional_force = additional_force

        # Oil slick parameters
        self.wind_coefficient = wind_coefficient
        self.wind_attenuation_coef = wind_attenuation_coef
        self.wind_dev_coef = wind_deviation_calibration_coef
        self.stokes_drift = stokes_drift
        self.wave_induced_vertical_dispersion = wave_induced_vertical_dispersion

    cdef void _increase_pool_size(OilslickParticleSet self, int size):
        """
        Resize particle pool arrays including oil slick extra arrays.

        Overrides the parent method to also resize addvelocity, addforce,
        dp, hp, and zinit arrays when the pool expands (e.g. particle sources).

        Parameters
        ----------
        size : int
            Number of new particles to add
        """
        cdef int old_npart = self.npart

        # call parent to resize base arrays
        LagrangianParticleSet._increase_pool_size(self, size)

        # compute actual size added (parent uses max(min_pool_size_update, size))
        cdef int size_added = self.npart - old_npart

        # resize extra arrays
        self.addvelocity = np.concatenate(
            (self.addvelocity, np.zeros((size_added, self.dim), dtype='d')), axis=0)
        self.addforce = np.concatenate(
            (self.addforce, np.zeros((size_added, self.dim), dtype='d')), axis=0)

    cdef void _parallel_split(OilslickParticleSet self, int offset, int count, int pool_size):
        """
        Split oil slick particle set for parallel computation.

        Extracts extra per-particle arrays before calling parent, then
        rebuilds addvelocity, addforce, dp, hp, and zinit for this rank.

        Parameters
        ----------
        offset : int
            Starting index in the active particle arrays
        count : int
            Number of active particles to keep for this rank
        pool_size : int
            Total pool size allocated for this rank
        """
        # Call parent to handle base arrays
        LagrangianParticleSet._parallel_split(self, offset, count, pool_size)

        # Rebuild extra arrays
        self.addvelocity = np.zeros((self.npart, self.dim), dtype='d')
        self.addforce = np.zeros((self.npart, self.dim), dtype='d')

    cpdef void _sort_active_particles(OilslickParticleSet self):
        """
        Sort active particles including oil slick extra arrays.

        Overrides the parent method to also reorder addvelocity, addforce,
        dp, hp, and zinit arrays consistently with the base arrays.
        """
        # compute sorting index before parent modifies inactive array
        sorting_idx = np.argsort(self.inactive)

        # sort extra arrays
        self.addvelocity = np.asarray(self.addvelocity)[sorting_idx, :]
        self.addforce = np.asarray(self.addforce)[sorting_idx, :]

        # call parent to sort base arrays
        LagrangianParticleSet._sort_active_particles(self)

    cdef void add_velocity_i(OilslickParticleSet self, 
            EulerianFieldSet fset, int i, double dt):
        """ 
        Add custom velocity to the particle i (for LSM1 model)

        Parameters
        ----------
            i : int, particle index
            time : float, time
            dt : float, time step
        """
        cdef:
            double[:] xp = self.last_position[i, :]
            double[:] up = self.last_velocity[i, :]
            double[:] ua = self.addvelocity[i, :]
            double zs, zb, h, hp_i, Cf
            double wind_coef
            double u10, ux10, uy10, u_wind, ux_wind, uy_wind, wdir, wind_deviation
            double Hs, Tp, wp, kp, u_wave, ux_wave, uy_wave
            double Kz_wave
            double epsilon_h = 1.e-3
            double u10_min = 1.e-2
            double eps_kp = 1.e-8
            bint debug = False
            int j

        # Check required opt field IDs
        if fset.fid_zb == -1:
            raise ValueError(
                "OilslickParticleSet.add_velocity_i: requires 'BOTTOM' optional field "
                "in EulerianFieldSet (fid_zb == -1)")
        if fset.fid_windx == -1 or fset.fid_windy == -1:
            raise ValueError(
                "OilslickParticleSet.add_velocity_i: requires 'WIND X' and 'WIND Y' optional fields "
                "in EulerianFieldSet (fid_windx={}, fid_windy={})".format(
                    fset.fid_windx, fset.fid_windy))

        # interpolate surface and bottom elevation
        zs = fset._interpolate_field_2d(fset.fid_zs, self.tri[i], -1, xp[0], xp[1])
        zb = fset._interpolate_field_2d(fset.fid_zb, self.tri[i], -1, xp[0], xp[1])
        h = zs - zb

        if self.dim==3:
            hp_i = max(EPSILON, zs - xp[2])
        else:
            hp_i = 0.

        # Wind effect:
        # ************
        # interpolate wind velocity stored in optfields 7 and 8
        ux10 = fset._interpolate_field_2d(fset.fid_windx, self.tri[i], -1, xp[0], xp[1])
        uy10 = fset._interpolate_field_2d(fset.fid_windy, self.tri[i], -1, xp[0], xp[1])
        u10 = sqrt(ux10**2 + uy10**2)

        # wind effect atenuation with depth if pseudo-3D
        if self.dim==3:
            wind_coef = self.wind_coefficient*exp(-self.wind_attenuation_coef*hp_i)
        else:
            wind_coef = self.wind_coefficient

        # wind deviation angle (Zhang and Ozer (1992)):
        if (u10 < 25.):
            wind_deviation = self.wind_dev_coef*pi*(40. - 8.*sqrt(u10))/180.
        else:
            wind_deviation = 0.
        wind_deviation = max(0., wind_deviation+self.wind_attenuation_coef*hp_i)
        ux_wind = wind_coef*( ux10*cos(wind_deviation) + uy10*sin(wind_deviation))
        uy_wind = wind_coef*(-ux10*sin(wind_deviation) + uy10*cos(wind_deviation))

        # Wave parameters:
        # ****************
        # Pierson and Moskowitz (1964) approximation
        if u10 < u10_min:
            Hs = 0.
            Tp = 1.
            wp = 0.
            kp = 0.
        else:
            Hs = 0.24*(u10**2)/GRAV
            Tp = 7.69*u10/GRAV
            wp = (2.*pi)/Tp  # peak angular frequency
            kp = wp**2./GRAV # peak wave number
            # finite depth computation (dispersion relation):
            for j in range(20):
                kp_prev = kp
                kp = wp**2./(GRAV*fmax(tanh(kp*h), EPSILON))
                if fabs(kp - kp_prev) < eps_kp*fabs(kp):
                    break

        # Wave effect:
        # ************
        if self.stokes_drift:
            # Kastrounis (2022)
            u_wave = ((Hs**2.)/8.)*wp*kp*exp(-2.*kp*hp_i)
            # Wang et al. (2008)
            #u_wave = ((Hs**2.)/8.)*wp*kp*cosh(2.*kp*(zs-zp))/(sinh(kp*h)**2.)
            # bounding Stokes' drift
            u_wind = sqrt(ux_wind**2. + uy_wind**2.)
            u_wave = min(u_wave, 0.2*u_wind)
            wdir = atan2(uy10, ux10)
            ux_wave = u_wave*cos(wdir)
            uy_wave = u_wave*sin(wdir)
        else:
            ux_wave = 0.
            uy_wave = 0.

        # Surface current majoration
        # **************************
        # (only in 2D, otherwise included in pseudo-3D velocity correction)
        if self.dim==2:
            Cf = fset.friction_calc(h)
            surf_corr = sqrt(0.5*Cf)/KAPPA
        else:
            surf_corr = 0.

        # Update additional velocity (m/s)
        # ********************************
        if h > epsilon_h:
            ua[0] = up[0]*surf_corr + ux_wind + ux_wave
            ua[1] = up[1]*surf_corr + uy_wind + uy_wave
        else:
            ua[0] = 0.
            ua[1] = 0.

        # Extra wave-induced vertical dispersion
        # **************************************
        if self.dim==3:
            if self.wave_induced_vertical_dispersion:
                Kz_wave = 0.028*(Hs**2./Tp)*exp(-2.*kp*hp_i)
                ua[2] = sqrt(2.*Kz_wave/dt)*random_gaussian_ziggurat()
            else:
                ua[2] = 0.

    cdef void add_force_i(OilslickParticleSet self,
            EulerianFieldSet fset, int i, double dt):
        """ 
        Add custom force to the particle i (for LSM2 model)

        Parameters
        ----------
            i : int, particle index
            time : float, time
            dt : float, time step
        """
        cdef:
            double[:] xp = self.last_position[i, :]
            double[:] up = self.last_velocity[i, :]
            double[:] fa = self.addforce[i, :]
            double[3] ua
            double ux, uy, uz
            double zs, zb, h, hp_i, Cf
            double wind_coef
            double u10, ux10, uy10, u_wind, ux_wind, uy_wind, wdir, wind_deviation
            double Hs, Tp, wp, kp, u_wave, ux_wave, uy_wave
            double Kz_wave
            double epsilon_h = 1.e-3
            double u10_min = 1.e-2
            double eps_kp = 1.e-8
            double unorm, Cd, Sp
            bint debug = False
            int j

        # LOCK EXPERIMENTAL 
        raise ValueError("Additional force not implemented yet, use LSM-1 instead")

        # # Initialize ua
        # ua[0] = 0.
        # ua[1] = 0.
        # ua[2] = 0.

        # # Check required opt field IDs
        # if fset.fid_zb == -1:
        #     raise ValueError(
        #         "OilslickParticleSet.add_velocity_i: requires 'BOTTOM' optional field "
        #         "in EulerianFieldSet (fid_zb == -1)")
        # if fset.fid_windx == -1 or fset.fid_windy == -1:
        #     raise ValueError(
        #         "OilslickParticleSet.add_velocity_i: requires 'WIND X' and 'WIND Y' optional fields "
        #         "in EulerianFieldSet (fid_windx={}, fid_windy={})".format(
        #             fset.fid_windx, fset.fid_windy))

        # # interpolate surface and bottom elevation
        # zs = fset._interpolate_field_2d(fset.fid_zs, self.tri[i], -1, xp[0], xp[1])
        # zb = fset._interpolate_field_2d(fset.fid_zb, self.tri[i], -1, xp[0], xp[1])
        # h = zs - zb

        # if self.dim==3:
        #     hp_i = max(EPSILON, zs - xp[2])
        # else:
        #     hp_i = 0.

        # # Wind effect:
        # # ************
        # # interpolate wind velocity stored in optfields 7 and 8
        # ux10 = fset._interpolate_field_2d(fset.fid_windx, self.tri[i], -1, xp[0], xp[1])
        # uy10 = fset._interpolate_field_2d(fset.fid_windy, self.tri[i], -1, xp[0], xp[1])
        # u10 = sqrt(ux10**2 + uy10**2)

        # # wind effect atenuation with depth if pseudo-3D
        # if self.dim==3:
        #     wind_coef = self.wind_coefficient*exp(-self.wind_attenuation_coef*hp_i)
        # else:
        #     wind_coef = self.wind_coefficient

        # # wind deviation angle (Zhang and Ozer (1992)):
        # if (u10 < 25.):
        #     wind_deviation = self.wind_dev_coef*pi*(40. - 8.*sqrt(u10))/180.
        # else:
        #     wind_deviation = 0.
        # wind_deviation = max(0., wind_deviation+self.wind_attenuation_coef*hp_i)
        # ux_wind = wind_coef*( ux10*cos(wind_deviation) + uy10*sin(wind_deviation))
        # uy_wind = wind_coef*(-ux10*sin(wind_deviation) + uy10*cos(wind_deviation))

        # # Wave parameters:
        # # ****************
        # # Pierson and Moskowitz (1964) approximation
        # if u10 < u10_min:
        #     Hs = 0.
        #     Tp = 1.
        #     wp = 0.
        #     kp = 0.
        # else:
        #     Hs = 0.24*(u10**2)/GRAV
        #     Tp = 7.69*u10/GRAV
        #     wp = (2.*pi)/Tp  # peak angular frequency
        #     kp = wp**2./GRAV # peak wave number
        #     # finite depth computation (dispersion relation):
        #     for j in range(20):
        #         kp_prev = kp
        #         kp = wp**2./(GRAV*fmax(tanh(kp*h), EPSILON))
        #         if fabs(kp - kp_prev) < eps_kp*fabs(kp):
        #             break

        # # Wave effect:
        # # ************
        # if self.stokes_drift:
        #     # Kastrounis (2022)
        #     u_wave = ((Hs**2.)/8.)*wp*kp*exp(-2.*kp*hp_i)
        #     # Wang et al. (2008)
        #     #u_wave = ((Hs**2.)/8.)*wp*kp*cosh(2.*kp*(zs-zp))/(sinh(kp*h)**2.)
        #     # bounding Stokes' drift
        #     u_wind = sqrt(ux_wind**2. + uy_wind**2.)
        #     u_wave = min(u_wave, 0.2*u_wind)
        #     wdir = atan2(uy10, ux10)
        #     ux_wave = u_wave*cos(wdir)
        #     uy_wave = u_wave*sin(wdir)
        # else:
        #     ux_wave = 0.
        #     uy_wave = 0.

        # # Surface current majoration
        # # **************************
        # # (only in 2D, otherwise included in pseudo-3D velocity correction)
        # if self.dim==2:
        #     Cf = fset.friction_calc(h)
        #     surf_corr = sqrt(0.5*Cf)/KAPPA
        # else:
        #     surf_corr = 0.

        # # Update additional velocity (m/s)
        # # ********************************
        # if h > epsilon_h:
        #     ua[0] = up[0]*surf_corr + ux_wind + ux_wave
        #     ua[1] = up[1]*surf_corr + uy_wind + uy_wave
        # else:
        #     ua[0] = 0.
        #     ua[1] = 0.

        # # Extra wave-induced vertical dispersion
        # # **************************************
        # if self.dim==3:
        #     if self.wave_induced_vertical_dispersion:
        #         Kz_wave = 0.028*(Hs**2./Tp)*exp(-2.*kp*hp_i)
        #         ua[2] = sqrt(2.*Kz_wave/dt)*random_gaussian_ziggurat()
        #     else:
        #         ua[2] = 0.

        # # Compute additional force Fa (N) based on ua 
        # # *******************************************
        # # We assume relaxation with drag force scaling A1=1/taup,
        # # and use the following equivalence under taup -> 0:
        # # Fa = (A1/A3)*ua, meaning up relax to u(fluid) + ua.
        # # With definition of LSM-2 and LSM-3 model constants, we have:
        # # Fa = (A1c/A3c)*ua*Cd*Ur

        # # interpolate fluid velocity at particle position
        # if fset.dim == 2:
        #     ux, uy = fset._interpolate_velocity_2d(\
        #         self.tri[i], xp[0], xp[1])
        #     uz = 0.
        #     if self.dim == 3:
        #         uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
        #             self.tri[i], xp[0], xp[1], xp[2], dt)
        #         uz += uz_corr
        #         ux *= uc_corr
        #         uy *= uc_corr
        # else:
        #     ux, uy, uz = fset._interpolate_velocity_3d(\
        #         self.tri[i], self.lowerlayer[i], 0, xp[0], xp[1], xp[2])

        # # compute characteristic time scale (a1=1/taup)
        # if self.dim == 2:
        #     unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2)
        # else:
        #     unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2 + (uz-up[2])**2)

        # unorm = fmax(EPSILON_LSM2, unorm)
        # Cd = self.drag_coefficient(unorm, self.diameter[i])
        # Sp = 1.5*self.diameter[i]  # particle projected area (m^2)

        # # ratio A1/A3
        # aux = (self.a1c/self.a3c)*Cd*unorm*Sp
        # aux = fmin(10., aux) # clipping to avoid explosion of Fa

        # if h > epsilon_h:
        #     fa[0] = aux*ua[0]
        #     fa[1] = aux*ua[1]
        # else:
        #     fa[0] = 0.
        #     fa[1] = 0.

        # if self.dim==3:
        #     fa[2] = aux*ua[2]
