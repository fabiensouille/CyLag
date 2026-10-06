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
from ..core.simutime cimport SimuTime
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..core.constants cimport CMU, EPSILON
from ..geom.xylocalizer cimport xy_localize
from .random_utils cimport generate_random_pair, generate_random_single
from libc.math cimport sqrt, fabs

cdef int update_state_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i):
    """
    Update kernel for the LSM-1 model, state vector: zp={xp}

    Parameters
    ----------
        simutime : cylag.SimuTime
        field_set : cylag.EulerianFieldSet
        particle_set : cylag.LagrangianParticleSet
        parameters : cylag.Parameters
    """
    cdef:
        int bnd_j = -1 # boundary edge crossed
        int time_scheme = parameters.time_scheme
        int diffmod = parameters.diffusion_model
        double dt = simutime.time_step
        double nux = parameters.horizontal_diffusivity
        double nuz = parameters.vertical_diffusivity
        double sigc = parameters.schmidt_number
        double buoy_vel
        double vert_vel_correction
        int tri0 = pset.tri[i]  # localization at the start of the step
        int ll0 = 0

    # Initialize
    # ~~~~~~~~~~
    pset.last_position[i, :] = pset.position[i, :]
    pset.last_velocity[i, :] = pset.velocity[i, :]
    if pset.dim==3:
        ll0 = pset.lowerlayer[i]

    # Advection:
    # ~~~~~~~~~~
    # Euler
    if time_scheme==1:
        euler_lsm1(simutime, fset, pset, i)

    # Second order Runge-Kutta
    elif time_scheme==2:
        bnd_j = rk2_lsm1(simutime, fset, pset, i)

    # Third order Runge-Kutta
    elif time_scheme==3:
        bnd_j = rk3_lsm1(simutime, fset, pset, i)

    # Fourth order Runge-Kutta
    elif time_scheme==4:
        bnd_j = rk4_lsm1(simutime, fset, pset, i)
    else:
        raise ValueError("Wrong time scheme")

    # Diffusion: Random Displacment Model (RDM)
    # ~~~~~~~~~~
    if diffmod!=0:
        diffusion_lsm1(
            simutime, fset, pset, i, diffmod, nux, nuz, sigc,
            parameters.rng_method, tri0, ll0)

    # Buoyancy velocity
    # ~~~~~~~~~~~~~~~~~
    if pset.dim==3:
        buoy_vel = pset.buoyancy_velocity(pset.diameter[i], 0, pset.rho0)
        pset.position[i, 2] += buoy_vel*dt
        pset.velocity[i, 2] += buoy_vel

    # Additional velocity
    # ~~~~~~~~~~~~~~~~~~~
    if pset.additional_velocity:
        pset.add_velocity_i(fset, i, simutime.time_step)
        # update state vector
        pset.position[i, 0] += pset.addvelocity[i, 0]*dt
        pset.position[i, 1] += pset.addvelocity[i, 1]*dt
        pset.velocity[i, 0] += pset.addvelocity[i, 0]
        pset.velocity[i, 1] += pset.addvelocity[i, 1]
        if pset.dim==3:
            pset.position[i, 2] += pset.addvelocity[i, 2]*dt
            pset.velocity[i, 2] += pset.addvelocity[i, 2]

    # localization
    # ~~~~~~~~~~~~
    if bnd_j==-1:
        pset.last_tri[i] = pset.tri[i]
        pset.tri[i], bnd_j = xy_localize(\
            fset.triangular_mesh,
            pset.last_tri[i],
            pset.last_position[i, 0],
            pset.last_position[i, 1],
            pset.position[i, 0], 
            pset.position[i, 1], 
            3, 0)

        if pset.dim==3:
            pset.lowerlayer[i] = fset.z_localize(\
                pset.tri[i],
                pset.position[i, 0],
                pset.position[i, 1],
                pset.position[i, 2])

    return bnd_j

cdef void euler_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i):
    """ Euler scheme for the LSM1 advection """
    cdef:
        double ux, uy, uz
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uc_corr = 1.0

    if fset.dim == 2:
        # interpolate mean fields at particle position
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        # interpolate mean fields at particle position
        ux, uy, uz = fset._interpolate_velocity_3d(\
            pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    # update particle state vector
    xp[0] += ux*dt
    xp[1] += uy*dt
    up[0] = ux
    up[1] = uy
    us[0] = ux
    us[1] = uy

    if pset.dim == 3:
        xp[2] += uz*dt
        up[2] = uz
        us[2] = uz

cdef int rk2_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i):
    """ Second order Runge-Kutta scheme for the LSM1 advection """
    cdef:
        double ux, uy, uz
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uz_corr = 0., uc_corr = 1.0
        double xtmp = 0., ytmp = 0., ztmp = 0.
        int tmp_tri
        int bnd_jj = -1
        int tmp_ll = 0

    # step 1
    # ------
    # interpolate mean fields at particle position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(pset.tri[i],\
            pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    up[0] = ux
    up[1] = uy
    us[0] = ux
    us[1] = uy
    if pset.dim == 3:
        up[2] = uz
        us[2] = uz

    # update particle state vector to half-step
    xtmp = xp[0] + 0.5*ux*dt
    ytmp = xp[1] + 0.5*uy*dt
    if pset.dim == 3:
        ztmp = xp[2] + 0.5*uz*dt

    # localization
    pset.last_tri[i] = pset.tri[i]
    tmp_tri, bnd_jj = xy_localize(\
        fset.triangular_mesh, pset.last_tri[i],
        pset.last_position[i, 0],
        pset.last_position[i, 1],
        xtmp, ytmp, 3, 0)
    if pset.dim == 3:
        tmp_ll = fset.z_localize(tmp_tri, xtmp, ytmp, ztmp)

    # stop if particle is going outside domain (euler step only)
    if bnd_jj != -1 or (pset.dim == 3 and (tmp_ll == -2 or tmp_ll == -3)):
        pset.tri[i] = tmp_tri
        if pset.dim == 3:
            pset.lowerlayer[i] = tmp_ll
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # step 2
    # ------
    # interpolate mean fields at half-step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(tmp_tri, xtmp, ytmp)
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # update particle state vector (full step)
    xp[0] += ux*dt
    xp[1] += uy*dt
    if pset.dim == 3:
        xp[2] += uz*dt

    return -1

cdef int rk3_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i):
    """ Third order Runge-Kutta scheme for the LSM1 advection """
    cdef:
        double ux, uy, uz
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uz_corr = 0.,uc_corr = 1.0
        double k1x = 0., k1y = 0., k1z = 0.
        double k2x = 0., k2y = 0., k2z = 0.
        double k3x = 0., k3y = 0., k3z = 0.
        double xtmp, ytmp, ztmp
        int tmp_tri
        int bnd_jj = -1
        int tmp_ll = 0

    # stage 1
    # -------
    # interpolate mean fields at particle position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(pset.tri[i],\
            pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    up[0] = ux
    up[1] = uy
    us[0] = ux
    us[1] = uy
    if pset.dim == 3:
        up[2] = uz
        us[2] = uz

    # calculate k1
    k1x = ux*dt
    k1y = uy*dt
    if pset.dim == 3:
        k1z = uz*dt

    # update to 1/3 step
    xtmp = xp[0] + (1./3.)*k1x
    ytmp = xp[1] + (1./3.)*k1y
    if pset.dim == 3:
        ztmp = xp[2] + (1./3.)*k1z
    else:
        ztmp = 0.

    # localization
    pset.last_tri[i] = pset.tri[i]
    tmp_tri, bnd_jj = xy_localize(\
        fset.triangular_mesh, pset.last_tri[i],
        pset.last_position[i, 0],
        pset.last_position[i, 1],
        xtmp, ytmp, 3, 0)
    if pset.dim == 3:
        tmp_ll = fset.z_localize(tmp_tri, xtmp, ytmp, ztmp)

    # stop if particle is going outside domain
    if bnd_jj != -1 or (pset.dim == 3 and (tmp_ll == -2 or tmp_ll == -3)):
        pset.tri[i] = tmp_tri
        if pset.dim == 3:
            pset.lowerlayer[i] = tmp_ll
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # stage 2
    # -------
    # interpolate mean fields at 1/3 step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(tmp_tri, xtmp, ytmp)
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # calculate k2
    k2x = ux*dt
    k2y = uy*dt
    if pset.dim == 3:
        k2z = uz*dt

    # update to 2/3 step
    xtmp = xp[0] + (2./3.)*k2x
    ytmp = xp[1] + (2./3.)*k2y
    if pset.dim == 3:
        ztmp = xp[2] + (2./3.)*k2z
    else:
        ztmp = 0.

    # localization
    pset.last_tri[i] = pset.tri[i]
    tmp_tri, bnd_jj = xy_localize(\
        fset.triangular_mesh, pset.last_tri[i],
        pset.last_position[i, 0],
        pset.last_position[i, 1],
        xtmp, ytmp, 3, 0)
    if pset.dim == 3:
        tmp_ll = fset.z_localize(tmp_tri, xtmp, ytmp, ztmp)

    # stop if particle is going outside domain
    if bnd_jj != -1 or (pset.dim == 3 and (tmp_ll == -2 or tmp_ll == -3)):
        pset.tri[i] = tmp_tri
        if pset.dim == 3:
            pset.lowerlayer[i] = tmp_ll
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # stage 3
    # -------
    # interpolate mean fields at 2/3 step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(tmp_tri, xtmp, ytmp)
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # calculate k3
    k3x = ux*dt
    k3y = uy*dt
    if pset.dim == 3:
        k3z = uz*dt

    # update (weighted average of k values)
    xp[0] += (0.25*k1x + 0.75*k3x)
    xp[1] += (0.25*k1y + 0.75*k3y)
    if pset.dim == 3:
        xp[2] += (0.25*k1z + 0.75*k3z)

    return -1

cdef int rk4_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i):
    """ Fourth order Runge-Kutta scheme for the LSM1 advection """
    cdef:
        double ux, uy, uz
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uz_corr = 0., uc_corr = 1.0
        double k1x = 0., k1y = 0., k1z = 0.
        double k2x = 0., k2y = 0., k2z = 0.
        double k3x = 0., k3y = 0., k3z = 0.
        double k4x = 0., k4y = 0., k4z = 0.
        double xtmp, ytmp, ztmp
        int tmp_tri
        int bnd_jj = -1
        int tmp_ll = 0

    # pseudo step 1
    # -------------
    # interpolate mean fields at particle position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(pset.tri[i],\
            pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    up[0] = ux
    up[1] = uy
    us[0] = ux
    us[1] = uy
    if pset.dim == 3:
        up[2] = uz
        us[2] = uz

    # calculate k1
    k1x = ux*dt
    k1y = uy*dt
    if pset.dim == 3:
        k1z = uz*dt

    # update to half-step
    xtmp = xp[0] + 0.5*k1x
    ytmp = xp[1] + 0.5*k1y
    if pset.dim == 3:
        ztmp = xp[2] + 0.5*k1z
    else:
        ztmp = 0.

    # localization
    pset.last_tri[i] = pset.tri[i]
    tmp_tri, bnd_jj = xy_localize(\
        fset.triangular_mesh, pset.last_tri[i],
        pset.last_position[i, 0],
        pset.last_position[i, 1],
        xtmp, ytmp, 3, 0)
    if pset.dim == 3:
        tmp_ll = fset.z_localize(tmp_tri, xtmp, ytmp, ztmp)

    # stop if particle is going outside domain (euler step only)
    if bnd_jj != -1 or (pset.dim == 3 and (tmp_ll == -2 or tmp_ll == -3)):
        pset.tri[i] = tmp_tri
        if pset.dim == 3:
            pset.lowerlayer[i] = tmp_ll
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # pseudo step 2
    # -------------
    # interpolate mean fields at half-step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(tmp_tri, xtmp, ytmp)
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # calculate k2
    k2x = ux*dt
    k2y = uy*dt
    if pset.dim == 3:
        k2z = uz*dt

    # update to half-step
    xtmp = xp[0] + 0.5*k2x
    ytmp = xp[1] + 0.5*k2y
    if pset.dim == 3:
        ztmp = xp[2] + 0.5*k2z
    else:
        ztmp = 0.

    # localization
    pset.last_tri[i] = pset.tri[i]
    tmp_tri, bnd_jj = xy_localize(\
        fset.triangular_mesh, pset.last_tri[i],
        pset.last_position[i, 0],
        pset.last_position[i, 1],
        xtmp, ytmp, 3, 0)
    if pset.dim == 3:
        tmp_ll = fset.z_localize(tmp_tri, xtmp, ytmp, ztmp)

    # stop if particle is going outside domain (euler step only)
    if bnd_jj != -1 or (pset.dim == 3 and (tmp_ll == -2 or tmp_ll == -3)):
        pset.tri[i] = tmp_tri
        if pset.dim == 3:
            pset.lowerlayer[i] = tmp_ll
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # pseudo step 3
    # -------------
    # interpolate mean fields at half-step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(tmp_tri, xtmp, ytmp)
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # calculate k3
    k3x = ux*dt
    k3y = uy*dt
    if pset.dim == 3:
        k3z = uz*dt

    # update to full step
    xtmp = xp[0] + k3x
    ytmp = xp[1] + k3y
    if pset.dim == 3:
        ztmp = xp[2] + k3z
    else:
        ztmp = 0.

    # localization
    pset.last_tri[i] = pset.tri[i]
    tmp_tri, bnd_jj = xy_localize(\
        fset.triangular_mesh, pset.last_tri[i],
        pset.last_position[i, 0],
        pset.last_position[i, 1],
        xtmp, ytmp, 3, 0)
    if pset.dim == 3:
        tmp_ll = fset.z_localize(tmp_tri, xtmp, ytmp, ztmp)

    # stop if particle is going outside domain (euler step only)
    if bnd_jj != -1 or (pset.dim == 3 and (tmp_ll == -2 or tmp_ll == -3)):
        pset.tri[i] = tmp_tri
        if pset.dim == 3:
            pset.lowerlayer[i] = tmp_ll
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # pseudo step 4
    # -------------
    # interpolate mean fields at full-step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(tmp_tri, xtmp, ytmp)
        uz = 0.
        # pseudo 3D corrections
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # calculate k4
    k4x = ux*dt
    k4y = uy*dt
    if pset.dim == 3:
        k4z = uz*dt

    # update (weighted average of k values)
    xp[0] += (k1x + 2.*k2x + 2.*k3x + k4x)/6.
    xp[1] += (k1y + 2.*k2y + 2.*k3y + k4y)/6.
    if pset.dim == 3:
        xp[2] += (k1z + 2.*k2z + 2.*k3z + k4z)/6.

    return -1

cdef void diffusion_lsm1(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i, int diffmod, 
        double nux, double nuz, double sigc,
        int rng_method, int tri0, int ll0):
    """ 
    Euler-Maruyama scheme for the LSM1 diffusion

    Note: only diffusion part is computed here, including the Ito drift
    dK/dx_i. K and its gradient are evaluated at the start-of-step position
    (last_position) in the start-of-step element (tri0, ll0).
    """
    cdef:
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double sqrtdt = sqrt(dt)
        double xix = 0., xiy = 0., xiz = 0.
        double diffx = 0., diffy = 0., diffz = 0.
        double driftx = 0., drifty = 0., driftz = 0.
        double Kp, dKdx, dKdy, dKdz
        double z0 = 0.

    # diffusion
    # ~~~~~~~~~
    if diffmod==2 or diffmod==3:
        # ---------------------
        # k-eps diffusion model
        # ---------------------
        if pset.dim == 3:
            z0 = pset.last_position[i, 2]
        Kp, dKdx, dKdy, dKdz = fset._interpolate_diffusivity_gradient_c(\
            tri0, ll0,
            pset.last_position[i, 0], pset.last_position[i, 1], z0, sigc)

        diffx = sqrt(2.*Kp)*sqrtdt
        diffy = diffx
        driftx = dKdx*dt
        drifty = dKdy*dt
        if pset.dim == 3:
            if diffmod==2:
                # constant vertical diffusivity: no vertical drift
                diffz = sqrt(2.*nuz)*sqrtdt
            else:
                diffz = diffx
                driftz = dKdz*dt
    else:
        # --------------------
        # constant diffusivity
        # --------------------
        diffx = sqrt(2.*nux)*sqrtdt
        diffy = diffx
        if pset.dim == 3:
            diffz = sqrt(2.*nuz)*sqrtdt

    # Wiener increment only when diffusivity is non-zero
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if diffx != 0.:
        xix, xiy = generate_random_pair(rng_method)
    if pset.dim == 3 and diffz != 0.:
        xiz = generate_random_single(rng_method)

    # update particle state vector
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    xp[0] += driftx + diffx*xix
    xp[1] += drifty + diffy*xiy
    if pset.dim == 3:
        xp[2] += driftz + diffz*xiz