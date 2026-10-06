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
from ..core.constants cimport CMU, EPSILON, EPSILON_LSM2, GRAV, C0KOLM
from ..geom.xylocalizer cimport xy_localize
from .random_utils cimport generate_random_pair, generate_random_single
from libc.math cimport fmax, fabs, sqrt, exp, expm1, isfinite

cdef int update_state_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i):
    """
    Update kernel for the LSM-2 model, state vector: zp={xp,Up}

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
        double nux = parameters.horizontal_diffusivity
        double nuz = parameters.vertical_diffusivity
        int nuopt = parameters.diffusion_lsm2_option
        double sigc = parameters.schmidt_number
        int tri0 = pset.tri[i]  # localization at the start of the step
        int ll0 = 0

    # Initialize
    # ~~~~~~~~~~
    pset.last_position[i, :] = pset.position[i, :]
    pset.last_velocity[i, :] = pset.velocity[i, :]
    if pset.dim==3:
        ll0 = pset.lowerlayer[i]

    # Additional force
    # ~~~~~~~~~~~~~~~~
    if pset.additional_force:
        pset.add_force_i(fset, i, simutime.time_step)

    # Advection-Diffusion 
    # ~~~~~~~~~~~~~~~~~~~
    # Diffusion on position or no diffusion
    if nuopt==1 or (nuopt>=2 and diffmod==0):
        # Euler
        if time_scheme==1:
            euler_lsm2(simutime, fset, pset, i)

        # Second order Runge-Kutta
        elif time_scheme==2:
            bnd_j = rk2_lsm2(simutime, fset, pset, i)

        # Third order Runge-Kutta
        elif time_scheme==3:
            bnd_j = rk3_lsm2(simutime, fset, pset, i)

        # Fourth order Runge-Kutta
        elif time_scheme==4:
            bnd_j = rk4_lsm2(simutime, fset, pset, i)

        # Direct integrator without diffusion
        elif time_scheme==5:
            di_ode_lsm2(simutime, fset, pset, i)
        else:
            raise ValueError("Wrong time scheme")

        # Diffusion
        if diffmod!=0:
            diffusion_lsm2(simutime, fset, pset, i,\
                diffmod, nux, nuz, sigc,
                parameters.rng_method, tri0, ll0)

    # Diffusion on velocity
    elif nuopt>=2 and diffmod!=0:
        # Euler-Maruyama
        if time_scheme==1:
            euler_maruyama_lsm2_u(simutime, fset, pset, i,\
                diffmod, nux, nuz, nuopt, sigc,
                parameters.rng_method)

        # Full Direct Integrator with diffusion
        elif time_scheme==5:
            di_lsm2(simutime, fset, pset, i, diffmod,\
                nux, nuz, nuopt, sigc,
                parameters.rng_method)
        else:
            raise ValueError("Wrong time scheme")
      
    else:
        raise ValueError("Wrong diffusion_lsm2_option")

    # localization
    # ~~~~~~~~~~~~
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

cdef void euler_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i):
    """ Euler scheme for the LSM2 advection """
    cdef:
        double ux, uy, uz
        double ax = 0., ay = 0., az = 0.
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uz_corr = 0., uc_corr = 1.0
        # LSM2 parameters
        bint admass = pset.added_mass_force
        double a0c = pset.a0c
        double a1c = pset.a1c
        double a2c = pset.a2c
        double a3c = pset.a3c
        double dp = pset.diameter[i]
        double vp = pset.volume[i]
        double unorm, Cd, a1

    # interpolate mean fields at particle position (conditional on fset.dim)
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                pset.tri[i], xp[0], xp[1], dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                pset.tri[i], pset.lowerlayer[i], 
                xp[0], xp[1], xp[2], dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2)
    else:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2 + (uz-up[2])**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update particle state vector
    us[0] = ux
    us[1] = uy
    up[0] += (ux - up[0])*dt*a1 + ax*a2c*dt
    up[1] += (uy - up[1])*dt*a1 + ay*a2c*dt
    xp[0] += up[0]*dt
    xp[1] += up[1]*dt

    if pset.dim == 3:
        us[2] = uz
        up[2] += (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        xp[2] += up[2]*dt

    if pset.additional_force:
        up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
        up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
        if pset.dim == 3:
            up[2] += (a3c/vp)*pset.addforce[i, 2]*dt

cdef int rk2_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i):
    """ Second order Runge-Kutta scheme for the LSM2 advection """
    cdef:
        double ux, uy, uz
        double ax = 0., ay = 0., az = 0.
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uz_corr = 0., uc_corr = 1.0
        # LSM2 parameters
        bint admass = pset.added_mass_force
        double a0c = pset.a0c
        double a1c = pset.a1c
        double a2c = pset.a2c
        double a3c = pset.a3c
        double dp = pset.diameter[i]
        double vp = pset.volume[i]
        double unorm, Cd, a1
        # RK2 parameters
        double k1x = 0., k1y = 0., k1z = 0.
        double k2x = 0., k2y = 0., k2z = 0.
        double f1x = 0., f1y = 0., f1z = 0.
        double f2x = 0., f2y = 0., f2z = 0.
        double xtmp = 0., ytmp = 0., ztmp = 0.
        int tmp_tri, bnd_jj = -1
        int tmp_ll = 0

    # step 1
    # ------
    # interpolate mean fields at particle position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                pset.tri[i], xp[0], xp[1], dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                pset.tri[i], pset.lowerlayer[i], 
                xp[0], xp[1], xp[2], dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2)
    else:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2 + (uz-up[2])**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update state (pseudo-step)
    k1x = up[0]*dt
    k1y = up[1]*dt
    f1x = (ux - up[0])*dt*a1 + ax*a2c*dt
    f1y = (uy - up[1])*dt*a1 + ay*a2c*dt
    
    xtmp = xp[0] + 0.5*k1x
    ytmp = xp[1] + 0.5*k1y
    if pset.dim == 3:
        k1z = up[2]*dt
        f1z = (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        ztmp = xp[2] + 0.5*k1z

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
        us[0] = ux
        us[1] = uy
        up[0] += (ux - up[0])*dt*a1 + ax*a2c*dt
        up[1] += (uy - up[1])*dt*a1 + ay*a2c*dt
        if pset.dim == 3:
            us[2] = uz
            up[2] += (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        if pset.additional_force:
            up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
            up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
            if pset.dim == 3:
                up[2] += (a3c/vp)*pset.addforce[i, 2]*dt
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # step 2
    # ------
    # interpolate mean fields at half-step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            tmp_tri, xtmp, ytmp)
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                tmp_tri, xtmp, ytmp, dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                tmp_tri, tmp_ll, xtmp, ytmp, ztmp, dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0]-0.5*f1x)**2 + (uy-up[1]-0.5*f1y)**2)
    else:
        unorm = sqrt((ux-up[0]-0.5*f1x)**2 + (uy-up[1]-0.5*f1y)**2 + (uz-up[2]-0.5*f1z)**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update state (pseudo-step)
    k2x = (up[0] + 0.5*f1x)*dt
    k2y = (up[1] + 0.5*f1y)*dt
    f2x = (ux - (up[0] + 0.5*f1x))*dt*a1 + ax*a2c*dt
    f2y = (uy - (up[1] + 0.5*f1y))*dt*a1 + ay*a2c*dt
    
    # update particle state vector
    us[0] = ux
    us[1] = uy
    up[0] += f2x
    up[1] += f2y
    xp[0] += k2x
    xp[1] += k2y

    if pset.dim == 3:
        k2z = (up[2] + 0.5*f1z)*dt
        f2z = (uz - (up[2] + 0.5*f1z))*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        us[2] = uz
        up[2] += f2z
        xp[2] += k2z

    if pset.additional_force:
        up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
        up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
        if pset.dim == 3:
            up[2] += (a3c/vp)*pset.addforce[i, 2]*dt

    return -1

cdef int rk3_lsm2(
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i):
    """ Third order Runge-Kutta scheme for the LSM2 advection """
    cdef:
        double ux, uy, uz
        double ax = 0., ay = 0., az = 0.
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uz_corr = 0., uc_corr = 1.0
        # LSM2 parameters
        bint admass = pset.added_mass_force
        double a0c = pset.a0c
        double a1c = pset.a1c
        double a2c = pset.a2c
        double a3c = pset.a3c
        double dp = pset.diameter[i]
        double vp = pset.volume[i]
        double unorm, Cd, a1
        # RK3 parameters
        double k1x = 0., k1y = 0., k1z = 0.
        double k2x = 0., k2y = 0., k2z = 0.
        double k3x = 0., k3y = 0., k3z = 0.
        double f1x = 0., f1y = 0., f1z = 0.
        double f2x = 0., f2y = 0., f2z = 0.
        double f3x = 0., f3y = 0., f3z = 0.
        double xtmp, ytmp, ztmp = 0.
        int tmp_tri, bnd_jj = -1
        int tmp_ll = 0

    # stage 1
    # -------
    # interpolate mean fields at particle position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                pset.tri[i], xp[0], xp[1], dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                pset.tri[i], pset.lowerlayer[i], \
                xp[0], xp[1], xp[2], dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2)
    else:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2 + (uz-up[2])**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update state (pseudo-step)
    k1x = up[0]*dt
    k1y = up[1]*dt
    f1x = (ux - up[0])*dt*a1 + ax*a2c*dt
    f1y = (uy - up[1])*dt*a1 + ay*a2c*dt
    
    xtmp = xp[0] + (1./3.)*k1x
    ytmp = xp[1] + (1./3.)*k1y
    if pset.dim == 3:
        k1z = up[2]*dt
        f1z = (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        ztmp = xp[2] + (1./3.)*k1z

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
        us[0] = ux
        us[1] = uy
        up[0] += (ux - up[0])*dt*a1 + ax*a2c*dt
        up[1] += (uy - up[1])*dt*a1 + ay*a2c*dt
        if pset.dim == 3:
            us[2] = uz
            up[2] += (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        if pset.additional_force:
            up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
            up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
            if pset.dim == 3:
                up[2] += (a3c/vp)*pset.addforce[i, 2]*dt
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # stage 2
    # -------
    # interpolate mean fields at 1/3 step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            tmp_tri, xtmp, ytmp)
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                tmp_tri, xtmp, ytmp, dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                tmp_tri, tmp_ll, xtmp, ytmp, ztmp, dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0]-(1./3.)*f1x)**2 + (uy-up[1]-(1./3.)*f1y)**2)
    else:
        unorm = sqrt((ux-up[0]-(1./3.)*f1x)**2 + (uy-up[1]-(1./3.)*f1y)**2 + (uz-up[2]-(1./3.)*f1z)**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update state (pseudo-step)
    k2x = (up[0] + (1./3.)*f1x)*dt
    k2y = (up[1] + (1./3.)*f1y)*dt
    f2x = (ux - (up[0] + (1./3.)*f1x))*dt*a1 + ax*a2c*dt
    f2y = (uy - (up[1] + (1./3.)*f1y))*dt*a1 + ay*a2c*dt
    
    xtmp = xp[0] + (2./3.)*k2x
    ytmp = xp[1] + (2./3.)*k2y

    if pset.dim == 3:
        k2z = (up[2] + (1./3.)*f1z)*dt
        f2z = (uz - (up[2] + (1./3.)*f1z))*dt*a1 + az*a2c*dt - GRAV*a0c*dt
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
        us[0] = ux
        us[1] = uy
        up[0] += (ux - up[0])*dt*a1 + ax*a2c*dt
        up[1] += (uy - up[1])*dt*a1 + ay*a2c*dt
        if pset.dim == 3:
            us[2] = uz
            up[2] += (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        if pset.additional_force:
            up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
            up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
            if pset.dim == 3:
                up[2] += (a3c/vp)*pset.addforce[i, 2]*dt
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # stage 3
    # -------
    # interpolate mean fields at 2/3 step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            tmp_tri, xtmp, ytmp)
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                tmp_tri, xtmp, ytmp, dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                tmp_tri, tmp_ll, xtmp, ytmp, ztmp, dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0]-(2./3.)*f2x)**2 + (uy-up[1]-(2./3.)*f2y)**2)
    else:
        unorm = sqrt((ux-up[0]-(2./3.)*f2x)**2 + (uy-up[1]-(2./3.)*f2y)**2 + (uz-up[2]-(2./3.)*f2z)**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update state (pseudo-step)
    k3x = (up[0] + (2./3.)*f2x)*dt
    k3y = (up[1] + (2./3.)*f2y)*dt
    f3x = (ux - (up[0] + (2./3.)*f2x))*dt*a1 + ax*a2c*dt
    f3y = (uy - (up[1] + (2./3.)*f2y))*dt*a1 + ay*a2c*dt

    # update particle state vector
    us[0] = ux
    us[1] = uy
    up[0] += (0.25*f1x + 0.75*f3x)
    up[1] += (0.25*f1y + 0.75*f3y)
    xp[0] += (0.25*k1x + 0.75*k3x)
    xp[1] += (0.25*k1y + 0.75*k3y)

    if pset.dim == 3:
        k3z = (up[2] + (2./3.)*f2z)*dt
        f3z = (uz - (up[2] + (2./3.)*f2z))*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        us[2] = uz
        up[2] += (0.25*f1z + 0.75*f3z)
        xp[2] += (0.25*k1z + 0.75*k3z)

    if pset.additional_force:
        up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
        up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
        if pset.dim == 3:
            up[2] += (a3c/vp)*pset.addforce[i, 2]*dt

    return -1

cdef int rk4_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i):
    """ Fourth order Runge-Kutta scheme for the LSM2 advection """
    cdef:
        double ux, uy, uz
        double ax = 0., ay = 0., az = 0.
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uz_corr = 0., uc_corr = 1.0
        # LSM2 parameters
        bint admass = pset.added_mass_force
        double a0c = pset.a0c
        double a1c = pset.a1c
        double a2c = pset.a2c
        double a3c = pset.a3c
        double dp = pset.diameter[i]
        double vp = pset.volume[i]
        double unorm, Cd, a1
        # RK4 parameters
        double k1x = 0., k1y = 0., k1z = 0.
        double k2x = 0., k2y = 0., k2z = 0.
        double k3x = 0., k3y = 0., k3z = 0.
        double k4x = 0., k4y = 0., k4z = 0.
        double f1x = 0., f1y = 0., f1z = 0.
        double f2x = 0., f2y = 0., f2z = 0.
        double f3x = 0., f3y = 0., f3z = 0.
        double f4x = 0., f4y = 0., f4z = 0.
        double xtmp, ytmp, ztmp = 0.
        int tmp_tri, bnd_jj = -1
        int tmp_ll = 0

    # pseudo step 1
    # ------
    # interpolate mean fields at particle position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                pset.tri[i], xp[0], xp[1], dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                pset.tri[i], pset.lowerlayer[i], 
                xp[0], xp[1], xp[2], dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2)
    else:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2 + (uz-up[2])**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update state (pseudo-step)
    k1x = up[0]*dt
    k1y = up[1]*dt
    f1x = (ux - up[0])*dt*a1 + ax*a2c*dt
    f1y = (uy - up[1])*dt*a1 + ay*a2c*dt
    xtmp = xp[0] + 0.5*k1x
    ytmp = xp[1] + 0.5*k1y
    
    if pset.dim == 3:
        k1z = up[2]*dt
        f1z = (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
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
        us[0] = ux
        us[1] = uy
        up[0] += (ux - up[0])*dt*a1 + ax*a2c*dt
        up[1] += (uy - up[1])*dt*a1 + ay*a2c*dt
        if pset.dim == 3:
            us[2] = uz
            up[2] += (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        if pset.additional_force:
            up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
            up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
            if pset.dim == 3:
                up[2] += (a3c/vp)*pset.addforce[i, 2]*dt
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # pseudo step 2
    # ------
    # interpolate mean fields at half-step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            tmp_tri, xtmp, ytmp)
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                tmp_tri, xtmp, ytmp, dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                tmp_tri, tmp_ll, xtmp, ytmp, ztmp, dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0]-0.5*f1x)**2 + (uy-up[1]-0.5*f1y)**2)
    else:
        unorm = sqrt((ux-up[0]-0.5*f1x)**2 + (uy-up[1]-0.5*f1y)**2 + (uz-up[2]-0.5*f1z)**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update state (pseudo-step)
    k2x = (up[0] + 0.5*f1x)*dt
    k2y = (up[1] + 0.5*f1y)*dt
    f2x = (ux - (up[0] + 0.5*f1x))*dt*a1 + ax*a2c*dt
    f2y = (uy - (up[1] + 0.5*f1y))*dt*a1 + ay*a2c*dt
    xtmp = xp[0] + 0.5*k2x
    ytmp = xp[1] + 0.5*k2y

    if pset.dim == 3:
        k2z = (up[2] + 0.5*f1z)*dt
        f2z = (uz - (up[2] + 0.5*f1z))*dt*a1 + az*a2c*dt - GRAV*a0c*dt
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
        us[0] = ux
        us[1] = uy
        up[0] += (ux - up[0])*dt*a1 + ax*a2c*dt
        up[1] += (uy - up[1])*dt*a1 + ay*a2c*dt
        if pset.dim == 3:
            us[2] = uz
            up[2] += (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        if pset.additional_force:
            up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
            up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
            if pset.dim == 3:
                up[2] += (a3c/vp)*pset.addforce[i, 2]*dt
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # pseudo step 3
    # ------
    # interpolate mean fields at half-step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            tmp_tri, xtmp, ytmp)
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                tmp_tri, xtmp, ytmp, dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                tmp_tri, tmp_ll, xtmp, ytmp, ztmp, dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0]-0.5*f2x)**2 + (uy-up[1]-0.5*f2y)**2)
    else:
        unorm = sqrt((ux-up[0]-0.5*f2x)**2 + (uy-up[1]-0.5*f2y)**2 + (uz-up[2]-0.5*f2z)**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update state (pseudo-step)
    k3x = (up[0] + 0.5*f2x)*dt
    k3y = (up[1] + 0.5*f2y)*dt
    f3x = (ux - (up[0] + 0.5*f2x))*dt*a1 + ax*a2c*dt
    f3y = (uy - (up[1] + 0.5*f2y))*dt*a1 + ay*a2c*dt
    xtmp = xp[0] + k3x
    ytmp = xp[1] + k3y

    if pset.dim == 3:
        k3z = (up[2] + 0.5*f2z)*dt
        f3z = (uz - (up[2] + 0.5*f2z))*dt*a1 + az*a2c*dt - GRAV*a0c*dt
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
        us[0] = ux
        us[1] = uy
        up[0] += (ux - up[0])*dt*a1 + ax*a2c*dt
        up[1] += (uy - up[1])*dt*a1 + ay*a2c*dt
        if pset.dim == 3:
            us[2] = uz
            up[2] += (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        if pset.additional_force:
            up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
            up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
            if pset.dim == 3:
                up[2] += (a3c/vp)*pset.addforce[i, 2]*dt
        xp[0] += up[0]*dt
        xp[1] += up[1]*dt
        if pset.dim == 3:
            xp[2] += up[2]*dt
        return bnd_jj

    # pseudo step 4
    # ------
    # interpolate mean fields at full-step position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            tmp_tri, xtmp, ytmp)
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                tmp_tri, xtmp, ytmp, ztmp, dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            tmp_tri, tmp_ll, 0, xtmp, ytmp, ztmp)

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                tmp_tri, xtmp, ytmp, dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                tmp_tri, tmp_ll, xtmp, ytmp, ztmp, dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0]-f3x)**2 + (uy-up[1]-f3y)**2)
    else:
        unorm = sqrt((ux-up[0]-f3x)**2 + (uy-up[1]-f3y)**2 + (uz-up[2]-f3z)**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # update state (pseudo-step)
    k4x = (up[0] + f3x)*dt
    k4y = (up[1] + f3y)*dt
    f4x = (ux - (up[0] + f3x))*dt*a1 + ax*a2c*dt
    f4y = (uy - (up[1] + f3y))*dt*a1 + ay*a2c*dt

    # update particle state vector
    us[0] = ux
    us[1] = uy
    up[0] += (1./6.)*(f1x + 2.*f2x + 2.*f3x + f4x)
    up[1] += (1./6.)*(f1y + 2.*f2y + 2.*f3y + f4y)
    xp[0] += (1./6.)*(k1x + 2.*k2x + 2.*k3x + k4x)
    xp[1] += (1./6.)*(k1y + 2.*k2y + 2.*k3y + k4y)

    if pset.dim == 3:
        k4z = (up[2] + f3z)*dt
        f4z = (uz - (up[2] + f3z))*dt*a1 + az*a2c*dt - GRAV*a0c*dt
        us[2] = uz
        up[2] += (1./6.)*(f1z + 2.*f2z + 2.*f3z + f4z)
        xp[2] += (1./6.)*(k1z + 2.*k2z + 2.*k3z + k4z)

    if pset.additional_force:
        up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
        up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
        if pset.dim == 3:
            up[2] += (a3c/vp)*pset.addforce[i, 2]*dt

    return -1

cdef void diffusion_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i, int diffmod, 
        double nux, double nuz, double sigc,
        int rng_method, int tri0, int ll0):
    """ 
    Euler-Maruyama scheme for the LSM2 diffusion on position

    Note: only diffusion part is computed here, including the Ito drift
    dK/dx_i. K and its gradient are evaluated at the start-of-step position
    (last_position) in the start-of-step element (tri0, ll0).
    """
    cdef:
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double xix = 0., xiy = 0., xiz = 0.
        double diffx = 0., diffy = 0., diffz = 0.
        double driftx = 0., drifty = 0., driftz = 0.
        double sqrtdt = sqrt(dt)
        double Kp = 0., dKdx = 0., dKdy = 0., dKdz = 0.
        double z0 = 0.

    # diffusion
    if diffmod == 2 or diffmod == 3:
        # k-eps diffusion model
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
            if diffmod == 2:
                # constant vertical diffusivity: no vertical drift
                diffz = sqrt(2.*nuz)*sqrtdt
            else:
                diffz = diffx
                driftz = dKdz*dt
    else:
        # constant diffusivity
        diffx = sqrt(2.*nux)*sqrtdt
        diffy = diffx
        if pset.dim == 3:
            diffz = sqrt(2.*nuz)*sqrtdt

    # In particular, option 1 with zero diffusion stays deterministic and
    # leaves the random stream untouched, just like direct options 2 and 3.
    if diffx != 0.:
        xix, xiy = generate_random_pair(rng_method)
    if pset.dim == 3 and diffz != 0.:
        xiz = generate_random_single(rng_method)

    # update particle state vector
    xp[0] += driftx + diffx*xix
    xp[1] += drifty + diffy*xiy
    if pset.dim == 3:
        xp[2] += driftz + diffz*xiz

cdef void euler_maruyama_lsm2_u(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i,  int diffmod,
        double nux, double nuz, 
        int nuopt, double sigc,
        int rng_method):
    """ 
    Euler-Maruyama scheme for the LSM2 advection-diffusion 
    
    Note: advection and diffusion are computed here.
    """
    cdef:
        double ux, uy, uz
        double ax = 0., ay = 0., az = 0.
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uz_corr = 0.,uc_corr = 1.0
        # LSM2 parameters
        bint admass = pset.added_mass_force
        double a0c = pset.a0c
        double a1c = pset.a1c
        double a2c = pset.a2c
        double a3c = pset.a3c
        double dp = pset.diameter[i]
        double vp = pset.volume[i]
        double unorm, Cd, a1
        double xix = 0., xiy = 0., xiz = 0.
        double diffx = 0., diffy = 0., diffz = 0.
        double sqrtdt = sqrt(dt)
        double tml = 0., turbeng = 0., epsilon = 0., nutmp = 0.

    # interpolate mean fields at particle position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    # compute dUs/dt for added mass and pressure forces
    if admass:
        if fset.dim == 2:
            ax, ay = fset._interpolate_acceleration_2d(\
                pset.tri[i], xp[0], xp[1], dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                pset.tri[i], pset.lowerlayer[i], 
                xp[0], xp[1], xp[2], dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2)
    else:
        unorm = sqrt((ux-up[0])**2 + (uy-up[1])**2 + (uz-up[2])**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm

    # diffusion terms
    if diffmod != 0:
        xix, xiy = generate_random_pair(rng_method)
        if pset.dim == 3:
            xiz = generate_random_single(rng_method)

    if diffmod == 2 or diffmod == 3:
        # k-eps diffusion model: Bi = A1*T_L*sqrt(C0*esp)
        if fset.dim == 2:
            turbeng = fset._interpolate_field_2d(fset.fid_turbeng,\
                pset.tri[i], -1, xp[0], xp[1])
            epsilon = fset._interpolate_field_2d(fset.fid_dissip,\
                pset.tri[i], -1, xp[0], xp[1])
        else:
            turbeng = fset._interpolate_field_3d(fset.fid_turbeng,\
                pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])
            epsilon = fset._interpolate_field_3d(fset.fid_dissip,\
                pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])
        
        epsilon = max(epsilon, EPSILON)
        # eddy time scale: T_L
        tml = (turbeng/epsilon)/(0.5 + (3./4.)*C0KOLM)
        diffx = a1*tml*sqrt(C0KOLM*epsilon)*sqrtdt
        diffy = diffx
        
        if pset.dim == 3:
            if diffmod == 2:
                # Option 2 specifies Bz; option 3 specifies Kz.
                if nuopt == 2:
                    diffz = nuz*sqrtdt
                else:
                    diffz = a1*sqrt(2.*nuz)*sqrtdt
            else:
                diffz = diffx
    else:
        # Constant diffusivity
        if nuopt == 2:
            # nui = Bi
            diffx = nux*sqrtdt
            diffy = diffx
            if pset.dim == 3:
                diffz = nuz*sqrtdt
        elif nuopt == 3:
            # nui = Ki and Bi = A1*sqrt(2*Ki)
            diffx = a1*sqrt(2.*nux)*sqrtdt
            diffy = diffx
            if pset.dim == 3:
                diffz = a1*sqrt(2.*nuz)*sqrtdt
        else:
            # not diffusion on U
            diffx = 0.
            diffy = 0.
            if pset.dim == 3:
                diffz = 0.

    # update particle state vector
    us[0] = ux
    us[1] = uy
    up[0] += (ux - up[0])*dt*a1 + ax*a2c*dt + diffx*xix
    up[1] += (uy - up[1])*dt*a1 + ay*a2c*dt + diffy*xiy
    xp[0] += up[0]*dt
    xp[1] += up[1]*dt

    if pset.dim == 3:
        us[2] = uz
        up[2] += (uz - up[2])*dt*a1 + az*a2c*dt - GRAV*a0c*dt + diffz*xiz
        xp[2] += up[2]*dt

    if pset.additional_force:
        up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
        up[1] += (a3c/vp)*pset.addforce[i, 1]*dt
        if pset.dim == 3:
            up[2] += (a3c/vp)*pset.addforce[i, 2]*dt

cdef void di_ode_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i):
    """ Use the shared exact drift; diffmod=0 never draws random numbers. """
    di_lsm2(simutime, fset, pset, i, 0, 0., 0., 2, 1., 1)

cpdef (double, double, double, double, double, double, double) lsm2_direct_coefficients(
        double a1, double dt):
    """
    Return frozen-step coefficients for one LSM-2 spatial component.

    The direct integrator freezes the scalar SDE over one time step:

    du = [a1*(U - u) + F]*dt + B*dW and dx = u*dt,

    where u is particle velocity, x is particle position, U is
    the interpolated mean-fluid velocity, a1 = 1/taup is the drag
    relaxation rate, F = A0*g + A2*af + A3*Fa is the constant
    acceleration/force contribution, B is the velocity-noise amplitude,
    and dW is a Wiener increment.

    Returns
    -------
    beta : double
        exp(-a1*dt), the velocity transition coefficient.
    D1 : double
        dt*phi1(-a1*dt) = (1-beta)/a1. This multiplies the initial velocity
        in the position mean and the force in the velocity mean.
    D2 : double
        dt - D1 = a1*dt**2*phi2(-a1*dt). This multiplies U in the
        position mean.
    force_x : double
        dt**2*phi2(-a1*dt) = D2/a1. This multiplies F in the position
        mean.
    Q_uu, Q_xu, Q_xx : double
        Entries of the conditional covariance for unit noise amplitude
        B=1: velocity variance, position-velocity covariance, and
        position variance. The physical covariance is B**2 * Q.

    Here phi1(z) = (exp(z)-1)/z and phi2(z) = (exp(z)-1-z)/z**2, 
    using their continuous limits at zero.
    Thus no reciprocal damping is needed when a1=0: the coefficients
    recover the ballistic integrated-Brownian-motion limit exactly.
    """
    cdef:
        double r = a1*dt
        double beta = exp(-r)
        double phi1, phi2, term, power2
        double D1, D2, force_x, q_uu, q_xu, q_xx
        int n

    if (not isfinite(a1) or not isfinite(dt) or a1 < 0. or dt < 0.
            or not isfinite(r)):
        raise ValueError("LSM2 requires finite a1 >= 0, dt >= 0 and a1*dt")

    if r < 0.1:
        # Terms n=0..12: truncation is below double precision at r=0.1.
        phi1 = 1.
        phi2 = 0.5
        term = 1.
        for n in range(1, 13):
            term *= -r/(n + 1.)
            phi1 += term
            phi2 += term/(n + 2.)
        D1 = dt*phi1
        D2 = dt*r*phi2
        force_x = dt*dt*phi2
    else:
        D1 = -expm1(-r)/a1
        D2 = dt - D1
        force_x = D2/a1

    if r == 0.:
        q_uu = dt
    else:
        q_uu = dt*(-expm1(-2.*r)/(2.*r))
    q_xu = 0.5*D1*D1

    if r < 1.:
        # Q_xx/dt**3 = sum (-r)**n * (2**(n+2)-2)/(n+3)!.
        # 24 terms retain double precision all the way to r=1, avoiding
        # the cancellation of dt - 2*D1 + Q_uu near zero.
        term = 1./6.
        power2 = 4.
        q_xx = 1./3.
        for n in range(1, 24):
            term *= -r/(n + 3.)
            power2 *= 2.
            q_xx += (power2 - 2.)*term
        q_xx *= dt*dt*dt
    else:
        q_xx = ((dt - 2.*D1 + q_uu)/a1)/a1

    return beta, D1, D2, force_x, q_uu, q_xu, q_xx

cdef void di_lsm2(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i,  int diffmod,
        double nux, double nuz, 
        int nuopt, double sigc,
        int rng_method):
    """ 
    Full direct integration for the LSM2 advection-diffusion 
    ~~> Full DI (exact integrator) for up, xp

    Analytical solution of the linear SDE system (per component):
        dUp = (Ctau - Up)/taup*dt + B*dW
        dXp = Up*dt
    with Ctau = U + Fp*taup, following the exact Ornstein-Uhlenbeck-process 
    update (Gillespie, 1996):
        E[Up]  = Ctau + (Up0 - Ctau)*beta
        E[Xp]  = Xp0 + Ctau*dt + (Up0 - Ctau)*taup*(1 - beta)
        Var[Gp]     = 0.5*B^2*taup*(1 - beta^2)
        Cov[Gp, Gx] = 0.5*B^2*taup^2*(1 - beta)^2
        Var[Gx]     = B^2*taup^2*(dt - 2*taup*(1 - beta)
                                  + 0.5*taup*(1 - beta^2))
    with beta = exp(-dt/taup). The correlated increments (Gp, Gx) are
    sampled with a 2x2 Cholesky decomposition, as in difull_lsm3.
    """
    cdef:
        double ux, uy, uz
        double ax = 0., ay = 0., az = 0.
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double[3] xp0 
        double[3] up0 
        double uz_corr = 0., uc_corr = 1.0
        bint admass = pset.added_mass_force
        double a0c = pset.a0c
        double a1c = pset.a1c
        double a2c = pset.a2c
        double a3c = pset.a3c
        double dp = pset.diameter[i]
        double vp = pset.volume[i]
        double unorm, Cd, a1
        double tml = 0., turbeng = 0., epsilon = 0.
        double beta, um_beta
        double D1, D2, force_x, q_uu, q_xu, q_xx
        double u_k, Fp
        double up_mean, xp_mean
        double[3] b2
        double var_gp, var_gx, cov_gp_gx
        double L_pp, L_xp, L_xx
        double Gp, Gx
        double xix = 0., xiy = 0.
        int k

    # --- Step 1: Parameter and Variable Preparation ---

    # Store initial state
    for k in range(pset.dim):
        xp0[k] = xp[k]
        up0[k] = up[k]

    # Interpolate Eulerian fields at particle position
    if fset.dim == 2:
        # interpolate velocity
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp0[0], xp0[1])
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp0[0], xp0[1], xp0[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
        # interpolate acceleration
        if admass:
            ax, ay = fset._interpolate_acceleration_2d(\
                pset.tri[i], xp0[0], xp0[1], dt, ux, uy)
            az = 0.
        else:
            ax, ay, az = 0., 0., 0.
    else:
        # interpolate velocity
        ux, uy, uz = fset._interpolate_velocity_3d(\
            pset.tri[i], pset.lowerlayer[i], 0, xp0[0], xp0[1], xp0[2])
        # interpolate acceleration
        if admass:
            ax, ay, az = fset._interpolate_acceleration_3d(\
                pset.tri[i], pset.lowerlayer[i], 
                xp0[0], xp0[1], xp0[2], dt, ux, uy, uz)
        else:
            ax, ay, az = 0., 0., 0.

    # Calculate relative velocity norm
    if pset.dim == 2:
        unorm = sqrt((ux - up0[0])**2 + (uy - up0[1])**2)
    else:
        unorm = sqrt((ux - up0[0])**2 + (uy - up0[1])**2 + (uz - up0[2])**2)
    unorm = fmax(EPSILON_LSM2, unorm)

    # Keep the exact damping rate, including zero (the ballistic limit).
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*Cd*unorm
    beta, D1, D2, force_x, q_uu, q_xu, q_xx = lsm2_direct_coefficients(a1, dt)
    um_beta = -expm1(-a1*dt)

    # --- Step 2: Diffusion Coefficients ---

    # Store B_k^2, not B_k^2/a1: finite even for undamped Brownian velocity.
    if diffmod == 0:
        b2[0] = 0.
        b2[1] = 0.
        b2[2] = 0.
    elif diffmod == 2 or diffmod == 3:
        # k-eps diffusion model: Bi = A1*T_L*sqrt(C0*esp)
        if fset.dim == 2:
            turbeng = fset._interpolate_field_2d(fset.fid_turbeng,\
                pset.tri[i], -1, xp0[0], xp0[1])
            epsilon = fset._interpolate_field_2d(fset.fid_dissip,\
                pset.tri[i], -1, xp0[0], xp0[1])
        else:
            turbeng = fset._interpolate_field_3d(fset.fid_turbeng,\
                pset.tri[i], pset.lowerlayer[i], 0, xp0[0], xp0[1], xp0[2])
            epsilon = fset._interpolate_field_3d(fset.fid_dissip,\
                pset.tri[i], pset.lowerlayer[i], 0, xp0[0], xp0[1], xp0[2])
        
        epsilon = fmax(epsilon, EPSILON)
        # eddy time scale: T_L
        tml = (turbeng/epsilon)/(0.5 + (3./4.)*C0KOLM)
        b2[0] = a1*a1*C0KOLM*epsilon*tml**2
        b2[1] = b2[0]
        # vertical component
        if diffmod == 2:
            if nuopt == 2:
                b2[2] = nuz*nuz
            else:
                b2[2] = 2.*nuz*a1*a1
        else:
            b2[2] = b2[0]
    else:
        # Constant diffusivity
        if nuopt == 2:
            # nui = Bi
            b2[0] = nux*nux
            b2[1] = b2[0]
            b2[2] = nuz*nuz
        elif nuopt == 3:
            # nui = Ki and Bi = A1*sqrt(2*Ki)
            b2[0] = 2.*nux*a1*a1
            b2[1] = b2[0]
            b2[2] = 2.*nuz*a1*a1
        else:
            # no diffusion on U
            b2[0] = 0.
            b2[1] = 0.
            b2[2] = 0.

    # --- Step 3: Update State for Each Dimension ---

    for k in range(pset.dim):
        # Get dimension-specific mean fluid velocity
        if k == 0: 
            u_k = ux
        elif k == 1: 
            u_k = uy
        else: 
            u_k = uz

        # Calculate external force Fp for this dimension
        if k == 0:
            Fp = a2c*ax
        elif k == 1:
            Fp = a2c*ay
        else:
            Fp = a2c*az - GRAV*a0c

        if pset.additional_force:
            Fp += (a3c/vp)*pset.addforce[i, k]

        # Calculate mean update (drift part)
        # up_mean = up0*beta + Ctau*(1 - beta)
        # xp_mean = xp0 + up0*taup*(1 - beta) + Ctau*(dt - taup*(1 - beta))
        # force_x = taup*D2, evaluated without cancellation or division by
        # zero; at a1=0 this gives xp0 + up0*dt + 0.5*Fp*dt**2.
        up_mean = up0[k]*beta + u_k*um_beta + Fp*D1
        xp_mean = xp0[k] + up0[k]*D1 + u_k*D2 + Fp*force_x

        # Deterministic steps must not consume the shared random stream.
        if b2[k] == 0. or dt == 0.:
            us[k] = u_k
            up[k] = up_mean
            xp[k] = xp_mean
            continue

        # Covariances of the stochastic increments (Gp, Gx)
        var_gp = b2[k]*q_uu
        cov_gp_gx = b2[k]*q_xu
        var_gx = b2[k]*q_xx

        # Cholesky decomposition of the 2x2 covariance matrix
        L_pp = sqrt(fmax(0., var_gp))
        L_xp = cov_gp_gx/L_pp if L_pp > 0. else 0.
        L_xx = sqrt(fmax(0., var_gx - L_xp**2))

        # Generate correlated stochastic increments [Gp, Gx]
        xix, xiy = generate_random_pair(rng_method)
        Gp = L_pp*xix
        Gx = L_xp*xix + L_xx*xiy

        # Final state update
        us[k] = u_k
        up[k] = up_mean + Gp
        xp[k] = xp_mean + Gx
