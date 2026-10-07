# -*- coding: utf-8 -*-
"""
Analytical advection
"""
import numpy as np

#==============================================================================
# LSM-1: LAGRANGIAN STOCHASTIC MODEL 1
#==============================================================================
def LSM1_analytic_advection_Euler(analytic_velocity_field, xp0, tf=1., dt=0.1):
    """
    Analytical advection of partilces - Euler scheme
    LSM1 : solves dxp(t) = uf(t, xp(t))*dt
    
    @param analytic_velocity_field (fun): analytic velocity field function
    @param xp0 (np.array((npart, 2)): array of intial position of particles
    @param tf (float): final time
    @param dt (float): time step
    """
    # time discretization
    times = np.arange(0, tf, dt)
    nt = len(times)
    # initialization
    npart = np.shape(xp0)[0]
    xp = np.empty((nt, npart, 2), dtype='d')
    xp[0, :, :] = xp0
    # time loop
    for k in range(nt-1):
        # loop on particles
        for i in range(npart):
            # pseudo step 1
            ux, uy = analytic_velocity_field(xp[k,i,0], xp[k,i,1])
            # update
            xp[k+1,i,0] = xp[k,i,0] + ux*dt
            xp[k+1,i,1] = xp[k,i,1] + uy*dt
    return xp

def LSM1_analytic_advection_RK2(analytic_velocity_field, xp0, tf=1., dt=0.1):
    """ 
    Analytical advection of partilces - RK2 scheme
    LSM1 : solves dxp(t) = uf(t, xp(t))*dt
    
    @param analytic_velocity_field (fun): analytic velocity field function
    @param xp0 (np.array((npart, 2)): array of intial position of particles
    @param tf (float): final time
    @param dt (float): time step
    """
    # time discretization
    times = np.arange(0, tf, dt)
    nt = len(times)
    # initialization
    npart = np.shape(xp0)[0]
    xp = np.empty((nt, npart, 2), dtype='d')
    xp[0, :, :] = xp0
    # time loop
    for k in range(nt-1):
        # loop on particles
        for i in range(npart):
            # pseudo step 1
            ux, uy = analytic_velocity_field(xp[k,i,0], xp[k,i,1])
            k1x = ux*dt
            k1y = uy*dt
            # pseudo step 2
            ux, uy = analytic_velocity_field(xp[k,i,0] + 0.5*k1x, xp[k,i,1] + 0.5*k1y)
            k2x = ux*dt
            k2y = uy*dt
            # update
            xp[k+1,i,0] = xp[k,i,0] + k2x
            xp[k+1,i,1] = xp[k,i,1] + k2y
    return xp

def LSM1_analytic_advection_RK4(analytic_velocity_field, xp0, tf=1., dt=0.1):
    """ 
    Analytical advection of partilces - RK4 scheme
    LSM1 : solves dxp(t) = uf(t, xp(t))*dt
    
    @param analytic_velocity_field (fun): analytic velocity field function
    @param xp0 (np.array((npart, 2)): array of intial position of particles
    @param tf (float): final time
    @param dt (float): time step
    """
    # time discretization
    times = np.arange(0, tf, dt)
    nt = len(times)
    # initialization
    npart = np.shape(xp0)[0]
    xp = np.empty((nt, npart, 2), dtype='d')
    xp[0, :, :] = xp0
    # time loop
    for k in range(nt-1):
        # loop on particles
        for i in range(npart):
            # pseudo step 1
            ux, uy = analytic_velocity_field(xp[k,i,0], xp[k,i,1])
            k1x = ux*dt
            k1y = uy*dt
            # pseudo step 2
            ux, uy = analytic_velocity_field(xp[k,i,0]+0.5*k1x, xp[k,i,1]+0.5*k1y)
            k2x = ux*dt
            k2y = uy*dt
            # pseudo step 3
            ux, uy = analytic_velocity_field(xp[k,i,0]+0.5*k2x, xp[k,i,1]+0.5*k2y)
            k3x = ux*dt
            k3y = uy*dt
            # pseudo step 4
            ux, uy = analytic_velocity_field(xp[k,i,0]+k3x, xp[k,i,1]+k3y)
            k4x = ux*dt
            k4y = uy*dt
            # update
            xp[k+1,i,0] = xp[k,i,0] + (1./6.)*(k1x + 2.*k2x + 2.*k3x + k4x)
            xp[k+1,i,1] = xp[k,i,1] + (1./6.)*(k1y + 2.*k2y + 2.*k3y + k4y)
    return xp
    
#==============================================================================
# LSM-1: LAGRANGIAN STOCHASTIC MODEL 2
#==============================================================================
def drag_coefficient(Re, law=0):
    """ 
    Computes the drag coefficient of a particle.
    
    @param Re (float): Reynolds number (Re = U*diameter/visc)
    @return Cd (float): drag coefficient
    """
    epsilon = 1.e-16
    # Schiller et Nauman, (1935) / Code Saturne
    if law==0:
        if Re <= 1000.:
            Cd = 24.*(1. + 0.15*Re**0.687)/max(epsilon, Re)
        else:
            Cd = 0.44
    # Almedeij (2008) / Joly (2015)
    elif law==1:
        phi1 = (24./Re)**10. + (21.*Re**-0.67)**10. + (4.*Re**-0.33)**10. + 0.4**10
        phi2 = 1./((0.148*Re**0.11)**-10. + (0.5)**-10.)
        phi3 = (1.57*(Re**-1.625)*1.e8 )**10.
        phi4 = 1./((6e-17*Re**2.63)**-10.  + 0.2**-10.)
        Cd = (1./( (phi1 + phi2)**-1. + phi3**-1. ) + phi4)**(1/10.)
    else:
        Cd = 0.44
    return Cd

def compute_LSM2_A1(uf, up, rhop=1000., rhof=1000., dp=0.01, nu=1.e-6, model=1, Ca=0.5, verbose=False):
    """
    Computes coefficient A1 of LSM2 model (A1 = 1./taup) 
    (taup = characteristic time scale of drag relaxation model)

    @param uf (array(2)): fluid velocity
    @param up (array(2)): particle velocity
    @param rhop (float): particle density
    @param rhof (float): fluid density
    @param dp (float): particle diameter
    @param nu (float): fluid kinematic viscosity
    @param model (int): 1: only drag force, 2: drag + momentum forces (default: 1)
    @param Ca (float): coefficient of added mass (for model: 2)

    @return A1 (float): inverse of characteristic time scale (1./taup)
    """
    # norm of relative velocity
    Unorm = np.sqrt((uf[0]-up[0])**2 + (uf[1]-up[1])**2)
    # particlar Reynolds number
    Re = Unorm*dp/nu
    # drag coefficient
    Cd = drag_coefficient(Re, law=0)
    # Sp/Vp for spherical particles
    surface_volume_ratio = 3./(2.*dp)
    if model==1:
        A1 = 0.5*(rhof/rhop)*surface_volume_ratio*Unorm*Cd
    else:
        aux = 0.5*(rhof/(rhop + Ca*rhof))
        A1 = aux*surface_volume_ratio*Unorm*Cd
    if verbose:
        print("Urel =", Unorm)
        print("Re =", Re)
        print("Cd =", Cd)
        print("A1 =", A1)
    return A1
    
def compute_LSM2_A2(rhop=1000., rhof=1000., Ca=0.5):
    """
    Computes coefficient A2 of LSM2 model
    
    @param rhop (float): particle density
    @param rhof (float): fluid density
    @param Ca (float): coefficient of added mass
    """
    A2 = (rhof*(1 + Ca))/(rhop + Ca*rhof)
    return A2

def LSM2_analytic_advection_DI(analytic_velocity_field, xp0, up0,
        tf=1., dt=0.1, model=1, analytic_acceleration_field=None,
        rhop=1000., rhof=1000., dp=0.01, nu=1.e-6):
    """
    Analytical advection of partilces - Direct integration method
    LSM2 : solves dxp(t) = up(t)*dt
                  dup(t) = A1*[uf(t, xp(t))-up(t)]*dt + A2*[duf(t, xp(t))/dt]*dt

    @param analytic_velocity_field (function): analytic velocity field function
    @param xp0 (np.array((npart, 2)): array of intial position of particles
    @param up0 (np.array((npart, 2)): array of intial velocity of particles
    @param tf (float): final time
    @param dt (float): time step
    @param model (float): 1: only drag force, 2: drag + momentum forces (default: 1)
    @param analytic_acceleration_field (function): analytic acceleration field function
        for model:2
    @param rhop (float): particle density
    @param rhof (float): fluid density
    @param dp (float): particle diameter
    @param nu (float): fluid kinematic viscosity
    """
    eps = 1.e-16
    
    # time discretization
    times = np.arange(0, tf, dt)
    nt = len(times)
    
    # initialization
    npart = np.shape(xp0)[0]
    xp = np.empty((nt, npart, 2), dtype='d')
    up = np.empty((nt, npart, 2), dtype='d')
    xp[0, :, :] = xp0
    up[0, :, :] = up0
    
    # time loop
    for k in range(nt-1):
        # loop on particles
        for i in range(npart):

            # compute model coefficients
            if model==1:
                # Model with drag force only:
                ux, uy = analytic_velocity_field(xp[k,i,0], xp[k,i,1])
                ax, ay = 0., 0.
                A1 = compute_LSM2_A1(np.array([ux, uy]), up[k,i,:],
                    rhop, rhof, dp, nu, model=model)
                A2 = 0.
            elif model==2:
                # Model with drag force + momentum:
                ux, uy = analytic_velocity_field(xp[k,i,0], xp[k,i,1])
                ax, ay = analytic_acceleration_field(xp[k,i,0], xp[k,i,1])
                A1 = compute_LSM2_A1(np.array([ux, uy]), up[k,i,:],
                    rhop, rhof, dp, nu, model=model)
                A2 = compute_LSM2_A2(rhop, rhof)

            if A1<=eps: # taup -> infty / A1 -> 0
                taup = 1./max(eps, A1)
                aux0 = 1.
                aux1 = 0.
                btau0 = ux
                btau1 = uy

            elif A1>=1./eps: # taup-> 0 / A1 -> infty
                taup = 0.
                aux0 = 0.
                aux1 = 1.
                btau0 = (ux + ax*A2*taup)
                btau1 = (uy + ay*A2*taup)
            
            else:
                taup = 1./A1
                aux0 = np.exp(-dt/taup)
                aux1 = 1. - aux0
                btau0 = (ux + ax*A2*taup)
                btau1 = (uy + ay*A2*taup)

            # update of up
            up[k+1,i,0] = up[k,i,0]*aux0 + btau0*aux1
            up[k+1,i,1] = up[k,i,1]*aux0 + btau1*aux1 
            
            # update of xp
            #xp[k+1,i,0] = xp[k,i,0] + 0.5*(up[k+1,i,0] + up[k,i,0])*dt
            #xp[k+1,i,1] = xp[k,i,1] + 0.5*(up[k+1,i,1] + up[k,i,1])*dt
            xp[k+1,i,0] = xp[k,i,0] + up[k,i,0]*taup*aux1 + btau0*(dt - taup*aux1)
            xp[k+1,i,1] = xp[k,i,1] + up[k,i,1]*taup*aux1 + btau1*(dt - taup*aux1)

    return xp

def LSM2_analytic_advection_Euler(analytic_velocity_field, xp0, up0, 
        tf=1., dt=0.1, up_upwinding=False, 
        model=1, analytic_acceleration_field=None,
        rhop=1000., rhof=1000., dp=0.01, nu=1.e-6):
    """
    Analytical advection of partilces - Euler scheme
    LSM2 : solves dxp(t) = up(t)*dt
                  dup(t) = A1*[uf(t, xp(t))-up(t)]*dt

    @param analytic_velocity_field (function): analytic velocity field function
    @param xp0 (np.array((npart, 2)): array of intial position of particles
    @param up0 (np.array((npart, 2)): array of intial velocity of particles
    @param tf (float): final time
    @param dt (float): time step
    @param up_upwinding (bool): 0.5*(up(n+1)+up(n)) for xp update instead of up(n)
    @param model (float): 1: only drag force, 2: drag + momentum forces (default: 1)
    @param analytic_acceleration_field (function): analytic acceleration field function
        for model:2
    @param rhop (float): particle density
    @param rhof (float): fluid density
    @param dp (float): particle diameter
    @param nu (float): fluid kinematic viscosity
    """
    eps = 1.e-16
    
    # time discretization
    times = np.arange(0, tf, dt)
    nt = len(times)

    # initialization
    npart = np.shape(xp0)[0]
    xp = np.empty((nt, npart, 2), dtype='d')
    up = np.empty((nt, npart, 2), dtype='d')
    xp[0, :, :] = xp0
    up[0, :, :] = up0

    # time loop
    for k in range(nt-1):
        # loop on particles
        for i in range(npart):
        
            # pseudo step 1
            ux, uy = analytic_velocity_field(xp[k,i,0], xp[k,i,1])

            if model==1:
                # Model with drag force only:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:],
                    rhop, rhof, dp, nu, model=model)
                A2 = 0.
                ax, ay = 0., 0.

            elif model==2:
                # Model with drag force + momentum:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:],
                    rhop, rhof, dp, nu, model=model)
                A2 = compute_LSM2_A2(rhop, rhof)
                ax, ay = analytic_acceleration_field(xp[k,i,0], xp[k,i,1])
            
            dupx = (ux - up[k,i,0])*dt*A1 + ax*A2*dt
            dupy = (uy - up[k,i,1])*dt*A1 + ay*A2*dt
            
            # update
            up[k+1,i,0] = up[k,i,0] + dupx
            up[k+1,i,1] = up[k,i,1] + dupy
            if up_upwinding:
                xp[k+1,i,0] = xp[k,i,0] + 0.5*(up[k+1,i,0] + up[k,i,0])*dt
                xp[k+1,i,1] = xp[k,i,1] + 0.5*(up[k+1,i,1] + up[k,i,1])*dt                
            else:
                xp[k+1,i,0] = xp[k,i,0] + up[k,i,0]*dt
                xp[k+1,i,1] = xp[k,i,1] + up[k,i,1]*dt
    return xp

def LSM2_analytic_advection_RK2(analytic_velocity_field, xp0, up0, 
        tf=1., dt=0.1, taup=None,
        model=1, analytic_acceleration_field=None,
        rhop=1000., rhof=1000., dp=0.01, nu=1.e-6):
    """
    Analytical advection of partilces - RK2 scheme
    LSM2 : solves dxp(t) = up(t)*dt
                  dup(t) = A1*[uf(t, xp(t))-up(t)]*dt

    @param analytic_velocity_field (function): analytic velocity field function
    @param xp0 (np.array((npart, 2)): array of intial position of particles
    @param up0 (np.array((npart, 2)): array of intial velocity of particles
    @param tf (float): final time
    @param dt (float): time step
    @param model (float): 1: only drag force, 2: drag + momentum forces (default: 1)
    @param analytic_acceleration_field (function): analytic acceleration field function
        for model:2
    @param rhop (float): particle density
    @param rhof (float): fluid density
    @param dp (float): particle diameter
    @param nu (float): fluid kinematic viscosity
    """
    eps = 1.e-16
    
    # time discretization
    times = np.arange(0, tf, dt)
    nt = len(times)
    
    # initialization
    npart = np.shape(xp0)[0]
    xp = np.empty((nt, npart, 2), dtype='d')
    up = np.empty((nt, npart, 2), dtype='d')
    xp[0, :, :] = xp0
    up[0, :, :] = up0
    
    # time loop
    for k in range(nt-1):
        # loop on particles
        for i in range(npart):
            # pseudo step 1
            # ~~~~~~~~~~~~~
            ux, uy = analytic_velocity_field(xp[k,i,0], xp[k,i,1])

            if model==1:
                # Model with drag force only:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:],
                    rhop, rhof, dp, nu, model=model)
                A2 = 0.
                ax, ay = 0., 0.
            elif model==2:
                # Model with drag force + momentum:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:],
                    rhop, rhof, dp, nu, model=model)
                A2 = compute_LSM2_A2(rhop, rhof)
                ax, ay = analytic_acceleration_field(xp[k,i,0], xp[k,i,1])

            k1x = up[k,i,0]*dt
            k1y = up[k,i,1]*dt            
            f1x = (ux - up[k,i,0])*dt*A1 + ax*A2*dt
            f1y = (uy - up[k,i,1])*dt*A1 + ay*A2*dt
            
            # pseudo step 2
            # ~~~~~~~~~~~~~
            ux, uy = analytic_velocity_field(xp[k,i,0]+0.5*k1x, xp[k,i,1]+0.5*k1y)
            
            if model==1:
                # Model with drag force only:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:]+[0.5*f1x, 0.5*f1y],
                    rhop, rhof, dp, nu, model=model)
                A2 = 0.
                ax, ay = 0., 0.
            elif model==2:
                # Model with drag force + momentum:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:]+[0.5*f1x, 0.5*f1y],
                    rhop, rhof, dp, nu, model=model)
                A2 = compute_LSM2_A2(rhop, rhof)
                ax, ay = analytic_acceleration_field(xp[k,i,0]+0.5*k1x, xp[k,i,1]+0.5*k1y)

            k2x = (up[k,i,0] + 0.5*f1x)*dt
            k2y = (up[k,i,1] + 0.5*f1y)*dt
            f2x = (ux - (up[k,i,0] + 0.5*f1x))*dt*A1 + ax*A2*dt
            f2y = (uy - (up[k,i,1] + 0.5*f1y))*dt*A1 + ay*A2*dt
            
            # update
            # ~~~~~~
            up[k+1,i,0] = up[k,i,0] + f2x
            up[k+1,i,1] = up[k,i,1] + f2y
            xp[k+1,i,0] = xp[k,i,0] + k2x
            xp[k+1,i,1] = xp[k,i,1] + k2y
    return xp

def LSM2_analytic_advection_RK4(analytic_velocity_field, xp0, up0, 
        tf=1., dt=0.1, taup=None,
        model=1, analytic_acceleration_field=None,
        rhop=1000., rhof=1000., dp=0.01, nu=1.e-6):
    """
    Analytical advection of partilces - RK2 scheme
    LSM2 : solves dxp(t) = up(t)*dt
                  dup(t) = A1*[uf(t, xp(t))-up(t)]*dt

    @param analytic_velocity_field (function): analytic velocity field function
    @param xp0 (np.array((npart, 2)): array of intial position of particles
    @param up0 (np.array((npart, 2)): array of intial velocity of particles
    @param tf (float): final time
    @param dt (float): time step
    @param model (float): 1: only drag force, 2: drag + momentum forces (default: 1)
    @param analytic_acceleration_field (function): analytic acceleration field function
        for model:2
    @param rhop (float): particle density
    @param rhof (float): fluid density
    @param dp (float): particle diameter
    @param nu (float): fluid kinematic viscosity
    """
    eps = 1.e-16
    
    # time discretization
    times = np.arange(0, tf, dt)
    nt = len(times)
    
    # initialization
    npart = np.shape(xp0)[0]
    xp = np.empty((nt, npart, 2), dtype='d')
    up = np.empty((nt, npart, 2), dtype='d')
    xp[0, :, :] = xp0
    up[0, :, :] = up0
    
    # time loop
    for k in range(nt-1):
        # loop on particles
        for i in range(npart):
            # pseudo step 1
            # ~~~~~~~~~~~~~
            ux, uy = analytic_velocity_field(xp[k,i,0], xp[k,i,1])

            if model==1:
                # Model with drag force only:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:],
                    rhop, rhof, dp, nu, model=model)
                A2 = 0.
                ax, ay = 0., 0.
            elif model==2:
                # Model with drag force + momentum:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:],
                    rhop, rhof, dp, nu, model=model)
                A2 = compute_LSM2_A2(rhop, rhof)
                ax, ay = analytic_acceleration_field(xp[k,i,0], xp[k,i,1])

            k1x = up[k,i,0]*dt
            k1y = up[k,i,1]*dt            
            f1x = (ux - up[k,i,0])*dt*A1 + ax*A2*dt
            f1y = (uy - up[k,i,1])*dt*A1 + ay*A2*dt
            
            # pseudo step 2
            # ~~~~~~~~~~~~~
            ux, uy = analytic_velocity_field(xp[k,i,0]+0.5*k1x, xp[k,i,1]+0.5*k1y)

            if model==1:
                # Model with drag force only:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:]+[0.5*f1x, 0.5*f1y],
                    rhop, rhof, dp, nu, model=model)
                A2 = 0.
                ax, ay = 0., 0.
            elif model==2:
                # Model with drag force + momentum:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:]+[0.5*f1x, 0.5*f1y],
                    rhop, rhof, dp, nu, model=model)
                A2 = compute_LSM2_A2(rhop, rhof)
                ax, ay = analytic_acceleration_field(xp[k,i,0]+0.5*k1x, xp[k,i,1]+0.5*k1y)

            k2x = (up[k,i,0] + 0.5*f1x)*dt
            k2y = (up[k,i,1] + 0.5*f1y)*dt
            f2x = (ux - (up[k,i,0] + 0.5*f1x))*dt*A1 + ax*A2*dt
            f2y = (uy - (up[k,i,1] + 0.5*f1y))*dt*A1 + ay*A2*dt
            
            # pseudo step 3
            # ~~~~~~~~~~~~~
            ux, uy = analytic_velocity_field(xp[k,i,0]+0.5*k2x, xp[k,i,1]+0.5*k2y)

            if model==1:
                # Model with drag force only:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:]+[0.5*f2x, 0.5*f2y],
                    rhop, rhof, dp, nu, model=model)
                A2 = 0.
                ax, ay = 0., 0.
            elif model==2:
                # Model with drag force + momentum:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:]+[0.5*f2x, 0.5*f2y],
                    rhop, rhof, dp, nu, model=model)
                A2 = compute_LSM2_A2(rhop, rhof)
                ax, ay = analytic_acceleration_field(xp[k,i,0]+0.5*k2x, xp[k,i,1]+0.5*k2y)

            k3x = (up[k,i,0] + 0.5*f2x)*dt
            k3y = (up[k,i,1] + 0.5*f2y)*dt
            f3x = (ux - (up[k,i,0] + 0.5*f2x))*dt*A1 + ax*A2*dt
            f3y = (uy - (up[k,i,1] + 0.5*f2y))*dt*A1 + ay*A2*dt
            
            # pseudo step 4
            # ~~~~~~~~~~~~~
            ux, uy = analytic_velocity_field(xp[k,i,0]+k3x, xp[k,i,1]+k3y)

            if model==1:
                # Model with drag force only:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:]+[f3x, f3y],
                    rhop, rhof, dp, nu, model=model)
                A2 = 0.
                ax, ay = 0., 0.
            elif model==2:
                # Model with drag force + momentum:
                A1 = compute_LSM2_A1([ux, uy], up[k,i,:]+[f3x, f3y],
                    rhop, rhof, dp, nu, model=model)
                A2 = compute_LSM2_A2(rhop, rhof)
                ax, ay = analytic_acceleration_field(xp[k,i,0]+k3x, xp[k,i,1]+k3y)

            k4x = (up[k,i,0] + f3x)*dt
            k4y = (up[k,i,1] + f3y)*dt
            f4x = (ux - (up[k,i,0] + f3x))*dt*A1 + ax*A2*dt
            f4y = (uy - (up[k,i,1] + f3y))*dt*A1 + ay*A2*dt
            
            # update
            # ~~~~~~
            up[k+1,i,0] = up[k,i,0] + (1./6.)*(f1x + 2.*f2x + 2.*f3x + f4x)
            up[k+1,i,1] = up[k,i,1] + (1./6.)*(f1y + 2.*f2y + 2.*f3y + f4y)
            xp[k+1,i,0] = xp[k,i,0] + (1./6.)*(k1x + 2.*k2x + 2.*k3x + k4x)
            xp[k+1,i,1] = xp[k,i,1] + (1./6.)*(k1y + 2.*k2y + 2.*k3y + k4y)
    return xp
