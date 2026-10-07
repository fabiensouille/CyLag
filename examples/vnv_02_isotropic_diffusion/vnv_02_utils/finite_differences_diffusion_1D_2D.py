# -*- coding: utf-8 -*-
"""
Analytical advection
"""
import numpy as np
import matplotlib.pylab as plt

def solve_heat_equation_1d(domain, dx, tf, CFL, nu, u0, bnd_type=0, ul=0., ur=0.):
    """
    Solve Heat equation in 1D

    :param domain (array(2)): [xmin, xmax] computational domain
    :param dx (float): space step
    :param tf (float): final time
    :param CFL (float): CFL number (if >0), constant time step (if <0)
    :param nu (float): diffusivity coefficient
    :param bnd_type (int): 0: Dirichlet, 1: Neumann
    :param ul (float): left dirichlet boundary condition (default=0.)
    :param ur (float): right dirichlet boundary condition (default=0.)
    :param u0 (function): initial condition
    :return u (array): solution to the heat equation
    """
    # initialize
    x = np.arange(domain[0], domain[1], dx, dtype='d')
    dt = CFL*(dx**2)/(2.*nu)
    times = np.arange(0., tf, dt, dtype='d')
    nx = len(x)
    nt = len(times)
    u = np.empty((nt, nx), dtype='d')
    # initial condition
    for i in range(nx):
        u[0, i] = u0(x[i])
    # solve heat equation
    alpha = (nu*dt)/(dx**2)
    # time loop
    for k in range(nt-1):
        # boundary conditions
        if bnd_type==0:
            # Dirichlet
            u[k, 0] = ul
            u[k,-1] = ur
        else:
            # Neumann
            u[k, 0] = u[k, 1]
            u[k,-1] = u[k,-2]
        # space loop
        for i in range(1, nx-1):
            u[k+1, i] = u[k, i] + alpha*(u[k, i+1] - 2.*u[k, i] + u[k, i-1])
    return times, x, u

def solve_heat_equation_2d(domainx, domainy, dx, dy, tf, CFL, nu, u0, bnd_type=0,
                           ul=0., ur=0., ud=0., uu=0.):
    """
    Solve Heat equation in 1D

    :param domainx (array(2)): [xmin, xmax] computational domain
    :param domainy (array(2)): [ymin, ymax] computational domain
    :param dx (float): horizontal space step
    :param dy (float): vertical space step
    :param tf (float): final time
    :param CFL (float): CFL number (if >0), constant time step (if <0)
    :param nu (float): diffusivity coefficient
    :param bnd_type (int): 0: Dirichlet, 1: Neumann
    :param ul (float): left dirichlet boundary condition (default=0.)
    :param ur (float): right dirichlet boundary condition (default=0.)
    :param ud (float): down dirichlet boundary condition (default=0.)
    :param uu (float): up dirichlet boundary condition (default=0.)
    :param u0 (function): initial condition
    :return u (array): solution to the heat equation
    """
    # initialize
    x = np.arange(domainx[0], domainx[1], dx, dtype='d')
    y = np.arange(domainy[0], domainy[1], dy, dtype='d')
    dt = CFL*(dx**2)/(4.*nu)
    times = np.arange(0., tf, dt, dtype='d')
    nx = len(x)
    ny = len(y)
    nt = len(times)
    u = np.empty((nt, nx, ny), dtype='d')
    # initial condition
    for i in range(nx):
        for j in range(ny):
            u[0, i, j] = u0(x[i], y[j])
    # solve heat equation
    alphax = (nu*dt)/(dx**2)
    alphay = (nu*dt)/(dy**2)
    # time loop
    for k in range(nt-1):
        # boundary conditions
        if bnd_type==0:
            # Dirichlet
            u[k, 0, :] = ul # left
            u[k,-1, :] = ur # right
            u[k, :, 0] = ud # down
            u[k, :,-1] = uu # up
        else:
            # Neumann
            u[k, 0, :] = u[k, 1, :] # left
            u[k,-1, :] = u[k,-2, :] # right
            u[k, :, 0] = u[k, :, 1] # down
            u[k, :,-1] = u[k, :,-2] # up
        # space loop
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u[k+1, i, j] = u[k, i, j] + \
                        alphax*(u[k, i+1, j] - 2.*u[k, i, j] + u[k, i-1, j]) + \
                        alphay*(u[k, i, j+1] - 2.*u[k, i, j] + u[k, i, j-1])
    return times, x, y, u

if __name__ == '__main__':

    # Solve heat equation in 1D
    def u0(x):
        u0 = 1.*np.exp(-5.*(x-5.)**2.)
        return u0

    times1, x1, u1d = solve_heat_equation_1d(domain=[0., 10.], dx=0.1, tf=1., CFL=0.9, nu=1., u0=u0, bnd_type=1)

    # plot result
    fig, ax = plt.subplots(nrows=1, figsize=(5., 5.))
    ax.plot(x1, u1d[0, :], color='k', ls=":", label='U0')
    ax.plot(x1, u1d[-1, :], color='b', ls="-", label='Uf')
    plt.legend()
    plt.show()

    # Solve heat equation in 2D
    def u0(x, y):
        u0 = 1.*np.exp(-5.*(x-5.)**2.-5.*(y-5.)**2.)
        return u0

    times, x2, y2, u2d = solve_heat_equation_2d(\
        domainx=[0., 10.], \
        domainy=[0., 10.], \
        dx=0.1, dy=0.1, tf=1., CFL=0.9, nu=1., u0=u0, bnd_type=1)

    # plot result
    fig, ax = plt.subplots(nrows=1, figsize=(5., 5.))
    ax.plot(x2, u2d[0, :, int(0.5*len(y2))], color='k', ls=":", label='U0')
    ax.plot(x2, u2d[-1, :, int(0.5*len(y2))], color='b', ls="-", label='Uf')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(nrows=1, figsize=(5., 5.))
    plt.pcolormesh(u2d[-1, :, :], cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
