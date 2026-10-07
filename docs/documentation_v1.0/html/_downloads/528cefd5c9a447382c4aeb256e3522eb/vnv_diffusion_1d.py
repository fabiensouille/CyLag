# -*- coding: utf-8 -*-
"""
Stochastic diffusion in 1D
===============================================================================

This example shows the basic principle of stochastic diffusion in 1D.
In this example, we : 

- solve 1D diffusion equation with finite differences for reference,
- solve the stochastic diffusion in 1D with a Gaussian initial condition, 
- plot trajectories of particles,
- compare results to the reference by computing estimates of density with two methods:

  - Kernel Smoothing (KS)
  - Nearest Grid Point (NGP) 

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt

np.random.seed(0)

sys.path.insert(0, os.path.abspath("../../examples/vnv_02_isotropic_diffusion"))
from vnv_02_utils.finite_differences_diffusion_1D_2D import solve_heat_equation_1d

#sphinx_gallery_thumbnail_number = 3

################################################################################
#
# Define function to model stochastic diffusion in 1D
# -----------------------------------------------------------------------------
#
def Stochastic_diffusion_1D(nt, dt, nu, Z0=0.):
    # Weiner increments
    dW = np.random.normal(0, np.sqrt(dt), nt)
    # Diffusion
    B = np.sqrt(2*nu)
    dW = dW*B 
    # Solution
    Z = np.cumsum(dW)
    Z += Z0
    return Z

################################################################################
#
# Solve diffusion equation with Finite Difference scheme for comparison
# -----------------------------------------------------------------------------
#
def u0(x):
    u0 = 1.*np.exp(-2.*(x-5.)**2.)
    return u0

times, x, u = solve_heat_equation_1d(
    domain=[0., 10.], 
    dx=0.05,
    tf=1., 
    CFL=0.95, 
    nu=1., 
    u0=u0, 
    bnd_type=0)

# Plot result
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()

List_of_time_to_plot = [0, 50, 150, 350, 800]

for k in List_of_time_to_plot:
    print("times plotted = ", times[k])

for idx, k in enumerate(List_of_time_to_plot):
    ax.plot(x, u[k, :], c=color[idx%len(color)], label='$t={:.2f}$s'.format(times[k]))
    
ax.set_xlabel("$x$ (m)")
ax.set_ylabel("$\\rho_{FD}(t,x)$")
ax.set_ylim([0.,1.])
plt.legend()
plt.grid()
plt.savefig('figs/diffusion1d_fd.png', dpi=300, format="png")
plt.show()


################################################################################
#
# Initialization particles positions based CDF fucntion of C0
# -----------------------------------------------------------------------------
#
npart = 1000
sampling = 1

np.random.seed(1)
xp0, massp, cdf = cylag.initialize_particles_from_pdf_1D(x, u[0, :], npart, epsilon=1e-8, sampling=sampling)

# integral of the initial condition
pdf0 = np.asarray(cylag.compute_density_1D_NGP(x, xp0, massp))
pdf2 = np.asarray(cylag.compute_density_1D_KS(x, xp0, massp, h=0.25, kernel=5))

# plot initial condition
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.))
#ax.plot(x, cdf, label='CDF of $p$', c=color[1])
ax.plot(xp0, 0.5*np.ones(npart), c="k", marker='o', lw=0., markersize=0.25, label="$x_p$")
ax.plot(x, pdf0, label='NGP', c=color[2], lw=1.)
ax.plot(x, pdf2, label='GKS', c=color[1], lw=1.8)
ax.plot(x, u[0, :], label='$\\rho_0$', c=color[0], ls='--', lw=1.)
plt.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$\\rho(0, x)$")
ax.set_ylim([0.,1.2])
plt.grid()
plt.savefig('figs/diffusion1d_initial_npart{}_s{}.png'.format(npart, sampling), dpi=300, format="png")
plt.show()

################################################################################
#
# Computing particle trajetories
# -----------------------------------------------------------------------------
#
nt = len(times)
dt = times[1]-times[0]
xp = np.empty((nt, npart))

# compute trajectories
for i in range(npart):
    np.random.seed(i*10)
    xp[:, i] = Stochastic_diffusion_1D(nt, dt=dt, nu=1., Z0=xp0[i])
    
################################################################################
#
# plot trajectories:
#
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()

for i in range(npart):
    ax.plot(times, xp[:, i], c=color[i%len(color)], lw=0.8)

ax.set_xlabel("$t$ (s)")
ax.set_ylabel("$x_p(t)$")
ax.set_ylim([0, 10])
#plt.legend()
plt.grid()
plt.savefig('figs/diffusion1d_trajectories_npart{}.png'.format(npart), dpi=300, format="png")
plt.show()

################################################################################
#
# Compute density with kernel smoothing method
# -----------------------------------------------------------------------------
#
pdf2 = np.asarray(cylag.compute_density_1D_KS(x, xp[0, :], massp, h=0.25, kernel=5))

# plot
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()

# plot density
ax.plot(x, pdf2, label='GKS', c=color[0], lw=1.8)
ax.plot(x, u[0, :], label='FD', c=color[0], ls='--', lw=1.)

List_of_time_to_plot = [50, 150, 350, 800]

for idx, k in enumerate(List_of_time_to_plot):
    # compute density
    pdf2 = np.asarray(cylag.compute_density_1D_KS(x, xp[k, :], massp, h=0.25, kernel=5))
    # plot density
    ax.plot(x, pdf2, c=color[idx+1], lw=1.8)
    ax.plot(x, u[k, :], c=color[idx+1], ls='--', lw=1.)

plt.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$\\rho(t, x)$")
ax.set_ylim([0.,1.])
plt.grid()
plt.savefig('figs/diffusion1d_ks_npart{}.png'.format(npart, sampling), dpi=300, format="png")
plt.show()

################################################################################
#
# .. note::
#   The density estimate computed from kernel smoothing fits the reference solution.
#   For a more accurate estimate, the number of particles should be increased until convergence.

################################################################################
#
# Compute density with the Nearest Grid Point method
# -----------------------------------------------------------------------------
#
pdf0 = np.asarray(cylag.compute_density_1D_NGP(x, xp[0, :], massp))

# plot
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()

# plot density
ax.plot(x, pdf0, label='NGP', c=color[0], lw=1.)
#ax.plot(x, pdf2, label='SPH', c=color[0], lw=1.8)
ax.plot(x, u[0, :], label='FD', c=color[0], ls='--', lw=1.)

List_of_time_to_plot = [50, 150, 350, 800]

for idx, k in enumerate(List_of_time_to_plot):
    # compute density
    pdf0 = np.asarray(cylag.compute_density_1D_NGP(x, xp[k, :], massp))
    # plot density
    ax.plot(x, pdf0, c=color[idx+1], lw=1.)
    ax.plot(x, u[k, :], c=color[idx+1], ls='--', lw=1.)

plt.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$\\rho(t, x)$")
ax.set_ylim([0.,1.])
plt.grid()
plt.savefig('figs/diffusion1d_ngp_npart{}.png'.format(npart, sampling), dpi=300, format="png")
plt.show()

################################################################################
#
# .. note::
#   Contrary to the kernel smoothing, the nearest grid point estimation is much more
#   chaotic which makes it difficult to rely on for physical interpretation.

