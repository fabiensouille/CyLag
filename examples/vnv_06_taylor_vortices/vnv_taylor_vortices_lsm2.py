# -*- coding: utf-8 -*-
"""
Taylor vortices - LSM-2
===============================================================================

In this example we illustrate advection of particles in an analytical rotational 
velocity field known as Taylor vortices.

Here we compare the different numerical schemes to solve the LSM-2 model.

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt

sys.path.insert(0, os.path.abspath("../../examples/vnv_01_couette_flow"))
from vnv_06_utils.taylor_field_set import taylor_eddies, \
    taylor_eddies_acceleration, meshgrid, compute_taylor_eddies_adim
from vnv_06_utils.analytical_advection import *

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Define main parameters and adimensional number
# -----------------------------------------------------------------------------
#

xp0 = np.array([[-2.,3.5]])
up0 = np.array([[0., 0.]])
up0[:,0], up0[:,1] = taylor_eddies(xp0[:,0], xp0[:,1])

rhop = 2000.
rhof = 1000.
dp = 0.05
nu = 1.e-6
Ar, Re, St = compute_taylor_eddies_adim(u0=1., lamb=5., rhop=rhop, rhof=rhof, dp=dp, nu=nu)

print("A =", Ar)
print("Re =", Re)
print("St =", St)

################################################################################
#
# Advection with LSM-2 model
# -----------------------------------------------------------------------------
#

# 2D domain
xx, yy = meshgrid(dx=0.5)
xx2, yy2 = meshgrid(dx=0.05)

# fluid velocity analitical field
ux, uy = taylor_eddies(xx, yy)
ux2, uy2 = taylor_eddies(xx2, yy2)

# LSM2 advection

# Euler
xp_lsm2_eul = LSM2_analytic_advection_Euler(
    taylor_eddies, 
    xp0,
    up0,
    tf=30., 
    dt=1.,
    model=2, 
    analytic_acceleration_field=taylor_eddies_acceleration,
    rhop=rhop, 
    rhof=rhof, 
    dp=dp,
    nu=nu)

# RK2
xp_lsm2_rk2 = LSM2_analytic_advection_RK2(
    taylor_eddies, 
    xp0,
    up0,
    tf=30., 
    dt=1.,
    model=2, 
    analytic_acceleration_field=taylor_eddies_acceleration,
    rhop=rhop, 
    rhof=rhof, 
    dp=dp,
    nu=nu)

# RK4
xp_lsm2_rk4 = LSM2_analytic_advection_RK4(
    taylor_eddies, 
    xp0,
    up0,
    tf=30., 
    dt=1.,
    model=2, 
    analytic_acceleration_field=taylor_eddies_acceleration,
    rhop=rhop, 
    rhof=rhof, 
    dp=dp,
    nu=nu)

# DI
xp_lsm2_di1 = LSM2_analytic_advection_DI(
    taylor_eddies, 
    xp0,
    up0,
    tf=30., 
    dt=1.,
    model=2, 
    analytic_acceleration_field=taylor_eddies_acceleration,
    rhop=rhop, 
    rhof=rhof, 
    dp=dp,
    nu=nu)

# adim of x coords
adim_coef = (2.*np.pi/5.)
xx, yy = xx*adim_coef, yy*adim_coef
xx2, yy2 = xx2*adim_coef, yy2*adim_coef
xp_lsm2_eul = xp_lsm2_eul*adim_coef
xp_lsm2_rk2 = xp_lsm2_rk2*adim_coef
xp_lsm2_rk4 = xp_lsm2_rk4*adim_coef
xp_lsm2_di1 = xp_lsm2_di1*adim_coef

# Plot 
fig, ax = plt.subplots(1, 1, figsize=(4., 4.))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()

# velocity
vel = np.sqrt(ux2**2 + uy2**2)
cs = ax.contourf(xx2, yy2, vel, cmap='RdBu_r', alpha=0.4)
#fig.colorbar(cs, ax=ax)
ax.quiver(xx, yy, ux, uy, scale=21)

# plot LSM1
ax.plot(xp_lsm2_eul[:,:,0], xp_lsm2_eul[:,:,1], label="Euler", marker='o', markersize=0., lw=2.5, c=color[0])
ax.plot(xp_lsm2_rk2[:,:,0], xp_lsm2_rk2[:,:,1], label="RK2", marker='d', markersize=0., lw=2.5, c=color[1])
ax.plot(xp_lsm2_rk4[:,:,0], xp_lsm2_rk4[:,:,1], label="RK4", marker='s', markersize=0., lw=2.5, c=color[2])
ax.plot(xp_lsm2_di1[:,:,0], xp_lsm2_di1[:,:,1], label="DI", marker='s', markersize=0., lw=2.5, c=color[3])

ax.set_xlabel("$x^*$")
ax.set_ylabel("$y^*$")
ax.set_xlim([-2.*np.pi, 2.*np.pi])
ax.set_ylim([-2.*np.pi, 2.*np.pi])
plt.legend()
#plt.grid()
plt.savefig('figs/vnv_2_taylor_vortices_lsm2.pdf', dpi=300, format="pdf")
plt.show()

