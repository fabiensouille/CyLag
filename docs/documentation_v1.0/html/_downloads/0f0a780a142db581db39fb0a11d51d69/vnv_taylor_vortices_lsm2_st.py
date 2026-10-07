# -*- coding: utf-8 -*-
"""
Taylor vortices - LSM-2 sensitivity to Strouhal number
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
# Advection with LSM-2 model
# -----------------------------------------------------------------------------
#
xp0 = np.array([[-4., 4.]])
up0 = np.array([[2., 0.]])

print((2.*np.pi/5.)*xp0)


# 2D domain
xx, yy = meshgrid(dx=0.5)
xx2, yy2 = meshgrid(dx=0.05)

# fluid velocity analitical field
ux, uy = taylor_eddies(xx, yy)
ux2, uy2 = taylor_eddies(xx2, yy2)

# particles 
StT = np.array([0.001, 0.01, 0.1, 0.5])

dp = StT*5./np.pi
for d in dp:
    Ar, Re, St = compute_taylor_eddies_adim(u0=1., lamb=5., rhop=2000., rhof=1000., dp=d, nu=1.e-6)
    print("~~~~~~~~~~~~~~~")
    print("dp =", d)
    print("A =", Ar)
    print("Re =", Re)
    print("St =", St)

# LSM2 advection
npart = np.shape(xp0)[0]

xp_st0 = LSM2_analytic_advection_DI(
    taylor_eddies, 
    xp0, 
    up0, 
    tf=100., 
    dt=1.,
    model=2, 
    analytic_acceleration_field=taylor_eddies_acceleration,
    rhop=2000., 
    rhof=1000., 
    dp=dp[0], 
    nu=1.e-6)

xp_st1 = LSM2_analytic_advection_DI(
    taylor_eddies, 
    xp0, 
    up0, 
    tf=100., 
    dt=1.,
    model=2, 
    analytic_acceleration_field=taylor_eddies_acceleration,
    rhop=2000., 
    rhof=1000., 
    dp=dp[1], 
    nu=1.e-6)
    
xp_st2 = LSM2_analytic_advection_DI(
    taylor_eddies, 
    xp0, 
    up0, 
    tf=100., 
    dt=1.,
    model=2, 
    analytic_acceleration_field=taylor_eddies_acceleration,
    rhop=2000., 
    rhof=1000., 
    dp=dp[2], 
    nu=1.e-6)

xp_st3 = LSM2_analytic_advection_DI(
    taylor_eddies, 
    xp0, 
    up0, 
    tf=100., 
    dt=1.,
    model=2, 
    analytic_acceleration_field=taylor_eddies_acceleration,
    rhop=2000., 
    rhof=1000., 
    dp=dp[3], 
    nu=1.e-6)

# adim of x coords
adim_coef = (2.*np.pi/5.)
xx, yy = xx*adim_coef, yy*adim_coef
xx2, yy2 = xx2*adim_coef, yy2*adim_coef
xp_st0 = xp_st0*adim_coef
xp_st1 = xp_st1*adim_coef
xp_st2 = xp_st2*adim_coef
xp_st3 = xp_st3*adim_coef

# Plot 
fig, ax = plt.subplots(1, 1, figsize=(4., 4.))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()

# velocity
vel = np.sqrt(ux2**2 + uy2**2)
cs = ax.contourf(xx2, yy2, vel, cmap='RdBu_r', alpha=0.2)
#fig.colorbar(cs, ax=ax)
ax.quiver(xx, yy, ux, uy, scale=21)

# plot
ax.plot(xp_st0[:,0,0], xp_st0[:,0,1], label="$St_T=0.001$", marker='s', markersize=2., lw=1.5, ls='-', c=color[0], markevery=20)
ax.plot(xp_st1[:,0,0], xp_st1[:,0,1], label="$St_T=0.01$", marker='+', markersize=2.5, lw=1.5, ls='-', c=color[1], markevery=20)
ax.plot(xp_st2[:,0,0], xp_st2[:,0,1], label="$St_T=0.1$", marker='o', markersize=2.5, lw=1.5, ls='-', c=color[2], markevery=20)
ax.plot(xp_st3[:,0,0], xp_st3[:,0,1], label="$St_T=0.5$", marker='^', markersize=2.5, lw=1.5, ls='-', c=color[3], markevery=20)
for i in range(1, npart):
    ax.plot(xp_st0[:,i,0], xp_st0[:,i,1], marker='s', markersize=2., lw=1.5, c=color[0], ls='-', markevery=20)
    ax.plot(xp_st1[:,i,0], xp_st1[:,i,1], marker='+', markersize=2.5, lw=1.5, c=color[1], ls='-', markevery=20)
    ax.plot(xp_st2[:,i,0], xp_st2[:,i,1], marker='o', markersize=2.5, lw=1.5, c=color[2], ls='-', markevery=20)
    ax.plot(xp_st3[:,i,0], xp_st3[:,i,1], marker='^', markersize=2.5, lw=1.5, c=color[3], ls='-', markevery=20)

ax.set_xlabel("$x^*$")
ax.set_ylabel("$y^*$")
ax.set_xlim([-2.*np.pi, 2.*np.pi])
ax.set_ylim([-2.*np.pi, 2.*np.pi])
plt.legend()
plt.grid()
plt.savefig('figs/vnv_2_taylor_vortices_lsm2di_st.pdf', dpi=300, format="pdf")
plt.show()

