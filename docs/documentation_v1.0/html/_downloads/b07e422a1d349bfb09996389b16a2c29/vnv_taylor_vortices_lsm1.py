# -*- coding: utf-8 -*-
"""
Taylor vortices - LSM-1
===============================================================================

In this example we illustrate advection of particles in an analytical rotational 
velocity field known as Taylor vortices.

Here we compare the different numerical schemes to solve the LSM-1 model.

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
# Advection with LSM-1 model
# -----------------------------------------------------------------------------
#

# 2D domain
xx, yy = meshgrid(dx=0.5)
xx2, yy2 = meshgrid(dx=0.05)

# fluid velocity analitical field
ux, uy = taylor_eddies(xx, yy)
ux2, uy2 = taylor_eddies(xx2, yy2)
ax2, ay2 = taylor_eddies_acceleration(xx2, yy2)

# initial positions
xp0 = np.array([[-2., 2.]])

# LSM1 advection
dt = 0.8
tf = 50. 
xp_lsm1_eul = LSM1_analytic_advection_Euler(taylor_eddies, xp0, tf=tf, dt=dt)
xp_lsm1_rk2 = LSM1_analytic_advection_RK2(taylor_eddies, xp0, tf=tf, dt=dt)
xp_lsm1_rk4 = LSM1_analytic_advection_RK4(taylor_eddies, xp0, tf=tf, dt=dt)

# adim of x coords
adim_coef = (2.*np.pi/5.)
xx, yy = xx*adim_coef, yy*adim_coef
xx2, yy2 = xx2*adim_coef, yy2*adim_coef
xp_lsm1_eul = xp_lsm1_eul*adim_coef
xp_lsm1_rk2 = xp_lsm1_rk2*adim_coef
xp_lsm1_rk4 = xp_lsm1_rk4*adim_coef

# Plot 
fig, ax = plt.subplots(1, 1, figsize=(6., 5))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()
# velocity
vel = np.sqrt(ux2**2 + uy2**2)
levels = np.linspace(0., 1., 11)
cs = ax.contourf(xx2, yy2, vel, levels, cmap='RdBu_r', alpha=0.4)
fig.colorbar(cs, ax=ax, label="$|\\textbf{U}_f|/U_0$")
ax.quiver(xx, yy, ux, uy, scale=21)
# plot LSM1 
ax.plot(xp_lsm1_eul[:,:,0], xp_lsm1_eul[:,:,1], label="Euler", marker='o', markersize=2., lw=1., c=color[0])
ax.plot(xp_lsm1_rk2[:,:,0], xp_lsm1_rk2[:,:,1], label="RK2", marker='s', markersize=2., lw=1., c=color[1])
ax.plot(xp_lsm1_rk4[:,:,0], xp_lsm1_rk4[:,:,1], label="RK4", marker='+', markersize=2., lw=1., c=color[2])
ax.set_xlabel("$x^*$")
ax.set_ylabel("$y^*$")
ax.set_xlim([-2.*np.pi, 2.*np.pi])
ax.set_ylim([-2.*np.pi, 2.*np.pi])
plt.legend()
plt.grid()
plt.savefig('figs/vnv_2_taylor_vortices_lsm1.pdf', dpi=300, format="pdf")
plt.show()

# plot 
fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()

# velocity
vel = np.sqrt(ux2**2 + uy2**2)
lw = 1.5*vel/vel.max()
#cs = ax.contourf(xx2, yy2, vel, cmap='RdBu_r', alpha=0.8)
#fig.colorbar(cs, ax=ax, label="$|U_f^*|$")
ax.streamplot(xx2, yy2, ux2, uy2, color=vel, cmap='RdBu_r', linewidth=lw, density=2., maxlength=10.0, arrowsize=0.5)
ax.plot(xp_lsm1_eul[:,:,0], xp_lsm1_eul[:,:,1], label="$Euler$", marker='o', markersize=2., lw=1., c=color[0])
ax.plot(xp_lsm1_rk2[:,:,0], xp_lsm1_rk2[:,:,1], label="$RK2$", marker='s', markersize=2., lw=1., c=color[1])
ax.plot(xp_lsm1_rk4[:,:,0], xp_lsm1_rk4[:,:,1], label="$RK4$", marker='+', markersize=2., lw=1., c=color[2])
ax.set_xlabel("$x^*$")
ax.set_ylabel("$y^*$")
ax.set_xlim([-2.*np.pi, 2.*np.pi])
ax.set_ylim([-2.*np.pi, 2.*np.pi])
plt.gca().set_aspect('equal')
plt.legend()
plt.savefig('figs/vnv_2_taylor_vortices_streamlines_lsm1.pdf', dpi=300, format="pdf")
plt.show()
   
