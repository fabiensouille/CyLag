# -*- coding: utf-8 -*-
"""
Couette flow - LSM-1 advection with Euler scheme 
===============================================================================

In this example we compute the advection of a particle in an analytical velocity field
with the LSM-1 model without diffusion (material pathlines) solved with the Euler scheme.

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt

sys.path.insert(0, os.path.abspath("../../examples/vnv_01_couette_flow"))
from vnv_01_utils.couette_field_set import couette_field_set, couette_flow_period

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Define EulerianFieldSet
# -----------------------------------------------------------------------------
#
fset, tri = couette_field_set(mesh="couette")

################################################################################
#
# We use a custom function that creates mesh and build vector field from 
# an analytical solution:
#
# .. math::
#    & U(r) = U_m * ((r-R_{0})/(R_{1}-R_{0})) \\
#    & \bold{u}(x,y) = (u_x(r,\theta), u_y(r,\theta)) \\
#
# where:
#
# .. math::
#    & u_x(r, \theta) = -U(r)\sin{\theta} \\
#    & u_y(r, \theta) =  U(r)\cos{\theta} 


################################################################################
#
# Define LagrangianParticleSet
# -----------------------------------------------------------------------------
#
# We define a single particle located at a radius :math:`r_0`

position0 = np.array([[0.3, 0.]])
r0 = np.sqrt(position0[0][0]**2 + position0[0][1]**2)

pset = cylag.LagrangianParticleSet(position0, dim=2)

################################################################################
#
# The period of rotation :math:`T` for a particle located at :math:`r_0` is given by
#
# .. math::
#   T(r_0) = \dfrac{2 \pi r_0}{U(r_0)}

T = couette_flow_period(r0)

print(T)

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
# 
# We set the final time to the period corresponding to the initial radius.
# The LSM-1 model is used without diffusion and with the Euler advection scheme.

parameters = {
    'final_time': T,
    'time_step': 0.4,
    'frozen_eulerian_fields': True,
    'model': 1,
    'time_scheme': 1,
    'diffusion_model': 0,
    'horizontal_diffusivity': 0.,
    'vertical_diffusivity': 0.,
    'boundary_conditions': True,
    'boundary_conditions_debug': False,
    'listing': True,
    'listing_printout_period': 1,
    'output_file': True,
    'output_printout_period': 1,
    'output_file_format': 'txt',
    'output_rep': 'resu_01',
    'output_file_name': 'particles_2d_euler',
    }

################################################################################
#
# Run cylag
# -----------------------------------------------------------------------------
cylag_solver = cylag.Solver(fset, pset, parameters)
cylag_solver.solve()

################################################################################
#
# Post-processing
# -----------------------------------------------------------------------------

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()
ax.set_aspect('equal')

# load particle file and get trajectory of particle
part = cylag.ParticlesIO.from_cylag_txt('resu_01/particles_2d_euler.txt')
rec = len(part.times)-1
time = part.times[rec]
traj = part.get_trajectories([0])

# plot velocity
fset, tri = couette_field_set()
ux = fset.get_field(1, time)
uy = fset.get_field(2, time)
velocity = np.sqrt(ux**2 + uy**2)
cs = ax.tricontourf(tri, velocity, levels=np.linspace(0., 1.0, 11), cmap='RdBu_r', alpha=0.75)
fig.colorbar(cs, ax=ax, label="$|\\textbf{U}_f|/U_m$")
ax.triplot(tri, lw=0.25, color='0.5')

# Plots direction of the velocity vector field
ax.quiver(tri.x, tri.y, ux, uy,
          units='xy', scale=15., zorder=3, color='0.35',
          width=0.007, headwidth=3., headlength=4.)

# plot trajectories
for tr in traj:
    ax.plot(tr[0:rec+1, 0], tr[0:rec+1, 1], 
            lw=1.5, marker='o', markersize=0., c=color[0],
            label="Euler")

# plot positions
ax.plot(part.xp[0][:], part.yp[0][:], c=color[0], lw=0., marker='o', markersize=1.)
ax.plot(part.xp[rec][:], part.yp[rec][:], c=color[0], lw=0., marker='o', markersize=1.)

plt.xlabel("$x/R_{1}$")
plt.ylabel("$y/R_{1}$")
plt.xlim([-1.25, 1.25])
plt.ylim([-1.25, 1.25])
plt.legend(loc=1)

#plt.savefig('figs/couette_flow_euler.pdf', dpi=300, format="pdf")
plt.savefig('figs/couette_flow_euler.png', dpi=300, format="png")
plt.show()

################################################################################
#
# .. note::
#   As illustrated on the Figure, the particle has drifted from its original radius
#   position due to numerical errors. As a consequence the particle has been advected 
#   by higher velocity and has done more than one turn. 
#   This could be fixed by decreasing the time step to increase accuracy or by
#   using higher order schemes.
# 

################################################################################
#
# Clean:
del fset
del pset
del parameters
del cylag_solver
del part
del rec
