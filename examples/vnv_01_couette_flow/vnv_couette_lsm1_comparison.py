# -*- coding: utf-8 -*-
"""
Couette flow - LSM-1 advection with different schemes (comparison)
===============================================================================

In this example we compare the following numerical scheme for the 
LSM-1 model without diffusion (material pathlines):

- **Euler**
- **Runge-Kutta 2**
- **Runge-Kutta 4**


"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt

sys.path.insert(0, os.path.abspath("../../examples/vnv_01_couette_flow"))
from vnv_01_utils.couette_field_set import couette_field_set, couette_flow_period, \
    couette_flow_analytical_solution

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Define function to run CyLag
# -----------------------------------------------------------------------------
#
def run_cylag_couette_lsm1(\
        position0=np.array([[0.4, 0.]]), 
        time_scheme=1, 
        dt=0.4, 
        nturn=1.5,
        output_file_name='particles_2d_euler',
        mesh="couette"):
        
    # define EulerianFieldSet 
    fset, tri = couette_field_set(mesh)

    # define LagrangianParticleSet
    r0 = np.sqrt(position0[0][0]**2 + position0[0][1]**2)
    T = couette_flow_period(r0)
    pset = cylag.LagrangianParticleSet(\
        position0, dim=2, initial_pool_size=1)

    # set general parameters
    parameters = {
        'final_time': nturn*T,
        'time_step': dt,
        'frozen_eulerian_fields': True,
        'model': 1,
        'time_scheme': time_scheme,
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
        'output_rep': 'resu_02',
        'output_file_name': output_file_name,
        }

    # run
    cylag_solver = cylag.Solver(fset, pset, parameters)
    cylag_solver.solve()

################################################################################
#
# Run model with different schemes for comparison
# -----------------------------------------------------------------------------
#
run_cylag_couette_lsm1(time_scheme=1, dt=0.4, output_file_name='particles_2d_E', nturn=3)
run_cylag_couette_lsm1(time_scheme=2, dt=0.8, output_file_name='particles_2d_RK2', nturn=3)
run_cylag_couette_lsm1(time_scheme=3, dt=1.2, output_file_name='particles_2d_RK3', nturn=3)
run_cylag_couette_lsm1(time_scheme=4, dt=1.6, output_file_name='particles_2d_RK4', nturn=3)

################################################################################
#
# Post-processing
# -----------------------------------------------------------------------------
#
# get particles 
part1 = cylag.ParticlesIO.from_cylag_txt('resu_02/particles_2d_E.txt')
part2 = cylag.ParticlesIO.from_cylag_txt('resu_02/particles_2d_RK2.txt')
part3 = cylag.ParticlesIO.from_cylag_txt('resu_02/particles_2d_RK3.txt')
part4 = cylag.ParticlesIO.from_cylag_txt('resu_02/particles_2d_RK4.txt')

# time
rec = len(part1.times)-1
time = part1.times[rec]
T = time

# load particles trajectories
traj1 = part1.get_trajectories([0])
traj2 = part2.get_trajectories([0])
traj3 = part3.get_trajectories([0])
traj4 = part4.get_trajectories([0])
positions1 = traj1[0]
positions2 = traj2[0]
positions3 = traj3[0]
positions4 = traj4[0]

# analytical trajectory
times = np.linspace(0., T, 60)
r, theta = couette_flow_analytical_solution(times, [0.4, 0.])
exact_pos = np.zeros((len(times), 2))
exact_pos[:, 0] = r*np.cos(theta)
exact_pos[:, 1] = r*np.sin(theta)

# field set
fset, tri = couette_field_set()
ux = fset.get_field(1, time)
uy = fset.get_field(2, time)
velocity = np.sqrt(ux**2 + uy**2)
print("Um =", np.max(velocity))

# Plot 
fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()
plt.gca().set_aspect('equal')

# plot velocity
cs = ax.tricontourf(tri, velocity, levels=np.arange(0., 1., 0.01), cmap='RdBu_r', alpha=0.4)
fig.colorbar(cs, ax=ax, label="$|\\textbf{U}_f|/U_m$")

# Plots direction of the velocity vector field
ax.quiver(tri.x, tri.y, ux, uy,
          units='xy', scale=15., zorder=3, color='0.5',
          width=0.007, headwidth=3., headlength=4.)
          
# plot particles
ax.plot(positions1[:, 0], positions1[:, 1],
    marker='+', markersize=3,  ls='-',  lw=1.5, c=color[0], label="Euler, $dt={}$".format(0.4))
ax.plot(positions2[:, 0], positions2[:, 1],
    marker='x', markersize=3,  ls='-', lw=1.5, c=color[1], label="RK2, $dt={}$".format(0.8))
ax.plot(positions3[:, 0], positions3[:, 1],
    marker='s', markersize=3,  ls='-', lw=1.5, c=color[2], label="RK3, $dt={}$".format(1.2))
ax.plot(positions4[:, 0], positions4[:, 1],
    marker='o', markersize=4,  ls='-', lw=1.25, c=color[3], label="RK4, $dt={}$".format(1.6))
    
# plot analytical trajectory
ax.plot(exact_pos[:, 0], exact_pos[:, 1], ls='-', lw=1., c='k', label="Exact")

plt.xlabel("$x/R_{1}$")
plt.ylabel("$y/R_{1}$")
plt.xlim([-1.25, 1.25])
plt.ylim([-1.25, 1.25])
plt.legend(loc=1)

#plt.savefig('figs/couette_flow_comparison.pdf', dpi=300, format="pdf")
plt.savefig('figs/couette_flow_comparison.png', dpi=300, format="png")
plt.show()

################################################################################
#
# Clean:
del part1
del part2
del part3
del part4
del rec
