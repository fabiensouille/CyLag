# -*- coding: utf-8 -*-
"""
Couette flow - LSM-1 convergence test
===============================================================================

In this example we analyse the convergence of the following numerical scheme for the 
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
    pset = cylag.LagrangianParticleSet(position0, dim=2)

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
        'output_rep': 'resu_03',
        'output_file_name': output_file_name,
        }

    # run
    cylag_solver = cylag.Solver(fset, pset, parameters)
    cylag_solver.solve()

################################################################################
#
# Define function to compute errors
# -----------------------------------------------------------------------------
#
def compute_error(traj, exact_traj):
    nt = len(traj)
    err = 0.
    for k in range(nt):
        dist = np.sqrt((traj[k, 0]-exact_traj[k, 0])**2 + (traj[k, 1]-exact_traj[k, 1])**2)
        err += dist
    return err/nt

################################################################################
#
# Define the initial position
# -----------------------------------------------------------------------------
#
position0 = np.array([[0.3, 0.]])
r0 = np.sqrt(position0[0][0]**2 + position0[0][1]**2)
T = couette_flow_period(r0)

################################################################################
#
# Define number of computations for the convergence and associated time steps
# -----------------------------------------------------------------------------
#

# number of computations
n_traj = 4

# error lists
errors1 = np.empty((n_traj))
errors2 = np.empty((n_traj))
errors3 = np.empty((n_traj))
errors4 = np.empty((n_traj))

# time steps
dts = np.linspace(0.5, 1., n_traj)

################################################################################
#
# Run model for each time step and scheme
# -----------------------------------------------------------------------------
#
for i in range(n_traj):
    # solve trajectory
    dt = dts[i]
    
    # solve trajectory
    run_cylag_couette_lsm1(position0, time_scheme=1, dt=dt, output_file_name='particles_2d_E', nturn=0.25, mesh="couette_refined")
    run_cylag_couette_lsm1(position0, time_scheme=2, dt=dt, output_file_name='particles_2d_RK2', nturn=0.25, mesh="couette_refined")
    run_cylag_couette_lsm1(position0, time_scheme=3, dt=dt, output_file_name='particles_2d_RK3', nturn=0.25, mesh="couette_refined")
    run_cylag_couette_lsm1(position0, time_scheme=4, dt=dt, output_file_name='particles_2d_RK4', nturn=0.25, mesh="couette_refined")
    
    # get particles 
    part1 = cylag.ParticlesIO.from_cylag_txt('resu_03/particles_2d_E.txt')
    part2 = cylag.ParticlesIO.from_cylag_txt('resu_03/particles_2d_RK2.txt')
    part3 = cylag.ParticlesIO.from_cylag_txt('resu_03/particles_2d_RK3.txt')
    part4 = cylag.ParticlesIO.from_cylag_txt('resu_03/particles_2d_RK4.txt')
    
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
    #times = np.arange(0., T, dts[i])
    times = np.asarray(part1.times)
    print(times)
    r, theta = couette_flow_analytical_solution(times, position0[0])
    exact_pos = np.zeros((len(times), 2))
    exact_pos[:, 0] = r * np.cos(theta)
    exact_pos[:, 1] = r * np.sin(theta)

    # plot trajectory error
    nei = np.shape(exact_pos)[0]
    errors1[i] = compute_error(positions1[0:nei, 0:2], exact_pos)
    errors2[i] = compute_error(positions2[0:nei, 0:2], exact_pos)
    errors3[i] = compute_error(positions3[0:nei, 0:2], exact_pos)
    errors4[i] = compute_error(positions4[0:nei, 0:2], exact_pos)

################################################################################
#
# Plot convergence
# -----------------------------------------------------------------------------
#
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()
plt.figure(figsize=(5.5,4.5))
plt.plot(dts, errors1, marker='s', markersize=3,  ls='-', lw=1.5, c=color[0], label="Euler")
plt.plot(dts, errors2, marker='s', markersize=3,  ls='-', lw=1.5, c=color[1], label="RK2")
plt.plot(dts, errors3, marker='s', markersize=3,  ls='-', lw=1.5, c=color[2], label="RK3")
plt.plot(dts, errors4, marker='s', markersize=3,  ls='-', lw=1.5, c=color[3], label="RK4")
plt.grid(which='major', color='grey', linestyle='--')
plt.grid(which='minor', color='grey', linestyle=':')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$dt$ (s)")
plt.ylabel("$E$ (m)")
plt.legend()
#plt.savefig('figs/couette_flow_convergence.pdf', dpi=300, format="pdf")
plt.savefig('figs/couette_flow_convergence.png', dpi=300, format="png")
plt.show()

################################################################################
#
# Clean:
del part1
del part2
del part3
del part4
del rec
