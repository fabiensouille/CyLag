# -*- coding: utf-8 -*-
"""
Flume 2D - LSM-1 sensitivity to Peclet number
===============================================================================

In this example we compute the advection-diffusion of particles in an 2D flume 
with the LSM-1 model for different Peclet numbers.

"""
import cylag
import os
import numpy as np
import matplotlib.pylab as plt

#sphinx_gallery_thumbnail_number = 4

################################################################################
#
# Define function to run CyLag
# -----------------------------------------------------------------------------
#
# In this exmple we use the LSM-1 model with diffusion and solve it using the Euler-Maruyama scheme.

def run_cylag(pe=1.e3, particle_file_name="particles_2d_pe"):

    print("Peclet number = ", pe)

    # define EulerianFieldSet :
    file_name = os.path.join('data', 'r2d_particles.slf')
    bnd_file = os.path.join('data', 'geo_particles.cli')
    fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)

    # define LagrangianParticleSet : 
    circ = cylag.circle(x0=-180., y0=500., r=10., n=180)
    position = cylag.init_positions_2d(poly=circ, grid_res=(20, 20))
    npart = position.shape[0]
    pset = cylag.LagrangianParticleSet(\
        position, 
        dim=2)

    # set general parameters :
    Um = 0.5      # mean velocity for computation of Pe
    Lc = 1700.    # length scale of the flume for computation of Pe
    nu = Um*Lc/pe # diffusivity computed from Pe
    
    parameters = {
        'final_time': 3600.,
        'time_step': 0.5,
        'frozen_eulerian_fields': False,
        'model': 1,
        'time_scheme': 1,
        'particle_velocity_init': 1,
        'diffusion_model': 1,
        'horizontal_diffusivity': nu,
        'boundary_conditions': True,
        'listing': True,
        'listing_printout_period': 1000,
        'output_file': True,
        'output_printout_period': 50,
        'output_rep': 'resu_02',
        'output_file_format': 'txt',
        'output_file_name': particle_file_name,
        }

    # traking algo run
    # ~~~~~~~~~~~~~~~~
    cylag_solver = cylag.Solver(fset, pset, parameters)
    cylag_solver.solve()

   
################################################################################
#
# Run CyLag for different Peclet number
# -----------------------------------------------------------------------------
# 

Pe_list = [1.e6, 1.e5, 1.e4, 5.e3]

for idx, Pe in enumerate(Pe_list):
    run_cylag(Pe, "particles_2d_pe_{}".format(idx))

################################################################################
#
# Post-processing
# -----------------------------------------------------------------------------
#

for idx, Pe in enumerate(Pe_list):

    part = cylag.ParticlesIO.from_cylag_txt('resu_02/particles_2d_pe_{}.txt'.format(idx))

    rec = len(part.times) - 1
    time = part.times[rec]

    traj_list = [0, 10, 20, 30, 40, 50, 100, 110, 120, 130, 140, 150]
    traj = part.get_trajectories(traj_list)

    # Define field set to plot velocity field:
    file_name = os.path.join('data', 'r2d_particles.slf')
    bnd_file = os.path.join('data', 'geo_particles.cli')
    fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)
    tri = fset.get_mtri_triangulation()
    ux = fset.get_field(1, time)
    uy = fset.get_field(2, time)
    velocity = np.sqrt(ux**2 + uy**2)

    # Plot result:
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    cylag.set_rcparams()
    ax.set_aspect('equal')

    # plot mesh and velocity
    levels = np.linspace(0.1, 0.7, 13)
    cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75, extend="both")
    fig.colorbar(cs, ax=ax, label="$|\\textbf{U}_f|$ (m/s)")
    ax.triplot(tri, lw=0.12, c='k')

    # plot trajectories
    for tr in traj:
        ax.plot(tr[0:rec+1, 0], tr[0:rec+1, 1], lw=0.35, marker='o', markersize=0., c='k')

    # plot initial and final positions
    ax.plot(part.xp[0][:], part.yp[0][:], c='b', lw=0., marker='o', markersize=1.5, label="$x_p(0)$")
    ax.plot(part.xp[rec][:], part.yp[rec][:], c='k', lw=0., marker='o', markersize=1.5, label="$x_p(t_f)$")

    plt.xlim([-250, 1550])
    plt.ylim([300, 700])
    plt.legend()
    plt.title("Peclet number = {}".format(Pe))
    plt.savefig("figs/cylag_particles_2d_pe_{}.png".format(idx), dpi=150, format="png")
    plt.show()

################################################################################
#
# Clean:
del part
del rec
