# -*- coding: utf-8 -*-
"""
Fish - IBM example with added trust force (LSM-2)
===============================================================================

In this example we illustrate the Idividual Based Model 
for fish behavior modelling with the LSM-2 model (added trust force).

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt

sys.path.insert(0, os.path.abspath("../../examples/vnv_12_utils"))

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Define EulerianFieldSet
# -----------------------------------------------------------------------------
#
t2d_file = os.path.join("data", "r2d_particles_fin.slf")
bnd_file = os.path.join("data", "geo_particles.cli")
fset = cylag.EulerianFieldSet.from_telemac2d(t2d_file, bnd_file=bnd_file)

################################################################################
#
# Define FishParticleSet
# -----------------------------------------------------------------------------
#

# polygon in which to add particles
poly = np.array([[-200., 380.],\
                 [-100., 380.],\
                 [-100., 620.],\
                 [-200., 620.]])

position = cylag.init_positions_2d(poly=poly, grid_res=(10, 20))
npart = position.shape[0]

################################################################################
#
# We illustrate the IBM fish model, which uses the following parameters:
#
# - swimming_model : int, swimming bahavior model
#
#   - 1 : Fish IBM - LSM1 (swimming velocity)
#   - 3 : Fish IBM - LSM2 (swimming thrust force)
#
# - swimming_threshold_fluid_velocity : threshold velocity that trigger change of state 
# - swimming_velocity_states : swimming velocity of fish for each latent state
# - swimming_dispersion : dispersion of the direction of swimming for latent states 2 and 3
# - swimming_thrust_forces : swimming thrust force of fish for each latent state

pset = cylag.FishParticleSet(\
    position,
    dim=2,
    particle_density=1000., 
    particle_diameter=0.1,
    drag_coefficient_model=0,
    fish_drag_coefficient=0.06,
    swimming_model=3,
    swimming_threshold_fluid_velocity=[0.25, 0.5],
    swimming_thrust_forces=[6.e-4, 6.e-3, 6.5e-3],
    swimming_dispersion=[0.025, 0.05])

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
# 
parameters = {
    'final_time': 6000.,
    'time_step': 10.,
    'frozen_eulerian_fields': True,
    'model': 2,
    'time_scheme': 5,
    'diffusion_model': 0,
    'boundary_conditions': True,
    'boundary_conditions_debug': False,
    'listing': True,
    'listing_printout_period': 10,
    'output_file': True,
    'output_printout_period': 1,
    'output_rep': 'resu_02',
    'output_file_format': 'txt',
    'output_file_name': 'particles_2d_ibm_ua',
    }

################################################################################
#
# Run
# -----------------------------------------------------------------------------
# 
cylag_solver = cylag.Solver(fset, pset, parameters)
cylag_solver.solve()


################################################################################
#
# Post-processing
# -----------------------------------------------------------------------------
# 

# particles I/O
part = cylag.ParticlesIO.from_cylag_txt('resu_02/particles_2d_ibm_ua.txt')
rec=600
time = part.times[rec]
traj_list=[0, 10, 20, 30, 100, 110, 120, 130]
traj = part.get_trajectories(traj_list)

# field set from telemac results
t2d_file = os.path.join("data", "r2d_particles_fin.slf")
bnd_file = os.path.join("data", "geo_particles.cli")
fset = cylag.EulerianFieldSet.from_telemac2d(t2d_file, bnd_file=bnd_file)
tri = fset.get_mtri_triangulation()

# get velocity at final time
ux = fset.get_field(1, time)
uy = fset.get_field(2, time)
velocity = np.sqrt(ux**2 + uy**2)

# plot result:
fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
cylag.set_rcparams()
ax.set_aspect('equal')

# plot mesh and velocity
levels = np.linspace(0., 0.75, 16)
cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.80)
fig.colorbar(cs, ax=ax, label="$\\textbf{U}_f$ (m/s)")
ax.triplot(tri, lw=0.12, c='k')

# plot trajectories
for tr in traj:
    ax.plot(tr[0:rec+1, 0], tr[0:rec+1, 1], lw=0.5, marker='o', markersize=0., c='k')

# plot fish particules
for i, tag in enumerate(part.tags[rec]):
    # compute orientation
    dx = part.fax[rec][i]
    dy = part.fay[rec][i]
    orientation = np.arctan2(dy, dx)
    # define custom marker
    marker = cylag.define_custom_marker(orientation)
    # plot
    ax.scatter(part.xp[rec][i], part.yp[rec][i], s=70, marker=marker, lw=0., c='k')

#plt.legend()
plt.savefig("figs/cylag_fish_ibm_fa.png", dpi=300, format="png")
plt.show()
