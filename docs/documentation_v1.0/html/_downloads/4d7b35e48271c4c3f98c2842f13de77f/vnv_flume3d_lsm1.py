# -*- coding: utf-8 -*-
"""
Flume 3D - LSM-1
===============================================================================

In this example we compute the advection-diffusion of particles in an 3D flume with the LSM-1 model.

"""
import cylag
import os
import numpy as np
import matplotlib.pylab as plt

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Define EulerianFieldSet
# -----------------------------------------------------------------------------
#
file_name = os.path.join('data', 'r3d_particles.slf')
bnd_file = os.path.join('data', 'geo_particles.cli')
fset = cylag.EulerianFieldSet.from_telemac3d(file_name, bnd_file=bnd_file)

print("fset times = ", np.asarray(fset.times))

################################################################################
#
# Define LagrangianParticleSet
# -----------------------------------------------------------------------------
#
circ = cylag.circle(x0=-150., y0=500., r=50., n=180)

position = cylag.init_positions_3d(
    poly=circ,\
    grid_res=(20, 20, 6), 
    zmin=260., 
    zmax=264.)

npart = position.shape[0]

print("number of particles = ", npart)

pset = cylag.LagrangianParticleSet(\
    position, 
    dim=3)

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
# 
# In this example we use the LSM-1 with diffusion.

parameters = {
    'final_time': 3400.,
    'time_step': 1.,
    'model': 1,
    'time_scheme': 1,
    'frozen_eulerian_fields': False,
    'particle_velocity_init': 1,
    'diffusion_model': 1,
    'horizontal_diffusivity': 1.e-3,
    'vertical_diffusivity': 1.e-3,
    'listing': True,
    'listing_printout_period': 100,
    'output_file': True,
    'output_printout_period': 100,
    'output_file_format': 'txt',
    'output_rep': 'resu_04',
    'output_file_name': 'particles_2d',
    }

################################################################################
#
# Run CyLag
# -----------------------------------------------------------------------------
# 
cylag_solver = cylag.Solver(fset, pset, parameters)
cylag_solver.solve()


################################################################################
#
# Post-processing
# -----------------------------------------------------------------------------

################################################################################
#
# Load particle file:

part = cylag.ParticlesIO.from_cylag_txt('resu_04/particles_2d.txt')

# recorded times:
print("list of recorded times =", part.times)
print("number of records = ", len(part.times))

# we plot final record:
rec = len(part.times) - 1
time = part.times[rec]

# select trajectories to plot from particle ID:
traj_list = [0, 10, 20, 30, 100, 110, 120, 130]
traj = part.get_trajectories(traj_list)

################################################################################
#
# Define field set to plot velocity field:

file_name = os.path.join('data', 'r2d_particles.slf')
bnd_file = os.path.join('data', 'geo_particles.cli')
fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)
tri = fset.get_mtri_triangulation()

# get velocity at final time
ux = fset.get_field(1, time)
uy = fset.get_field(2, time)
velocity = np.sqrt(ux**2 + uy**2)

print("Umax =", np.max(velocity))

################################################################################
#
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
    ax.plot(tr[0:rec+1, 0], tr[0:rec+1, 1], lw=0.5, marker='o', markersize=0., c='k')

# plot initial and final positions
ax.plot(part.xp[0][:], part.yp[0][:], c='b', lw=0., marker='o', markersize=1.5, label="$x_p(0)$")
ax.plot(part.xp[rec][:], part.yp[rec][:], c='k', lw=0., marker='o', markersize=1.5, label="$x_p(t_f)$")

plt.xlim([-250, 1550])
plt.ylim([300, 700])
plt.legend()
plt.savefig("figs/cylag_particles_3d_lsm1.png", dpi=150, format="png")
plt.show()

################################################################################
#
# Plot result in (x-z) 2D plan:

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
cylag.set_rcparams()

# plot trajectories
for tr in traj:
    ax.plot(tr[0:rec+1, 0], tr[0:rec+1, 2], lw=0.5, marker='o', markersize=0., c='k')

# plot initial and final positions
ax.plot(part.xp[0][:], part.zp[0][:], c='b', lw=0., marker='o', markersize=1.5, label="$x_p(0)$")
ax.plot(part.xp[rec][:], part.zp[rec][:], c='k', lw=0., marker='o', markersize=1.5, label="$x_p(t_f)$")

plt.xlabel("$x$")
plt.ylabel("$z$")
plt.xlim([-250, 1550])
plt.grid()
plt.legend()
plt.savefig("figs/cylag_particles_3d_vertical.png", dpi=300, format="png")
plt.show()

################################################################################
#
# Clean:
del fset
del pset
del parameters
del cylag_solver
del part
del rec
