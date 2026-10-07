# -*- coding: utf-8 -*-
"""
Tide - LSM-1 advection-diffusion
===============================================================================

In this example we compute the advection-diffusion of particles in a coastal 
environment with the LSM-1 model.

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Define EulerianFieldSet
# -----------------------------------------------------------------------------
#
file_name = os.path.join('data', 'r2d_tide-jmj_type.slf')
bnd_file = os.path.join('data', 'geo_tide.cli')
fset = cylag.EulerianFieldSet.from_telemac2d(
    file_name, 
    bnd_file=bnd_file,
    optional_fields=['TURBULENT ENERG.', 'DISSIPATION'])

print(np.asarray(fset.times))
print("Final time in t2d result file : ", np.asarray(fset.times)[-1])

################################################################################
#
# Define LagrangianParticleSet
# -----------------------------------------------------------------------------
#
circ = cylag.circle(x0=198000., y0=147000., r=2000., n=360)
ini_position = cylag.init_positions_2d(poly=circ, grid_res=(5, 5))

npart = np.shape(ini_position)[0]
print('Initial number of particles = ', npart)

pset = cylag.LagrangianParticleSet(
    ini_position, 
    dim=2)

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
# 
parameters = {
    'initial_time': 3600.,
    'final_time': 13*3600.,
    'time_step': 10.,
    'model': 1,
    'time_scheme': 1,
    'frozen_eulerian_fields': False,
    'diffusion_model': 2,
    'listing': True,
    'listing_printout_period': 100,
    'output_file': True,
    'output_printout_period': 100,
    'output_file_format': 'txt',
    'output_rep': 'resu_01',
    'output_file_name': 'particles_2d',
    }

################################################################################
#
# Run cylag
# -----------------------------------------------------------------------------
#
cylag_solver = cylag.Solver(fset, pset, parameters)
cylag_solver.solve()

################################################################################
#
# Post-processing
# -----------------------------------------------------------------------------
#
# Load particles :
part = cylag.ParticlesIO.from_cylag_txt('resu_01/particles_2d.txt')

# recorded times :
print("list of recorded times =", part.times)
print("number of records = ", len(part.times))

rec = len(part.times) - 1
time = part.times[rec]
tags = part.tags[rec]
traj = part.get_trajectories(tags)

################################################################################
#
# Field set from telemac results :
file_name = os.path.join('data', 'r2d_tide-jmj_type.slf')
bnd_file = os.path.join('data', 'geo_tide.cli')
fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)
tri = fset.get_mtri_triangulation()

# get velocity at final time
ux = fset.get_field(1, time)
uy = fset.get_field(2, time)
velocity = np.sqrt(ux**2 + uy**2)

################################################################################
#
# Plot result :
cylag.set_rcparams()
fig, ax = plt.subplots(1, 1, figsize=(8.5, 7))
ax.set_aspect('equal')

# plot mesh and velocity
levels = np.linspace(0., np.max(velocity)+0.1, 20)
cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.80)
fig.colorbar(cs, ax=ax, label="$U_f$ (m/s)")
ax.triplot(tri, lw=0.1, c='k')
img = ax.quiver(tri.x, tri.y, ux, uy, velocity, cmap='RdBu_r')

# plot trajectories
for tr in traj:
    ax.plot(tr[0:rec+1, 0], tr[0:rec+1, 1], lw=0.35, marker='o', markersize=0., c="k")

# plot positions
ax.plot(part.xp[0][:], part.yp[0][:], c='b', lw=0., marker='o', markersize=1.8, label="$x_p(0)$")
ax.plot(part.xp[rec][:], part.yp[rec][:], c='r', lw=0., marker='o', markersize=2.2, label="$x_p(t_f)$")

plt.xlabel("$x$ (m)")
plt.ylabel("$y$ (m)")
plt.legend()
plt.savefig('figs/cylag_tide_lsm1.png', dpi=300, format="png")
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
