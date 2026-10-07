# -*- coding: utf-8 -*-
"""
Tide poly-disperse - LSM-1
===============================================================================

In this example we compute the advection-diffusion of particles in a coastal 
environment in pseudo-3D with a poly-dipserse distribution of particles.
We use the LSM-1 model.

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

#sphinx_gallery_thumbnail_number = 2

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
    optional_fields=['BOTTOM'],
    friction_model=1,
    friction_coefficient=25.)
    
print("Final time in t2d result file : ", np.asarray(fset.times)[-1])
print("List of fields     : ", fset.get_field_list()[0])
print("List of fields IDs : ", fset.get_field_list()[1])

################################################################################
#
# Define LagrangianParticleSet
# -----------------------------------------------------------------------------
#

circ = cylag.circle(x0=204000., y0=148000., r=800., n=360)

# here we define a single layer on the vertical axis, located in z=8.5
ini_position = cylag.init_positions_3d(
  poly=circ, 
  grid_res=(20, 20),
  z_method=1,
  z_npart=1,
  zmin=8.4,
  zmax=8.6)

npart = np.shape(ini_position)[0]

pset = cylag.LagrangianParticleSet(\
    ini_position, 
    dim=3,
    particle_density=990.,
    particle_diameter=1.e-3,
    particle_diameter_distribution=["uniform", 0.5e-3],
    buoyancy_velocity_model=1, 
    compute_depth=True,
    )
    
#print("particle diameter : ", np.asarray(pset.diameter))

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
#
parameters = {
    'initial_time': 3600.,
    'final_time': 6.*3600.,
    'time_step': 10.,
    'model': 1,
    'time_scheme': 1,
    'frozen_eulerian_fields': False,
    'diffusion_model': 1,
    'horizontal_diffusivity': 1.e-3,
    'vertical_diffusivity': 5.e-3,
    'water_density': 1025.,
    'rebound_damping_coef': 0.25,
    'stranding_probability': 0.5,
    'listing': True,
    'listing_printout_period': 100,
    'output_file': True,
    'output_printout_period': 10,
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

################################################################################
#
# Load particles :
part = cylag.ParticlesIO.from_cylag_txt('resu_01/particles_2d.txt')

# recorded times :
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
zs = fset.get_field(0, time)

################################################################################
#
# Plot result :
cylag.set_rcparams()
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_aspect('equal')

# plot mesh and velocity
levels = np.linspace(0., np.max(velocity)+0.1, 20)
cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.80)
fig.colorbar(cs, ax=ax, label="$U_f$ (m/s)")
ax.triplot(tri, lw=0.1, c='k')
img = ax.quiver(tri.x, tri.y, ux, uy, velocity, cmap='RdBu_r')

# plot final positions
bounds = [1.e-4, 1.5e-4, 1.e-3, 1.5e-3, 1.e-2]
cmap_disc = cmap_disc = ListedColormap(
    plt.cm.Spectral(np.linspace(0., 0.8, len(bounds)-1)))
norm_disc = BoundaryNorm(bounds, cmap_disc.N)
sc = ax.scatter(part.xp[rec][:], part.yp[rec][:], c=part.diameter[rec][:], 
    lw=0., s=3.75, cmap=cmap_disc, norm=norm_disc, label="$x_p$")
fig.colorbar(sc, ax=ax, label="$d_p$", ticks=bounds, extend="max")

plt.xlabel("$x$ (m)")
plt.ylabel("$y$ (m)")
plt.legend()
plt.savefig('figs/cylag_tide_lsm1.png', dpi=300, format="png")
plt.show()

################################################################################
#
# Plot result in (x-z) 2D plan:
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
cylag.set_rcparams()
colors, _, _ = cylag.get_default_color_palette()

# get particle diameter to plot each trajectory with different color
diameter = part.diameter[0][:]
norm = Normalize(vmin=diameter.min(), vmax=diameter.max())
cmap = plt.colormaps["RdBu_r"]

# plot trajectories
for idx, tr in enumerate(traj):
    diam = diameter[idx]
    ax.plot(tr[0:rec+1, 0], tr[0:rec+1, 2], lw=0.5, 
        marker='o', markersize=0., 
        c=cmap(norm(diam)) )

# plot initial and final positions
ax.plot(part.xp[0][:], part.zp[0][:], c='b', lw=0., marker='o', markersize=1.5, label="$\\textbf{x}_p(0)$")
ax.plot(part.xp[rec][:], part.zp[rec][:], c='k', lw=0., marker='o', markersize=1.5, label="$\\textbf{x}_p(t_f)$")

plt.xlabel("$x$")
plt.ylabel("$z$")
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
