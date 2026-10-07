# -*- coding: utf-8 -*-
"""
Multiple sources 
===============================================================================

In this example we illustrate how to add particle with sources.

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
file_name = os.path.join('data', 'r2d_flume.slf')
bnd_file = os.path.join('data', 'geo_flume.cli')
fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)

################################################################################
#
# Define LagrangianParticleSet
# -----------------------------------------------------------------------------
#
# .. note::
#    it is important to initialize ``initial_pool_size`` with a rough estimate 
#    of the number of particles that will enter the domain with source.
#    The ``min_pool_size_update`` corresponds to the size of the increase in pool 
#    size if the number of particles reach the ``initial_pool_size`` value.
#    This allocation change impacts performance, so it is recommended to fix
#    a value that is not too low for ``min_pool_size_update``.

pset = cylag.LagrangianParticleSet(\
    None, 
    dim=2, 
    initial_pool_size=10, 
    min_pool_size_update=10)

################################################################################
#
# Define particle sources
# -----------------------------------------------------------------------------
#
# Source 1:
source1_poly = np.array([[5.1, 0.5], [6.1, 0.5], [6.1, 1.5], [5.1, 1.5]])
source1_scheduler = cylag.schedules(time_delta=5.)
source1 = cylag.ParticleSource(
    source1_scheduler, 
    source1_poly,
    xy_method=1, 
    xy_npart=4)

################################################################################
#
# Source 2:
source2_poly = np.array([[10.1, 0.5], [11.1, 0.5], [11.1, 1.5], [10.1, 1.5]])
source2_scheduler = cylag.schedules(iteration_delta=40)
source2 = cylag.ParticleSource(
    source2_scheduler, 
    source2_poly, \
    xy_method=2, 
    xy_density=5.)

sources = [source1, source2]

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
# 
parameters = {
    'final_time': 20.,
    'time_step': 0.1,
    'frozen_eulerian_fields': True,
    'particle_velocity_init': 1,
    'model': 1,
    'time_scheme': 1,
    'diffusion_model': 0,
    'horizontal_diffusivity': 3.e-5,
    'vertical_diffusivity': 0.,
    'boundary_conditions': True,
    'boundary_conditions_debug': False,
    'listing': True,
    'listing_printout_period': 10,
    'output_file': True,
    'output_printout_period': 10,
    'output_file_format': 'txt',
    'output_file_name': 'particles_2d',
    'output_rep': 'resu_02',
    'bnd_statistics': True,
    'bnd_statistics_file_name': 'bndstat',
    }

################################################################################
#
# Run CyLag
# -----------------------------------------------------------------------------
# 
cylag_solver = cylag.Solver(fset, pset, parameters, sources)
cylag_solver.solve()

################################################################################
#
# Post-processing
# -----------------------------------------------------------------------------

# Load particle file:
part = cylag.ParticlesIO.from_cylag_txt('resu_02/particles_2d.txt')

# time
rec = len(part.times) - 1 
time = part.times[rec]

# field set from telemac results
file_name = os.path.join('data', 'r2d_flume.slf')
bnd_file = os.path.join('data', 'geo_flume.cli')
fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)
tri = fset.get_mtri_triangulation()

# get velocity at final time
ux = fset.get_field(1, 0)
uy = fset.get_field(2, 0)
velocity = np.sqrt(ux**2 + uy**2)

# plot result:
cylag.set_rcparams()
fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
ax.set_aspect('equal')

# plot mesh and velocity
levels = np.linspace(0., 1., 11)
cs = ax.tricontourf(tri, velocity, cmap='RdBu_r')
fig.colorbar(cs, ax=ax, label="$\\textbf{U}_f$ (m/s)")
ax.triplot(tri, lw=0.12, c='k')

# plot ini positions
ax.plot(part.xp[0][:], part.yp[0][:], c='b', lw=0., marker='o', markersize=1., label="$x_p(0)$")

# plot final positions
ax.plot(part.xp[rec][:], part.yp[rec][:], c='k', lw=0., marker='o', markersize=1., label="$x_p(t_f)$")
for i in range(len(part.xp[rec][:])):
    ax.text(part.xp[rec][i], part.yp[rec][i], "{}".format(part.tags[rec][i]))

#plt.legend()
plt.savefig("figs/cylag_particles_2d_source_multiple.png", dpi=300, format="png")
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
