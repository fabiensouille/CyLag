# -*- coding: utf-8 -*-
"""
Fish - CBM example with added velocity (LSM-1)
===============================================================================

In this example we illustrate the Collective Based Model 
for fish behavior modelling with the LSM-1 model (added velocity).

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt

sys.path.insert(0, os.path.abspath("../../examples/vnv_12_utils"))
from vnv_12_utils.simple_fset import fset_2d

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Define EulerianFieldSet
# -----------------------------------------------------------------------------
#
tri, fset = fset_2d()

################################################################################
#
# Define FishParticleSet
# -----------------------------------------------------------------------------
#

circ = cylag.circle(x0=0., y0=0., r=70., n=180)
position = cylag.init_positions_2d(poly=circ, grid_res=(15, 15))

npart = position.shape[0]

velocity = np.ones((npart, 2), dtype='d')
for j in range(npart):
    velocity[j, 0] *= np.random.uniform(-1, 1)
    velocity[j, 1] *= np.random.uniform(-1, 1)

################################################################################
#
# We illustrate the CBM fish model, which uses the following parameters:
#
# - swimming_model : int, swimming bahavior model
#
#   - 2 : Fish CBM - LSM1 (swimming velocity)
#   - 4 : Fish CBM - LSM2 (swimming thrust force)
#    
# - attraction_radius : attraction radius for the CBM model (default: 10.)
# - orientation_radius : orientation radius for the CBM model (default: 5.)
# - repulsion_radius : repulsion radius for the CBM model (default: .5)
# - blind_angle : angle defining the cone of blindness of fish for the CBM model (default: 60.)
#

pset = cylag.FishParticleSet(\
    position, 
    velocity, 
    dim=2,
    swimming_model=2,
    attraction_radius = 12.,
    orientation_radius = 10.,
    repulsion_radius = 2.,
    blind_angle = 180.)

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
# 
parameters = {
    'final_time': 300.,
    'time_step': 10.,
    'frozen_eulerian_fields': True,
    'model': 1,
    'time_scheme': 1,
    'diffusion_model': 0,
    'horizontal_diffusivity': 0.,
    'vertical_diffusivity': 0.,
    'boundary_conditions': True,
    'boundary_conditions_debug': False,
    'listing': True,
    'listing_printout_period': 10,
    'output_file': True,
    'output_printout_period': 1,
    'output_rep': 'resu_03',
    'output_file_format': 'txt',
    'output_file_name': 'test_cbm_ua',
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
part = cylag.ParticlesIO.from_cylag_txt('resu_03/test_cbm_ua.txt')
rec = len(part.times) - 1
time = part.times[rec]

# field set from telemac results
tri, fset = fset_2d()

# plot result:
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
cylag.set_rcparams()
ax.set_aspect('equal')


# plot fish particules
for i, tag in enumerate(part.tags[rec]):
    # compute orientation
    dx = part.ua[rec][i]
    dy = part.va[rec][i]
    orientation = np.arctan2(dy, dx)
    # define custom marker
    marker = cylag.define_custom_marker(orientation)
    # plot
    ax.scatter(part.xp[rec][i], part.yp[rec][i], s=70, marker=marker, lw=0., c='k')

plt.xlim([-100., 100.])
plt.ylim([-100., 100.])
#plt.legend()
plt.savefig("figs/test_cbm_ua_2d_{}.png".format(rec), dpi=300, format="png")
plt.show()

