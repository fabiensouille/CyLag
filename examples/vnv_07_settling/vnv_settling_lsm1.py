# -*- coding: utf-8 -*-
"""
Settling with the LSM-1 model
===============================================================================

In this example we compute the settling of particle in a simple 3D domain
with the LSM-1 model.

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt

sys.path.insert(0, os.path.abspath("../../examples/vnv_07_settling"))
from vnv_07_utils.custom_field_set import fset_3d

#sphinx_gallery_thumbnail_number = 1


################################################################################
#
# Define EulerianFieldSet
# -----------------------------------------------------------------------------
#
fset = fset_3d()

################################################################################
#
# Define LagrangianParticleSet
# -----------------------------------------------------------------------------
#
# With the LSM-1 model the settling velocity is added with analytical solution
# using ``buoyancy_velocity_model=1``.

position = np.asarray([\
    [5.1, 5.1, 19.45], [2.1, 2.1, 19.45]])

velocity = np.asarray([\
    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

pset = cylag.LagrangianParticleSet(\
    position, 
    velocity, 
    dim=3,
    initial_pool_size=2,
    particle_density= 2500., 
    particle_diameter=1.e-3,
    buoyancy_velocity_model=1)

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
# 

parameters = {
    'final_time': 10.,
    'time_step': 0.01,
    'frozen_eulerian_fields': True,
    'model': 1,
    'time_scheme': 1,
    'particle_velocity_init': 0,
    'diffusion_model': 1,
    'vertical_diffusivity': 1e-3,
    'boundary_conditions': True,
    'rebound_damping_coef': 0.5,
    'listing': True,
    'listing_printout_period': 100,
    'output_file': True,
    'output_printout_period': 1,
    'output_rep': 'resu_01',
    'output_file_format': 'txt',
    'output_file_name': 'particles_3d',
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

# particles I/O
part = cylag.ParticlesIO.from_cylag_txt('resu_01/particles_3d.txt')

# timeseries of particle 0
zt = np.asarray(part.zp)
wt = np.asarray(part.wp)
zt0 = zt[:,0]
wt0 = wt[:,0]

# plot settling velocity vs time (to check when equlibrium is reached)
fig, ax = plt.subplots(1, 2, figsize=(6, 3.5))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()
ax[0].plot(part.times, zt0, c=color[0], label="Z")
ax[0].set_xlim([0., 10.])
#ax[0].set_ylim([15., 20.])
ax[0].set_xlabel('$t$ (s)')
ax[0].set_ylabel('$z_p(t)$')
ax[0].grid()
ax[1].plot(part.times, wt0, c=color[1], label="W")
ax[1].set_xlim([0., 10.])
#ax[1].set_ylim([0., 10.])
ax[1].set_xlabel('$t$ (s)')
ax[1].set_ylabel('$w_p(t)$')
ax[1].grid()
plt.savefig("figs/settling_wp_vs_time_lsm1.png", dpi=300, format="png")
plt.show()

################################################################################
#
# Clean:
del fset
del pset
del parameters
del cylag_solver
del part
