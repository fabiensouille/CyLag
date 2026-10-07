# -*- coding: utf-8 -*-
"""
Settling comparison without diffusion
===============================================================================

In this example we compute the settling of particle in a simple 3D domain
with all LSM models without diffusion.

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
# Define run case function
# -----------------------------------------------------------------------------
#
def run_case(lsm, nu_z):
    # Define EulerianFieldSet
    fset = fset_3d()

    # Define LagrangianParticleSet
    position = np.asarray([\
        [5.1, 5.1, 19.5], [2.1, 2.1, 19.5]])

    velocity = np.asarray([\
        [0.0, 0.0, -0.01], [0.0, 0.0, -0.01]])

    # here we select the 
    pset = cylag.LagrangianParticleSet(\
        position, 
        velocity, 
        dim=3, 
        initial_pool_size=2,
        particle_density=2500., 
        particle_diameter=10.e-3,
        added_mass_force=True, 
        added_mass_coef=0.5,
        drag_coefficient_model=1,
        drag_coefficient=0.44,
        buoyancy_velocity_model=1)

    if lsm==1:
        sc = 1
    else:
        sc = 5

    # Set general parameters
    parameters = {
        'final_time': 10.,
        'time_step': 0.005,
        'frozen_eulerian_fields': True,
        'model': lsm,
        'time_scheme': sc,
        'particle_velocity_init': 0,
        'diffusion_model': 1,
        'diffusion_lsm2_option': 3,
        'diffusion_lsm3_tl_vertical': 1.e-2,
        'horizontal_diffusivity': 0.,
        'vertical_diffusivity': nu_z,
        'boundary_conditions': False,
        'rebound_damping_coef': 0.5,
        'listing': True,
        'listing_printout_period': 100,
        'output_file': True,
        'output_printout_period': 1,
        'output_rep': 'resu_{0:02d}'.format(lsm),
        'output_file_format': 'txt',
        'output_file_name': 'particles_3d',
        }

    # Run cylag
    cylag_solver = cylag.Solver(fset, pset, parameters)
    cylag_solver.solve()


################################################################################
#
# Run different LSM cases
# -----------------------------------------------------------------------------

nu_z = 0.
plt_marker = True

# Run
run_case(lsm=1, nu_z=nu_z)
run_case(lsm=2, nu_z=nu_z)
run_case(lsm=3, nu_z=nu_z)

# plot
cylag.set_rcparams()
fig, ax = plt.subplots(1, 2, figsize=(6, 3.))
color,_,_ = cylag.get_default_color_palette()
line_styles = ['-', '--', '-.']
markers = ['o', 's', '^']

for i in range(1, 4):

    label = "LSM-{0}".format(i)

    part = cylag.ParticlesIO.from_cylag_txt('resu_{0:02d}/particles_3d.txt'.format(i))

    # timeseries of particle 0
    zt = np.asarray(part.zp)
    wt = np.asarray(part.wp)
    zt0 = zt[:,0]
    wt0 = wt[:,0]

    if not plt_marker:
        markers = [None, None, None]

    # plot settling velocity vs time (to check when equlibrium is reached)
    ax[0].plot(part.times, zt0, c=color[i-1], label=label, ls=line_styles[i-1], 
               marker=markers[i-1], markersize=4, markevery=10)
    ax[0].set_xlim([0., 1.])
    ax[0].set_ylim([18.75, 19.75])
    ax[0].set_xlabel('$t$ (s)')
    ax[0].set_ylabel('$z_p(t)$ (m)')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(part.times, wt0, c=color[i-1], label=label, ls=line_styles[i-1], 
               marker=markers[i-1], markersize=4, markevery=10)
    ax[1].set_xlim([0., 1.])
    ax[1].set_ylim([-0.7, 0.])
    ax[1].set_xlabel('$t$ (s)')
    ax[1].set_ylabel('$w_p(t)$ (m/s)')
    ax[1].grid()


w_still = -np.sqrt(4.*10.e-3*9.80665*(2500./1000. - 1.)/(3.*0.44))
ax[1].axhline(w_still, c='k', lw=0.8, ls='-', label='Analytical')
ax[1].legend()

plt.tight_layout()
#plt.savefig("figs/settling_wp_vs_time_lsm_comparison.png", dpi=300, format="png")
plt.savefig("figs/settling_wp_vs_time_lsm_comparison.pdf", dpi=300, format="pdf")
plt.show()
