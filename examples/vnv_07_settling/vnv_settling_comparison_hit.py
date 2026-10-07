# -*- coding: utf-8 -*-
"""
Settling comparison in weak homogeneous isotropic turbulence
===============================================================================

Compare LSM-1, LSM-2 and LSM-3 in a zero-mean-flow water column with prescribed,
spatially uniform, stationary turbulence. This is an idealized maintained HIT
case away from walls.
The same long-time, constant-drag diffusivity is used in all three models.
"""
import cylag
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
from cylag.lsm.random_utils import pcg32_seed

sys.path.insert(0, os.path.abspath("../../examples/vnv_07_settling"))
from vnv_07_utils.custom_field_set import fset_3d

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Prescribe a weak HIT case and derive equivalent constant coefficients
# -----------------------------------------------------------------------------
C_MU = 0.09
C0 = 1.2
U_RMS = 0.1                  # m/s, nominal Eulerian component RMS
TURBULENCE_LENGTH = 0.01      # m

def hit_coefficients(u_rms=U_RMS, length_scale=TURBULENCE_LENGTH):
    """Return uniform (k, epsilon, T_L, K) using CyLag's Langevin closure."""
    k = 1.5*u_rms**2
    epsilon = C_MU**0.75*k**1.5/length_scale
    tl = (k/epsilon)/(0.5 + 0.75*C0)
    diffusivity = 0.5*C0*epsilon*tl**2
    return k, epsilon, tl, diffusivity

K_HIT, EPS_HIT, TL_HIT, NU_HIT = hit_coefficients()

################################################################################
#
# Define run case function
# -----------------------------------------------------------------------------
#
def run_case(lsm, nu_z=NU_HIT, tl=TL_HIT):
    """Run an ensemble with common K and T_L in all three directions."""

    fset = fset_3d()

    # Define LagrangianParticleSet
    npart = 50
    position = np.tile([5.1, 5.1, 19.5], (npart, 1))
    velocity = np.tile([0.0, 0.0, 0.0],  (npart, 1))

    # Zero initial seen-fluid fluctuations: LSM-3 develops its stationary
    # distribution over a few T_L. The early-time curves are transients.
    pset = cylag.LagrangianParticleSet(\
        position, 
        velocity, 
        dim=3, 
        initial_pool_size=npart,
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
        'diffusion_lsm3_tl_vertical': tl,
        'horizontal_diffusivity': 0.,
        'vertical_diffusivity': nu_z,
        'rng_method': 2,
        'boundary_conditions': False,
        'listing': True,
        'listing_printout_period': 100,
        'output_file': True,
        'output_printout_period': 1,
        'output_rep': 'resu_{0:02d}'.format(lsm),
        'output_file_format': 'txt',
        'output_file_name': 'particles_3d',
        }

    cylag_solver = cylag.Solver(fset, pset, parameters)
    cylag_solver.solve()


################################################################################
#
# Run different LSM cases
# -----------------------------------------------------------------------------

nu_z = NU_HIT
tl_z = TL_HIT

print("Weak HIT: nominal u_rms={:.3e} m/s, ell={:.3e} m".format(U_RMS, TURBULENCE_LENGTH))
print("k={:.6e} m2/s2, epsilon={:.6e} m2/s3".format(K_HIT, EPS_HIT))
print("T_L={:.6e} s, Kx=Ky=Kz={:.6e} m2/s".format(tl_z, nu_z))
print("LSM-3 stationary RMS(Us_i)={:.6e} m/s; dt/T_L={:.3e}".format(np.sqrt(nu_z/tl_z), 0.005/tl_z))

# Run
run_case(lsm=1, nu_z=nu_z, tl=tl_z)
run_case(lsm=2, nu_z=nu_z, tl=tl_z)
run_case(lsm=3, nu_z=nu_z, tl=tl_z)

# plot
cylag.set_rcparams()
fig, ax = plt.subplots(1, 2, figsize=(6, 3.))
color,_,_ = cylag.get_default_color_palette()
line_styles = ['-', '--', '-.']
markers = ['o', 's', '^']

for i in range(1, 4):

    label = "LSM-{0}".format(i)

    part = cylag.ParticlesIO.from_cylag_txt('resu_{0:02d}/particles_3d.txt'.format(i))

    # Mean +/- one ensemble standard deviation (not a confidence interval).
    # In LSM-1, Wp is the drift velocity, not a derivative of Brownian position.
    zt = np.asarray(part.zp)
    wt = np.asarray(part.wp)
    for axis, values in zip(ax, (zt, wt)):
        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
        axis.plot(part.times, mean, c=color[i-1], label=label,
                  ls=line_styles[i-1])
        axis.fill_between(part.times, mean-std, mean+std,
                          color=color[i-1], alpha=0.12)

# For this 1-cm particle, Schiller-Naumann is in its Cd=0.44 branch at
# equilibrium, also used by LSM-1 here. This is a still-water reference,
# not an imposed equality for nonlinear drag in turbulent flow.
w_still = -np.sqrt(4.*10.e-3*9.80665*(2500./1000. - 1.)/(3.*0.44))
ax[1].axhline(w_still, c='k', lw=0.8, ls='-', label='Analytical')

ax[0].set_ylabel(r'$\langle z_p(t)\rangle$ (m)')
ax[1].set_ylabel(r'$\langle w_p(t)\rangle$ (m/s)')

for axis in ax:
    axis.set_xlim([0., 1.])
    axis.set_xlabel('$t$ (s)')
    axis.grid()
    axis.legend()

ax[0].set_ylim([18.75, 19.75])

#fig.suptitle('Weak HIT: ensemble means and one-standard-deviation bands')

plt.tight_layout()
#plt.savefig("figs/settling_wp_vs_time_lsm_comparison_hit.png", dpi=300, format="png")
plt.savefig("figs/settling_wp_vs_time_lsm_comparison_hit.pdf", dpi=300, format="pdf")
plt.show()
