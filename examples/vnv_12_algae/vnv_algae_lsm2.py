# -*- coding: utf-8 -*-
"""
Algae - LSM-2 advection-diffusion
===============================================================================

In this example we compute the advection-diffusion of algae in a meandering 
channel with the LSM-2 model

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
file_name = os.path.join('data', 'r2d_meander.slf')
bnd_file = os.path.join('data', 'geo_meander.cli')

fset = cylag.EulerianFieldSet.from_telemac2d(
    file_name,
    bnd_file=bnd_file, 
    optional_fields=['TURBULENT ENERG.', 'DISSIPATION'])

print("fset times = ", np.asarray(fset.times))

################################################################################
#
# Define AlgaeParticleSet
# -----------------------------------------------------------------------------
#
# The ``AlgaeParticleSet`` module uses a custom drag force function.
#
# The drag force is expressed as follows:
#
# .. math::
#   F_d = \dfrac{1}{2} \rho_f S_p C_d |U|U
#
# Different models can be selected for the :math:`C_d` coefficient 
# via the ``drag_coefficient_model`` parameter:
#
#  - 0: Constant drag model, Cd=cst
#  - 1: Schiller et Nauman, (1935),
#  - 2: Almedeij (2008),
#  - 3: Algae formula: Cd = exp(self.Cd_a - self.Cd_b*log(Re)) 
#
# When the Algae formula is selected the coefficients of the law
# can be adjusted to fit different species, such as:
#
#  - 1: Iridaea Flaccida : Cd_a=6.822121, Cd_b=0.800627, 
#  - 2: Pelvetiopsis Limitata : Cd_a=8.214783, Cd_b=0.877036,
#  - 3: Gigartina Leptorhynchos : Cd_a=6.773712, Cd_b=0.774252.
#
poly = np.array([[-6.085,-0.270],\
                 [-5.800,-0.750],\
                 [-4.958,-0.260],\
                 [-5.235,0.220]])

position = cylag.init_positions_2d(poly=poly, grid_res=(50, 50))
npart = position.shape[0]

print('Number of particles = ', npart)

pset = cylag.AlgaeParticleSet(\
    position, 
    dim=2,
    particle_density=1000.,
    particle_diameter=0.01,
    drag_coefficient_model=3, 
    added_mass_force=True,
    added_mass_coef=0.5,
    Cd_a=6.822121,
    Cd_b=0.800627)

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
# 
parameters = {
    'final_time': 85.,
    'time_step': 0.1,
    'frozen_eulerian_fields': True,
    'model': 2,
    'time_scheme': 5,
    'particle_velocity_init': 1,
    'diffusion_model': 2,
    'diffusion_lsm2_option': 1,
    'schmidt_number': 0.72,
    'listing': True,
    'listing_printout_period': 100,
    'rebound_damping_coef': 0.,
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

rec = len(part.times) - 1
time = part.times[rec]

################################################################################
#
# Field set from telemac results :
file_name = os.path.join('data', 'r2d_meander.slf')
bnd_file = os.path.join('data', 'geo_meander.cli')
fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)
tri = fset.get_mtri_triangulation()

# get velocity at final time
ux = fset.get_field(1, time)
uy = fset.get_field(2, time)
velocity = np.sqrt(ux**2 + uy**2)

################################################################################
#
# Plot result :
fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()
ax.set_aspect('equal')

# plot mesh and velocity
levels = np.linspace(0.15, 0.25, 11)
cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.80)
fig.colorbar(cs, ax=ax, label="$\\textbf{U}_f$ (m/s)")
ax.triplot(tri, lw=0.12, c='grey')

# plot ini positions
ax.plot(part.xp[0][:], part.yp[0][:], c='b', lw=0., marker='o', markersize=1., label="$x_p(0)$")

# plot final positions
i1 = int(1.*rec/4.)
i2 = int(2.*rec/4.)
i3 = int(3.*rec/4.)
ax.plot(part.xp[i1][:], part.yp[i1][:], c=color[4], lw=0., marker='o', markersize=1., label="$x_p(t_f/4)$")
ax.plot(part.xp[i2][:], part.yp[i2][:], c=color[1], lw=0., marker='o', markersize=1., label="$x_p(2t_f/4)$")
ax.plot(part.xp[i3][:], part.yp[i3][:], c=color[2], lw=0., marker='o', markersize=1., label="$x_p(3t_f/4)$")
ax.plot(part.xp[rec][:], part.yp[rec][:], c=color[0], lw=0., marker='o', markersize=1., label="$x_p(t_f)$")

plt.xlabel("$x$ (m)")
plt.ylabel("$y$ (m)")
plt.legend()
plt.savefig("figs/cylag_algae_lsm2.png", dpi=300, format="png")
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
