# -*- coding: utf-8 -*-
"""
Statistics
===============================================================================

In this example we analyse particles going through control sections.

"""
import cylag
import os
import numpy as np
import matplotlib.pylab as plt

#sphinx_gallery_thumbnail_number = 2

################################################################################
#
# Define EulerianFieldSet
# -----------------------------------------------------------------------------
#
file_name = os.path.join('data', 'r2d_tide-jmj_type.slf')
bnd_file = os.path.join('data', 'geo_tide.cli')
fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)

print("Final time in t2d result file : ", np.asarray(fset.times)[-1])

################################################################################
#
# Define LagrangianParticleSet
# -----------------------------------------------------------------------------
#
circ = cylag.circle(x0=198000., y0=147000., r=2000., n=360)
ini_position = cylag.init_positions_2d(poly=circ, grid_res=(10, 10))

npart = np.shape(ini_position)[0]

print('Initial number of particles = ', npart)

pset = cylag.LagrangianParticleSet(
    ini_position, 
    dim=2, 
    initial_pool_size=npart)

################################################################################
#
# Define control sections
# -----------------------------------------------------------------------------
# 
control_poly1 = np.array([\
    [192500., 145000.], [193500., 155000.]])
    
control_poly2 = np.array([\
    [192000., 145000.], [191500., 148000.],\
    [191000., 151000.], [191000., 158000.]])
    
control_section1 = cylag.ControlSection(control_poly1)
control_section2 = cylag.ControlSection(control_poly2)

control_sections = [control_section1, control_section2]

################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
# 
parameters = {
    'initial_time': 7200.,
    'final_time': 21600.,
    'time_step': 10.,
    'model': 1,
    'time_scheme': 1,
    'frozen_eulerian_fields': False,
    'diffusion_model': 1,
    'listing': True,
    'listing_printout_period': 100,
    'output_file': True,
    'output_printout_period': 100,
    'output_file_format': 'txt',
    'output_file_name': 'particles_2d',
    'bnd_statistics': True,
    }

################################################################################
#
# Run CyLag
# -----------------------------------------------------------------------------
# 
cylag_solver = cylag.Solver(\
    fset, 
    pset, 
    parameters,
    control_sections=control_sections)

cylag_solver.solve()


################################################################################
#
# Post-processing : plot 2D results
# -----------------------------------------------------------------------------

# particles I/O
part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d.txt')
rec = len(part.times) - 1
time = part.times[rec]
tags = part.tags[rec]
traj = part.get_trajectories(tags)
npart = len(tags)

# field set from telemac results
file_name = os.path.join('data', 'r2d_tide-jmj_type.slf')
bnd_file = os.path.join('data', 'geo_tide.cli')
fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)
tri = fset.get_mtri_triangulation()

# get velocity at final time
ux = fset.get_field(1, time)
uy = fset.get_field(2, time)
velocity = np.sqrt(ux**2 + uy**2)

# plot result
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.set_aspect('equal')
cylag.set_rcparams()

# plot mesh and velocity
levels = np.linspace(0., np.max(velocity), 40)
cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.80)
fig.colorbar(cs, ax=ax, label="$U_f$ (m/s)")
ax.triplot(tri, lw=0.1, c='k')

# plot trajectories
for tr in traj:
    ax.plot(tr[0:rec+1, 0], tr[0:rec+1, 1], lw=0.5, marker='o', markersize=0., c="k")

# plot ini positions
ax.plot(part.xp[0][:], part.yp[0][:], c='k', lw=0., marker='o', markersize=1., label="$x_p(0)$")

# plot final positions
ax.plot(part.xp[rec][:], part.yp[rec][:], c='b', lw=0., marker='o', markersize=1., label="$x_p(t_f)$")

# plot control polygon
control_poly1 = np.array([\
    [192500., 145000.], [193500., 155000.]])
control_poly2 = np.array([\
    [192000., 145000.], [191500., 148000.],\
    [191000., 151000.], [191000., 158000.]])
ax.plot(control_poly1[:, 0], control_poly1[:, 1], ls='-', color='g', lw=1.5)
ax.plot(control_poly2[:, 0], control_poly2[:, 1], ls='-', color='r', lw=1.5)

plt.legend()
plt.savefig('figs/cylag_tide_stats.png', dpi=300, format="png")
plt.show()

################################################################################
#
# Post-processing : plot control section statistics
# -----------------------------------------------------------------------------
#

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
c0, c1, c2 = cylag.get_default_color_palette()
cylag.set_rcparams()

nsec = 2
for i in range(nsec):
    # load stats file
    data = np.loadtxt("particles/cs_stats_{}.txt".format(i+1), delimiter=',', skiprows=2)
    times = data[:, 0]
    npart = data[:, 1]

    # other statistics
    cumul = np.cumsum(npart)

    ax[0].plot(times, npart, marker='+', color=c0[i], lw=0.5, label='Control section n°{}'.format(i))
    ax[1].plot(times, cumul, marker='', color=c0[i], lw=1.5, label='Control section n°{}'.format(i))

ax[0].grid()
ax[0].set_xlabel("$t$ (s)")
ax[0].set_ylabel("$N$")
ax[0].legend()
ax[1].grid()
ax[1].set_xlabel("$t$ (s)")
ax[1].set_ylabel("$N_{cumul}$")
ax[1].legend()

plt.savefig("figs/cylag_csec_stats.png", dpi=300, format="png")
plt.show()

################################################################################
# 
# .. note::
#   The same information can be extracted from boundaries using the option
#   ``'bnd_statistics': True``.

################################################################################
#
# Clean:
del fset
del pset
del parameters
del cylag_solver
del part
del rec
