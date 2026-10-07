# -*- coding: utf-8 -*-
"""
Oil slick advection-diffusion in pseudo-3D
===============================================================================

In this example we compute the advection-diffusion of oil with the oil slick module of CyLag.
We use the LSM-1 model.

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

#sphinx_gallery_thumbnail_number = 2

################################################################################
#
# Define EulerianFieldSet
# -----------------------------------------------------------------------------
#
# The oil slick module requires to load the wind field as an optional field in EulerianFieldSet.
# Also the friction model and friction coefficient used in the Eulerian simulation
# need to be specified so that current velocity correction can be computed.

file_name = os.path.join('data', 'r2d_tide-jmj_type_wind.slf')
bnd_file = os.path.join('data', 'geo_tide.cli')
opt_fields = ['BOTTOM', 'WIND X', 'WIND Y']

fset = cylag.EulerianFieldSet.from_telemac2d(
    file_name, 
    bnd_file=bnd_file,
    optional_fields=opt_fields,
    friction_model=1,
    friction_coefficient=50.)

print("Final time in t2d result file : ", np.asarray(fset.times)[-1])
print("List of fields     : ", fset.get_field_list()[0])
print("List of fields IDs : ", fset.get_field_list()[1])

################################################################################
#
# Define LagrangianParticleSet
# -----------------------------------------------------------------------------
#
# With the ``OilslickParticleSet`` 
# particles are advected horizontally by the combined effect of current drift,
# wind drift, wave drift (Stokes). The velocity of each drift component
# account for the particle immersion in the water column (vertical velocity profile). 
# The vertical displacment of particles is driven by vertical dispersion and buoyancy.
# We consider a polydisperse oil slick defined with a Weibull distribution.

# here we define a single layer of particleq=s on the vertical axis, located in z=8.5
#circ = cylag.circle(x0=204000., y0=148000., r=800., n=360)
circ = cylag.circle(x0=198000., y0=147000., r=1500., n=360)

ini_position = cylag.init_positions_3d(
    poly=circ, 
    grid_res=(75, 75),
    z_method=1,
    z_npart=1,
    zmin=101.,
    zmax=102.)
    
npart = np.shape(ini_position)[0]

pset = cylag.OilslickParticleSet(\
    ini_position, 
    dim=3,
    particle_density=995.,
    particle_diameter=10.e-3,
    particle_diameter_distribution=["weibull", 1.8],
    compute_depth=True,
    wind_coefficient=0.025,
    wind_attenuation_coef=0.5,
    wind_deviation_calibration_coef=1.,
    stokes_drift=True,
    wave_induced_vertical_dispersion=True)
    
################################################################################
#
# Set general parameters
# -----------------------------------------------------------------------------
#
parameters = {
    'initial_time': 3600.,
    'final_time': 6.*3600.,
    'time_step': 25.,
    'model': 1,
    'time_scheme': 1,
    'frozen_eulerian_fields': False,
    'diffusion_model': 1,
    'horizontal_diffusivity': 1.e-2,
    'vertical_diffusivity': 1.e-2,
    'water_density': 1025.,
    'rebound_damping_coef': 0.25,
    'stranding_probability': 0.5,
    'listing': True,
    'listing_printout_period': 100,
    'output_file': True,
    'output_printout_period': 10,
    'output_file_format': 'txt',
    'output_rep': 'resu_02',
    'output_file_name': 'particles_2d',
    'stranding_output': True,
    'stranding_output_file_name': 'stranded_particles',
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
part = cylag.ParticlesIO.from_cylag_txt('resu_02/particles_2d.txt')

# recorded times :
print("list of recorded times =", part.times)
print("number of records = ", len(part.times))

max_rec = len(part.times) - 1
rec_list = [max_rec//2, max_rec]

print(rec_list)

for idx, rec in enumerate(rec_list):

    time = part.times[rec]

    # Field set from telemac results :
    tri = fset.get_mtri_triangulation()
    ux = fset.get_field(1, time)
    uy = fset.get_field(2, time)
    velocity = np.sqrt(ux**2 + uy**2)
    wind_ux = fset.get_field(5, time)
    wind_uy = fset.get_field(6, time)
    wind_vel = np.sqrt(wind_ux**2 + wind_uy**2)

    # Plot result :
    cylag.set_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=(10., 7))
    ax.set_aspect('equal')

    # plot mesh and velocity
    levels = np.linspace(0., np.max(velocity)+0.1, 20)
    cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.80)
    fig.colorbar(cs, ax=ax, label="$U_f$ (m/s)")
    ax.triplot(tri, lw=0.1, c='k')
    img = ax.quiver(tri.x, tri.y, ux, uy, velocity, cmap='RdBu_r')
    #img = ax.quiver(tri.x, tri.y, wind_ux, wind_uy, wind_vel, cmap='magma')

    # plot final positions
    bounds = [0., 0.5, 1., 1.5, 2.]
    cmap_disc = cmap_disc = ListedColormap(
        plt.cm.bone(np.linspace(0., 0.8, len(bounds)-1)))
    norm_disc = BoundaryNorm(bounds, cmap_disc.N)
    sc = ax.scatter(part.xp[rec][:], part.yp[rec][:], c=part.depth[rec][:], 
        lw=0., s=2.5, cmap=cmap_disc, norm=norm_disc, label="$x_p(t_f)$")
    fig.colorbar(sc, ax=ax, label="Particle depth (m)", ticks=bounds, extend="max")

    # plot stranded particles
    if idx==len(rec_list)-1:
        stranded_particles = np.loadtxt('resu_02/stranded_particles.txt', skiprows=2, delimiter=",")
        ax.plot(stranded_particles[:, 0], stranded_particles[:, 1], c='r', lw=0., marker='o', markersize=3., label="Stranded")

    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.legend()
    plt.savefig('figs/cylag_tide_lsm1_{}.png'.format(idx), dpi=300, format="png")
    plt.show()

################################################################################
#
# Clean:
del fset
del pset
del parameters
del cylag_solver
del part
