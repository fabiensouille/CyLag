#!/usr/bin/env python
# coding: utf-8
"""
Runing tide example in online mode 
===============================================================================

In this example we illustrate how to run CyLag in online mode with openTELEMAC
API (TelAPY).

"""
import os
import argparse
import numpy as np
from os import path, environ
import matplotlib.pyplot as plt
# Import telapy
import telapy.api.t2d
from mpi4py import MPI
# MPI.Init()
from telapy.api.t2d import Telemac2d
# Import Postel functionalities
from data_manip.extraction.telemac_file import TelemacFile
from postel.plot2d import *
# Import cylag
import cylag

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Initialization of TelAPY
# -----------------------------------------------------------------------------
#

# Create an instance of Telemac
cas_file = path.join('t2d_tide-jmj_type.cas')
my_case = Telemac2d(cas_file, lang=1, comm=MPI.COMM_WORLD)
my_case.set_case()
my_case.init_state_default()

################################################################################
#
# Initialization of CyLag
# -----------------------------------------------------------------------------
#
bnd_file = path.join('geo_tide.cli')
geo_file = path.join('geo_tide.slf')
res = TelemacFile(geo_file)

# define cylag mesh
bnd_points, _ = cylag.read_cli(bnd_file)
triangular_mesh = cylag.TriangularMesh(\
    res.tri.x, 
    res.tri.y, 
    res.tri.triangles, 
    boundary_points=bnd_points)

# define with blank hydro state
meshsize = len(res.tri.x)
elevation = np.ones((1, meshsize), dtype='d')
velocity_x = np.zeros((1, meshsize), dtype='d')
velocity_y = np.zeros((1, meshsize), dtype='d')
fset = cylag.EulerianFieldSet(
    triangular_mesh=triangular_mesh,
    dim=2, nlayers=1,
    times=np.zeros(1),
    elevation=elevation,
    velocity_x=velocity_x,
    velocity_y=velocity_y)
del res

# define particles initial positions
gridres = 6
circ = cylag.circle(x0=198000., y0=147000., r=2000., n=360)
ini_position = cylag.init_positions_2d(poly=circ, grid_res=(gridres, gridres))
npart = np.shape(ini_position)[0]
print('Initial number of particles = ', npart)

position = np.copy(ini_position)
pset = cylag.LagrangianParticleSet(position, dim=2, initial_pool_size=npart)

################################################################################
#
# Setup main simulation parameters
# -----------------------------------------------------------------------------
#

# Get the number of time steps.
final_time = my_case.get('MODEL.TMAX')
nite = my_case.get('MODEL.NTIMESTEPS')
telemac_dt = my_case.get('MODEL.DT')

# set general parameters for the traking algo
parameters = {
    'initial_time': 0.,
    'final_time': nite*telemac_dt,
    'time_step': telemac_dt, 
    'model': 1,
    'time_scheme': 1,
    'frozen_eulerian_fields': True,
    'listing': False,
    'output_file': True,
    'output_printout_period': 10,
    'output_file_format': 'txt',
    'output_file_name': 'particles_2d',
    }

# cylag solver
cylag_solver = cylag.Solver(fset, pset, parameters)

################################################################################
#
# Run simulation in online mode
# -----------------------------------------------------------------------------
#

for i in range(nite):
    # FORWARD TELEMAC
    # ~~~~~~~~~~~~~~~
    my_case.run_one_time_step()

    # get hydro state
    zb = my_case.get_array('MODEL.BOTTOMELEVATION')
    h = my_case.get_array('MODEL.WATERDEPTH')
    u = my_case.get_array('MODEL.VELOCITYU')
    v = my_case.get_array('MODEL.VELOCITYV')
    elevation = h - zb

    # FORWARD CYLAG
    # ~~~~~~~~~~~~~
    cylag_solver.field_set._tmp_elevation[:] = elevation[:]
    cylag_solver.field_set._tmp_velocity_x[:] = u[:]
    cylag_solver.field_set._tmp_velocity_y[:] = v[:]

    # increment tracking algo
    cylag_solver.forward(telemac_dt)

################################################################################
#
# Delete instances
# -----------------------------------------------------------------------------
#
my_case.finalize()
del my_case


################################################################################
#
# Post-processing
# -----------------------------------------------------------------------------
#

################################################################################
#
# Load telemac result file :
#
res_file = path.join('r2d_tide-jmj_type.slf')
res = TelemacFile(res_file)
h = res.get_data_value('WATER DEPTH', -1)
ux = res.get_data_value('VELOCITY U', -1)
uy = res.get_data_value('VELOCITY V', -1)
velocity = np.sqrt(ux**2 + uy**2)

################################################################################
#
# Load particles :
#
part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d.txt')
pos_ini = np.asarray([[part.xp[0][i], part.yp[0][i]] for i in range(part.npart[0])])
pos_fin = np.asarray([[part.xp[-1][i], part.yp[-1][i]] for i in range(part.npart[-1])])

# load particles trajectories
traj = part.get_trajectories(range(part.npart[0]))

################################################################################
#
# Plot result :
#
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_aspect('equal')
plot2d_scalar_map(fig, ax, res.tri, velocity, data_name='Velocity (m/s)', alpha = .75, cmap_name='RdBu_r')
plot2d_triangle_mesh(ax, res.tri, x_label='X (m)', y_label='Y (m)', color='k', linewidth=0.1)

# plot trajectories
for tr in traj:
    ax.plot(tr[:, 0], tr[:, 1], lw=0.35, marker='o', markersize=0, c="k")

# plot particules positions
ax.plot(
    np.array(pos_ini)[:, 0],
    np.array(pos_ini)[:, 1],
    c='b', lw=0., marker='o', markersize=2.5, label="Initial positions")
ax.plot(
    np.array(pos_fin)[:, 0],
    np.array(pos_fin)[:, 1],
    c='r', lw=0., marker='o', markersize=2.5, label="Final positions")
    
plt.show()

################################################################################
#
# Clean:
del fset
del pset
del parameters
del cylag_solver
del part
