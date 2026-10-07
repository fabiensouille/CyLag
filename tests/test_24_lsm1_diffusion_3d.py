# -*- coding: utf-8 -*-
import unittest
import time
import copy
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

PRINTOUT = False

def fset_3d():
    # load mesh
    root = path.dirname(__file__)
    meshx = np.loadtxt(path.join(root, "data", "square_meshx.txt"))
    meshy = np.loadtxt(path.join(root, "data", "square_meshy.txt"))
    triangles = np.loadtxt(path.join(root, "data", "square_meshtri.txt"), dtype='int32')
    bnd_file = path.join(root, "data", "square_geo.cli")
    # matplotlib triangulation
    tri = Triangulation(meshx, meshy, triangles)
    trifinder = tri.get_trifinder()
    # read boundaries
    bnd_points, _ = cylag.read_cli(bnd_file)
    cylagtri = cylag.TriangularMesh(meshx, meshy, triangles, boundary_points=bnd_points)
    # field_set
    nlayers = 11
    meshsize = len(tri.x)
    elevation = np.zeros((1, meshsize*nlayers), dtype='d')
    velocity_x = np.zeros((1, meshsize*nlayers), dtype='d')
    velocity_y = np.zeros((1, meshsize*nlayers), dtype='d')
    velocity_z = np.zeros((1, meshsize*nlayers), dtype='d')
    zs = 1.
    zb = 0.
    size = len(meshx)
    dz = (zs - zb)/max((nlayers-1), 1)
    zlayers = np.array([dz*i for i in range(nlayers)])
    for i in range(nlayers):
        elevation[0, i*size:(i+1)*size] = zlayers[i]
    fset = cylag.EulerianFieldSet(
        triangular_mesh=cylagtri,
        dim=3, nlayers=nlayers,
        times=np.zeros(1),
        elevation=elevation,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
        trifinder=trifinder)
    return fset, tri

class CheckSolverAnalytical(unittest.TestCase):

    def test_solver_3d_lsm1_advection_euler(self):
        # fset
        fset, tri = fset_3d()
        # particle set
        npart = 50
        res = int(np.sqrt(npart))
        circ = cylag.circle(x0=0., y0=0., r=1000., n=50)
        position = cylag.init_positions_3d(\
            poly=circ, grid_res=(res, res, 6), zmin=0.1, zmax=0.9)
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=3, initial_pool_size=position.shape[0])
        # cylag solver
        parameters = {
            'final_time': 1000.,
            'time_step': 10.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 1,
            'diffusion_model': 1,
            'horizontal_diffusivity': 500.,
            'vertical_diffusivity': 1.e-6,
            'listing': True,
            'output_file': PRINTOUT,
            'output_printout_period': 1,
            'output_file_format': 'txt',
            'output_file_name': 'particles_3d',
            }
        cylag_solver = cylag.Solver(fset, pset, parameters)
        cylag_solver.solve()
        # particles_io class for post-processing
        if parameters['output_file']:
            part = cylag.ParticlesIO.from_cylag_txt('particles/particles_3d.txt')
            traj = part.get_trajectories([0,1,2,3])
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()    
        # check result
        # plot
        if PRINTOUT:
            print(xp)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            ax.triplot(tri, lw=0.25, color='0.75')
            # plot particles
            ax.plot(apos[:, 0], apos[:, 1], c='r', lw=0., marker='o', markersize=3, label="Analytic")
            ax.plot(xp[:, 0], xp[:, 1], c='k', lw=0., marker='o', markersize=2, label="Final positions")
            if parameters['output_file']:
                for tr in traj:
                    ax.plot(tr[:, 0], tr[:, 1], lw=1., marker='o', markersize=1, c='k')
            plt.legend()
            plt.show()
            plt.close(fig)
        # clean 
        del tri
        del fset
        del pset
        del cylag_solver

if __name__ == "__main__":
    unittest.main()
