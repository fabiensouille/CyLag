# -*- coding: utf-8 -*-
import unittest
import time
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation
import matplotlib.cm as cm

PRINTOUT = False

class CheckSolver(unittest.TestCase):

    def test_solver_2d(self):
        root = path.dirname(__file__)
        t2d_file = path.join(root, "data", "r2d_test.slf")
        bnd_file = path.join(root, "data", "geo_test.cli")
        # optional fields
        opt_fields = ['WATER DEPTH']
        # field_set
        fset = cylag.EulerianFieldSet.from_telemac2d(\
            t2d_file, bnd_file=bnd_file, optional_fields=opt_fields)
        # particle set
        npart = 25
        res = int(np.sqrt(npart))
        circ = cylag.circle(x0=0., y0=500., r=50., n=50)
        position = cylag.init_positions_2d(poly=circ, grid_res=(res, res))
        #position = np.array([[0., 500.], [0., 450.], [-100., 550.]])
        #velocity = np.array([[0., 0.], [0., 0.], [0., 0.]])
        pset = cylag.LagrangianParticleSet(\
            position, dim=2, initial_pool_size=10000)
        # solver
        parameters = {
            'final_time': 4000.,
            'time_step': 10.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'diffusion_model': 0,
            'listing': True,
            'listing_printout_period': 20,
            'output_file': PRINTOUT,
            'output_printout_period': 10,
            'output_file_format': 'txt',
            'output_file_name': 'particles_2d',
            }
        cylag_solver = cylag.Solver(fset, pset, parameters)
        cylag_solver.solve()
        # particles_io class for post-processing
        if parameters['output_file']:
            part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d.txt')
            traj = part.get_trajectories([0, 1, 2])
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(14, 3.5))
            ax.set_aspect('equal')
            tri = Triangulation(fset.triangular_mesh.x,\
                                fset.triangular_mesh.y,\
                                fset.triangular_mesh.triangles)
            # plot velocity
            u = fset.get_field(1, parameters['final_time'], layer=-1)
            v = fset.get_field(2, parameters['final_time'], layer=-1)
            velocity = np.sqrt(u**2 + v**2)
            levels = np.linspace(0., np.max(velocity), 20)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.80)
            fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
            ax.triplot(tri, lw=0.25, color='0.75')
            # plot particles
            ax.plot(position[:, 0], position[:, 1], c='b', lw=0., marker='o', markersize=1)
            ax.plot(xp[:, 0], xp[:, 1], c='k', lw=0., marker='o', markersize=2)
            if parameters['output_file']:
                for tr in traj:
                    ax.plot(tr[:, 0], tr[:, 1], lw=.5, marker='o', markersize=0.5, c='k')
            #plt.legend()
            plt.show()
            plt.close(fig)
        # clean 
        del pset
        del fset
        del cylag_solver

    def test_solver_3d(self):
        root = path.dirname(__file__)
        t3d_file = path.join(root, "data", "r3d_test.slf")
        bnd_file = path.join(root, "data", "geo_test.cli")
        # field_set
        fset = cylag.EulerianFieldSet.from_telemac3d(\
            t3d_file, bnd_file=bnd_file)
        # particle set
        npart = 25
        res = int(np.sqrt(npart))
        circ = cylag.circle(x0=0., y0=500., r=50., n=50)
        position = cylag.init_positions_3d(\
            poly=circ, grid_res=(res, res, 10), zmin=259., zmax=261.)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=3, initial_pool_size=10000)
        # solver
        parameters = {
            'final_time': 2520.,
            'time_step': 10.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'diffusion_model': 0,
            'boundary_conditions': True,
            'listing': True,
            'listing_printout_period': 1,
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
            traj = part.get_trajectories([0, 1, 2])
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(14, 3.5))
            ax.set_aspect('equal')
            tri = Triangulation(fset.triangular_mesh.x,\
                                fset.triangular_mesh.y,\
                                fset.triangular_mesh.triangles)
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=fset.nlayers-1)
            v = fset.get_field(2, parameters['final_time'], layer=fset.nlayers-1)
            velocity = np.sqrt(u**2 + v**2)
            levels = np.linspace(0., np.max(velocity), 20)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.80)
            fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
            ax.triplot(tri, lw=0.25, color='0.75')
            # plot particles
            ax.plot(position[:, 0], position[:, 1], c='b', lw=0., marker='o', markersize=1)
            ax.plot(xp[:, 0], xp[:, 1], c='k', lw=0., marker='o', markersize=1)
            if parameters['output_file']:
                for tr in traj:
                    ax.plot(tr[:, 0], tr[:, 1], lw=.5, marker='o', markersize=0.5, c='k')
            #plt.legend()
            plt.show()
            plt.close(fig)
        # clean 
        del pset
        del fset
        del cylag_solver

if __name__ == "__main__":
    unittest.main()
