# -*- coding: utf-8 -*-
import unittest
import time
import os
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation
import matplotlib.cm as cm

PRINTOUT = False

class CheckSolverParallel(unittest.TestCase):

    def test_solver_2d_seq_vs_par(self):
        root = path.dirname(__file__)
        t2d_file = path.join(root, "data", "r2d_test.slf")
        bnd_file = path.join(root, "data", "geo_test.cli")
        # optional fields
        opt_fields = ['WATER DEPTH']
        # field_set
        fset = cylag.EulerianFieldSet.from_telemac2d(\
            t2d_file, bnd_file=bnd_file, optional_fields=opt_fields)
        # particles
        npart = 25
        res = int(np.sqrt(npart))
        circ = cylag.circle(x0=0., y0=500., r=50., n=50)
        pos = cylag.init_positions_2d(poly=circ, grid_res=(res, res))
        # main params
        parameters = {
            'final_time': 1000.,
            'time_step': 10.,
            'model': 1,
            'time_scheme': 2,
            'particle_velocity_init': 1,
            'frozen_eulerian_fields': True,
            'diffusion_model': 0,
            'listing': True,
            'listing_printout_period': 20,
            'output_file': True,
            'output_printout_period': 10,
            'output_file_format': 'txt',
            }
        # seq solver
        pset = cylag.LagrangianParticleSet(\
            pos, dim=2, initial_pool_size=10000)
        parameters.update({'output_file_name': 'particles_2d_seq',})
        cylag_solver_seq = cylag.Solver(fset, pset, parameters)
        cylag_solver_seq.solve()
        # par solver
        pset = cylag.LagrangianParticleSet(\
            pos, dim=2, initial_pool_size=10000)
        parameters.update({'output_file_name': 'particles_2d_par',})
        cylag_solver_par = cylag.SolverParallel(fset, pset, parameters)
        cylag_solver_par.solve()
        # comparison seq vs par
        part_seq = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d_seq.txt')
        part_par = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d_par.txt')
        position_seq = part_seq.xp[-1]
        velocity_seq = part_seq.up[-1]
        position_par = part_par.xp[-1]
        velocity_par = part_par.up[-1]
        np.testing.assert_allclose(position_seq, position_par, rtol=1e-7)
        np.testing.assert_allclose(velocity_seq, velocity_par, rtol=1e-7)
        del part_seq
        del part_par
        os.system('rm -r particles')
        # get particles last position
        xp, up, us = cylag_solver_par.particle_set.get_state()
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
            ax.plot(pos[:, 0], pos[:, 1], c='b', lw=0., marker='o', markersize=1)
            ax.plot(xp[:, 0], xp[:, 1], c='k', lw=0., marker='o', markersize=2)
            #plt.legend()
            plt.show()
            plt.close(fig)
        # clean
        del fset
        del pset
        del cylag_solver_seq
        del cylag_solver_par

if __name__ == "__main__":
    unittest.main()
