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

PRINTOUT = False

class CheckSources(unittest.TestCase):

    def test_2d_source(self):
        root = path.dirname(__file__)
        t2d_file = path.join(root, "data", "r2d_flume.slf")
        bnd_file = path.join(root, "data", "geo_flume.cli")
        # field_set
        fset = cylag.EulerianFieldSet.from_telemac2d(t2d_file, bnd_file=bnd_file)
        # particle set
        pset = cylag.LagrangianParticleSet(\
            None, dim=2, initial_pool_size=1, min_pool_size_update=1)
        # Source 1:
        source1_poly = np.array([[5.1, 0.5], [6.1, 0.5], [6.1, 1.5], [5.1, 1.5]])
        source1_scheduler = cylag.schedules(time_delta=5.)
        source1 = cylag.ParticleSource(source1_scheduler, source1_poly, \
            xy_method=1, xy_npart=4)
        sources = [source1]
        # solver
        parameters = {
            'final_time': 20.,
            'time_step': 0.1,
            'frozen_eulerian_fields': True,
            'particle_velocity_init': 1,
            'model': 1,
            'time_scheme': 1,
            'diffusion_model': 0,
            'horizontal_diffusivity': 3.e-5,
            'vertical_diffusivity': 0.,
            'boundary_conditions': True,
            'boundary_conditions_debug': False,
            'listing': True,
            'listing_printout_period': 1,
            'output_file': True,
            'output_printout_period': 1,
            'output_file_format': 'txt',
            'output_file_name': 'particles_2d',
            'bnd_statistics': False,
            }
        cylag_solver = cylag.Solver(fset, pset, parameters, sources)
        cylag_solver.solve()
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        xpa = np.array([[10.35, 0.75], [10.85, 0.75], [10.35, 1.25],
                        [10.85, 1.25], [ 5.45, 0.75], [ 5.95, 0.75],
                        [ 5.45, 1.25], [ 5.95, 1.25], [15.25, 0.75],
                        [15.75, 0.75], [15.25, 1.25], [15.75, 1.25]])
        upa = np.array([[1., 0.], [1., 0.], [1., 0.],
                        [1., 0.], [1., 0.], [1., 0.],
                        [1., 0.], [1., 0.], [1., 0.],
                        [1., 0.], [1., 0.], [1., 0.]])
        if PRINTOUT:
            print(xp)
            print(up)
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        np.testing.assert_allclose(up, upa, rtol=1e-7)
        # Test IO:
        part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d.txt')
        xp10 = part.xp[100]
        xp20 = part.xp[200]
        xpa10 = [15.35, 15.75, 15.25, 15.75, 10.25, 10.75, 10.25, 10.75]
        xpa20 = [10.35, 10.85, 10.35, 10.85, 5.45, 5.95, 5.45, 5.95, 15.25, 15.75, 15.25, 15.75]
        np.testing.assert_allclose(xp10, xpa10, rtol=1e-7)
        np.testing.assert_allclose(xp20, xpa20, rtol=1e-7)
        # plot
        if PRINTOUT:
            traj = part.get_trajectories([0, 1, 2])
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
            ax.plot(xp[:, 0], xp[:, 1], c='k', lw=0., marker='o', markersize=2)
            if parameters['output_file']:
                for tr in traj:
                    ax.plot(tr[:, 0], tr[:, 1], lw=.5, marker='o', markersize=0.5, c='k')
            #plt.legend()
            plt.show()
            plt.close(fig)
        # clean 
        os.system('rm -r particles')
        del pset
        del fset
        del cylag_solver

if __name__ == "__main__":
    unittest.main()
