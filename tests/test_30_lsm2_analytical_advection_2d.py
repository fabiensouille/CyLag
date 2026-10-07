# -*- coding: utf-8 -*-
import unittest
import time
import copy
import os
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

PRINTOUT = False

class radial_velocity_field_2d():
    def __init__(self, omega=2.*np.pi/3600.):
        self.omega = omega

    def get(self, x, y):
        r, phi = np.sqrt(x**2 + y**2), np.arctan2(x, y)
        velx =  self.omega*r*np.cos(phi)
        vely = -self.omega*r*np.sin(phi)
        return velx, vely

    def get_field_on_2dmesh(self, tri):
        meshx = tri.x
        meshy = tri.y
        size = len(meshx)
        velx = np.empty(size)
        vely = np.empty(size)
        velx[:], vely[:] = self.get(meshx[:], meshy[:])
        return velx, vely

def fset_2d():
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
    # set analytical velocity field:
    field = radial_velocity_field_2d()
    velx, vely = field.get_field_on_2dmesh(tri)
    velocity = np.sqrt(velx**2 + vely**2)
    # field_set
    meshsize = len(tri.x)
    elevation = np.ones((1, meshsize), dtype='d')
    velocity_x = np.zeros((1, meshsize), dtype='d')
    velocity_y = np.zeros((1, meshsize), dtype='d')
    velocity_x[0, :] = velx
    velocity_y[0, :] = vely
    fset = cylag.EulerianFieldSet(
        triangular_mesh=cylagtri,
        dim=2, nlayers=1,
        times=np.zeros(1),
        elevation=elevation,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        trifinder=trifinder)
    return fset, tri

class CheckSolverAnalytical(unittest.TestCase):

    def test_solver_2d_lsm2_advection_euler(self):
        # fset and tri
        fset, tri = fset_2d()
        # particle set
        position = np.asarray([\
            [5000., 0.], [5000., 500.], [1000., 0.], [1000., 500.]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=2, initial_pool_size=1000)
        # cylag solver
        parameters = {
            'final_time': 3600.,
            'time_step': 1.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 2,
            'particle_velocity_init': 1,
            'diffusion_model': 0,
            'diffusion_lsm2_option': 1,
            'listing': True,
            'output_file': PRINTOUT,
            'output_printout_period': 1,
            'output_file_format': 'txt',
            'output_file_name': 'particles_2d',
            }
        cylag_solver = cylag.Solver(fset, pset, parameters)
        cylag_solver.solve()
        # particles_io class for post-processing
        if parameters['output_file']:
            part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d.txt')
            traj = part.get_trajectories([0])
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        # check result
        xpa = np.asarray([\
            [5.02240714e+03, 1.59416032e-02],
            [5.02231846e+03, 5.02247697e+02],
            [1.01149457e+03, 3.88291522e-02],
            [1.01090280e+03, 5.05494818e+02]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            print(xp)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=-1)
            v = fset.get_field(2, parameters['final_time'], layer=-1)
            velocity = np.sqrt(u**2 + v**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
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
            os.system('rm -r particles')
        # clean 
        del tri
        del fset
        del pset
        del cylag_solver

    def test_solver_2d_lsm2_advection_rk2(self):
        # fset and tri
        fset, tri = fset_2d()
        # particle set
        position = np.asarray([\
            [5000., 0.], [5000., 500.], [1000., 0.], [1000., 500.]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=2, initial_pool_size=1000)
        # cylag solver
        parameters = {
            'final_time': 3600.,
            'time_step': 1.,
            'time_scheme': 2,
            'frozen_eulerian_fields': True,
            'model': 2,
            'particle_velocity_init': 1,
            'diffusion_model': 0,
            'diffusion_lsm2_option': 1,
            'listing': True,
            'output_file': PRINTOUT,
            'output_printout_period': 1,
            'output_file_format': 'txt',
            'output_file_name': 'particles_2d',
            }
        cylag_solver = cylag.Solver(fset, pset, parameters)
        cylag_solver.solve()
        # particles_io class for post-processing
        if parameters['output_file']:
            part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d.txt')
            traj = part.get_trajectories([0])
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        # check result
        xpa = np.asarray([\
            [5.04996265e+03, 1.42800005e-01],
            [5.04986094e+03, 5.05129760e+02],
            [1.01703735e+03, 8.91474852e-02],
            [1.01641878e+03, 5.08313138e+02]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            print(xp)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=-1)
            v = fset.get_field(2, parameters['final_time'], layer=-1)
            velocity = np.sqrt(u**2 + v**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
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
            os.system('rm -r particles')
        # clean 
        del tri
        del fset
        del pset
        del cylag_solver

    def test_solver_2d_lsm2_advection_rk4(self):
        # fset and tri
        fset, tri = fset_2d()
        # particle set
        position = np.asarray([\
            [5000., 0.], [5000., 500.], [1000., 0.], [1000., 500.]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=2, initial_pool_size=1000)
        # cylag solver
        parameters = {
            'final_time': 3600.,
            'time_step': 1.,
            'time_scheme': 4,
            'frozen_eulerian_fields': True,
            'model': 2,
            'particle_velocity_init': 1,
            'diffusion_model': 0,
            'diffusion_lsm2_option': 1,
            'listing': True,
            'output_file': PRINTOUT,
            'output_printout_period': 1,
            'output_file_format': 'txt',
            'output_file_name': 'particles_2d',
            }
        cylag_solver = cylag.Solver(fset, pset, parameters)
        cylag_solver.solve()
        # particles_io class for post-processing
        if parameters['output_file']:
            part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d.txt')
            traj = part.get_trajectories([0])
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        # check result
        xpa = np.asarray([\
            [5.04996154e+03, 1.58905815e-01],
            [5.04985822e+03, 5.05145755e+02],
            [1.01703715e+03, 9.23901032e-02],
            [1.01641697e+03, 5.08316281e+02]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            print(xp)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=-1)
            v = fset.get_field(2, parameters['final_time'], layer=-1)
            velocity = np.sqrt(u**2 + v**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
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
            os.system('rm -r particles')
        # clean 
        del tri
        del fset
        del pset
        del cylag_solver

    def test_solver_2d_lsm2_advection_di1(self):
        # fset and tri
        fset, tri = fset_2d()
        # particle set
        position = np.asarray([\
            [5000., 0.], [5000., 500.], [1000., 0.], [1000., 500.]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=2, initial_pool_size=1000)
        # cylag solver
        parameters = {
            'final_time': 3600.,
            'time_step': 20.,
            'time_scheme': 5,
            'frozen_eulerian_fields': True,
            'model': 2,
            'particle_velocity_init': 1,
            'diffusion_model': 0,
            'diffusion_lsm2_option': 1,
            'listing': True,
            'output_file': PRINTOUT,
            'output_printout_period': 1,
            'output_file_format': 'txt',
            'output_file_name': 'particles_2d',
            }
        cylag_solver = cylag.Solver(fset, pset, parameters)
        cylag_solver.solve()
        # particles_io class for post-processing
        if parameters['output_file']:
            part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d.txt')
            traj = part.get_trajectories([0])
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        # Same reference as the zero-noise SDE branch below: both must use
        # the initial, not the updated, velocity in the exact displacement.
        xpa = np.asarray([\
            [5.58482914e+03, 1.46424824e+01],
            [5.58333674e+03, 5.73120494e+02],
            [1.12027542e+03, 3.17155011e+00],
            [1.11839840e+03, 5.63135678e+02]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            print(xp)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=-1)
            v = fset.get_field(2, parameters['final_time'], layer=-1)
            velocity = np.sqrt(u**2 + v**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
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
            os.system('rm -r particles')
        # clean 
        del tri
        del fset
        del pset
        del cylag_solver

    def test_solver_2d_lsm2_advection_di(self):
        # fset and tri
        fset, tri = fset_2d()
        # particle set
        position = np.asarray([\
            [5000., 0.], [5000., 500.], [1000., 0.], [1000., 500.]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=2, initial_pool_size=1000)
        # cylag solver
        parameters = {
            'final_time': 3600.,
            'time_step': 20.,
            'time_scheme': 5,
            'frozen_eulerian_fields': True,
            'model': 2,
            'particle_velocity_init': 1,
            'diffusion_model': 1,
            'diffusion_lsm2_option': 2,
            'horizontal_diffusivity': 0.,
            'vertical_diffusivity': 0.,
            'listing': True,
            'output_file': PRINTOUT,
            'output_printout_period': 1,
            'output_file_format': 'txt',
            'output_file_name': 'particles_2d',
            }
        cylag_solver = cylag.Solver(fset, pset, parameters)
        cylag_solver.solve()
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        # check result
        xpa = np.asarray([\
            [5.58482914e+03, 1.46424824e+01],
            [5.58333674e+03, 5.73120494e+02],
            [1.12027542e+03, 3.17155011e+00],
            [1.11839840e+03, 5.63135678e+02]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            print(xp)
            part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d.txt')
            traj = part.get_trajectories([0])
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=-1)
            v = fset.get_field(2, parameters['final_time'], layer=-1)
            velocity = np.sqrt(u**2 + v**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
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
            os.system('rm -r particles')
        # clean 
        del tri
        del fset
        del pset
        del cylag_solver

    def test_solver_2d_lsm2_advection_euler_maruyama(self):
        # fset and tri
        fset, tri = fset_2d()
        # particle set
        position = np.asarray([\
            [5000., 0.], [5000., 500.], [1000., 0.], [1000., 500.]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=2, initial_pool_size=1000)
        # cylag solver
        parameters = {
            'final_time': 50.,
            'time_step': 0.005,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 2,
            'particle_velocity_init': 1,
            'diffusion_model': 1,
            'diffusion_lsm2_option': 2,
            'horizontal_diffusivity': 0.,
            'vertical_diffusivity': 0.,
            'listing': True,
            'output_file': PRINTOUT,
            'output_printout_period': 100,
            'output_file_format': 'txt',
            'output_file_name': 'particles_2d_em',
            }
        cylag_solver = cylag.Solver(fset, pset, parameters)
        cylag_solver.solve()
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        # check result
        xpa = np.array([
            [4981.651198, -435.835882],
            [5025.233608,   62.329215],
            [ 996.422963,  -87.174511],
            [1040.00244 ,  411.033803]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            print(xp)
            part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d_em.txt')
            traj = part.get_trajectories([0,1,2,3])
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=-1)
            v = fset.get_field(2, parameters['final_time'], layer=-1)
            velocity = np.sqrt(u**2 + v**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
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
            os.system('rm -r particles')
        # clean 
        del tri
        del fset
        del pset
        del cylag_solver

if __name__ == "__main__":
    unittest.main()
