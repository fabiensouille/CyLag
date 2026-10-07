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

class radial_velocity_field_3d():
    def __init__(self, omega=2.*np.pi/3600.):
        self.omega = omega
        self.zs = 1.
        self.zb = 0.

    def get(self, x, y, z):
        r, phi = np.sqrt(x**2 + y**2), np.arctan2(x, y)
        velx =  self.omega*r*np.cos(phi)*z
        vely = -self.omega*r*np.sin(phi)*z
        velz = 0.
        return velx, vely, velz

    def get_field_on_3dmesh(self, tri, nlayers=1):
        meshx = tri.x
        meshy = tri.y
        size = len(meshx)
        elev = np.empty(size*nlayers)
        velx = np.empty(size*nlayers)
        vely = np.empty(size*nlayers)
        velz = np.empty(size*nlayers)
        dz = (self.zs - self.zb)/max((nlayers-1), 1)
        zlayers = np.array([dz*i for i in range(nlayers)])
        for i in range(nlayers):
            elev[i*size:(i+1)*size] = zlayers[i]
            velx[i*size:(i+1)*size],\
            vely[i*size:(i+1)*size],\
            velz[i*size:(i+1)*size] = self.get(meshx[:], meshy[:], zlayers[i])
        return elev, velx, vely, velz

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
    # set analytical velocity field:
    nlayers = 11
    field = radial_velocity_field_3d()
    elev, velx, vely, velz = field.get_field_on_3dmesh(tri, nlayers=nlayers)
    velocity = np.sqrt(velx**2 + vely**2 + velz**2)
    # field_set
    meshsize = len(tri.x)
    elevation = np.zeros((1, meshsize*nlayers), dtype='d')
    velocity_x = np.zeros((1, meshsize*nlayers), dtype='d')
    velocity_y = np.zeros((1, meshsize*nlayers), dtype='d')
    velocity_z = np.zeros((1, meshsize*nlayers), dtype='d')
    elevation[0, :] = elev
    velocity_x[0, :] = velx
    velocity_y[0, :] = vely
    velocity_z[0, :] = velz
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
        position = np.asarray([\
            [5000., 0., 0.5], [5000., 500., 0.9999],\
            [1000., 0., 0.5], [1000., 500., 0.9999]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=3, initial_pool_size=4)
        # cylag solver
        ntour = 1
        parameters = {
            'final_time': ntour*3600.,
            'time_step': 10.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 1,
            'diffusion_model': 0,
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
        xpa = np.asarray([\
            [-5.06900817e+03, -4.04228110e-01,  5.00000000e-01],
            [ 5.28103619e+03,  5.34857127e+02,  9.99900000e-01],
            [-1.01380163e+03, -8.08456219e-02,  5.00000000e-01],
            [ 1.05567231e+03,  5.29507814e+02,  9.99900000e-01]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=10)
            v = fset.get_field(2, parameters['final_time'], layer=10)
            w = fset.get_field(3, parameters['final_time'], layer=10)
            velocity = np.sqrt(u**2 + v**2 + w**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
            ax.triplot(tri, lw=0.25, color='0.75')
            # plot particles
            analyticpos = np.asarray([[-5000., 0., 0.5], [5000., 500., 1.],
                                      [-1000., 0., 0.5], [1000., 500., 1.]])
            ax.plot(analyticpos[:, 0], analyticpos[:, 1],\
                c='r', lw=0., marker='o', markersize=3, label="Analytic")
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

    def test_solver_3d_lsm1_advection_RK2(self):
        # fset
        fset, tri = fset_3d()
        # particle set
        position = np.asarray([\
            [5000., 0., 0.5], [5000., 500., 0.9999],\
            [1000., 0., 0.5], [1000., 500., 0.9999]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=3, initial_pool_size=4)
        # cylag solver
        ntour = 1
        parameters = {
            'final_time': ntour*3600.,
            'time_step': 200.,
            'time_scheme': 2,
            'frozen_eulerian_fields': True,
            'model': 1,
            'diffusion_model': 0,
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
        #print(xp)
        # check result
        xpa = np.asarray([\
            [-5.00982318e+03,  7.91738560e+01,  5.00000000e-01],
            [ 5.19393649e+03, -1.16404904e+02,  9.99900000e-01],
            [-1.00196464e+03,  1.58347712e+01,  5.00000000e-01],
            [ 1.08914758e+03,  3.87197910e+02,  9.99900000e-01]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=10)
            v = fset.get_field(2, parameters['final_time'], layer=10)
            w = fset.get_field(3, parameters['final_time'], layer=10)
            velocity = np.sqrt(u**2 + v**2 + w**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
            ax.triplot(tri, lw=0.25, color='0.75')
            # plot particles
            analyticpos = np.asarray([[-5000., 0., 0.5], [5000., 500., 1.],
                                      [-1000., 0., 0.5], [1000., 500., 1.]])
            ax.plot(analyticpos[:, 0], analyticpos[:, 1],\
                c='r', lw=0., marker='o', markersize=3, label="Analytic")
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

    def test_solver_3d_lsm1_advection_RK4(self):
        # fset
        fset, tri = fset_3d()
        # particle set
        position = np.asarray([\
            [5000., 0., 0.5], [5000., 500., 0.9999],\
            [1000., 0., 0.5], [1000., 500., 0.9999]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=3, initial_pool_size=4)
        # cylag solver
        ntour = 1
        parameters = {
            'final_time': ntour*3600.,
            'time_step': 200.,
            'time_scheme': 4,
            'frozen_eulerian_fields': True,
            'model': 1,
            'diffusion_model': 0,
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
        #print(xp)
        # check result
        xpa = np.asarray([\
            [-4.99998240e+03, -1.20144922e-01,  5.00000000e-01],
            [ 4.99819692e+03,  5.06745590e+02,  9.99900000e-01],
            [-9.99996480e+02, -2.40289843e-02,  5.00000000e-01],
            [ 9.99090798e+02,  5.01259730e+02,  9.99900000e-01]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=10)
            v = fset.get_field(2, parameters['final_time'], layer=10)
            w = fset.get_field(3, parameters['final_time'], layer=10)
            velocity = np.sqrt(u**2 + v**2 + w**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
            ax.triplot(tri, lw=0.25, color='0.75')
            # plot particles
            analyticpos = np.asarray([[-5000., 0., 0.5], [5000., 500., 1.],
                                      [-1000., 0., 0.5], [1000., 500., 1.]])
            ax.plot(analyticpos[:, 0], analyticpos[:, 1],\
                c='r', lw=0., marker='o', markersize=3, label="Analytic")
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

    def test_solver_3d_lsm1_advection_euler_maruyama(self):
        # fset
        fset, tri = fset_3d()
        # particle set
        position = np.asarray([\
            [5000., 0., 0.5], [5000., 500., 0.9999],\
            [1000., 0., 0.5], [1000., 500., 0.9999]])
        apos = copy.copy(position)
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, dim=3, initial_pool_size=4)
        # cylag solver
        ntour = 1
        parameters = {
            'final_time': ntour*3600.,
            'time_step': 10.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 1,
            'diffusion_model': 1,
            'horizontal_diffusivity': 0.,
            'vertical_diffusivity': 0.,
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
        xpa = np.asarray([\
            [-5.06900817e+03, -4.04228110e-01,  5.00000000e-01],
            [ 5.28103619e+03,  5.34857127e+02,  9.99900000e-01],
            [-1.01380163e+03, -8.08456219e-02,  5.00000000e-01],
            [ 1.05567231e+03,  5.29507814e+02,  9.99900000e-01]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=10)
            v = fset.get_field(2, parameters['final_time'], layer=10)
            w = fset.get_field(3, parameters['final_time'], layer=10)
            velocity = np.sqrt(u**2 + v**2 + w**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
            ax.triplot(tri, lw=0.25, color='0.75')
            # plot particles
            analyticpos = np.asarray([[-5000., 0., 0.5], [5000., 500., 1.],
                                      [-1000., 0., 0.5], [1000., 500., 1.]])
            ax.plot(analyticpos[:, 0], analyticpos[:, 1],\
                c='r', lw=0., marker='o', markersize=3, label="Analytic")
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
