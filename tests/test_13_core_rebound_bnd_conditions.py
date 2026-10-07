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

def plot_triangles_id(ax, triangulation, indices):
    """ Plot triangulation.triangles id on mesh """
    for i in range(triangulation.triangles.shape[0]):
        if i in indices:
            v0, v1, v2 = triangulation.triangles[i]
            xc = np.mean([triangulation.x[v0], triangulation.x[v1], triangulation.x[v2]])
            yc = np.mean([triangulation.y[v0], triangulation.y[v1], triangulation.y[v2]])
            ax.text(xc, yc, "{}".format(i), fontsize=8, color='r')

def plot_cylag_edges(ax, triangulation, indices, color='k'):
    """ Plot cylag edges """
    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    edges = np.asarray(triangulation.edges)
    colors = ['k', 'r', 'g', 'b']

    #for i in range(edges.shape[0]):
    for i in indices:
        v0, v1 = edges[i, 0], edges[i, 1]
        xs = [triangulation.x[v0], triangulation.x[v1]]
        ys = [triangulation.y[v0], triangulation.y[v1]]
        j = triangulation.edges_labels[i]
        ax.plot(xs, ys, c=colors[j], lw=1.5, marker='o', markersize=5)

def plot_cylag_boundary_edges(ax, triangulation):
    """ Plot cylag boundary edges """
    colors = ['k', 'r', 'g', 'b']

    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    bnd_edges = np.asarray(triangulation.boundary_edges)

    for i in range(bnd_edges.shape[0]):
        v0, v1 = bnd_edges[i, 0], bnd_edges[i, 1]
        xs = [triangulation.x[v0], triangulation.x[v1]]
        ys = [triangulation.y[v0], triangulation.y[v1]]
        ax.plot(xs, ys, c=colors[bnd_edges[i, 2]], lw=1.5, marker='+', markersize=2)

def normal2u(v0, v1):
    """ Computes normal of edge """
    n = np.zeros( (2), dtype='d')
    norm = np.sqrt((v1[0]-v0[0])**2 + (v1[1]-v0[1])**2)
    n[0] = (v0[1] - v1[1])/norm
    n[1] = (v1[0] - v0[0])/norm
    return n
    
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

class CheckRebounds(unittest.TestCase):

    def test_velocity_orientation_change(self):
        u0 = np.array([1., 0.])
        up = np.array([1., 0.])
        A  = np.array([0., 0.])
        B  = np.array([1., 1.])
        # 
        nn = normal2u(A, B)
        # rotation of up
        up0 = up[0]
        up[0] = up0*nn[0] + up[1]*nn[1]
        up[1] =-up0*nn[1] + up[1]*nn[0]
        # invert normal component of the velocity
        up[0] = -1.*up[0]
        # inverse rotation
        up0 = up[0]
        up[0] = up0*nn[0] - up[1]*nn[1]
        up[1] = up0*nn[1] + up[1]*nn[0]
        if PRINTOUT:
            # plot
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            plt.plot([-1., -1.+u0[0]], [-1., -1.+u0[1]], lw=1., marker='o', markersize=10, c='b')
            plt.plot([-1., -1.+up[0]], [-1., -1.+up[1]], lw=1., marker='o', markersize=10, c='b')
            plt.plot(A[0], A[1], lw=1., marker='o', markersize=10, c='r')
            plt.plot(B[0], B[1], lw=1., marker='o', markersize=10, c='r')
            plt.show()

    def test_2D_rebound_specular(self):
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
        # particle set
        position = np.asarray([\
            [7500., 7500.], [7500., 7500.], [7500., 7500.],
            [7500., 7500.], [7500., 7500.], [7500., 7500.],
            [7500., 7500.], [7500., 7500.], [7500., 7500.],
            [7500., 7500.], [7500., 7500.]])
        velocity = 10.*np.asarray([\
            [1., 0. ], [1., 0.1], [1., 0.2],
            [1., 0.3], [1., 0.4], [1., 0.5],
            [1., 0.6], [1., 0.7], [1., 0.8],
            [1., 0.9], [1., 1. ]])
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, velocity, dim=2, initial_pool_size=12)
        # cylag solver
        parameters = {
            'final_time': 5000.,
            'time_step': 50.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 0,
            'particle_velocity_init': 0,
            'boundary_conditions': True,
            'boundary_conditions_debug': False,
            'max_wall_rebounds': 4,
            'rebound_damping_coef': 0., # 0: no energy loss / 1: total energy loss
            'listing': False,
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
            traj = part.get_trajectories(range(position.shape[0]))
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        #print(np.asarray(xp))
        # check
        xpa =np.array([\
            [ 2900.024,  7500.   ],
            [ 2900.024,  7700.012],
            [ 2900.024,  2700.012],
            [ 2900.024, -2299.988],
            [ 2900.024, -7299.988],
            [ 2900.024, -7700.012],
            [ 2900.024, -2700.012],
            [ 2900.024,  2299.988],
            [ 2900.024,  7299.988],
            [ 2900.024,  7900.024]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            ax.triplot(tri, lw=0.25, color='0.75')
            plot_cylag_boundary_edges(ax, fset.triangular_mesh)
            plot_cylag_edges(ax, fset.triangular_mesh, [574], color='r')
            plot_triangles_id(ax, fset.triangular_mesh, [4178, 8477])
            # plot particles
            ax.plot(xp[:, 0], xp[:, 1], c='r', lw=0., marker='o', markersize=2, label="Final positions")
            if parameters['output_file']:
                for tr in traj:
                    ax.plot(tr[:, 0], tr[:, 1], lw=0.5, marker='o', markersize=1)
            plt.legend()
            plt.show()
            plt.close(fig)
        # clean
        del tri
        del cylagtri
        del pset
        del fset
        del cylag_solver

    def test_3D_rebound_specular(self):
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
        # particle set
        position = np.asarray([\
            [7500., 7500., 0.5], [7500., 7500., 0.5], [7500., 7500., 0.5],
            [7500., 7500., 0.5], [7500., 7500., 0.5], [7500., 7500., 0.5],
            [7500., 7500., 0.5], [7500., 7500., 0.5], [7500., 7500., 0.5],
            [7500., 7500., 0.5], [7500., 7500., 0.5]])
        velocity = 10.*np.asarray([\
            [1., 0. , 0.], [1., 0.1, 0.], [1., 0.2, 0.],
            [1., 0.3, 0.], [1., 0.4, 0.], [1., 0.5, 0.],
            [1., 0.6, 0.], [1., 0.7, 0.], [1., 0.8, 0.],
            [1., 0.9, 0.], [1., 1., 0. ]])
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, velocity, dim=3, initial_pool_size=12)
        # cylag solver
        ntour = 1
        parameters = {
            'final_time': 5000.,
            'time_step': 50.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 0,
            'particle_velocity_init': 0,
            'boundary_conditions': True,
            'boundary_conditions_debug': False,
            'max_wall_rebounds': 4,
            'rebound_damping_coef': 0.,
            'listing': False,
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
            traj = part.get_trajectories(range(position.shape[0]))
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        #print(np.asarray(xp))
        # check
        xpa =np.array([\
            [ 2900.024,  7500.   , 0.5],
            [ 2900.024,  7700.012, 0.5],
            [ 2900.024,  2700.012, 0.5],
            [ 2900.024, -2299.988, 0.5],
            [ 2900.024, -7299.988, 0.5],
            [ 2900.024, -7700.012, 0.5],
            [ 2900.024, -2700.012, 0.5],
            [ 2900.024,  2299.988, 0.5],
            [ 2900.024,  7299.988, 0.5],
            [ 2900.024,  7900.024, 0.5]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            ax.triplot(tri, lw=0.25, color='0.75')
            plot_cylag_boundary_edges(ax, fset.triangular_mesh)
            plot_cylag_edges(ax, fset.triangular_mesh, [574], color='r')
            plot_triangles_id(ax, fset.triangular_mesh, [4178, 8477])
            # plot particles
            ax.plot(xp[:, 0], xp[:, 1], c='r', lw=0., marker='o', markersize=2, label="Final positions")
            if parameters['output_file']:
                for tr in traj:
                    ax.plot(tr[:, 0], tr[:, 1], lw=0.5, marker='o', markersize=1)
            plt.legend()
            plt.show()
            plt.close(fig)
            del part
        # clean 
        del tri
        del cylagtri
        del pset
        del fset
        del cylag_solver

    def test_3D_rebound_specular_2(self):
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
        # particle set
        position = np.asarray([\
            [7500., 7500., 0.5], [7500., 7500., 0.5], [7500., 7500., 0.5],
            [7500., 7500., 0.5], [7500., 7500., 0.5], [7500., 7500., 0.5],
            [7500., 7500., 0.5], [7500., 7500., 0.5], [7500., 7500., 0.5],
            [7500., 7500., 0.5], [7500., 7500., 0.5]])
        velocity = 10.*np.asarray([\
            [1., 0. , 0.0006], [1., 0.1, -0.0006], [1., 0.2, 0.],
            [1., 0.3, 0.0005], [1., 0.4, -0.0005], [1., 0.5, 0.],
            [1., 0.6, 0.0004], [1., 0.7, -0.0004], [1., 0.8, 0.],
            [1., 0.9, 0.0003], [1., 1.,  -0.0003]])
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, velocity, dim=3, initial_pool_size=12)
        # cylag solver
        ntour = 1
        parameters = {
            'final_time': 5000.,
            'time_step': 50.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 0,
            'particle_velocity_init': 0,
            'boundary_conditions': True,
            'boundary_conditions_debug': False,
            'max_wall_rebounds': 4,
            'rebound_damping_coef': 0.,
            'listing': False,
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
            traj = part.get_trajectories(range(position.shape[0]))
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        #print(np.asarray(xp))
        # check
        xpa =np.array([\
            [ 2.900024e+03,  7.500000e+03,  9.000000e-01],
            [ 2.900024e+03,  7.700012e+03,  1.000000e-01],
            [ 2.900024e+03,  2.700012e+03,  5.000000e-01],
            [ 2.900024e+03, -2.299988e+03,  1.000000e+00],
            [ 2.900024e+03, -7.299988e+03,  0.000000e+00],
            [ 2.900024e+03, -7.700012e+03,  5.000000e-01],
            [ 2.900024e+03, -2.700012e+03,  9.000000e-01],
            [ 2.900024e+03,  2.299988e+03,  1.000000e-01],
            [ 2.900024e+03,  7.299988e+03,  5.000000e-01],
            [ 2.900024e+03,  7.900024e+03,  9.000000e-01]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            ax.triplot(tri, lw=0.25, color='0.75')
            plot_cylag_boundary_edges(ax, fset.triangular_mesh)
            plot_cylag_edges(ax, fset.triangular_mesh, [574], color='r')
            plot_triangles_id(ax, fset.triangular_mesh, [4178, 8477])
            # plot particles
            ax.plot(xp[:, 0], xp[:, 1], c='r', lw=0., marker='o', markersize=2, label="Final positions")
            if parameters['output_file']:
                for tr in traj:
                    ax.plot(tr[:, 0], tr[:, 1], lw=0.5, marker='o', markersize=1)
            plt.legend()
            plt.show()
            plt.close(fig)
        # clean 
        del tri
        del cylagtri
        del pset
        del fset
        del cylag_solver
        

    def test_2D_rebound_specular_large_dt(self):
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
        # particle set
        position = np.asarray([[7500., 7500.]])
        velocity = 10.*np.asarray([[1., 0.4]])
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, velocity, dim=2, initial_pool_size=12)
        # cylag solver
        parameters = {
            'final_time': 5000.,
            'time_step': 500.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 0,
            'particle_velocity_init': 0,
            'boundary_conditions': True,
            'boundary_conditions_debug': False,
            'max_wall_rebounds': 4,
            'rebound_damping_coef': 0., # 0: no energy loss / 1: total energy loss
            'listing': False,
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
            traj = part.get_trajectories(range(position.shape[0]))
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        #print(np.asarray(xp))
        # check
        xpa =np.array([[ 2900.024, -7299.988 ]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            ax.triplot(tri, lw=0.25, color='0.75')
            plot_cylag_boundary_edges(ax, fset.triangular_mesh)
            plot_cylag_edges(ax, fset.triangular_mesh, [574], color='r')
            plot_triangles_id(ax, fset.triangular_mesh, [4178, 8477])
            # plot particles
            ax.plot(xp[:, 0], xp[:, 1], c='r', lw=0., marker='o', markersize=2, label="Final positions")
            if parameters['output_file']:
                for tr in traj:
                    ax.plot(tr[:, 0], tr[:, 1], lw=0.5, marker='o', markersize=1)
            plt.legend()
            plt.show()
            plt.close(fig)
        # clean
        del tri
        del cylagtri
        del pset
        del fset
        del cylag_solver
        
    def test_2D_rebound_diffuse(self):
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
        # particle set
        position = np.asarray([\
            [7500., 7500.], [7500., 7500.], [7500., 7500.],
            [7500., 7500.], [7500., 7500.], [7500., 7500.],
            [7500., 7500.], [7500., 7500.], [7500., 7500.],
            [7500., 7500.], [7500., 7500.]])
        velocity = 10.*np.asarray([\
            [1., 0. ], [1., 0.1], [1., 0.2],
            [1., 0.3], [1., 0.4], [1., 0.5],
            [1., 0.6], [1., 0.7], [1., 0.8],
            [1., 0.9], [1., 1. ]])
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, velocity, dim=2, initial_pool_size=12)
        # cylag solver
        parameters = {
            'final_time': 5000.,
            'time_step': 50.,
            'time_scheme': 1,
            'frozen_eulerian_fields': True,
            'model': 0,
            'particle_velocity_init': 0,
            'boundary_conditions': True,
            'boundary_conditions_type': 2,
            'wall_roughness_angle': 35.0,
            'boundary_conditions_debug': False,
            'max_wall_rebounds': 4,
            'rebound_damping_coef': 0., # 0: no energy loss / 1: total energy loss
            'listing': False,
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
            traj = part.get_trajectories(range(position.shape[0]))
        # get particles last position
        xp, up, us = cylag_solver.particle_set.get_state()
        #print(np.asarray(xp))
        # check
        xpa =np.array([\
            [ 2900.024,  7500.   ],
            [ 2900.024,  7700.012],
            [ 2900.024,  2700.012],
            [ 2900.024, -2299.988],
            [ 2900.024, -7299.988],
            [ 2900.024, -7700.012],
            [ 2900.024, -2700.012],
            [ 2900.024,  2299.988],
            [ 2900.024,  7299.988],
            [ 2900.024,  7900.024]])
        #np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            ax.triplot(tri, lw=0.25, color='0.75')
            plot_cylag_boundary_edges(ax, fset.triangular_mesh)
            plot_cylag_edges(ax, fset.triangular_mesh, [574], color='r')
            plot_triangles_id(ax, fset.triangular_mesh, [4178, 8477])
            # plot particles
            ax.plot(xp[:, 0], xp[:, 1], c='r', lw=0., marker='o', markersize=2, label="Final positions")
            if parameters['output_file']:
                for tr in traj:
                    ax.plot(tr[:, 0], tr[:, 1], lw=0.5, marker='o', markersize=1)
            plt.legend()
            plt.show()
            plt.close(fig)
        # clean
        del tri
        del cylagtri
        del pset
        del fset
        del cylag_solver

if __name__ == "__main__":
    unittest.main()
