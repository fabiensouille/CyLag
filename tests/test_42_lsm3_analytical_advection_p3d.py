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
    opt_fields = np.zeros((3, 1, meshsize), dtype='d')
    opt_fields_names = ['BOTTOM', 'TURBULENT ENERG.', 'DISSIPATION']
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
        opt_fields=opt_fields,
        opt_fields_names=opt_fields_names,
        trifinder=trifinder,
        friction_model=1,
        friction_coefficient=40.)
    return fset, tri

class CheckSolverAnalytical(unittest.TestCase):

    def test_solver_p3d_lsm2_advection_di(self):
        # fset
        fset, tri = fset_2d()
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
            'final_time': 3600.,
            'time_step': 1.,
            'time_scheme': 5,
            'frozen_eulerian_fields': True,
            'model': 3,
            'particle_velocity_init': 1,
            'diffusion_model': 2,
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
            [ 4.75450561e+03, -1.83294844e+03,  5.00000000e-01],
            [ 2.33409108e+03, -4.58085258e+03,  9.99900000e-01],
            [ 9.57929441e+02, -3.69170032e+02,  5.00000000e-01],
            [ 8.55082445e+02, -7.73654172e+02,  9.99900000e-01]])
        np.testing.assert_allclose(xp, xpa, rtol=1e-7)
        # plot
        if PRINTOUT:
            print(xp)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            # plot velocity 
            u = fset.get_field(1, parameters['final_time'], layer=10)
            v = fset.get_field(2, parameters['final_time'], layer=10)
            velocity = np.sqrt(u**2 + v**2)
            levels = np.arange(0., 15., 1)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.75)
            #fig.colorbar(cs, ax=ax, label="")
            #ax.quiver(tri.x, tri.y, u, v, scale=300., color='k')
            ax.triplot(tri, lw=0.25, color='0.75')
            # plot particles
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
