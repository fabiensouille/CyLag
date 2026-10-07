# -*- coding: utf-8 -*-
import unittest
import time
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

PRINTOUT = False

class CheckEulerianFieldSet(unittest.TestCase):

    def test_eulerian_field_set_init_2d(self):
        root = path.dirname(__file__)
        t2d_file = path.join(root, "data", "r2d_test.slf")
        bnd_file = path.join(root, "data", "geo_test.cli")
        # optional fields
        opt_fields = ['WATER DEPTH']
        # initialize 
        eulerian_field_set = cylag.EulerianFieldSet.from_telemac2d(\
            t2d_file, bnd_file=bnd_file, optional_fields=opt_fields)
        # matplotlib triangulation and trifinder
        tri = Triangulation(\
            eulerian_field_set.triangular_mesh.x,\
            eulerian_field_set.triangular_mesh.y,\
            eulerian_field_set.triangular_mesh.triangles)
        # get velocity field 
        #print(np.asarray(eulerian_field_set.times))
        #print(np.asarray(eulerian_field_set.field_names))
        time = 3600. # in times
        time = 3752. # not in times
        elev = eulerian_field_set.get_field(0, time)
        velx = eulerian_field_set.get_field(1, time)
        vely = eulerian_field_set.get_field(2, time)
        velocity = np.sqrt(velx**2 + vely**2)
        # get optional field
        h = eulerian_field_set.get_field(4, time)
        #print(np.asarray(eulerian_field_set.opt_fields_names))
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(14, 3.5))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot velocity
            #levels = np.arange(0., 1., 0.05)
            #cs = ax.tricontourf(tri, velocity, levels=levels, cmap='jet')
            #fig.colorbar(cs, ax=ax)
            # plot water depth
            levels = np.arange(4., 15., 0.1)
            cs = ax.tricontourf(tri, h, levels=levels, cmap='jet')
            fig.colorbar(cs, ax=ax)
            plt.show()
        # clean 
        del tri
        del eulerian_field_set

    def test_eulerian_field_set_init_3d(self):
        root = path.dirname(__file__)
        t3d_file = path.join(root, "data", "r3d_test.slf")
        bnd_file = path.join(root, "data", "geo_test.cli")
        # optional fields
        opt_fields = None
        # initialize 
        eulerian_field_set = cylag.EulerianFieldSet.from_telemac3d(\
            t3d_file, bnd_file=bnd_file, optional_fields=opt_fields)
        # matplotlib triangulation and trifinder
        tri = Triangulation(\
            eulerian_field_set.triangular_mesh.x,\
            eulerian_field_set.triangular_mesh.y,\
            eulerian_field_set.triangular_mesh.triangles)
        # get velocity field 
        #print(np.asarray(eulerian_field_set.times))
        #print(np.asarray(eulerian_field_set.field_names))
        print(eulerian_field_set.nlayers)
        time = 0. # in times
        elev = eulerian_field_set.get_field(0, time, layer=5)
        velx = eulerian_field_set.get_field(1, time, layer=5)
        vely = eulerian_field_set.get_field(2, time, layer=5)
        velz = eulerian_field_set.get_field(3, time, layer=5)
        velocity = np.sqrt(velx**2 + vely**2 + velz**2)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(14, 3.5))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot velocity
            levels = np.arange(0., 1., 0.05)
            cs = ax.tricontourf(tri, velocity, levels=levels, cmap='jet')
            fig.colorbar(cs, ax=ax)
            plt.show()
        # clean
        del tri
        del eulerian_field_set

if __name__ == "__main__":
    unittest.main()
