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
        opt_fields = ['WATER DEPTH', 'BOTTOM']
        # initialize
        eulerian_field_set = cylag.EulerianFieldSet.from_telemac2d(\
            t2d_file, bnd_file=bnd_file, optional_fields=opt_fields)
        # matplotlib triangulation and trifinder
        tri = Triangulation(\
            eulerian_field_set.triangular_mesh.x,\
            eulerian_field_set.triangular_mesh.y,\
            eulerian_field_set.triangular_mesh.triangles)
        trifinder = tri.get_trifinder()
        # get velocity field
        #print(np.asarray(eulerian_field_set.times))
        #print(np.asarray(eulerian_field_set.field_names))
        time = 3600. # in times
        time = 3752. # not in times
        elev = eulerian_field_set.get_field(0, time)
        velx = eulerian_field_set.get_field(1, time)
        vely = eulerian_field_set.get_field(2, time)
        velocity = np.sqrt(velx**2 + vely**2)
        # interpolation test
        point = np.array([395., 444.])
        point = np.array([375., 571.])
        # localize point 
        itri = trifinder(point[0], point[1])
        # interpolate with cylag
        eulerian_field_set._set_record(time, 0)
        eulerian_field_set._time_interpolation(time)
        interp_h = eulerian_field_set._interpolate_field_2d(4, itri, -1, point[0], point[1])
        interp_zb = eulerian_field_set._interpolate_field_2d(5, itri, -1, point[0], point[1])
        interp_el = eulerian_field_set._interpolate_field_2d(0, itri, -1, point[0], point[1])
        interp_ux, interp_uy = eulerian_field_set._interpolate_velocity_2d(itri, point[0], point[1])
        if PRINTOUT:
            print("interp_h", interp_h)
            print("interp_zb", interp_zb)
            print("interp_el", interp_el)
            print("interp_ux", interp_ux)
            print("interp_uy", interp_uy)
        # interpolate with cylag using wrappers and integrated trifinder
        interp_el1 = eulerian_field_set.interpolate_field(0, point, time)
        interp_ux1, interp_uy1 = eulerian_field_set.interpolate_velocity(point, time)
        if PRINTOUT:
            print("interp_el", interp_el)
            print("interp_ux", interp_ux)
            print("interp_uy", interp_uy)
        # checks
        np.testing.assert_equal(interp_el, interp_el1)
        np.testing.assert_equal(interp_ux, interp_ux1)
        np.testing.assert_equal(interp_uy, interp_uy1)
        # interpolate with TelemacFile or mtri for comparison
        # TODO
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
        trifinder = tri.get_trifinder()
        # get velocity field 
        #print(np.asarray(eulerian_field_set.times))
        #print(np.asarray(eulerian_field_set.field_names))
        #print(eulerian_field_set.nlayers)
        time = 0. # in times
        elev = eulerian_field_set.get_field(0, time, layer=5)
        velx = eulerian_field_set.get_field(1, time, layer=5)
        vely = eulerian_field_set.get_field(2, time, layer=5)
        velz = eulerian_field_set.get_field(3, time, layer=5)
        velocity = np.sqrt(velx**2 + vely**2 + velz**2)
        # interpolation test
        point = np.array([375., 571., 260.])
        # localize point 
        itri = trifinder(point[0], point[1])
        # interpolate with cylag
        eulerian_field_set._set_record(time, 0)
        eulerian_field_set._time_interpolation(time)
        interp_el = eulerian_field_set._interpolate_field_3d(\
            0, itri, 0, 1, point[0], point[1], point[2])
        interp_ux, interp_uy, interp_uz = \
            eulerian_field_set._interpolate_velocity_3d(\
            itri,  0, 1, point[0], point[1], point[2])
        if PRINTOUT:
            print("interp_el", interp_el)
            print("interp_ux", interp_ux)
            print("interp_uy", interp_uy)
            print("interp_uz", interp_uz)
        # interpolate with cylag and manual z localization
        lower_layer = eulerian_field_set.z_localize(itri, point[0], point[1], point[2])
        interp_el1 = eulerian_field_set._interpolate_field_3d(\
            0, itri, lower_layer, 0, point[0], point[1], point[2])
        interp_ux1, interp_uy1, interp_uz1 = \
            eulerian_field_set._interpolate_velocity_3d(\
            itri,  0, 1, point[0], point[1], point[2])
        if PRINTOUT:
            print("interp_el", interp_el1)
            print("interp_ux", interp_ux1)
            print("interp_uy", interp_uy1)
            print("interp_uz", interp_uz1)
        # interpolate with cylag using wrappers and integrated trifinder
        interp_el2 = eulerian_field_set.interpolate_field(0, point, time)
        interp_ux2, interp_uy2, interp_uz2 = eulerian_field_set.interpolate_velocity(point, time)
        if PRINTOUT:
            print("interp_el", interp_el2)
            print("interp_ux", interp_ux2)
            print("interp_uy", interp_uy2)
            print("interp_uz", interp_uz2)
        # checks
        np.testing.assert_equal(interp_el, interp_el1)
        np.testing.assert_equal(interp_el, interp_el2)
        np.testing.assert_equal(interp_ux, interp_ux1)
        np.testing.assert_equal(interp_ux, interp_ux2)
        np.testing.assert_equal(interp_uy, interp_uy1)
        np.testing.assert_equal(interp_uy, interp_uy2)
        np.testing.assert_equal(interp_uz, interp_uz1)
        np.testing.assert_equal(interp_uz, interp_uz2)
        # interpolate with TelemacFile or mtri for comparison
        # TODO
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
        del eulerian_field_set
        del tri

if __name__ == "__main__":
    unittest.main()
