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

def plot_polygon(ax, poly, c):
    polygon = Polygon([[0, 0], [0, 0]], facecolor=c, alpha=0.25)
    polygon.set_xy(np.column_stack([poly[:, 0], poly[:, 1]]))
    ax.add_patch(polygon)

class CheckInit(unittest.TestCase):

    def test_init_method_2d(self):
        xy_method = 2
        # rectangle poly
        rectangle = np.array(([0., 0.], [1., 0.], [1., 1.], [0., 1.]))
        pos0 = cylag.init_positions_2d(poly=rectangle, poly_area=1.,\
            xy_method=xy_method, grid_res=[2, 2], xy_npart=16, xy_density=100)
        # circle poly
        circ = cylag.circle(x0=2., y0=2., r=1., n=180)
        pos1 = cylag.init_positions_2d(poly=circ, poly_area=np.pi,\
            xy_method=xy_method, grid_res=[2, 2], xy_npart=16, xy_density=100)
        # poly
        poly = np.array(([2., 0.], [2.5, 0.], [3., 1.], [2., 0.5]))
        pos2 = cylag.init_positions_2d(poly=poly, poly_area=1.,\
            xy_method=xy_method, grid_res=[2, 2], xy_npart=16, xy_density=100)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # init 0
            plot_polygon(ax, rectangle, c='b')
            plt.plot(pos0[:, 0], pos0[:, 1], lw=0., marker='o', markersize=10, c='b')
            # init 1
            plot_polygon(ax, circ, c='r')
            plt.plot(pos1[:, 0], pos1[:, 1], lw=0., marker='o', markersize=10, c='r')
            # init 2
            plot_polygon(ax, poly, c='g')
            plt.plot(pos2[:, 0], pos2[:, 1], lw=0., marker='o', markersize=10, c='g')
            plt.show()

    def test_init_method_3d(self):
        xy_method = 0
        z_method = 2
        # rectangle poly
        rectangle = np.array(([0., 0.], [1., 0.], [1., 1.], [0., 1.]))
        pos0 = cylag.init_positions_3d(poly=rectangle, poly_area=1.,\
            xy_method=xy_method, grid_res=[2, 2, 2], xy_npart=16, xy_density=100,\
            z_method=z_method, z_npart=2, z_density=1., zmin=0., zmax=2.)
        #print(pos0)
        # circle poly
        circ = cylag.circle(x0=2., y0=2., r=1., n=180)
        pos1 = cylag.init_positions_3d(poly=circ, poly_area=np.pi,\
            xy_method=xy_method, grid_res=[2, 2, 2], xy_npart=16, xy_density=100,\
            z_method=z_method, z_npart=2, z_density=1., zmin=0., zmax=2.)
        # poly
        poly = np.array(([2., 0.], [2.5, 0.], [3., 1.], [2., 0.5]))
        pos2 = cylag.init_positions_3d(poly=poly, poly_area=1.,\
            xy_method=xy_method, grid_res=[2, 2, 2], xy_npart=16, xy_density=100,\
            z_method=z_method, z_npart=2, z_density=1., zmin=0., zmax=2.)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # init 0
            plot_polygon(ax, rectangle, c='b')
            plt.plot(pos0[:, 0], pos0[:, 1], lw=0., marker='o', markersize=10, c='b')
            # init 1
            plot_polygon(ax, circ, c='r')
            plt.plot(pos1[:, 0], pos1[:, 1], lw=0., marker='o', markersize=10, c='r')
            # init 2
            plot_polygon(ax, poly, c='g')
            plt.plot(pos2[:, 0], pos2[:, 1], lw=0., marker='o', markersize=10, c='g')
            plt.show()

    def test_init_2d(self):
        # grid resolution
        npart = 100
        res = int(np.sqrt(npart))
        # rectangle poly
        rectangle = np.array(([0., 0.], [1., 0.], [1., 1.], [0., 1.]))
        pos0 = cylag.init_positions_2d(poly=rectangle, poly_area=1., grid_res=(res, res))
        # circle poly
        circ = cylag.circle(x0=2., y0=2., r=1., n=25)
        pos1 = cylag.init_positions_2d(poly=circ, poly_area=1., grid_res=(res, res))
        # poly
        poly = np.array(([2., 0.], [2.5, 0.], [3., 1.], [2., 0.5]))
        pos2 = cylag.init_positions_2d(poly=poly, poly_area=1., grid_res=(res, res))
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # init 0
            plot_polygon(ax, rectangle, c='b')
            plt.plot(pos0[:, 0], pos0[:, 1], lw=0., marker='o', markersize=10, c='b')
            # init 1
            plot_polygon(ax, circ, c='r')
            plt.plot(pos1[:, 0], pos1[:, 1], lw=0., marker='o', markersize=10, c='r')
            # init 2
            plot_polygon(ax, poly, c='g')
            plt.plot(pos2[:, 0], pos2[:, 1], lw=0., marker='o', markersize=10, c='g')
            plt.show()

    def test_init_3d(self):
        # grid resolution
        npart = 100
        res = int(np.sqrt(npart))
        zmin = -1.
        zmax = 10.
        # rectangle poly
        rectangle = np.array(([0., 0.], [1., 0.], [1., 1.], [0., 1.]))
        pos0 = cylag.init_positions_3d(poly=rectangle, poly_area=1., grid_res=(res, res, res), zmin=zmin, zmax=zmax)
        # circle poly
        circ = cylag.circle(x0=2., y0=2., r=1., n=25)
        pos1 = cylag.init_positions_3d(poly=circ, poly_area=1.,grid_res=(res, res, res), zmin=zmin, zmax=zmax)
        # poly
        poly = np.array(([2., 0.], [2.5, 0.], [3., 1.], [2., 0.5]))
        pos2 = cylag.init_positions_3d(poly=poly, poly_area=1., grid_res=(res, res, res), zmin=zmin, zmax=zmax)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # init 0
            plot_polygon(ax, rectangle, c='b')
            plt.plot(pos0[:, 0], pos0[:, 1], lw=0., marker='o', markersize=10, c='b')
            # init 1
            plot_polygon(ax, circ, c='r')
            plt.plot(pos1[:, 0], pos1[:, 1], lw=0., marker='o', markersize=10, c='r')
            # init 2
            plot_polygon(ax, poly, c='g')
            plt.plot(pos2[:, 0], pos2[:, 1], lw=0., marker='o', markersize=10, c='g')
            plt.show()

if __name__ == "__main__":
    unittest.main()
