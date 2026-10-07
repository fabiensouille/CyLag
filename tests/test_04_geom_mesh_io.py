# -*- coding: utf-8 -*-
import os
import unittest
import time
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

PRINTOUT = False

class CheckTriangularMeshIO(unittest.TestCase):

    def test_mesh_IO_t2d(self):
        root = path.dirname(__file__)
        t2d_file = path.join(root, "data", "r2d_flume.slf")
        bnd_file = path.join(root, "data", "geo_flume.cli")
        # field_set
        fset = cylag.EulerianFieldSet.from_telemac2d(t2d_file, bnd_file=bnd_file)
        cylagtri = fset.triangular_mesh
        tri = Triangulation(cylagtri.x, cylagtri.y, cylagtri.triangles)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot cylag boundaries
            cylag.plot_cylag_boundary_points(ax, cylagtri)
            cylag.plot_cylag_boundary_edges(ax, cylagtri)
            plt.show()
        # clean
        del tri
        del fset
        del cylagtri

    def test_mesh_IO_hdf5_write(self):
        root = path.dirname(__file__)
        t2d_file = path.join(root, "data", "r2d_flume.slf")
        bnd_file = path.join(root, "data", "geo_flume.cli")
        # field_set
        fset = cylag.EulerianFieldSet.from_telemac2d(t2d_file, bnd_file=bnd_file)
        cylagtri = fset.triangular_mesh
        # save hdf5 cylag mesh
        hdf5_file = path.join(root, "data", "geo_flume.h5")
        cylagtri.save_mesh_hdf5(file_name=hdf5_file)
        # clean 
        del fset
        del cylagtri
        
    def test_mesh_IO_hdf5_read(self):
        root = path.dirname(__file__)
        h2d_file = path.join(root, "data", "geo_flume.h5")
        # field_set
        cylagtri = cylag.TriangularMesh.from_hdf5(h2d_file)
        tri = Triangulation(cylagtri.x, cylagtri.y, cylagtri.triangles)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot cylag boundaries
            cylag.plot_cylag_boundary_points(ax, cylagtri)
            cylag.plot_cylag_boundary_edges(ax, cylagtri)
            plt.show()
        # clean 
        del tri
        del cylagtri
       
if __name__ == "__main__":
    unittest.main()
