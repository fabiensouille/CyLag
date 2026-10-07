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

class CheckFsetIO(unittest.TestCase):

    def test_fset_IO_t2d(self):
        root = path.dirname(__file__)
        t2d_file = path.join(root, "data", "r2d_flume.slf")
        bnd_file = path.join(root, "data", "geo_flume.cli")
        # field_set
        fset0 = cylag.EulerianFieldSet.from_telemac2d(t2d_file, bnd_file=bnd_file)
        # save fset in h5
        fset_file = path.join(root, "data", "cylag_fset_file.h5")
        mesh_file = path.join(root, "data", "cylag_mesh_file.h5")
        fset0.save_hdf5(fset_file=fset_file, mesh_file=mesh_file)
        # load fset from h5
        fset = cylag.EulerianFieldSet.from_hdf5(fset_file=fset_file, mesh_file=mesh_file)
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
        del fset0
        del fset
        del cylagtri
        os.system("rm {}".format(fset_file))
        os.system("rm {}".format(mesh_file))

if __name__ == "__main__":
    unittest.main()
