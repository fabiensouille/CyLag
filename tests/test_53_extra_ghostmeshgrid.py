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
from matplotlib import rc

PRINTOUT = False

class CheckGhostMeshGrid(unittest.TestCase):

    def test_1d_ghost_mesh_grid(self):
        # primal mesh (eulerian space)
        xmin = -2.1
        xmax = 21.4
        primal_x = np.arange(xmin, xmax, 0.7)
        # length of interaction (to compute linear transform from primal to ghost)
        h = 1.
        # number of ghost cells
        ghost_mesh = cylag.GhostMeshGrid1D(np.min(primal_x), np.max(primal_x), h)
        # ghost mesh
        ghost_x = np.arange(0, ghost_mesh.ng, 1)
        # test
        x_test_primal = 5.
        x_test_ghost = ghost_mesh.primal_to_ghost(x_test_primal)
        x_test_primal0 = ghost_mesh.ghost_to_primal(x_test_ghost)
        np.testing.assert_equal(x_test_primal0, x_test_primal)
        np.testing.assert_equal(x_test_ghost, 7.1)
        if PRINTOUT:        
            print("x_test_primal :", x_test_primal)
            print("x_test_ghost :", x_test_ghost)
            print("x_test_primal0 :", x_test_primal0)
            print("tranform =", np.asarray(ghost_mesh.transform))
            print("ng =", ghost_mesh.ng)
            print("ghost_x =", ghost_x)
        # plot
        if PRINTOUT:
            cylag.set_rcparams()
            rc('legend', framealpha=0.9)
            color,_,_ = cylag.get_default_color_palette()
            fig, ax = plt.subplots(1, 1, figsize=(8., 2.))
            # plot meshes
            ax.plot(primal_x, 0.*np.ones(len(primal_x)), marker='x', c=color[1], markersize=10, lw=0.)
            ax.plot(ghost_x, np.ones(len(ghost_x)), marker='s', c=color[2], markersize=6, lw=0.)
            # plot ghost mesh
            x_ghost = ghost_mesh.primal_to_ghost(primal_x)
            ax.plot(x_ghost, np.ones(len(x_ghost)), marker='x', c=color[3], markersize=6, lw=0.)
            for i in range(len(x_ghost)):
                ax.plot([primal_x, x_ghost], [0., 1.], c=color[3], lw=0.1, ls=":")
            #ax.set_xlim([xmin-0.5, xmax+0.5])
            ax.set_ylim([-0.5, 1.5])
            plt.grid()
            plt.show()
            plt.close(fig)
        
if __name__ == "__main__":
    unittest.main()
