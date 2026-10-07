# -*- coding: utf-8 -*-
import random
import unittest
import time
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt

PRINTOUT = False

class CheckLocalizerZ(unittest.TestCase):

    def test_zlocalizer_compute_llindex(self):
        root = path.dirname(__file__)
        # set mesh
        zs, zb, nl = 10., 0., 11
        z_layers = np.linspace(zb, zs, nl)
        # find lower layer index of z
        z = (zs-zb)*0.55
        lz1 = cylag.compute_lower_layer_index(nl, zs, zb, z)
        lz2 = cylag.c_compute_lower_layer_index(nl, zs, zb, z)
        np.testing.assert_equal(lz1, 5)
        np.testing.assert_equal(lz2, 5)

    def test_zlocalizer_compute_llindex_cputimes(self):
        root = path.dirname(__file__)
        # set mesh
        zs, zb, nl = 10., 0., 11
        z_layers = np.linspace(zb, zs, nl)
        # find lower layer index of z
        nruns = 1000
        z = []
        for i in range(nruns):
            random.seed(i)
            z.append((zs-zb)*random.random())
        # run py
        t0 = time.time()
        for i in range(nruns):
            lz1 = cylag.compute_lower_layer_index(nl, zs, zb, z[i])
        t1 = time.time()
        cputime_py = t1 - t0
        # run cy
        t0 = time.time()
        for i in range(nruns):
            lz2 = cylag.c_compute_lower_layer_index(nl, zs, zb, z[i])
        t1 = time.time()
        cputime_cy = t1 - t0
        # run times
        if PRINTOUT:
            print("cputime Python = ", cputime_py)
            print("cputime Cython = ", cputime_cy)
            print("Cython is x{} faster".format(cputime_py/cputime_cy))

    def test_zlocalizer_compute_llindex_array(self):
        root = path.dirname(__file__)
        # set mesh
        zs, zb, nl = 10., 0., 11
        z_layers = np.linspace(zb, zs, nl)
        # find lower layer index of z
        z_arr = np.array([(zs-zb)*0.55, (zs-zb)*0.74])
        lz_arr1 = cylag.compute_lower_layer_index(\
            nl, zs*np.ones(2), zb*np.ones(2), z_arr)
        lz_arr2 = cylag.c_compute_lower_layer_index_array(\
            nl, zs*np.ones(2), zb*np.ones(2), z_arr)
        np.testing.assert_equal(np.asarray(lz_arr1), np.asarray([5, 7]))
        np.testing.assert_equal(np.asarray(lz_arr2), np.asarray([5, 7]))

    def test_zlocalizer_compute_llindex_array_cputimes(self):
        root = path.dirname(__file__)
        # set mesh
        zs, zb, nl = 10., 0., 11
        z_layers = np.linspace(zb, zs, nl)
        # find lower layer index of z
        nruns = 1000
        npoin = 1000
        z = []
        for i in range(nruns):
            random.seed(i)
            z.append(np.asarray([(zs-zb)*random.random() for i in range(npoin)]))
        # run py
        t0 = time.time()
        for i in range(nruns):
            lz = cylag.compute_lower_layer_index(\
                nl, zs*np.ones(npoin), zb*np.ones(npoin), z[i])
        t1 = time.time()
        cputime_py = t1 - t0
        # run cy
        t0 = time.time()
        for i in range(nruns):
            lz = cylag.c_compute_lower_layer_index_array(\
                nl, zs*np.ones(npoin), zb*np.ones(npoin), z[i])
        t1 = time.time()
        cputime_cy = t1 - t0
        # run times
        if PRINTOUT:
            print("cputime Python = ", cputime_py)
            print("cputime Cython = ", cputime_cy)
            print("Cython is x{} faster".format(cputime_py/cputime_cy))

    def test_zlocalizer_compute_llindex_random(self):
        root = path.dirname(__file__)
        # set mesh
        zs, zb, nl = 10., 0., 11
        z_layers = np.linspace(zb, zs, nl)
        # find lower layer index of z
        z = (zs-zb)*random.random()
        #lz = cylag.compute_lower_layer_index(nl, zs, zb, z)
        lz = cylag.c_compute_lower_layer_index(nl, zs, zb, z)
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_aspect('equal')
            ax.plot(z_layers, np.ones(nl),
                    c='r', lw=1., marker='o', markersize=5, label="zlayers")
            ax.plot(z, 1., c='b', lw=1., marker='o', markersize=5, label="target")
            ax.plot(z_layers[lz], 1.,
                    c='b', lw=1., marker='x', markersize=5, label="lower_layer_index")
            ax.set_ylim([0, 2.5])
            ax.set_xlim([-1, 11])
            plt.legend()
            plt.show()
            plt.close(fig)
        # find lower layer index of z list
        z_arr = (zs-zb)*np.array([random.random(), random.random()])
        #lz_arr = cylag.compute_lower_layer_index(nl, zs, zb, z_arr)
        lz_arr = cylag.c_compute_lower_layer_index_array(nl, zs*np.ones(2), zb*np.ones(2), z_arr)
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_aspect('equal')
            ax.plot(z_layers, np.ones(nl), c='r', lw=1., marker='o', markersize=5, label="zlayers")
            for i, z in enumerate(z_arr):
                ax.plot(z, 1., c='b', lw=1., marker='o', markersize=5)
                ax.plot(z_layers[lz_arr[i]], 1., c='b', lw=1., marker='x', markersize=5)
            ax.set_ylim([0, 2.5])
            ax.set_xlim([-1, 11])
            plt.legend()
            plt.show()
            plt.close(fig)

if __name__ == "__main__":
    unittest.main()
