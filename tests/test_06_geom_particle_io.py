# -*- coding: utf-8 -*-
import os
import unittest
import time
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt

PRINTOUT = False

class CheckParticleIO(unittest.TestCase):

    def test_fset_IO_t2d(self):
        root = path.dirname(__file__)
        particle_file = path.join(root, "data", "particles_2d.txt")

        # Load with particle IO
        t0 = time.time()
        part = cylag.ParticlesIO.from_cylag_txt(particle_file)
        t1 = time.time()
        print("CPU Time : ", t1-t0)
        
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            rec = 10
            xx = part.xp[rec][:]
            yy = part.yp[rec][:]
            ax.scatter(xx, yy, c='r', lw=0., s=10., label="$x_p(t)$")
            plt.show()

        # clean
        del part

if __name__ == "__main__":
    unittest.main()
