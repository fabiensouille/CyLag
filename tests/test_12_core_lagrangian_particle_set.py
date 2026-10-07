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

class CheckLagrangianParticleSet(unittest.TestCase):

    def test_init_2d(self):
        position = np.array([[0., 0.], [1., 1.], [2., 3.]])
        velocity = np.array([[1., 1.], [0., 0.], [6., 4.]])
        fluidvel = np.array([[0., 0.], [6., 6.], [0., 0.]])
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, velocity, fluid_velocity_seen=fluidvel,\
            dim=2, initial_pool_size=5)
        # check init
        np.testing.assert_equal(position, pset.position[0:3, :])
        np.testing.assert_equal(velocity, pset.velocity[0:3, :])
        np.testing.assert_equal(fluidvel, pset.fluid_velocity_seen[0:3, :])
        # plot
        if PRINTOUT:
            for i in range(pset.npart):
                pset.show_status(i)
        # clean 
        del pset

    def test_init_3d(self):
        position = np.array([[0., 0., 0.], [1., 1., 1.], [2., 3., 3.]])
        velocity = np.array([[1., 1., 1.], [0., 0., 0.], [6., 4., 4.]])
        fluidvel = np.array([[0., 0., 0.], [6., 6., 6.], [0., 0., 0.]])
        # init particle set
        pset = cylag.LagrangianParticleSet(\
            position, velocity, fluid_velocity_seen=fluidvel,\
            dim=3, initial_pool_size=5)
        # check init
        np.testing.assert_equal(position, pset.position[0:3, :])
        np.testing.assert_equal(velocity, pset.velocity[0:3, :])
        np.testing.assert_equal(fluidvel, pset.fluid_velocity_seen[0:3, :])
        # plot
        if PRINTOUT:
            for i in range(pset.npart):
                pset.show_status(i)
        # clean 
        del pset

    def test_add(self):
        position = np.array([[0., 0.], [1., 1.], [2., 3.]])
        velocity = np.array([[1., 1.], [0., 0.], [6., 4.]])
        fluidvel = np.array([[0., 0.], [6., 6.], [0., 0.]])
        # initialize pset
        pset = cylag.LagrangianParticleSet(\
            position, velocity, fluid_velocity_seen=fluidvel,\
            dim=2, initial_pool_size=3, min_pool_size_update=2)
        # plot
        if PRINTOUT:
            for i in range(pset.npart):
                pset.show_status(i)
        # add particle
        new_pos = np.array([[5., 5.]])
        pset.add(new_pos)
        # checks
        np.testing.assert_equal(np.concatenate((position, new_pos), axis=0),\
                                pset.position[0:4, :])
        np.testing.assert_equal(pset.particle_diameter, pset.diameter[3])
        np.testing.assert_equal(pset.particle_volume, pset.volume[3])
        # plot
        if PRINTOUT:
            for i in range(pset.npart):
                pset.show_status(i)
        # clean 
        del pset

    def test_delete(self):
        position = np.array([[0., 0.], [1., 1.], [2., 3.]])
        velocity = np.array([[1., 1.], [0., 0.], [6., 4.]])
        fluidvel = np.array([[0., 0.], [6., 6.], [0., 0.]])
        # initialize pset
        pset = cylag.LagrangianParticleSet(\
            position, velocity, fluid_velocity_seen=fluidvel,\
            dim=2, initial_pool_size=3, min_pool_size_update=2)
        # plot
        if PRINTOUT:
            for i in range(pset.npart):
                pset.show_status(i)
        # del particle
        pset.delete(np.array([2], dtype='int32'))
        # checks
        np.testing.assert_equal(1, pset.inactive[2])
        # plot
        if PRINTOUT:
            for i in range(pset.npart):
                pset.show_status(i)
        # clean 
        del pset

    def test_add_performance(self):
        position = np.array([[0., 0.], [1., 1.], [2., 3.]])
        velocity = np.array([[1., 1.], [0., 0.], [6., 4.]])
        fluidvel = np.array([[0., 0.], [6., 6.], [0., 0.]])
        # initialize pset
        pset = cylag.LagrangianParticleSet(\
            position, velocity, fluid_velocity_seen=fluidvel,\
            dim=2, initial_pool_size=3, min_pool_size_update=1)
        # plot
        if PRINTOUT:
            for i in range(pset.npart):
                pset.show_status(i)
        # deleting particle 2
        pset.delete(np.array([2], dtype='int32'))
        # add particle
        new_pos = np.array([[5., 5.]])
        pset.add(new_pos)
        # plot
        if PRINTOUT:
            for i in range(pset.npart):
                pset.show_status(i)
        # clean 
        del pset

    def test_mem_alloc(self):
        nmod = 1000
        position = np.array([[-5., -5.]])
        velocity = np.array([[4., 4.]])
        fluidvel = np.array([[6., 6.]])
        # Case 1: heavy malloc / min pool size = 1 
        pset = cylag.LagrangianParticleSet(\
            position, velocity, fluid_velocity_seen=fluidvel,\
            dim=2, initial_pool_size=1, min_pool_size_update=1)
        t0 = time.time()
        for i in range(nmod):
            new_pos = np.array([[float(i), float(i)]])
            pset.add(new_pos)
        pset.delete(np.arange(1, nmod+1, dtype='int32'))
        t1 = time.time()
        cpu_time0 = t1 - t0
        # Case 1: heavy malloc / min pool size = 10
        pset = cylag.LagrangianParticleSet(\
            position, velocity, fluid_velocity_seen=fluidvel,\
            dim=2, initial_pool_size=1+nmod, min_pool_size_update=nmod)
        t0 = time.time()
        for i in range(nmod):
            new_pos = np.array([[float(i), float(i)]])
            pset.add(new_pos)
        pset.delete(np.arange(1, nmod+1, dtype='int32'))
        t1 = time.time()
        cpu_time1 = t1 - t0
        if PRINTOUT:
            print(" ")
            print("cpu_time0 :", cpu_time0)
            print("cpu_time1 :", cpu_time1)
        # clean 
        del pset

if __name__ == "__main__":
    unittest.main()
