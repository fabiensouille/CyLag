# -*- coding: utf-8 -*-
import os
import unittest
import time
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt
import matplotlib.tri as mtri

PRINTOUT = False
NON_REGRESSION_TEST = True

def run_cylag_2d(res=50, scheme=1, lsm=0, diffusion_model=0):
    root = path.dirname(__file__)
    # define field set
    t2d_file = path.join(root, "data", "r2d_test.slf")
    bnd_file = path.join(root, "data", "geo_test.cli")
    fset = cylag.EulerianFieldSet.from_telemac2d(t2d_file, bnd_file=bnd_file)
    circ = cylag.circle(x0=0., y0=500., r=50., n=50)
    position = cylag.init_positions_2d(poly=circ, grid_res=(res, res))
    npart = np.shape(position)[0]
    pset = cylag.LagrangianParticleSet(\
        position, dim=2, initial_pool_size=npart)
    # set general parameters for the traking algo
    final_time = 4000.
    parameters = {
        'final_time': final_time, 
        'time_step': 10., 
        'time_scheme': scheme,
        'frozen_eulerian_fields': False,
        'model': lsm,
        'rng_method': 1,
        'diffusion_model': diffusion_model,
        'particle_velocity_init': 1,
        'horizontal_diffusivity': 0.001,
        'vertical_diffusivity': 0.001,
        'listing': False,
        'boundary_conditions': True,
        'output_file': False,
        'output_printout_period': 10,
        }
    # traking algo run
    cylag_solver = cylag.Solver(fset, pset, parameters)
    t0 = time.time()
    cylag_solver.solve()
    t1 = time.time()
    cpu_time = t1 - t0
    return cpu_time

def run_cylag_3d(res=50, scheme=1, lsm=0, diffusion_model=0):
    root = path.dirname(__file__)
    t3d_file = path.join(root, "data", "r3d_test.slf")
    bnd_file = path.join(root, "data", "geo_test.cli")
    # field_set
    fset = cylag.EulerianFieldSet.from_telemac3d(\
        t3d_file, bnd_file=bnd_file)
    # define particles initial positions
    poly = np.array([[-100., 450.],\
                     [ 100., 450.],\
                     [ 100., 550.],\
                     [-100., 550.]])
    position = cylag.init_positions_3d(\
        poly=poly, grid_res=(res, res, 2), zmin=259., zmax=261.)
    npart = position.shape[0]
    pset = cylag.LagrangianParticleSet(\
        position, dim=3, initial_pool_size=100000)
    # set general parameters for the traking algo
    parameters = {
        'final_time': 4000., 
        'time_step': 10., 
        'time_scheme': scheme,
        'frozen_eulerian_fields': True,
        'model': lsm,
        'rng_method': 1,
        'diffusion_model': diffusion_model,
        'particle_velocity_init': 1,
        'horizontal_diffusivity': 0.001,
        'vertical_diffusivity': 0.001,
        'listing': False,
        'boundary_conditions': True,
        'output_file': False,
        'output_printout_period': 10,
        }
    # traking algo run
    cylag_solver = cylag.Solver(fset, pset, parameters)
    t0 = time.time()
    cylag_solver.solve()
    t1 = time.time()
    cpu_time = t1 - t0
    return cpu_time

class CheckPerformance2d(unittest.TestCase):

    def test_perf_t2d(self):
        root = path.dirname(__file__)
        npart = [200, 600]

        ref_times = np.array([0.10, 0.26])
        ref_times_release = np.array([0.085, 0.242])
        ref_times_debug = np.array([0.094, 0.259])
        ref_times_check = ref_times + 0.1*ref_times[-1] # margin for check
        cpu_times_0 = np.empty(len(npart), dtype='d')
        cpu_times_1 = np.empty(len(npart), dtype='d')
        cpu_times_2 = np.empty(len(npart), dtype='d')
        cpu_times_3 = np.empty(len(npart), dtype='d')
        cpu_times_m = np.empty(len(npart), dtype='d') 

        for i, nn in enumerate(npart):
            cpu_times_0[i] = run_cylag_2d(int(np.sqrt(nn)), scheme=1, lsm=1, diffusion_model=0)
            cpu_times_1[i] = run_cylag_2d(int(np.sqrt(nn)), scheme=1, lsm=1, diffusion_model=1)
            cpu_times_2[i] = run_cylag_2d(int(np.sqrt(nn)), scheme=5, lsm=2, diffusion_model=0)
            cpu_times_3[i] = run_cylag_2d(int(np.sqrt(nn)), scheme=5, lsm=2, diffusion_model=1)
            cpu_times_m[i] = np.max([cpu_times_0[i], cpu_times_1[i], cpu_times_2[i], cpu_times_3[i]])

        # check non regression
        if NON_REGRESSION_TEST:
            np.testing.assert_array_less(cpu_times_0, ref_times_check)
            np.testing.assert_array_less(cpu_times_1, ref_times_check)
            np.testing.assert_array_less(cpu_times_2, ref_times_check)
            np.testing.assert_array_less(cpu_times_3, ref_times_check)

        # plot
        if PRINTOUT:
            print("CPU Times 0 =", cpu_times_0)
            print("CPU Times 1 =", cpu_times_1)
            print("CPU Times 2 =", cpu_times_2)
            print("CPU Times 3 =", cpu_times_3)
            print("CPU Times max =", cpu_times_m)
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(npart, ref_times, ls='--', marker='o', c='k', label="Ref")
            ax.plot(npart, ref_times_release, ls='-.', marker='s', c='0.5', label="Release")
            ax.plot(npart, ref_times_debug, ls=':', marker='s', c='0.5', label="Debug")
            ax.plot(npart, cpu_times_0, marker='x', c='b', label="CPU Times (LSM1 - no diffusion)")
            ax.plot(npart, cpu_times_1, marker='x', c='r', label="CPU Times (LSM1 - constant K)")
            ax.plot(npart, cpu_times_2, marker='x', c='c', label="CPU Times (LSM2 - no diffusion)")
            ax.plot(npart, cpu_times_3, marker='x', c='m', label="CPU Times (LSM2 - constant K)")
            plt.grid()
            plt.legend()
            plt.show()
            plt.close(fig)

    def test_perf_t3d(self):
        root = path.dirname(__file__)
        npart = [200, 600]

        ref_times = np.array([0.33, 0.80])
        ref_times_release = np.array([0.26, 0.7])
        ref_times_debug = np.array([0.30, 0.77])
        ref_times_check = ref_times + 0.1*ref_times[-1]
        cpu_times_0 = np.empty(len(npart), dtype='d')
        cpu_times_1 = np.empty(len(npart), dtype='d')
        cpu_times_m = np.empty(len(npart), dtype='d')

        for i, nn in enumerate(npart):
            cpu_times_0[i] = run_cylag_3d(int(np.sqrt(nn)), scheme=1, lsm=1, diffusion_model=0)
            cpu_times_1[i] = run_cylag_3d(int(np.sqrt(nn)), scheme=1, lsm=1, diffusion_model=1)
            cpu_times_m[i] = np.max([cpu_times_0[i], cpu_times_1[i]])

        # check non regression
        if NON_REGRESSION_TEST:
            np.testing.assert_array_less(cpu_times_0, ref_times_check)
            np.testing.assert_array_less(cpu_times_1, ref_times_check)

        # plot
        if PRINTOUT:
            print("CPU Times 0 =", cpu_times_0)
            print("CPU Times 1 =", cpu_times_1)
            print("CPU Times max =", cpu_times_m)
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(npart, ref_times, ls='--', marker='o', c='k', label="Ref")
            ax.plot(npart, ref_times_release, ls='-.', marker='s', c='0.5', label="Release")
            ax.plot(npart, ref_times_debug, ls=':', marker='s', c='0.5', label="Debug")
            ax.plot(npart, cpu_times_0, marker='x', c='b', label="CPU Times (no diffusion)")
            ax.plot(npart, cpu_times_1, marker='x', c='r', label="CPU Times (constant K)")
            plt.grid()
            plt.legend()
            plt.show()
            plt.close(fig)

if __name__ == "__main__":
    unittest.main()
