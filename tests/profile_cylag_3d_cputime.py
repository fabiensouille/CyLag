# -*- coding: utf-8 -*-
"""
ParticleTracking algo - t3d_particles case

"""
import time
import numpy as np
from os import path, environ
import matplotlib.pylab as plt
import matplotlib.tri as mtri
import cylag
from data_manip.extraction.telemac_file import TelemacFile
from postel.plot2d import *
import pstats, cProfile

def benchmark(res=100):
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
    final_time = 4000.
    parameters = {
        'final_time': final_time, 
        'time_step': 5.,
        'time_scheme': 1,
        'frozen_eulerian_fields': True,
        'model': 1,
        'boundary_conditions': True,
        'listing': False,
        'output_file': False,
        }

    # traking algo run
    cylag_solver = cylag.Solver(fset, pset, parameters)

    profiler = cProfile.Profile()
    profiler.enable()

    t0 = time.time()
    cylag_solver.solve()
    t1 = time.time()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('tottime').print_stats()
    stats.sort_stats('tottime').dump_stats("Profile_stats")

    cpu_time = t1 - t0
    print("CPU Time : ", cpu_time)
    return cpu_time

if __name__ == "__main__":

    npart = [50, 200, 500, 1000]
    ref_times = [0.14766812324523926, 0.4311378002166748, 0.9918620586395264, 1.9053785800933838]
    cpu_times = []

    for i, nn in enumerate(npart):
        cpu_time = benchmark(int(np.sqrt(nn)))
        cpu_times.append(cpu_time)

    print(cpu_times)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(npart, ref_times, ls='--', marker='o', c='r', label="Ref")
    ax.plot(npart, cpu_times, marker='x', c='b', label="CPU Times")

    plt.grid()
    plt.legend()
    plt.show()
    plt.close(fig)

