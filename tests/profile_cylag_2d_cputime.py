# -*- coding: utf-8 -*-
"""
Test ParticleTracking algo - t2d_tide case

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

def benchmark(res=50):
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
    final_time = 4500.
    parameters = {
        'final_time': final_time, 
        'time_step': 10., 
        'time_scheme': 1,
        'frozen_eulerian_fields': False,
        'model': 1,
        'diffusion': False,
        'horizontal_diffusivity': 0.,
        'vertical_diffusivity': 0.,
        'listing': False,
        'boundary_conditions': True,
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

    ref_times = [0.018800735473632812, 0.07321619987487793, 0.19443511962890625, 0.40314602851867676]
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

