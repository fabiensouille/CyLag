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

"""
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   336249    0.700    0.000    1.517    0.000 xylocalizer.pyx:338(xy_localize_point_from_path)
  1086300    0.427    0.000    0.616    0.000 intersections.pyx:106(intersec2d_segment_segment)
   767719    0.375    0.000    0.546    0.000 eulerian_field_set.pyx:1007(_interpolate_velocity_2d)
   767719    0.295    0.000    3.072    0.000 update_state_lsm1.pyx:30(update_state_lsm1)
  1178112    0.271    0.000    0.386    0.000 utils.pyx:387(point_in_triangle_dotp)
   767779    0.236    0.000    2.105    0.000 xylocalizer.pyx:27(xy_localize)
      400    0.191    0.000    3.313    0.008 solver.pyx:402(forward)
  4345200    0.189    0.000    0.189    0.000 utils.pyx:256(orientation)
  1178112    0.167    0.000    0.553    0.000 xylocalizer.pyx:126(point_in_tri)
   767719    0.127    0.000    0.673    0.000 update_state_lsm1.pyx:131(euler_lsm1)
  2874958    0.116    0.000    0.116    0.000 utils.pyx:360(pintri)
  1536638    0.094    0.000    0.094    0.000 eulerian_field_set.pyx:662(_get_tmp_from_id)
   767719    0.045    0.000    0.046    0.000 compute_bc.pyx:35(compute_boundary_condition)
   767719    0.043    0.000    0.043    0.000 eulerian_field_set.pyx:732(_cache_interp_factors)
   767719    0.035    0.000    0.035    0.000 eulerian_field_set.pyx:756(_use_cached_factors)
      400    0.003    0.000    0.004    0.000 eulerian_field_set.pyx:820(_time_interpolation)
      990    0.001    0.000    0.001    0.000 lagrangian_particle_set.pyx:754(delete_single)
        1    0.000    0.000    3.313    3.313 solver.pyx:371(solve)
     2400    0.000    0.000    0.000    0.000 eulerian_field_set.pyx:577(_get_field_slice)
       60    0.000    0.000    0.000    0.000 intersections.pyx:27(compute_symetric_point2d)
     1200    0.000    0.000    0.000    0.000 eulerian_field_set.pyx:627(_get_field_from_id)
      400    0.000    0.000    0.000    0.000 eulerian_field_set.pyx:799(_set_record)
      400    0.000    0.000    0.000    0.000 zlocalizer.pyx:125(c_compute_lower_index)
       60    0.000    0.000    0.000    0.000 utils.pyx:83(normal2)
       60    0.000    0.000    0.000    0.000 utils.pyx:61(sub2)
      400    0.000    0.000    0.000    0.000 simutime.pyx:77(increment)
      120    0.000    0.000    0.000    0.000 utils.pyx:43(dot2)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.000    0.000 solver.pyx:576(flush_statistics)
"""

def benchmark(res=10):
    root = path.dirname(__file__)
    # define eulerian space from telemac3d results
    t2d_file = path.join(root, "data", "r2d_test.slf")
    bnd_file = path.join(root, "data", "geo_test.cli")
    fset = cylag.EulerianFieldSet.from_telemac2d(t2d_file, bnd_file=bnd_file)

    # define particles initial positions
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
        'time_scheme': 1,
        'frozen_eulerian_fields': False,
        'listing': False,
        'boundary_conditions': True,
        'output_file': False,
        }

    # traking algo run
    cylag_solver = cylag.Solver(fset, pset, parameters)

    profiler = cProfile.Profile()
    profiler.enable()
    
    cylag_solver.solve()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('tottime').print_stats()
    stats.sort_stats('tottime').dump_stats("Profile_stats")

if __name__ == "__main__":

    benchmark(50)
    #cProfile.run("benchmark(10)", filename="Profile.txt", sort=-1)

