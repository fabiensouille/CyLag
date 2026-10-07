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

def plot_scheduler(ax, scheduler, text, yplot):
    simutime = cylag.SimuTime(
        print_progress=False,
        final_time=5.,
        time_step=5./35.)
    while not simutime.is_finished:
        ax.plot([simutime.time,], [yplot,], 'bx')
        if scheduler.now(simutime):
            ax.plot([simutime.time], [yplot], 'ro')
        simutime.increment()
    ax.text(0., yplot+0.25, text)

class CheckScheduler(unittest.TestCase):

    def test_scheduler_plot(self):
        if PRINTOUT:
            # Plot scheduler
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            plot_scheduler(ax, cylag.schedules(time_delta=1.5), "Every 1.5 seconds", yplot=1)
            plot_scheduler(ax, cylag.schedules(iteration_delta=5), "Every 5 time iterations", yplot=2)
            plot_scheduler(ax, cylag.schedules(times=[0.5, 1., 2.5, 3.]), "At given times", yplot=3)
            plot_scheduler(ax, cylag.schedules(iterations=[10, 12, 14, 16]), "At given time iterations", yplot=4)
            plot_scheduler(ax, cylag.schedules(count=4), "4 equaly time-separated", yplot=5)
            plot_scheduler(ax, cylag.schedules(always=True), "Always", yplot=6)
            plt.grid()
            plt.show()
        else:
            pass

    def test_scheduler_time_delta(self):
        scheduler = cylag.schedules(time_delta=1.5)  
        scheduler_times = []
        scheduler_times_ref = [0.0, 1.4285714285714282, 2.999999999999999, 4.428571428571428]
        simutime = cylag.SimuTime(
            print_progress=False,
            final_time=5.,
            time_step=5./35.)
        while not simutime.is_finished:
            if scheduler.now(simutime):
                scheduler_times.append(simutime.time)
            simutime.increment()
        if PRINTOUT:
            print(scheduler_times)
        np.testing.assert_equal(scheduler_times, scheduler_times_ref)

    def test_scheduler_iteration_delta(self):
        scheduler = cylag.schedules(iteration_delta=5)
        scheduler_times = []
        scheduler_times_ref = [0.0, 0.7142857142857142, 1.4285714285714282, 2.1428571428571423,\
                               2.8571428571428563, 3.5714285714285703, 4.285714285714285]
        simutime = cylag.SimuTime(
            print_progress=False,
            final_time=5.,
            time_step=5./35.)
        while not simutime.is_finished:
            if scheduler.now(simutime):
                scheduler_times.append(simutime.time)
            simutime.increment()
        if PRINTOUT:
            print(scheduler_times)
        np.testing.assert_equal(scheduler_times, scheduler_times_ref)

    def test_scheduler_times(self):
        scheduler = cylag.schedules(times=[0.5, 1., 2.5, 3.])
        scheduler_times = []
        scheduler_times_ref = [0.42857142857142855, 0.9999999999999998, 2.428571428571428, 2.999999999999999]
        simutime = cylag.SimuTime(
            print_progress=False,
            final_time=5.,
            time_step=5./35.)
        while not simutime.is_finished:
            if scheduler.now(simutime):
                scheduler_times.append(simutime.time)
            simutime.increment()
        if PRINTOUT:
            print(scheduler_times)
        np.testing.assert_equal(scheduler_times, scheduler_times_ref)

    def test_scheduler_iterations(self):
        scheduler = cylag.schedules(iterations=[10, 12, 14, 16])
        scheduler_times = []
        scheduler_times_ref =[1.4285714285714282, 1.7142857142857137, 1.9999999999999993, 2.285714285714285]
        simutime = cylag.SimuTime(
            print_progress=False,
            final_time=5.,
            time_step=5./35.)
        while not simutime.is_finished:
            if scheduler.now(simutime):
                scheduler_times.append(simutime.time)
            simutime.increment()
        if PRINTOUT:
            print(scheduler_times)
        np.testing.assert_equal(scheduler_times, scheduler_times_ref)

    def test_scheduler_count(self):
        scheduler = cylag.schedules(count=4)
        scheduler_times = []
        scheduler_times_ref = [0.0, 1.571428571428571, 3.2857142857142847, 4.857142857142858]
        simutime = cylag.SimuTime(
            print_progress=False,
            final_time=5.,
            time_step=5./35.)
        while not simutime.is_finished:
            if scheduler.now(simutime):
                scheduler_times.append(simutime.time)
            simutime.increment()
        if PRINTOUT:
            print(scheduler_times)
        np.testing.assert_equal(scheduler_times, scheduler_times_ref)

    def test_scheduler_always(self):
        scheduler = cylag.schedules(always=True)
        scheduler_times = []
        simutime = cylag.SimuTime(
            print_progress=False,
            final_time=5.,
            time_step=5./35.)
        while not simutime.is_finished:
            if scheduler.now(simutime):
                scheduler_times.append(simutime.time)
            simutime.increment()
        if PRINTOUT:
            print(scheduler_times)

if __name__ == "__main__":
    unittest.main()
