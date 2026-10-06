# cython: profile=False
#
# -------------------------------------------------------------------------------------------------
#  Project name: CyLag
#  Copyright (C) 2023 Fabien Souille
#
#  This program is free: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License published by the Free Software Foundation,
#  either version 3 of the license, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License for details.
#
#  You should have received a copy of the GNU General Public License
#  with this program. If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------------------------------
#
import numpy as np

cdef class Scheduler:
    """ Base scheduler class """
    def __init__(self):
        pass

    cpdef bint now(Scheduler self, SimuTime simutime):
        """ Returns True if scheduler trigger condition is valid """
        print("I entered class scheduler")
        return False

cdef class TimeDeltaScheduler(Scheduler):
    """ Schedule a task every time_delta """
    def __init__(self, time_delta, ini_time=None):
        self.time_delta = time_delta
        self.next_time = ini_time
        self.iteration = -1

    cpdef bint now(TimeDeltaScheduler self, SimuTime simutime):
        if simutime.time <= self.next_time and \
                self.next_time < simutime.time + simutime.time_step:
            self.next_time += self.time_delta
            self.iteration += 1
            return True
        else:
            return False

cdef class IterationDeltaScheduler(Scheduler):
    """Schedule a task for every constant time iteration delta"""
    def __init__(self, iteration_delta):
        self.iteration = -1
        self.iteration_delta = iteration_delta

    cpdef bint now(IterationDeltaScheduler self, SimuTime simutime):
        if simutime.iteration % self.iteration_delta == 0:
            self.iteration += 1
            return True

cdef class IterationScheduler(Scheduler):
    """Schedule a task for a list of time iterations"""
    def __init__(self, iterations):
        self.iteration = -1
        self.iterations = iterations

    cpdef bint now(IterationScheduler self, SimuTime simutime):
        if simutime.iteration in self.iterations:
            self.iteration += 1
            return True
        else:
            return False

cdef class TimeScheduler(Scheduler):
    """Schedule a task for a list of times"""
    def __init__(self, times):
        self.iteration = 0
        self.times = np.array(times, dtype='d')
        self.ntimes = len(times)

    cpdef bint now(TimeScheduler self, SimuTime simutime):
        cdef bint remaining = self.iteration < self.ntimes
        if remaining and simutime.time <= self.times[self.iteration] and \
                self.times[self.iteration] < simutime.time + simutime.time_step:
            self.iteration += 1
            return True
        else:
            return False

cdef class CountScheduler(Scheduler):
    """Schedule a number of task between 0 and final_time"""
    def __init__(self, count):
        self.iteration = 0
        self.count = count
        self.is_ready = True

    cpdef bint now(CountScheduler self, SimuTime simutime):
        cdef:
            bint remaining = self.iteration < self.count
            double[:] times = np.linspace(0, simutime.final_time, self.count)

        if remaining and simutime.time <= times[self.iteration] and \
                times[self.iteration] < simutime.time + simutime.time_step:
            self.iteration += 1
            return True
        else:
            return False

cdef class AlwaysScheduler(Scheduler):
    """Schedule a task for every iteration"""
    def __init__(self):
        pass

    cpdef bint now(AlwaysScheduler self, SimuTime simutime):
        self.iteration += 1
        return True

cpdef Scheduler schedules(ini_time=0., time_delta=None, iteration_delta=None, times=None,
                          iterations=None, count=None, never=None,
                          always=None):
    """
    Create a scheduler for a task.

    At most two (simutime and only one argument) of the 7 arguments available must be passed. Each
    argument creates a different type of scheduler:

    Parameters
    ----------
    ini_time : float
        Time at which the scheduler is initilised.
    time_delta : float
        Time delta between two tasks.
    iteration_delta: int
        Number of iterations between two tasks.
    times: list (float)
        Task times.
    iterations: list (int)
        Task time iterations.
    count: int
        Number of tasks, equally time-separated.
    never: bool
        Task is never triggered.
    always: bool
        Task is triggered at each time iteration.

    Returns
    -------
    Scheduler
        Scheduler object.
    """

    # Check arguments.
    args = [time_delta, iteration_delta, times, iterations, count, never, always]
    args_count = len(args) - args.count(None) 
    if args_count == 0:
        raise ValueError("At least one of these argument required: \
        time_delta, iteration_delta, times, iterations, count, never, always")
    if args_count > 1:
        raise ValueError("Only one of these arguments possible: \
        time_delta, iteration_delta, times, iterations, count, never, always , got {}".format(args_count))

    if time_delta is not None:
        return TimeDeltaScheduler(time_delta, ini_time)

    elif iteration_delta is not None:
        return IterationDeltaScheduler(iteration_delta)

    elif times is not None:
        return TimeScheduler(times)

    elif iterations is not None:
        return IterationScheduler(iterations)

    elif count is not None:
        return CountScheduler(count)

    elif always is not None:
        return AlwaysScheduler()

    raise ValueError("No argument found")
