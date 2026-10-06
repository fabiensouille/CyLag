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
from libc.limits cimport INT_MAX
import numpy as np
from libc.math cimport fmod

cdef class SimuTime:
    """
    Parameters
    ----------
        final_time : float, simulation final time
        time_step : float, time step of the resolution
        print_progress : bool, if True, print progress at each time step
        maxiteration : int, maximum number of iterations

    Attributes
    ----------
        time : float, current time
        time_step : float, time step
        final_time : float, simulation final time
        times : double[:], array of time steps
        nt : int, number of time steps
        iteration : int, current iteration
        maxiteration : int, maximum number of iterations
        is_finished : bool, simulation status
        print_progress : bool, if True, print progress at each time step
    """
    def __init__(self,
                 print_progress,
                 initial_time=0.,
                 final_time=0.,
                 time_step=0.,
                 maxiteration=INT_MAX):

        # check input values
        if final_time <= initial_time:
            raise ValueError("final_time must be greater than initial_time")
        if time_step <= 0.:
            raise ValueError("time_step must be greater than 0")
        if maxiteration <= 0:
            raise ValueError("maxiteration must be greater than 0")
        if final_time-initial_time < time_step:
            raise ValueError("final_time - initial_time must be greater than time_step")
        if final_time-initial_time==0.:
            raise ValueError("final_time - initial_time must be greater than 0")

        # initialize attributes
        self.time = initial_time
        self.initial_time = initial_time
        self.time_step = time_step
        self.final_time = final_time
        self.nt = int((final_time - initial_time)/time_step)
        self.times = np.linspace(initial_time, final_time, self.nt + 1)
        self.iteration = 0
        self.maxiteration = maxiteration
        self.is_finished = False
        self.print_progress = print_progress

    cpdef void increment(SimuTime self):
        """
        Increment time
        """
        self.time += self.time_step
        self.iteration += 1

        if self.time >= self.final_time or self.iteration >= self.maxiteration:
            self.is_finished = True

    cpdef void progress(SimuTime self):
        """
        Print progress
        """
        # convert time to days, hours, minutes, seconds
        cdef int jj, hh, mm, ss, percent, filled
        cdef double progress, duration
        cdef str progress_bar

        jj, hh, mm, ss = self.convert_time_to_jjhhmmss(self.time)

        duration = self.final_time - self.initial_time
        if duration <= 0.:
            progress = 1.
        else:
            progress = (self.time - self.initial_time) / duration
            progress = min(1., max(0., progress))

        percent = <int>(progress * 100.)
        filled = <int>(progress * 15.)
        progress_bar = "[{}{}{}] {:3d}%".format(
            "=" * (filled-1), ">" if filled > 0 else "", "." * (15 - filled), percent)

        print("Iter = {:10d} ; "\
              "Dt = {:4.4f}s ; "\
              "Time = {:02d}d {:02d}:{:02d}:{:02d} ; {}"\
              .format(self.iteration,
                      self.time_step,
                      jj, hh, mm, ss,
                      progress_bar))

    cpdef tuple convert_time_to_jjhhmmss(SimuTime self, double time_in_seconds):
        """
        Converts time in seconds to days, hours, minutes, and seconds
        """
        jj = int(time_in_seconds/86400.0)
        hh = int(fmod(time_in_seconds, 86400.0)/3600.0)
        mm = int(fmod(time_in_seconds, 3600.0)/60.0)
        ss = int(fmod(time_in_seconds, 60.0))
        return jj, hh, mm, ss