from ..core.simutime cimport SimuTime

"""
+-----------------------------+----------------------------------------+
| Class                       | Schedules a task ...                   |
+=============================+========================================+
| IterationDeltaScheduler     | every constant time iterations delta   |
+-----------------------------+----------------------------------------+
| TimeDeltaScheduler          | every constant time delta              |
+-----------------------------+----------------------------------------+
| IterationScheduler          | at given time iterations               |
+-----------------------------+----------------------------------------+
| TimeScheduler               | at given times                         |
+-----------------------------+----------------------------------------+
| CountScheduler              | a number times, equally time-separted  |
+-----------------------------+----------------------------------------+
| NeverScheduler              | never                                  |
+-----------------------------+----------------------------------------+
| AlwaysScheduler             | at every time iterations               |
+-----------------------------+----------------------------------------+
"""

cdef class Scheduler:
    cpdef bint now(Scheduler self, SimuTime simutime)
    cdef public int iteration

cdef class TimeDeltaScheduler(Scheduler):
    cdef:
        double time_delta
        double next_time
    cpdef bint now(TimeDeltaScheduler self, SimuTime simutime)

cdef class IterationDeltaScheduler(Scheduler):
    cdef public int iteration_delta
    cpdef bint now(IterationDeltaScheduler self, SimuTime simutime)

cdef class IterationScheduler(Scheduler):
    cdef public list iterations
    cpdef bint now(IterationScheduler self, SimuTime simutime)

cdef class TimeScheduler(Scheduler):
    cdef public double[:] times
    cdef public int ntimes
    cpdef bint now(TimeScheduler self, SimuTime simutime)

cdef class CountScheduler(Scheduler):
    cdef public int count
    cdef public bint is_ready
    cdef public double[:] times
    cpdef bint now(CountScheduler self, SimuTime simutime)

cdef class NeverScheduler(Scheduler):
    cpdef bint now(NeverScheduler self, SimuTime simutime)

cdef class AlwaysScheduler(Scheduler):
    cpdef bint now(AlwaysScheduler self, SimuTime simutime)

cpdef Scheduler schedules(ini_time=?, time_delta=?, iteration_delta=?, times=?,
                          iterations=?, count=?, never=?, always=?)
