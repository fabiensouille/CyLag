# -*- coding: utf-8 -*-
"""
Scheduler
===============================================================================

In this example we show the different options of the ``Scheduler`` class.

"""
import numpy as np
import matplotlib.pylab as plt
import cylag

################################################################################
#
# Scheduler plot function
# -----------------------------------------------------------------------------
def plot_scheduler(ax, scheduler, text, yplot):

    simutime = cylag.SimuTime(
        print_progress=False,
        final_time=5.,
        time_step=5./35.)

    while not simutime.is_finished:
        print(simutime.time)
        ax.plot([simutime.time,], [yplot,], 'bx')
        if scheduler.now(simutime):
            print("now")
            ax.plot([simutime.time], [yplot], 'go')

        simutime.increment()

    ax.text(0., yplot+0.25, text)

################################################################################
#
# Scheduler
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(7, 6))

plot_scheduler(ax, cylag.schedules(time_delta=1.5), "Every 1.5 seconds", yplot=1)
plot_scheduler(ax, cylag.schedules(iteration_delta=5), "Every 5 time iterations", yplot=2)
plot_scheduler(ax, cylag.schedules(times=[0.5, 1., 2.5, 3.]), "At given times", yplot=3)
plot_scheduler(ax, cylag.schedules(iterations=[10, 12, 14, 16]), "At given time iterations", yplot=4)
plot_scheduler(ax, cylag.schedules(count=4), "4 equaly time-separated", yplot=5)
plot_scheduler(ax, cylag.schedules(always=True), "Always", yplot=6)

plt.grid()
plt.show()
