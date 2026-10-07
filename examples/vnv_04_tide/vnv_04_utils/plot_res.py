# -*- coding: utf-8 -*-
"""
Plot results

"""
import cylag
import os
import time
import numpy as np
from os import path, environ
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from postel.plot2d import *

def plot_particles(tri, fset, part, rec=0,
        plot_scatter=True,
        plot_mesh=True,
        plot_velocity_field=True,
        plot_velocity_vectors=False,
        plot_traj=False,
        traj_length=None,
        plot_background_map=False,
        background_map_file='',
        save_fig=False,
        fig_name='',
        show_fig=True,
        color='k',
        scatter_cmap='copper',
        **kwargs):

    """ Plot particles """
    time = part.times[rec]
    xx = part.xp[rec][:]
    yy = part.yp[rec][:]
    tags = part.tags[rec]

    # plot
    cylag.set_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_aspect('equal')

    xlim = [np.min(tri.x), np.max(tri.x)]
    ylim = [np.min(tri.y), np.max(tri.y)]

    # plot background map
    if plot_background_map:
        plot2d_image(
            ax, image_file=background_map_file, 
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            aspect=1., alpha=1.)

    # velocity
    ux = fset.get_field(1, time)
    uy = fset.get_field(2, time)
    velocity = np.sqrt(ux**2 + uy**2)
    if plot_velocity_field:
        # tripcolor
        img = ax.tripcolor(tri, velocity, shading='gouraud', cmap='RdBu_r', vmin=0., vmax=2.)
        fig.colorbar(img, ax=ax, label="$U_f$ (m/s)")

    # mesh
    if plot_mesh:
        ax.triplot(tri, lw=0.2, c='k')
                    
    # Plotting vectors
    if plot_velocity_vectors:
        norm = mcolors.Normalize(vmin=0.0, vmax=1.6)
        img = ax.quiver(tri.x, tri.y, ux, uy, velocity, norm=norm, cmap='RdBu_r')

    # trajectories
    if plot_traj:
        traj = part.get_trajectories(tags)
        if traj_length is not None:
            rec0 = max(0, int(rec-traj_length))
        else:
            rec0 = 0
        for tr in traj:
            ax.plot(tr[rec0:rec+1, 0], tr[rec0:rec+1, 1], lw=0.2, marker='o', markersize=0, c="w")

    # positions
    if plot_scatter:
        ax.scatter(xx, yy, s=1., marker='o', c=tags, cmap=scatter_cmap, alpha=1.,
             linewidths=0, **kwargs)
    else:
        ax.plot(xx, yy, c=color, linewidth=0., marker='o', markersize=0.5, **kwargs)

    # Compute and plot date
    sec = time-part.times[0]
    d = sec // 86400
    ax.text(0.08, 0.08,
      f"J{int(d):02} {int((sec//3600)%24):02}:{int((sec//60)%60):02}",
      transform=ax.transAxes, va="top", fontsize=14)

    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    #ax.grid()

    if save_fig:
        plt.savefig("{}{:05d}.png".format(fig_name, rec), dpi=300, format="png")
    if show_fig:
        plt.show()

    plt.close(fig)

if __name__ == "__main__":

    # define eulerian space from telemac3d results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    file_name = path.join('..', 'data', 'r2d_tide-jmj_type.slf')
    bnd_file = path.join('..', 'data', 'geo_tide.cli')
    fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)
    tri = fset.get_mtri_triangulation()

    # particles_io class for post-processing
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = 'lsm1'
    part = cylag.ParticlesIO.from_cylag_txt('../resu_03/particles_2d.txt')

    # Post processing
    # ~~~~~~~~~~~~~~~
    recs = [1, 20, len(part.times)-1]

    # 'RdGy', 'PRGn', 'YlGn'
    for rec in recs:
        plot_particles(
            tri,
            fset,
            part,
            rec=rec,
            plot_scatter=False,
            plot_mesh=False,
            plot_velocity_field=True,
            plot_velocity_vectors=True,
            plot_traj=False,
            traj_length=10,
            save_fig=True,
            fig_name='../figs/t2d_tide_{}_'.format(model),
            plot_background_map=True,
            background_map_file='../data/background_map_b.png',
            show_fig=True,
            scatter_cmap='magma',
            color='k')
