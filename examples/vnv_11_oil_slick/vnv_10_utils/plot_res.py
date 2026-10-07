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
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
        wind_rose=True,
        wind_x=0.2, 
        wind_y=-0.15,
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
    fig, ax = plt.subplots(1, 1, figsize=(10., 7.))
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
        bounds = [0., 0.5, 1., 1.5, 2.]
        cmap_disc = cmap_disc = ListedColormap(
            plt.cm.bone(np.linspace(0., 0.5, len(bounds)-1)))
        norm_disc = BoundaryNorm(bounds, cmap_disc.N)
        sc = ax.scatter(part.xp[rec][:], part.yp[rec][:], c=part.zp[rec][:], 
            lw=0., s=1.75, cmap=cmap_disc, norm=norm_disc, label="$x_p(t_f)$")
        fig.colorbar(sc, ax=ax, label="Particle depth (m)", ticks=bounds, extend="max")
    else:
        ax.plot(xx, yy, c=color, linewidth=0., marker='o', markersize=0.5, **kwargs)

    if wind_rose:
        # wind components
        wind_speed = np.hypot(wind_x, wind_y)
        wind_dir = (np.degrees(np.arctan2(wind_x, wind_y)) + 360) % 360
        # create inset
        axins = inset_axes(ax, width="12%", height="12%", loc="upper right")
        axins.set_aspect("equal")
        axins.add_patch(plt.Circle((0, 0), 1, fill=False, lw=1))
        # compass direction -> cartesian angle
        theta = np.deg2rad(90 - wind_dir)
        # arrow length scaled by speed
        max_speed = 20.0  # adapt if needed

        u = wind_x / wind_speed
        v = wind_y / wind_speed
        axins.annotate(
            "",
            xy=(0.8*u, 0.8*v),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1,
                color="r",
                mutation_scale=20,
            ),
        )

        axins.text(0, 1.15, "N", ha="center", va="center", fontsize=7)
        axins.text(1.15, 0, "E", ha="center", va="center", fontsize=7)
        axins.text(0, -1.15, "S", ha="center", va="center", fontsize=7)
        axins.text(-1.15, 0, "W", ha="center", va="center", fontsize=7)
        axins.text(0, -1.5, f"Wind", ha="center", fontsize=7)
        axins.set_xlim(-1.3, 1.3)
        axins.set_ylim(-1.6, 1.3)
        axins.axis("off")

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
    part = cylag.ParticlesIO.from_cylag_txt('../resu_for_video/particles_2d.txt')

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
            plot_scatter=True,
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
