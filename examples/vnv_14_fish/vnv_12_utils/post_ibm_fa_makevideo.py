# -*- coding: utf-8 -*-
"""
Test ParticleTracking algo - t2d_tide case

"""
import os
import time
import numpy as np
from os import path, environ
import matplotlib.pylab as plt
import matplotlib.tri as mtri
import cylag
from data_manip.extraction.telemac_file import TelemacFile
from plot_params import *

def plot_particles(tri, fset, part, rec=0, fig_name='test', **kwargs):
    """ Plot particles """
    time = part.times[rec]
    xx = part.xp[rec][:]
    yy = part.yp[rec][:]
    tags = part.tags[rec]
    npart = len(tags)

    # get velocity at final time
    nl = fset.nlayers
    ux = fset.get_field(1, time, layer=4)
    uy = fset.get_field(2, time, layer=4)
    velocity = np.sqrt(ux**2 + uy**2)

    # plot
    set_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
    ax.set_aspect('equal')
    xlim = [np.min(tri.x), np.max(tri.x)]
    ylim = [np.min(tri.y), np.max(tri.y)]

    # plot mesh and velocity
    levels = np.linspace(0., 0.75, 16)
    cs = ax.tricontourf(tri, velocity, levels=levels, cmap='RdBu_r', alpha=0.80)
    fig.colorbar(cs, ax=ax, label="$\\textbf{U}_f$ (m/s)")
    ax.triplot(tri, lw=0.12, c='grey')

    # plot fish particules
    for i, tag in enumerate(part.tags[rec]):
        # compute orientation
        dx = part.fax[rec][i]
        dy = part.fay[rec][i]
        orientation = np.arctan2(dy, dx)
        # define custom marker
        marker = cylag.define_custom_marker(orientation)
        # plot
        ax.scatter(part.xp[rec][i], part.yp[rec][i], s=100, marker=marker, lw=0., **kwargs)

    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")

    plt.savefig("{}{:05d}.png".format(fig_name, rec), dpi=300, format="png")
    plt.close(fig)

if __name__ == "__main__":

    # define eulerian space from telemac3d results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    t2d_file = path.join("data", "r2d_particles_fin.slf")
    bnd_file = path.join("data", "geo_particles.cli")
    fset = cylag.EulerianFieldSet.from_telemac2d(t2d_file, bnd_file=bnd_file)
    tri = mtri.Triangulation(fset.triangular_mesh.x,\
                             fset.triangular_mesh.y,\
                             fset.triangular_mesh.triangles)

    # particles_io class for post-processing
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    part = cylag.ParticlesIO.from_cylag_txt('particles/particles_2d_ibm_fa.txt')

    # Post processing
    # ~~~~~~~~~~~~~~~

    # Plot particles at a given rec
    plot_particles(tri, fset, part, rec=40, fig_name='figs/video/rec', c='k')

    # Make video from particles plots:
    # generate .png files
    plot_every = 1
    for rec, time in enumerate(part.times):
        if rec%plot_every == 0:
            print("~~> plotting rec={}, time={}".format(rec, time))
            plot_particles(tri, fset, part, rec, fig_name='figs/video/rec', c='k')

    # make mp4 from .png files
    cylag.ffmpeg_img_to_video(
        'figs/video/', 'rec', 'video', 60)

    # remove png files
    os.system("rm figs/video/*.png")
