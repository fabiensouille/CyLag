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

def fset_2d():
    root = path.dirname(__file__)
    meshx = np.loadtxt(path.join(root, "data", "square_meshx.txt"))
    meshy = np.loadtxt(path.join(root, "data", "square_meshy.txt"))
    triangles = np.loadtxt(path.join(root, "data", "square_meshtri.txt"), dtype='int32')
    bnd_file = path.join(root, "data", "square_geo.cli")

    meshx = meshx/100.
    meshy = meshy/100.
    tri = mtri.Triangulation(meshx, meshy, triangles)
    trifinder = tri.get_trifinder()
    meshsize = len(tri.x)
    
    bnd_points = cylag.read_cli(bnd_file)
    cylagtri = cylag.TriangularMesh(meshx, meshy, triangles, boundary_points=bnd_points)
    elevation = np.ones((1, meshsize), dtype='d')
    velocity_x = np.zeros((1, meshsize), dtype='d')
    velocity_y = np.zeros((1, meshsize), dtype='d')

    fset = cylag.EulerianFieldSet(
        triangular_mesh=cylagtri,
        dim=2, nlayers=1,
        times=np.zeros(1),
        elevation=elevation,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        trifinder=trifinder)
    return tri, fset

def plot_particles(tri, fset, part, rec=0, fig_name='test', **kwargs):
    """ Plot particles """
    time = part.times[rec]
    xx = part.xp[rec][:]
    yy = part.yp[rec][:]
    tags = part.tags[rec]
    npart = len(tags)

    # plot
    set_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=(6., 6.))
    ax.set_aspect('equal')

    # plot mesh and velocity
    #ax.triplot(tri, lw=0.12, c='grey')

    # plot fish particules
    for i, tag in enumerate(part.tags[rec]):
        # compute orientation
        dx = part.ua[rec][i]
        dy = part.va[rec][i]
        orientation = np.arctan2(dy, dx)
        # define custom marker
        marker = cylag.define_custom_marker(orientation)
        # plot
        ax.scatter(part.xp[rec][i], part.yp[rec][i], s=100, marker=marker, lw=0., **kwargs)

    ax.set_xlim([-100., 100.])
    ax.set_ylim([-100., 100.])
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")

    plt.savefig("{}{:05d}.png".format(fig_name, rec), dpi=300, format="png")
    plt.close(fig)

if __name__ == "__main__":

    # define eulerian space from telemac3d results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tri, fset = fset_2d()

    # particles_io class for post-processing
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    part = cylag.ParticlesIO.from_cylag_txt('particles/test_cbm_ua.txt')

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
        'figs/video/', 'rec', 'video', 30)

    # remove png files
    os.system("rm figs/video/*.png")
