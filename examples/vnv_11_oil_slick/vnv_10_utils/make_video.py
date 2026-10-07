# -*- coding: utf-8 -*-
"""
Make video from resu

"""
import cylag
import os
import time
import numpy as np
from os import path, environ
import matplotlib.pylab as plt
import matplotlib.tri as mtri
from postel.plot2d import *
from plot_res import plot_particles

def ffmpeg_img_to_video(img_folder, img_name, videoname, framerate):
    """
    Use ffmpeg to convert a list of png into a mp4 video
    """
    os.system("ffmpeg -r {} -i {}%05d.png -vcodec libx264 -crf 25 {}.mp4"
              .format(framerate, img_folder+img_name, videoname))

if __name__ == "__main__":

    # define eulerian space from telemac3d results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    file_name = path.join('..', 'data', 'r2d_tide-jmj_type.slf')
    bnd_file = path.join('..', 'data', 'geo_tide.cli')
    fset = cylag.EulerianFieldSet.from_telemac2d(file_name, bnd_file=bnd_file)
    tri = fset.get_mtri_triangulation()

    # particles_io class for post-processing
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    part = cylag.ParticlesIO.from_cylag_txt('../resu_for_video/particles_2d.txt')

    # Make video from particles plots:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # generate .png files
    plot_every = 1
    for rec, time in enumerate(part.times):
        if rec%plot_every == 0:
            print("~~> plotting rec={}, time={}".format(rec, time))
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
                fig_name='../figs/video/rec',
                plot_background_map=True,
                background_map_file='../data/background_map_b.png',
                show_fig=False,
                scatter_cmap='magma',
                color='k')
                
    # make mp4 from .png files
    ffmpeg_img_to_video(
        '../figs/video/', 'rec', 'video', 40)

    # remove png files
    os.system("rm ../figs/video/*.png")
