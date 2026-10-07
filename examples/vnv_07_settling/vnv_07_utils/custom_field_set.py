# -*- coding: utf-8 -*-
"""
Custom field set
"""
from os import path
import math
import numpy as np
from matplotlib.tri import Triangulation
import cylag

def fset_3d(zb=0., zs=20., nlayers=10):

    # load mesh
    meshx = np.loadtxt(path.join("data", "square_meshx.txt"))
    meshy = np.loadtxt(path.join("data", "square_meshy.txt"))
    meshx /= 1000.
    meshy /= 1000.
    triangles = np.loadtxt(path.join("data", "square_meshtri.txt"), dtype='int32')
    bnd_file = path.join("data", "square_geo.cli")
    
    # read boundaries
    bnd_points, _ = cylag.read_cli(bnd_file)
    cylagtri = cylag.TriangularMesh(meshx, meshy, triangles, boundary_points=bnd_points)
    
    # field_set
    meshsize = len(cylagtri.x)
    elevation = np.zeros((1, meshsize*nlayers), dtype='d')
    velocity_x = np.zeros((1, meshsize*nlayers), dtype='d')
    velocity_y = np.zeros((1, meshsize*nlayers), dtype='d')
    velocity_z = np.zeros((1, meshsize*nlayers), dtype='d')
    dz = (zs - zb)/max((nlayers-1), 1)
    zlayers = np.array([dz*i for i in range(nlayers)])
    for i in range(nlayers):
        elevation[0, i*meshsize:(i+1)*meshsize] = zlayers[i]
        
    return cylag.EulerianFieldSet(
        triangular_mesh=cylagtri,
        dim=3, nlayers=nlayers,
        times=np.zeros(1),
        elevation=elevation,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z)
        
def plot2d_mesh():
    fset = fset_3d()
    tri = Triangulation(\
        fset.triangular_mesh.x,\
        fset.triangular_mesh.y,\
        fset.triangular_mesh.triangles)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_aspect('equal')
    ax.triplot(tri, lw=0.25, color='0.75')
    plt.show()
