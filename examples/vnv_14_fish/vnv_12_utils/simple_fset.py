# -*- coding: utf-8 -*-
from os import path
import cylag
import numpy as np
from matplotlib.tri import Triangulation

def fset_2d():
    meshx = np.loadtxt(path.join("data", "square_meshx.txt"))
    meshy = np.loadtxt(path.join("data", "square_meshy.txt"))
    triangles = np.loadtxt(path.join("data", "square_meshtri.txt"), dtype='int32')
    bnd_file = path.join("data", "square_geo.cli")

    meshx = meshx/100.
    meshy = meshy/100.
    tri = Triangulation(meshx, meshy, triangles)
    trifinder = tri.get_trifinder()
    meshsize = len(tri.x)
    
    bnd_points, _ = cylag.read_cli(bnd_file)
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
