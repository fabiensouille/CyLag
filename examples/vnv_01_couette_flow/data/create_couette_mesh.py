# -*- coding: utf-8 -*-
"""
Couette flow analytical advection validation
"""
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

def create_couette_triangulation(n_angles=60, n_radii=15, min_radius=0.25, max_radius=1.):
    # First create the x and y coordinates of the points.
    radii = np.linspace(min_radius, max_radius, n_radii)
    angles = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += math.pi / n_angles
    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()

    # Create the Triangulation; no triangles specified so Delaunay triangulation
    # created.
    triang = Triangulation(x, y)

    # Mask off unwanted triangles.
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid * xmid + ymid * ymid < min_radius * min_radius, 1, 0)
    triang.set_mask(mask)
    return x, y, triang

def write_couette_mesh(mesh_name, meshx, meshy, triang, min_radius=0.25):
    xmid = meshx[triang.triangles].mean(axis=1)
    ymid = meshy[triang.triangles].mean(axis=1)
    mask = np.where(xmid * xmid + ymid * ymid < min_radius * min_radius, 1, 0)
    triang.set_mask(mask)
    # save mesh x,y
    np.savetxt(mesh_name + "_meshx.txt", meshx, fmt='%.16f')
    np.savetxt(mesh_name + "_meshy.txt", meshy, fmt='%.16f')
    # save mesh triangles where not masked
    ntri = np.shape(mask)[0]
    f = open(mesh_name + "_meshtri.txt", "w")
    for i in range(ntri):
        if mask[i]==0:
            #print(triang.triangles[i])
            f.write('{} {} {} \n'.format(\
                triang.triangles[i][0],
                triang.triangles[i][1],
                triang.triangles[i][2]))
    f.close()

if __name__ == '__main__':

    # MESH 1:
    # ~~~~~~~
    # mesh name
    mesh_name = "couette"
    # Creating a Triangulation
    min_radius = 0.25
    meshx, meshy, triang = create_couette_triangulation(\
        n_angles=60, n_radii=15, min_radius=0.25, max_radius=1.)
    write_couette_mesh(\
        mesh_name, meshx, meshy, triang, min_radius=0.25)
    # read mesh
    x = np.loadtxt(mesh_name + "_meshx.txt")
    y = np.loadtxt(mesh_name + "_meshy.txt")
    triangles = np.loadtxt(mesh_name + "_meshtri.txt", dtype='int32')
    # Plot 
    fig, ax = plt.subplots(1, 1, figsize=(6., 5))
    plt.gca().set_aspect('equal')
    tri = Triangulation(x, y, triangles)
    ax.triplot(tri, lw=0.5, color='0.75')
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.grid()
    plt.show()
    
    # MESH 2: refined (for convergence)
    # ~~~~~~~
    # mesh name
    mesh_name = "couette_refined"
    # Creating a Triangulation
    min_radius = 0.25
    meshx, meshy, triang = create_couette_triangulation(\
        n_angles=360, n_radii=100, min_radius=0.25, max_radius=1.)
    write_couette_mesh(\
        mesh_name, meshx, meshy, triang, min_radius=0.25)
    # read mesh
    x = np.loadtxt(mesh_name + "_meshx.txt")
    y = np.loadtxt(mesh_name + "_meshy.txt")
    triangles = np.loadtxt(mesh_name + "_meshtri.txt", dtype='int32')
    # Plot 
    fig, ax = plt.subplots(1, 1, figsize=(6., 5))
    plt.gca().set_aspect('equal')
    tri = Triangulation(x, y, triangles)
    ax.triplot(tri, lw=0.5, color='0.75')
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.grid()
    plt.show()
