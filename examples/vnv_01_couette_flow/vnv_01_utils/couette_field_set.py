# -*- coding: utf-8 -*-
"""
Couette field set
"""
from os import path
import math
import numpy as np
from matplotlib.tri import Triangulation
import cylag

def create_couette_triangulation(n_angles=60, n_radii=15, min_radius=0.25, max_radius=1.):
    """ 
    Create Couette mesh
    """
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
    """ 
    Write Couette mesh
    """
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


class radial_velocity_field_2d():
    """
    Couette flow velocity
    ~~~~~~~~~~~~~~~~~~~~~
    U(r) = U_m * ((r-R_{0})/(R_{1}-R_{0}))
    \vec{u}(x,y) = (ux(x,y), uy(x,y)) = (ux(r,\theta), uy(r,\theta)) where:
    ux(r, \theta) = -U(r)\sin{\theta)}
    uy(r, \theta) =  U(r)\cos{\theta)}
    """
    def __init__(self, umax=1., min_radius=0.25, max_radius=1.):
        self.umax = umax
        self.min_radius = min_radius
        self.max_radius = max_radius

    def get(self, x, y):
        radius = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        vel = self.umax*(radius-self.min_radius)/(self.max_radius-self.min_radius)
        u = -vel*np.sin(theta)
        v =  vel*np.cos(theta)
        return u, v

    def get_field_on_2dmesh(self, tri):
        meshx = tri.x
        meshy = tri.y
        size = len(meshx)
        velx = np.empty(size)
        vely = np.empty(size)
        velx[:], vely[:] = self.get(meshx[:], meshy[:])
        return velx, vely

def couette_flow_period(radius, umax=1., min_radius=0.25, max_radius=1.):
    """
    Couette flow period of rotation for a given radius
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    T(r) = (2.\pi r)/U(r)
    """
    p = 2.*np.pi*radius
    u = umax*(radius-min_radius)/(max_radius-min_radius)
    return p/u
    
def couette_flow_analytical_solution(time, initial_position):
    """ 
    Couette flow analytical solution 
    """
    x0 = initial_position[0]
    y0 = initial_position[1]
    radius = np.sqrt(x0**2 + y0**2)
    period = couette_flow_period(radius, umax=1., min_radius=0.25, max_radius=1.)
    theta0 = np.arctan2(y0, x0)
    theta =  time*2.*np.pi/period
    theta += theta0
    return radius, theta

def polar_coordinates(x, y):
    """ 
    Get polar coordinates 
    """
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return radius, theta
  
def couette_field_set(mesh="couette"):
    """ 
    Initialize couette flow field set 
    """
    # load mesh
    meshx = np.loadtxt(path.join("data", mesh+"_meshx.txt"))
    meshy = np.loadtxt(path.join("data", mesh+"_meshy.txt"))
    triangles = np.loadtxt(path.join("data", mesh+"_meshtri.txt"), dtype='int32')
    
    # matplotlib triangulation
    tri = Triangulation(meshx, meshy, triangles)
    trifinder = tri.get_trifinder()
    
    # create cylag triangular mesh
    cylagtri = cylag.TriangularMesh(meshx, meshy, triangles)
    
    # set analytical velocity field:
    field = radial_velocity_field_2d()
    velx, vely = field.get_field_on_2dmesh(tri)
    velocity = np.sqrt(velx**2 + vely**2)
    
    # field_set
    meshsize = len(tri.x)
    elevation = np.ones((1, meshsize), dtype='d')
    velocity_x = np.zeros((1, meshsize), dtype='d')
    velocity_y = np.zeros((1, meshsize), dtype='d')
    velocity_x[0, :] = velx
    velocity_y[0, :] = vely

    fset = cylag.EulerianFieldSet(
        triangular_mesh=cylagtri,
        dim=2, nlayers=1,
        times=np.zeros(1),
        elevation=elevation,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        trifinder=trifinder)

    return fset, tri

