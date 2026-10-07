# -*- coding: utf-8 -*-
import unittest
import time
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

PRINTOUT = False

def plot_triangles_id(ax, triangulation):
    """ Plot triangulation.triangles id on mesh """
    for i in range(triangulation.triangles.shape[0]):
        v0, v1, v2 = triangulation.triangles[i]
        xc = np.mean([triangulation.x[v0], triangulation.x[v1], triangulation.x[v2]])
        yc = np.mean([triangulation.y[v0], triangulation.y[v1], triangulation.y[v2]])
        ax.text(xc, yc, "{}".format(i), fontsize=8, color='r')

def plot_cylag_triangle(ax, triangulation, indices, color='b', alpha=0.25):
    """ Plot cylag triangle """
    # convert cylag memoryviews into numpy arrays
    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    triangles = np.asarray(triangulation.triangles)
    for index in indices:
        polygon = Polygon([[0, 0], [0, 0]], facecolor=color, alpha=alpha)
        if index==-1:
            points = [0, 0, 0]
        else:
            points = triangles[index]
        xs = x[points]
        ys = y[points]
        polygon.set_xy(np.column_stack([xs, ys]))
        ax.add_patch(polygon)

class CheckXYLocalizer(unittest.TestCase):

    def test_xylocalizer_global(self):
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        # matplotlib triangulation and trifinder
        tri = Triangulation(x, y, triangles)
        trifinder = tri.get_trifinder()
        # define cylag triangulation with bnd from cli
        cylagtri = cylag.TriangularMesh(x, y, triangles)
        # number of tests
        ntest = 10
        for i in range(ntest):
            # point to locate
            point = [np.random.uniform(20., 60., 1)[0],\
                     np.random.uniform(30., 60., 1)[0]]
            # localize with matplotlib trifinder
            k_mtri = trifinder(point[0], point[1])
            # localize with cylag
            k_cylag1 = cylag.xy_localize_point(cylagtri, point[0], point[1])
            # check
            np.testing.assert_equal(k_mtri, k_cylag1)
            if PRINTOUT:
                print("==========================")
                print("trifinder      :", k_mtri)
                print("cylag method 1 :", k_cylag1)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot point to locate
            plt.plot(point[0], point[1], color='b', marker='o', markersize=6, lw=0, label="new pos")
            # plot triangle
            plot_cylag_triangle(ax, cylagtri, [k_mtri], color='b', alpha=0.5)
            plot_cylag_triangle(ax, cylagtri, [k_cylag1], color='r', alpha=0.15)
            # plot triangles labels
            plot_triangles_id(ax, tri)
            plt.legend()
            plt.show()
        # clean 
        del tri
        del cylagtri

    def test_xylocalizer_in_neighbors(self):
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        # matplotlib triangulation and trifinder
        tri = Triangulation(x, y, triangles)
        trifinder = tri.get_trifinder()
        # define cylag triangulation with bnd from cli
        cylagtri = cylag.TriangularMesh(x, y, triangles)
        # Build triangle_adjacency for neighbor search (not built by default)
        cylagtri._init_triangles_adjacency()
        # number of tests
        ntest = 10
        for i in range(ntest):
            # point to locate
            point = [np.random.uniform(20., 60., 1)[0],\
                     np.random.uniform(30., 60., 1)[0]]
            # localize with matplotlib trifinder
            k_mtri = trifinder(point[0], point[1])
            # random pick a neighbor
            k_neigh = -1
            while k_neigh ==-1:
                i_neigh = np.random.randint(1, cylagtri.triangle_adjacency.shape[1], 1)
                k_neigh = np.asarray(cylagtri.triangle_adjacency)[k_mtri, i_neigh][0]
            # localize with cylag
            k_cylag1 = cylag.xy_localize_point_in_neighbourhood(cylagtri, k_neigh, point[0], point[1])
            # check
            np.testing.assert_equal(k_mtri, k_cylag1)
            if PRINTOUT:
                print("==========================")
                print("neihbor        :", k_neigh)
                print("trifinder      :", k_mtri)
                print("cylag method 1 :", k_cylag1)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot point to locate
            plt.plot(point[0], point[1], color='b', marker='o', markersize=6, lw=0, label="new pos")
            # plot triangle
            plot_cylag_triangle(ax, cylagtri, [k_mtri], color='b', alpha=0.5)
            plot_cylag_triangle(ax, cylagtri, [k_cylag1], color='r', alpha=0.15)
            # plot triangles labels
            plot_triangles_id(ax, tri)
            plt.legend()
            plt.show()
        # clean 
        del tri
        del cylagtri
            
    def test_xylocalizer_from_path(self):
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        # matplotlib triangulation and trifinder
        tri = Triangulation(x, y, triangles)
        trifinder = tri.get_trifinder()
        # define cylag triangulation with bnd from cli
        cylagtri = cylag.TriangularMesh(x, y, triangles)
        # number of tests
        ntest = 10
        for i in range(ntest):
            # point to locate
            point = [np.random.uniform(20., 60., 1)[0],\
                     np.random.uniform(30., 60., 1)[0]]
            # localize with matplotlib trifinder
            k_mtri = trifinder(point[0], point[1])
            # random pick previous point
            point0 = [np.random.uniform(20., 60., 1)[0],\
                      np.random.uniform(30., 60., 1)[0]]
            k0 = trifinder(point0[0], point0[1])
            # localize with cylag
            k_cylag1, _ = cylag.xy_localize_point_from_path(cylagtri, k0, point0[0], point0[1], point[0], point[1])
            # check
            np.testing.assert_equal(k_mtri, k_cylag1)
            if PRINTOUT:
                print("==========================")
                print("neihbor        :", k0)
                print("trifinder      :", k_mtri)
                print("cylag method 1 :", k_cylag1)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot point to locate
            plt.plot(point0[0], point0[1], color='k', marker='o', markersize=6, lw=0, label="old pos")
            plt.plot(point[0], point[1], color='b', marker='o', markersize=6, lw=0, label="new pos")
            plt.plot([point0[0],point[0]], [point0[1],point[1]], ls="--", c='b')
            # plot triangle
            plot_cylag_triangle(ax, cylagtri, [k_mtri], color='b', alpha=0.5)
            plot_cylag_triangle(ax, cylagtri, [k_cylag1], color='r', alpha=0.15)
            # plot triangles labels
            plot_triangles_id(ax, tri)
            plt.legend()
            plt.show()
        # clean 
        del tri
        del cylagtri

    def test_xylocalizer_benchmark_close_point(self):
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        # matplotlib triangulation and trifinder
        tri = Triangulation(x, y, triangles)
        trifinder = tri.get_trifinder()
        # define cylag triangulation with bnd from cli
        cylagtri = cylag.TriangularMesh(x, y, triangles)
        # Build triangle_adjacency for neighbor search (not built by default)
        cylagtri._init_triangles_adjacency()
        # point to locate
        point = [np.random.uniform(20., 60., 1)[0],\
                 np.random.uniform(30., 60., 1)[0]]
        point = [31., 45.]
        k_neigh = 74 # previous tri (for xy_localize_point_in_neighbourhood)
        point0 = [40., 48.] # previous triangle (for xy_localize_point_from_path)
        # localize with matplotlib trifinder
        t0 = time.time()
        k_mtri = trifinder(point[0], point[1])
        t1 = time.time()
        cpu_time0 = t1 - t0
        # localize with cylag
        t0 = time.time()
        k_cylag1 = cylag.xy_localize_point(cylagtri, point[0], point[1])
        t1 = time.time()
        cpu_time1 = t1 - t0
        t0 = time.time()
        k_cylag2 = cylag.xy_localize_point_in_neighbourhood(cylagtri, k_neigh, point[0], point[1])
        t1 = time.time()
        cpu_time2 = t1 - t0
        t0 = time.time()
        k_cylag3, _ = cylag.xy_localize_point_from_path(cylagtri, k_neigh, point0[0], point0[1], point[0], point[1])
        t1 = time.time()
        cpu_time3 = t1 - t0
        if PRINTOUT:
            print("trifinder      :", k_mtri,   "/ CPU Time :", cpu_time0)
            print("cylag method 1 :", k_cylag1, "/ CPU Time :", cpu_time1)
            print("cylag method 2 :", k_cylag2, "/ CPU Time :", cpu_time2)
            print("cylag method 3 :", k_cylag3, "/ CPU Time :", cpu_time3)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot point to locate
            plt.plot(point0[0], point0[1], color='k', marker='o', markersize=6, lw=0, label="old pos")
            plt.plot(point[0], point[1], color='b', marker='o', markersize=6, lw=0, label="new pos")
            plt.plot([point0[0],point[0]], [point0[1],point[1]], ls="--", c='b')
            # plot triangle
            plot_cylag_triangle(ax, cylagtri, [k_mtri], color='b', alpha=0.5)
            plot_cylag_triangle(ax, cylagtri, [k_cylag1], color='r', alpha=0.15)
            plot_cylag_triangle(ax, cylagtri, [k_cylag2], color='r', alpha=0.15)
            plot_cylag_triangle(ax, cylagtri, [k_cylag3], color='r', alpha=0.15)
            # plot triangles labels
            plot_triangles_id(ax, tri)
            plt.legend()
            plt.show()
        # clean 
        del tri
        del cylagtri

    def test_xylocalizer_benchmark_any_point(self):
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "square_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "square_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "square_meshtri.txt"), dtype='int32')
        # matplotlib triangulation and trifinder
        tri = Triangulation(x, y, triangles)
        trifinder = tri.get_trifinder()
        # define cylag triangulation with bnd from cli
        cylagtri = cylag.TriangularMesh(x, y, triangles)
        # Build triangle_adjacency for neighbor search (not built by default)
        cylagtri._init_triangles_adjacency()
        # number of tests
        ntest = 100
        cpu_time0 = 0.
        cpu_time1 = 0.
        cpu_time2 = 0.
        cpu_time3 = 0.
        cpu_time4 = 0.
        for i in range(ntest):
            # point to locate
            point = [np.random.uniform(0., 5000., 1)[0],\
                     np.random.uniform(0., 5000., 1)[0]]
            # localize with matplotlib trifinder
            t0 = time.time()
            k_mtri = trifinder(point[0], point[1])
            t1 = time.time()
            cpu_time0 += t1 - t0
            # random pick previous point
            point0 = [np.random.uniform(0., 1000., 1)[0],\
                      np.random.uniform(0., 1000., 1)[0]]
            k0 = trifinder(point0[0], point0[1])
            # localize with cylag
            t0 = time.time()
            k_cylag1 = cylag.xy_localize_point(cylagtri, point[0], point[1])
            t1 = time.time()
            cpu_time1 += t1 - t0
            # localize with cylag - from neighbors
            t0 = time.time()
            k_cylag2 = cylag.xy_localize_point_in_neighbourhood(cylagtri, k0, point[0], point[1])
            if k_cylag2 == -1:
                k_cylag2 = cylag.xy_localize_point(cylagtri, point[0], point[1])
            t1 = time.time()
            cpu_time2 += t1 - t0
            np.testing.assert_equal(k_mtri, k_cylag2)
            # localize with cylag - from path            
            t0 = time.time()
            k_cylag3, _ = cylag.xy_localize_point_from_path(cylagtri, k0, point0[0], point0[1], point[0], point[1])
            t1 = time.time()
            cpu_time3 += t1 - t0
            np.testing.assert_equal(k_mtri, k_cylag3)
            # localize with cylag - using kdtree
            t0 = time.time()
            k_cylag4 = cylag.xy_localize_point_kdtree(cylagtri, point[0], point[1], 10)
            t1 = time.time()
            cpu_time4 += t1 - t0
            np.testing.assert_equal(k_mtri, k_cylag4)
            #if PRINTOUT:
            #    print("=====================================================")
            #    print("trifinder      :", k_mtri,   "/ CPU Time :", cpu_time0)
            #    print("cylag method 1 :", k_cylag1, "/ CPU Time :", cpu_time1)
            #    print("cylag method 2 :", k_cylag2, "/ CPU Time :", cpu_time2)
            #    print("cylag method 3 :", k_cylag3, "/ CPU Time :", cpu_time3)
            #    print("cylag method 4 :", k_cylag4, "/ CPU Time :", cpu_time4)
        # plot
        if PRINTOUT:
            print("CPU Time matplotlib trifinder        :", cpu_time0)
            print("CPU Time cylag method 1 (brut force) :", cpu_time1)
            print("CPU Time cylag method 2 (neighbors)  :", cpu_time2)
            print("CPU Time cylag method 3 (path)       :", cpu_time3)
            print("CPU Time cylag method 4 (KDTree)     :", cpu_time4)
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot point to locate
            plt.plot(point0[0], point0[1], color='k', marker='o', markersize=6, lw=0, label="old pos")
            plt.plot(point[0], point[1], color='b', marker='o', markersize=6, lw=0, label="new pos")
            plt.plot([point0[0],point[0]], [point0[1],point[1]], ls="--", c='b')
            # plot triangle
            plot_cylag_triangle(ax, cylagtri, [k_mtri], color='b', alpha=0.5)
            plot_cylag_triangle(ax, cylagtri, [k_cylag1], color='r', alpha=0.15)
            plot_cylag_triangle(ax, cylagtri, [k_cylag2], color='r', alpha=0.15)
            plot_cylag_triangle(ax, cylagtri, [k_cylag3], color='r', alpha=0.15)
            plot_cylag_triangle(ax, cylagtri, [k_cylag4], color='g', alpha=0.15)
            # plot triangles labels
            #plot_triangles_id(ax, tri)
            plt.legend()
            plt.show()
        # clean 
        del tri
        del cylagtri

    def test_xylocalizer_benchmark(self):
       # load test arrays
       root = path.dirname(__file__)
       x = np.loadtxt(path.join(root, "data", "square_meshx.txt"))
       y = np.loadtxt(path.join(root, "data", "square_meshy.txt"))
       triangles = np.loadtxt(path.join(root, "data", "square_meshtri.txt"), dtype='int32')
       # matplotlib triangulation and trifinder
       tri = Triangulation(x, y, triangles)
       trifinder = tri.get_trifinder()
       # define cylag triangulation with bnd from cli
       cylagtri = cylag.TriangularMesh(x, y, triangles)
       # Build triangle_adjacency for neighbor search (not built by default)
       cylagtri._init_triangles_adjacency()
       # number of tests
       ntest = 10
       cpu_time0 = 0.
       cpu_time1 = 0.
       cpu_time2 = 0.
       cpu_time3 = 0.
       for i in range(ntest):
           # point to locate
           point = [np.random.uniform(0., 2000., 1)[0],\
                    np.random.uniform(0., 2000., 1)[0]]
           # localize with matplotlib trifinder
           t0 = time.time()
           k_mtri = trifinder(point[0], point[1])
           t1 = time.time()
           cpu_time0 += t1 - t0
           # random pick previous point
           point0 = [np.random.uniform(0., 1000., 1)[0],\
                     np.random.uniform(0., 1000., 1)[0]]
           k0 = trifinder(point0[0], point0[1])
           # localize with cylag
           t0 = time.time()
           k_cylag1, _ = cylag.xy_localize(cylagtri, k0, point0[0], point0[1], point[0], point[1], 0, 1)
           t1 = time.time()
           cpu_time1 += t1 - t0
           # localize with cylag - from neighbors
           t0 = time.time()
           k_cylag2, _ = cylag.xy_localize(cylagtri, k0, point0[0], point0[1], point[0], point[1], 1, 1)
           if k_cylag2 == -1:
               k_cylag2, _ = cylag.xy_localize(cylagtri, k0, point0[0], point0[1], point[0], point[1], 0, 1)
           t1 = time.time()
           cpu_time2 += t1 - t0
           np.testing.assert_equal(k_mtri, k_cylag2)
           # localize with cylag - from path
           t0 = time.time()
           k_cylag3, _ = cylag.xy_localize(cylagtri, k0, point0[0], point0[1], point[0], point[1], 2, 1)
           t1 = time.time()
           cpu_time3 += t1 - t0
           np.testing.assert_equal(k_mtri, k_cylag3)
           #if PRINTOUT:
           #    print("=====================================================")
           #    print("trifinder      :", k_mtri,   "/ CPU Time :", cpu_time0)
           #    print("cylag method 1 :", k_cylag1, "/ CPU Time :", cpu_time1)
           #    print("cylag method 2 :", k_cylag2, "/ CPU Time :", cpu_time2)
           #    print("cylag method 3 :", k_cylag3, "/ CPU Time :", cpu_time3)
       # plot
       if PRINTOUT:
           print("CPU Time matplotlib trifinder        :", cpu_time0)
           print("CPU Time cylag method 1 (KDTree)     :", cpu_time1)
           print("CPU Time cylag method 2 (neighbors)  :", cpu_time2)
           print("CPU Time cylag method 3 (path)       :", cpu_time3)
           fig, ax = plt.subplots(figsize=(8, 8))
           plt.gca().set_aspect('equal')
           # plot mesh
           plt.triplot(tri, lw=0.5, color='0.5')
           # plot point to locate
           plt.plot(point0[0], point0[1], color='k', marker='o', markersize=6, lw=0, label="old pos")
           plt.plot(point[0], point[1], color='b', marker='o', markersize=6, lw=0, label="new pos")
           plt.plot([point0[0],point[0]], [point0[1],point[1]], ls="--", c='b')
           # plot triangle
           plot_cylag_triangle(ax, cylagtri, [k_mtri], color='b', alpha=0.5)
           plot_cylag_triangle(ax, cylagtri, [k_cylag1], color='r', alpha=0.15)
           plot_cylag_triangle(ax, cylagtri, [k_cylag2], color='r', alpha=0.15)
           plot_cylag_triangle(ax, cylagtri, [k_cylag3], color='r', alpha=0.15)
           # plot triangles labels
           #plot_triangles_id(ax, tri)
           plt.legend()
           plt.show()
       # clean 
       del tri
       del cylagtri

if __name__ == "__main__":
    unittest.main()
