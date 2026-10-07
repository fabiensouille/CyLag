# -*- coding: utf-8 -*-
import unittest
import time
from os import path
import numpy as np
import cylag
import matplotlib.tri as mtri
import matplotlib.pylab as plt

PRINTOUT = False

class CheckIntersections(unittest.TestCase):

    def test_symmetric_2d_0(self):
        # Segment 
        A = np.array([1., 2.])
        B = np.array([3., 4.])
        # Point to compute sym
        D = np.array([3., 1.])
        # Compute symmetric point
        damp0 = 0.
        damp1 = 1.
        P = cylag.compute_symetric_point2d(A, B, D, damp0)
        Q = cylag.compute_symetric_point2d(A, B, D, damp1)
        # plot 
        np.testing.assert_equal(np.asarray(P), np.array([0., 4.]))
        np.testing.assert_equal(np.asarray(Q), np.array([1.5, 2.5]))
        # check
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            ax.plot([A[0], B[0]], [A[1], B[1]], marker='o', c="b")
            ax.text(A[0], A[1], "A", c="b", fontsize=20)
            ax.text(B[0], B[1], "B", c="b", fontsize=20)
            ax.plot([D[0], P[0]], [D[1], P[1]], marker='o', c="k")
            ax.text(D[0], D[1], "D", c="k", fontsize=20)
            ax.plot([P[0], Q[0]], [P[1], Q[1]], marker='o', c="r", lw=0)
            ax.text(P[0], P[1], "Sym, damp={}".format(damp0), c="r", fontsize=20)
            ax.text(Q[0], Q[1], "Sym, damp={}".format(damp1), c="r", fontsize=20)
            plt.show()
            plt.close(fig)
            
    def test_symmetric_2d_1(self):
        # Segment 
        A = np.array([2.5, 3.5])
        B = np.array([3., 4.])
        # Point to compute sym
        D = np.array([3., 1.])
        # Compute symmetric point
        damp0 = 0.
        damp1 = 1.
        P = cylag.compute_symetric_point2d(A, B, D, damp0)
        Q = cylag.compute_symetric_point2d(A, B, D, damp1)
        # plot 
        np.testing.assert_equal(np.asarray(P), np.array([0., 4.]))
        np.testing.assert_equal(np.asarray(Q), np.array([1.5, 2.5]))
        # check
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            ax.plot([A[0], B[0]], [A[1], B[1]], marker='o', c="b")
            ax.text(A[0], A[1], "A", c="b", fontsize=20)
            ax.text(B[0], B[1], "B", c="b", fontsize=20)
            ax.plot([D[0], P[0]], [D[1], P[1]], marker='o', c="k")
            ax.text(D[0], D[1], "D", c="k", fontsize=20)
            ax.plot([P[0], Q[0]], [P[1], Q[1]], marker='o', c="r", lw=0)
            ax.text(P[0], P[1], "Sym, damp={}".format(damp0), c="r", fontsize=20)
            ax.text(Q[0], Q[1], "Sym, damp={}".format(damp1), c="r", fontsize=20)
            plt.show()
            plt.close(fig)

    def test_symmetric_3d_0(self):
        # Face
        A = np.array([1., 2., 0.])
        B = np.array([3., 4., 0.])
        C = np.array([1., 2., 1.])
        # Point to compute sym
        D = np.array([3., 1., 1.])
        # Compute symmetric point
        damp0 = 0.
        damp1 = 1.
        P = cylag.compute_symetric_point3d(A, B, C, D, damp0)
        Q = cylag.compute_symetric_point3d(A, B, C, D, damp1)
        # plot 
        np.testing.assert_equal(np.asarray(P), np.array([0., 4., 1.]))
        np.testing.assert_equal(np.asarray(Q), np.array([1.5, 2.5, 1.]))
        # check
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_aspect('equal')
            ax.plot([A[0], B[0]], [A[1], B[1]], marker='o', c="b")
            ax.text(A[0], A[1], "A", c="b", fontsize=20)
            ax.text(B[0], B[1], "B", c="b", fontsize=20)
            ax.plot([D[0], P[0]], [D[1], P[1]], marker='o', c="k")
            ax.text(D[0], D[1], "D", c="k", fontsize=20)
            ax.plot([P[0], Q[0]], [P[1], Q[1]], marker='o', c="r", lw=0)
            ax.text(P[0], P[1], "Sym, damp={}".format(damp0), c="r", fontsize=20)
            ax.text(Q[0], Q[1], "Sym, damp={}".format(damp1), c="r", fontsize=20)
            plt.show()
            plt.close(fig)

    def test_intersect_2d(self):
        # Segment 1
        list_A = [np.array([1., 1.]), np.array([1., 1.]), np.array([1., 2.]), np.array([5., 2.]), np.array([4., 1.])]
        list_B = [np.array([4., 4.]), np.array([4., 4.]), np.array([3., 4.]), np.array([7., 4.]), np.array([6., 3.])]
        # Segment 2
        list_C = [np.array([2., 2.]), np.array([3., 1.]), np.array([1., 4.]), np.array([6., 5.]), np.array([6., 5.])]
        list_D = [np.array([5., 5.]), np.array([6., 4.]), np.array([3., 1.]), np.array([8., 3.]), np.array([8., 3.])]
        stat = []
        for i in range(len(list_A)):
            stat.append(cylag.intersec2d_segment_segment(list_A[i], list_B[i], list_C[i], list_D[i]))
        # check res:
        ref_stat = [1, 0, 1, 1, 0]
        np.testing.assert_equal(ref_stat, stat)
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            for i in range(len(list_A)):
                ax.plot([list_A[i][0], list_B[i][0]], [list_A[i][1], list_B[i][1]], marker='o', label="{}".format(i))
                ax.plot([list_C[i][0], list_D[i][0]], [list_C[i][1], list_D[i][1]], marker='o', label="{}".format(i))
            plt.legend()
            plt.show()
            plt.close(fig)
#            
    def test_intersect_2d_point(self):
        # Segment 1
        A = np.array([0., 0.])
        B = np.array([0., 4.])
        # Segment 2
        C = np.array([-1., 1.])
        D = np.array([4., 1.])
        # intersection point
        intx, inty = cylag.intersec2d_get_intersection_point(A[0], A[1], B[0], B[1], C[0], C[1], D[0], D[1])
        np.testing.assert_equal(abs(intx), 0.)
        np.testing.assert_equal(inty, 1.)
        if PRINTOUT:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.plot([A[0], B[0]], [A[1], B[1]], marker='o', label="")
            ax.plot([C[0], D[0]], [C[1], D[1]], marker='o', label="")
            plt.plot([intx], [inty], marker='o', c='k')
            plt.legend()
            plt.show()
            plt.close(fig)

    def test_intersect_3d_triangle_ray(self):
        # Trangles
        A = np.array([0., 0., 0.])
        B = np.array([0., 1., 0.])
        C = np.array([0., 0., 1.])
        # intersection
        I1 = np.array([0., 0., 0.])
        I2 = np.array([0., 0., 0.])
        ind1 = cylag.MT_intersec3d_triangle_ray(A, B, C,\
            np.array([-1., 0.5, 0.25]), np.array([1., 0.5, 0.25]), I1)
        ind2 = cylag.MT_intersec3d_triangle_ray(A, B, C,\
            np.array([-1., 10., 0.25]), np.array([1., 10., 0.25]), I2)
        # check res:
        np.testing.assert_equal(ind1, 1)
        np.testing.assert_equal(I1, np.array([0., 0.5, 0.25]))
        np.testing.assert_equal(ind2, 0)
        np.testing.assert_equal(I2, np.array([0., 0., 0.]))
        if PRINTOUT:
            print(I1, ind1)
            print(I2, ind2)
            
    def test_intersect_3d_triangle_segment(self):
        # Trangles
        A = np.array([0., 0., 0.])
        B = np.array([0., 1., 0.])
        C = np.array([0., 0., 1.])
        # intersection
        I1 = np.array([0., 0., 0.])
        I2 = np.array([0., 0., 0.])
        ind1 = cylag.intersec3d_triangle_segment(A, B, C,\
            np.array([-1., 0.5, 0.25]), np.array([1., 0.5, 0.25]), I1)
        ind2 = cylag.intersec3d_triangle_segment(A, B, C,\
            np.array([0.1, 0.5, 0.25]), np.array([1., 0.5, 0.25]), I2)
        # check res:
        np.testing.assert_equal(ind1, 1)
        np.testing.assert_equal(I1, np.array([0., 0.5, 0.25]))
        np.testing.assert_equal(ind2, 0)
        np.testing.assert_equal(I2, np.array([0., 0., 0.]))
        if PRINTOUT:
            print(I1, ind1)
            print(I2, ind2)

if __name__ == "__main__":
    unittest.main()
