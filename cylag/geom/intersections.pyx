# cython: profile=False
#
# -------------------------------------------------------------------------------------------------
#  Project name: CyLag
#  Copyright (C) 2023 Fabien Souille
#
#  This program is free: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License published by the Free Software Foundation,
#  either version 3 of the license, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License for details.
#
#  You should have received a copy of the GNU General Public License
#  with this program. If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------------------------------
#
import numpy as np
from ..geom.utils cimport dot, dot2, sub, sub2, cross, normal, normal2
from ..geom.utils cimport orientation, on_segment, _orientation2d, _on_segment2d

# Symetric point comutation

cpdef double[:] compute_symetric_point2d(
        double[:] point0, double[:] point1,
        double[:] point, double damp):
    """
    Computes symetric of a point from a segment defined by (point0, point1)

    Parameters
    ----------
    point0 : array_like, shape (2,)
        First point of the segment.
    point1 : array_like, shape (2,)
        Second point of the segment.
    point : array_like, shape (2,)
        Point to compute the symetric.
    damp : float
        Damping factor for the symetric point computation.

    Returns
    -------
    psym : ndarray, shape (2,)
        Symetric point of the input point with respect to the segment.
    """
    cdef:
        double[:] psym = np.zeros( (2), dtype='d')
        double[:] u = np.zeros( (2), dtype='d')
        double[:] n = np.zeros( (2), dtype='d')
        double norm2, distance
        double coef = 1. - damp/2.

    normal2(point0, point1, n)
    norm2 = dot2(n, n)
    sub2(point, point0, u)
    distance = dot2(u, n)
    psym[0] = point[0] - 2.*((distance*coef)/norm2)*n[0]
    psym[1] = point[1] - 2.*((distance*coef)/norm2)*n[1]
    return psym

cpdef double[:] compute_symetric_point3d(
        double[:] point0, double[:] point1, double[:] point2,
        double[:] point, double damp):
    """
    Computes symetric psym of a point from a plane defined by (point0, point1, point2)

    Parameters
    ----------
    point0 : array_like, shape (3,)
        First point of the plane.
    point1 : array_like, shape (3,)
        Second point of the plane.
    point2 : array_like, shape (3,)
        Third point of the plane.
    point : array_like, shape (3,)
        Point to compute the symetric.
    damp : float
        Damping factor for the symetric point computation.

    Returns
    -------
    psym : ndarray, shape (3,)
        Symetric point of the input point with respect to the plane.
    """
    cdef:
        double[:] psym = np.zeros( (3), dtype='d')
        double[:] u = np.zeros( (3), dtype='d')
        double[:] n = np.zeros( (3), dtype='d')
        double norm2, distance
        double coef = 1. - damp/2.

    normal(point0, point1, point2, n)
    norm2 = dot(n, n)       
    sub(point, point0, u)
    distance = dot(u, n)
    psym[0] = point[0] - 2.*((distance*coef)/norm2)*n[0]
    psym[1] = point[1] - 2.*((distance*coef)/norm2)*n[1]
    psym[2] = point[2] - 2.*((distance*coef)/norm2)*n[2]
    return psym

# 2D intersection functions

cpdef int intersec2d_segment_segment(
        double[:] point0, double[:] point1, double[:] seg_point0, double[:] seg_point1):
    """
    Find if two segments intersects (point0, point1) and a ray (seg_point0, seg_point1)

    https://www.algotree.org/algorithms/computational_geometry/line_segment_intersection/

    Parameters
    ----------
    point0 : array_like, shape (2,)
        First point of the first segment.
    point1 : array_like, shape (2,)
        Second point of the first segment.
    seg_point0 : array_like, shape (2,)
        First point of the second segment.
    seg_point1 : array_like, shape (2,)
        Second point of the second segment.

    Returns
    -------
    int
        0 = disjoint (no intersection)
        1 = intersect in unique point
    """
    cdef:
        int or_1, or_2, or_3, or_4

    or_1 = orientation(point0, point1, seg_point0)
    or_2 = orientation(point0, point1, seg_point1)
    or_3 = orientation(seg_point0, seg_point1, point0)
    or_4 = orientation(seg_point0, seg_point1, point1)

    if (or_1 != or_2 and or_3 != or_4) :
        return 1
    if (or_1 == 0 and on_segment(point0, point1, seg_point0)):
        return 1
    if (or_2 == 0 and on_segment(point0, point1, seg_point1)):
        return 1
    if (or_3 == 0 and on_segment(seg_point0, seg_point1, point0)):
        return 1
    if (or_4 == 0 and on_segment(seg_point0, seg_point1, point1)):
        return 1
    return 0

cdef int c_intersec2d_segment_segment(
        double p0x, double p0y, double p1x, double p1y,
        double s0x, double s0y, double s1x, double s1y):
    """
    Fast scalar segment-segment intersection test (no memoryview / allocation overhead).

    Tests whether segment (p0, p1) intersects segment (s0, s1).

    Parameters
    ----------
    p0x, p0y : float
        Coordinates of the first point of the first segment.
    p1x, p1y : float
        Coordinates of the second point of the first segment.
    s0x, s0y : float
        Coordinates of the first point of the second segment.
    s1x, s1y : float
        Coordinates of the second point of the second segment.

    Returns
    -------
    int
        0 = disjoint (no intersection)
        1 = intersect in unique point
    """
    cdef int or_1, or_2, or_3, or_4

    or_1 = _orientation2d(p0x, p0y, p1x, p1y, s0x, s0y)
    or_2 = _orientation2d(p0x, p0y, p1x, p1y, s1x, s1y)
    or_3 = _orientation2d(s0x, s0y, s1x, s1y, p0x, p0y)
    or_4 = _orientation2d(s0x, s0y, s1x, s1y, p1x, p1y)

    if or_1 != or_2 and or_3 != or_4:
        return 1
    if or_1 == 0 and _on_segment2d(p0x, p0y, p1x, p1y, s0x, s0y):
        return 1
    if or_2 == 0 and _on_segment2d(p0x, p0y, p1x, p1y, s1x, s1y):
        return 1
    if or_3 == 0 and _on_segment2d(s0x, s0y, s1x, s1y, p0x, p0y):
        return 1
    if or_4 == 0 and _on_segment2d(s0x, s0y, s1x, s1y, p1x, p1y):
        return 1
    return 0

cpdef (double, double) intersec2d_get_intersection_point(\
        double x1, double y1,
        double x2, double y2,
        double x3, double y3,
        double x4, double y4):
    """ 
    Find intersection point of two segments 
    
    Parameters
    ----------
    x1, y1 : float
        Coordinates of the first point of the first segment.
    x2, y2 : float
        Coordinates of the second point of the first segment.
    x3, y3 : float
        Coordinates of the first point of the second segment.
    x4, y4 : float
        Coordinates of the second point of the second segment.
        
    Returns
    -------
    px, py : float
        Coordinates of the intersection point.
    """
    cdef: 
        double px, py
    px= ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)) 
    py= ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    return px, py

# 3D intersection functions

cpdef int MT_intersec3d_triangle_ray(
        double[:] point0, double[:] point1, double[:] point2,
        double[:] seg_point0, double[:] seg_point1, double[:] intersection_point):
    """
    Find intersection of 3D triangle (point0, point1, point2) and a ray (seg_point0, seg_point1)

    Implementation of Möller and Trumbore algorithm. 
    See article: "Fast, Minimum Storage Ray/Triangle Intersection".

    Parameters
    ----------
    point0 : array_like, shape (3,)
        First point of the triangle.
    point1 : array_like, shape (3,)
        Second point of the triangle.
    point2 : array_like, shape (3,)
        Third point of the triangle.
    seg_point0 : array_like, shape (3,)
        First point of the ray.
    seg_point1 : array_like, shape (3,)
        Second point of the ray.
    intersection_point : array_like, shape (3,)
        Intersection point of the triangle and the ray (if they intersect).

    Returns
    -------
    int
        0 = disjoint (no intersection)
        1 = intersect in unique point
    """
    cdef:
        double[:] edge1 = np.zeros( (3), dtype='d')
        double[:] edge2 = np.zeros( (3), dtype='d')
        double[:] direction = np.zeros( (3), dtype='d')
        double[:] pvec = np.zeros( (3), dtype='d')
        double[:] tvec = np.zeros( (3), dtype='d')
        double[:] qvec = np.zeros( (3), dtype='d')
        double u, v, det, inv_det, r
        double EPSILON = 1.e-16

    sub(point1, point0, edge1)
    sub(point2, point0, edge2)
    sub(seg_point1, seg_point0, direction)

    cross(direction, edge2, pvec)
    det = dot(edge1, pvec)

    if det > -EPSILON and det < EPSILON: return 0
    
    inv_det = 1.0/det
    
    sub(seg_point0, point0, tvec)
    u = dot(tvec, pvec)*inv_det
    if u < 0.0 or u > 1.0: return 0

    cross(tvec, edge1, qvec)

    v = dot(direction, qvec)*inv_det
    if v < 0.0 or u + v > 1.0: return 0

    r = dot(edge2, qvec)*inv_det
    intersection_point[0] = seg_point0[0] + r * direction[0]
    intersection_point[1] = seg_point0[1] + r * direction[1]
    intersection_point[2] = seg_point0[2] + r * direction[2]

    return 1

cpdef int intersec3d_triangle_segment(
        double[:] point0, double[:] point1, double[:] point2,
        double[:] seg_point0, double[:] seg_point1, double[:] intersection_point):
    """
    Find intersection of 3D triangle (point0, point1, point2) and a segment (seg_point0, seg_point1)

    Implementation of
    http://geomalgorithms.com/a06-_intersect-2.html
    http://geomalgorithms.com/a06-_intersect-2.html#intersect3D_RayTriangle%28%29

    Parameters
    ----------
    point0 : array_like, shape (3,)
        First point of the triangle.
    point1 : array_like, shape (3,)
        Second point of the triangle.
    point2 : array_like, shape (3,)
        Third point of the triangle.
    seg_point0 : array_like, shape (3,)
        First point of the segment.
    seg_point1 : array_like, shape (3,)
        Second point of the segment.
    intersection_point : array_like, shape (3,)
        Intersection point of the triangle and the segment (if they intersect).

    Returns
    -------
    int
       -1 = triangle is degenerate (no intersection)
        0 = disjoint (no intersection)
        1 = intersect in unique point
        2 = segment lies in triangle plane
    """
    cdef:
        double[:] u = np.zeros( (3), dtype='d')
        double[:] v = np.zeros( (3), dtype='d')
        double[:] n = np.zeros( (3), dtype='d')
        double[:] direction = np.zeros( (3), dtype='d')
        double[:] w0 = np.zeros( (3), dtype='d')
        double[:] w = np.zeros( (3), dtype='d')
        double r, a, b
        double  uu, uv, vv, wu, wv, D
        double s,t
        double EPSILON = 1.e-16
        bint Check_parallel = False

    sub(point1, point0, u)
    sub(point2, point0, v)
    cross(u, v, n)

    if n[0]==0.0 and n[1]==0.0 and n[2]==0.0: return -1

    sub(seg_point1, seg_point0, direction)
    sub(seg_point0, point0, w0)
    a = -dot(n, w0)
    b = dot(n, direction)

    if Check_parallel:
        if abs(b) < EPSILON: 
            if a == 0.0: #Segment lies in triangle plane.
                return 2
            else:        #Segment is disjoint from plane.
                return 0

    r = a / b
    if r < 0.0 or 1.0 < r: # No intersect.
        return 0

    intersection_point[0] = seg_point0[0] + r * direction[0]
    intersection_point[1] = seg_point0[1] + r * direction[1]
    intersection_point[2] = seg_point0[2] + r * direction[2]

    uu = dot(u, u)
    uv = dot(u, v)
    vv = dot(v, v)
    sub(intersection_point, point0, w)
    wu = dot(w, u)
    wv = dot(w, v)
    D = uv*uv - uu*vv

    s = (uv*wv - vv*wu) / D
    if s < 0.0 or s > 1.0: # intersection_point is outside tri
        return 0
    t = (uv*wu - uu*wv) / D
    if t < 0.0 or (s + t) > 1.0: # intersection_point is outside tri
        return 0

    return 1
