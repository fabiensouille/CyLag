# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
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
from libc.math cimport fmin, fmax, sqrt, fabs

# Basic 2d vectorial functions
cdef double det22(double[:] v1, double[:] v2) noexcept nogil:
    """ 
    Determinant 2x2 
    
    Parameters
    ----------
    v1 : array_like, shape (2,)
        First vector.
    v2 : array_like, shape (2,)
        Second vector.

    Returns
    -------
    double
        Result of the determinant v1 x v2.
    """
    return v1[0]*v2[1] - v1[1]*v2[0]

cdef double dot2(double[:] v1, double[:] v2) noexcept nogil:
    """ 
    Dot product of vectors v1 and v2 
    
    Parameters
    ----------
    v1 : array_like, shape (2,)
        First vector.
    v2 : array_like, shape (2,)
        Second vector.

    Returns
    -------
    double
        Result of the dot product v1 . v2.
    """
    return v1[0]*v2[0] + v1[1]*v2[1]

cdef void sub2(double[:] v1, double[:] v2, double[:] dest) noexcept nogil:
    """ 
    Substraction of vectors v1 and v2 
    
    Parameters
    ----------
    v1 : array_like, shape (2,)
        First vector.
    v2 : array_like, shape (2,)
        Second vector.

    Returns
    -------
    dest : ndarray, shape (2,)
        Result of the substraction v1 - v2.
    """
    dest[0] = v1[0] - v2[0]
    dest[1] = v1[1] - v2[1]

cdef void normal2(double[:] v0, double[:] v1, double[:] n) noexcept nogil:
    """ 
    Computes normal of edge (v0, v1) 
    
    Parameters
    ----------
    v0 : array_like, shape (2,)
        First vertex of the edge.
    v1 : array_like, shape (2,)
        Second vertex of the edge.

    Returns
    -------
    n : ndarray, shape (2,)
        Normal vector of the edge.
    """
    n[0] = v0[1] - v1[1]
    n[1] = v1[0] - v0[0]
    
cdef void normal2u(double[:] v0, double[:] v1, double[:] n) noexcept nogil:
    """ 
    Computes normal of edge (v0, v1) (normalized)

    Parameters
    ----------
    v0 : array_like, shape (2,)
        First vertex of the edge.
    v1 : array_like, shape (2,)
        Second vertex of the edge.

    Returns
    -------
    n : ndarray, shape (2,)
        Normalized normal vector of the edge.
    """
    cdef:
        double norm
    norm = sqrt((v1[0]-v0[0])**2 + (v1[1]-v0[1])**2)
    n[0] = (v0[1] - v1[1])/norm
    n[1] = (v1[0] - v0[0])/norm

# Basic 3d vectorial functions
cdef void cross(double[:] v1, double[:] v2, double[:] dest) noexcept nogil:
    """ 
    Cross product of vectors v1 and v2 

    Parameters
    ----------
    v1 : array_like, shape (3,)
        First vector.
    v2 : array_like, shape (3,)
        Second vector.
    dest : ndarray, shape (3,)
        Destination for the cross product v1 x v2.
    """
    dest[0] = v1[1]*v2[2] - v1[2]*v2[1]
    dest[1] = v1[2]*v2[0] - v1[0]*v2[2]
    dest[2] = v1[0]*v2[1] - v1[1]*v2[0]

cdef double dot(double[:] v1, double[:] v2) noexcept nogil:
    """ 
    Dot product of vectors v1 and v2 

    Parameters
    ----------
    v1 : array_like, shape (3,)
        First vector.
    v2 : array_like, shape (3,)
        Second vector.

    Returns
    -------
    double
        Result of the dot product v1 . v2.
    """
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

cdef void sub(double[:] v1, double[:] v2, double[:] dest) noexcept nogil:
    """ 
    Substraction of vectors v1 and v2 

    Parameters
    ----------
    v1 : array_like, shape (3,)
        First vector.
    v2 : array_like, shape (3,)
        Second vector.
    dest : ndarray, shape (3,)
        Destination for the substraction v1 - v2.
    """
    dest[0] = v1[0] - v2[0]
    dest[1] = v1[1] - v2[1]
    dest[2] = v1[2] - v2[2]

cdef void normal(double[:] v0, double[:] v1, double[:] v2, double[:] dest) noexcept nogil:
    """ 
    Computes normal of triangle v0,v1,v2 

    Parameters
    ----------
    v0 : array_like, shape (3,)
        First vertex of the triangle.
    v1 : array_like, shape (3,)
        Second vertex of the triangle.
    v2 : array_like, shape (3,)
        Third vertex of the triangle.
    dest : ndarray, shape (3,)
        Destination for the normal vector of the triangle.
    """
    dest[0] = (v1[1]-v0[1])*(v2[2]-v0[2]) - (v1[2]-v0[2])*(v2[1]-v0[1])
    dest[1] = (v1[2]-v0[2])*(v2[0]-v0[0]) - (v1[0]-v0[0])*(v2[2]-v0[2])
    dest[2] = (v1[0]-v0[0])*(v2[1]-v0[1]) - (v1[1]-v0[1])*(v2[0]-v0[0])

# Area computations
cdef double compute_triangle_area(\
        double xa, double ya,\
        double xb, double yb,\
        double xc, double yc) noexcept nogil:
    return 0.5*fabs((xb-xa)*(yc-ya)-(yb-ya)*(xc-xa))

cpdef double polygon_area(double[:] x, double[:] y) noexcept nogil:
    """ 
    Compute the area of a simple polygon 

    Parameters
    ----------
    x : array_like, shape (n,)
        x-coordinates of the polygon vertices.
    y : array_like, shape (n,)
        y-coordinates of the polygon vertices.

    Returns
    -------
    area : double
        Area of the polygon.
    """
    cdef:
        int n = x.shape[0]
        int i
        double area = 0.0
    if n < 3:
        return 0.0
    for i in range(n - 1):
        area += x[i] * y[i+1] - x[i+1] * y[i]
    area += x[n-1] * y[0] - x[0] * y[n-1]
    return 0.5 * fabs(area)

# Functions used for segment-segment intersection
cdef int orientation(double[:] v0, double[:] v1, double[:] v2) noexcept nogil:
    """ 
    Determines the orientation of the triplet (v0, v1, v2)

    Parameters
    ----------
    v0 : array_like, shape (2,)
        First vertex of the triplet.
    v1 : array_like, shape (2,)
        Second vertex of the triplet.
    v2 : array_like, shape (2,)
        Third vertex of the triplet.
    
    Returns
    -------
    int
        0 if collinear, 1 if clockwise, 2 if counterclockwise.
    """
    delta = (v0[1] - v1[1])*(v2[0] - v0[0]) - \
            (v0[0] - v1[0])*(v2[1] - v0[1])
    if delta == 0.:
        return 0
    elif delta > 0.:
        return 1
    return 2

cdef int _orientation2d(
        double v0x, double v0y,
        double v1x, double v1y,
    double v2x, double v2y) noexcept nogil:
    """
    Determines the orientation of the triplet (v0, v1, v2)

    Parameters
    ----------
    v0x, v0y : double
        Coordinates of the first vertex of the triplet.
    v1x, v1y : double
        Coordinates of the second vertex of the triplet.
    v2x, v2y : double
        Coordinates of the third vertex of the triplet. 

    Returns
    -------
    int
        0 if collinear, 1 if clockwise, 2 if counterclockwise.
    """
    cdef double delta = (v0y - v1y) * (v2x - v0x) - (v0x - v1x) * (v2y - v0y)
    if delta == 0.:
        return 0
    elif delta > 0.:
        return 1
    return 2

cdef bint on_segment(double[:] v0, double[:] v1, double[:] v2) noexcept nogil:
    """
    Check if the point v2 lies on the line segment v0v1.

    Parameters
    ----------
    v0 : array_like, shape (2,)
        First endpoint of the segment.
    v1 : array_like, shape (2,)
        Second endpoint of the segment.
    v2 : array_like, shape (2,)
        Point to check.

    Returns
    -------
    bool
        True if v2 lies on the segment v0v1, False otherwise.
    """
    if (v2[0] <= fmax(v1[0], v0[0]) and v2[0] >= fmin(v1[0], v0[0]) and
        v2[1] <= fmax(v1[1], v0[1]) and v2[1] >= fmin(v1[1], v0[1])):
        return 1
    return 0

cdef bint _on_segment2d(
        double v0x, double v0y,
        double v1x, double v1y,
    double v2x, double v2y) noexcept nogil:
    """
    Check if the point (v2x, v2y) lies on the line segment (v0x, v0y)-(v1x, v1y).

    Parameters
    ----------
    v0x, v0y : double
        Coordinates of the first endpoint of the segment.
    v1x, v1y : double
        Coordinates of the second endpoint of the segment.
    v2x, v2y : double
        Coordinates of the point to check.

    Returns
    -------
    bool
        True if (v2x, v2y) lies on the segment (v0x, v0y)-(v1x, v1y), False otherwise.
    """
    if (v2x <= fmax(v1x, v0x) and v2x >= fmin(v1x, v0x) and
        v2y <= fmax(v1y, v0y) and v2y >= fmin(v1y, v0y)):
        return 1
    return 0
    
# Point in triangle tests
cdef inline double pintri(\
        double x, double y,\
        double xa, double ya,\
        double xb, double yb,\
        double xc, double yc) noexcept nogil:
    """
    Determines if the point (x, y) is inside the triangle defined by (xa, ya), (xb, yb), (xc, yc).

    Parameters
    ----------
    x, y : double
        Coordinates of the point to check.
    xa, ya : double
        Coordinates of the first vertex of the triangle.
    xb, yb : double
        Coordinates of the second vertex of the triangle.
    xc, yc : double
        Coordinates of the third vertex of the triangle.

    Returns
    -------
    bool
        True if the point (x, y) is inside the triangle, False otherwise.
    """
    return ( (xb-xa)*( y-ya) - ( x-xa)*(yb-ya) ) \
         * ( ( x-xa)*(yc-ya) - (xc-xa)*( y-ya) )

cdef bint point_in_triangle_dotp(\
        double x , double y ,\
        double x0, double y0,\
        double x1, double y1,\
        double x2, double y2) noexcept nogil:
    """ 
    Point in triangle test using dot product

    Parameters
    ----------
    x, y : double
        Coordinates of the point to check.
    x0, y0 : double
        Coordinates of the first vertex of the triangle.
    x1, y1 : double
        Coordinates of the second vertex of the triangle.
    x2, y2 : double
        Coordinates of the third vertex of the triangle.

    Returns
    -------
    bool
        True if the point (x, y) is inside the triangle, False otherwise.
    """
    return pintri(x,y, x0,y0, x1,y1, x2,y2) >=0 \
       and pintri(x,y, x1,y1, x2,y2, x0,y0) >=0 \
       and pintri(x,y, x2,y2, x0,y0, x1,y1) >=0

cdef bint point_in_triangle_bbox(\
        double x , double y ,\
        double x0, double y0,\
        double x1, double y1,\
        double x2, double y2) noexcept nogil:
    """ 
    Point in triangle bounding box test

    Parameters
    ----------
    x, y : double
        Coordinates of the point to check.
    x0, y0 : double
        Coordinates of the first vertex of the triangle.
    x1, y1 : double
        Coordinates of the second vertex of the triangle.
    x2, y2 : double
        Coordinates of the third vertex of the triangle.

    Returns
    -------
    bool
        True if the point (x, y) is inside the triangle's bounding box, False otherwise.
    """
    return x>=fmin(x0, fmin(x1, x2)) and \
           x<=fmax(x0, fmax(x1, x2)) and \
           y>=fmin(y0, fmin(y1, y2)) and \
           y<=fmax(y0, fmax(y1, y2))
