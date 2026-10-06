# cython: profile=False
#
import numpy as np
from ..geom.intersections cimport intersec2d_segment_segment
from ..geom.intersections cimport c_intersec2d_segment_segment

cdef class ControlSection():
    """
    Control sections class

    usage: to count particles that cross the control section

    Parameters
    ----------
    poly : (npoint, 2) array of float, polygon defining the control section
    """
    def __init__(self, poly):
        self.nseg = poly.shape[1]-1
        self.poly = poly

    cpdef bint cross(ControlSection self, double[:] pos_0, double[:] pos_1):
        """ 
        Check if trajectory cross the control section 

        Parameters
        ----------
        pos_0 : double[:]
            Starting position of the trajectory segment
        pos_1 : double[:]
            Ending position of the trajectory segment

        Returns
        -------
        bint
            0: trajectory does not cross the control section
            1: trajectory crosses the control section
        """
        cdef:
            int i
            double[:] seg_0 = np.empty((2), dtype='d')
            double[:] seg_1 = np.empty((2), dtype='d')

        # loop on segments of the control section:
        for i in range(self.nseg):
            seg_0[0] = self.poly[i, 0]
            seg_0[1] = self.poly[i, 1]
            seg_1[0] = self.poly[i + 1, 0]
            seg_1[1] = self.poly[i + 1, 1]
            
            # check if segment [pos_0, pos_1] cross the segment [seg_0, seg_1]
            if intersec2d_segment_segment(pos_0, pos_1, seg_0, seg_1):
                return 1

        return 0

    cdef bint c_cross(ControlSection self, double x0, double y0, double x1, double y1):
        """
        Fast scalar check if trajectory crosses the control section.
        No memoryview or numpy allocation overhead.

        Parameters
        ----------
        x0 : double
            Starting x-coordinate of the trajectory segment
        y0 : double
            Starting y-coordinate of the trajectory segment
        x1 : double
            Ending x-coordinate of the trajectory segment
        y1 : double
            Ending y-coordinate of the trajectory segment

        Returns
        -------
        bint
            0: trajectory does not cross the control section
            1: trajectory crosses the control section
        """
        cdef int i
        for i in range(self.nseg):
            if c_intersec2d_segment_segment(
                    x0, y0, x1, y1,
                    self.poly[i, 0], self.poly[i, 1],
                    self.poly[i + 1, 0], self.poly[i + 1, 1]):
                return 1
        return 0
