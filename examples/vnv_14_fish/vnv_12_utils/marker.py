# -*- coding: utf-8 -*-
import numpy as np

def define_custom_marker(alpha, shape=0):
    from matplotlib.path import Path
    # basic form
    if shape==0:
        A = (+0.1, 0)     # head
        B = (0, -0.5)    # lower curv
        C = (-1.5,+0.)   # upper tail 
        D = (-1.5,-0.)   # lower tail 
        E = ( 0, 0.5)    # upper curv
    else:
        A = (+1, 0)      # head
        B = (0, -0.8)    # lower curv
        C = (-1.5,+0.3)  # upper tail 
        D = (-1.5,-0.3)  # lower tail 
        E = ( 0, 0.8)    # upper curv
    verts = [A, B, C, D, E, A]
    # rotation of vertices
    for idx, pt in enumerate(verts):
        xnew = pt[0]*np.cos(alpha) - pt[1]*np.sin(alpha)
        ynew = pt[0]*np.sin(alpha) + pt[1]*np.cos(alpha)
        verts[idx] = (xnew, ynew)
    # path codes
    codes = [
        Path.MOVETO, #begin the figure in the lower right
        Path.CURVE3, #start a 3 point curve with the control point in lower left
        Path.LINETO, #end curve in the upper left
        Path.LINETO, #end curve in the upper left
        Path.CURVE3, #start a new 3 point curve with the upper right as a control point
        Path.LINETO, #end curve in lower right
    ]
    path = Path(verts,codes)
    return path
