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
import os
import numpy as np
from matplotlib.patches import Polygon
from matplotlib import rc
from matplotlib.path import Path

# Plotting style parameters
# =========================
def set_rcparams():
    """ Set plot style parameters """
    rc('figure', autolayout=True)
    rc('text.latex', preamble=r'\usepackage{lmodern}')
    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=12)
    rc('legend', framealpha=0.7)
    rc('axes', xmargin=0.)
    rc('axes', ymargin=0.)
    rc('axes', labelsize=14)
    rc('axes', labelpad=4.0)
    rc('grid', color='b0b0b0')
    rc('grid', linestyle=':')
    rc('grid', linewidth=.8)
    rc('grid', alpha=1.)

def get_default_color_palette():
    colors1 = ['#002d74', '#e85113', '#1fa12e', '#c9d200', '#f49e00', '#006ab3', '#3381ff', '#f2855a', '#54de64']
    colors2 = ['#3381ff', '#f2855a', '#54de64', '#f5ff33', '#ffb833', '#33adff', '#99c0ff', '#f7b9a1', '#a9efb1']
    colors3 = ['#99c0ff', '#f7b9a1', '#a9efb1', '#faff99', '#ffdb99', '#99d6ff', '#f7b9a1', '#a9efb1', '#faff99']
    return colors1, colors2, colors3

def get_default_markers():
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'd', 'H']
    return markers

# Plot mesh attributes
# ====================
def plot_cylag_triangle(ax, triangulation, indices, color='b', alpha=0.25):
    """ 
    Plot cylag triangle 

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    triangulation : object
        The triangulation object containing the triangles.
    indices : list of int
        Indices of the triangles to plot.
    color : str, optional
        Color of the triangles (default is 'b').
    alpha : float, optional
        Transparency of the triangles (default is 0.25).
    """
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

def plot_cylag_boundary_points(ax, triangulation, colors=['k', 'r', 'g', 'b']):
    """ 
    Plot cylag boundary points 
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    triangulation : object
        The triangulation object containing the boundary points.
    colors : list of str, optional
        List of colors for the points (default is ['k', 'r', 'g', 'b']).
    """
    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    bnd_pts = np.asarray(triangulation.boundary_points)

    for i in range(bnd_pts.shape[0]):
        v0 = bnd_pts[i, 0]
        xs = [triangulation.x[v0]]
        ys = [triangulation.y[v0]]
        ax.plot(xs, ys, c=colors[bnd_pts[i, 1]], lw=1.5, marker='o', markersize=5)

def plot_cylag_boundary_edges(ax, triangulation, colors=['k', 'r', 'g', 'b']):
    """ 
    Plot cylag boundary edges 
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    triangulation : object
        The triangulation object containing the boundary edges.
    colors : list of str, optional
        List of colors for the edges (default is ['k', 'r', 'g', 'b']).
    """
    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    bnd_edges = np.asarray(triangulation.boundary_edges)

    for i in range(bnd_edges.shape[0]):
        v0, v1 = bnd_edges[i, 0], bnd_edges[i, 1]
        xs = [triangulation.x[v0], triangulation.x[v1]]
        ys = [triangulation.y[v0], triangulation.y[v1]]
        ax.plot(xs, ys, c=colors[bnd_edges[i, 2]], lw=1.5, marker='+', markersize=2)
        
def plot_cylag_edges(ax, triangulation, indices, colors=['k', 'r', 'g', 'b']):
    """ 
    Plot cylag edges 
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    triangulation : object
        The triangulation object containing the edges.
    indices : list of int
        Indices of the edges to plot.
    colors : list of str, optional
        List of colors for the edges (default is ['k', 'r', 'g', 'b']).
    """
    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    edges = np.asarray(triangulation.edges)
    for i in indices:
        v0, v1 = edges[i, 0], edges[i, 1]
        xs = [triangulation.x[v0], triangulation.x[v1]]
        ys = [triangulation.y[v0], triangulation.y[v1]]
        j = triangulation.edges_labels[i]
        ax.plot(xs, ys, c=colors[j], lw=1.5, marker='o', markersize=5)
        
def plot_cylag_edgetri_map(ax, triangulation, indices):
    """ 
    Plot cylag edges triangles map 
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    triangulation : object
        The triangulation object containing the triangles and their neighbors.
    indices : list of int
        Indices of the edges to plot.
    """
    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    edges = np.asarray(triangulation.edges)
    edgetri = np.asarray(triangulation.edgetri_map)
    for i in indices:
        tri0, tri1 = edgetri[i, 0], edgetri[i, 1]
        plot_cylag_triangle(ax, triangulation, [tri0], color='r', alpha=0.25)
        plot_cylag_triangle(ax, triangulation, [tri1], color='r', alpha=0.25)
        v0, v1 = edges[i, 0], edges[i, 1]
        xs = [triangulation.x[v0], triangulation.x[v1]]
        ys = [triangulation.y[v0], triangulation.y[v1]]
        ax.plot(xs, ys, c='b', lw=1.5, marker='o', markersize=5)

def plot_cylag_triedge_map(ax, triangulation, indices):
    """ 
    Plot cylag edges triangles map 
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    triangulation : object
        The triangulation object containing the triangles and their neighbors.
    indices : list of int
        Indices of the triangles to plot.
    """
    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    edges = np.asarray(triangulation.edges)
    triedge = np.asarray(triangulation.triedge_map)
    for i in indices:
        e0, e1, e2 = triedge[i, 0], triedge[i, 1], triedge[i, 2]
        plot_cylag_triangle(ax, triangulation, [i], color='b', alpha=0.25)
        plot_cylag_edges(ax, triangulation, [e0, e1, e2], colors='r')

def plot_cylag_adjacent_triangles(ax, triangulation, indices, color='b', alpha=0.25):
    """ 
    Plot cylag triangle

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    triangulation : object
        The triangulation object containing the triangles and their neighbors.
    indices : list of int
        Indices of the triangles to plot.
    color : str, optional
        Color of the neighboring triangles (default is 'b').
    alpha : float, optional
        Transparency of the neighboring triangles (default is 0.25).
    """
    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    triangles = np.asarray(triangulation.triangles)
    for index in indices:
        # plot main triangles
        polygon = Polygon([[0, 0], [0, 0]], facecolor='b', alpha=alpha)
        if index==-1:
            points = [0, 0, 0]
        else:
            points = triangles[index]
        xs = x[points]
        ys = y[points]
        polygon.set_xy(np.column_stack([xs, ys]))
        ax.add_patch(polygon)
        # plot neighboring triangles
        neighbors_id = np.asarray(triangulation.triangle_adjacency[index, :])
        plot_cylag_triangle(ax, triangulation, neighbors_id, color=color, alpha=0.25)
        
def plot_cylag_neighbor_triangles(ax, triangulation, indices, color='b', alpha=0.25):
    """
    Plot cylag triangle and its neighbors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    triangulation : object
        The triangulation object containing the triangles and their neighbors.
    indices : list of int
        Indices of the triangles to plot.
    color : str, optional
        Color of the neighboring triangles (default is 'b').
    alpha : float, optional
        Transparency of the neighboring triangles (default is 0.25).
    """
    x = np.asarray(triangulation.x)
    y = np.asarray(triangulation.y)
    triangles = np.asarray(triangulation.triangles)
    for index in indices:
        # plot main triangles
        polygon = Polygon([[0, 0], [0, 0]], facecolor='b', alpha=alpha)
        if index==-1:
            points = [0, 0, 0]
        else:
            points = triangles[index]
        xs = x[points]
        ys = y[points]
        polygon.set_xy(np.column_stack([xs, ys]))
        ax.add_patch(polygon)
        # plot neighboring triangles
        neighbors_id = np.asarray(triangulation.triangle_neighbors[index, :])
        plot_cylag_triangle(ax, triangulation, neighbors_id, color=color, alpha=0.25)

# Make video
# ==========
def ffmpeg_img_to_video(img_folder, img_name, videoname, framerate=30):
    """
    Use ffmpeg to convert a list of png into a mp4 video

    Remarks
    -------
    - This function requires ffmpeg to be installed on the system.
    - Numbering of the images must be: img_name00001.png, img_name00002.png, etc.

    Parameters
    ----------
    img_folder : str
        Path to the folder containing the images.
    img_name : str
        Name of the images (without the number and extension).
    videoname : str
        Name of the output video (without extension).
    framerate : int
        Framerate of the output video (default: 30).
    """
    os.system("ffmpeg -r {} -i {}%05d.png -vcodec libx264 \
              -pix_fmt yuv420p -crf 25 -movflags +faststart {}.mp4"
              .format(framerate, img_folder+img_name, videoname))

# Custom markers for plots:
# =========================
def define_custom_marker(alpha, shape=0, marker_type='fish'):
    """
    Define a custom marker for plots.

    Parameters
    ----------
    alpha : float
        Rotation angle of the marker in radians.
    shape : int, optional
        Shape variant of the marker (default is 0).
    marker_type : str, optional
        Type of the marker (default is 'fish').

    Returns
    -------
    path : matplotlib.path.Path
        Path object representing the custom marker.
    """
    if marker_type=='fish':
        # basic form
        if shape==0:
            A = (+0.1, 0)    # head
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
    else:
        raise ValueError("Unknown marker type: {}".format(marker_type))

import numpy as np

# Polygon i2s reader
# ==================
def read_i2s_polygon(file_path: str) -> np.ndarray:
    """
    Read polygon coordinates from an I2S file.

    Parameters
    ----------
    file_path : str
        Path to the I2S file.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2) containing (x, y) coordinates.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    try:
        header_end = next(i for i, line in enumerate(lines) if ":EndHeader" in line)
    except StopIteration:
        raise ValueError("Keyword ':EndHeader' not found in I2S file.")

    try:
        n_points = int(lines[header_end + 1].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("Invalid or missing point count after ':EndHeader'.") from exc

    try:
        return np.array(
            [list(map(float, line.split()[:2]))
             for line in lines[header_end + 2 : header_end + 2 + n_points]])
    except (ValueError, IndexError) as exc:
        raise ValueError("Invalid coordinate data in I2S file.") from exc