__version__ = '0.1'

from .triangular_mesh import TriangularMesh
from .intersections import compute_symetric_point3d, compute_symetric_point2d,\
    intersec2d_segment_segment, intersec2d_get_intersection_point, \
    MT_intersec3d_triangle_ray, intersec3d_triangle_segment
from .read_cli import read_cli
from .xylocalizer import xy_localize_point, xy_localize_point_in_neighbourhood,\
    xy_localize_point_from_path, xy_localize, xy_localize_point_kdtree
from .zlocalizer import \
compute_lower_layer_index, c_compute_lower_layer_index_array, c_compute_lower_layer_index,\
compute_lower_index, c_compute_lower_index
