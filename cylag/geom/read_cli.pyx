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
from ..core.constants cimport BND_WALL_REF, MAX_NUMBER_OF_OPEN_BND

def read_cli(file_name, vertex_id_column=11):
    """ 
    Read Telemac .cli boundary condition file 
    
    Parameters
    ----------
    file_name : str
        Path to the .cli file.
    vertex_id_column : int, optional
        Column index for vertex IDs (default is 11).

    Returns
    -------
    boundary_points : ndarray, shape (m, 2)
        Array containing vertex IDs and their corresponding boundary labels.
    new_boundary_count : int
        Number of new boundaries found in the .cli file.
    """
    cdef:
        bint new_boundary
        int new_boundary_count
        long[:] bnd_ids

    # init tables
    boundary_points = []

    # for indexing boundaries
    new_boundary = False
    new_boundary_count = 0
    bnd_ids = np.arange(1, MAX_NUMBER_OF_OPEN_BND+1, 1)

    # Read file .cli
    f = open(file_name, 'r')

    while True:
        # check number of open boundaries
        if new_boundary_count>MAX_NUMBER_OF_OPEN_BND:
            raise ValueError("Exceeding maximum number of open boundary (= {})"\
                .format(MAX_NUMBER_OF_OPEN_BND))
    
        line = f.readline()

        # end of file
        if not line:
            break

        line_l = line.split()

        # boundary points labels
        if int(line_l[0])==2 and int(line_l[1])==2 and int(line_l[2])==2:
            boundary_point_label = BND_WALL_REF
            new_boundary = False
        else:
            if new_boundary == False:
                new_boundary_count += 1
            boundary_point_label = bnd_ids[new_boundary_count-1]
            new_boundary = True

        # boundary points
        boundary_points.append([int(line_l[vertex_id_column])-1, boundary_point_label])
            
    f.close()

    boundary_points = np.asarray(boundary_points).astype(np.int32)
    
    # cut boundary fix
    nb = boundary_points.shape[0]
    if boundary_points[0, 1] != BND_WALL_REF and boundary_points[nb-1, 1] != BND_WALL_REF:
        last_bnd_value = boundary_points[nb-1, 1]
        boundary_points=np.where(boundary_points==last_bnd_value, boundary_points[0, 1], boundary_points)

    return boundary_points, new_boundary_count
