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
import matplotlib.pylab as plt

cdef class GhostMeshGrid1D:
    """
    Ghost Mesh Grid 1D

    usage: to spacially sort large amount of particles efficiently
    
    properties:
        - Ghost mesh grid point are not stored, only the number of nodes
        - Linear transform of original space to account for interaction length
          i.e. ghost mesh dx = 1 <=> dx = interaction length on primal mesh
        - Ghost mesh grid nodes start at 0 and end at GhostMeshGrid1D.ng
          i.e. nodes = np.arange(0, GhostMeshGrid1D.ng, 1)

    Parameters
    ----------
    xmin: float, x lower bound
    xmax: float, x upper bound
    interaction_length: float, length of interaction
        used to define the disorsion ratio
    """
    def __init__(self, xmin, xmax, interaction_length):
        # Ghost mesh attributes
        self.ng = int((xmax-xmin)//interaction_length) + 1
        self.transform = np.array([-xmin, 1./interaction_length], dtype='d')

        # Sorting attributes        
        self.sorting_idx   = np.empty((0), dtype='int64')
        self.cellid_filled = np.empty((0), dtype='int64')
        self.cellid_jump   = np.empty((0), dtype='int64')

    cpdef double primal_to_ghost(GhostMeshGrid1D self, double x):
        """ Convert coordinate from primal to ghost space """
        return (x+self.transform[0])*self.transform[1]

    cpdef double ghost_to_primal(GhostMeshGrid1D self, double x):
        """ Convert coordinate from ghost to primal space """
        return (x/self.transform[1])-self.transform[0]

    cpdef int get_ghostcell_id(GhostMeshGrid1D self, double x):
        """ Get ghost cell id from coordinate in primal space """
        xg = self.primal_to_ghost(x)
        return int(xg)

    cpdef double[:] set_lookup_tables(self, double[:] nodes):
        """  
        Create cell id lookup tables and return sorted 
        array based on ghost cell ids

        Parameters
        ----------
        nodes: array, nodes to locate and sort
        
        Returns
        -------
        nodes: array, sorted array
        cellid: array, ghost cell id in ascending order
        cellid_filled: array, id of filled ghost cell
        cellid_jump: array, id of ghost cell change in cell_id
        """
        cdef:
            int i
            int ni = len(nodes)
            int nif
            long[:] cellid     = np.empty((ni),   dtype='int64')
            long[:] cellid_ref = np.empty((ni+1), dtype='int64')
            long[:] tmp_arri   = np.empty((ni),   dtype='int64')
            double[:] tmp_arrd = np.empty((ni),   dtype='d')

        # compute ghost cell id for each point
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i in range(ni):
            cellid[i] = self.get_ghostcell_id(nodes[i])

        # sort cellid and nodes
        # ~~~~~~~~~~~~~~~~~~~~~
        self.sorting_idx = np.argsort(cellid)
        for i in range(ni):
            tmp_arri[i] = cellid[self.sorting_idx[i]]
            tmp_arrd[i] = nodes[self.sorting_idx[i]]
        cellid[:] = tmp_arri[:]
        nodes[:]  = tmp_arrd[:]

        # create cell_id_jump, cell_id_filled
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # extend cell id with one fake cell to get last jump
        cellid_ref[0:ni] = cellid[0:ni]
        cellid_ref[ni]   = cellid[ni-1]
        # remove duplicates with numpy
        self.cellid_filled, self.cellid_jump = np.unique(cellid_ref, return_index=True)
        nif = len(self.cellid_filled)
        # delete last fake cell
        self.cellid_filled = self.cellid_filled[0:nif-1] 

        return nodes

    cpdef double[:] sort_table(self, double[:] table):
        """ Sort table based on the same sorting index as cellid """
        cdef:             
            int i
            int ni = len(table)
            double[:] tmp_arrd = np.empty((ni),   dtype='d')
        for i in range(ni):
            tmp_arrd[i] = table[self.sorting_idx[i]]
        table[:]  = tmp_arrd[:]
        return table


cdef class GhostMeshGrid2D:
    """
    Ghost Mesh Grid 2D

    usage: to spacially sort large amount of particles efficiently

    properties:
        - Ghost mesh grid point are not stored, only the number of nodes
        - Linear transform of original space to account for interaction length
          i.e. ghost mesh dx = 1 <=> dx = interaction length on primal mesh
        - Ghost mesh grid nodes start at [0, 0] and end at 
          [GhostMeshGrid1D.ngx, GhostMeshGrid1D.ngy]
    
    Parameters
    ----------
    xmin: float, x lower bound
    xmax: float, x upper bound
    ymin: float, y lower bound
    ymax: float, y upper bound
    interaction_length: float, length of interaction
        used to define the disorsion ratio
    """
    def __init__(self, xmin, xmax, ymin, ymax, interaction_length):
        self.ngx = int((xmax-xmin)//interaction_length) + 1
        self.ngy = int((ymax-ymin)//interaction_length) + 1
        self.ng = self.ngx*self.ngy
        self.transform = np.array([-xmin, -ymin, 1./interaction_length], dtype='d')
        
        # Sorting attributes        
        self.sorting_idx   = np.empty((0), dtype='int64')
        self.cellid_filled = np.empty((0), dtype='int64')
        self.cellid_jump   = np.empty((0), dtype='int64')
        
    cpdef double[:] primal_to_ghost(GhostMeshGrid2D self, double[:] P):
        " Convert coordinate from primal to ghost space """
        cdef: 
            double[:] M = np.empty((2), dtype='d')
        M[0] = (P[0]+self.transform[0])*self.transform[2]
        M[1] = (P[1]+self.transform[1])*self.transform[2]
        return M

    cpdef double[:] ghost_to_primal(GhostMeshGrid2D self, double[:] P):
        """ Convert coordinate from ghost to primal space """
        cdef: 
            double[:] M = np.empty((2), dtype='d')
        M[0] = (P[0]/self.transform[2])-self.transform[0]
        M[1] = (P[1]/self.transform[2])-self.transform[1]
        return M

    cpdef int get_ghostcell_idx(GhostMeshGrid2D self, double[:] P):
        """ Get ghost cell id from coordinate in primal space """
        P = self.primal_to_ghost(P)
        return int(P[0])
    
    cpdef int get_ghostcell_idy(GhostMeshGrid2D self, double[:] P):
        """ Get ghost cell id from coordinate in primal space """
        P = self.primal_to_ghost(P)
        return int(P[1])
    
    cpdef int get_ghostcell_id(GhostMeshGrid2D self, double[:] P):
        """ Get ghost cell id from coordinate in primal space """
        cdef: 
            int[:] I = np.empty((2), dtype='int32')
        P = self.primal_to_ghost(P)
        I[0] = int(P[0])
        I[1] = int(P[1])
        return self.id_from_grid_ij(I)

    cpdef int[:] grid_ij_from_id(GhostMeshGrid2D self, int k):
        cdef: 
            int[:] I = np.empty((2), dtype='int32')
        I[0] = k%self.ngx
        I[1] = k//self.ngx
        return I
    
    cpdef int id_from_grid_ij(GhostMeshGrid2D self, int[:] I):
        return I[0] + I[1]*self.ngx
    
    cpdef double[:,:] set_lookup_tables(self, double[:,:] nodes):
        """  
        Create cell id lookup tables and return sorted 
        array based on ghost cell ids

        Parameters
        ----------
        nodes: array, nodes to locate and sort

        Returns
        -------
        nodes: array, sorted array
        cellid: array, ghost cell id in ascending order
        cellid_filled: array, id of filled ghost cell
        cellid_jump: array, id of ghost cell change in cell_id
        """
        cdef:
            int i
            int ni = len(nodes[:, 0])
            int nif
            long[:] cellid     = np.empty((ni),   dtype='int64')
            long[:] cellid_ref = np.empty((ni+1), dtype='int64')
            long[:] tmp_arri   = np.empty((ni),   dtype='int64')
            double[:] P = np.empty((2), dtype='d')
            double[:,:] tmp_arrd = np.empty((ni,2),   dtype='d')
            
        # compute ghost cell id for each point
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i in range(ni):
            P[0] = nodes[i, 0]
            P[1] = nodes[i, 1]
            cellid[i] = int(self.get_ghostcell_id(P))

        # sort cellid and nodes
        # ~~~~~~~~~~~~~~~~~~~~~
        self.sorting_idx = np.argsort(cellid)
        for i in range(ni):
            tmp_arri[i] = cellid[self.sorting_idx[i]]
            tmp_arrd[i, 0] = nodes[self.sorting_idx[i], 0]
            tmp_arrd[i, 1] = nodes[self.sorting_idx[i], 1]
        cellid[:]  = tmp_arri[:]
        nodes[:,:] = tmp_arrd[:,:]

        # create cell_id_jump, cell_id_filled
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # extend cell id with one fake cell to get last jump
        cellid_ref[0:ni] = cellid[0:ni]
        cellid_ref[ni]   = cellid[ni-1]
        # remove duplicates with numpy
        self.cellid_filled, self.cellid_jump = np.unique(cellid_ref, return_index=True)
        nif = len(self.cellid_filled)
        # delete last fake cell
        self.cellid_filled = self.cellid_filled[0:nif-1]

        return nodes

    cpdef double[:] sort_table(self, double[:] table):
        """ Sort table based on the same sorting index as cellid """
        cdef:             
            int i
            int ni = len(table)
            double[:] tmp_arrd = np.empty((ni),   dtype='d')
        for i in range(ni):
            tmp_arrd[i] = table[self.sorting_idx[i]]
        table[:]  = tmp_arrd[:]
        return table
