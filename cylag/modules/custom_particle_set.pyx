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
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet

cdef class CustomParticleSet(LagrangianParticleSet):
    """
    Custom particle class for user defined additional velocity or force
    
    Parameters
    ----------
    additional_velocity : bool, activate custom model based on added velocity
        WARNING: compatible with LSM-1 formulation only
    additional_force : bool, activate custom model based on added force
        WARNING: compatible with LSM-2 and 3 formulations only
    """
    def __init__(self,\
            position=None, velocity=None, fluid_velocity_seen=None,\
            dim=2, initial_pool_size=10000, min_pool_size_update=1000,
            particle_density=1000., particle_diameter=0.01,
            drag_coefficient_model=1, drag_coefficient=0.44,
            added_mass_force=False, added_mass_coef=0.5,
            buoyancy_velocity_model=0, buoyancy_velocity=0.0,
            compute_depth=False, max_stranded=None,
            additional_velocity=False, additional_force=False):

        # intialize parent class
        LagrangianParticleSet.__init__(self,\
            position, velocity, fluid_velocity_seen,\
            dim, initial_pool_size, min_pool_size_update,
            particle_density, particle_diameter,
            drag_coefficient_model, drag_coefficient,
            added_mass_force, added_mass_coef,
            buoyancy_velocity_model, buoyancy_velocity,
            compute_depth, max_stranded)

        # additional velocity and force
        self.addvelocity = np.zeros((self.npart, self.dim), dtype='d')
        self.addforce = np.zeros((self.npart, self.dim), dtype='d')
        self.additional_velocity = additional_velocity
        self.additional_force = additional_force

    cdef void _increase_pool_size(CustomParticleSet self, int size):
        """
        Resize particle pool arrays including custom extra arrays.

        Overrides the parent method to also resize addvelocity and addforce
        arrays when the pool expands (e.g. particle sources).

        Parameters
        ----------
        size : int
            Number of new particles to add
        """
        cdef int old_npart = self.npart

        # call parent to resize base arrays
        LagrangianParticleSet._increase_pool_size(self, size)

        # compute actual size added (parent uses max(min_pool_size_update, size))
        cdef int size_added = self.npart - old_npart

        # resize extra arrays
        self.addvelocity = np.concatenate(
            (self.addvelocity, np.zeros((size_added, self.dim), dtype='d')), axis=0)
        self.addforce = np.concatenate(
            (self.addforce, np.zeros((size_added, self.dim), dtype='d')), axis=0)

    cpdef void _sort_active_particles(CustomParticleSet self):
        """
        Sort active particles including custom extra arrays.

        Overrides the parent method to also reorder addvelocity and addforce
        arrays consistently with the base arrays.
        """
        # compute sorting index before parent modifies inactive array
        sorting_idx = np.argsort(self.inactive)

        # sort extra arrays
        self.addvelocity = np.asarray(self.addvelocity)[sorting_idx, :]
        self.addforce = np.asarray(self.addforce)[sorting_idx, :]

        # call parent to sort base arrays
        LagrangianParticleSet._sort_active_particles(self)

    cdef void _parallel_split(CustomParticleSet self, int offset, int count, int pool_size):
        """
        Split custom particle set for parallel computation.

        Calls parent to handle base arrays, then rebuilds addvelocity
        and addforce arrays for this rank.

        Parameters
        ----------
        offset : int
            Starting index in the active particle arrays
        count : int
            Number of active particles to keep for this rank
        pool_size : int
            Total pool size allocated for this rank
        """
        # Call parent to handle base arrays
        LagrangianParticleSet._parallel_split(self, offset, count, pool_size)

        # Rebuild extra arrays
        self.addvelocity = np.zeros((self.npart, self.dim), dtype='d')
        self.addforce = np.zeros((self.npart, self.dim), dtype='d')

    cdef void add_velocity_i(CustomParticleSet self, EulerianFieldSet fset, int i, double dt):
        """ 
        Additional velocity (Ua) to the particle i (for LSM1 model)

        Parameters
        ----------
            i : int, particle index
            time : float, time
            dt : float, time step
        """
        cdef:
            double[:] xp = self.position[i, :]
            double[:] up = self.velocity[i, :]
            double[:] us = self.fluid_velocity_seen[i, :]
            double[:] ua = self.addvelocity[i, :]

        # TODO : define additional velocity (m/s)
        #
        # for example:
        ua[0] = 1.
        ua[1] = 1.
        if self.dim==3:
            ua[2] = 0.

    cdef void add_force_i(CustomParticleSet self, EulerianFieldSet fset, int i, double dt):
        """ 
        Additional force (Fa) to the particle i (for LSM2/3 model)

        Note : for equivalence with LSM1 model, the additional force can be defined as:
            Fa = (A1/A3)*Ua, meaning Up relax to Uf + Ua.

        Parameters
        ----------
            i : int, particle index
            time : float, time
            dt : float, time step
        """
        cdef:
            double[:] xp = self.position[i, :]
            double[:] up = self.velocity[i, :]
            double[:] us = self.fluid_velocity_seen[i, :]
            double[:] fa = self.addforce[i, :]

        # TODO : define additional force (N)
        #
        # for example:
        fa[0] = 0.
        fa[1] =-1.e-4
        if self.dim==3:
            fa[2] = 0.
