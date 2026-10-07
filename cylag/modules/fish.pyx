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
from scipy.stats import wrapcauchy
from libc.math cimport sqrt, cos, sin, acos, fabs
from ..core.constants cimport EPSILON
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..lsm.random_utils cimport random_uniform

cdef class FishParticleSet(LagrangianParticleSet):
    """
    Fish particle class 

    Parameters
    ----------
    fish_mass : float, mass of the fish (default: 0.3)
    fish_length : float, length of the fish (default: 0.152)

    swimming_model : int, swimming bahavior model
        1 : Fish IBM - LSM1 (swimming velocity)
        2 : Fish CBM - LSM1 (swimming velocity)
        3 : Fish IBM - LSM2 (swimming thrust force)
        4 : Fish CBM - LSM2 (swimming thrust force)
        
    swimming_threshold_fluid_velocity : (2) array of float
        threshold velocity that trigger change of state 
        (default : [0.15, 0.25])
    swimming_velocity_states : (3) array of float
        swimming velocity of fish for each latent state
        (default : [0.15, 0.25, 0.5])
    swimming_dispersion : (2) array of float
        dispersion of the direction of swimming for 
        latent states 2 and 3
        (default : [0.1, 0.2])
    swimming_thrust_forces : (3) array of float
        swimming thrust force of fish for each latent state
        (default : [6.e-4, 6.e-3, 6.5e-3])
        
    attraction_radius : float
        attraction radius for the CBM model (default: 10.)
    orientation_radius : float
        orientation radius for the CBM model (default: 5.)
    repulsion_radius : float
        repulsion radius for the CBM model (default: .5)
    blind_angle : float
        angle defining the cone of blindness of fish
        for the CBM model (default: 60.)
    """
    def __init__(self,\
            position=None, velocity=None, fluid_velocity_seen=None,\
            dim=2, initial_pool_size=10000, min_pool_size_update=1000,
            particle_density=1000., particle_diameter=0.1,
            particle_diameter_distribution=None,
            drag_coefficient_model=0, fish_drag_coefficient=0.06,
            added_mass_force=True, added_mass_coef=0.5,
            buoyancy_velocity_model=0, buoyancy_velocity=0.0,
            compute_depth=False, max_stranded=None,
            swimming_model=1,
            swimming_threshold_fluid_velocity=[0.15, 0.25],
            swimming_velocity_states=[0.15, 0.25, 0.5],
            swimming_dispersion=[0.1, 0.2], 
            swimming_thrust_forces=[6.e-4, 6.e-3, 6.5e-3],
            attraction_radius=10.,
            orientation_radius=5.,
            repulsion_radius=1.,
            blind_angle=30.):

        LagrangianParticleSet.__init__(self,\
            position, velocity, fluid_velocity_seen,\
            dim, initial_pool_size, min_pool_size_update,
            particle_density, particle_diameter,
            particle_diameter_distribution,
            drag_coefficient_model, fish_drag_coefficient, 
            added_mass_force, added_mass_coef,
            buoyancy_velocity_model, buoyancy_velocity,
            compute_depth, max_stranded)

        # Swmiming model parameters
        self.swimming_model = swimming_model
        
        # IBM model parameters
        self.swimming_threshold_fluid_velocity = np.asarray(swimming_threshold_fluid_velocity)
        self.swimming_velocity_states = np.asarray(swimming_velocity_states)
        self.swimming_thrust_forces = np.asarray(swimming_thrust_forces)
        self.swimming_dispersion = np.asarray(swimming_dispersion)

        # CBM model parameters
        self.attraction_radius = attraction_radius 
        self.orientation_radius = orientation_radius
        self.repulsion_radius = repulsion_radius
        self.blind_angle = blind_angle

        # additional velocity and force
        self.addvelocity = np.zeros((self.npart, self.dim), dtype='d')
        self.addforce = np.zeros((self.npart, self.dim), dtype='d')

        # particle properties
        if swimming_model==1 or swimming_model==2:
            # Swimming velocity (LSM1)
            self.additional_velocity = True
            self.additional_force = False
        elif swimming_model==3 or swimming_model==4:
            # Thrust force (LSM2)
            self.additional_velocity = False
            self.additional_force = True
        else:
            raise ValueError("WARNING : Unknown fish behaviour model")

    cdef void _increase_pool_size(FishParticleSet self, int size):
        """
        Resize particle pool arrays including fish extra arrays.

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

    cpdef void _sort_active_particles(FishParticleSet self):
        """
        Sort active particles including fish extra arrays.

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

    cdef void _parallel_split(FishParticleSet self, int offset, int count, int pool_size):
        """
        Split fish particle set for parallel computation.

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

    cdef void add_velocity_i(FishParticleSet self, EulerianFieldSet fset, int i, double dt):
        """ 
        Add custom velocity to the particle i (for LSM1 model)

        Parameters
        ----------
            i : int, particle index
            time : float, time
            dt : float, time step
        """
        if self.swimming_model==1:
            self.add_velocity_behavior_i_ibm1(i, dt)
        elif self.swimming_model == 2:
            self.add_velocity_behavior_i_cbm1(i, dt)

    cdef void add_force_i(FishParticleSet self, EulerianFieldSet fset, int i, double dt):
        """ 
        Add custom force to the particle i (for LSM2 model)

        Parameters
        ----------
            i : int, particle index
            time : float, time
            dt : float, time step
        """
        if self.swimming_model == 3:
            self.add_force_behavior_i_ibm1(i, dt)
        #elif self.swimming_model == 4:
        #    self.add_force_behavior_i_cbm1(i, dt)

    cdef void add_velocity_behavior_i_ibm1(FishParticleSet self, int i, double dt):
        """ Fish velocity individual behavior model """
        cdef:
            double[:] xp = self.last_position[i, :]
            double[:] up = self.last_velocity[i, :]
            double[:] us = self.fluid_velocity_seen[i, :]
            double[:] ub = self.addvelocity[i, :]
            double[:] us_trig = self.swimming_threshold_fluid_velocity
            double[:] swim_dir_dispersion = self.swimming_dispersion
            double swim_norm
            double swim_dir
            double us_norm

        # compute latent state
        # ~~~~~~~~~~~~~~~~~~~~
        # Fluid velocity seen norm at particle position
        us_norm = sqrt(us[0]**2 + us[1]**2)

        # low current
        if us_norm < us_trig[0]:
            state = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
        # intermediate current
        elif us_norm >= us_trig[0] and us_norm < us_trig[1]:
            state = np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1])
        # high current
        elif us_norm >= us_trig[1]:
            state = np.random.choice([0, 1, 2], p=[0.05, 0.15, 0.8])

        # compute swimming parameters (intensity, direction)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # State 0 : random walk
        if state==0:
            swim_norm = self.swimming_velocity_states[0]
            swim_dir = np.arctan2(ub[1], ub[0])
            gammat = wrapcauchy.rvs(1.-self.swimming_dispersion[0], size=1)
            swim_dir += gammat
            #swim_dir = 2.*np.pi*random_uniform() # totally random dir (not realistic)

        # State 1 : swimming against current
        elif state==1:
            swim_norm = self.swimming_velocity_states[1]
            if us_norm > 0.:
                # inverse of current direction
                alpha = np.arctan2(us[1], us[0]) + np.pi
                # swimming orientation relative to the current
                if swim_dir_dispersion[0] > 0.:
                    gammat = wrapcauchy.rvs(1.-swim_dir_dispersion[0], size=1)
                else:
                    gammat = 0.
                # swimming orientation relative to the ground
                swim_dir = alpha + gammat
            else:
                swim_dir = 2.*np.pi*random_uniform()
        # State 2 : swimming burst
        else:
            swim_norm = self.swimming_velocity_states[2]
            if us_norm > 0.:
                # inverse of current direction
                alpha = np.arctan2(us[1], us[0]) + np.pi
                # swimming orientation relative to the current
                if swim_dir_dispersion[1] > 0.:
                    gammat = wrapcauchy.rvs(1.-swim_dir_dispersion[1], size=1)
                else:
                    gammat = 0.
                # swimming orientation relative to the ground
                swim_dir = alpha + gammat
            else:
                swim_dir = 2.*np.pi*random_uniform()

        # compute behavior
        # ~~~~~~~~~~~~~~~~
        ub[0] = swim_norm*cos(swim_dir)
        ub[1] = swim_norm*sin(swim_dir)

        #if self.dim==3: # TODO implement 3D

    cdef void add_force_behavior_i_ibm1(FishParticleSet self, int i, double dt):
        """ Fish velocity individual behavior model """
        cdef:
            double[:] xp = self.last_position[i, :]
            double[:] up = self.last_velocity[i, :]
            double[:] us = self.fluid_velocity_seen[i, :]
            double[:] fb = self.addforce[i, :]
            double[:] us_trig = self.swimming_threshold_fluid_velocity
            double[:] swim_dir_dispersion = self.swimming_dispersion
            double mass = self.density*self.volume[i]
            double us_norm
            double swim_norm
            double swim_dir
            double alpha, gammat

        #if self.dim==2: # TODO implement 3D

        # compute latent state
        # ~~~~~~~~~~~~~~~~~~~~
        # Fluid velocity seen norm at particle position
        us_norm = sqrt(us[0]**2 + us[1]**2)

        # low current
        if us_norm < us_trig[0]:
            state = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
        # intermediate current
        elif us_norm >= us_trig[0] and us_norm < us_trig[1]:
            state = np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1])
        # high current
        elif us_norm >= us_trig[1]:
            state = np.random.choice([0, 1, 2], p=[0.05, 0.15, 0.8])

        # compute swimming parameters (intensity, direction)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # State 0 : random walk
        if state==0:
            swim_norm = self.swimming_thrust_forces[0]
            swim_dir = np.arctan2(fb[1], fb[0])
            gammat = wrapcauchy.rvs(1.-self.swimming_dispersion[0], size=1)
            swim_dir += gammat
            #swim_dir = 2.*np.pi*random_uniform() # totally random dir (not realistic)

        # State 1 : swimming against current
        elif state==1:
            swim_norm = self.swimming_thrust_forces[1]
            if us_norm > 0.:
                # inverse of current direction
                alpha = np.arctan2(us[1], us[0]) + np.pi
                # swimming orientation relative to the current
                if swim_dir_dispersion[0] > 0.:
                    gammat = wrapcauchy.rvs(1.-swim_dir_dispersion[0], size=1)
                else:
                    gammat = 0.
                # swimming orientation relative to the ground
                swim_dir = alpha + gammat
            else:
                swim_dir = 2.*np.pi*random_uniform()

        # State 2 : swimming burst
        else:
            swim_norm = self.swimming_thrust_forces[2]
            if us_norm > 0.:
                # inverse of current direction
                alpha = np.arctan2(us[1], us[0]) + np.pi
                # swimming orientation relative to the current
                if swim_dir_dispersion[1] > 0.:
                    gammat = wrapcauchy.rvs(1.-swim_dir_dispersion[1], size=1)
                else:
                    gammat = 0.
                # swimming orientation relative to the ground
                swim_dir = alpha + gammat
            else:
                swim_dir = 2.*np.pi*random_uniform()

        # compute behavior
        # ~~~~~~~~~~~~~~~~
        fb[0] = swim_norm*cos(swim_dir)
        fb[1] = swim_norm*sin(swim_dir)

    cdef void add_velocity_behavior_i_cbm1(FishParticleSet self, int i, double dt):
        """
        Fish velocity collective behavior model 
        
        Collective Memory and Spatial Sorting in Animal Groups
        Iain D. Couzin, Jens Krausew, Richard Jamesz, Graeme D. Ruxtony and Nigel R. Franks
        """
        cdef:
            int j
            int nja, njo, njr
            double[:] xp = self.position[i, :]
            double[:] up0 = self.last_velocity[i, :]
            double[:] up = self.velocity[i, :]
            double[:] ub = self.addvelocity[i, :]
            double[:] rij = np.zeros((self.dim), dtype='d') # unit vector pointing towards j 
            double rijnorm
            double us_norm, up_norm, upj_norm
            double swim_norm, swim_dir
            double ra = self.attraction_radius
            double ro = self.orientation_radius
            double rr = self.repulsion_radius
            double ba = self.blind_angle
            double dsa0, dso0, dsr0
            double dsa1, dso1, dsr1
            double gammat
            double alphaij

        # TODO: extend to 3D

        # initialize
        up_norm = sqrt(up0[0]**2 + up0[1]**2)
        nja = 0
        njo = 0
        njr = 0
        dsa0, dsa1 = 0., 0.
        dsr0, dsr1 = 0., 0.
        dso0 = up0[0]/max(up_norm, EPSILON)
        dso1 = up0[1]/max(up_norm, EPSILON)

        # loop on neighbor particles
        for j in range(self.npart):
            if j != i:
                rij[0] = self.position[j, 0] - xp[0]
                rij[1] = self.position[j, 1] - xp[1]
                rijnorm = max(sqrt(rij[0]**2 + rij[1]**2), EPSILON)

                # angle between velocity vector and rij
                if ba > 0.:
                    alphaij = acos((rij[0]*up0[0]+rij[1]*up0[1])/(rijnorm*up_norm))
                    alphaij = fabs(alphaij)*180./np.pi
                else:
                    alphaij = 0.

                # repulsion
                if rijnorm <= rr:
                    njr += 1
                    dsr0 -= rij[0]/rijnorm
                    dsr1 -= rij[1]/rijnorm

                # orientation
                if rijnorm > rr and rijnorm <= ro:
                    upj_norm = max(sqrt(self.last_velocity[j, 0]**2 +\
                                        self.last_velocity[j, 1]**2), EPSILON)

                    # check if in angle of vision
                    if alphaij <= 360.-ba:
                        njo += 1
                        dso0 += self.last_velocity[j, 0]/upj_norm
                        dso1 += self.last_velocity[j, 1]/upj_norm

                # attraction
                if rijnorm > ro and rijnorm <= ra:
            
                    # check if in angle of vision
                    if alphaij <= 360.-ba:
                        nja += 1
                        dsa0 += rij[0]/rijnorm
                        dsa1 += rij[1]/rijnorm

        # compute behavior
        # ~~~~~~~~~~~~~~~~
        swim_norm = self.swimming_velocity_states[0]

        # repulsion has the highest priority
        if njr > 0:
            ub[0] = swim_norm*dsr0
            ub[1] = swim_norm*dsr1
        else:
            # orientation and attraction
            if njo > 0 and nja == 0:
                ub[0] = swim_norm*dso0
                ub[1] = swim_norm*dso1
            elif nja > 0 and njo == 0:
                ub[0] = swim_norm*dsa0
                ub[1] = swim_norm*dsa1
            elif nja > 0 and njo > 0:
                ub[0] = swim_norm*0.5*(dsa0 + dso0)
                ub[1] = swim_norm*0.5*(dsa1 + dso1)
            else:
                ub[0] = swim_norm*dso0
                ub[1] = swim_norm*dso1

        # dispersion of the swimming direction
        if self.swimming_dispersion[0] > 0.:
            swim_dir = np.arctan2(ub[1], ub[0])
            gammat = wrapcauchy.rvs(1.-self.swimming_dispersion[0], size=1)
            swim_dir += gammat
            ub[0] = swim_norm*cos(swim_dir)
            ub[1] = swim_norm*sin(swim_dir)
