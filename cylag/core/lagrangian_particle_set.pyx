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
from ..core.constants cimport NU0, EPSILON, DEFAULT_INITIAL_POOL_SIZE, GRAV
from ..core.particle_distribution cimport draw_particle_diameter, check_particle_distribution
from ..core.eulerian_field_set cimport EulerianFieldSet
from libc.math cimport fmax, sqrt, fabs

cdef class LagrangianParticleSet:
    """
    Particles class containg particles

    Parameters
    ----------
    position : ndarray, shape (npart, dim), dtype float64, optional
        particle coordinates x,y,z (default: None)
    velocity : ndarray, shape (npart, dim), dtype float64, optional
        particle velocity up,vp,wp (default: None)
    fluid_velocity_seen : ndarray, shape (npart, dim), dtype float64, optional
        fluid velocity seen us,vs,ws (default: None)
    dim : int, optional
        dimension 2 or 3 (default: 2)
    initial_pool_size : int, optional
        initial size of particle pool (default: 10000)
    min_pool_size_update : int, optional
        minimum size increment for pool expansion (default: 1000)
    ini_tag : int, optional
        particle first tag (default: 0)
    particle_density : float, optional
        particle density (default: 1000.)
    particle_diameter : float, optional
        mean particle diameter (default: 0.01)
    particle_diameter_distribution : list, optional
        particle diameter distribution (default: None)
        - None: constant diameter, monodisperse
        - ['uniform', delta]: U(particle_diameter-delta, particle_diameter+delta)
        - ['normal', std]: N(particle_diameter, std)
        - ['lognormal', gsd]: LN(particle_diameter, gsd)
        - ['weibull', shape]: scale=particle_diameter/tgamma(1 + 1/shape)
        - ['tabulated', None]: user-defined tabulated distribution [dp, CDF]
                               (see cylag/core/particle_distribution.pyx)
    drag_coefficient_model : int, optional
        model for the drag coefficient (default: 1)
        - 0: Constant drag model
        - 1: Schiller et Nauman, (1935)
        - 2: Almedeij (2008)
    drag_coefficient : float, optional
        constant drag coefficient for drag_coefficient_model=0 (default: 0.44)
    added_mass_force : bool, optional
        activate added mass force (default: False)
    added_mass_coef : float, optional
        added mass coefficient (default: 0.5)
    buoyancy_velocity_model : int, optional
        model for the buoyancy velocity for the LSM-1 model (default: 1)
        - 0: Constant buoyancy velocity model
        - 1: Analytical solution
    buoyancy_velocity : float, optional
        particle buoyancy velocity (default: 0.)
    compute_depth : bool, optional
        compute particle depth to free surface (default: False)
        add depth to output files if True (only for 3D)
    max_stranded : int, optional
        maximum number of stranded particles (default: None)
        set to initial_pool_size if None

    Attributes
    ----------
    density : float
        particle density
    diameter : float
        particle diameter
    volume : float
        particle volume
    surface : float
        particle surface
    position : ndarray, shape (npart_pool, dim), dtype float64
        particle coordinates x,y,z
    velocity : ndarray, shape (npart_pool, dim), dtype float64
        particle velocity up,vp,wp
    fluid_velocity_seen : ndarray, shape (npart_pool, dim), dtype float64
        fluid velocity seen us,vs,ws
    tag : ndarray, shape (npart_pool,), dtype int32
        tag index of a particle
    npart : int
        total number of particles in pool
    npart_active : int
        number of active particles
    inactive : ndarray, shape (npart_pool,), dtype int32
        0 if active particle, 1 if inactive particle
    tri : ndarray, shape (npart_pool,), dtype int32
        triangle containing particle
        -1 if not in a triangle (aside domain)
        -2 if new particle
    lowerlayer : ndarray, shape (npart_pool,), dtype int32
        lower layer containing particle (z localization)
        -2 if below bottom, -3 if above free surface
    last_tri : ndarray, shape (npart_pool,), dtype int32
        previous triangle containing particle
    last_position : ndarray, shape (npart_pool, dim), dtype float64
        particle previous coordinates x,y,z
    last_velocity : ndarray, shape (npart_pool, dim), dtype float64
        particle previous velocity
    dim : int
        Dimension (2 or 3)
    min_pool_size_update : int
        Minimum size increment for pool expansion
    maxtag : int
        Maximum tag assigned to particles
    drag_coef_model : int
        Drag coefficient model type
    drag_coef : float
        Drag coefficient value
    added_mass_force : bool
        Whether added mass force is activated
    added_mass_coef : float
        Added mass coefficient
    additional_velocity : bool
        Whether additional velocity is considered
    additional_force : bool
        Whether additional force is considered
    a0c : double
        Model coefficient for buoyancy
    a1c : double
        Model coefficient for drag
    a2c : double
        Model coefficient for momentum
    a3c : double
        Model coefficient for additional force
    max_stranded : int
        Maximum number of stranded particles
    compute_depth : bool
        Whether to compute particle depth to free surface
    depth : ndarray, shape (npart_pool,), dtype float64
        Particle depth to free surface (only for 3D)
    stranded_count : int
        Number of stranded particles
    stranded_positions : ndarray, shape (max_stranded, 3), dtype float64
        Stranded particles positions
    """
    def __init__(LagrangianParticleSet self,\
            position=None, velocity=None, fluid_velocity_seen=None,\
            dim=2, initial_pool_size=None, min_pool_size_update=1000,
            particle_density=1000., particle_diameter=0.01,
            particle_diameter_distribution=None,
            drag_coefficient_model=1, drag_coefficient=0.44,
            added_mass_force=False, added_mass_coef=0.5,
            buoyancy_velocity_model=0, buoyancy_velocity=0.0,
            compute_depth=False, max_stranded=None):

        # dimension
        if position is not None and initial_pool_size is None:
            self.npart = np.shape(position)[0]
        else:
            self.npart = initial_pool_size if initial_pool_size is not None\
                else DEFAULT_INITIAL_POOL_SIZE

        self.dim = dim
        self.min_pool_size_update = min_pool_size_update

        # pool of particles
        self.npart_active = 0
        self.inactive = np.ones(self.npart, dtype='int32')

        # Particle properties
        self.rho0 = 1000. # default water density; overridden by solver from parameters.water_density
        self.density = particle_density

        # particle mean diameter and volume
        self.particle_diameter = particle_diameter
        self.particle_volume = np.pi*(self.particle_diameter**3)/6.

        # particle diameter distribution
        if particle_diameter_distribution is not None:
            # poly-disperse distribution
            self.diameter_distribution = particle_diameter_distribution[0]
            self.diameter_distribution_param = particle_diameter_distribution[1]
            # check distribution parameters
            check_particle_distribution(self.particle_diameter,\
                self.diameter_distribution, self.diameter_distribution_param)
        else:
            # mono-disperse distribution
            self.diameter_distribution = 'monodisperse'
            self.diameter_distribution_param = 0.0

        self.diameter = np.zeros(self.npart, dtype='d')
        self.volume = np.zeros(self.npart, dtype='d')
        self._initialize_particle_distribution()

        # Particle state vector: z_p = {x_p, u_p, u_s}
        self.position = np.zeros((self.npart, self.dim), dtype='d')
        self.velocity = np.zeros((self.npart, self.dim), dtype='d')
        self.fluid_velocity_seen = np.zeros((self.npart, self.dim), dtype='d')

        # Track depth to surface if requested
        self.compute_depth = compute_depth
        if compute_depth and self.dim != 3:
            raise ValueError("compute_depth can only be True for 3D pset")
        if self.compute_depth:
            self.depth = np.zeros(self.npart, dtype='d')
        else:
            self.depth = None

        # id of particles for i/o
        self.tag = np.arange(self.npart, dtype='int32')
        self.maxtag = self.tag[self.npart-1]

        # stranded particle buffer
        self.max_stranded = max_stranded if max_stranded is not None else self.npart
        self.stranded_count = 0
        self.stranded_positions = np.zeros((self.max_stranded, 3), dtype='d')

        # Initialize particle state vector
        self._init_state_vector(position, velocity, fluid_velocity_seen)

        # localization in eulerian mesh
        self.tri = -1*np.ones((self.npart), dtype='int32')
        self.lowerlayer = np.zeros((self.npart), dtype='int32')
        self.last_tri = -1*np.ones((self.npart), dtype='int32')
        self.last_position = np.empty((self.npart, self.dim), dtype='d')
        self.last_velocity = np.empty((self.npart, self.dim), dtype='d')
        self.last_position[:, :] = self.position[:, :]
        self.last_velocity[:, :] = self.velocity[:, :]

        # Model parameters
        self.drag_coef_model = drag_coefficient_model
        self.drag_coef = drag_coefficient
        self.added_mass_force = added_mass_force
        self.added_mass_coef = added_mass_coef
        
        # LSM1 parameters
        self.buoyancy_velocity_model = buoyancy_velocity_model
        self.buoyancy_velocity_value = buoyancy_velocity

        # Parameter for behavior modules
        self.additional_velocity = False
        self.additional_force = False

        # Constant component if LSM2/3 model coefficients: A0, A1, A2, A3
        # Are initialized in _initialize_model_coefficients() called by solver
        self.a0c = 0. # buoyancy
        self.a1c = 0. # drag
        self.a2c = 0. # momentum
        self.a3c = 0. # additional force

    def _init_state_vector(self, position=None, velocity=None, fluid_velocity_seen=None):
        """ 
        Initialize state vector (at class initialization)

        Fills the inactive particles with zeros and sets the active particles based on the provided arrays.

        Parameters
        ----------
        position : ndarray, shape (npart, dim), dtype float64, optional
            Initial particle positions
        velocity : ndarray, shape (npart, dim), dtype float64, optional
            Initial particle velocities
        fluid_velocity_seen : ndarray, shape (npart, dim), dtype float64, optional
            Initial fluid velocities seen by particles
            
        Raises
        ------
        ValueError
            If dimensions don't match or arrays are incompatible
        """
        cdef:
            int[:] tmpi1, tmpi0
            double[:,:] tmpd0

        # init particle shape
        pshp = np.shape(position)

        if position is not None:
            # check size 
            if pshp[0] > self.npart:
                raise ValueError('size of position > initial_pool_size')

            # temporary array used for initialization
            tmpi0 = np.zeros((pshp[0]), dtype='int32')
            tmpi1 = np.ones((self.npart-pshp[0]), dtype='int32')
            tmpd0 = np.zeros((self.npart-pshp[0], self.dim), dtype='d')
        
            # check dimension
            if pshp[1] != self.dim:
                raise ValueError('dim of position != LagrangianParticleSet.dim')
            # init
            #self.position[0:pshp[0], :] = position[:, :]
            self.position = np.concatenate((position, tmpd0), axis=0)
            self.inactive = np.concatenate((tmpi0, tmpi1), axis=0)
            self.npart_active = pshp[0]

        if velocity is not None:
            # check dimension
            pshp = np.shape(velocity)
            if position is None:
                raise ValueError('position must be != None')
            if pshp[0] != self.npart_active:
                raise ValueError('size of velocity != size of position')
            if pshp[1] != self.dim:
                raise ValueError('dim of velocity != LagrangianParticleSet.dim')
            # init
            #self.velocity[0:pshp[0], :] = velocity[:, :]
            self.velocity = np.concatenate((velocity, tmpd0), axis=0)

        if fluid_velocity_seen is not None:
            # check dimension
            pshp = np.shape(fluid_velocity_seen)
            if position is None:
                raise ValueError('position must be != None')
            if velocity is None:
                raise ValueError('velocity must be != None')
            if pshp[0] != self.npart_active:
                raise ValueError('size of fluid_velocity_seen != size of velocity')
            if pshp[1] != self.dim:
                raise ValueError('dim of fluid_velocity_seen != LagrangianParticleSet.dim')
            # init
            #self.fluid_velocity_seen[0:pshp[0], :] = fluid_velocity_seen[:,:]
            self.fluid_velocity_seen = np.concatenate((fluid_velocity_seen, tmpd0), axis=0)

    cdef void _initialize_particle_distribution(LagrangianParticleSet self):
        """
        Initialize particle diameter and volume based on the specified distribution.
        """
        cdef int i
        for i in range(self.npart):
            self.diameter[i] = draw_particle_diameter(
                self.particle_diameter,
                self.diameter_distribution,
                self.diameter_distribution_param)
            self.volume[i] = np.pi*(self.diameter[i]**3)/6.

    cdef void _initialize_model_coefficients(LagrangianParticleSet self):
        """ 
        Initialize constant component if LSM2/3 model coefficients: A0, A1, A2, A3 
        (called by solver when using LSM2 or LSM3 model)

        Constant components are defined as follows:
        A0 = A0c
        A1 = A1c * Cd * Ur * Sp/Vp
        A2 = A2c
        A3 = A3c / Vp
        """
        cdef:
            double rhop = self.density
            double Ca = self.added_mass_coef
            double aux

        if self.added_mass_force==0:
            self.a0c = (rhop - self.rho0)/rhop
            self.a1c = 0.5*(self.rho0/rhop)
            self.a2c = 0.
            self.a3c = 1./rhop
        else:
            aux = 1./(rhop + Ca*self.rho0)
            self.a0c = (rhop - self.rho0)*aux
            self.a1c = 0.5*self.rho0*aux
            self.a2c = self.rho0*(1. + Ca)*aux
            self.a3c = aux

    def show_status(LagrangianParticleSet self, int i):
        """ 
        Print particle status information
        
        Parameters
        ----------
        i : int
            Index of particle to display
        """
        print("~~~~~~~~~~~~~~~~~~~~~")
        print("Particle n°", i)
        print("position            =", np.asarray(self.position[i, :]))
        print("velocity            =", np.asarray(self.velocity[i, :]))
        print("fluid_velocity_seen =", np.asarray(self.fluid_velocity_seen[i, :]))
        print("tag                 =", self.tag[i])
        print("inactive            =", self.inactive[i])
        print("tri                 =", self.tri[i])
        print("lowerlayer          =", self.lowerlayer[i])
        print("last_tri            =", self.last_tri[i])
        print("last_position       =", np.asarray(self.last_position[i,:]))

    def get_state(self):
        """ 
        Returns particles state vector of active particles
        
        Returns
        -------
        position : ndarray, shape (npart_active, dim), dtype float64
            Positions of active particles
        velocity : ndarray, shape (npart_active, dim), dtype float64
            Velocities of active particles
        fluid_velocity_seen : ndarray, shape (npart_active, dim), dtype float64
            Fluid velocities seen by active particles
            
        Notes
        -----
        For post-processing only. Sorts active particles before returning.
        """
        cdef int na
        
        na = self.npart_active

        self._sort_active_particles()
        # return state vector
        return np.asarray(self.position[0:na]),\
               np.asarray(self.velocity[0:na]),\
               np.asarray(self.fluid_velocity_seen[0:na])

    def get_particle_properties(self):
        """
        Returns particles properties of active particles
        
        Returns
        -------
        diameter : ndarray, shape (npart_active,), dtype float64
            Diameters of active particles
        volume : ndarray, shape (npart_active,), dtype float64
            Volumes of active particles
            
        Notes
        -----
        For post-processing only. Sorts active particles before returning.
        """
        cdef int na
        
        na = self.npart_active

        self._sort_active_particles()
        # return particle properties
        return np.asarray(self.diameter[0:na]),\
               np.asarray(self.volume[0:na])

    cdef void _parallel_split(LagrangianParticleSet self, int offset, int count, int pool_size):
        """
        Split particle set for parallel computation.

        Distributes particles among ranks, keeping only particles [offset:offset+count] from active particles.
        The pool is filled using padding with inactive particles to reach the specified pool_size.

        Keeps only particles [offset:offset+count] from active particles
        and resizes the pool to pool_size. Subclasses should override and
        call the parent method to handle their extra per-particle arrays.

        Parameters
        ----------
        offset : int
            Starting index in the active particle arrays
        count : int
            Number of active particles to keep for this rank
        pool_size : int
            Total pool size allocated for this rank
        """
        cdef int new_npart = max(pool_size, count)
        cdef int pad = new_npart - count

        # Extract slices of active particles for this rank
        dp_slice = np.asarray(self.diameter)[offset:offset+count].copy()
        vol_slice = np.asarray(self.volume)[offset:offset+count].copy()
        pos_slice = np.asarray(self.position)[offset:offset+count, :].copy()
        vel_slice = np.asarray(self.velocity)[offset:offset+count, :].copy()
        fvs_slice = np.asarray(self.fluid_velocity_seen)[offset:offset+count, :].copy()
        if self.compute_depth:
            depth_slice = np.asarray(self.depth)[offset:offset+count].copy()

        # Padding arrays (inactive particles) to fill the pool
        pad_2d = np.zeros((pad, self.dim), dtype='d')
        pad_1d = np.zeros(pad, dtype='d')
        pad_1d_int0 = np.zeros(pad, dtype='int32')
        pad_1d_int1 = np.ones(pad, dtype='int32')
        pad_1d_intm1 = -1*np.ones(pad, dtype='int32')

        # Rebuild dimensions
        self.npart = new_npart
        self.npart_active = count

        # Rebuild diameter and volume
        self.diameter = np.concatenate((dp_slice, pad_1d), axis=0)
        self.volume = np.concatenate((vol_slice, pad_1d), axis=0)

        # Rebuild state vector
        self.position = np.concatenate((pos_slice, pad_2d), axis=0)
        self.velocity = np.concatenate((vel_slice, pad_2d), axis=0)
        self.fluid_velocity_seen = np.concatenate((fvs_slice, pad_2d), axis=0)

        # Rebuild depth
        if self.compute_depth:
            self.depth = np.concatenate((depth_slice, pad_1d), axis=0)

        # Rebuild inactive flags
        inactive_active = np.zeros(count, dtype='int32')
        self.inactive = np.concatenate((inactive_active, pad_1d_int1), axis=0)

        # Rebuild tags (preserve original tags for this rank's particles)
        tag_slice = np.asarray(self.tag)[offset:offset+count].copy()
        pad_tags = np.arange(count, new_npart, dtype='int32')
        self.tag = np.concatenate((tag_slice, pad_tags), axis=0)
        self.maxtag = max(new_npart - 1, int(np.max(tag_slice)) if count > 0 else 0)

        # Rebuild localization arrays
        self.tri = -1 * np.ones(new_npart, dtype='int32')
        self.lowerlayer = np.zeros(new_npart, dtype='int32')
        self.last_tri = -1 * np.ones(new_npart, dtype='int32')
        self.last_position = np.concatenate((pos_slice.copy(), pad_2d), axis=0)
        self.last_velocity = np.concatenate((vel_slice.copy(), pad_2d), axis=0)

    cdef void _increase_pool_size(LagrangianParticleSet self, int size):
        """ 
        Resize particle pool arrays in case of pool size change

        Sets new particles as inactive and fills them with default values. 
        Updates the pool size accordingly.
        
        Parameters
        ----------
        size : int
            Number of new particles to add
            
        Notes
        -----
        Increases all arrays (position, velocity, etc.) by max(min_pool_size_update, size)
        """
        cdef: 
            int old_npart, size_to_add
            int[:] tmpi0, tmpi1, tmpi1m
            double[:,:] tmpd0
        
        # Store old size
        old_npart = self.npart

        # to avoid excessive allocation of memory when adding particles
        size_to_add = max(self.min_pool_size_update, size)

        # init temporary tables
        tmpi0 = np.zeros((size_to_add), dtype='int32')
        tmpi1 = np.ones((size_to_add), dtype='int32')
        tmpi1m = -1*np.ones((size_to_add), dtype='int32')
        tmpd0 = np.zeros((size_to_add, self.dim), dtype='d')

        # dimension
        self.npart += size_to_add
        self.inactive = np.concatenate((self.inactive, tmpi1), axis=0)

        # diameter and volume
        self.diameter = np.concatenate((self.diameter, tmpd0[:, 0]), axis=0)
        self.volume = np.concatenate((self.volume, tmpd0[:, 1]), axis=0)

        # state vector
        self.position = np.concatenate((self.position, tmpd0), axis=0)
        self.velocity = np.concatenate((self.velocity, tmpd0), axis=0)
        self.fluid_velocity_seen = np.concatenate((self.fluid_velocity_seen, tmpd0), axis=0)

        # particle depth
        if self.compute_depth:
            self.depth = np.concatenate((self.depth, tmpd0[:, 0]), axis=0)

        # tag
        newtags = np.arange(self.maxtag+1, self.maxtag+1+size_to_add, dtype='int32')
        self.tag = np.concatenate((self.tag, newtags), axis=0)
        self.maxtag += size_to_add

        # localization in eulerian mesh
        self.tri = np.concatenate((self.tri, tmpi1m), axis=0)
        self.lowerlayer = np.concatenate((self.lowerlayer, tmpi0), axis=0)
        self.last_tri = np.concatenate((self.last_tri, tmpi0), axis=0)
        self.last_position = np.concatenate((self.last_position, tmpd0), axis=0)
        self.last_velocity = np.concatenate((self.last_velocity, tmpd0), axis=0)

    cpdef void _sort_active_particles(LagrangianParticleSet self):
        """ 
        Sort active particles 
        
        Sorting is done based on the inactive array,
        so that active particles are at the beginning of the arrays.
        """

        sorting_idx = np.argsort(self.inactive)

        # dimension
        self.inactive = np.asarray(self.inactive)[sorting_idx]

        # diameter and volume
        self.diameter = np.asarray(self.diameter)[sorting_idx]
        self.volume = np.asarray(self.volume)[sorting_idx]

        # state vector
        self.position = np.asarray(self.position)[sorting_idx, :]
        self.velocity = np.asarray(self.velocity)[sorting_idx, :]
        self.fluid_velocity_seen = np.asarray(self.fluid_velocity_seen)[sorting_idx, :]

        # particle depth
        if self.compute_depth:
            self.depth = np.asarray(self.depth)[sorting_idx]

        # tag
        self.tag = np.asarray(self.tag)[sorting_idx]

        # localization in eulerian mesh
        self.tri = np.asarray(self.tri)[sorting_idx]
        self.lowerlayer = np.asarray(self.lowerlayer)[sorting_idx]
        self.last_tri = np.asarray(self.last_tri)[sorting_idx]
        self.last_position = np.asarray(self.last_position)[sorting_idx, :]
        self.last_velocity = np.asarray(self.last_velocity)[sorting_idx, :]

    cpdef void add(LagrangianParticleSet self, double[:,:] position):
        """ 
        Add particles to the particle set:
        - Set inactive particles to active.
        - If no inactive particles are available, increase the pool size.

        Parameters
        ----------
        position : ndarray, shape (nadd, dim), dtype float64
            Positions of particles to add

        Notes
        -----
        State initialization that requires fset are done in solver, therefore:
        - Localization in the eulerian mesh is set to -2 (new particle) 
          to trigger initialization in the solver.
        - Only initializes diameter, volume and position (not localization). 
        - Particle velocity and fluid velocity seen are set to zero.
        Automatically increases pool size if necessary.
        """
        cdef:
            int i, j
            int nactive = self.npart_active
            int nadd = np.shape(position)[0]
            int pool_size = self.npart - self.npart_active
            bint method = 1 # method of add (0: with sorting /1: without sorting)

        if nadd > 0:
            # increase particle pool size if necessary
            if nadd > pool_size:
                self._increase_pool_size(nadd-pool_size)
        
            # method 0 : with active particle sorting
            # =======================================
            if method == 0:
                # sort active particles
                self._sort_active_particles()

                # add particles
                # ~~~~~~~~~~~~~
                # set active
                self.inactive[nactive:nactive+nadd] = 0

                # define properties
                for i in range(nadd):
                    self.diameter[nactive+i] = draw_particle_diameter(
                        self.particle_diameter,
                        self.diameter_distribution,
                        self.diameter_distribution_param)
                    self.volume[nactive+i] = np.pi*(self.diameter[nactive+i]**3)/6.

                # define position
                self.position[nactive:nactive+nadd, :] = position[:,:]

                # set localization to -2 for triggering initialization in the solver
                self.tri[nactive:nactive+nadd] = -2*np.ones((nadd), dtype='int32')
                self.npart_active += nadd

            # method 1 : without active particle sorting
            # ==========================================
            if method == 1:
                # counter for added particles
                j = 0

                # add particles
                # ~~~~~~~~~~~~~
                for i in range(self.npart):

                    if self.inactive[i] == 1:
                        # set active
                        self.inactive[i] = 0

                        # define properties
                        self.diameter[i] = draw_particle_diameter(
                            self.particle_diameter,
                            self.diameter_distribution,
                            self.diameter_distribution_param)
                        self.volume[i] = np.pi*(self.diameter[i]**3)/6.

                        # define position
                        self.position[i, :] = position[j, :]

                        # set localization to -2 for triggering initialization in the solver
                        self.tri[i] = -2
                        self.npart_active += 1

                        # increment counter for added particles
                        j += 1

                    if j == nadd:
                        break

    cpdef void delete(LagrangianParticleSet self, int[:] particles_id):
        """ 
        Delete particles from the particle set
        
        Parameters
        ----------
        particles_id : ndarray, shape (ndel,), dtype int32
            Indices of particles to delete (must be active)
            
        Notes
        -----
        Erases particles from the active set and moves them to the pool of inactive particles
        """
        cdef:
            int i
            double[:] tmpd0 = np.zeros((self.dim), dtype='d')

        # deactivate particle i
        for i in particles_id:
            # only delete active particle
            if self.inactive[i] == 0:
                self.inactive[i] = 1
                self.diameter[i] = 0.
                self.volume[i] = 0.
                self.position[i, :] = tmpd0[:]
                self.velocity[i, :] = tmpd0[:]
                self.fluid_velocity_seen[i, :] = tmpd0[:]
                if self.compute_depth:
                    self.depth[i] = 0.
                self.tri[i] = -1
                self.lowerlayer[i] = 0
                self.last_tri[i] = -1
                self.last_position[i, :] = tmpd0[:]
                self.last_velocity[i, :] = tmpd0[:]
                self.npart_active -= 1
                # reset tag for next usage
                self.tag[i] = self.maxtag+1
                self.maxtag += 1

        # TODO: if pool size too big, deallocate memory
        #       self.position = np.delete(self.position, particle_id, 0)

    cpdef void delete_single(LagrangianParticleSet self, int i):
        """ 
        Delete single particle
        
        Parameters
        ----------
        i : int
            Index of particle to delete (must be active)
            
        Notes
        -----
        Erases particle from the active set and moves it to the pool of inactive particles
        """
        cdef:
            double[:] tmpd0 = np.zeros((self.dim), dtype='d')

        # only delete active particle
        if self.inactive[i] == 0:
            self.inactive[i] = 1
            self.diameter[i] = 0.
            self.volume[i] = 0.
            self.position[i, :] = tmpd0[:]
            self.velocity[i, :] = tmpd0[:]
            self.fluid_velocity_seen[i, :] = tmpd0[:]
            if self.compute_depth:
                self.depth[i] = 0.
            self.tri[i] = -1
            self.lowerlayer[i] = 0
            self.last_tri[i] = -1
            self.last_position[i, :] = tmpd0[:]
            self.last_velocity[i, :] = tmpd0[:]
            self.npart_active -= 1
            # reset tag for next usage
            self.tag[i] = self.maxtag+1
            self.maxtag += 1

    cpdef double drag_coefficient(LagrangianParticleSet self, double U, double dp):
        """ 
        Computes the drag coefficient for a given relative velocity and diameter.

        Parameters
        ----------
        U : double
            Relative velocity magnitude
        dp : double
            Particle diameter

        Returns
        -------
        Cd : double
            Drag coefficient
        """
        cdef:
            double Re = fmax(EPSILON, U*dp/NU0)
            double phi1, phi2, phi3, phi4
            double Cd

        # Constant drag
        if self.drag_coef_model==0:
            Cd = self.drag_coef

        # Schiller et Nauman (1935)
        elif self.drag_coef_model==1:
            if Re <= 1000.:
                Cd = (24./Re)*(1. + 0.15*Re**0.687)
            else:
                Cd = 0.44

        # Almedeij (2008)
        elif self.drag_coef_model==2:
            phi1 = (24./Re)**10. + (21.*Re**-0.67)**10. + (4.*Re**-0.33)**10. + 0.4**10
            phi2 = 1./((0.148*Re**0.11)**-10. + (0.5)**-10.)
            phi3 = (1.57*(Re**-1.625)*1.e8 )**10.
            phi4 = 1./((6e-17*Re**2.63)**-10.  + 0.2**-10.)
            Cd = (1./( (phi1 + phi2)**-1. + phi3**-1.) + phi4)**(1/10.)
        else:
            Cd = 0.44

        return Cd

    cpdef double buoyancy_velocity(LagrangianParticleSet self, double dp,
            int model_hr, double rho0):
        """ 
        Computes the buoyancy velocity of a particle.

        Note: The buoyancy velocity is the terminal settling velocity of a particle in a fluid, 
        considering the effects of buoyancy and drag : it is natively included in LSM2 and 3.
        This function is therefore only called for LSM1.

        Parameters
        ----------
        model_hr : int, model option for high Reynolds number (dp>1.e-3)
            0 : Analytical solution with constant drag coefficient 
            1 : Van Rijn (1985)
        rho0 : float, water density

        Returns
        -------
        wb : float, buoyancy velocity
        """
        cdef:
            double rhop = self.density
            double Cd = self.drag_coef
            double s = rhop/rho0
            double wb

        # constant given by user
        # ~~~~~~~~~~~~~~~~~~~~~~
        if self.buoyancy_velocity_model==0:
            wb = self.buoyancy_velocity_value

        # analytical solution
        # ~~~~~~~~~~~~~~~~~~~
        else:
            # Low Reynolds : Stokes' law
            if dp < 1.e-4:
                wb = fabs(s-1.)*(GRAV*dp**2)/(18.*NU0)

            # High Reynolds : 
            elif dp > 1.e-3:

                # Analytic turbulent range with constant drag coefficient:
                if model_hr==0:
                    wb = sqrt(fabs(s-1.)*(4.*dp*GRAV)/(3.*Cd))

                # Van Rijn (1985):        
                else:
                    wb = 1.1*sqrt((s-1.)*GRAV*dp)

            # Intermediate : Ruby and Zanke (1977)
            else:
                wb = (10.*NU0/dp)*(sqrt(1.+ fabs(s-1.)*GRAV*dp**3/(100.*NU0**2)) - 1.)

            if s>1.:
                wb *= -1.

        return wb

    cdef void add_velocity_i(LagrangianParticleSet self, 
            EulerianFieldSet fset, int i, double dt):
        pass

    cdef void add_force_i(LagrangianParticleSet self, 
            EulerianFieldSet fset, int i, double dt):
        pass