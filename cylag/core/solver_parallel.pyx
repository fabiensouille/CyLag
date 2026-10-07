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
import time
import numpy as np
try:
    #import mpi4py
    #mpi4py.rc.initialize = False
    from mpi4py import MPI
except ImportError:
    raise ImportError("mpi4py is required for SolverParallel")
from ..core.constants cimport *
from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..lsm.init_lsm2_fluctuations cimport update_lsm2_fluctuations
from ..lsm.init_lsm3_fluctuations cimport update_lsm3_fluctuations
from ..lsm.update_state_test cimport update_state_test
from ..lsm.update_state_lsm1 cimport update_state_lsm1
from ..lsm.update_state_lsm2 cimport update_state_lsm2
from ..lsm.update_state_lsm3 cimport update_state_lsm3
from ..geom.xylocalizer cimport xy_localize
from ..geom.zlocalizer cimport c_compute_lower_index, c_compute_lower_layer_index
from ..io.write_particles import prepare_output, merge_results, merge_stranded_results, write_stranded_particles_txt_2d
from ..io.write_particles cimport write_particles
from ..io.write_statistics import prepare_secstats_output, merge_secstats_results
from ..io.write_statistics cimport write_secstats, write_secstats_buffer
from ..boundary_conditions.compute_bc cimport compute_boundary_condition
from ..extra.control_shapes cimport ControlSection
from ..lsm.random_utils cimport random_gaussian_pair, random_gaussian_ziggurat, pcg32_advance

cdef class SolverParallel:
    """
    Hybrid Eulerian-Lagrangian solver with MPI parallelization

    Parameters
    ----------
    field_set : cylag.EulerianFieldSet
        Eulerian field set containing velocity and optional fields
    particle_set : cylag.LagrangianParticleSet
        Lagrangian particle set (will be split among processes)
    parameters : dict, optional
        Model parameters (default: {})
    sources : list of cylag.ParticleSource, optional
        List of particle sources (default: None)
    control_sections : list, optional
        List of control sections for statistics (default: None)
    comm : MPI.Comm, optional
        MPI communicator (default: MPI.COMM_WORLD)
        
    Attributes
    ----------
    particle_set : cylag.LagrangianParticleSet
        Lagrangian particle set (local to this process)
    field_set : cylag.EulerianFieldSet
        Eulerian field set
    parameters : cylag.Parameters
        Model parameters
    sources : list of cylag.ParticleSource
        List of particle sources
    control_sections : list
        List of control sections
    simutime : cylag.SimuTime
        Simulation time manager
    comm : MPI.Comm
        MPI communicator
    rank : int
        Process rank in MPI communicator
    size : int
        Total number of processes in MPI communicator
    """
    def __init__(self,
                 field_set,
                 particle_set,
                 parameters={},
                 sources=None,
                 control_sections=None,
                 comm=None):

        cdef int i, ntimes

        if comm is None:
            #if not MPI.Is_initialized():
            #    MPI.Init()
            #    self._mpi_owner = 1
            #else:
            #    self._mpi_owner = 0
            self.comm = MPI.COMM_WORLD
        else:
            #self._mpi_owner = 0
            self.comm = comm

        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # All ranks start from the same PCG32 state: jump each rank to its own sub-stream
        if self.size > 1:
            pcg32_advance(<unsigned long long>self.rank << 48)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Parallel load distribution
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check size 
        if self.size >= MAX_NUMBER_OF_PROC:
            raise ValueError("Number of processes >= MAX_NUMBER_OF_PROC, check size")
        elif self.size == 1:
            print("WARNING, parallel solver with nproc = 1, consider using standard solver")

        # Parallel load distribution - split particles
        if self.rank == 0:
            # total pool split
            nperproc0 = particle_set.npart // self.size
            lastproc0 = particle_set.npart % self.size
            npart0 = [max(1, nperproc0) for i in range(self.size)]
            npart0[self.size-1] += lastproc0

            # active particles split
            if particle_set.npart_active > 0:
                nperproc = particle_set.npart_active // self.size
                lastproc = particle_set.npart_active % self.size
                npartp = [nperproc for i in range(self.size)]
                npartp[self.size-1] += lastproc
            else:
                npartp = [0 for i in range(self.size)]
        else:
            npart0 = None
            npartp = None

        npart0 = self.comm.bcast(npart0, root=0) # min pool size
        npartp = self.comm.bcast(npartp, root=0) # active particles count

        # ~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize main classes
        # ~~~~~~~~~~~~~~~~~~~~~~~
        self.parameters = Parameters(parameters)
        if self.rank == 0: 
            self.parameters.check_values()
        else:
            self.parameters.listing = 0

        self.simutime = SimuTime(
            self.parameters.listing,
            self.parameters.initial_time,
            self.parameters.final_time,
            self.parameters.time_step)

        self.sources = sources

        self.control_sections = control_sections
        if self.control_sections is not None:
            self.parameters.control_sections = True
            self.parameters.ncsec = len(self.control_sections)
        else:
            self.parameters.control_sections = False
            self.parameters.ncsec = 0

        # initialize particle_set (as many as size)
        self.particle_set = particle_set
        if self.size > 1:
            # split active particles between procs using _parallel_split
            self.particle_set._parallel_split(
                self.rank * npartp[0], npartp[self.rank], npart0[self.rank])

        # field_set
        self.field_set = field_set

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Print initialization information
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.parameters.listing:
            print('================================================================================')
            print('|                                INITIALIZATION                                |')
            print('================================================================================')
            self.print_init_str()

        # ~~~~~~~~~~~~~
        # Sanity checks
        # ~~~~~~~~~~~~~
        ntimes = self.field_set.ntimes

        # if 2D hydro with 3D particle set:
        # check that free surface and bottom fields are present
        if self.rank == 0:
            if self.field_set.dim != self.particle_set.dim:
                if self.field_set.dim == 2 and self.particle_set.dim == 3:
                    if self.field_set.fid_zs == -1 or self.field_set.fid_zb == -1:
                        raise ValueError(
                            "2D EulerianFieldSet with 3D LagrangianParticleSet : "\
                            "EulerianFieldSet should contain free surface and bottom fields")
                else:
                    raise ValueError(
                        "EulerianFieldSet dimension ({}) does not match "\
                        "LagrangianParticleSet set dimension ({})".format(
                            self.field_set.dim, self.particle_set.dim))

        # don't allow time extrapolation
        if self.rank == 0:
            if self.parameters.frozen_eulerian_fields!=1 and \
               self.parameters.final_time>self.field_set.times[ntimes-1]:
                raise ValueError("final_time > field_set maximum time")

        # set frozen hydro
        if self.parameters.frozen_eulerian_fields==1:
            self.field_set.frozen_hydro = True
        else:
            self.field_set.frozen_hydro = False

        if self.rank == 0:
            if ntimes==1 and self.field_set.frozen_hydro == False:
                raise ValueError("EulerianFieldSet has only one time step,\
                                use frozen_eulerian_fields=True")

        # set initial record from initial time
        if self.field_set.frozen_hydro == False:
            self.field_set.record = c_compute_lower_index(\
                self.field_set.times, self.parameters.initial_time, 0)
        self.field_set._interp_done = False
        self.field_set._prev_valid = False

        # check behavior model compatibility with lsm
        if self.rank==0:
            # check additional velocity/force compatibility with lsm
            if self.particle_set.additional_velocity and\
              (self.parameters.model==2 or self.parameters.model==3):
                raise ValueError("Additional velocity only compatible with LSM1")

            if self.particle_set.additional_force and\
               self.parameters.model==1:
                raise ValueError("Additional force not compatible with LSM1")

            # check diffusion fields
            if self.parameters.diffusion_model==2 or \
               self.parameters.diffusion_model==3:
                if self.field_set.fid_turbeng == -1 or \
                   self.field_set.fid_dissip == -1:
                    raise ValueError(
                        "LSM: diffusion_model={} requires 'TURBULENT ENERG.' and "
                        "'DISSIPATION' fields in EulerianFieldSet "
                        "(fid_turbeng={}, fid_dissip={})".format(
                            self.parameters.diffusion_model, \
                            self.field_set.fid_turbeng, \
                            self.field_set.fid_dissip))

        # ~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize particle set
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # propagate water density from parameters to particle set
        self.particle_set.rho0 = self.parameters.water_density

        # initialize lsm2,3 forces coefficients
        if self.parameters.model==2 or self.parameters.model==3:
            self.particle_set._initialize_model_coefficients()

        # fields at initial time are needed to initialize particle states
        if self.field_set.frozen_hydro == False:
            self.field_set._time_interpolation(self.parameters.initial_time)

        # initialize particle state
        # loop on active particles
        for i in range(self.particle_set.npart):
            if self.particle_set.inactive[i]==0:
                self.initialize_particle_state(i)

        # ~~~~~~~~~~~~~~~
        # Prepare outputs
        # ~~~~~~~~~~~~~~~
        # prepare output files and repository
        if self.parameters.output_file==1:
            self.parameters.output_file_name += "#{}".format(self.rank)
            prepare_output(
                self.particle_set,
                self.parameters.output_rep,
                self.parameters.output_file_format,
                self.parameters.output_file_name,
                self.parameters.model)

        if self.parameters.stranding_output:
            self.parameters.stranding_output_file_name += "#{}".format(self.rank)

        if self.parameters.bnd_statistics==1 or \
           self.parameters.control_sections==1:
            self.parameters.bnd_statistics_file_name += "#{}".format(self.rank)
            self.parameters.control_sections_file_name += "#{}".format(self.rank)
            prepare_secstats_output(
                self.parameters.bnd_statistics,
                self.field_set.triangular_mesh.nopen,
                self.parameters.output_rep,
                self.parameters.bnd_statistics_file_name,
                self.parameters.control_sections,
                self.parameters.ncsec,
                self.parameters.control_sections_file_name)

        # write particle_set initial positions
        if self.parameters.output_file==1:
            write_particles(self.particle_set, self.simutime.time, 0, self.size, self.rank, self.parameters)

        # initialize statistics buffers
        cdef int nt_buf = self.simutime.nt
        cdef int nopen_buf = self.field_set.triangular_mesh.nopen
        cdef int ncsec_buf = self.parameters.ncsec
        self._bnd_stats_times = np.zeros(nt_buf, dtype='d')
        self._bnd_stats_data = np.zeros((nt_buf, max(1, nopen_buf)), dtype='int32')
        self._bnd_stats_count = 0
        self._csec_stats_times = np.zeros(nt_buf, dtype='d')
        self._csec_stats_data = np.zeros((nt_buf, max(1, ncsec_buf)), dtype='int32')
        self._csec_stats_count = 0

    cpdef void initialize_particle_state(SolverParallel self, int i):
        """ 
        Initialize particle state
        
        Performs horizontal and vertical localization of the particle,
        and optionally initializes particle velocity from mean fluid velocity.
        
        Parameters
        ----------
        i : int
            Particle index
        """
        cdef:
            double ux, uy, uz
            double turbeng, epsilon, tml, sigma_s, sigma_p
            double unorm, Cd, a1, taup, b2tau0, b2tau1, b2tau2
            double xix, xiy, xiz
            double zs, zb
            double[:] xp = self.particle_set.position[i,:]
            double[:] up = self.particle_set.velocity[i,:]
            double[:] us = self.particle_set.fluid_velocity_seen[i,:]

        # horizontal localization
        if self.field_set.trifinder is not None:
            self.particle_set.tri[i] = self.field_set.trifinder(\
                xp[0], xp[1])
        else:
            self.particle_set.tri[i], _ = xy_localize(\
                self.field_set.triangular_mesh, -1, 0., 0.,
                xp[0], xp[1], 0, 0)

        if self.particle_set.tri[i] == -1:
            self.particle_set.delete_single(i)
            print("WARNING: Particle {} is ouside triangulation".format(i))

        # vertical localization (skipped if particle was just deleted)
        if self.particle_set.dim==3 and self.particle_set.inactive[i]==0:

            # get zs and zb from field_set
            zs = self.field_set._interpolate_field_2d(
                self.field_set.fid_zs,
                self.particle_set.tri[i], 
                self.field_set.nlayers-1,
                xp[0], xp[1])

            if self.field_set.dim==2:
                # pseudo-3D: get zb from 2D field_set
                zb = self.field_set._interpolate_field_2d(
                    self.field_set.fid_zb,
                    self.particle_set.tri[i], 
                    0, xp[0], xp[1])
            else:
                # 3D: get zb from zs at layer 0
                zb = self.field_set._interpolate_field_2d(
                    self.field_set.fid_zs,
                    self.particle_set.tri[i], 
                    0, xp[0], xp[1])

            self.particle_set.lowerlayer[i] = c_compute_lower_layer_index(\
                self.field_set.nlayers, zs, zb, xp[2])

            # resets particle state if particle is outside domain
            if xp[2] < zb:
                xp[2] = zb + EPSILON
                print("WARNING: Particle {} is below bottom zb={}, "\
                    "set zp=zb".format(i, zb))
                
            if xp[2] > zs:
                xp[2] = zs - EPSILON
                print("WARNING: Particle {} is above free surface zs={}, "\
                    "set zp=zs".format(i, zs))

        # only initialize particle state if particle is active
        if self.particle_set.inactive[i]==0:

            # initialize particle depth
            if self.particle_set.compute_depth:
                zs = self.field_set._interpolate_field_2d(
                    self.field_set.fid_zs,
                    self.particle_set.tri[i], 
                    self.field_set.nlayers-1,
                    xp[0], xp[1])
                self.particle_set.depth[i] = zs - xp[2]

            # initialize particle velocity from mean fluid velocity
            if self.parameters.particle_velocity_init==1:
                if self.field_set.dim == 2:
                    ux, uy = self.field_set._interpolate_velocity_2d(\
                        self.particle_set.tri[i], xp[0], xp[1])
                    up[0] = ux
                    up[1] = uy
                else:
                    ux, uy, uz = self.field_set._interpolate_velocity_3d(\
                        self.particle_set.tri[i], 
                        self.particle_set.lowerlayer[i], 0,
                        xp[0], xp[1], xp[2])
                    up[0] = ux
                    up[1] = uy
                    up[2] = uz

                # Add up fluctuation for LSM2 (diffusion-on-velocity variant)
                if self.parameters.model == 2 and \
                self.parameters.diffusion_lsm2_option >= 2 and \
                self.parameters.diffusion_model != 0:
                    # relative velocity ~ 0 since Up = mean flow at this point
                    fluct1, fluct2, fluct3 = update_lsm2_fluctuations(\
                        self.field_set,
                        self.particle_set,
                        self.parameters,
                        i)
                    up[0] += fluct1
                    up[1] += fluct2
                    if self.particle_set.dim == 3:
                        up[2] += fluct3

                # initialize particle velocity seen
                if self.parameters.particle_velocity_init==1:
                    if self.particle_set.dim == 2:
                        us[0] = up[0]
                        us[1] = up[1]
                    else:
                        us[0] = up[0]
                        us[1] = up[1]
                        us[2] = up[2]

                # Add us fluctuation for LSM3 
                if self.parameters.model == 3:
                    fluct1, fluct2, fluct3 = update_lsm3_fluctuations(\
                        self.field_set,
                        self.particle_set,
                        self.parameters,
                        i)
                    us[0] += fluct1
                    us[1] += fluct2
                    if self.particle_set.dim == 3:
                        us[2] += fluct3

    cpdef void solve(SolverParallel self):
        """
        Main solver: solves lagrangian particle tracking problem in parallel
        
        Iterates over all time steps from initial to final time,
        calling forward() at each iteration. After completion,
        synchronizes processes and merges results.
        """
        cdef:
            int l
            double start, end

        if self.parameters.listing:
            print('================================================================================')
            print('|                                 TIME LOOP                                    |')
            print('================================================================================')

        start = time.time()
        for l in range(self.simutime.nt):
            if not self.simutime.is_finished:
                self.forward(self.simutime.time_step)

        end = time.time()
        if self.parameters.listing:
            print('================================================================================')
            print('|                                    END                                       |')
            print('================================================================================')
            print('Problem.solve() completed in {}s (wall time)'.format(end-start))

        # Flush buffered statistics to disk
        self.flush_statistics()

        self.comm.barrier()

        # merge result files 
        self.merge_results()
        
        # mpi finalize
        #if self._mpi_owner and not MPI.Is_finalized():
        #    MPI.Finalize()

    cpdef void forward(SolverParallel self, double time_step):
        """
        Main loop: advances simulation by one time step
        
        Performs time interpolation of fields, adds particles from sources
        (with parallel distribution), updates particle states, applies
        boundary conditions, computes statistics, and writes outputs.
        
        Parameters
        ----------
        time_step : double
            Time step duration
        """
        cdef:
            int i, isrc, icsec
            int dim = self.field_set.dim
            int ite = self.simutime.iteration
            int npart = self.particle_set.npart
            int npart_new
            int bnd_j = -1
            int nopen = self.field_set.triangular_mesh.nopen
            int[:] bnd_stats = np.zeros((nopen), dtype='int32')
            int[:] csec_stats = np.zeros((self.parameters.ncsec), dtype='int32')
            double time = self.simutime.time
            double[:,:] new_positions
            double zs

        # compute time interpolation of mean fields
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.field_set._time_interpolation(time)

        # Add particles from sources
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.sources is not None:
        
            # loop on sources
            isrc = 0
            for source in self.sources:
                isrc += 1
                if source.release_condition(self.simutime, self.field_set):
                    # define global new positions from source
                    new_positions = source.init_positions(
                        self.field_set, self.particle_set.dim)

                    # Parallel load distribution
                    if self.rank == 0:
                        npart_new = new_positions.shape[0]
                        nperproc = npart_new // self.size
                        lastproc = npart_new % self.size
                        npartp = [nperproc for i in range(self.size)]
                        npartp[self.size-1] += lastproc
                    else:
                        npartp = None
                    npartp = self.comm.bcast(npartp, root=0)

                    # Add particles
                    if self.size > 1:
                        self.particle_set.add(new_positions[\
                            self.rank*npartp[0]:\
                            self.rank*npartp[0]+npartp[self.rank], :])
                    else:
                        self.particle_set.add(new_positions)

                    if self.parameters.listing and self.rank == 0:
                        print(' ~~> Adding {} particles from source {}'\
                            .format(new_positions.shape[0], isrc))

        # Main loop on particles
        # ~~~~~~~~~~~~~~~~~~~~~~
        for i in range(npart):
            if self.particle_set.inactive[i] == 0:

                # set state of added particles
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # check if particle was added at previous time step (itri==-2)
                if self.particle_set.tri[i] == -2:
                    self.initialize_particle_state(i)

            if self.particle_set.inactive[i] == 0:

                # update state vector 
                # ~~~~~~~~~~~~~~~~~~~
                # TESTING KERNEL
                if self.parameters.model==0:
                    bnd_j = update_state_test(\
                        self.simutime,
                        self.field_set,
                        self.particle_set,
                        self.parameters, i)

                # LSM-1 MODEL
                elif self.parameters.model==1:
                    bnd_j = update_state_lsm1(\
                        self.simutime,
                        self.field_set,
                        self.particle_set,
                        self.parameters, i)

                # LSM-2 MODEL
                elif self.parameters.model==2:
                    bnd_j = update_state_lsm2(\
                        self.simutime,
                        self.field_set,
                        self.particle_set,
                        self.parameters, i)

                # LSM-3 MODEL
                else:
                    bnd_j = update_state_lsm3(\
                        self.simutime,
                        self.field_set,
                        self.particle_set,
                        self.parameters, i)

                # boundary conditions
                # ~~~~~~~~~~~~~~~~~~~
                if self.parameters.boundary_conditions==1:
                    compute_boundary_condition(\
                        self.simutime,
                        self.field_set,
                        self.particle_set,
                        self.parameters,
                        i, bnd_j,
                        bnd_stats)
                else:
                    # sanity check only
                    if self.particle_set.tri[i]==PART_LOC_O:
                        self.particle_set.delete_single(i)
                        
                    if self.particle_set.lowerlayer[i]==PART_LOC_B:
                        self.particle_set.delete_single(i)
                        
                    if self.particle_set.lowerlayer[i]==PART_LOC_A:
                        self.particle_set.delete_single(i)

                # compute particle depth if requested
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if self.particle_set.compute_depth and \
                    self.particle_set.inactive[i] == 0 and \
                    self.particle_set.dim == 3:

                    zs = self.field_set._interpolate_field_2d(
                        self.field_set.fid_zs,
                        self.particle_set.tri[i], self.field_set.nlayers-1,
                        self.particle_set.position[i, 0],
                        self.particle_set.position[i, 1])

                    self.particle_set.depth[i] = zs - self.particle_set.position[i, 2]

                # Control sections
                # ~~~~~~~~~~~~~~~~
                if self.control_sections is not None:
                    for icsec in range(self.parameters.ncsec):
                        if (<ControlSection>self.control_sections[icsec]).c_cross(
                                self.particle_set.last_position[i, 0],
                                self.particle_set.last_position[i, 1],
                                self.particle_set.position[i, 0],
                                self.particle_set.position[i, 1]):
                            csec_stats[icsec] += 1

        # Time increment
        # ~~~~~~~~~~~~~~
        self.simutime.increment()
        ite = self.simutime.iteration

        # Write output particle_set file
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.parameters.output_file:
            if ite%self.parameters.output_printout_period == 0:
                write_particles(
                    self.particle_set,
                    self.simutime.time,
                    self.simutime.iteration,
                    self.size, self.rank,
                    self.parameters)

        # Buffer boundary statistics
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.parameters.bnd_statistics:
            self._bnd_stats_times[self._bnd_stats_count] = time
            self._bnd_stats_data[self._bnd_stats_count, :nopen] = bnd_stats
            self._bnd_stats_count += 1
            
        # Buffer control sections statistics
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.control_sections is not None:
            self._csec_stats_times[self._csec_stats_count] = time
            self._csec_stats_data[self._csec_stats_count, :self.parameters.ncsec] = csec_stats
            self._csec_stats_count += 1
            
        # Print progress
        # ~~~~~~~~~~~~~~
        if self.rank == 0:
            if self.simutime.print_progress:
                if ite%self.parameters.listing_printout_period == 0:
                    self.simutime.progress()

    cpdef void merge_results(SolverParallel self):
        """
        Merge output files after parallel computation
        
        Combines particle output files, boundary statistics, and control
        section statistics from all processes into single output files.
        Only executed by rank 0 process.
        """
        if self.rank == 0:
            if self.parameters.output_file:
                merge_results(self.size,
                    self.parameters.output_rep, 
                    self.parameters.output_file_name,
                    self.parameters.output_file_format)
                
            if self.parameters.bnd_statistics:
                merge_secstats_results(self.size,
                    self.parameters.output_rep,
                    self.parameters.bnd_statistics_file_name, 
                    self.field_set.triangular_mesh.nopen)

            if self.parameters.control_sections:
                merge_secstats_results(self.size,
                    self.parameters.output_rep,
                    self.parameters.control_sections_file_name, 
                    self.parameters.ncsec)

            if self.parameters.stranding_output:
                merge_stranded_results(self.size,
                    self.parameters.output_rep,
                    self.parameters.stranding_output_file_name)

    cpdef void flush_statistics(SolverParallel self):
        """
        Flush buffered statistics to disk

        Writes all accumulated boundary and control section statistics
        to their respective output files. Each file is opened only once,
        avoiding the I/O overhead of per-time-step writes.

        This method is called automatically at the end of solve().
        It can also be called manually when using forward() directly.
        After flushing, the internal buffer counters are reset to zero.
        """
        cdef int nopen

        # Write boundary statistics buffer
        if self.parameters.bnd_statistics and self._bnd_stats_count > 0:
            nopen = self.field_set.triangular_mesh.nopen
            write_secstats_buffer(
                self._bnd_stats_times,
                self._bnd_stats_data,
                self._bnd_stats_count, nopen,
                self.parameters.output_rep,
                self.parameters.bnd_statistics_file_name)
            self._bnd_stats_count = 0

        # Write stranded particle output file
        if self.parameters.stranding_output and self.particle_set.stranded_count > 0:
            write_stranded_particles_txt_2d(
                self.size, self.rank,
                self.parameters.output_rep,
                self.parameters.stranding_output_file_name,
                self.particle_set.stranded_positions,
                self.particle_set.stranded_count)
            self.particle_set.stranded_count = 0

        # Write control sections statistics buffer
        if self.control_sections is not None and self._csec_stats_count > 0:
            write_secstats_buffer(
                self._csec_stats_times,
                self._csec_stats_data,
                self._csec_stats_count, self.parameters.ncsec,
                self.parameters.output_rep,
                self.parameters.control_sections_file_name)
            self._csec_stats_count = 0

    def print_init_str(self):
        """
        Prints a formatted string summarizing the initialization of the solver.
        """
        cdef str simudim, time_scheme_str, diffusion_str, vel_init_str

        if self.field_set.dim == 2 and self.particle_set.dim == 3:
            simudim = "Pseudo-3D"
        else:
            simudim = "{}D".format(self.field_set.dim)

        if self.parameters.time_scheme == 1:
            time_scheme_str = "Euler"
        elif self.parameters.time_scheme == 2:
            time_scheme_str = "Runge-Kutta 2"
        elif self.parameters.time_scheme == 3:
            time_scheme_str = "Runge-Kutta 3"
        elif self.parameters.time_scheme == 4:
            time_scheme_str = "Runge-Kutta 4"
        elif self.parameters.time_scheme == 5:
            time_scheme_str = "Direct Integrator"
        else:
            time_scheme_str = "Unknown"

        if self.parameters.diffusion_model == 0:
            diffusion_str = "No diffusion"
        elif self.parameters.diffusion_model == 1:
            diffusion_str = "Constant horiz. and vert. diffusivity"
        elif self.parameters.diffusion_model == 2:
            diffusion_str = "k-epsilon model 2D / constant vert. diffusivity"
        elif self.parameters.diffusion_model == 3:
            diffusion_str = "k-epsilon model 3D"
        else:
            diffusion_str = "Unknown"

        if self.parameters.particle_velocity_init == 0:
            vel_init_str = "Initialized from given values"
        elif self.parameters.particle_velocity_init == 1:
            vel_init_str = "Initialized from field set"
        else:
            vel_init_str = "Unknown"

        if self.parameters.boundary_conditions == 0:
            bc_type_str = "None"
        else:
            if self.parameters.boundary_conditions_type == 1:
                bc_type_str = "Specular rebound"
            elif self.parameters.boundary_conditions_type == 2:
                bc_type_str = "Diffuse rebound"

        print(' ~~> Simulation type :', simudim)
        print(' ~~> Lagrangian Stochastic Model : LSM-{}'.format(self.parameters.model))
        print(' ~~> Integration method :', time_scheme_str)
        print(' ~~> Time interval : [{0:.3f}, {1:.3f}] - dt = {2:.3f}'.format(
            self.parameters.initial_time,
            self.parameters.final_time,
            self.parameters.time_step))
        print(' ~~> Diffusion model :', diffusion_str)
        print(' ~~> Particle velocity :', vel_init_str)
        print(' ~~> Wall boundary conditions :', bc_type_str)
