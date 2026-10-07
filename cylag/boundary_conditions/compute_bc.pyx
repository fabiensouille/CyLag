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
from libc.math cimport fabs, sqrt, fmin, fmax, sin, cos
from ..core.constants cimport BND_WALL_REF, BND_OPEN_REF,\
    PART_LOC_B, PART_LOC_A, PART_LOC_O, _DEG_TO_RAD, EPSILON_REBOUND
from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..geom.utils cimport normal2u
from ..geom.xylocalizer cimport xy_localize
from ..geom.intersections cimport compute_symetric_point2d
from ..geom.zlocalizer cimport c_compute_lower_layer_index
from ..lsm.random_utils cimport generate_random_single, random_uniform

cpdef void compute_boundary_condition(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i, int bnd_j,
        int[:] bnd_stats):
    """
    Compute boundary conditions
    
    Handles wall rebound, open boundaries, bottom/surface reflections,
    and updates boundary statistics.

    Parameters
    ----------
    simutime : cylag.SimuTime
        Simulation time manager
    fset : cylag.EulerianFieldSet
        Eulerian field set containing mesh and fields
    pset : cylag.LagrangianParticleSet
        Lagrangian particle set
    parameters : cylag.Parameters
        Model parameters
    i : int
        Particle index
    bnd_j : int
        Boundary edge index (or -1 if no boundary crossed)
    bnd_stats : int[:]
        Array for accumulating boundary crossing statistics
    """
    cdef:
        double time = simutime.time
        int dim = fset.dim
        int i0, i1
        int bnd_jj
        int rebound_count = 0
        int[:,:] edges = fset.triangular_mesh.edges
        int[:] edges_labels = fset.triangular_mesh.edges_labels
        double[:] x = fset.triangular_mesh.x
        double[:] y = fset.triangular_mesh.y
        double[:] edge_vtx0
        double[:] edge_vtx1
        double[:] new_position
        double[:] intersect
        double[:] nn
        double zbp, zsp, dz, hp
        double theta, cos_t, sin_t, nnx, nny
        double damp = min(parameters.rebound_damping_coef, 1.0-EPSILON_REBOUND)
        double strand = parameters.stranding_probability
        double roughness_angle = parameters.wall_roughness_angle*_DEG_TO_RAD
        double max_theta = 85.0*_DEG_TO_RAD
        int max_rebounds = parameters.max_wall_rebounds
        int bnd_type = parameters.boundary_conditions_type
        bint debug = parameters.boundary_conditions_debug
        bint delete_if_failed_rebound = 1
        bint strand_condition = 0

    # boundary edge crossed:
    # ~~~~~~~~~~~~~~~~~~~~~~
    if bnd_j!=-1:

        # Stranding condition
        # -------------------
        # strand_condition == 0 : particle is kept (we compute rebound)
        # strand_condition == 1 : particle is stranded (we delete and skip rebound)
        if edges_labels[bnd_j] == BND_WALL_REF and strand > 0.:
            strand_condition = random_uniform() < strand
        else:
            strand_condition = 0

        if debug and strand_condition==1:
            print(" ~~~> particle {} stranded on wall bnd edge {} ".format(i, bnd_j))

        # append stranded particle position
        if parameters.stranding_output and strand_condition==1:
            if pset.stranded_count < pset.max_stranded:
                if dim==2:
                    zval = 0.0
                else:
                    zval = pset.position[i, 2]
                pset.stranded_positions[pset.stranded_count, 0] = pset.position[i, 0]
                pset.stranded_positions[pset.stranded_count, 1] = pset.position[i, 1]
                pset.stranded_positions[pset.stranded_count, 2] = zval
                pset.stranded_count += 1
            else:
                print("Warning: stranded particle buffer is full\
                       Increase max_stranded in LagrangianParticleSet")

        # wall boundary conditions
        # ------------------------
        if edges_labels[bnd_j] == BND_WALL_REF and strand_condition==0:

            # init points and normal
            edge_vtx0 = np.empty((dim), dtype='d')
            edge_vtx1 = np.empty((dim), dtype='d')
            new_position = np.empty((dim), dtype='d')
            intersect = np.empty((dim), dtype='d')
            nn = np.empty((dim), dtype='d')
        
            if debug:
                print(" ~~~> particle {} cross wall bnd edge {} ".format(i, bnd_j))

            # recursive rebound until inside domain, stop if:
            #  > particle is inside domain (pset.tri[i]!=-1)
            #  > number of rebound has reached max rebound allowed (rebound_max)
            bnd_jj = bnd_j # last crossed bnd edge
            rebound_count = 0

            while pset.tri[i]==PART_LOC_O and rebound_count < max_rebounds:

                if debug:
                    print(" ~~~> computing rebound of particle {}".format(i))
                    print("rebound n°:", rebound_count+1)
                    print("pos before outside:", pset.last_position[i, 0], pset.last_position[i, 1])
                    print("tri before outside:", pset.last_tri[i])
                    print("pos before rebound:", pset.position[i, 0], pset.position[i, 1])
                    print("tri before rebound:", pset.tri[i])

                # bnd intersected edges
                i0 = edges[bnd_jj, 0]
                i1 = edges[bnd_jj, 1]
                edge_vtx0[0], edge_vtx0[1] = x[i0], y[i0]
                edge_vtx1[0], edge_vtx1[1] = x[i1], y[i1]

                # diffuse rebound 
                # ~~~~~~~~~~~~~~~~
                # compute symetric
                new_position = compute_symetric_point2d(\
                    edge_vtx0, edge_vtx1, pset.position[i], damp)

                # new position
                pset.position[i, 0] = new_position[0]
                pset.position[i, 1] = new_position[1]

                if debug:
                    print("pos after rebound:", pset.position[i, 0], pset.position[i, 1])

                # localize new position
                pset.tri[i], bnd_jj = xy_localize(\
                    fset.triangular_mesh,
                    pset.last_tri[i],
                    pset.last_position[i, 0],
                    pset.last_position[i, 1],
                    pset.position[i, 0],
                    pset.position[i, 1],
                    3, debug)

                if debug:
                    print("tri after rebound:", pset.tri[i])
                    print("last bnd edge crossed:", bnd_jj)

                # rebounded particle crossed an open boundary: it leaves the domain
                if pset.tri[i]==PART_LOC_O and bnd_jj!=-1 and\
                   edges_labels[bnd_jj]>=BND_OPEN_REF:
                    if parameters.bnd_statistics:
                        bnd_stats[edges_labels[bnd_jj]-1] += 1
                    pset.delete_single(i)
                    break

                # compute particle velocity after rebound
                if parameters.model==0 or parameters.model>=2:
                    # compute normal
                    normal2u(edge_vtx0, edge_vtx1, nn)

                    # perturb normal for diffuse (rough wall) reflection
                    if bnd_type==2 and roughness_angle > 0.:
                        theta = generate_random_single(parameters.rng_method)*roughness_angle
                        theta = fmax(-max_theta, fmin(max_theta, theta))
                        cos_t = cos(theta)
                        sin_t = sin(theta)
                        nnx = nn[0]*cos_t - nn[1]*sin_t
                        nny = nn[0]*sin_t + nn[1]*cos_t
                        nn[0] = nnx
                        nn[1] = nny

                    # rotation of up
                    up0 = pset.velocity[i, 0]
                    pset.velocity[i, 0] = up0*nn[0] + pset.velocity[i, 1]*nn[1]
                    pset.velocity[i, 1] =-up0*nn[1] + pset.velocity[i, 1]*nn[0]

                    # invert normal component of the velocity
                    pset.velocity[i, 0] = -1.*pset.velocity[i, 0]

                    # inverse rotation
                    up0 = pset.velocity[i, 0]
                    pset.velocity[i, 0] = up0*nn[0] - pset.velocity[i, 1]*nn[1]
                    pset.velocity[i, 1] = up0*nn[1] + pset.velocity[i, 1]*nn[0]
                    pset.velocity[i, 0] *= 1.-damp
                    pset.velocity[i, 1] *= 1.-damp
                    if pset.dim==3:
                        pset.velocity[i, 2] *= 1.-damp

                # compute fluid velocity seen after rebound
                if parameters.model==3:
                    # compute normal
                    normal2u(edge_vtx0, edge_vtx1, nn)

                    # perturb normal for diffuse (rough wall) reflection
                    if bnd_type==2 and roughness_angle > 0.:
                        theta = generate_random_single(parameters.rng_method)*roughness_angle
                        theta = fmax(-max_theta, fmin(max_theta, theta))
                        cos_t = cos(theta)
                        sin_t = sin(theta)
                        nnx = nn[0]*cos_t - nn[1]*sin_t
                        nny = nn[0]*sin_t + nn[1]*cos_t
                        nn[0] = nnx
                        nn[1] = nny

                    # rotation of up
                    up0 = pset.fluid_velocity_seen[i, 0]
                    pset.fluid_velocity_seen[i, 0] = up0*nn[0] + pset.fluid_velocity_seen[i, 1]*nn[1]
                    pset.fluid_velocity_seen[i, 1] =-up0*nn[1] + pset.fluid_velocity_seen[i, 1]*nn[0]

                    # invert normal component of the velocity
                    pset.fluid_velocity_seen[i, 0] = -1.*pset.fluid_velocity_seen[i, 0]

                    # inverse rotation
                    up0 = pset.fluid_velocity_seen[i, 0]
                    pset.fluid_velocity_seen[i, 0] = up0*nn[0] - pset.fluid_velocity_seen[i, 1]*nn[1]
                    pset.fluid_velocity_seen[i, 1] = up0*nn[1] + pset.fluid_velocity_seen[i, 1]*nn[0]

                # increment rebound count
                rebound_count += 1
                
                # get out of loop if failed to localize
                if pset.tri[i]==PART_LOC_O and bnd_jj==-1:
                    if debug:
                        print("Failed to localize particle after rebound ")
                    break

            # vertical relocalization in the new triangle after the rebound
            if pset.dim==3 and pset.tri[i]!=PART_LOC_O:
                pset.lowerlayer[i] = fset.z_localize(\
                    pset.tri[i],
                    pset.position[i, 0],
                    pset.position[i, 1],
                    pset.position[i, 2])

        # Open boundary condition
        # ------------------------
        if edges_labels[bnd_j] >= BND_OPEN_REF:
            pset.delete_single(i)

            if debug:
                print(" ~~~> particle {} cross open bnd edge {} ".format(i, bnd_j))
            
            # Open boundary statistics
            if parameters.bnd_statistics:
                bnd_stats[edges_labels[bnd_j]-1] += 1

    # Vertical boundary conditions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Only if 3D (fset.dim==3 and pset.dim==3) or 
    #   peudo 3D (fset.dim==2 and pset.dim==3)
    if (fset.dim==3 and pset.dim==3) or (fset.dim==2 and pset.dim==3):

        # particle below bottom
        # ~~~~~~~~~~~~~~~~~~~~~
        if pset.lowerlayer[i]==PART_LOC_B and pset.tri[i]!=PART_LOC_O:

            if debug:
                print(" ~~~> particle {} below bottom".format(i))

            # interpolate bottom and free surface
            if fset.dim==2 and pset.dim==3:
                zbp = fset._interpolate_field_2d(fset.fid_zb,\
                    pset.tri[i], 0,
                    pset.position[i, 0],
                    pset.position[i, 1])
                zsp = fset._interpolate_field_2d(fset.fid_zs,\
                    pset.tri[i], 0,
                    pset.position[i, 0],
                    pset.position[i, 1])
            else:
                zbp = fset._interpolate_field_2d(fset.fid_zs,\
                    pset.tri[i], 0,
                    pset.position[i, 0],
                    pset.position[i, 1])
                zsp = fset._interpolate_field_2d(fset.fid_zs,\
                    pset.tri[i], fset.nlayers-1,
                    pset.position[i, 0],
                    pset.position[i, 1])

            # compute z distances
            dz = fabs(zbp - pset.position[i, 2]) # overshoot distance
            hp = fmax(0., zsp - zbp) # water depth at particle position
            
            # vertical reflexion (cannot bounce above half the water depth)
            pset.position[i, 2] = fmin(zbp + (1.-damp)*dz, zbp + 0.5*hp)
            pset.lowerlayer[i] = c_compute_lower_layer_index(\
                fset.nlayers, zsp, zbp, pset.position[i, 2])

            # reflect vertical velocity upward for LSM-2 and LSM-3
            if parameters.model >= 2:
                pset.velocity[i, 2] = fabs(pset.velocity[i, 2])*(1. - damp)
            if parameters.model == 3:
                pset.fluid_velocity_seen[i, 2] = fabs(pset.fluid_velocity_seen[i, 2])*(1. - damp)

        # particle above surface
        # ~~~~~~~~~~~~~~~~~~~~~~
        if pset.lowerlayer[i]==PART_LOC_A and pset.tri[i]!=PART_LOC_O:
        
            if debug:
                print(" ~~~> particle {} above surface".format(i))
                
            # interpolate bottom and free surface
            if fset.dim==2 and pset.dim==3:
                zbp = fset._interpolate_field_2d(fset.fid_zb,\
                    pset.tri[i], 0,
                    pset.position[i, 0],
                    pset.position[i, 1])
                zsp = fset._interpolate_field_2d(fset.fid_zs,\
                    pset.tri[i], 0,
                    pset.position[i, 0],
                    pset.position[i, 1])
            else:
                zbp = fset._interpolate_field_2d(fset.fid_zs,\
                    pset.tri[i], 0,
                    pset.position[i, 0],
                    pset.position[i, 1])
                zsp = fset._interpolate_field_2d(fset.fid_zs,\
                    pset.tri[i], fset.nlayers-1,
                    pset.position[i, 0],
                    pset.position[i, 1])

            # compute z distances
            dz = fabs(pset.position[i, 2] - zsp) # overshoot distance
            hp = fmax(0., zsp - zbp) # water depth at particle position

            # vertical reflexion (cannot bounce below half the water depth)
            pset.position[i, 2] = fmax(zsp - (1.-damp)*dz, zsp - 0.5*hp)
            pset.lowerlayer[i] = c_compute_lower_layer_index(\
                fset.nlayers, zsp, zbp, pset.position[i, 2])

            # reflect vertical velocity downward for LSM-2 and LSM-3
            if parameters.model >= 2:
                pset.velocity[i, 2] = -fabs(pset.velocity[i, 2])*(1. - damp)
            if parameters.model == 3:
                pset.fluid_velocity_seen[i, 2] = -fabs(pset.fluid_velocity_seen[i, 2])*(1. - damp)

    # Delete particles if stranded or if failed rebound
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if pset.inactive[i] == 0:

        # delete particle if failed rebound or stranding
        if delete_if_failed_rebound or strand_condition:
            if pset.tri[i]==PART_LOC_O:
                pset.delete_single(i)
                if debug:
                    print(" ~~~> particle {} deleted after failed rebound".format(i))
                
            if pset.lowerlayer[i]==PART_LOC_B:
                pset.delete_single(i)
                if debug:
                    print(" ~~~> particle {} deleted after failed rebound".format(i))
                
            if pset.lowerlayer[i]==PART_LOC_A:
                pset.delete_single(i)
                if debug:
                    print(" ~~~> particle {} deleted after failed rebound".format(i))
