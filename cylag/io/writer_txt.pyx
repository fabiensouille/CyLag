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
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..io.parallel_tags cimport set_parallel_tag

cpdef void write_txt_2d(\
    LagrangianParticleSet pset,
    double time, int iteration, int size, int rank,
    str outputdir, str file_name, int model):
    """
    Write lagrangian particles to txt (2D version)

    Parameters
    ----------
    pset : cylag.LagrangianParticleSet
    time : flaot, simulation time
    iteration (int): iteration
    file_name (str): output file name
    """
    cdef:
        int j
        int npart = pset.npart
        int npa = pset.npart_active
        int[:] inactive = pset.inactive
        int tag

    # selection of optional attributes to write
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    has_zp = hasattr(pset, 'zp')
    has_hp = hasattr(pset, 'hp')

    if pset.diameter_distribution=='monodisperse':
        has_diameter = False
    else:
        has_diameter = True

    # write output file
    # ~~~~~~~~~~~~~~~~~
    fileout = os.path.join(outputdir, file_name+'.txt')

    # write time
    f = open(fileout, 'a')
    f.write("Time = {:.8f}, ".format(time))
    f.write("Iter = {:16d}, ".format(iteration))
    f.write("Npart = {:16d}\n".format(npa))

    # write particules
    for j in range(npart):
        if inactive[j] == 0:
        
            # tag modif only in parallel
            if size > 1:
                tag = set_parallel_tag(size, rank, pset.tag[j])
            else:
                tag = pset.tag[j]
            
            # write particle data
            f.write('{:16d}, '.format(tag))
            f.write('{:.16e}, '.format(pset.position[j, 0]))
            f.write('{:.16e}, '.format(pset.position[j, 1]))
            if has_zp:
                f.write('{:.16e}, '.format(pset.zp[j]))
            elif has_hp:
                f.write('{:.16e}, '.format(pset.hp[j]))
            else:
                f.write('{:1.0f}, '.format(0.))
            f.write('{:.8e}, '.format(pset.velocity[j, 0]))
            f.write('{:.8e}, '.format(pset.velocity[j, 1]))
            f.write('{:1.0f}, '.format(0.))
            if model>=2:
                f.write('{:.8e}, '.format(pset.fluid_velocity_seen[j, 0]))
                f.write('{:.8e}, '.format(pset.fluid_velocity_seen[j, 1]))
                f.write('{:1.0f}, '.format(0.))
            if pset.additional_velocity:
                f.write('{:.8e}, '.format(pset.addvelocity[j, 0]))
                f.write('{:.8e}, '.format(pset.addvelocity[j, 1]))
                f.write('{:1.0f}, '.format(0.))
            if pset.additional_force:
                f.write('{:.8e}, '.format(pset.addforce[j, 0]))
                f.write('{:.8e}, '.format(pset.addforce[j, 1]))
                f.write('{:1.0f}, '.format(0.))
            if has_diameter:
                f.write('{:.8e}, '.format(pset.diameter[j]))
            f.write('\n')

    f.close()

cpdef void write_txt_3d(\
    LagrangianParticleSet pset,
    double time, int iteration, int size, int rank, 
    str outputdir, str file_name, int model):
    """
    Write lagrangian particles to txt (3D version)

    Parameters
    ----------
    pset : cylag.LagrangianParticleSet
    time : flaot, simulation time
    iteration (int): iteration
    file_name (str): output file name
    """
    cdef:
        int j
        int npart = pset.npart
        int npa = pset.npart_active
        int[:] inactive = pset.inactive
        int tag

    # selection of optional attributes to write
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if pset.diameter_distribution=='monodisperse':
        has_diameter = False
    else:
        has_diameter = True

    # write output file
    # ~~~~~~~~~~~~~~~~~
    fileout = os.path.join(outputdir, file_name+'.txt')

    # write time
    f = open(fileout, 'a')
    f.write("Time = {:.8f}, ".format(time))
    f.write("Iter = {:16d}, ".format(iteration))
    f.write("Npart = {:16d}\n".format(npa))

    # write particules
    for j in range(npart):
        if inactive[j] == 0:

            # tag modif only in parallel
            if size > 1:
                tag = set_parallel_tag(size, rank, pset.tag[j])
            else:
                tag = pset.tag[j]

            # write particle data
            f.write('{:16d}, '.format(tag))
            f.write('{:.16e}, '.format(pset.position[j, 0]))
            f.write('{:.16e}, '.format(pset.position[j, 1]))
            f.write('{:.16e}, '.format(pset.position[j, 2]))
            f.write('{:.8e}, '.format(pset.velocity[j, 0]))
            f.write('{:.8e}, '.format(pset.velocity[j, 1]))
            f.write('{:.8e}, '.format(pset.velocity[j, 2]))
            if model>=2:
                f.write('{:.8e}, '.format(pset.fluid_velocity_seen[j, 0]))
                f.write('{:.8e}, '.format(pset.fluid_velocity_seen[j, 1]))
                f.write('{:.8e}, '.format(pset.fluid_velocity_seen[j, 2]))
            if pset.additional_velocity:
                f.write('{:.8e}, '.format(pset.addvelocity[j, 0]))
                f.write('{:.8e}, '.format(pset.addvelocity[j, 1]))
                f.write('{:.8e}, '.format(pset.addvelocity[j, 2]))
            if pset.additional_force:
                f.write('{:.8e}, '.format(pset.addforce[j, 0]))
                f.write('{:.8e}, '.format(pset.addforce[j, 1]))
                f.write('{:.8e}, '.format(pset.addforce[j, 2]))
            if has_diameter:
                f.write('{:.8e}, '.format(pset.diameter[j]))
            if pset.compute_depth:
                f.write('{:.8e}, '.format(pset.depth[j]))
            f.write('\n')

    f.close()