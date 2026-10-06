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

cpdef void write_vtk_2d(\
    LagrangianParticleSet pset,
    double time, int iteration, int size, int rank,
    str outputdir, str file_name, int model):
    """
    Write lagrangian particles to vtk (2D version)

    Parameters
    ----------
    pset : cylag.LagrangianParticleSet
    time : float
        simulation time
    iteration : int
        iteration
    file_name : str
        output file name
    """
    cdef:
        int j
        int npart = pset.npart
        int npa = pset.npart_active
        double[:,:] pos = pset.position
        double[:,:] vel = pset.velocity
        double[:,:] svel = pset.fluid_velocity_seen
        int tagj
        int[:] inactive = pset.inactive

    # select optional attributes to write
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if pset.diameter_distribution=='monodisperse':
        has_diameter = False
    else:
        has_diameter = True

    # write output file
    # ~~~~~~~~~~~~~~~~~
    fileout = os.path.join(outputdir, file_name + "_{}.vtk".format(iteration))

    # Write vtk
    f = open(fileout, 'w')
    f.write('# vtk DataFile Version 2.0\n')
    f.write('particle positions (Points 2D)\n')
    f.write('ASCII\n')
    f.write("DATASET UNSTRUCTURED_GRID\n")

    # Time
    f.write("FIELD FIELDDATA 2\n")
    f.write("TIME 1 1 DOUBLE\n")
    f.write("{}\n".format(time))
    f.write("CYCLE 1 1 INT\n")
    f.write("{}\n".format(iteration))

    f.write('POINTS {} FLOAT\n'.format(npa))
    for j in range(npart):
        if inactive[j] == 0:
            f.write('{} {} {}\n'.format(pos[j, 0], pos[j, 1], 0.))

    # Cells are the particules.
    f.write('CELLS {} {}\n'.format(npa, 2*npa))
    cnt = 0
    for j in range(npart):
        if inactive[j] == 0:
            f.write('1 {}\n'.format(cnt))
            cnt += 1

    # All cells are of type VTK_VERTEX
    f.write('CELL_TYPES {}\n'.format(npa))
    for j in range(npart):
        if inactive[j] == 0:
            f.write('1\n')

    # Give all cell one index.
    f.write('POINT_DATA {}\n'.format(npa))
    f.write('SCALARS Index INT\n')
    f.write('LOOKUP_TABLE default\n')
    for j in range(npart):
        if inactive[j] == 0:
            f.write('{}\n'.format(j))

    # Give all cell its tag.
    f.write('SCALARS tag INT\n')
    f.write('LOOKUP_TABLE default\n')
    for j in range(npart):
        if inactive[j] == 0:
            # tag modif only in parallel
            if size > 1:
                tagj = set_parallel_tag(size, rank, pset.tag[j])
            else:
                tagj = pset.tag[j]
            f.write('{}\n'.format(tagj))

    # Give all cell its velocity
    f.write("VECTORS velocity FLOAT\n")
    for j in range(npart):
        if inactive[j] == 0:
            f.write('{} {} {}\n'.format(vel[j, 0], vel[j, 1], 0.))

    # Give all cell its diameter
    if has_diameter:
        f.write("SCALARS diameter FLOAT\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(npart):
            if inactive[j] == 0:
                f.write('{}\n'.format(pset.diameter[j]))

    # Give all cell its fluid velocity seen
    if model>=2:
        f.write("VECTORS fluid_velocity_seen FLOAT\n")
        for j in range(npart):
            if inactive[j] == 0:
                f.write('{} {} {}\n'.format(svel[j, 0], svel[j, 1], 0.))

    # Additional velocity
    if pset.additional_velocity:
        f.write("VECTORS additional_velocity FLOAT\n")
        for j in range(npart):
            if inactive[j] == 0:
                f.write('{} {} {}\n'.format(\
                    pset.addvelocity[j, 0],\
                    pset.addvelocity[j, 1], 0.))

    # Additional force
    if pset.additional_force:
        f.write("VECTORS additional_force FLOAT\n")
        for j in range(npart):
            if inactive[j] == 0:
                f.write('{} {} {}\n'.format(\
                    pset.addforce[j, 0],\
                    pset.addforce[j, 1], 0.))
    f.close()

cpdef void write_vtk_3d(\
    LagrangianParticleSet pset,
    double time, int iteration, int size, int rank, 
    str outputdir, str file_name, int model):
    """
    Write lagrangian particles to vtk (3D version)

    Parameters
    ----------
    pset : cylag.LagrangianParticleSet
    time : float
        simulation time
    iteration : int
        iteration
    file_name : str
        output file name
    """
    cdef:
        int j
        int npart = pset.npart
        int npa = pset.npart_active
        double[:,:] pos = pset.position
        double[:,:] vel = pset.velocity
        double[:,:] svel = pset.fluid_velocity_seen
        int tagj
        int[:] inactive = pset.inactive

    # selection of optional attributes to write
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if pset.diameter_distribution=='monodisperse':
        has_diameter = False
    else:
        has_diameter = True

    # write output file
    # ~~~~~~~~~~~~~~~~~
    fileout = os.path.join(outputdir, file_name + "_{}.vtk".format(iteration))

    # Write vtk
    f = open(fileout, 'w')
    f.write('# vtk DataFile Version 2.0\n')
    f.write('particle positions (Points 3D)\n')
    #f.write('Time = {}\n'.format(time))
    f.write('ASCII\n')
    f.write("DATASET UNSTRUCTURED_GRID\n")

    # Time
    f.write("FIELD FIELDDATA 2\n")
    f.write("TIME 1 1 DOUBLE\n")
    f.write("{}\n".format(time))
    f.write("CYCLE 1 1 INT\n")
    f.write("{}\n".format(iteration))

    # Particules coordinates (x,y,z)
    f.write('POINTS {} FLOAT\n'.format(npa))
    for j in range(npart):
        if inactive[j] == 0:
            f.write('{} {} {}\n'.format(pos[j, 0], pos[j, 1], pos[j, 2]))

    # Cells are the particules.
    f.write('CELLS {} {}\n'.format(npa, 2*npa))
    cnt = 0
    for j in range(npart):
        if inactive[j] == 0:
            f.write('1 {}\n'.format(cnt))
            cnt += 1

    # All cells are of type VTK_VERTEX
    f.write('CELL_TYPES {}\n'.format(npa))
    for j in range(npart):
        if inactive[j] == 0:
            f.write('1\n')

    # Give all cell one index.
    f.write('POINT_DATA {}\n'.format(npa))
    f.write('SCALARS Index INT\n')
    f.write('LOOKUP_TABLE default\n')
    for j in range(npart):
        if inactive[j] == 0:
            f.write('{}\n'.format(j))

    # Give all cell its tag.
    f.write('SCALARS tag INT\n')
    f.write('LOOKUP_TABLE default\n')
    for j in range(npart):
        if inactive[j] == 0:
            # tag modif only in parallel
            if size > 1:
                tagj = set_parallel_tag(size, rank, pset.tag[j])
            else:
                tagj = pset.tag[j]
            f.write('{}\n'.format(tagj))

    # Give all cell its elevation (for warp by scalar filter of paraview).
    f.write('SCALARS elevation FLOAT\n')
    f.write('LOOKUP_TABLE default\n')
    for j in range(npart):
        if inactive[j] == 0:
            f.write('{}\n'.format(pos[j, 2]))

    # Give all cell its velocity
    f.write("VECTORS velocity FLOAT\n")
    for j in range(npart):
        if inactive[j] == 0:
            f.write('{} {} {}\n'.format(vel[j, 0], vel[j, 1], vel[j, 2]))

    # Give all cell its diameter
    if has_diameter:
        f.write("SCALARS diameter FLOAT\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(npart):
            if inactive[j] == 0:
                f.write('{}\n'.format(pset.diameter[j]))

    # Give all cell its fluid velocity seen
    if model>=2:
        f.write("VECTORS fluid_velocity_seen FLOAT\n")
        for j in range(npart):
            if inactive[j] == 0:
                f.write('{} {} {}\n'.format(svel[j, 0], svel[j, 1], svel[j, 2]))

    # particle depth
    if pset.compute_depth:
        f.write('SCALARS depth FLOAT\n')
        f.write('LOOKUP_TABLE default\n')
        for j in range(npart):
            if inactive[j] == 0:
                f.write('{}\n'.format(pset.depth[j]))

    # Additional velocity
    if pset.additional_velocity:
        f.write("VECTORS additional_velocity FLOAT\n")
        for j in range(npart):
            if inactive[j] == 0:
                f.write('{} {} {}\n'.format(\
                    pset.addvelocity[j, 0],\
                    pset.addvelocity[j, 1],\
                    pset.addvelocity[j, 2]))

    # Additional force
    if pset.additional_force:
        f.write("VECTORS additional_force FLOAT\n")
        for j in range(npart):
            if inactive[j] == 0:
                f.write('{} {} {}\n'.format(\
                    pset.addforce[j, 0],\
                    pset.addforce[j, 1],\
                    pset.addforce[j, 2]))
    f.close()
