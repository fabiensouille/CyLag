# cython: profile=False
"""
Write particles functions

Date: 18-01-2023
Author: FABIEN SOUILLE
"""
import os
import glob
import numpy as np
from ..core.parameters cimport Parameters
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from .writer_vtk import write_vtk_2d, write_vtk_3d
from .writer_txt import write_txt_2d, write_txt_3d
from ..io.parallel_tags cimport set_parallel_tag
from ..io import ParticlesIO

def prepare_output(pset, outputdir, output_file_format, output_file_name, model=1):
    """
    Prepare output files and repository

    Parameters
    ----------
    output_file_format : str, format of particle result file 
    output_file_name : str, name of particle result file
    """
    cdef int i

    # Prepare output dir
    os.makedirs(outputdir, exist_ok=True)

    # Prepare txt output file
    if output_file_format == 'txt' or output_file_format == 'all':
        fileout = os.path.join(outputdir, output_file_name+'.txt')
        # write Header (overwrites any pre-existing file)
        f = open(fileout, 'w')
        f.write('cylag particles result file\n')
        f.write('Variables: Tags, Xp, Yp, Zp, Up, Vp, Wp, ')
        if model>=2:
            f.write('Us, Vs, Ws, ')
        if pset.additional_velocity:
            f.write('Ua, Va, Wa, ')
        if pset.additional_force:
            f.write('Fax, Fay, Faz, ')
        if pset.diameter_distribution != 'monodisperse':
            f.write('dp, ')
        if pset.compute_depth:
            f.write('depth, ')
        f.write('\n')
        f.close()

cpdef void write_particles(\
        LagrangianParticleSet particle_set,\
        double time, int iteration, int size, int rank,
        Parameters params):
    """ 
    Write lagrangian particles

    Parameters
    ----------
    particle_set : cylag.LagrangianParticleSet
    iteration : int, lagrangian time iteration
    params : dict, lagrangian algo parameters
    """
    # VTK writer
    if params.output_file_format == 'vtk' or params.output_file_format == 'all':
        if particle_set.dim == 2:
            write_vtk_2d(particle_set, time, iteration, size, rank,\
                params.output_rep,
                params.output_file_name,
                params.model)

        elif particle_set.dim == 3:
            write_vtk_3d(particle_set, time, iteration, size, rank,\
                params.output_rep,
                params.output_file_name,
                params.model)

    # TXT writer
    if params.output_file_format == 'txt' or params.output_file_format == 'all':
        if particle_set.dim == 2:
            write_txt_2d(particle_set, time, iteration, size, rank,\
                params.output_rep,
                params.output_file_name,
                params.model)

        elif particle_set.dim == 3:
            write_txt_3d(particle_set, time, iteration, size, rank,\
                params.output_rep,
                params.output_file_name,
                params.model)

def merge_results(size, outputdir, file_name, output_file_format='txt', debug=False):
    """ 
    Merge cylag result files after parallel run
    
    Parameters
    ----------
    size : int, number of result files to merge (number of proc)
    outputdir : str, output directory
    file_name : str, name of the result files (with '#' rank suffix)
    output_file_format : str, format of particle result file ('txt', 'vtk', 'all')
    debug : bool, if True, do not remove temporary files
    """
    cdef:
        int i, j

    if output_file_format == 'txt' or output_file_format == 'all':
        _merge_txt_results(size, outputdir, file_name, debug)

    if output_file_format == 'vtk' or output_file_format == 'all':
        _merge_vtk_results(size, outputdir, file_name, debug)

def _merge_txt_results(size, outputdir, file_name, debug=False):
    """ 
    Merge cylag .txt result files after parallel run
    """
    cdef:
        int i, j

    fileref = file_name.split('#')[0]
    fileout = os.path.join(outputdir, fileref+'.txt')

    # read all part files
    part_list = []
    for prc in range(size):
        filep = os.path.join(outputdir, fileref+'#{}'.format(prc)+'.txt')
        part_list.append(ParticlesIO.from_cylag_txt(filep))

    # Prepare txt output file
    if os.path.isfile(fileout):
        os.remove(fileout)

    # write Header
    f = open(fileout, 'w')
    f.write('cylag particles result file\n')
    f.write('Variables: Tags, Xp, Yp, Zp, Up, Vp, Wp, ')
    if part_list[0].us is not None:
        f.write('Us, Vs, Ws, ')
    if part_list[0].ua is not None:
        f.write('Ua, Va, Wa, ')
    if part_list[0].fax is not None:
        f.write('Fax, Fay, Faz, ')
    has_diameter = any(np.any(np.asarray(d) != 0.) for p in part_list for d in p.diameter)
    has_depth = any(np.any(np.asarray(d) != 0.) for p in part_list for d in p.depth)
    if has_diameter:
        f.write('dp, ')
    if has_depth:
        f.write('depth, ')
    f.write('\n')
    f.close()

    for i in range(len(part_list[0].times)):

        f = open(fileout, 'a')

        f.write("Time = {:.8f}, ".format(part_list[0].times[i]))
        f.write("Iter = {:16d}, ".format(part_list[0].iterations[i]))

        npart_totp = 0
        for part in part_list:
            npart_totp += part.npart[i]
        f.write("Npart = {:16d}\n".format(npart_totp))

        for part in part_list:
            for j in range(part.npart[i]):
                f.write('{:16d}, '.format(part.tags[i][j]))
                f.write('{:.16e}, '.format(part.xp[i][j]))
                f.write('{:.16e}, '.format(part.yp[i][j]))
                f.write('{:.16e}, '.format(part.zp[i][j]))
                f.write('{:.8e}, '.format(part.up[i][j]))
                f.write('{:.8e}, '.format(part.vp[i][j]))
                f.write('{:.8e}, '.format(part.wp[i][j]))
                if part.us is not None:
                    f.write('{:.8e}, '.format(part.us[i][j]))
                    f.write('{:.8e}, '.format(part.vs[i][j]))
                    f.write('{:.8e}, '.format(part.ws[i][j]))
                if part.ua is not None:
                    f.write('{:.8e}, '.format(part.ua[i][j]))
                    f.write('{:.8e}, '.format(part.va[i][j]))
                    f.write('{:.8e}, '.format(part.wa[i][j]))
                if part.fax is not None:
                    f.write('{:.8e}, '.format(part.fax[i][j]))
                    f.write('{:.8e}, '.format(part.fay[i][j]))
                    f.write('{:.8e}, '.format(part.faz[i][j]))
                if has_diameter:
                    f.write('{:.8e}, '.format(part.diameter[i][j]))
                if has_depth:
                    f.write('{:.8e}, '.format(part.depth[i][j]))
                f.write('\n')

        f.close()

    # remove parallel solver temporary files
    if not debug:
        for prc in range(size):
            filep = os.path.join(outputdir, fileref+'#{}'.format(prc)+'.txt')
            if os.path.exists(filep):
                os.remove(filep)

def _merge_vtk_results(size, outputdir, file_name, debug=False):
    """ 
    Merge VTK result files after parallel run

    Parameters
    ----------
    size : int, number of MPI processes
    outputdir : str, output directory
    file_name : str, base file name with '#' rank suffix
    debug : bool, if True, do not remove temporary files
    """
    cdef:
        int ite, prc, npa, total_npa, idx, j

    fileref = file_name.split('#')[0]

    # Find all iterations from rank 0's files
    rank0_prefix = fileref + '#0_'
    iterations = []
    for fname in os.listdir(outputdir):
        if fname.startswith(rank0_prefix) and fname.endswith('.vtk'):
            ite_str = fname[len(rank0_prefix):len(fname)-len('.vtk')]
            if ite_str.isdigit():
                iterations.append(int(ite_str))
    iterations.sort()

    for ite in iterations:
        all_header = []
        all_points = []
        all_data_sections = []
        total_npa = 0

        for prc in range(size):
            filep = os.path.join(outputdir,
                fileref + '#{}'.format(prc) + '_{}.vtk'.format(ite))
            with open(filep, 'r') as f:
                lines = f.readlines()

            # Parse file structure
            idx = 0
            # Header: lines 0..3
            if prc == 0:
                all_header = lines[0:4]
            idx = 4

            # FIELD FIELDDATA section until POINTS
            field_lines = []
            while idx < len(lines) and not lines[idx].startswith('POINTS'):
                field_lines.append(lines[idx])
                idx += 1
            if prc == 0:
                all_header.extend(field_lines)

            # POINTS npa FLOAT
            npa = int(lines[idx].split()[1])
            idx += 1
            all_points.extend(lines[idx:idx+npa])
            idx += npa

            # CELLS npa 2*npa - skip
            idx += 1 + npa  # CELLS header + cell lines

            # CELL_TYPES npa - skip
            idx += 1 + npa  # CELL_TYPES header + type lines

            # POINT_DATA npa - skip header
            idx += 1

            # Parse SCALARS and VECTORS data sections
            data_sections = []
            while idx < len(lines):
                line = lines[idx].strip()
                if line.startswith('SCALARS'):
                    section = {'type': 'SCALARS', 'header': lines[idx],
                               'lookup': lines[idx+1], 'values': []}
                    idx += 2
                    while idx < len(lines) and \
                          not lines[idx].strip().startswith(('SCALARS', 'VECTORS')):
                        section['values'].append(lines[idx])
                        idx += 1
                    data_sections.append(section)
                elif line.startswith('VECTORS'):
                    section = {'type': 'VECTORS', 'header': lines[idx],
                               'values': []}
                    idx += 1
                    while idx < len(lines) and \
                          not lines[idx].strip().startswith(('SCALARS', 'VECTORS')):
                        section['values'].append(lines[idx])
                        idx += 1
                    data_sections.append(section)
                else:
                    idx += 1

            if prc == 0:
                all_data_sections = [
                    {'type': s['type'], 'header': s['header'],
                     'lookup': s.get('lookup', ''),
                     'values': list(s['values'])}
                    for s in data_sections]
            else:
                for si in range(len(data_sections)):
                    if si < len(all_data_sections):
                        all_data_sections[si]['values'].extend(
                            data_sections[si]['values'])

            total_npa += npa

        # Write merged file
        fileout = os.path.join(outputdir, fileref + '_{}.vtk'.format(ite))
        with open(fileout, 'w') as f:
            for line in all_header:
                f.write(line)
            f.write('POINTS {} FLOAT\n'.format(total_npa))
            for line in all_points:
                f.write(line)
            f.write('CELLS {} {}\n'.format(total_npa, 2*total_npa))
            for j in range(total_npa):
                f.write('1 {}\n'.format(j))
            f.write('CELL_TYPES {}\n'.format(total_npa))
            for j in range(total_npa):
                f.write('1\n')
            f.write('POINT_DATA {}\n'.format(total_npa))
            for section in all_data_sections:
                f.write(section['header'])
                if section['type'] == 'SCALARS':
                    f.write(section['lookup'])
                for line in section['values']:
                    f.write(line)

    # Remove parallel solver temporary files
    if not debug:
        for ite in iterations:
            for prc in range(size):
                filep = os.path.join(outputdir,
                    fileref + '#{}'.format(prc) + '_{}.vtk'.format(ite))
                if os.path.exists(filep):
                    os.remove(filep)

cpdef void write_stranded_particles_txt_2d(\
    int size, int rank,
    str outputdir, str file_name,
    double[:,:] stranded_positions,
    int stranded_count):
    """
    Write stranded particles output file after simulation.

    Parameters
    ----------
    size : int
        Number of MPI ranks.
    rank : int
        Current MPI rank.
    outputdir : str
        Output directory.
    file_name : str
        Base file name for stranded particle output.
    stranded_tags : int[:]
        Stranded particle tag buffer.
    stranded_positions : double[:,:]
        Stranded particle position buffer.
    stranded_count : int
        Number of stranded particles recorded.
    """
    cdef int i
    cdef int tag
    cdef double x, y, z
    cdef str fileout
    cdef str fileref

    if stranded_count <= 0:
        return

    os.makedirs(outputdir, exist_ok=True)
    fileout = os.path.join(outputdir, file_name + '.txt')

    if os.path.isfile(fileout):
        os.remove(fileout)

    with open(fileout, 'w') as f:
        f.write('cylag stranded particles file\n')
        f.write('Variables: Xp, Yp, Zp\n')
        for i in range(stranded_count):
            x = stranded_positions[i, 0]
            y = stranded_positions[i, 1]
            z = stranded_positions[i, 2]
            f.write('{:.16e}, '.format(x))
            f.write('{:.16e}, '.format(y))
            f.write('{:.16e}\n'.format(z))

def merge_stranded_results(size, outputdir, file_name, debug=False):
    """
    Merge stranded particle txt files after parallel run.
    """
    fileref = file_name.split('#')[0]
    fileout = os.path.join(outputdir, fileref + '.txt')

    # remove existing merged output
    if os.path.isfile(fileout):
        os.remove(fileout)

    parts = []
    for prc in range(size):
        filep = os.path.join(outputdir, fileref + '#{}'.format(prc) + '.txt')
        if not os.path.isfile(filep):
            continue
        with open(filep, 'r') as f:
            lines = f.readlines()
        if len(lines) <= 2:
            continue
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            parts.append(line)

    if parts:
        with open(fileout, 'w') as f:
            f.write('cylag stranded particles file\n')
            f.write('Variables: Xp, Yp, Zp\n')
            for line in parts:
                f.write(line + '\n')

    if not debug:
        for prc in range(size):
            filep = os.path.join(outputdir, fileref + '#{}'.format(prc) + '.txt')
            if os.path.isfile(filep):
                os.remove(filep)