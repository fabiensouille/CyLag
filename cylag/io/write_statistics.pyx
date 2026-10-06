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
import numpy as np

def prepare_secstats_output(\
        bnd_statistics=False, nopen=0, 
        outputdir="particles", bnd_file_name="bnd_stats",
        cs_statistics=False, ncsec=0, cs_file_name="cs_stats"):
    """
    Prepare control section and bnd statistics files
    """
    # Prepare output dir
    os.makedirs(outputdir, exist_ok=True)

    # Prepare bnd statistics output file
    if bnd_statistics:
        for i in range(1, nopen+1):
            fileout = os.path.join(outputdir, bnd_file_name+'_{}.txt'.format(i))
            # check if file exists
            if os.path.isfile(fileout):
                os.system("rm {}".format(fileout))
            else:
                os.system("touch {}".format(fileout))
            # write Header
            f = open(fileout, 'a')
            f.write('cylag boundary statistics file\n')
            f.write('Variables: Time, Npart')
            f.write('\n')
            f.close()

    # Prepare control sections statistics output file
    if cs_statistics:
        for i in range(1, ncsec+1):
            fileout = os.path.join(outputdir, cs_file_name+'_{}.txt'.format(i))
            # check if file exists
            if os.path.isfile(fileout):
                os.system("rm {}".format(fileout))
            else:
                os.system("touch {}".format(fileout))
            # write Header
            f = open(fileout, 'a')
            f.write('cylag control section file\n')
            f.write('Variables: Time, Npart')
            f.write('\n')
            f.close()

cpdef void write_secstats(\
        int[:] stats, int nsec, double time, str outputdir, str bnd_file_name):
    """
    Write statistics on 2D control section or horizontal boundary condition
    """
    cdef:
        int i

    for i in range(1, nsec+1):
        #if stats[i] > 0: 
        fileout = os.path.join(outputdir, bnd_file_name+'_{}.txt'.format(i))
        f = open(fileout, 'a')
        f.write("{:.8f}, ".format(time))
        f.write("{:16d}  ".format(stats[i-1]))
        f.write('\n')
        f.close()

cpdef void write_secstats_buffer(\
        double[:] times, int[:,:] data, int nsteps, int nsec,\
        str outputdir, str bnd_file_name):
    """
    Write buffered statistics on 2D control section or horizontal boundary condition.

    Writes all buffered time steps at once, opening each file only once
    to avoid repeated file I/O overhead.

    Parameters
    ----------
    times : double[:]
        Array of times for each buffered step
    data : int[:,:]
        Array of shape (nsteps, nsec) with statistics per step and section
    nsteps : int
        Number of buffered time steps to write
    nsec : int
        Number of sections
    outputdir : str
        Output directory
    bnd_file_name : str
        Base file name for output
    """
    cdef:
        int i, j

    for i in range(1, nsec+1):
        fileout = os.path.join(outputdir, bnd_file_name+'_{}.txt'.format(i))
        f = open(fileout, 'a')
        for j in range(nsteps):
            f.write("{:.8f}, ".format(times[j]))
            f.write("{:16d}  ".format(data[j, i-1]))
            f.write('\n')
        f.close()

def merge_secstats_results(size, outputdir, file_name, nsec, debug=False):
    """
    Merge boundary statistics result files after parallel run

    Parameters
    ----------
    size : int, number of MPI processes
    outputdir : str, output directory
    file_name : str, base file name with '#' for rank number
    nsec : int, number of sections
    debug : bool, if True, do not remove temporary files
    """
    cdef:
        int i, j, k
        int nt

    # loop on nsec
    for i in range(1, nsec+1):
        fileref = file_name.split('#')[0]
        fileout = os.path.join(outputdir, fileref+'_{}.txt'.format(i))

        # remove pre-existing output file
        if os.path.isfile(fileout):
            os.remove(fileout)

        # read all files
        data = []
        for k in range(size):
            filek = os.path.join(outputdir,\
                fileref+'#{}'.format(k) + '_{}.txt'.format(i))
            datak = np.loadtxt(filek, delimiter=',', skiprows=2, ndmin=2)
            data.append(datak)

        # merge results
        nt = data[0].shape[0]
        global_data = np.zeros((nt, 2), dtype='d')
        for j in range(nt):
            global_data[j, 0] = data[0][j, 0]
            for k in range(size):
                global_data[j, 1] += data[k][j, 1]

        # save in txt
        f = open(fileout, 'w')
        f.write('cylag boundary statistics file\n')
        f.write('Variables: Time, Npart')
        f.write('\n')
        for j in range(nt):
            f.write("{:.8f}, ".format(global_data[j, 0]))
            f.write("{:16d}  ".format(int(global_data[j, 1])))
            f.write('\n')
        f.close()

    # remove parallel solver temporary files
    if not debug:
        for i in range(1, nsec+1):
            for k in range(size):
                filek = os.path.join(outputdir,\
                    fileref+'#{}'.format(k) + '_{}.txt'.format(i))
                os.system("rm {}".format(filek))
