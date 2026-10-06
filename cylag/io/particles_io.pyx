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
from libc.math cimport sqrt
from ..extra.statistics_from_particles cimport sph_smoothing_kernel

class ParticlesIO():
    """
    Particles result class for post-processing purposes
    Rmk: use to load particles results of different format and export

    Parameters
    ----------
    times (array(ntime)): computation times
    iterations (array(ntime)): computation iterations
    npart (array(ntime)): number of particles at each time step
    tags (List[0:ntime][0:npart]): particles tags
    xp (List[0:ntime][0:npart]): x coordinates of particles
    yp (List[0:ntime][0:npart]): y coordinates of particles
    zp (List[0:ntime][0:npart]): z coordinates of particles

    Attributes
    ----------
    ntimes (int): number of time steps
    times (array(ntime)): computation times
    iterations (array(ntime)): computation iterations
    npart (array(ntime)): number of particles at each time step
    tags (List[0:ntime][0:npart]): particles tags
    xp (List[0:ntime][0:npart]): x coordinates of particles
    yp (List[0:ntime][0:npart]): y coordinates of particles
    zp (List[0:ntime][0:npart]): z coordinates of particles
    up (List[0:ntime][0:npart], optional): x velocity of particles
    vp (List[0:ntime][0:npart], optional): y velocity of particles
    wp (List[0:ntime][0:npart], optional): z velocity of particles
    us (List[0:ntime][0:npart], optional): x fluid velocity seen by particles
    vs (List[0:ntime][0:npart], optional): y fluid velocity seen by particles
    ws (List[0:ntime][0:npart], optional): z fluid velocity seen by particles
    ua (List[0:ntime][0:npart], optional): x additional velocity of particles
    va (List[0:ntime][0:npart], optional): y additional velocity of particles
    wa (List[0:ntime][0:npart], optional): z additional velocity of particles
    fax (List[0:ntime][0:npart], optional): x additional force of particles
    fay (List[0:ntime][0:npart], optional): y additional force of particles
    faz (List[0:ntime][0:npart], optional): z additional force of particles
    density (List[0:ntime][0:npart], optional): density estimate at particle positions
    diameter (List[0:ntime][0:npart], optional): particle diameter

    Methods
    -------
    get_trajectory(tag): get trajectory in time of tagged particle
    get_trajectories(tags): get multiple trajectories
    write(file_name, output_format): write cylag particle output file
    from_telemac3d(file_name): return ParticlesIO class from telemac3d result
    """
    def __init__(self, times, iterations, tags, xp, yp, zp,\
                 up=None, vp=None, wp=None,\
                 us=None, vs=None, ws=None,\
                 ua=None, va=None, wa=None,\
                 fax=None, fay=None, faz=None,\
                 diameter=None, depth=None):
        # particle general properties
        self.times = times
        self.iterations = iterations
        self.ntimes = np.shape(times)[0]
        self.tags = tags
        self.xp = xp
        self.yp = yp
        self.zp = zp
        self.npart = np.empty((self.ntimes), dtype='int32')
        self._count_particles()
        # particle velocity
        self.up = self._init_particle_attribute(up)
        self.vp = self._init_particle_attribute(vp)
        self.wp = self._init_particle_attribute(wp)
        # particle fluid velocity seen
        self.us = us
        self.vs = vs
        self.ws = ws
        # particle additional velocity or force
        self.ua = ua
        self.va = va
        self.wa = wa
        self.fax = fax
        self.fay = fay
        self.faz = faz
        # other particle properties
        self.density = self._init_particle_attribute(None)
        self.diameter = self._init_particle_attribute(diameter)
        self.depth = self._init_particle_attribute(depth)

    def _count_particles(self):
        """ 
        Count the number of particles at each time step 
        
        Returns
        -------
        npart (array(ntime)): number of particles at each time step
        """
        cdef int i, ntimes

        ntimes = self.ntimes
        for i in range(ntimes):
            self.npart[i] = len(self.tags[i])

    def _init_particle_attribute(self, val):
        """ 
        Initialize particle attribute (set same shape as positions) 
        
        Returns
        -------
        attribute (List[0:ntime][0:npart]): particle attribute
        """
        cdef int i, ni, nj

        if val is None:
            attribute = []
            ni = len(self.tags)
            for i in range(ni):
                nj = len(self.tags[i])
                attribute.append(np.zeros((nj), dtype='d'))
            return attribute
        else:
            return val

    def get_trajectories(self, tags):
        """ 
        Get particles trajectories in time from particule id list
        
        Parameters
        ----------
        tags (List[int]): list of particle tags

        Returns
        -------
        trajectories (List[0:len(tags)][0:ntime][3]): 
            list of particle trajectories (x, y, z) in time
        """
        cdef int tag

        trajectories = []
        for tag in tags:
            traj = self.get_trajectory(tag)
            trajectories.append(traj)
        return trajectories

    def get_trajectory(self, tag):
        """ 
        Get particle trajectory in time from particule id 
        
        Parameters
        ----------
        tag (int): particle tag

        Returns
        -------
        trajectory (array(ntime, 3)): particle trajectory (x, y, z) in time
        """
        cdef:
            int i, j, npart, itag
            int[:] tags_i
            double[:] xp_i, yp_i, zp_i

        trajectory = []
        itag = int(tag)
        for i in range(self.ntimes):
            tags_i = self.tags[i]
            xp_i = self.xp[i]
            yp_i = self.yp[i]
            zp_i = self.zp[i]
            npart = tags_i.shape[0]
            for j in range(npart):
                if tags_i[j] == itag:
                    trajectory.append([xp_i[j], yp_i[j], zp_i[j]])
                    break
        return np.asarray(trajectory)

    def get_mean_trajectory(self):
        """ 
        Get the mean trajectory (3D: x, y, z)

        Returns
        -------
        trajectory (array(ntime, 3)): mean particle trajectory (x, y, z) in time
        """
        cdef int i, j, nj
        cdef double xmean, ymean, zmean
        cdef double[:] xp_i, yp_i, zp_i

        trajectory = []
        for i in range(self.ntimes):
            xp_i = self.xp[i]
            yp_i = self.yp[i]
            zp_i = self.zp[i]
            nj = xp_i.shape[0]
            xmean = 0.
            ymean = 0.
            zmean = 0.
            for j in range(nj):
                xmean += xp_i[j]
                ymean += yp_i[j]
                zmean += zp_i[j]
            xmean /= nj
            ymean /= nj
            zmean /= nj            
            trajectory.append([xmean, ymean, zmean])
        return np.asarray(trajectory)

    def get_mean_trajectory_2d(self, threshold=None):
        """ 
        Get the mean trajectory (2D: x, y)
        
        Parameters
        ----------
        threshold (float, optional): z threshold; particles
            with z > threshold are excluded from the mean computation

        Returns
        -------
        trajectory (array(ntime, 2)): mean particle trajectory (x, y) in time
        """
        cdef int i, j, nj, count
        cdef double xmean, ymean, threshold_value
        cdef bint use_threshold
        cdef double[:] xp_i, yp_i, zp_i

        trajectory = []
        threshold_value = 0.
        use_threshold = threshold is not None
        if use_threshold:
            threshold_value = threshold
        for i in range(self.ntimes):
            xp_i = self.xp[i]
            yp_i = self.yp[i]
            zp_i = self.zp[i]
            nj = xp_i.shape[0]
            xmean = 0.
            ymean = 0.
            count = 0
            for j in range(nj):
                if use_threshold and zp_i[j] > threshold_value:
                    continue
                xmean += xp_i[j]
                ymean += yp_i[j]
                count += 1
            if count > 0:
                xmean /= count
                ymean /= count
            trajectory.append([xmean, ymean])
        return np.asarray(trajectory)

    def compute_density(self, virtual_mass=1., h=100., kernel=5, dim=2):
        """
        Get density estimate at particle positions.

        Parameters
        ----------
        virtual_mass : float, optional
            Virtual mass of particles.
        h : float, optional
            Smoothing length.
        kernel : int, optional
            Smoothing kernel identifier.
        dim : int, optional
            Dimension (2 or 3).

        Returns
        -------
        density (List[0:ntime][0:npart]): density estimate at particle positions
        """
        cdef:
            int i, j, k
            int npart
            double dist
            double dx, dy, dz
            double xj, yj, zj
            double rho
            double influence
            double[:] xp_i, yp_i, zp_i, density_i

        # loop on times
        for i in range(self.ntimes):
            npart = <int> self.npart[i]
            xp_i = self.xp[i]
            yp_i = self.yp[i]
            zp_i = self.zp[i]
            density_i = self.density[i]
            # loop on particles
            for j in range(npart):
                xj = xp_i[j]
                yj = yp_i[j]
                zj = zp_i[j]
                rho = 0.
                for k in range(npart):
                    dx = xj - xp_i[k]
                    dy = yj - yp_i[k]
                    if dim == 3:
                        dz = zj - zp_i[k]
                        dist = sqrt(dx*dx + dy*dy + dz*dz)
                    else:
                        dist = sqrt(dx*dx + dy*dy)
                    influence = sph_smoothing_kernel(h, dist, kernel, dim)
                    rho += virtual_mass * influence
                density_i[j] = rho

    def compute_density_at_given_rec(self, i=0, virtual_mass=1., h=100., kernel=5, dim=2):
        """
        Get density estimate at particle positions for a given record.

        Parameters
        ----------
        i : int, optional
            Record index at which to compute density.
        virtual_mass : float, optional
            Virtual mass of particles.
        h : float, optional
            Smoothing length.
        kernel : int, optional
            Smoothing kernel identifier.
        dim : int, optional
            Dimension (2 or 3).

        Returns
        -------
        density (List[0:npart]): density estimate at particle positions for the given record
        """
        cdef:
            int j, k
            int npart=<int> self.npart[i]
            double dist
            double dx, dy, dz
            double xj, yj, zj
            double rho
            double influence
            double[:] xp_i, yp_i, zp_i, density_i

        xp_i = self.xp[i]
        yp_i = self.yp[i]
        zp_i = self.zp[i]
        density_i = self.density[i]
        # loop on particles
        for j in range(npart):
            xj = xp_i[j]
            yj = yp_i[j]
            zj = zp_i[j]
            rho = 0.
            for k in range(npart):
                dx = xj - xp_i[k]
                dy = yj - yp_i[k]
                if dim == 3:
                    dz = zj - zp_i[k]
                    dist = sqrt(dx*dx + dy*dy + dz*dz)
                else:
                    dist = sqrt(dx*dx + dy*dy)
                influence = sph_smoothing_kernel(h, dist, kernel, dim)
                rho += virtual_mass * influence
            density_i[j] = rho

    def write_cylag_txt(self, file_name):
        """ 
        Write particle file in .txt format 
        
        Parameters
        ----------
        file_name (str): output file name
        """
        cdef:
            int i, j
            double time

        # Prepare output dir
        outputdir = os.path.join("particles")
        if not os.path.exists(outputdir):
            os.system("mkdir {}".format(outputdir))

        # Prepare txt output file
        fileout = os.path.join('particles', file_name+'.txt')
        if os.path.isfile(fileout):
            os.system("rm {}".format(fileout))
        else:
            os.system("touch {}".format(fileout))

        # write Header
        f = open(fileout, 'a')
        f.write('cylag particles result file\n')
        f.write('Variables: Tags, Xp, Yp, Zp, Up, Vp, Wp, ')
        if self.us is not None:
            f.write('Us, Vs, Ws, ')
        if self.ua is not None:
            f.write('Ua, Va, Wa, ')
        if self.fax is not None:
            f.write('Fax, Fay, Faz, ')
        has_diameter = len(self.diameter) > 0 and self.diameter[0].sum() != 0.
        if has_diameter:
            f.write('dp, ')
        has_depth = len(self.depth) > 0 and self.depth[0].sum() != 0.
        if has_depth:
            f.write('depth, ')
        f.write('\n')

        for i, time in enumerate(self.times):
            # write time
            f = open(fileout, 'a')
            f.write("Time = {:.8f}, ".format(time))
            f.write("Iter = {:16d}, ".format(self.iterations[i]))
            f.write("Npart = {:16d}\n".format(self.npart[i]))
            # write particules
            for j in range(self.npart[i]):
                f.write('{:16d}, '.format(self.tags[i][j]))
                f.write('{:.16e}, '.format(self.xp[i][j]))
                f.write('{:.16e}, '.format(self.yp[i][j]))
                f.write('{:.16e}, '.format(self.zp[i][j]))
                f.write('{:.8e}, '.format(self.up[i][j]))
                f.write('{:.8e}, '.format(self.vp[i][j]))
                f.write('{:.8e}, '.format(self.wp[i][j]))
                if self.us is not None:
                    f.write('{:.8e}, '.format(self.us[i][j]))
                    f.write('{:.8e}, '.format(self.vs[i][j]))
                    f.write('{:.8e}, '.format(self.ws[i][j]))
                if self.ua is not None:
                    f.write('{:.8e}, '.format(self.ua[i][j]))
                    f.write('{:.8e}, '.format(self.va[i][j]))
                    f.write('{:.8e}, '.format(self.wa[i][j]))
                if self.fax is not None:
                    f.write('{:.8e}, '.format(self.fax[i][j]))
                    f.write('{:.8e}, '.format(self.fay[i][j]))
                    f.write('{:.8e}, '.format(self.faz[i][j]))
                has_diameter = len(self.diameter) > 0 and self.diameter[0].sum() != 0.
                if has_diameter:
                    f.write('{:.8e}, '.format(self.diameter[i][j]))
                if has_depth:
                    f.write('{:.8e}, '.format(self.depth[i][j]))
                f.write('\n')
            f.close()

    @staticmethod
    def from_cylag_txt(file_name):
        """ 
        Initialize particles from cylag .txt file

        Parameters
        ----------
        file_name (str): input file name
        """
        cdef:
            int i, lid, nparttoread
            bint read_us, read_ua, read_fa, read_diameter

        # init tables
        iterations = []
        times = []
        tags = []
        xp, yp, zp = [], [], []
        up, vp, wp = [], [], []
        # read
        f = open(file_name, 'r')
        # read header
        line = f.readline()
        line = f.readline()
        if 'Us' in line:
            read_us = True
            us, vs, ws = [], [], []
        else:
            read_us = False
            us, vs, ws = None, None, None
        if 'Ua' in line:
            read_ua = True
            ua, va, wa = [], [], []
        else:
            read_ua = False
            ua, va, wa = None, None, None
        if 'Fax' in line:
            read_fa = True
            fax, fay, faz = [], [], []
        else:
            read_fa = False
            fax, fay, faz = None, None, None
        if 'dp' in line:
            read_diameter = True
            diameter = []
        else:
            read_diameter = False
            diameter = None
        if 'depth' in line:
            read_depth = True
            depth = []
        else:
            read_depth = False
            depth = None
        # read particles raw data
        while True:
            line = f.readline()
            if line.split(" = ")[0]=="Time":
                blockline = line.replace('=', ',').split(",")
                times.append(float(blockline[1]))
                iterations.append(int(blockline[3]))
                nparttoread = int(blockline[5])
                block_tags = np.empty(nparttoread, dtype='int32')
                block_xp = np.empty(nparttoread, dtype='d')
                block_yp = np.empty(nparttoread, dtype='d')
                block_zp = np.empty(nparttoread, dtype='d')
                block_ux = np.empty(nparttoread, dtype='d')
                block_uy = np.empty(nparttoread, dtype='d')
                block_uz = np.empty(nparttoread, dtype='d')
                if read_us:
                    block_usx = np.empty(nparttoread, dtype='d')
                    block_usy = np.empty(nparttoread, dtype='d')
                    block_usz = np.empty(nparttoread, dtype='d')
                if read_ua:
                    block_uax = np.empty(nparttoread, dtype='d')
                    block_uay = np.empty(nparttoread, dtype='d')
                    block_uaz = np.empty(nparttoread, dtype='d')
                if read_fa:
                    block_fax = np.empty(nparttoread, dtype='d')
                    block_fay = np.empty(nparttoread, dtype='d')
                    block_faz = np.empty(nparttoread, dtype='d')
                if read_diameter:
                    block_diameter = np.empty(nparttoread, dtype='d')
                if read_depth:
                    block_depth = np.empty(nparttoread, dtype='d')
                for i in range(nparttoread):
                    line = f.readline()
                    datline = line.split(",")
                    block_tags[i] = int(datline[0])
                    block_xp[i] = float(datline[1])
                    block_yp[i] = float(datline[2])
                    block_zp[i] = float(datline[3])
                    block_ux[i] = float(datline[4])
                    block_uy[i] = float(datline[5])
                    block_uz[i] = float(datline[6])
                    lid = 6
                    if read_us:
                        block_usx[i] = float(datline[lid+1])
                        block_usy[i] = float(datline[lid+2])
                        block_usz[i] = float(datline[lid+3])
                        lid += 3
                    if read_ua:
                        block_uax[i] = float(datline[lid+1])
                        block_uay[i] = float(datline[lid+2])
                        block_uaz[i] = float(datline[lid+3])
                        lid += 3
                    if read_fa:
                        block_fax[i] = float(datline[lid+1])
                        block_fay[i] = float(datline[lid+2])
                        block_faz[i] = float(datline[lid+3])
                        lid += 3
                    if read_diameter:
                        block_diameter[i] = float(datline[lid+1])
                        lid += 1
                    if read_depth:
                        block_depth[i] = float(datline[lid+1])
                        lid += 1
                tags.append(block_tags)
                xp.append(block_xp)
                yp.append(block_yp)
                zp.append(block_zp)
                up.append(block_ux)
                vp.append(block_uy)
                wp.append(block_uz)
                if read_us:
                    us.append(block_usx)
                    vs.append(block_usy)
                    ws.append(block_usz)
                if read_ua:
                    ua.append(block_uax)
                    va.append(block_uay)
                    wa.append(block_uaz)
                if read_fa:
                    fax.append(block_fax)
                    fay.append(block_fay)
                    faz.append(block_faz)
                if read_diameter:
                    diameter.append(block_diameter)
                if read_depth:
                    depth.append(block_depth)
            # end of file
            if not line:
                break
        f.close()
        return ParticlesIO(times, iterations, tags, \
                           xp, yp, zp, up, vp, wp, us, vs, ws,\
                           ua, va, wa, fax, fay, faz, diameter, depth)

    @staticmethod
    def from_telemac3d(file_name):
        """ 
        Initialize particles from telemac3d .dat file (tecplot format)

        Parameters
        ----------
        file_name (str): input file name
        """
        cdef int i, nparttoread

        # init tables
        times = []
        tags = []
        xp = []
        yp = []
        zp = []
        # read tecplot file_name
        f = open(file_name, 'r')
        # read header
        line = f.readline()
        line = f.readline()
        # read particles raw data
        while True:
            line = f.readline()
            if line.split(",")[0]=="ZONE DATAPACKING=POINT":
                blockline = line.replace('=', ',').split(",")
                nparttoread = int(blockline[5])
                times.append(float(blockline[7]))
                block_tags = np.empty(nparttoread, dtype='int32')
                block_xp = np.empty(nparttoread, dtype='d')
                block_yp = np.empty(nparttoread, dtype='d')
                block_zp = np.empty(nparttoread, dtype='d')
                for i in range(nparttoread):
                    line = f.readline()
                    datline = line.split(",")
                    block_tags[i] = int(datline[0])
                    block_xp[i] = float(datline[1])
                    block_yp[i] = float(datline[2])
                    block_zp[i] = float(datline[3])
                tags.append(block_tags)
                xp.append(block_xp)
                yp.append(block_yp)
                zp.append(block_zp)
            # end of file
            if not line:
                break
        f.close()
        return ParticlesIO(times, None, tags, xp, yp, zp)
