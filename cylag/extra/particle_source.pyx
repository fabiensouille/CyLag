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
from libc.math cimport sqrt
from ..core.constants cimport INTERN_REF
from ..core.simutime cimport SimuTime
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..extra.initial_conditions import init_positions_2d, init_positions_3d
from ..extra.initial_conditions import circle
from ..lsm.random_utils cimport random_uniform
from ..geom.utils cimport polygon_area

cdef class ParticleSource:
    """ 
    ParticleSource class : adds particle during computation
    
    Parameters
    ----------
    scheduler : cylag.Scheduler, class used to trigger add particles (pset.add)
    poly : (np, 2) array fo float
        contains x, y positions of polygon shape
    xy_method : int, method of initialization, (default: 0)
        0: grid_res [number of particles along x and y axis],
        1: npart [total number of particles],
        2: density [number of particles/m^-2)]
    grid_res : (int, int, int) number of points on on x, y, z axis (default: [5, 5, 1])
    xy_npart : int, number of particles in poly (default: 25)
    xy_density : float, number of particle per unit area in poly (default: 25.)
    z_method : int, method of initialization, (default: 0)
        0: grid_res [number of particles along z axis],
        1: z_npart [number of particles layers],
        2: z_density [number of particles/m^-1)]
    z_npart : int, number of particles on vertical axis (default: 1)
    z_density : float, number of particle per unit length on vertical axis (default: 1.)    
    zmin : float, z min for vertical init
    zmax : float, z max for vertical init
    """
    def __init__(self, scheduler, poly, \
            grid_res=np.array([5, 5, 1], dtype='int32'),\
            xy_method=0, xy_npart=25, xy_density=25.,\
            z_method=0, z_npart=1, z_density=1., zmin=0., zmax=1.,):

        self.scheduler = scheduler
        self.poly = poly
        self.poly_area = polygon_area(self.poly[:, 0], self.poly[:, 1])
        self.grid_res = grid_res
        self.xy_method = xy_method
        self.xy_npart = xy_npart
        self.xy_density = xy_density
        self.z_method = z_method
        self.z_npart = z_npart
        self.z_density = z_density
        self.zmin = zmin
        self.zmax = zmax

    cpdef int release_condition(ParticleSource self, \
            SimuTime simutime, EulerianFieldSet fset):
        return self.scheduler.now(simutime)

    cpdef init_positions(ParticleSource self, EulerianFieldSet fset, int dim=-1):
        """ 
        Initialize particle positions

        Parameters
        ----------
        fset : cylag.EulerianFieldSet
            Eulerian field set
        dim : int, optional
            Dimension of the generated positions (2 or 3). Defaults to the
            field set dimension; the solver passes the particle set dimension
            so that a 3D particle set works with a 2D field set.
        """
        if dim == -1:
            dim = fset.dim

        if dim==2:
            positions = init_positions_2d(self.poly, self.poly_area,
                self.xy_method, self.grid_res, self.xy_npart, self.xy_density)
        else:
            positions = init_positions_3d(self.poly, self.poly_area,
                self.xy_method, self.grid_res, self.xy_npart, self.xy_density,
                self.z_method, self.z_npart, self.z_density, self.zmin, self.zmax)

        return positions        

cdef class ParticleSourceFromField:
    """ 
    ParticleSourceFromField class : add particles during computation from Field values.
    
    Two options are avaible val (value) and var (variation) for adding particles,
    - val: the value of the field is compared to a threshold
    - var: the value of the field variation (field[n+1]-field[n]) is compared to a threshold
    
    The following comparison options are available (see criteria parameter): 
    - eq (==), 
    - neq (!=), 
    - geq (>=),
    - leq (<=), 
    - gt (>), 
    - lt (<).
    
    Parameters
    ----------
    scheduler : cylag.Scheduler, class used to trigger add particles
    field_id_criteria : int, id of the criteria field used for adding particles
    field_id_density : int, id of the density field used for adding particles, 
        the field must contain a density which unit is [kg/m2]
    criteria : str, criteria unsed for adding particle, must be equal to 'val_operation' 
        or 'var_operation' in which operation can be replaced by eq, neq, geq, leq, gt or lt
    threshold: float, field value or variation threshold
    virtual_mass : float, virtual mass associated with particles in [kg] (default: 1.)
        If virtual_mass == 1.: npart = density*cell_area
        If virtual_mass != 1.: npart = density*cell_area/virtual_mass
    single_release : bool, if True only release particles once when criteria is met
    """
    def __init__(self, scheduler,\
            field_id_criteria, field_id_density,\
            criteria, threshold,\
            virtual_mass=1.,\
            single_release=False):

        self.scheduler = scheduler
        self.field_id_criteria = field_id_criteria
        self.field_id_density = field_id_density
        self.criteria = criteria
        self.threshold = threshold
        self.virtual_mass = virtual_mass
        self.loc = np.empty((0), dtype='int64')
        self.single_release = single_release
        self.release_rec = -1
        
    cpdef int release_condition(ParticleSourceFromField self, \
            SimuTime simutime, EulerianFieldSet fset):

        cdef:
            int rec = fset.record
            int fid = self.field_id_criteria
            double[:,:] field
            double[:] field0
            double[:] field1
            double[:] fdiff
            int nloc

        if self.scheduler.now(simutime):

            # Variation criteria: 
            # ~~~~~~~~~~~~~~~~~~
            if self.criteria.split('_')[0] == 'var':

                # get field values at neighboring time steps
                field = fset._get_field_from_id(fid)
                field0 = fset._get_field_slice(field, rec, -1)
                field1 = fset._get_field_slice(field, rec+1, -1)

                # get field variation and compare to threshold
                fdiff = np.asarray(field1) - np.asarray(field0)
                
                # nodes where criteria is met
                if self.criteria == 'var_eq':
                    self.loc = np.where(np.asarray(fdiff) == self.threshold)[0]
                elif self.criteria == 'var_neq':
                    self.loc = np.where(np.asarray(fdiff) != self.threshold)[0]
                elif self.criteria == 'var_geq':
                    self.loc = np.where(np.asarray(fdiff) >= self.threshold)[0]
                elif self.criteria == 'var_leq':
                    self.loc = np.where(np.asarray(fdiff) <= self.threshold)[0]
                elif self.criteria == 'var_gt':
                    self.loc = np.where(np.asarray(fdiff) > self.threshold)[0]
                elif self.criteria == 'var_lt':
                    self.loc = np.where(np.asarray(fdiff) < self.threshold)[0]
                else:
                    raise ValueError("Unkwown criteria")

                nloc = np.shape(self.loc)[0] # total number of nodes

                # check if release particles
                if nloc > 0:
                    if self.single_release:
                        if rec != self.release_rec:
                            self.release_rec = rec
                            return 1
                    else:
                        self.release_rec = rec
                        return 1

            # Value criteria: add particles if field_id_criteria meets condition
            # ~~~~~~~~~~~~~~
            elif self.criteria.split('_')[0] == 'val':

                # get field value
                field = fset._get_field_from_id(fid)
                field0 = fset._get_field_slice(field, rec, -1)

                # nodes where criteria is met
                if self.criteria == 'val_eq':
                    self.loc = np.where(np.asarray(field0) == self.threshold)[0]
                elif self.criteria == 'val_neq':
                    self.loc = np.where(np.asarray(field0) != self.threshold)[0]
                elif self.criteria == 'val_geq':
                    self.loc = np.where(np.asarray(field0) >= self.threshold)[0]
                elif self.criteria == 'val_leq':
                    self.loc = np.where(np.asarray(field0) <= self.threshold)[0]
                elif self.criteria == 'val_gt':
                    self.loc = np.where(np.asarray(field0) > self.threshold)[0]
                elif self.criteria == 'val_lt':
                    self.loc = np.where(np.asarray(field0) < self.threshold)[0]
                else:
                    raise ValueError("Unkwown criteria")

                nloc = np.shape(self.loc)[0] # total number of nodes

                # check if release particles 
                if nloc > 0:
                    if self.single_release:
                        if rec != self.release_rec:
                            self.release_rec = rec
                            return 1
                    else:
                        self.release_rec = rec
                        return 1

        return 0

    cpdef init_positions(ParticleSourceFromField self, EulerianFieldSet fset, int dim=-1):
        """
        Initialize particle positions
        """
        cdef:
            int nloc = np.shape(self.loc)[0]
            int i, j, k
            int rec = fset.record
            int npart_new
            double[:,:] density_field
            double[:] density_field0
            double density
            double locx, locy
            double x0, y0
            double[:,:] positions = np.zeros((0, 2), dtype='d')
            int method = 1

        if fset.dim!=2:
            raise ValueError("Dimension must be 2")

        if dim==3:
            raise ValueError(
                "ParticleSourceFromField only generates 2D positions, "
                "it cannot be used with a 3D LagrangianParticleSet")

        # get density field
        density_field = fset._get_field_from_id(self.field_id_density)
        density_field0 = fset._get_field_slice(density_field, rec, -1)
        density = 0.

        # loop on loc (where to add particle patch)
        for i in range(nloc):
            k = self.loc[i] # node id

            # check if internal node
            if fset.triangular_mesh.vertex_labels[k]==INTERN_REF:

                # compute area of the dual cell centered on node k
                dual_cell_area = 0.
                nadj = np.shape(fset.triangular_mesh.vertex_adjacency)[1]
                for j in range(nadj):
                    itri = fset.triangular_mesh.vertex_adjacency[k, j]
                    if itri != -1:
                        dual_cell_area += abs(fset.triangular_mesh.signed_area[itri])/3.

                # define the zone to add particles at node k (circular zone)
                density = density_field0[k]/self.virtual_mass
                r0 = sqrt(dual_cell_area/np.pi)  # radius of circle used to add particles

                # METHOD 0: ADD SINGLE PARTICLE AT NODE K
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if method == 0:
                    postmp = np.zeros((1, 2), dtype='d')
                    postmp[0, 0] = fset.triangular_mesh.x[k]
                    postmp[0, 1] = fset.triangular_mesh.y[k]

                # METHOD 1: ADD RANDOMLY IN CRICULAR AREA AROUND NODE K
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif method == 1:
                    # number of particles to add at node k
                    if density != 0.:
                        npart_new = density*dual_cell_area
                        npart_new = max(1, npart_new)
                        postmp = np.zeros((npart_new, 2), dtype='d')
                        x0 = fset.triangular_mesh.x[k]
                        y0 = fset.triangular_mesh.y[k]
                        for j in range(npart_new):
                            postmp_r = r0*sqrt(random_uniform())
                            postmp_a = 2.*np.pi*random_uniform()
                            postmp[j, 0] = x0 + postmp_r*np.cos(postmp_a)
                            postmp[j, 1] = y0 + postmp_r*np.sin(postmp_a)
                    else:
                        postmp = np.zeros((0, 2), dtype='d')

                # METHOD 2: ADD IN CIRCULAR AREA WITH "init_positions_2d" (not optimized)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif method == 2:
                    if density != 0.: 
                        circ = circle(
                            x0=fset.triangular_mesh.x[k],
                            y0=fset.triangular_mesh.y[k],
                            r=r0, n=180)
                        postmp = init_positions_2d(circ,\
                            poly_area=dual_cell_area,\
                            xy_method=2,\
                            xy_density=density)
                    else:
                        postmp = np.zeros((0, 2), dtype='d')

                positions = np.concatenate((positions, postmp), axis=0)

        return positions
