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
import h5py
import numpy as np
import matplotlib.tri as mtri
from ..geom.zlocalizer cimport c_compute_lower_index, c_compute_lower_layer_index
from ..geom.read_cli import read_cli
from ..geom.triangular_mesh cimport TriangularMesh
from ..core.constants cimport PART_LOC_A, PART_LOC_B, PART_LOC_O, EPSILON, KAPPA, GRAV, CMU
from libc.math cimport fabs, fmax, fmin, log, sqrt, exp

cdef class EulerianFieldSet:
    """
    EulerianFieldSet class containing Eulerian mesh and fields

    Parameters
    ----------
    triangular_mesh : cylag.TriangularMesh
        Triangular mesh containing geometry and connectivity
    dim : int, optional
        Dimension of the mesh and fields (2 or 3, default: 2)
    nlayers : int, optional
        Number of layers for 3D mesh (default: 1)
    times : ndarray, shape (ntimes,), dtype float64
        Times at which fields were recorded
    elevation : ndarray, shape (ntimes, npoints*nlayers), dtype float64
        Elevation field for each point of the mesh
    velocity_x : ndarray, shape (ntimes, npoints*nlayers), dtype float64
        X-coordinate of the velocity field
    velocity_y : ndarray, shape (ntimes, npoints*nlayers), dtype float64
        Y-coordinate of the velocity field
    velocity_z : ndarray, shape (ntimes, npoints*nlayers), dtype float64, optional
        Z-coordinate of the velocity field (3D only)
    opt_fields : ndarray, shape (nopt, ntimes, npoints*nlayers), dtype float64, optional
        Optional fields (default: None)
    opt_fields_names : list of str, shape (nopt,), optional
        Names of the optional fields (default: None)
    friction_model : int, optional
        Friction model used in the Eulerian simulation (default: 1)
    friction_coefficient : double, optional
        Friction coefficient used in the Eulerian simulation (default: 40.)
    trifinder : matplotlib.tri.TriFinder, optional
        Triangulation finder for fast point localization (default: None)

    Attributes
    ----------
    triangular_mesh : cylag.TriangularMesh
        Triangular mesh
    dim : int
        Dimension of the mesh and fields (2 or 3)
    npoints : int
        Number of points in the triangulation
    nlayers : int
        Number of layers for 3D mesh
    ntimes : int
        Number of time steps
    times : ndarray, shape (ntimes,), dtype float64
        Times at which fields were recorded
    elevation : ndarray, shape (ntimes, npoints*nlayers), dtype float64
        Elevation field
    velocity_x : ndarray, shape (ntimes, npoints*nlayers), dtype float64
        X-velocity field
    velocity_y : ndarray, shape (ntimes, npoints*nlayers), dtype float64
        Y-velocity field
    velocity_z : ndarray, shape (ntimes, npoints*nlayers), dtype float64
        Z-velocity field (3D only)
    opt_fields : ndarray, shape (nopt, ntimes, npoints*nlayers), dtype float64
        Optional fields
    opt_fields_names : list of str, shape (nopt,)
        Names of optional fields
    nopt : int
        Number of optional fields
    field_names : list of str, shape (4,)
        Names of main fields
    friction_model : int
        Friction model used in the Eulerian simulation
    friction_coefficient : double
        Friction coefficient used in the Eulerian simulation
    frozen_hydro : bool
        Whether time interpolation is deactivated (single time step)
    record : int
        Current time record index
    time_interp_factor : double
        Time interpolation factor
    trifinder : matplotlib.tri.TriFinder
        Triangulation finder

    Public methods
    --------------
    from_telemac2d(file_name) : 
        Build EulerianFieldSet from Telemac2d result file
    from_telemac3d(file_name) : 
        Build EulerianFieldSet from Telemac3d result file
    from_hdf5(fset_file, mesh_file) :
        Create EulerianFieldSet from HDF5 files
    save_hdf5(fset_file, mesh_file) :
        Save EulerianFieldSet to HDF5 files
    get_field(field_id, time, layer) : 
        Get field at specified time/layer
    z_localize(k, x, y, z) :
        Return lower layer index of point in 3D mesh
    interpolate_field(field_id, point, time) :
        Interpolate field on point
    interpolate_velocity(point, time) :
        Interpolate velocity on point
    """
    def __init__(self,
                 triangular_mesh,
                 dim=2,
                 nlayers=1,
                 times=None,
                 elevation=None,
                 velocity_x=None,
                 velocity_y=None,
                 velocity_z=None,
                 opt_fields=None,
                 opt_fields_names=None,
                 trifinder=None,
                 friction_model=1,
                 friction_coefficient=40.):

        # Mesh
        self.triangular_mesh = triangular_mesh
        self.npoints = triangular_mesh.nv
        self.nlayers = nlayers
        self.dim = dim
        self.trifinder = trifinder
        
        # Time parameters
        self.times = times
        self.ntimes = len(times)
        self.record = 0
        self.time_interp_factor = 1.
        if len(self.times) == 1:
            self.frozen_hydro = True
        else:
            self.frozen_hydro = False
            # Validate times are strictly increasing and no duplicates
            for i in range(self.ntimes - 1):
                dt = times[i+1] - times[i]
                if dt <= EPSILON:
                    raise ValueError(
                        f"Invalid times array: times must be strictly increasing. "
                        f"Found times[{i}]={times[i]:.6e} and times[{i+1}]={times[i+1]:.6e} "
                        f"with dt={dt:.6e} (must be > {EPSILON:.6e}). "
                        f"Check input data for duplicate or reversed time values.")

        # Check fields dimension
        self.check_dim(elevation)
        self.check_dim(velocity_x)
        self.check_dim(velocity_y)
        if self.dim == 3:
            self.check_dim(velocity_z)
            
        # Fields
        self.elevation = elevation
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.velocity_z = velocity_z
        if self.dim == 3:
            self.check_regular_layers()
        self.field_names = [\
            'ELEVATION', 'VELOCITY U', 'VELOCITY V', 'VELOCITY W']

        if opt_fields is not None:
            assert opt_fields_names is not None
            self.nopt = len(opt_fields_names)
        else:
            self.nopt = 0
        self.opt_fields = opt_fields
        self.opt_fields_names = opt_fields_names

        # Named field IDs
        # Main fields are always at fixed positions
        self.fid_zs = 0
        self.fid_ux = 1
        self.fid_uy = 2
        self.fid_uz = 3
        # Optional field IDs: resolved by name, -1 if not loaded
        self.fid_turbeng = -1
        self.fid_dissip = -1
        self.fid_zb = -1
        self.fid_windx = -1
        self.fid_windy = -1
        if self.opt_fields_names is not None:
            _opt = list(self.opt_fields_names)
            if 'TURBULENT ENERG.' in _opt:
                self.fid_turbeng= 4 + _opt.index('TURBULENT ENERG.')
            elif 'TURBULENT ENERGY' in _opt:
                self.fid_turbeng= 4 + _opt.index('TURBULENT ENERGY')
            elif 'ENERGIE TURBUL.' in _opt:
                self.fid_turbeng= 4 + _opt.index('ENERGIE TURBUL.')
            if 'DISSIPATION' in _opt:
                self.fid_dissip = 4 + _opt.index('DISSIPATION')
            if 'BOTTOM' in _opt:
                self.fid_zb    = 4 + _opt.index('BOTTOM')
            elif 'FOND' in _opt:
                self.fid_zb    = 4 + _opt.index('FOND')
            if 'WIND X' in _opt:
                self.fid_windx = 4 + _opt.index('WIND X')
            elif 'VENT X' in _opt:
                self.fid_windx = 4 + _opt.index('VENT X')
            if 'WIND Y' in _opt:
                self.fid_windy = 4 + _opt.index('WIND Y')
            elif 'VENT Y' in _opt:
                self.fid_windy = 4 + _opt.index('VENT Y')

        # Friction parameters for pseudo 3D velocity corrections
        self.friction_model = friction_model
        self.friction_coefficient = friction_coefficient

        # temporary arrays
        self._tmp_elevation = np.empty((self.npoints*self.nlayers), dtype='d')
        self._tmp_velocity_x = np.empty((self.npoints*self.nlayers), dtype='d')
        self._tmp_velocity_y = np.empty((self.npoints*self.nlayers), dtype='d')
        self._tmp_velocity_z = np.empty((self.npoints*self.nlayers), dtype='d')
        self._tmp_z0 = np.empty((self.npoints*self.nlayers), dtype='d')
        self._tmp_u0 = np.empty((self.npoints*self.nlayers), dtype='d')
        self._tmp_v0 = np.empty((self.npoints*self.nlayers), dtype='d')
        self._tmp_w0 = np.empty((self.npoints*self.nlayers), dtype='d')

        if opt_fields is not None:
            self._tmp_opt_fields = np.empty((self.nopt, self.npoints*self.nlayers), dtype='d')
        else:
            self._tmp_opt_fields = None

        # Initialize interpolation factor cache
        self._cached_triangle = -1
        self._cached_x = 0.0
        self._cached_y = 0.0
        self._cached_f0 = 0.0
        self._cached_f1 = 0.0
        self._cached_f2 = 0.0
        self._cache_valid = False
        self._interp_done = False
        self._prev_valid = False

        self._init_tmp()

    def check_dim(self, field):
        """ 
        Check field dimension against expected shape
        
        Parameters
        ----------
        field : ndarray, shape (ntimes, npoints) or (ntimes, npoints*nlayers), dtype float64
            Field array to check
            
        Raises
        ------
        ValueError
            If field dimension doesn't match expected shape
        """
        if self.dim == 2:
            ref_shape = (self.ntimes, self.npoints)
        if self.dim == 3:
            ref_shape = (self.ntimes, self.npoints*self.nlayers)
        shape = np.shape(field)
        if shape != ref_shape:
            raise ValueError("Field have wrong dimension, expected {}, found {}"\
                             .format((self.ntimes, self.npoints), shape))

    def check_regular_layers(self, rtol=1e-3):
        """
        Check that the 3D vertical layers are regularly spaced

        Layer l must satisfy z_l = zb + l*(zs - zb)/(nlayers - 1) at every node
        at the first time record, as assumed by the vertical localization.

        Parameters
        ----------
        rtol : float, optional
            Tolerance on the layer elevation, relative to the layer thickness
            (default: 1e-3)

        Raises
        ------
        ValueError
            If vertical layers are not regularly spaced
        """
        nl = self.nlayers
        if self.dim != 3 or nl < 2:
            return
        frac = (np.arange(nl)/(nl - 1.))[:, None]
        z = np.asarray(self.elevation)[0].reshape(nl, self.npoints)
        zb = z[0]
        hl = (z[-1] - zb)/(nl - 1.)
        expected = zb[None, :] + frac*(z[-1] - zb)[None, :]
        tol = rtol*np.abs(hl) + 1e-6*(1. + np.abs(zb))
        err = np.abs(z - expected) - tol[None, :]
        if np.any(err > 0.):
            l, i = np.unravel_index(np.argmax(err), err.shape)
            raise ValueError(
                f"Vertical layers are not regularly spaced (CyLag requires "
                f"uniform sigma layers): at first time record, node {i}, layer {l}, "
                f"elevation is {z[l, i]:.6e}, expected {expected[l, i]:.6e}.")

    @staticmethod
    def from_telemac2d(file_name, 
          main_var_names=['FREE SURFACE', 'VELOCITY U', 'VELOCITY V'],
          bnd_file=None, optional_fields=None, init_trifinder=True,
          friction_model=1, friction_coefficient=40.):
        """
        Build EulerianFieldSet from telemac2d result file

        Parameters
        ----------
        file_name : str, file name of the telemac2d result file
        bnd_file : str, file name of the telemac2d cli file (default : None)
        main_var_names: list of str, names of the main EulerianFieldSet variables
            (i.e. elevation, velocity u, velocity v) in the telemac result file
            (default: ['FREE SURFACE', 'VELOCITY U', 'VELOCITY V'])
        friction_model : int, optional
            Friction model used in the Eulerian simulation (for pseudo 3D) (default: 1)
        friction_coefficient : double, optional
            Friction coefficient used in the Eulerian simulation (for pseudo 3D) (default: 40.)
        optional_fields : List of str, names of the optional_fields to load
            (default : None)
        init_trifinder : bool, initialize matplotlib triangulation trifinder
            allows quicker initialization of particles localization in trimesh
            (default : True)

        Return
        ------
        EulerianFieldSet (object)
        """
        cdef:
            int k, i
            int niter, npoin, nopt
        
        from data_manip.extraction.telemac_file import TelemacFile
        res = TelemacFile(file_name)

        # Mesh initialization
        if bnd_file is not None:
            bnd_pts, nopen = read_cli(bnd_file)
        else:
            bnd_pts = None
            nopen = 0

        triangular_mesh = TriangularMesh.from_matplotlib(\
            res.tri, boundary_points=bnd_pts, nopen=nopen)

        # init trifinder for fast search algo
        if init_trifinder:
            trifinder = res.tri.get_trifinder()
        else:
            trifinder = None

        # reading fields
        times = res.times - res.times[0]
        niter = res.ntimestep
        npoin = res.npoin2
        elev = np.empty((niter, npoin), dtype='d')
        velx = np.empty((niter, npoin), dtype='d')
        vely = np.empty((niter, npoin), dtype='d')
        for k in range(niter):
            elev[k, :] = res.get_data_value(main_var_names[0], k)
            velx[k, :] = res.get_data_value(main_var_names[1], k)
            vely[k, :] = res.get_data_value(main_var_names[2], k)
        velz = None

        if optional_fields is not None:
            nopt = len(optional_fields)
            opt_fields = np.empty((nopt, niter, npoin), dtype='d')
            for i, field_name in enumerate(optional_fields):
                for k in range(niter):
                    opt_fields[i, k, :] = res.get_data_value(field_name, k)
        else:
            opt_fields = None

        # delete TelemacFile
        del res

        return EulerianFieldSet(\
           triangular_mesh, 2, 1, times, elev, velx, vely, velz,\
           opt_fields, optional_fields, trifinder,
           friction_model=friction_model,
           friction_coefficient=friction_coefficient)

    @staticmethod
    def from_telemac3d(file_name, 
            main_var_names=['ELEVATION Z', 'VELOCITY U', 'VELOCITY V', 'VELOCITY W'],
            bnd_file=None, optional_fields=None, init_trifinder=True,
            friction_model=1, friction_coefficient=40.):
        """
        Build EulerianFieldSet from telemac3d result file

        Parameters
        ----------
        file_name : str, file name of the telemac3d result file
        bnd_file : str, file name of the telemac3d boundary cli file (default : None)
        main_var_names: list of str, names of the main EulerianFieldSet variables
            (i.e. elevation, velocity u, velocity v, velocity w) in the telemac3d result file
            (default: ['ELEVATION Z', 'VELOCITY U', 'VELOCITY V', 'VELOCITY W'])
        optional_fields : List of str, names of the optional_fields to load
            (default : None)
        init_trifinder : bool, initialize matplotlib triangulation trifinder
            allows quicker initialization of particles localization in trimesh
            (default : True)
        friction_model : int, optional
            Friction model used in the Eulerian simulation (for pseudo 3D) (default: 1)
        friction_coefficient : double, optional
            Friction coefficient used in the Eulerian simulation (for pseudo 3D) (default: 40.)

        Return
        ------
        EulerianFieldSet (object)
        """
        cdef:
            int k, i
            int niter, npoin, nopt

        from data_manip.extraction.telemac_file import TelemacFile
        res = TelemacFile(file_name)

        # Mesh initialization 
        if bnd_file is not None:
            bnd_pts, nopen = read_cli(bnd_file)
        else:
            bnd_pts = None
            nopen = 0

        triangular_mesh = TriangularMesh.from_matplotlib(\
            res.tri, boundary_points=bnd_pts, nopen=nopen)

        # init trifinder for fast search algo
        if init_trifinder:
            trifinder = res.tri.get_trifinder()
        else:
            trifinder = None

        # reading basic fields
        times = res.times - res.times[0]
        niter = res.ntimestep
        npoin = res.npoin2
        nplan = res.nplan
        elev = np.empty((niter, npoin*nplan), dtype='d')
        velx = np.empty((niter, npoin*nplan), dtype='d')
        vely = np.empty((niter, npoin*nplan), dtype='d')
        velz = np.empty((niter, npoin*nplan), dtype='d')
        for k in range(niter):
            elev[k, :] = res.get_data_value(main_var_names[0], k)
            velx[k, :] = res.get_data_value(main_var_names[1], k)
            vely[k, :] = res.get_data_value(main_var_names[2], k)
            velz[k, :] = res.get_data_value(main_var_names[3], k)

        # reading optional fields
        if optional_fields is not None:
            nopt = len(optional_fields)
            opt_fields = np.empty((nopt, niter, npoin*nplan), dtype='d')
            for i, field_name in enumerate(optional_fields):
                for k in range(niter):
                    opt_fields[i, k, :] = res.get_data_value(field_name, k)
        else:
            opt_fields = None

        # delete TelemacFile
        del res

        return EulerianFieldSet(\
            triangular_mesh, 3, nplan, times, elev, velx, vely, velz,\
            opt_fields, optional_fields, trifinder,
            friction_model=friction_model,
            friction_coefficient=friction_coefficient)

    @staticmethod
    def from_hdf5(fset_file="cylag_fset_file.h5",\
                  mesh_file="cylag_mesh_file.h5",\
                  optional_fields=None,
                  init_trifinder=True,
                  friction_model=1,
                  friction_coefficient=40.):
        """
        Create EulerianFieldSet from hdf5 cylag files
        
        Parameters
        ----------
        fset_file : str, file name of the cylag fset file
        mesh_file : str, file name of the cylag mesh file
        optional_fields : List of str, names of the optional_fields to load
            (default : None)
        init_trifinder : bool, initialize matplotlib triangulation trifinder
            allows quicker initialization of particles localization in trimesh
            (default : True)

        Return
        ------
        EulerianFieldSet (object)
        """
        cdef int i

        # Load mesh from mesh file
        triangular_mesh = TriangularMesh.from_hdf5(file_name=mesh_file)

        # Set trifinder 
        if init_trifinder:
            tri = mtri.Triangulation(\
                triangular_mesh.x,\
                triangular_mesh.y,\
                triangular_mesh.triangles)
            trifinder = tri.get_trifinder()
        else:
            trifinder = None

        # Load EulerianFieldSet params from fset file
        with h5py.File(fset_file, 'r') as f:
            # Mesh
            dim = f['dim'][:][0]
            nplan = f['nplan'][:][0]
            # Times
            times = f['times'][:]
            # Fields
            elev = f['elevation'][:]
            velx = f['velocity_x'][:]
            vely = f['velocity_y'][:]
            if dim==3:
                velz = f['velocity_z'][:]
            # Opt Fields
            nopt = f['nopt'][:][0]
            if nopt != 0:
                # names saved in the file, unless provided by the user
                if optional_fields is None:
                    if 'opt_fields_names' in f:
                        optional_fields = [n.decode('utf-8') if isinstance(n, bytes) else str(n)
                                           for n in f['opt_fields_names'][:]]
                    else:
                        optional_fields = ["opt_field_{}".format(i) for i in range(nopt)]
                opt_fields = f['opt_fields'][:]
            else:
                optional_fields = None
                opt_fields = None
        f.close()

        if dim==2:
            return EulerianFieldSet(\
               triangular_mesh, 2, 1, times, elev, velx, vely, None,\
               opt_fields, optional_fields, trifinder,
               friction_model=friction_model,
               friction_coefficient=friction_coefficient)
        elif dim==3:
            return EulerianFieldSet(\
                triangular_mesh, 3, nplan, times, elev, velx, vely, velz,\
                opt_fields, optional_fields, trifinder,
                friction_model=friction_model,
                friction_coefficient=friction_coefficient)
        else:
            raise ValueError("Wrong dimension in fset_file")

    def save_hdf5(EulerianFieldSet self,\
            fset_file="cylag_fset_file.h5",\
            mesh_file="cylag_mesh_file.h5"):
        """
        Save EulerianFieldSet in hdf5
        
        Parameters
        ----------
        fset_file : str, file name of the cylag fset file
        mesh_file : str, file name of the cylag mesh file
        """
        cdef int i

        # Save mesh in mesh file
        self.triangular_mesh.save_mesh_hdf5(mesh_file)

        # Save EulerianFieldSet in fset file
        with h5py.File(fset_file, 'w') as f:
            # Mesh
            f.create_dataset('dim', data=np.array([self.dim]))
            f.create_dataset('nplan', data=np.array([self.nlayers]))
            # Times
            f.create_dataset('times', data=np.array(self.times))
            # Fields
            f.create_dataset('elevation', data=np.array(self.elevation))
            f.create_dataset('velocity_x', data=np.array(self.velocity_x))
            f.create_dataset('velocity_y', data=np.array(self.velocity_y))
            if self.dim==3:
                f.create_dataset('velocity_z', data=np.array(self.velocity_z))
            # Opt Fields
            f.create_dataset('nopt', data=np.array([self.nopt]))
            if self.nopt != 0:
                f.create_dataset('opt_fields', data=np.array(self.opt_fields))
                f.create_dataset('opt_fields_names',
                    data=[str(n).encode('utf-8') for n in self.opt_fields_names],
                    dtype=h5py.string_dtype())
        f.close()

    def get_mtri_triangulation(self):
        """
        Returns matplotlib triangulation (for post-processing only)
        """
        return mtri.Triangulation(
            self.triangular_mesh.x,\
            self.triangular_mesh.y,\
            self.triangular_mesh.triangles)

    def get_field_list(self):
        """
        Returns list of field names (for post-processing only)
        """
        field_list = self.field_names + list(self.opt_fields_names)
        field_id_list = list(range(len(field_list)))
        return field_list, field_id_list

    cpdef double[:] _get_field_slice(EulerianFieldSet self, double[:,:] field, int k, int l):
        """ 
        Get field array slice at specified time and layer indices
        
        Parameters
        ----------
        field : ndarray, shape (ntimes, npoints*nlayers), dtype float64
            Field array to slice
        k : int
            Time index
        l : int
            Layer index (-1 for all layers)
            
        Returns
        -------
        ndarray, shape (npoints,) or (npoints*nlayers,), dtype float64
            Sliced field array
        """
        if self.dim == 2:
            return field[k, 0:self.npoints]
        else:
            if l != -1:
                return field[k, l*self.npoints:(l+1)*self.npoints]
            else:
                return field[k, :]
            
    cpdef double[:] _get_tmp_slice(EulerianFieldSet self, double[:] field, int l):
        """ 
        Get temporary array slice at specified layer index
        
        Parameters
        ----------
        field : ndarray, shape (npoints*nlayers,), dtype float64
            Temporary field array
        l : int
            Layer index (-1 for all layers)
            
        Returns
        -------
        ndarray, shape (npoints,) or (npoints*nlayers,), dtype float64
            Sliced temporary array
        """
        if self.dim == 2:
            return field
        else:
            if l != -1:
                return field[l*self.npoints:(l+1)*self.npoints]
            else:
                return field[:]

    cpdef double[:,:] _get_field_from_id(EulerianFieldSet self, int field_id):
        """ 
        Get field array from field ID
        
        Parameters
        ----------
        field_id : int
            Field identifier (0=elevation, 1=vel_x, 2=vel_y, 3=vel_z, 4+=opt_fields)
            
        Returns
        -------
        ndarray, shape (ntimes, npoints*nlayers), dtype float64
            Field array
            
        Raises
        ------
        ValueError
            If field_id is invalid
        """
        cdef:
            double[:,:] field
        if field_id == 0:
            field = self.elevation
        elif field_id == 1:
            field = self.velocity_x
        elif field_id == 2:
            field = self.velocity_y
        elif field_id == 3:
            field = self.velocity_z
        elif field_id > 3 and field_id <= 3 + self.nopt:
            field = self.opt_fields[field_id-4, :, :]
        else:
            raise ValueError("Wrong field id")
        return field

    cpdef double[:] _get_tmp_from_id(EulerianFieldSet self, int field_id):
        """ 
        Get temporary array from field ID
        
        Parameters
        ----------
        field_id : int
            Field identifier (0=elevation, 1=vel_x, 2=vel_y, 3=vel_z, 4+=opt_fields)
            
        Returns
        -------
        ndarray, shape (npoints*nlayers,), dtype float64
            Temporary field array
            
        Raises
        ------
        ValueError
            If field_id is invalid
        """
        cdef:
            double[:] field
        if field_id == 0:
            field = self._tmp_elevation
        elif field_id == 1:
            field = self._tmp_velocity_x
        elif field_id == 2:
            field = self._tmp_velocity_y
        elif field_id == 3:
            field = self._tmp_velocity_z
        elif field_id > 3 and field_id <= 3 + self.nopt:
            field = self._tmp_opt_fields[field_id-4, :]
        else:
            raise ValueError("Wrong field id")
        return field

    cpdef void _init_tmp(EulerianFieldSet self):
        """ 
        Initialize temporary arrays with current time step data
        
        Copies field data from the current record to temporary arrays
        for interpolation operations.
        """
        cdef:
            int size = self.nlayers*self.npoints
            int record = 0
            double[:,:] field
            double[:] field0
            double[:] field_tmp
            double[:] aux = np.empty((size), dtype='d')

        # initialize fields
        for field_id in range(self.dim+1):
            field = self._get_field_from_id(field_id)
            aux = self._get_field_slice(field, record, -1)
            field_tmp = self._get_tmp_from_id(field_id)
            field_tmp[:] = aux[:]

        # initialize opt fields
        for field_id in range(4, 4+self.nopt):
            field = self._get_field_from_id(field_id)
            aux = self._get_field_slice(field, record, -1)
            field_tmp = self._get_tmp_from_id(field_id)
            field_tmp[:] = aux[:]
            
        # initialize previous arrays
        self._tmp_z0[:] = self._tmp_elevation[:]
        self._tmp_u0[:] = self._tmp_velocity_x[:]
        self._tmp_v0[:] = self._tmp_velocity_y[:]
        self._tmp_w0[:] = self._tmp_velocity_z[:]

    cpdef void _cache_interp_factors(EulerianFieldSet self, int k, double x, double y):
        """ 
        Cache barycentric interpolation factors for reuse
        
        Parameters
        ----------
        k : int
            Triangle index
        x, y : double
            Point coordinates
        """
        cdef:
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
        
        self._cached_triangle = k
        self._cached_x = x
        self._cached_y = y
        self._cached_f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
        self._cached_f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
        self._cached_f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]
        self._cache_valid = True

    cpdef bint _use_cached_factors(EulerianFieldSet self, int k, double x, double y):
        """ 
        Check if cached interpolation factors can be reused
        
        Parameters
        ----------
        k : int
            Triangle index
        x, y : double
            Point coordinates
            
        Returns
        -------
        bool
            True if cache is valid and matches current point
        """
        return (self._cache_valid and 
                self._cached_triangle == k and 
                abs(self._cached_x - x) < EPSILON and 
                abs(self._cached_y - y) < EPSILON)

    cpdef double[:] _get_tmp(EulerianFieldSet self, int field_id, int l):
        """ 
        Get temporary array for field at specified layer
        
        Parameters
        ----------
        field_id : int
            Field identifier
        l : int
            Layer index
            
        Returns
        -------
        ndarray, shape (npoints,) or (npoints*nlayers,), dtype float64
            Temporary field array at specified layer
        """
        cdef:
            double[:] tmp
        tmp = self._get_tmp_from_id(field_id)
        tmp = self._get_tmp_slice(tmp, l)
        return tmp

    cpdef void _set_record(EulerianFieldSet self, double time, bint follow):
        """ 
        Set time record index from time value
        
        Parameters
        ----------
        time : double
            Time value
        follow : bool
            If True, search from current record; if False, search from beginning
        """
        if self.frozen_hydro==0:
            # search record in times
            if follow==0:
                self.record = c_compute_lower_index(self.times, time, 0)
            # search in times from last record instead of all records
            else:
                self.record = c_compute_lower_index(self.times, time, self.record)
        else:
            self.record = 0

    cpdef void _time_interpolation(EulerianFieldSet self, double time):
        """ 
        Compute time interpolation of all fields and store in temporary arrays
        
        Parameters
        ----------
        time : double
            Time at which to interpolate
        """
        cdef:
            int ntmp
            int size = self.nlayers*self.npoints
            int nl = self.nlayers
            int rec
            double time_fact
            double[:] times = self.times
            double[:,:] field
            double[:] field0
            double[:] field1
            double[:] field_tmp

        # number of tmp fields
        if self.dim==2:
            ntmp = 3
        else:
            ntmp = 4

        # only interpolate if not frozen hydro
        if self.frozen_hydro==0:

            # saves previous velocity for acceleration computation
            self._tmp_z0[:] = self._tmp_elevation[:]
            self._tmp_u0[:] = self._tmp_velocity_x[:]
            self._tmp_v0[:] = self._tmp_velocity_y[:]
            self._tmp_w0[:] = self._tmp_velocity_z[:]

            # previous fields are only meaningful from the second interpolation on
            self._prev_valid = self._interp_done
            self._interp_done = True

            # time interp factor                
            self._set_record(time, 1)
            rec = self.record
            dt = times[rec] - times[rec+1]
            if abs(dt) < EPSILON:
                raise ValueError(
                    f"Time interpolation failed: duplicate or too-close time values detected "
                    f"at records {rec} and {rec+1} (times[{rec}]={times[rec]:.6e}, "
                    f"times[{rec+1}]={times[rec+1]:.6e}, dt={dt:.6e}). "
                    f"Check input data for time consistency.")
            time_fact = (time-times[rec+1])/dt

            # time interpolation of main fields
            for field_id in range(ntmp):
                # get field values at neighboring time steps
                field = self._get_field_from_id(field_id)
                field0 = self._get_field_slice(field, rec, -1)
                field1 = self._get_field_slice(field, rec+1, -1)
                # interpolate and store in temporary array
                field_tmp = self._get_tmp_from_id(field_id)
                for i in range(size):
                    field_tmp[i] = time_fact*field0[i] + (1.-time_fact)*field1[i]

            # time interpolation of optionnal fields
            for field_id in range(4, 4+self.nopt):
                # get field values at neighboring time steps
                field = self._get_field_from_id(field_id)
                field0 = self._get_field_slice(field, rec, -1)
                field1 = self._get_field_slice(field, rec+1, -1)
                # interpolate and store in temporary array
                field_tmp = self._get_tmp_from_id(field_id)
                for i in range(size):
                    field_tmp[i] = time_fact*field0[i] + (1.-time_fact)*field1[i]

    cpdef void _time_interpolation_field(EulerianFieldSet self, int field_id, double time):
        """ 
        Compute time interpolation of a specific field and store in temporary array
        
        Parameters
        ----------
        field_id : int
            Field identifier to interpolate
        time : double
            Time at which to interpolate
            
        Raises
        ------
        ValueError
            If field_id is invalid for 2D case
        """
        cdef:
            int size = self.nlayers*self.npoints
            int nl = self.nlayers
            int rec = self.record
            double time_fact
            double[:] times = self.times
            double[:,:] field
            double[:] field0
            double[:] field1
            double[:] field_tmp

        if self.dim==2:
            if field_id == 3:
                raise ValueError("Wrong field id")

        # only interpolate if not frozen hydro
        if self.frozen_hydro==0:

            # time interp factor
            dt = times[rec] - times[rec+1]
            if abs(dt) < EPSILON:
                raise ValueError(
                    f"Time interpolation failed: duplicate or too-close time values detected "
                    f"at records {rec} and {rec+1} (times[{rec}]={times[rec]:.6e}, "
                    f"times[{rec+1}]={times[rec+1]:.6e}, dt={dt:.6e}). "
                    f"Check input data for time consistency.")
            time_fact = (time-times[rec+1])/dt

            # get field values at neighboring time steps
            field = self._get_field_from_id(field_id)
            field0 = self._get_field_slice(field, rec, -1)
            field1 = self._get_field_slice(field, rec+1, -1)

            # interpolate and store in temporary array
            field_tmp = self._get_tmp_from_id(field_id)
            for i in range(size):
                field_tmp[i] = time_fact*field0[i] + (1.-time_fact)*field1[i]

    cpdef double _interpolate_field_2d(EulerianFieldSet self,\
            int field_id, int k, int l, double x, double y):
        """
        Interpolate field in 2D
        
        Parameters
        ----------
        field_id : int
            Index of field to interpolate
        k : int
            Index of triangle containing point
        l : int
            Layer index (set to -1 in 2D)
        x, y : double
            Point coordinates
            
        Returns
        -------
        double
            Interpolated field value at point
            
        Raises
        ------
        ValueError
            If point is not in triangulation
        """
        cdef:
            int i0, i1, i2
            double f0, f1, f2
            double[:] field
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
            int[:,:] triangles = self.triangular_mesh.triangles

        # force layer to -1 in 2D
        if self.dim == 2:
            field = self._get_tmp_from_id(field_id)
        else:
            field = self._get_tmp(field_id, l)

        if k == -1:
            raise ValueError("Failed interpolation (0), point not in triangulation")

        # Use cached factors if available, otherwise compute and cache
        if self._use_cached_factors(k, x, y):
            f0 = self._cached_f0
            f1 = self._cached_f1
            f2 = self._cached_f2
        else:
            f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
            f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
            f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]
            self._cache_interp_factors(k, x, y)

        # interpolate field on point
        i0 = triangles[k, 0]
        i1 = triangles[k, 1]
        i2 = triangles[k, 2]
        field_on_point = field[i0]*f0 + field[i1]*f1 + field[i2]*f2

        return field_on_point

    cpdef (double, double) _interpolate_velocity_2d(\
            EulerianFieldSet self, int k,\
            double x, double y):
        """ 
        Interpolate velocity field in 2D
        
        Parameters
        ----------
        k : int
            Index of triangle containing point
        x, y : double
            Point coordinates
            
        Returns
        -------
        tuple of double
            Interpolated velocity components (u, v) at point
            
        Raises
        ------
        ValueError
            If point is not in triangulation
        """
        cdef:
            int i0, i1, i2
            double f0, f1, f2
            double[:] u, v
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
            int[:,:] triangles = self.triangular_mesh.triangles
            double u_on_points
            double v_on_points

        if k == -1:
            raise ValueError("Failed interpolation (1), point not in triangulation")

        u = self._get_tmp_from_id(self.fid_ux)
        v = self._get_tmp_from_id(self.fid_uy)
        
        # Use cached factors if available, otherwise compute and cache
        if self._use_cached_factors(k, x, y):
            f0 = self._cached_f0
            f1 = self._cached_f1
            f2 = self._cached_f2
        else:
            f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
            f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
            f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]
            self._cache_interp_factors(k, x, y)

        # interpolate field on point
        i0 = triangles[k, 0]
        i1 = triangles[k, 1]
        i2 = triangles[k, 2]
        u_on_points = u[i0]*f0 + u[i1]*f1 + u[i2]*f2
        v_on_points = v[i0]*f0 + v[i1]*f1 + v[i2]*f2

        return u_on_points, v_on_points

    cpdef int z_localize(EulerianFieldSet self, int k,\
            double x, double y, double z):
        """
        Locate point vertically in 3D mesh

        Parameters
        ----------
        k : int
            Index of triangle containing point (x,y)
        x, y, z : double
            Point coordinates
            
        Returns
        -------
        int
            Lower layer index containing point, or PART_LOC_B/A if outside
        """
        cdef:
            int nl = self.nlayers
            int i0, i1, i2
            int lowerlayer
            double f0, f1, f2
            double zbp, zsp
            double[:] zb, zs
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
            int[:,:] triangles = self.triangular_mesh.triangles

        # point outside the triangulation: no vertical localization
        if k == -1:
            return PART_LOC_O

        # precompute interpolation factors
        i0 = triangles[k, 0]
        i1 = triangles[k, 1]
        i2 = triangles[k, 2]
        f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
        f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
        f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]

        # Computation of zb and zs at point P
        if self.dim == 2:
            zb = self._get_tmp(self.fid_zb, -1)
            zs = self._get_tmp(self.fid_zs, -1)
        else:
            zb = self._get_tmp(self.fid_zs, 0)
            zs = self._get_tmp(self.fid_zs, nl-1)

        zbp = zb[i0]*f0 + zb[i1]*f1 + zb[i2]*f2
        zsp = zs[i0]*f0 + zs[i1]*f1 + zs[i2]*f2

        # Computation of lower layer index
        if self.dim == 2:
            lowerlayer = 0
        else:
            lowerlayer = c_compute_lower_layer_index(nl, zsp, zbp, z)

        # localize point on z axis
        # (check if points are above surface or below bottom)
        if z < zbp:
            lowerlayer = PART_LOC_B
        elif z > zsp:
            lowerlayer = PART_LOC_A

        return lowerlayer

    cpdef double _interpolate_field_3d(EulerianFieldSet self,\
            int field_id, int k, int lowerlayer, bint zlocalize,
            double x, double y, double z):
        """ 
        Interpolate field in 3D
        
        Parameters
        ----------
        field_id : int
            Index of field to interpolate
        k : int
            Index of triangle containing point
        lowerlayer : int
            Index of lower layer
        zlocalize : bool
            Whether to localize point vertically
        x, y, z : double
            Point coordinates
            
        Returns
        -------
        double
            Interpolated field value at point
            
        Raises
        ------
        ValueError
            If point is outside domain
        """
        cdef:
            int nl = self.nlayers
            int i0, i1, i2
            int ju, jl
            double f0, f1, f2, zfactors0, zfactors1
            double zbp, zsp, zlp, zup, resl, resu
            double[:] zb, zs
            double[:] elev, field
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
            int[:,:] triangles = self.triangular_mesh.triangles

        if k == -1:
            raise ValueError("Failed interpolation (2), point not in triangulation")

        # precompute interpolation factors
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        i0 = triangles[k, 0]
        i1 = triangles[k, 1]
        i2 = triangles[k, 2]
        
        # compute interpolation factors.
        f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
        f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
        f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]

        # localize point on z axis
        if zlocalize == 1:
            # point needs to be vertically localized
            # Computation of zb and zs at point P
            zb = self._get_tmp(self.fid_zs, 0)
            zs = self._get_tmp(self.fid_zs, nl-1)
            zbp = zb[i0]*f0 + zb[i1]*f1 + zb[i2]*f2
            zsp = zs[i0]*f0 + zs[i1]*f1 + zs[i2]*f2
            # Computation of lower layer index
            lowerlayer = c_compute_lower_layer_index(nl, zsp, zbp, z)
        
        # (check if points are above surface or below bottom)
        if lowerlayer == PART_LOC_B:
            raise ValueError("Failed interpolation, point is below bottom")

        elif  lowerlayer == PART_LOC_A:
            raise ValueError("Failed interpolation, point is above surface")

        # interpolate field on point
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        elev = self._get_tmp_from_id(self.fid_zs)
        field = self._get_tmp_from_id(field_id)

        # interpolate elevation on lower and upper layers 
        jl = lowerlayer*self.npoints
        ju = (lowerlayer+1)*self.npoints
        zlp = elev[jl + i0]*f0 + elev[jl + i1]*f1 + elev[jl + i2]*f2
        zup = elev[ju + i0]*f0 + elev[ju + i1]*f1 + elev[ju + i2]*f2

        # precompute 1d vertical interp factors
        dz = zlp - zup
        if abs(dz) < EPSILON:
            raise ValueError(
                f"3D interpolation failed: degenerate vertical layer detected "
                f"at triangle {k}, layer {lowerlayer}. Lower elevation (zlp={zlp:.6e}) "
                f"equals upper elevation (zup={zup:.6e}), dz={dz:.6e}. "
                f"Check mesh vertical structure and elevation field consistency.")
        zf0 = (z - zup)/dz
        zf1 = (zlp - z)/dz

        # interpolate data on lower and upper layers
        resl = field[jl + i0]*f0 + field[jl + i1]*f1 + field[jl + i2]*f2
        resu = field[ju + i0]*f0 + field[ju + i1]*f1 + field[ju + i2]*f2

        # vertical interpolation
        field_on_point = zf0*resl + zf1*resu

        return field_on_point

    cpdef (double, double, double) _interpolate_velocity_3d(\
            EulerianFieldSet self,\
            int k, int lowerlayer, bint zlocalize,\
            double x, double y, double z):
        """ 
        Interpolate velocity field in 3D
        
        Parameters
        ----------
        k : int
            Index of triangle containing point
        lowerlayer : int
            Index of lower layer
        zlocalize : bool
            Whether to localize point vertically
        x, y, z : double
            Point coordinates
            
        Returns
        -------
        tuple of double
            Interpolated velocity components (u, v, w) at point
            
        Raises
        ------
        ValueError
            If point is outside domain
        """
        cdef:
            int nl = self.nlayers
            int i0, i1, i2
            int ju, jl
            double f0, f1, f2, zf0, zf1
            double zbp, zsp, zlp, zup, resl, resu
            double[:] zb, zs
            double[:] elev, u, v, w
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
            int[:,:] triangles = self.triangular_mesh.triangles
            
        if k == -1:
            raise ValueError("Failed interpolation (3), point not in triangulation")

        # precompute interpolation factors
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        i0 = triangles[k, 0]
        i1 = triangles[k, 1]
        i2 = triangles[k, 2]
        
        # compute interpolation factors.
        f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
        f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
        f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]

        # localize point on z axis
        if zlocalize == 1:
            # point needs to be vertically localized
            # Computation of zb and zs at point P
            zb = self._get_tmp(self.fid_zs, 0)
            zs = self._get_tmp(self.fid_zs, nl-1)
            zbp = zb[i0]*f0 + zb[i1]*f1 + zb[i2]*f2
            zsp = zs[i0]*f0 + zs[i1]*f1 + zs[i2]*f2
            # Computation of lower layer index
            lowerlayer = c_compute_lower_layer_index(nl, zsp, zbp, z)
        
        # (check if points are above surface or below bottom)
        if lowerlayer == PART_LOC_B:
            raise ValueError("Failed interpolation (3b), point is below bottom")

        elif  lowerlayer == PART_LOC_A:
            raise ValueError("Failed interpolation (3c), point is above surface")

        # interpolate field on point
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        elev = self._get_tmp_from_id(self.fid_zs)
        u = self._get_tmp_from_id(self.fid_ux)
        v = self._get_tmp_from_id(self.fid_uy)
        w = self._get_tmp_from_id(self.fid_uz)

        # interpolate elevation on lower and upper layers 
        jl = lowerlayer*self.npoints
        ju = (lowerlayer+1)*self.npoints
        zlp = elev[jl + i0]*f0 + elev[jl + i1]*f1 + elev[jl + i2]*f2
        zup = elev[ju + i0]*f0 + elev[ju + i1]*f1 + elev[ju + i2]*f2

        # precompute 1d vertical interp factors
        dz = zlp - zup
        if abs(dz) < EPSILON:
            raise ValueError(
                f"3D velocity interpolation failed: degenerate vertical layer detected "
                f"at triangle {k}, layer {lowerlayer}. Lower elevation (zlp={zlp:.6e}) "
                f"equals upper elevation (zup={zup:.6e}), dz={dz:.6e}. "
                f"Check mesh vertical structure and elevation field consistency.")
        zf0 = (z - zup)/dz
        zf1 = (zlp - z)/dz

        # interpolate data 
        u_on_points = zf0*(u[jl + i0]*f0 + u[jl + i1]*f1 + u[jl + i2]*f2)\
                    + zf1*(u[ju + i0]*f0 + u[ju + i1]*f1 + u[ju + i2]*f2)
        v_on_points = zf0*(v[jl + i0]*f0 + v[jl + i1]*f1 + v[jl + i2]*f2)\
                    + zf1*(v[ju + i0]*f0 + v[ju + i1]*f1 + v[ju + i2]*f2)
        w_on_points = zf0*(w[jl + i0]*f0 + w[jl + i1]*f1 + w[jl + i2]*f2)\
                    + zf1*(w[ju + i0]*f0 + w[ju + i1]*f1 + w[ju + i2]*f2)

        return u_on_points, v_on_points, w_on_points

    cpdef (double, double) _interpolate_acceleration_2d(\
            EulerianFieldSet self, int k,\
            double x, double y, double dt,
            double ua, double va):
        """ 
        Interpolate the derivative of the mean fluid velocity along a path in 2D

        Returns dU/dt = dU/dt|_x + (Ua . grad) U, where the Eulerian time
        derivative is zero for frozen fields, and the gradient is constant per
        P1 triangle. Ua = U gives the fluid material derivative, Ua = Up the
        derivative along the particle trajectory.
        
        Parameters
        ----------
        k : int
            Index of triangle containing point
        x, y : double
            Point coordinates
        dt : double
            Time step
        ua, va : double
            Velocity components advecting the field (convective term)
            
        Returns
        -------
        tuple of double
            Acceleration components (ax, ay) at point
            
        Raises
        ------
        ValueError
            If point is not in triangulation
        """
        cdef:
            int i0, i1, i2
            double f0, f1, f2
            double[:] u = self._tmp_velocity_x
            double[:] v = self._tmp_velocity_y
            double[:] u0 = self._tmp_u0
            double[:] v0 = self._tmp_v0
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
            int[:,:] triangles = self.triangular_mesh.triangles
            double dudx, dudy, dvdx, dvdy
            double dudt = 0., dvdt = 0.

        if k == -1:
            raise ValueError("Failed interpolation (6), point not in triangulation")

        i0 = triangles[k, 0]
        i1 = triangles[k, 1]
        i2 = triangles[k, 2]

        # convective term
        dudx = u[i0]*gradx[k, 0] + u[i1]*gradx[k, 1] + u[i2]*gradx[k, 2]
        dudy = u[i0]*grady[k, 0] + u[i1]*grady[k, 1] + u[i2]*grady[k, 2]
        dvdx = v[i0]*gradx[k, 0] + v[i1]*gradx[k, 1] + v[i2]*gradx[k, 2]
        dvdy = v[i0]*grady[k, 0] + v[i1]*grady[k, 1] + v[i2]*grady[k, 2]

        # Eulerian time derivative, unavailable at first interpolation or if frozen hydro
        if self.frozen_hydro==0 and self._prev_valid:
            f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
            f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
            f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]
            dudt = ((u[i0] - u0[i0])*f0 + (u[i1] - u0[i1])*f1 + (u[i2] - u0[i2])*f2)/dt
            dvdt = ((v[i0] - v0[i0])*f0 + (v[i1] - v0[i1])*f1 + (v[i2] - v0[i2])*f2)/dt

        return dudt + ua*dudx + va*dudy, dvdt + ua*dvdx + va*dvdy

    cpdef (double, double, double) _interpolate_acceleration_3d(\
            EulerianFieldSet self,\
            int k, int lowerlayer,\
            double x, double y, double z, double dt,
            double ua, double va, double wa):
        """ 
        Interpolate the derivative of the mean fluid velocity along a path in 3D

        Returns dU/dt = dU/dt|_x + (Ua . grad) U, where the Eulerian time
        derivative is zero for frozen fields. Horizontal derivatives are taken
        at fixed elevation z, i.e. they include the slope of the layer surfaces.
        Ua = U gives the fluid material derivative, Ua = Up the derivative
        along the particle trajectory.
        
        Parameters
        ----------
        k : int
            Index of triangle containing point
        lowerlayer : int
            Index of lower layer
        x, y, z : double
            Point coordinates
        dt : double
            Time step
        ua, va, wa : double
            Velocity components advecting the field (convective term)
            
        Returns
        -------
        tuple of double
            Acceleration components (ax, ay, az) at point
            
        Raises
        ------
        ValueError
            If point is not in triangulation
        """
        cdef:
            int i0, i1, i2
            int ju, jl
            double f0, f1, f2, s, dz
            double zlp, zup, zlx, zly, zux, zuy
            double[:] elev
            double[:] u = self._tmp_velocity_x
            double[:] v = self._tmp_velocity_y
            double[:] w = self._tmp_velocity_z
            double[:] u0 = self._tmp_u0
            double[:] v0 = self._tmp_v0
            double[:] w0 = self._tmp_w0
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
            int[:,:] triangles = self.triangular_mesh.triangles
            double un, dudx, dudy, dudz
            double vn, dvdx, dvdy, dvdz
            double wn, dwdx, dwdy, dwdz
            double dudt = 0., dvdt = 0., dwdt = 0.

        if k == -1:
            raise ValueError("Failed interpolation (7), point not in triangulation")

        i0 = triangles[k, 0]
        i1 = triangles[k, 1]
        i2 = triangles[k, 2]
        f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
        f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
        f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]

        # layer elevations and slopes at point
        elev = self._get_tmp_from_id(self.fid_zs)
        jl = lowerlayer*self.npoints
        ju = (lowerlayer+1)*self.npoints
        zlp = elev[jl + i0]*f0 + elev[jl + i1]*f1 + elev[jl + i2]*f2
        zup = elev[ju + i0]*f0 + elev[ju + i1]*f1 + elev[ju + i2]*f2
        dz = zup - zlp
        if abs(dz) < EPSILON:
            raise ValueError(
                f"3D acceleration interpolation failed: degenerate vertical layer detected "
                f"at triangle {k}, layer {lowerlayer}. Lower elevation (zlp={zlp:.6e}) "
                f"equals upper elevation (zup={zup:.6e}), dz={dz:.6e}. "
                f"Check mesh vertical structure and elevation field consistency.")

        s = (z - zlp)/dz
        zlx = elev[jl + i0]*gradx[k, 0] + elev[jl + i1]*gradx[k, 1] + elev[jl + i2]*gradx[k, 2]
        zly = elev[jl + i0]*grady[k, 0] + elev[jl + i1]*grady[k, 1] + elev[jl + i2]*grady[k, 2]
        zux = elev[ju + i0]*gradx[k, 0] + elev[ju + i1]*gradx[k, 1] + elev[ju + i2]*gradx[k, 2]
        zuy = elev[ju + i0]*grady[k, 0] + elev[ju + i1]*grady[k, 1] + elev[ju + i2]*grady[k, 2]

        # convective term
        un, dudx, dudy, dudz = _prism_value_gradient(\
            u, jl, ju, i0, i1, i2, k, f0, f1, f2,\
            gradx, grady, s, dz, zlx, zly, zux, zuy)
        vn, dvdx, dvdy, dvdz = _prism_value_gradient(\
            v, jl, ju, i0, i1, i2, k, f0, f1, f2,\
            gradx, grady, s, dz, zlx, zly, zux, zuy)
        wn, dwdx, dwdy, dwdz = _prism_value_gradient(\
            w, jl, ju, i0, i1, i2, k, f0, f1, f2,\
            gradx, grady, s, dz, zlx, zly, zux, zuy)

        # Eulerian time derivative, unavailable at initial time or if frozen hydro
        if self.frozen_hydro==0 and self._prev_valid:
            dudt = (un - _prism_value_gradient(\
                u0, jl, ju, i0, i1, i2, k, f0, f1, f2,\
                gradx, grady, s, dz, zlx, zly, zux, zuy)[0])/dt
            dvdt = (vn - _prism_value_gradient(\
                v0, jl, ju, i0, i1, i2, k, f0, f1, f2,\
                gradx, grady, s, dz, zlx, zly, zux, zuy)[0])/dt
            dwdt = (wn - _prism_value_gradient(\
                w0, jl, ju, i0, i1, i2, k, f0, f1, f2,\
                gradx, grady, s, dz, zlx, zly, zux, zuy)[0])/dt

        return (dudt + ua*dudx + va*dudy + wa*dudz,
                dvdt + ua*dvdx + va*dvdy + wa*dvdz,
                dwdt + ua*dwdx + va*dwdy + wa*dwdz)

    cpdef double _interpolate_surface_velocity(\
            EulerianFieldSet self, int k,\
            double x, double y, double dt,
            double zsp):
        """ 
        Interpolate free surface velocity (dzs/dt) 
        
        Parameters
        ----------
        k : int
            Index of triangle containing point
        x, y : double
            Point coordinates
        dt : double
            Time step
        zsp : double
            Current surface elevation at point
            
        Returns
        -------
        tuple of double
            Interpolated elevation evolution components (dzs/dt) at point
            
        Raises
        ------
        ValueError
            If point is not in triangulation
        """
        cdef:
            int i0, i1, i2
            double f0, f1, f2
            double[:] z0 = self._tmp_z0
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
            int[:,:] triangles = self.triangular_mesh.triangles
            double z0p, dzsdt_on_points

        if k == -1:
            raise ValueError("Failed interpolation (6), point not in triangulation")

        # cannot compute acceleration at first interpolation or if frozen hydro
        if self.frozen_hydro==1 or not self._prev_valid:
            dzsdt_on_points = 0.

        else:
            # precompute interpolation factors
            f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
            f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
            f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]

            # interpolate velocities u^{n-1} on point
            i0 = triangles[k, 0]
            i1 = triangles[k, 1]
            i2 = triangles[k, 2]

            z0p = z0[i0]*f0 + z0[i1]*f1 + z0[i2]*f2

            # approximate elevation evolution
            dzsdt_on_points = (zsp - z0p)/dt

        return dzsdt_on_points

    cdef (double, double, double, double) _interpolate_diffusivity_gradient_c(\
            EulerianFieldSet self, int k, int lowerlayer,\
            double x, double y, double z, double sigc):
        """
        Interpolate the k-epsilon diffusivity K = (Cmu/sigc)*k^2/epsilon and its gradient

        K is evaluated at the mesh nodes (epsilon floored at EPSILON) and then
        interpolated with the P1 (2D) or prism (3D) basis, so that the stochastic
        amplitude sqrt(2K) and the drift grad(K) derive from the same discrete field.
        In 3D the horizontal derivatives are taken at fixed elevation z, i.e. they
        include the slope of the layer surfaces. In 2D (including pseudo-3D) dK/dz = 0.

        Parameters
        ----------
        k : int
            Index of triangle containing point
        lowerlayer : int
            Index of lower layer (3D only)
        x, y, z : double
            Point coordinates (z only used in 3D)
        sigc : double
            Schmidt number

        Returns
        -------
        tuple of double
            (K, dK/dx, dK/dy, dK/dz) at point

        Raises
        ------
        ValueError
            If point is outside domain or turbulence fields are not loaded
        """
        cdef:
            int i0, i1, i2
            int jl, ju
            double fac = CMU/sigc
            double f0, f1, f2, s, dz
            double kn0, kn1, kn2
            double Kp, dKdx, dKdy, dKdz = 0.
            double Kl, Ku, gxl, gyl, gxu, gyu
            double zlx, zly, zux, zuy, zlp, zup
            double[:] kt, ep, elev
            double[:,:] det = self.triangular_mesh.det
            double[:,:] gradx = self.triangular_mesh.gradx
            double[:,:] grady = self.triangular_mesh.grady
            int[:,:] triangles = self.triangular_mesh.triangles

        if k == -1:
            raise ValueError("Failed diffusivity interpolation, point not in triangulation")
        if self.fid_turbeng == -1 or self.fid_dissip == -1:
            raise ValueError(
                "Diffusivity gradient requires turbulent energy and dissipation fields")

        i0 = triangles[k, 0]
        i1 = triangles[k, 1]
        i2 = triangles[k, 2]
        f0 = det[k, 0] + x*gradx[k, 0] + y*grady[k, 0]
        f1 = det[k, 1] + x*gradx[k, 1] + y*grady[k, 1]
        f2 = det[k, 2] + x*gradx[k, 2] + y*grady[k, 2]

        kt = self._get_tmp_from_id(self.fid_turbeng)
        ep = self._get_tmp_from_id(self.fid_dissip)

        if self.dim == 2:
            kn0 = _k_eps_diffusivity(kt[i0], ep[i0], fac)
            kn1 = _k_eps_diffusivity(kt[i1], ep[i1], fac)
            kn2 = _k_eps_diffusivity(kt[i2], ep[i2], fac)
            Kp = kn0*f0 + kn1*f1 + kn2*f2
            dKdx = kn0*gradx[k, 0] + kn1*gradx[k, 1] + kn2*gradx[k, 2]
            dKdy = kn0*grady[k, 0] + kn1*grady[k, 1] + kn2*grady[k, 2]
            return fmax(0., Kp), dKdx, dKdy, 0.

        if lowerlayer == PART_LOC_B:
            raise ValueError("Failed diffusivity interpolation, point is below bottom")
        elif lowerlayer == PART_LOC_A:
            raise ValueError("Failed diffusivity interpolation, point is above surface")

        jl = lowerlayer*self.npoints
        ju = (lowerlayer + 1)*self.npoints
        elev = self._get_tmp_from_id(self.fid_zs)

        # lower layer: nodal K, value and horizontal gradient
        kn0 = _k_eps_diffusivity(kt[jl + i0], ep[jl + i0], fac)
        kn1 = _k_eps_diffusivity(kt[jl + i1], ep[jl + i1], fac)
        kn2 = _k_eps_diffusivity(kt[jl + i2], ep[jl + i2], fac)
        Kl = kn0*f0 + kn1*f1 + kn2*f2
        gxl = kn0*gradx[k, 0] + kn1*gradx[k, 1] + kn2*gradx[k, 2]
        gyl = kn0*grady[k, 0] + kn1*grady[k, 1] + kn2*grady[k, 2]
        zlp = elev[jl + i0]*f0 + elev[jl + i1]*f1 + elev[jl + i2]*f2
        zlx = elev[jl + i0]*gradx[k, 0] + elev[jl + i1]*gradx[k, 1] + elev[jl + i2]*gradx[k, 2]
        zly = elev[jl + i0]*grady[k, 0] + elev[jl + i1]*grady[k, 1] + elev[jl + i2]*grady[k, 2]

        # upper layer
        kn0 = _k_eps_diffusivity(kt[ju + i0], ep[ju + i0], fac)
        kn1 = _k_eps_diffusivity(kt[ju + i1], ep[ju + i1], fac)
        kn2 = _k_eps_diffusivity(kt[ju + i2], ep[ju + i2], fac)
        Ku = kn0*f0 + kn1*f1 + kn2*f2
        gxu = kn0*gradx[k, 0] + kn1*gradx[k, 1] + kn2*gradx[k, 2]
        gyu = kn0*grady[k, 0] + kn1*grady[k, 1] + kn2*grady[k, 2]
        zup = elev[ju + i0]*f0 + elev[ju + i1]*f1 + elev[ju + i2]*f2
        zux = elev[ju + i0]*gradx[k, 0] + elev[ju + i1]*gradx[k, 1] + elev[ju + i2]*gradx[k, 2]
        zuy = elev[ju + i0]*grady[k, 0] + elev[ju + i1]*grady[k, 1] + elev[ju + i2]*grady[k, 2]

        dz = zup - zlp
        if fabs(dz) < EPSILON:
            raise ValueError(
                f"3D diffusivity interpolation failed: degenerate vertical layer detected "
                f"at triangle {k}, layer {lowerlayer}. Lower elevation (zlp={zlp:.6e}) "
                f"equals upper elevation (zup={zup:.6e}), dz={dz:.6e}. "
                f"Check mesh vertical structure and elevation field consistency.")

        # K(x,y,z) = Kl + s*(Ku - Kl) with s = (z - zl(x,y))/(zu(x,y) - zl(x,y))
        s = (z - zlp)/dz
        Kp = (1. - s)*Kl + s*Ku
        dKdz = (Ku - Kl)/dz
        dKdx = (1. - s)*gxl + s*gxu - dKdz*((1. - s)*zlx + s*zux)
        dKdy = (1. - s)*gyl + s*gyu - dKdz*((1. - s)*zly + s*zuy)

        return fmax(0., Kp), dKdx, dKdy, dKdz

    def _interpolate_diffusivity_gradient(self, int k, int lowerlayer,
            double x, double y, double z, double sigc):
        return self._interpolate_diffusivity_gradient_c(
            k, lowerlayer, x, y, z, sigc)

    def get_field(self, field_id, time, layer=-1):
        """
        Get field at specified time and layer

        WARNING: DO NOT CALL INSIDE MAIN TIME LOOP, PRE-POST ONLY
        If time is not in recorded times, linear interpolation is performed.

        Parameters
        ----------
        field_id : int
            Index of the field to extract, ordered as self.field_names,
            for optional fields start at index = 4
        time : float
            Extraction time
        layer : int, optional
            Layer to extract (only for 3D, default: -1)
            
        Returns
        -------
        ndarray, shape (npoints,), dtype float64
            Array containing field values
        """
        self._set_record(time, 0)
        self._time_interpolation_field(field_id, time)
        res = self._get_tmp(field_id, layer)
        return np.asarray(res)

    def interpolate_velocity(self, point, time, k=None):
        """
        Interpolate velocity field at a point
        
        WARNING: DO NOT CALL INSIDE MAIN TIME LOOP, PRE-POST ONLY
        
        Parameters
        ----------
        point : ndarray, shape (2,) or (3,), dtype float64
            Coordinates of the point (2D or 3D)
        time : float
            Time at which to interpolate
        k : int, optional
            Triangle index containing point (if None, will be computed)
            
        Returns
        -------
        tuple of double
            Interpolated velocity components (u, v) for 2D or (u, v, w) for 3D
            
        Raises
        ------
        ValueError
            If point is not in triangulation
        """
        # time interpolation
        self._set_record(time, 0)
        self._time_interpolation(time)

        # localize point
        if k is None:
            k = self.trifinder(point[0], point[1])
        if k ==-1:
            raise ValueError("Failed interpolation (4), point not in triangulation")

        # interpolate
        if self.dim == 2:
            ux, uy = self._interpolate_velocity_2d(\
                k, point[0], point[1])
            return ux, uy
        else:
            ux, uy, uz = self._interpolate_velocity_3d(\
                k, 0, 1, point[0], point[1], point[2])
            return ux, uy, uz

    def interpolate_field(self, field_id, point, time, k=None):
        """
        Interpolate field at a point
        
        WARNING: DO NOT CALL INSIDE MAIN TIME LOOP, PRE-POST ONLY
        
        Parameters
        ----------
        field_id : int
            Index of field to interpolate
        point : ndarray, shape (2,) or (3,), dtype float64
            Coordinates of the point (2D or 3D)
        time : float
            Time at which to interpolate
        k : int, optional
            Triangle index containing point (if None, will be computed)
            
        Returns
        -------
        field_on_point : double
            Interpolated field value at point
            
        Raises
        ------
        ValueError
            If point is not in triangulation
        """
        # time interpolation
        self._set_record(time, 0)
        self._time_interpolation_field(field_id, time)
        
        # localize point
        if k is None:
            k = self.trifinder(point[0], point[1])
        if k ==-1:
            raise ValueError("Failed interpolation (5), point not in triangulation")

        if self.dim == 2:
            field_on_point = self._interpolate_field_2d(\
                field_id, k, -1, point[0], point[1])
        else:
            field_on_point = self._interpolate_field_3d(\
                field_id, k, 0, 1, point[0], point[1], point[2])
        return field_on_point

    cpdef (double, double) pseudo_3d_velocity_corrections(EulerianFieldSet self,
            int k, double x, double y, double z, double dt, double ux, double uy):
        """
        Compute the vertical and horizontal velocity corrections for pseudo-3D simulations

        The vertical velocity keeps the relative elevation sigma=(z-zb)/h of
        fluid particles advected by the reconstructed log-law velocity Uc:
        w = (1-sigma)*Dzb/Dt + sigma*Dzs/Dt, with D/Dt = d/dt + Uc.grad.

        Parameters
        ----------
        k : int
            Index of the triangle containing the particle
        x : double
            x-coordinate of the particle
        y : double
            y-coordinate of the particle
        z : double
            z-coordinate of the particle
        dt : double
            Time step for the velocity correction
        ux : double
            Depth-averaged fluid velocity along x at the particle position
        uy : double
            Depth-averaged fluid velocity along y at the particle position

        Returns
        -------
        (double, double)
            (vertical velocity, horizontal correction factor)
        """
        cdef:
            double[:] zs_n = self._get_tmp(self.fid_zs, -1)
            double[:] zb_n = self._get_tmp(self.fid_zb, -1)
            double[:,:] gx = self.triangular_mesh.gradx
            double[:,:] gy = self.triangular_mesh.grady
            int[:,:] tri = self.triangular_mesh.triangles
            int i0 = tri[k, 0], i1 = tri[k, 1], i2 = tri[k, 2]
            double zs, zb, h, sigma, Cf, z0, hor, uc, vc
            double dzsdx, dzsdy, dzbdx, dzbdy, dzsdt, w

        zs = self._interpolate_field_2d(self.fid_zs, k, -1, x, y)
        zb = self._interpolate_field_2d(self.fid_zb, k, -1, x, y)
        h = fmax(EPSILON, zs - zb)
        sigma = fmin(1., fmax(0., (z - zb)/h))

        # horizontal log-law correction
        Cf = self.friction_calc(h)
        z0 = fmax(EPSILON, h*exp(-(1. + KAPPA/sqrt(Cf/2.))))
        if z-zb <= z0:
            hor = 0.0
        else:
            hor = log((z - zb)/z0)/(log(h/z0) - 1.)
        uc = ux*hor
        vc = uy*hor

        # P1 gradients, constant per triangle
        dzsdx = zs_n[i0]*gx[k,0] + zs_n[i1]*gx[k,1] + zs_n[i2]*gx[k,2]
        dzsdy = zs_n[i0]*gy[k,0] + zs_n[i1]*gy[k,1] + zs_n[i2]*gy[k,2]
        dzbdx = zb_n[i0]*gx[k,0] + zb_n[i1]*gx[k,1] + zb_n[i2]*gx[k,2]
        dzbdy = zb_n[i0]*gy[k,0] + zb_n[i1]*gy[k,1] + zb_n[i2]*gy[k,2]

        # vertical velocity correction based on surface velocity and horizontal gradients
        # note : dzb/dt is assumed to be zero, true for pure hydrodynamics, 
        #        but could be important for morphodynamics
        dzsdt = self._interpolate_surface_velocity(k, x, y, dt, zs)
        w = (1. - sigma)*(uc*dzbdx + vc*dzbdy) \
          + sigma*(dzsdt+ uc*dzsdx + vc*dzsdy)
        return w, hor

    cpdef double friction_calc(EulerianFieldSet self, double h):
        """
        Compute friction coefficient based on water depth and friction model

        Parameters
        ----------
        h : double
            Water depth

        Returns
        -------
        Cf : double
            Friction coefficient based on the selected friction model
        """
        cdef:
            double frict_mod = self.friction_model
            double frict_coef = self.friction_coefficient

        # Define surface velocity coefficient from friction law
        if frict_mod==0:
            # Chezy
            Cf = 2.*GRAV/frict_coef**2
        elif frict_mod==1:
            # Strikler
            Cf = 2.*GRAV/(frict_coef**2)/(fmax(EPSILON, h)**(1./3.))
        elif frict_mod==2:
            # Manning
            Cf = 2.*GRAV*(frict_coef**2)/(fmax(EPSILON, h)**(1./3.))
        elif frict_mod==3:
            # Nikuradse
            aux = 30.*h/(frict_coef*exp(1.))
            aux = max(1.001, h*11.036/frict_coef)
            Cf = 2./(log(aux)/KAPPA)**2
        else:
            raise ValueError("Invalid friction model")

        return Cf

cdef inline double _k_eps_diffusivity(double k, double eps, double fac) noexcept:
    """ Nodal k-epsilon diffusivity K = (Cmu/sigc)*k^2/eps with floored eps """
    return fac*k*k/fmax(eps, EPSILON)

cdef inline (double, double, double, double) _prism_value_gradient(\
        double[:] fld, int jl, int ju, int i0, int i1, int i2, int k,
        double f0, double f1, double f2,
        double[:,:] gradx, double[:,:] grady,
        double s, double dz, double zlx, double zly, double zux, double zuy) noexcept:
    """
    Value and gradient of a nodal prism field at relative height s in [0, 1]

    F = (1-s)*Fl + s*Fu with s = (z - zl(x,y))/dz and dz = zu - zl. The
    horizontal derivatives are taken at fixed z (layer-surface slopes zl*, zu*).
    """
    cdef:
        double fl, fu, gxl, gyl, gxu, gyu, dfdz

    fl = fld[jl + i0]*f0 + fld[jl + i1]*f1 + fld[jl + i2]*f2
    fu = fld[ju + i0]*f0 + fld[ju + i1]*f1 + fld[ju + i2]*f2
    gxl = fld[jl + i0]*gradx[k, 0] + fld[jl + i1]*gradx[k, 1] + fld[jl + i2]*gradx[k, 2]
    gyl = fld[jl + i0]*grady[k, 0] + fld[jl + i1]*grady[k, 1] + fld[jl + i2]*grady[k, 2]
    gxu = fld[ju + i0]*gradx[k, 0] + fld[ju + i1]*gradx[k, 1] + fld[ju + i2]*gradx[k, 2]
    gyu = fld[ju + i0]*grady[k, 0] + fld[ju + i1]*grady[k, 1] + fld[ju + i2]*grady[k, 2]
    dfdz = (fu - fl)/dz
    return ((1. - s)*fl + s*fu,
            (1. - s)*gxl + s*gxu - dfdz*((1. - s)*zlx + s*zux),
            (1. - s)*gyl + s*gyu - dfdz*((1. - s)*zly + s*zuy),
            dfdz)