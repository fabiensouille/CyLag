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
from ..core.eulerian_field_set cimport EulerianFieldSet

def convert_fset2d_from_slf_to_h5(t2d_file, bnd_file=None,\
        main_var_names=['FREE SURFACE', 'VELOCITY U', 'VELOCITY V'], 
        optional_fields=None,\
        fset_file="cylag_fset_file.h5",\
        mesh_file="cylag_mesh_file.h5"):
    """
    Convert TELEMAC result file to h5 cylag result file

    Parameters
    ----------
    t2d_file : str, file name of the telemac2d result file
    bnd_file : str, file name of the telemac2d cli file (default : None)
    main_var_names: list of str, names of the main EulerianFieldSet variables
        (i.e. elevation, velocity u, velocity v) in the telemac result file
        (default: ['ELEVATION Z', 'VELOCITY U', 'VELOCITY V', 'VELOCITY W'])
    optional_fields : List of str, names of the optional_fields to load
        (default : None)

    Outputs Files
    -------------
    fset_file : str, file name of the cylag fset file 
    mesh_file : str, file name of the cylag mesh file
    """
    cdef:
        int i, j
        EulerianFieldSet fset

    # create fset from telemac .slf and .cli files
    fset = EulerianFieldSet.from_telemac2d(t2d_file,\
        main_var_names=main_var_names,\
        bnd_file=bnd_file,\
        optional_fields=optional_fields)

    # changes opt_fields names if duplicates to avoid errors in saving h5
    if optional_fields is not None:
        for i in range(len(optional_fields)):
            for j in range(i):
                if fset.opt_fields_names[i] == optional_fields[j]:
                    fset.opt_fields_names[i] = 'NONE{}'.format(i)

    # save h5 cylag files
    fset.save_hdf5(fset_file=fset_file, mesh_file=mesh_file)

def convert_fset3d_from_slf_to_h5(t3d_file, bnd_file=None,\
        main_var_names=['ELEVATION Z', 'VELOCITY U', 'VELOCITY V', 'VELOCITY W'], 
        optional_fields=None,\
        fset_file="cylag_fset_file.h5",\
        mesh_file="cylag_mesh_file.h5"):
    """
    Convert TELEMAC result file to h5 cylag result file

    Parameters
    ----------
    t2d_file : str, file name of the telemac2d result file
    bnd_file : str, file name of the telemac2d cli file (default : None)
    main_var_names: list of str, names of the main EulerianFieldSet variables
        (i.e. elevation, velocity u, velocity v) in the telemac result file
        (default: ['ELEVATION Z', 'VELOCITY U', 'VELOCITY V', 'VELOCITY W'])
    optional_fields : List of str, names of the optional_fields to load
        (default : None)

    Outputs Files
    -------------
    fset_file : str, file name of the cylag fset file 
    mesh_file : str, file name of the cylag mesh file
    """
    cdef:
        int i, j
        EulerianFieldSet fset

    # create fset from telemac .slf and .cli files
    fset = EulerianFieldSet.from_telemac3d(t3d_file,\
        main_var_names=main_var_names,\
        bnd_file=bnd_file,\
        optional_fields=optional_fields)

    # changes opt_fields names if duplicates to avoid errors in saving h5
    if optional_fields is not None:
        for i in range(len(optional_fields)):
            for j in range(i):
                if fset.opt_fields_names[i] == optional_fields[j]:
                    fset.opt_fields_names[i] = 'NONE{}'.format(i)

    # save h5 cylag files
    fset.save_hdf5(fset_file=fset_file, mesh_file=mesh_file)
