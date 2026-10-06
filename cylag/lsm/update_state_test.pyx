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
from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..core.constants cimport *
from ..geom.xylocalizer cimport xy_localize

cdef int update_state_test(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i):
    """
    Update kernel for the LSM-1 model, state vector: zp={xp}

    Parameters
    ----------
        simutime : cylag.SimuTime
        field_set : cylag.EulerianFieldSet
        particle_set : cylag.LagrangianParticleSet
        parameters : cylag.Parameters
    """
    cdef:
        int bnd_j = -1 # boundary edge crossed
        int time_scheme = parameters.time_scheme
        double nux = parameters.horizontal_diffusivity
        double nuz = parameters.vertical_diffusivity
        double ux, uy, uz
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]

    # Initialize
    # ~~~~~~~~~~
    pset.last_position[i, :] = pset.position[i, :]
    pset.last_velocity[i, :] = pset.velocity[i, :]

    # Update state vector
    # ~~~~~~~~~~~~~~~~~~~
    if pset.dim == 2:
        pset.position[i, 0] += pset.velocity[i, 0]*dt
        pset.position[i, 1] += pset.velocity[i, 1]*dt
    else:
        pset.position[i, 0] += pset.velocity[i, 0]*dt
        pset.position[i, 1] += pset.velocity[i, 1]*dt
        pset.position[i, 2] += pset.velocity[i, 2]*dt

    # localization
    # ~~~~~~~~~~~~
    pset.last_tri[i] = pset.tri[i]
    pset.tri[i], bnd_j = xy_localize(\
        fset.triangular_mesh, pset.last_tri[i],
        pset.last_position[i, 0],
        pset.last_position[i, 1],
        pset.position[i, 0],
        pset.position[i, 1],
        2, 0)

    if pset.dim==3:
        pset.lowerlayer[i] = fset.z_localize(\
            pset.tri[i], 
            pset.position[i, 0], 
            pset.position[i, 1],
            pset.position[i, 2])
    
    return bnd_j
