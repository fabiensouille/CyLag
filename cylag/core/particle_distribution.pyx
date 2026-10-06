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
from ..core.constants cimport MIN_PARTICLE_DIAMETER, MAX_PARTICLE_DIAMETER
from libc.math cimport tgamma, log

# ==================================================================
# Use a tabulated distribution for particle diameters - user defined
# ==================================================================
# description: contains a list of tuples (diameter, CDF value)
cdef double[5][2] TABULATED_PARTICLE_DISTRIBUTION = [
    [0.5, 0.1],
    [1.0, 0.2],
    [2.0, 0.3],
    [3.0, 0.5],
    [4.0, 1.0]
]

# ==================================================================
# Particle diameter distribution functions
# ==================================================================
cpdef void check_particle_distribution(double mean_diameter, str dist, double param):
    """
    Check the validity of the particle diameter distribution parameters.

    Parameters
    ----------
    mean_diameter : float
        Mean particle diameter.
    dist : str
        Distribution type
    param : float
        Distribution parameter
    """
    if dist == 'uniform':
        if mean_diameter-param < 0.:
            raise ValueError(f"Uniform : mean_diameter-param must be positive, "\
                              "got {}-{}".format(mean_diameter, param))
        if param <= 0.:
            raise ValueError(f"Uniform : param must be positive, got {param}")

    elif dist == 'normal':
        if mean_diameter <= 0.:
            raise ValueError(f"Normal : mean_diameter must be positive, got {mean_diameter}")
        if param <= 0.:
            raise ValueError(f"Normal : param (std) must be positive, got {param}")

    elif dist == 'lognormal':
        if mean_diameter <= 0.:
            raise ValueError(f"Lognormal : mean_diameter must be positive, got {mean_diameter}")
        if param < 1.:
            raise ValueError(f"Lognormal : param (GSD) must be >= 1, got {param}")

    elif dist == 'weibull':
        if mean_diameter <= 0.:
            raise ValueError(f"Weibull : mean_diameter must be positive, got {mean_diameter}")
        if param < 1:
            raise ValueError(f"Weibull : param (shape) must be >= 1, got {param}")

    elif dist == 'tabulated':
        dp_tab = TABULATED_PARTICLE_DISTRIBUTION[:, 0]
        cdf_tab = TABULATED_PARTICLE_DISTRIBUTION[:, 1]

        if not np.all(np.diff(dp_tab) > 0):
            raise ValueError("Tabulated : diameters must be strictly increasing")
        if not np.all(np.diff(cdf_tab) >= 0):
            raise ValueError("Tabulated : CDF values must be non-decreasing")

    else:
        raise ValueError(f"Unknown particle diameter distribution: {dist}")

cpdef double draw_particle_diameter(double mean_diameter, str dist, double param):
    """
    Draw a particle diameter from a specified distribution and
    rejects out-of-bound values

    Parameters
    ----------
    mean_diameter : float
        Mean particle diameter.
    dist : str
        Distribution type: 'uniform', 'normal', 'lognormal', 'weibull'
    param : float
        Distribution parameter:
        - 'uniform': 
            param is delta, uniform in [mean-diameter-delta, mean-diameter+delta]
        - 'normal':
            param is std, normal with mean=mean_diameter, std=std
        - 'lognormal': 
            mean_diameter is the arithmetic number-mean diameter E[D].
            param is the geometric standard deviation, GSD >= 1.
            Internally: sigma_log = log(GSD) / mu_log = log(mean_diameter) - sigma_log**2 / 2
        - 'weibull': 
            mean_diameter is the arithmetic number-mean diameter E[D].
            param is shape, Weibull with scale=mean_diameter/tgamma(1 + 1/shape)

    Returns
    -------
    float
        Drawn particle diameter.
    """
    cdef:
        double diameter
        double mu, scale
        double[:] dp_tab, cdf_tab
        int max_attempts = 100

    # Mono-disperse 
    # ~~~~~~~~~~~~~
    if dist is None or dist == 'none' or dist == 'monodisperse':
        return mean_diameter

    # Poly-disperse
    # ~~~~~~~~~~~~~
    else:
        # rejection sampling to avoid out of bound diameter
        for attempt in range(max_attempts):
            if dist == 'uniform':
                diameter = mean_diameter + (np.random.uniform(0.0, 1.0)*2.0 - 1.0)*param

            elif dist == 'normal':
                diameter = np.random.normal(loc=mean_diameter, scale=param)

            elif dist == 'lognormal':
                mu = log(mean_diameter) - 0.5 * (log(param)**2)
                diameter = np.random.lognormal(mean=mu, sigma=log(param))

            elif dist == 'weibull':
                scale = mean_diameter/tgamma(1.0 + 1.0 / param)
                diameter = scale*(-log(1.0 - np.random.uniform(0.0, 1.0)))**(1.0/param)

            elif dist == 'tabulated':
                dp_tab = TABULATED_PARTICLE_DISTRIBUTION[:, 0]
                cdf_tab = TABULATED_PARTICLE_DISTRIBUTION[:, 1]
                diameter = np.interp(np.random.random(), cdf_tab, dp_tab)
            else:
                raise ValueError(f"Unknown particle diameter distribution: {dist}")
                
            if MIN_PARTICLE_DIAMETER <= diameter <= MAX_PARTICLE_DIAMETER:
                return diameter

        raise RuntimeError(
            f"Unable to draw an admissible diameter from distribution '{dist}' "
            f"after {max_attempts} attempts. "
            f"mean_diameter={mean_diameter}, param={param}, "
            f"bounds=[{MIN_PARTICLE_DIAMETER}, {MAX_PARTICLE_DIAMETER}]")