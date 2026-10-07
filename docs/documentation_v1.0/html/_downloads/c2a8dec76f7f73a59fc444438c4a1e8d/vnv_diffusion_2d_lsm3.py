# -*- coding: utf-8 -*-
"""
Homogeneous isotropic turbulence with the LSM-3
===============================================================================

Validation of the LSM-3 model and of its stochastic time integration scheme
(Direct Integrator ``difull_lsm3``, ``time_scheme=5``) on the reference test
case of a point source dispersion in a stationary homogeneous isotropic
turbulent flow, i.e. with uniform mean fields and uniform timescales.

The Monte-Carlo statistics extracted from the particle set are compared with
the exact second-order moments of the LSM-3 system, which can be derived
analytically when all mean fields and timescales are constant.

**Model equations** (constant mean fields, zero mean velocity, no external
force, 2D so that gravity does not enter the equations):

.. math::

    & dx_{p,i} = U_{p,i} dt, \\\\
    & dU_{p,i} = \\dfrac{U_{s,i}-U_{p,i}}{\\tau_p} dt, \\\\
    & dU_{s,i} = -\\dfrac{U_{s,i}}{T_L} dt + \\sigma_s dW_i,
    \\quad \\sigma_s = \\sqrt{C_0 \\varepsilon}.

where 

- :math:`x_p` is the particle position, 
- :math:`U_p` the particle velocity,
- :math:`U_s` the fluid velocity seen by the particle,
- :math:`\\tau_p=1/\mathcal{A}_1` the particle relaxation timescale,
- :math:`T_L` the Lagrangian timescale of the fluid velocity seen.

**Exact solution**: this system is linear with additive noise, hence
:math:`(x_p, U_p, U_s)` is a Gaussian process. 
With the deterministic initial
condition :math:`x_p=U_p=U_s=0` (point source at rest), the covariance matrix
is :math:`P(t) = \\sigma_s^2 \\int_0^t v(s) v(s)^T ds` with
:math:`v(s) = e^{As}(0,0,1)^T`, that is

.. math::

    & v_3(s) = \\alpha(s) = e^{-s/T_L}, \\\\
    & v_2(s) = C \\left[ \\alpha(s) - \\beta(s) \\right],
      \\quad \\beta(s) = e^{-s/\\tau_p},
      \\quad C = \\dfrac{T_L}{T_L-\\tau_p}, \\\\
    & v_1(s) = C \\left[ T_L (1-\\alpha(s)) - \\tau_p (1-\\beta(s)) \\right],

so that all the entries of :math:`P(t)` are obtained in closed form (see
:func:`lsm3_moments` below).

In the fluid particle limit :math:`\\tau_p \\ll T_L` retained here 
:math:`(U_p \\to U_s \\equiv U)`, these
moments reduce to the classical results of the simplified Langevin model
(with :math:`\\sigma_u^2 = C_0 \\varepsilon T_L/2 = U_\\alpha^2`,
where :math:`U_\\alpha` is the stationary RMS of :math:`U`):

.. math::

    & \\langle U^2 \\rangle = \\sigma_u^2 (1-e^{-2t/T_L}), \\\\
    & \\langle x_p U \\rangle = \\sigma_u^2 T_L (1-e^{-t/T_L})^2, \\\\
    & \\langle x_p^2 \\rangle = 2 \\sigma_u^2 T_L
      \\left[ t - \\frac{T_L}{2}(1-e^{-t/T_L})(3-e^{-t/T_L}) \\right],

- **Limit case 1: short-time tracer regime** : with :math:`\\tau_p \\ll t \\ll T_L`

.. math::

    \\langle U^2 \\rangle \\sim C_0 \\varepsilon t, \\quad
    \\langle x_p U \\rangle \\sim C_0 \\varepsilon t^2/2, \\quad
    \\langle x_p^2 \\rangle \\sim C_0 \\varepsilon t^3/3.


- **Limit case 2: diffusive regime** : with :math:`t \\gg T_L \\gg \\tau_p`

.. math::

    \\langle U^2 \\rangle \\to U_\\alpha^2, \\quad
    \\langle x_p U \\rangle \\to U_\\alpha^2 T_L, \\quad
    \\langle x_p^2 \\rangle \\to 2 U_\\alpha^2 T_L t .


In the following, two sets of simulations are performed:

- a **short-time tracer regime** case, with :math:`\\Delta t = 0.05 T_L` and
  :math:`t_f = 6 T_L`,
- a **diffusive** case, with :math:`\\Delta t = 200 T_L` and
  :math:`t_f = 24000 T_L`, which checks that the exponential (Direct
  Integration) scheme remains exact for arbitrarily large time steps.
  
.. note::

    Unlike the conventional ballistic regime, where particles have an initial velocity variance and 
    :math:`\\langle x_p^2 \\rangle  \\propto t^2`, 
    this example releases particles at rest, so stochastic forcing first generates their velocity 
    and the short-time dispersion follows :math:`\\langle x_p^2 \\rangle  \\propto t^3`.
    (for ballistic regime see: Balvet, G., Minier, J. P., Henry, C., Roustan, Y., & Ferrand, M. (2023). 
    A time-step-robust algorithm to compute particle trajectories in 3-D unstructured meshes
    for Lagrangian stochastic methods. Monte Carlo Methods and Applications, 29(2), 95-126).

"""
import os
from matplotlib.pyplot import ylim
import numpy as np
import matplotlib.pylab as plt
from matplotlib.tri import Triangulation

import cylag

#sphinx_gallery_thumbnail_number = 2

################################################################################
#
# Physical parameters
# -----------------------------------------------------------------------------
#
# The turbulence is homogeneous, isotropic and stationary. The turbulent
# kinetic energy and the dissipation rate are chosen so that the Lagrangian
# timescale is 1 s and the turbulent velocity scale is 1 m/s, using the CyLag
# definitions :math:`T_L = (k/eps)/(1/2 + 3*C0/4)` and :math:`\sigma_u^2 = C0*eps*T_L/2`.

C0KOLM = 1.2   # Kolmogorov constant (cylag/core/constants.pyx)
NU0 = 1.3e-6   # water kinematic viscosity (cylag/core/constants.pyx)

UALPHA = 1.0                       # turbulent velocity scale (m/s)
TL = 1.0                           # Lagrangian timescale (s)
EPS = 2.*UALPHA**2/(C0KOLM*TL)     # dissipation rate (m2/s3)
TKE = (0.5 + 0.75*C0KOLM)*TL*EPS   # turbulent kinetic energy (m2/s2)
SIG2 = C0KOLM*EPS                  # sigma_s**2 = C0*eps

# Particle properties: neutrally buoyant Stokes particles whose relaxation
# timescale taup = (rho_p/rho_f)*dp**2/(18*nu) is much smaller than T_L, so
# that particles behave as fluid particles (tracer limit).
RHOW = 1000.
RHOP = 1000.
DP = 1.e-4
TAUP = (RHOP/RHOW)*DP**2/(18.*NU0)

# Number of stochastic particles released from the point source
NPART = 20000

print(" ~~> C0*eps          = {:.4f} m2/s3".format(SIG2))
print(" ~~> dissipation     = {:.4f} m2/s3".format(EPS))
print(" ~~> turbulent energ = {:.4f} m2/s2".format(TKE))
print(" ~~> T_L             = {:.4e} s".format(TL))
print(" ~~> tau_p           = {:.4e} s (tau_p/T_L = {:.2e})".format(
    TAUP, TAUP/TL))


################################################################################
#
# Exact second-order moments of the LSM-3 system
# -----------------------------------------------------------------------------
#
def lsm3_moments(t, tl, taup, sig2):
    """
    Exact second-order moments of {xp, Up, Us} for constant mean fields.

    Parameters
    ----------
    t : ndarray
        Times at which the moments are evaluated.
    tl : float
        Lagrangian timescale of the fluid velocity seen.
    taup : float
        Particle relaxation timescale (assumed constant, Stokes regime).
    sig2 : float
        Diffusion coefficient of the Langevin equation, sigma_s**2 = C0*eps.

    Returns
    -------
    dict
        Moments <xx>, <xu>, <xs>, <uu>, <us>, <ss>, where x is the particle
        position, u the particle velocity and s the fluid velocity seen.
    """
    if abs(tl - taup) < 1.e-12*tl:
        raise ValueError("degenerate case tl == taup, not implemented")

    t = np.asarray(t, dtype='d')
    alpha = np.exp(-t/tl)
    beta = np.exp(-t/taup)
    cc = tl/(tl - taup)

    # elementary integrals over [0, t]
    i_a = tl*(1. - alpha)
    i_b = taup*(1. - beta)
    i_aa = 0.5*tl*(1. - alpha**2)
    i_bb = 0.5*taup*(1. - beta**2)
    i_ab = (tl*taup/(tl + taup))*(1. - alpha*beta)

    mom = {}
    mom['ss'] = sig2*i_aa
    mom['us'] = sig2*cc*(i_aa - i_ab)
    mom['uu'] = sig2*cc**2*(i_aa - 2.*i_ab + i_bb)
    mom['xs'] = sig2*cc*(tl*(i_a - i_aa) - taup*(i_a - i_ab))
    mom['xu'] = sig2*cc**2*((tl - taup)*(i_a - i_b)
                            - tl*i_aa + (tl + taup)*i_ab - taup*i_bb)
    mom['xx'] = sig2*cc**2*(tl**2*(t - 2.*i_a + i_aa)
                            - 2.*tl*taup*(t - i_a - i_b + i_ab)
                            + taup**2*(t - 2.*i_b + i_bb))
    return mom


################################################################################
#
# Eulerian field set
# -----------------------------------------------------------------------------
#
# A coarse triangulation of a square domain is used: the mean velocity is zero
# and the turbulent fields are uniform, so the mesh only has to be large enough
# to contain the particle cloud during the whole simulation.
#
def build_field_set(domain_size, npts=21):
    """ Uniform (zero velocity, constant k and epsilon) 2D field set """
    dx = 2.*domain_size/(npts - 1)

    # the grid is slightly shifted so that the point source located at (0, 0)
    # lies neither on a mesh node nor on a mesh edge
    xg = np.linspace(-domain_size, domain_size, npts) + 0.137*dx
    yg = np.linspace(-domain_size, domain_size, npts) + 0.271*dx
    xx, yy = np.meshgrid(xg, yg)

    meshx = np.ascontiguousarray(xx.ravel(), dtype='d')
    meshy = np.ascontiguousarray(yy.ravel(), dtype='d')

    tri = Triangulation(meshx, meshy)
    triangles = np.ascontiguousarray(tri.triangles, dtype='int32')
    cylagtri = cylag.TriangularMesh(meshx, meshy, triangles)

    nnodes = meshx.shape[0]
    elevation = np.ones((1, nnodes), dtype='d')
    velocity_x = np.zeros((1, nnodes), dtype='d')
    velocity_y = np.zeros((1, nnodes), dtype='d')

    # k-epsilon fields, required by the LSM-3 Langevin equation
    opt_fields = np.empty((2, 1, nnodes), dtype='d')
    opt_fields[0, 0, :] = TKE
    opt_fields[1, 0, :] = EPS

    fset = cylag.EulerianFieldSet(
        triangular_mesh=cylagtri,
        dim=2, nlayers=1,
        times=np.zeros(1),
        elevation=elevation,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        opt_fields=opt_fields,
        opt_fields_names=['TURBULENT ENERG.', 'DISSIPATION'],
        trifinder=tri.get_trifinder())

    return fset


################################################################################
#
# Monte-Carlo simulation
# -----------------------------------------------------------------------------
#
# All the particles are released at the origin with a zero velocity
# (``particle_velocity_init=0``), which is the initial condition used for the
# analytical solution. The moments are averaged over the particles and over the
# two directions x and y, since the turbulence is isotropic.
#
def run_case(time_step, final_time, domain_size, npart=NPART, npts=21):
    """ Run CyLag and return the Monte-Carlo moments at every time step """
    fset = build_field_set(domain_size, npts=npts)

    # point source: all the particles are released at the origin
    position = np.zeros((npart, 2), dtype='d')
    pset = cylag.LagrangianParticleSet(
        position, dim=2,
        particle_density=RHOP,
        particle_diameter=DP,
        drag_coefficient_model=1,
        added_mass_force=False)

    parameters = {
        'final_time': final_time,
        'time_step': time_step,
        'frozen_eulerian_fields': True,
        'model': 3,
        'time_scheme': 5,
        'particle_velocity_init': 0,
        'diffusion_model': 2,
        'water_density': RHOW,
        'boundary_conditions': False,
        'listing': False,
        'output_file': False,
        }

    solver = cylag.Solver(fset, pset, parameters)

    nt = solver.simutime.nt
    times = np.zeros(nt + 1)
    keys = ['xx', 'xu', 'xs', 'uu', 'us', 'ss']
    mom = {key: np.zeros(nt + 1) for key in keys}

    for ite in range(nt):
        solver.forward(time_step)
        xp, up, us = pset.get_state()
        times[ite+1] = solver.simutime.time
        mom['xx'][ite+1] = np.mean(xp*xp)
        mom['xu'][ite+1] = np.mean(xp*up)
        mom['xs'][ite+1] = np.mean(xp*us)
        mom['uu'][ite+1] = np.mean(up*up)
        mom['us'][ite+1] = np.mean(up*us)
        mom['ss'][ite+1] = np.mean(us*us)

    if pset.npart_active != npart:
        print("WARNING: {} particles left the domain".format(
            npart - pset.npart_active))

    return times, mom


################################################################################
#
# Post-processing utilities
# -----------------------------------------------------------------------------
#
# The 99% confidence intervals of the Monte-Carlo estimators are obtained from
# the Gaussian relations :math:`\text{Var}(x^2) = 2\langle x^2\rangle^2` and
# :math:`\text{Var}(xU) = \langle x^2\rangle\langle U^2\rangle + \langle xU\rangle^2`, 
# with :math:`nsample = 2*npart` independent samples (npart particles times two directions).
#
def confidence_interval(ana, npart, quantile=2.576):
    """ Half-width of the 99% confidence interval of the MC estimators """
    nsample = 2.*npart
    half = {}
    half['xx'] = quantile*np.sqrt(2./nsample)*ana['xx']
    half['uu'] = quantile*np.sqrt(2./nsample)*ana['uu']
    half['ss'] = quantile*np.sqrt(2./nsample)*ana['ss']
    half['xu'] = quantile*np.sqrt((ana['xx']*ana['uu'] + ana['xu']**2)/nsample)
    return half


def plot_case(times, mom, ana, half, fname, title, ylim2=None):
    """ Plot <xx>, <UU> and <xU> against the analytical solution """
    cylag.set_rcparams()
    colors, _, _ = cylag.get_default_color_palette()
    tstar = times/TL

    fig, axes = plt.subplots(1, 3, figsize=(15., 4.2))
    fig.suptitle(title)

    panels = [
        ('xx', UALPHA**2*TL**2, "$\\langle x_p^2 \\rangle / (U_\\alpha T_L)^2$"),
        ('uu', UALPHA**2, "$\\langle U_p^2 \\rangle / U_\\alpha^2$"),
        ('xu', UALPHA**2*TL, "$\\langle x_p U_p \\rangle / (U_\\alpha^2 T_L)$"),
        ]

    for ax, (key, scale, label) in zip(axes, panels):
        ax.plot(tstar, ana[key]/scale, c='k', lw=1.5, label="Analytical")
        ax.plot(tstar, (ana[key] + half[key])/scale, c='0.6', lw=0.8, ls='--',
                label="$99\\%$ confidence interval")
        ax.plot(tstar, (ana[key] - half[key])/scale, c='0.6', lw=0.8, ls='--')
        ax.plot(tstar, mom[key]/scale, c=colors[2], lw=0., marker='x',
                markersize=4, label="CyLag (LSM-3, DI)")
        ax.set_xlabel("$t/T_L$")
        ax.set_ylabel(label)

    if ylim2 is not None:
        axes[1].set_ylim(ylim2)

    axes[0].legend(loc='best', fontsize=8)
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(os.path.join("figs", fname), dpi=200, format="png")
    plt.show()


def print_errors(times, mom, ana, half, tstart):
    """ Print the relative error on the moments for t > tstart """
    mask = times > tstart
    print(" ~~> relative errors for t > {:.3f} s:".format(tstart))
    for key in ['xx', 'uu', 'xu', 'ss']:
        err = np.abs(mom[key][mask] - ana[key][mask])/np.abs(ana[key][mask])
        inside = np.abs(mom[key][mask] - ana[key][mask]) <= half[key][mask]
        print("     <{}> : mean = {:8.2e} ; max = {:8.2e} ; "
              "within 99% CI : {:5.1f}%".format(
                  key, np.mean(err), np.max(err), 100.*np.mean(inside)))


################################################################################
#
# Case 1 : short-time tracer regime
# -----------------------------------------------------------------------------
#
# A small time step (:math:`\Delta t = 0.05\,T_L`) is used and the simulation 
# is run over
# :math:`6\,T_L`. The particle velocity variance grows linearly from zero
# (:math:`\langle U^2\rangle \sim C_0 \varepsilon t`) before saturating at 
# :math:`U_\alpha^2`, while :math:`\langle x^2\rangle` goes from
# the short-time tracer :math:`t^3` behavior to the diffusive linear growth.
#
dt_bal = 0.05*TL
tf_bal = 6.*TL

times_bal, mom_bal = run_case(dt_bal, tf_bal, domain_size=50.)
ana_bal = lsm3_moments(times_bal, TL, TAUP, SIG2)
half_bal = confidence_interval(ana_bal, NPART)

plot_case(times_bal, mom_bal, ana_bal, half_bal,
          "cylag_lsm3_hit_shorttimetracer.png",
          "Point source dispersion in HIT - $\\Delta t = 0.05\\,T_L$")

print_errors(times_bal, mom_bal, ana_bal, half_bal, tstart=0.5*TL)


################################################################################
#
# Case 2 : diffusive regime with a very large time step
# -----------------------------------------------------------------------------
#
# The time step is now 200 times larger than the Lagrangian timescale. The
# exponential (Direct Integration) scheme must still reproduce the exact
# statistics, and in particular the dispersion coefficient
# :math:`K = \langle x^2\rangle/(2\,t) \to U_\alpha^2 T_L`.
#
dt_dif = 200.*TL
tf_dif = 24000.*TL

times_dif, mom_dif = run_case(dt_dif, tf_dif, domain_size=2000.)
ana_dif = lsm3_moments(times_dif, TL, TAUP, SIG2)
half_dif = confidence_interval(ana_dif, NPART)

plot_case(times_dif, mom_dif, ana_dif, half_dif,
          "cylag_lsm3_hit_diffusive.png",
          "Point source dispersion in HIT - $\\Delta t = 200\\,T_L$",
          ylim2=(0.98, 1.02))

print_errors(times_dif, mom_dif, ana_dif, half_dif, tstart=dt_dif)

# dispersion coefficient
kdiff = mom_dif['xx'][1:]/(2.*times_dif[1:])
print(" ~~> dispersion coefficient K = {:.4f} m2/s "
      "(theory: {:.4f} m2/s)".format(np.mean(kdiff), UALPHA**2*TL))


################################################################################
#
# Clean:
del times_bal, mom_bal, ana_bal, half_bal
del times_dif, mom_dif, ana_dif, half_dif

