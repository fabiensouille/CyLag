# -*- coding: utf-8 -*-
"""
Homogeneous isotropic turbulence with the LSM-2
===============================================================================

Validation of the LSM-2 model with diffusion on particle velocity (
``diffusion_lsm2_option=2``) and its stochastic time integration scheme
(Direct Integrator ``difull_lsm2``, ``time_scheme=5``) on the reference test
case of a point source dispersion in a uniform zero-mean flow.

The Monte-Carlo statistics are compared with exact second-order moments derived
analytically from the linear SDE system.

**Model equations** (zero mean flow, no gravity, 2D):

.. math::

    & dx_{p,i} = U_{p,i}\\,dt, \\qquad \\\\
    & dU_{p,i} = -\\frac{U_{p,i}}{\\tau_p}\\,dt + B\\,dW_i,

where :math:`\\tau_p = 1/\\mathcal{A}_1` is the particle relaxation time and
:math:`B` is the stochastic diffusion coefficient set by
``param:horizontal_diffusivity``.

**Exact analytical solution** (zero initial conditions):

By the Fubini–Itô theorem, the solutions can be written as stochastic
integrals:

.. math::

    U_{p,i}(t) &= B \\int_0^t e^{-(t-s)/\\tau_p}\\,dW_i(s), \\\\
    x_{p,i}(t) &= B \\int_0^t \\tau_p\\bigl(1 - e^{-(t-s)/\\tau_p}\\bigr)\\,dW_i(s).

Applying the Itô isometry gives the three non-zero second-order moments:

.. math::

    \\langle U_p^2 \\rangle &= \\sigma_u^2 (1-\\beta^2), \\\\
    \\langle x_p U_p \\rangle &= \\sigma_u^2 \\tau_p (1-\\beta)^2, \\\\
    \\langle x_p^2 \\rangle &= 2 \\sigma_u^2 \\tau_p
      \\left[t - 2\\tau_p(1-\\beta) + \\frac{\\tau_p}{2}(1-\\beta^2)\\right],

where :math:`\\sigma_u^2 = B^2 \\tau_p /2` and :math:`\\beta = e^{-t/\\tau_p}`.

- **Limit case 1: short-time regime** : with :math:`t \\ll \\tau_p`

.. math::
  
  \\langle U_p^2\\rangle \\sim B^2 t,  \\quad
  \\langle x_pU_p\\rangle \\sim B^2 t^2/2,  \\quad
  \\langle x_p^2\\rangle \\sim B^2 t^3/3.
  
- **Limit case 2: diffusive regime** : with :math:`t \\gg \\tau_p`

.. math::

  \\langle U_p^2\\rangle \\to \\sigma_u^2,  \\quad
  \\langle x_pU_p\\rangle \\to \\sigma_u^2\\tau_p,  \\quad
  \\langle x_p^2\\rangle \\to 2K\\,t 
  
with :math:`K = \\sigma_u^2 \\tau_p = B^2 \\tau_p^2 /2`.

.. note::

  **why** :math:`\\tau_p` **is constant (Stokes regime)**

  In the Stokes regime (:math:`\\text{Re} = d_p|U_p|/\\nu \\ll 1`), the Schiller
  drag gives :math:`C_d = 24/\\text{Re}`, so
  :math:`1/ \\tau_p = \\mathcal{A}_1 = 18\\rho_f\\nu/(\\rho_p d_p^2) = \\text{const}`.
  The Stokes condition is :math:`\\text{Re} = d_p\\sigma_u/\\nu \\ll 1`, i.e.
  :math:`B \\ll \\nu/(d_p\\sqrt{\\tau_p/2})`.


In the following, two sets of simulations are performed:

- a **short-time** case, with :math:`\\Delta t = 0.05\\,\\tau_p` and
  :math:`t_f = 6\\,\\tau_p`,
- a **diffusive** case, with :math:`\\Delta t = 200\\,\\tau_p` and
  :math:`t_f = 24000\\,\\tau_p`, which checks that the Direct Integration
  scheme reproduces the correct dispersion coefficient
  :math:`K = \\sigma_u^2\\tau_p` even for very large time steps.

"""
import os
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
# Particle parameters: neutrally buoyant Stokes particle.
# In the Stokes regime (:math:`Re << 1`), the Schiller drag gives
# :math:`\tau_p = const`, independent of the velocity.

NU0 = 1.3e-6    # kinematic viscosity of water (cylag/core/constants.pyx)

RHOW = 1000.    # water density (kg/m3)
RHOP = 1000.    # particle density (kg/m3), neutrally buoyant
DP = 1.e-4      # particle diameter (m)

# Stokes relaxation timescale (valid for Re << 1)
TAUP = RHOP*DP**2/(18.*RHOW*NU0)

# Velocity stochastic diffusion coefficient B (m/s^(3/2)):
# in the SDE dU_p = ... + B*dW, B*dW has units m/s.
# In CyLag, this is set via horizontal_diffusivity (for diffusion_lsm2_option=2).
# B must be small enough that Re = dp*sigma_u/nu << 1 (Stokes criterion).
# In Stokes regime (Re << 1), the Schiller drag gives a1 = 18*rho_f*nu/(rho_p*dp**2)
# (the |Up - Uf| factor cancels), so tau_p stays constant throughout the simulation.
# With B = 0.001: Re ~ 0.001, Schiller correction to tau_p < 0.05% (well within MC noise).
B = 0.001

# b2tau = B^2 * taup (key parameter entering all moments)
B2TAU = TAUP * B**2

# Steady-state velocity variance: sigma_u^2 = b2tau/2
SIGMA_U = np.sqrt(B2TAU/2.)

# Check Stokes regime: Re = dp*sigma_u/nu must be << 1 for taup = const
RE_STOKES = DP*SIGMA_U/NU0

# Diffusion coefficient of xp in the diffusive limit: K = sigma_u^2 * taup
K_DIFF = SIGMA_U**2*TAUP

# Number of stochastic particles released from the point source
NPART = 20000

print(" ~~> tau_p   = {:.4e} s".format(TAUP))
print(" ~~> B       = {:.4f} m/s^(3/2)".format(B))
print(" ~~> b2tau   = {:.4e} m^2/s^2".format(B2TAU))
print(" ~~> sigma_u = {:.4e} m/s  (sqrt(b2tau/2))".format(SIGMA_U))
print(" ~~> Re_Stokes = {:.4f}  (must be << 1 for taup=const)".format(RE_STOKES))
print(" ~~> K_diff  = {:.4e} m^2/s  (= sigma_u^2 * taup)".format(K_DIFF))


################################################################################
#
# Exact second-order moments of the LSM-2 system
# -----------------------------------------------------------------------------
#
def lsm2_moments(t, taup, b2tau):
    """
    Exact second-order moments of (xp, Up) for constant tau_p and B.

    System (zero mean flow, zero initial conditions):

        dx_p = U_p dt,
        dU_p = -(U_p/tau_p) dt + B dW,   b2tau = B^2 * tau_p.

    Parameters
    ----------
    t : ndarray
        Times at which the moments are evaluated.
    taup : float
        Particle relaxation timescale (Stokes, constant).
    b2tau : float
        Diffusion parameter B^2 * tau_p.

    Returns
    -------
    dict with keys 'uu', 'xu', 'xx':
        Second-order moments <Up^2>, <xp*Up>, <xp^2>.
    """
    t = np.asarray(t, dtype='d')
    beta = np.exp(-t/taup)
    mom = {}
    mom['uu'] = 0.5*b2tau*(1. - beta**2)
    mom['xu'] = 0.5*b2tau*taup*(1. - beta)**2
    mom['xx'] = b2tau*taup*(t - 2.*taup*(1. - beta) + 0.5*taup*(1. - beta**2))
    return mom


################################################################################
#
# Eulerian field set
# -----------------------------------------------------------------------------
#
# A coarse triangulation of a square domain with zero mean velocity.
# No turbulent fields (k, epsilon) are required because diffusion_model=1
# (constant diffusivity) is used.
#
def build_field_set(domain_size, npts=21):
    """ Zero-velocity 2D field set (no k-eps fields needed) """
    dx = 2.*domain_size/(npts - 1)

    # the point source at (0, 0) lies neither on a node nor on an edge
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

    fset = cylag.EulerianFieldSet(
        triangular_mesh=cylagtri,
        dim=2, nlayers=1,
        times=np.zeros(1),
        elevation=elevation,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        trifinder=tri.get_trifinder())

    return fset


################################################################################
#
# Monte-Carlo simulation
# -----------------------------------------------------------------------------
#
# All particles are released at the origin with Up=0 (particle_velocity_init=0),
# matching the zero initial conditions of the analytical solution.
# The moments are averaged over particles and both directions (isotropic diffusion).
#
def run_case(time_step, final_time, domain_size, npart=NPART, npts=21):
    """ Run CyLag LSM-2 and return Monte-Carlo moments at every time step """
    fset = build_field_set(domain_size, npts=npts)

    position = np.zeros((npart, 2), dtype='d')
    pset = cylag.LagrangianParticleSet(
        position, dim=2,
        particle_density=RHOP,
        particle_diameter=DP,
        drag_coefficient_model=1,  # Schiller: constant a1 in Stokes regime
        added_mass_force=False)

    parameters = {
        'final_time': final_time,
        'time_step': time_step,
        'frozen_eulerian_fields': True,
        'model': 2,
        'time_scheme': 5,              # Direct Integrator (difull_lsm2)
        'particle_velocity_init': 0,   # Up(0) = 0
        'diffusion_model': 1,          # constant diffusivity
        'diffusion_lsm2_option': 2,    # diffusion on Up with B = horizontal_diffusivity
        'horizontal_diffusivity': B,   # stochastic diffusion coefficient B
        'water_density': RHOW,
        'boundary_conditions': False,
        'listing': False,
        'output_file': False,
        }

    solver = cylag.Solver(fset, pset, parameters)

    nt = solver.simutime.nt
    times = np.zeros(nt + 1)
    mom = {key: np.zeros(nt + 1) for key in ['xx', 'xu', 'uu']}

    for ite in range(nt):
        solver.forward(time_step)
        xp, up, _ = pset.get_state()
        times[ite+1] = solver.simutime.time
        mom['xx'][ite+1] = np.mean(xp*xp)
        mom['xu'][ite+1] = np.mean(xp*up)
        mom['uu'][ite+1] = np.mean(up*up)

    if pset.npart_active != npart:
        print("WARNING: {} particles left the domain".format(
            npart - pset.npart_active))

    return times, mom


################################################################################
#
# Post-processing utilities
# -----------------------------------------------------------------------------
#
# For a Gaussian random vector (xp, Up), the variances of the MC estimators are:
#   Var(Up^2) = 2*<Up^2>^2,
#   Var(xp^2) = 2*<xp^2>^2,
#   Var(xp*Up) = <xp^2>*<Up^2> + <xp*Up>^2,
# with nsample = 2*npart independent samples (npart particles x 2 directions).
#
def confidence_interval(ana, npart, quantile=2.576):
    """ Half-width of the 99% confidence interval of the MC estimators """
    nsample = 2.*npart
    half = {}
    half['uu'] = quantile*np.sqrt(2./nsample)*ana['uu']
    half['xx'] = quantile*np.sqrt(2./nsample)*ana['xx']
    half['xu'] = quantile*np.sqrt((ana['xx']*ana['uu'] + ana['xu']**2)/nsample)
    return half


def plot_case(times, mom, ana, half, fname, title, ylim_uu=None):
    """ Plot <xp^2>, <Up^2> and <xp Up> against the analytical solution """
    cylag.set_rcparams()
    colors, _, _ = cylag.get_default_color_palette()
    tstar = times/TAUP

    fig, axes = plt.subplots(1, 3, figsize=(15., 4.2))
    fig.suptitle(title)

    panels = [
        ('xx', SIGMA_U**2*TAUP**2,
         "$\\langle x_p^2 \\rangle / (\\sigma_u \\tau_p)^2$"),
        ('uu', SIGMA_U**2,
         "$\\langle U_p^2 \\rangle / \\sigma_u^2$"),
        ('xu', SIGMA_U**2*TAUP,
         "$\\langle x_p U_p \\rangle / (\\sigma_u^2 \\tau_p)$"),
        ]

    for ax, (key, scale, label) in zip(axes, panels):
        ax.plot(tstar, ana[key]/scale, c='k', lw=1.5, label="Analytical")
        ax.plot(tstar, (ana[key] + half[key])/scale, c='0.6', lw=0.8, ls='--',
                label="$99\\%$ confidence interval")
        ax.plot(tstar, (ana[key] - half[key])/scale, c='0.6', lw=0.8, ls='--')
        ax.plot(tstar, mom[key]/scale, c=colors[2], lw=0., marker='x',
                markersize=4, label="CyLag (LSM-2, DI)")
        ax.set_xlabel("$t/\\tau_p$")
        ax.set_ylabel(label)

    if ylim_uu is not None:
        axes[1].set_ylim(ylim_uu)

    axes[0].legend(loc='best', fontsize=8)
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(os.path.join("figs", fname), dpi=200, format="png")
    plt.show()


def print_errors(times, mom, ana, half, tstart):
    """ Print the relative error on the moments for t > tstart """
    mask = times > tstart
    print(" ~~> relative errors for t > {:.4e} s:".format(tstart))
    for key in ['xx', 'uu', 'xu']:
        err = np.abs(mom[key][mask] - ana[key][mask])/np.abs(ana[key][mask])
        inside = np.abs(mom[key][mask] - ana[key][mask]) <= half[key][mask]
        print("     <{}> : mean = {:8.2e} ; max = {:8.2e} ; "
              "within 99% CI : {:5.1f}%".format(
                  key, np.mean(err), np.max(err), 100.*np.mean(inside)))


################################################################################
#
# Case 1 : short-time inertial regime
# -----------------------------------------------------------------------------
#
# A small time step (dt = 0.05*tau_p) resolves the transient: <Up^2> grows
# linearly from zero (short-time inertial: <Up^2> ~ B^2*t) before saturating at sigma_u^2,
# while <xp^2> goes from t^3 growth to the diffusive linear regime.
#
dt_bal = 0.05*TAUP
tf_bal = 6.*TAUP

times_bal, mom_bal = run_case(dt_bal, tf_bal, domain_size=0.01)
ana_bal = lsm2_moments(times_bal, TAUP, B2TAU)
half_bal = confidence_interval(ana_bal, NPART)

plot_case(times_bal, mom_bal, ana_bal, half_bal,
          "cylag_lsm2_hit_shorttimeinertial.png",
          "LSM-2 point source dispersion - $\\Delta t = 0.05\\,\\tau_p$")

print_errors(times_bal, mom_bal, ana_bal, half_bal, tstart=0.5*TAUP)


################################################################################
#
# Case 2 : diffusive regime with a very large time step
# -----------------------------------------------------------------------------
#
# The time step is now 200 times larger than tau_p. The Direct Integration
# scheme must still recover the correct dispersion coefficient K = sigma_u^2*taup.
#
dt_dif = 200.*TAUP
tf_dif = 24000.*TAUP

times_dif, mom_dif = run_case(dt_dif, tf_dif, domain_size=0.01)
ana_dif = lsm2_moments(times_dif, TAUP, B2TAU)
half_dif = confidence_interval(ana_dif, NPART)

plot_case(times_dif, mom_dif, ana_dif, half_dif,
          "cylag_lsm2_hit_diffusive.png",
          "LSM-2 point source dispersion - $\\Delta t = 200\\,\\tau_p$",
          ylim_uu=(0.98, 1.02))

print_errors(times_dif, mom_dif, ana_dif, half_dif, tstart=dt_dif)

# dispersion coefficient K estimated from Monte-Carlo
kdiff = mom_dif['xx'][1:]/(2.*times_dif[1:])
print(" ~~> dispersion coefficient K = {:.4e} m^2/s "
      "(theory: {:.4e} m^2/s)".format(np.mean(kdiff), K_DIFF))


################################################################################
#
# Clean:
del times_bal, mom_bal, ana_bal, half_bal
del times_dif, mom_dif, ana_dif, half_dif
