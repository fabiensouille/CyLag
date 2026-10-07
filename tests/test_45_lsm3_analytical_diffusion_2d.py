# -*- coding: utf-8 -*-
import unittest
from os import path
import numpy as np
import cylag
from matplotlib.tri import Triangulation
from cylag.lsm.random_utils import pcg32_seed

PRINTOUT = False

# Physical parameters
C0KOLM = 1.2
NU0 = 1.3e-6
UALPHA = 1.0
TL = 1.0
EPS = 2.*UALPHA**2/(C0KOLM*TL)
TKE = (0.5 + 0.75*C0KOLM)*TL*EPS
SIG2 = C0KOLM*EPS
RHOW = 1000.
RHOP = 1000.
DP = 1.e-4
TAUP = (RHOP/RHOW)*DP**2/(18.*NU0)
K_DIFF = UALPHA**2*TL
NPART = 10000

def fset_2d(diffusion_model=2):
    # load mesh
    root = path.dirname(__file__)
    meshx = np.loadtxt(path.join(root, "data", "square_meshx.txt"))
    meshy = np.loadtxt(path.join(root, "data", "square_meshy.txt"))
    triangles = np.loadtxt(path.join(root, "data", "square_meshtri.txt"),
                           dtype='int32')
    bnd_file = path.join(root, "data", "square_geo.cli")
    # matplotlib triangulation
    tri = Triangulation(meshx, meshy, triangles)
    trifinder = tri.get_trifinder()
    # read boundaries
    bnd_points, _ = cylag.read_cli(bnd_file)
    cylagtri = cylag.TriangularMesh(
        meshx, meshy, triangles, boundary_points=bnd_points)
    # field_set
    meshsize = len(tri.x)
    elevation = np.ones((1, meshsize), dtype='d')
    velocity_x = np.zeros((1, meshsize), dtype='d')
    velocity_y = np.zeros((1, meshsize), dtype='d')
    optional_fields = {}
    if diffusion_model == 2:
        opt_fields = np.empty((2, 1, meshsize), dtype='d')
        opt_fields[0, 0, :] = TKE
        opt_fields[1, 0, :] = EPS
        optional_fields = {
            'opt_fields': opt_fields,
            'opt_fields_names': ['TURBULENT ENERG.', 'DISSIPATION'],
            }
    fset = cylag.EulerianFieldSet(
        triangular_mesh=cylagtri,
        dim=2, nlayers=1,
        times=np.zeros(1),
        elevation=elevation,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        trifinder=trifinder,
        **optional_fields)
    return fset, tri

def lsm3_moments(t, tl=TL, taup=TAUP, sig2=SIG2):
    """Exact second-order moments of (xp, Up, Us)."""
    if abs(tl - taup) < 1.e-12*tl:
        raise ValueError("degenerate case tl == taup, not implemented")
    t = np.asarray(t, dtype='d')
    alpha = np.exp(-t/tl)
    beta = np.exp(-t/taup)
    cc = tl/(tl - taup)
    i_a = tl*(1. - alpha)
    i_b = taup*(1. - beta)
    i_aa = 0.5*tl*(1. - alpha**2)
    i_bb = 0.5*taup*(1. - beta**2)
    i_ab = tl*taup/(tl + taup)*(1. - alpha*beta)
    moments = {}
    moments['ss'] = sig2*i_aa
    moments['us'] = sig2*cc*(i_aa - i_ab)
    moments['uu'] = sig2*cc**2*(i_aa - 2.*i_ab + i_bb)
    moments['xs'] = sig2*cc*(
        tl*(i_a - i_aa) - taup*(i_a - i_ab))
    moments['xu'] = sig2*cc**2*(
        (tl - taup)*(i_a - i_b) - tl*i_aa
        + (tl + taup)*i_ab - taup*i_bb)
    moments['xx'] = sig2*cc**2*(
        tl**2*(t - 2.*i_a + i_aa)
        - 2.*tl*taup*(t - i_a - i_b + i_ab)
        + taup**2*(t - 2.*i_b + i_bb))
    return moments

def confidence_interval(moments, quantile=2.576):
    """99% confidence half-width of each Gaussian moment estimator."""
    nsample = 2.*NPART
    half = {}
    for key in ['xx', 'uu', 'ss']:
        half[key] = quantile*np.sqrt(2./nsample)*moments[key]
    for key, var1, var2 in [
            ('xu', 'xx', 'uu'),
            ('xs', 'xx', 'ss'),
            ('us', 'uu', 'ss')]:
        half[key] = quantile*np.sqrt(
            (moments[var1]*moments[var2] + moments[key]**2)/nsample)
    return half

class CheckLSM3AnalyticalDiffusion2D(unittest.TestCase):

    def run_lsm3_case(self, time_step, final_time, diffusion_model=2, tl=TL):
        """Run one LSM-3 case and return all six second moments."""
        pcg32_seed(42, 0xda3e06d11bb2b203)
        fset, tri = fset_2d(diffusion_model)
        position = np.zeros((NPART, 2), dtype='d')
        pset = cylag.LagrangianParticleSet(
            position, dim=2,
            particle_density=RHOP,
            particle_diameter=DP,
            drag_coefficient_model=1,
            added_mass_force=False)

        parameters = {
            'final_time': final_time,
            'time_step': time_step,
            'time_scheme': 5,
            'frozen_eulerian_fields': True,
            'model': 3,
            'particle_velocity_init': 0,
            'diffusion_model': diffusion_model,
            'horizontal_diffusivity': K_DIFF if diffusion_model == 1 else 0.,
            # Distinct scales catch use of a hard-coded or vertical value.
            # For model 2 the horizontal input must be ignored in favor of k-eps.
            'diffusion_lsm3_tl_horizontal': tl if diffusion_model == 1 else 4.*TL,
            'diffusion_lsm3_tl_vertical': 7.*TL,
            'rng_method': 2,
            'water_density': RHOW,
            'boundary_conditions': False,
            'listing': False,
            'output_file': False,
            }

        cylag_solver = cylag.Solver(fset, pset, parameters)
        nt = cylag_solver.simutime.nt
        times = np.zeros(nt + 1, dtype='d')
        keys = ['xx', 'xu', 'xs', 'uu', 'us', 'ss']
        moments = {key: np.zeros(nt + 1, dtype='d') for key in keys}

        for ite in range(nt):
            cylag_solver.forward(time_step)
            xp, up, us = pset.get_state()
            times[ite+1] = cylag_solver.simutime.time
            moments['xx'][ite+1] = np.mean(xp*xp)
            moments['xu'][ite+1] = np.mean(xp*up)
            moments['xs'][ite+1] = np.mean(xp*us)
            moments['uu'][ite+1] = np.mean(up*up)
            moments['us'][ite+1] = np.mean(up*us)
            moments['ss'][ite+1] = np.mean(us*us)

        self.assertEqual(pset.npart_active, NPART)

        if PRINTOUT:
            print(" ~~> LSM-3 case: dt/TL = {:.2e}, tf/TL = {:.2e}".format(
                time_step/tl, final_time/tl))

        del tri
        del fset
        del pset
        del cylag_solver
        return times, moments

    def assert_moments_in_confidence_interval(self, times, moments, tstart,
                                              tl=TL, sig2=SIG2):
        """Compare the six simulated moments with their exact values."""
        analytical = lsm3_moments(times, tl=tl, sig2=sig2)
        half = confidence_interval(analytical)
        mask = times >= tstart

        for key in ['xx', 'xu', 'xs', 'uu', 'us', 'ss']:
            inside = (np.abs(moments[key][mask] - analytical[key][mask])
                      <= half[key][mask])
            ratio = np.mean(inside)
            if PRINTOUT:
                print("     <{}>: {:.1f}% within the 99% CI".format(
                    key, 100.*ratio))
            self.assertGreaterEqual(
                ratio, 0.90,
                msg="<{}>: insufficient agreement with analytical solution"
                    .format(key))

    def test_analytical_asymptotic_behaviors(self):
        # Tracer limit: tau_p << t << T_L.
        t_short = 0.05*TL
        short = lsm3_moments(t_short)
        short_limit = {
            'uu': SIG2*t_short,
            'ss': SIG2*t_short,
            'us': SIG2*t_short,
            'xu': 0.5*SIG2*t_short**2,
            'xs': 0.5*SIG2*t_short**2,
            'xx': SIG2*t_short**3/3.,
            }
        for key in ['xx', 'xu', 'xs', 'uu', 'us', 'ss']:
            np.testing.assert_allclose(short[key], short_limit[key], rtol=0.08)

        # Diffusive limit: t >> T_L >> tau_p.
        t_long = 1.e4*TL
        long = lsm3_moments(t_long)
        np.testing.assert_allclose(long['ss'], UALPHA**2, rtol=1.e-12)
        np.testing.assert_allclose(long['uu'], UALPHA**2, rtol=5.e-4)
        np.testing.assert_allclose(long['us'], UALPHA**2, rtol=5.e-4)
        np.testing.assert_allclose(long['xu'], K_DIFF, rtol=5.e-4)
        np.testing.assert_allclose(long['xs'], K_DIFF, rtol=5.e-4)
        np.testing.assert_allclose(long['xx'], 2.*K_DIFF*t_long, rtol=2.e-4)

    def test_solver_2d_lsm3_short_time_analytical_diffusion(self):
        # Resolve the transient from rest to stationary velocity statistics.
        time_step = 0.05*TL
        final_time = 6.*TL
        times, moments = self.run_lsm3_case(time_step, final_time)
        self.assert_moments_in_confidence_interval(
            times, moments, tstart=0.5*TL)

    def test_solver_2d_lsm3_diffusive_limit(self):
        # Verify the direct integrator with a time step much larger than T_L.
        time_step = 200.*TL
        final_time = 24000.*TL
        times, moments = self.run_lsm3_case(time_step, final_time)
        self.assert_moments_in_confidence_interval(
            times, moments, tstart=time_step)

        # In the long-time limit, <xp^2>/(2t) must converge to Ualpha^2*T_L.
        kdiff = moments['xx'][1:]/(2.*times[1:])
        np.testing.assert_allclose(np.mean(kdiff), K_DIFF, rtol=0.03)

    def test_solver_2d_lsm3_constant_k_short_time(self):
        # Keep K fixed while changing TL: Bs**2 = 2*K/TL**2 and Var(Us) = K/TL.
        for tl in (0.5, 2.5):
            with self.subTest(diffusion_model=1, tl_horizontal=tl):
                time_step = 0.05*tl
                final_time = 6.*tl
                times, moments = self.run_lsm3_case(
                    time_step, final_time, diffusion_model=1, tl=tl)
                self.assert_moments_in_confidence_interval(
                    times, moments, tstart=0.5*tl,
                    tl=tl, sig2=2.*K_DIFF/tl**2)

    def test_solver_2d_lsm3_constant_k_diffusive_limit(self):
        # The long-time diffusivity must remain K, independently of TL.
        for tl in (0.5, 2.5):
            with self.subTest(diffusion_model=1, tl_horizontal=tl):
                time_step = 200.*tl
                final_time = 24000.*tl
                times, moments = self.run_lsm3_case(
                    time_step, final_time, diffusion_model=1, tl=tl)
                self.assert_moments_in_confidence_interval(
                    times, moments, tstart=time_step,
                    tl=tl, sig2=2.*K_DIFF/tl**2)

                kdiff = moments['xx'][1:]/(2.*times[1:])
                np.testing.assert_allclose(np.mean(kdiff), K_DIFF, rtol=0.03)

if __name__ == "__main__":
    unittest.main()
