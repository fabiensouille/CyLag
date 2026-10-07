# -*- coding: utf-8 -*-
import unittest
from os import path
import numpy as np
import cylag
from matplotlib.tri import Triangulation
from cylag.lsm.random_utils import pcg32_seed

PRINTOUT = False

# Particle and fluid parameters
NU0 = 1.3e-6
RHOW = 1000.
RHOP = 1000.
DP = 1.e-4
TAUP = RHOP*DP**2/(18.*RHOW*NU0)
B = 0.001
B2TAU = TAUP*B**2
SIGMA_U2 = B2TAU/2.
K_DIFF = SIGMA_U2*TAUP
# Homogeneous k-epsilon fields chosen so that
# Bp = (TL/TAUP)*sqrt(C0KOLM*EPS) = B and K = Bp**2*TAUP**2/2.
C0KOLM = 1.2
TL = 0.25
EPS = 2.*K_DIFF/(C0KOLM*TL**2)
TKE = (0.5 + 0.75*C0KOLM)*TL*EPS
NPART = 10000

def fset_2d(diffusion_model=1):
    # load mesh
    root = path.dirname(__file__)
    meshx = np.loadtxt(path.join(root, "data", "square_meshx.txt"))
    meshy = np.loadtxt(path.join(root, "data", "square_meshy.txt"))
    triangles = np.loadtxt(path.join(root, "data", "square_meshtri.txt"), dtype='int32')
    bnd_file = path.join(root, "data", "square_geo.cli")
    # matplotlib triangulation
    tri = Triangulation(meshx, meshy, triangles)
    trifinder = tri.get_trifinder()
    # read boundaries
    bnd_points, _ = cylag.read_cli(bnd_file)
    cylagtri = cylag.TriangularMesh(meshx, meshy, triangles,
                                    boundary_points=bnd_points)
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

def lsm2_moments(t):
    """Exact LSM-2 moments for zero initial position and velocity."""
    t = np.asarray(t, dtype='d')
    beta = np.exp(-t/TAUP)
    moments = {}
    moments['uu'] = 0.5*B2TAU*(1. - beta**2)
    moments['xu'] = 0.5*B2TAU*TAUP*(1. - beta)**2
    moments['xx'] = B2TAU*TAUP*(
        t - 2.*TAUP*(1. - beta) + 0.5*TAUP*(1. - beta**2))
    return moments

def confidence_interval(moments, quantile=2.576):
    """99% Monte-Carlo confidence half-width for 2*NPART samples."""
    nsample = 2.*NPART
    half = {}
    half['uu'] = quantile*np.sqrt(2./nsample)*moments['uu']
    half['xx'] = quantile*np.sqrt(2./nsample)*moments['xx']
    half['xu'] = quantile*np.sqrt(
        (moments['xx']*moments['uu'] + moments['xu']**2)/nsample)
    return half

class CheckLSM2AnalyticalDiffusion2D(unittest.TestCase):

    def run_lsm2_case(self, time_step, final_time, diffusion_model=1,
                      diffusion_lsm2_option=2):
        """Run equivalent Bp-, K-, or k-epsilon-based diffusion from rest."""
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
            'model': 2,
            'particle_velocity_init': 0,
            'diffusion_model': diffusion_model,
            'diffusion_lsm2_option': diffusion_lsm2_option,
            # Model 2 must derive Bp from the fields, not this input.
            'horizontal_diffusivity': (0. if diffusion_model == 2 else
                                      B if diffusion_lsm2_option == 2 else K_DIFF),
            'vertical_diffusivity': 0.,
            'rng_method': 2,
            'water_density': RHOW,
            'boundary_conditions': False,
            'listing': False,
            'output_file': False,
            }

        cylag_solver = cylag.Solver(fset, pset, parameters)
        nt = cylag_solver.simutime.nt
        times = np.zeros(nt + 1, dtype='d')
        moments = {key: np.zeros(nt + 1, dtype='d')
                   for key in ['xx', 'xu', 'uu']}

        for ite in range(nt):
            cylag_solver.forward(time_step)
            xp, up, _ = pset.get_state()
            times[ite+1] = cylag_solver.simutime.time
            moments['xx'][ite+1] = np.mean(xp*xp)
            moments['xu'][ite+1] = np.mean(xp*up)
            moments['uu'][ite+1] = np.mean(up*up)

        self.assertEqual(pset.npart_active, NPART)

        if PRINTOUT:
            print(" ~~> LSM-2 case: dt/taup = {:.2e}, tf/taup = {:.2e}".format(
                time_step/TAUP, final_time/TAUP))

        del tri
        del fset
        del pset
        del cylag_solver
        return times, moments

    def assert_moments_in_confidence_interval(self, times, moments, tstart):
        """Compare Monte-Carlo moments with the analytical solution."""
        analytical = lsm2_moments(times)
        half = confidence_interval(analytical)
        mask = times >= tstart

        for key in ['xx', 'xu', 'uu']:
            error = np.abs(moments[key][mask] - analytical[key][mask])
            inside = error <= half[key][mask]
            ratio = np.mean(inside)
            if PRINTOUT:
                print("     <{}>: {:.1f}% within the 99% CI".format(
                    key, 100.*ratio))
            # A fraction criterion is robust to isolated Monte-Carlo outliers
            # while still checking the complete time series.
            self.assertGreaterEqual(
                ratio, 0.90,
                msg="<{}>: insufficient agreement with analytical solution"
                    .format(key))

    def test_analytical_asymptotic_behaviors(self):
        # Short-time inertial limit: t << tau_p
        t_short = 1.e-4*TAUP
        short = lsm2_moments(t_short)
        short_limit = {
            'uu': B**2*t_short,
            'xu': 0.5*B**2*t_short**2,
            'xx': B**2*t_short**3/3.,
            }
        for key in ['xx', 'xu', 'uu']:
            np.testing.assert_allclose(short[key], short_limit[key], rtol=3.e-4)

        # Long-time diffusive limit: t >> tau_p
        t_long = 1.e4*TAUP
        long = lsm2_moments(t_long)
        self.assertAlmostEqual(long['uu']/SIGMA_U2, 1., places=12)
        self.assertAlmostEqual(long['xu']/(SIGMA_U2*TAUP), 1., places=12)
        np.testing.assert_allclose(long['xx'], 2.*K_DIFF*t_long, rtol=2.e-4)

    def check_short_time_diffusion(self, diffusion_model, diffusion_lsm2_option):
        # Resolve the inertial transient up to velocity-variance saturation.
        time_step = 0.05*TAUP
        final_time = 6.*TAUP
        times, moments = self.run_lsm2_case(
            time_step, final_time, diffusion_model, diffusion_lsm2_option)
        self.assert_moments_in_confidence_interval(
            times, moments, tstart=0.5*TAUP)

    def check_diffusive_limit(self, diffusion_model, diffusion_lsm2_option):
        # Verify the direct integrator with a time step much larger than tau_p.
        time_step = 200.*TAUP
        final_time = 24000.*TAUP
        times, moments = self.run_lsm2_case(
            time_step, final_time, diffusion_model, diffusion_lsm2_option)
        self.assert_moments_in_confidence_interval(
            times, moments, tstart=time_step)

        # In the long-time limit, <x_p^2>/(2t) must converge to K_DIFF.
        kdiff = moments['xx'][1:]/(2.*times[1:])
        np.testing.assert_allclose(np.mean(kdiff), K_DIFF, rtol=0.03)

    def test_solver_2d_lsm2_short_time_analytical_diffusion(self):
        self.check_short_time_diffusion(diffusion_model=1, diffusion_lsm2_option=2)

    def test_solver_2d_lsm2_diffusive_limit(self):
        self.check_diffusive_limit(diffusion_model=1, diffusion_lsm2_option=2)

    def test_solver_2d_lsm2_constant_k_short_time(self):
        # Option 3 takes K and computes Bp = sqrt(2*K)/taup.
        self.check_short_time_diffusion(diffusion_model=1, diffusion_lsm2_option=3)

    def test_solver_2d_lsm2_constant_k_diffusive_limit(self):
        self.check_diffusive_limit(diffusion_model=1, diffusion_lsm2_option=3)

    def test_solver_2d_lsm2_k_epsilon_short_time(self):
        # Both velocity-diffusion options use Bp = TL*sqrt(C0*eps)/taup.
        for option in (2, 3):
            with self.subTest(diffusion_model=2, diffusion_lsm2_option=option):
                self.check_short_time_diffusion(2, option)

    def test_solver_2d_lsm2_k_epsilon_diffusive_limit(self):
        for option in (2, 3):
            with self.subTest(diffusion_model=2, diffusion_lsm2_option=option):
                self.check_diffusive_limit(2, option)

if __name__ == "__main__":
    unittest.main()
