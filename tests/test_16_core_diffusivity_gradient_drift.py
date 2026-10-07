# -*- coding: utf-8 -*-
import unittest
from os import path
import numpy as np
import cylag
from matplotlib.tri import Triangulation
from cylag.lsm.random_utils import pcg32_seed

CMU = 0.09
SIGC = 0.72
EPS = 1.e-3
NPART = 20000

def mesh_and_tri():
    root = path.dirname(__file__)
    meshx = np.loadtxt(path.join(root, "data", "square_meshx.txt"))
    meshy = np.loadtxt(path.join(root, "data", "square_meshy.txt"))
    triangles = np.loadtxt(path.join(root, "data", "square_meshtri.txt"), dtype='int32')
    bnd_points, _ = cylag.read_cli(path.join(root, "data", "square_geo.cli"))
    tri = Triangulation(meshx, meshy, triangles)
    cylagtri = cylag.TriangularMesh(meshx, meshy, triangles,
                                    boundary_points=bnd_points)
    return cylagtri, tri

def tke_from_diffusivity(kdiff):
    """k such that (Cmu/sigc)*k^2/EPS = K."""
    return np.sqrt(kdiff*EPS/(CMU/SIGC))

def fset_2d(kdiff_nodes):
    cylagtri, tri = mesh_and_tri()
    npts = len(tri.x)
    opt_fields = np.empty((2, 1, npts), dtype='d')
    opt_fields[0, 0, :] = tke_from_diffusivity(kdiff_nodes)
    opt_fields[1, 0, :] = EPS
    fset = cylag.EulerianFieldSet(
        triangular_mesh=cylagtri, dim=2, nlayers=1, times=np.zeros(1),
        elevation=np.ones((1, npts)),
        velocity_x=np.zeros((1, npts)),
        velocity_y=np.zeros((1, npts)),
        opt_fields=opt_fields,
        opt_fields_names=['TURBULENT ENERG.', 'DISSIPATION'],
        trifinder=tri.get_trifinder())
    return fset, tri


class CheckDiffusivityGradient(unittest.TestCase):

    def test_gradient_2d_linear(self):
        cylagtri, tri = mesh_and_tri()
        a, b, c = 2.e-3, -1.e-3, 50.
        fset, tri = fset_2d(c + a*tri.x + b*tri.y)
        x, y = 1234.5, -2345.6
        k = int(tri.get_trifinder()(x, y))
        kp, dkdx, dkdy, dkdz = fset._interpolate_diffusivity_gradient(
            k, 0, x, y, 0., SIGC)
        self.assertAlmostEqual(kp, c + a*x + b*y, places=9)
        self.assertAlmostEqual(dkdx, a, places=12)
        self.assertAlmostEqual(dkdy, b, places=12)
        self.assertEqual(dkdz, 0.)

    def test_gradient_2d_uniform(self):
        fset, tri = fset_2d(np.full(4624, 3.))
        k = int(tri.get_trifinder()(10., 20.))
        kp, dkdx, dkdy, dkdz = fset._interpolate_diffusivity_gradient(
            k, 0, 10., 20., 0., SIGC)
        self.assertAlmostEqual(kp, 3., places=9)
        self.assertAlmostEqual(dkdx, 0., places=12)
        self.assertAlmostEqual(dkdy, 0., places=12)

    def test_gradient_3d_sloping_layers(self):
        # K linear in physical (x, y, z): exact on sloping prisms only if the
        # horizontal derivative is taken at fixed elevation.
        cylagtri, tri = mesh_and_tri()
        npts, nl = len(tri.x), 3
        a, b, c = 1.e-3, 0.5, 100.
        zb = -5. + 1.e-3*tri.x
        zs = 20. + 2.e-4*tri.y
        elev = np.concatenate([zb + (zs - zb)*l/(nl - 1) for l in range(nl)])
        kdiff = c + a*np.tile(tri.x, nl) + b*elev
        opt_fields = np.empty((2, 1, npts*nl), dtype='d')
        opt_fields[0, 0, :] = tke_from_diffusivity(kdiff)
        opt_fields[1, 0, :] = EPS
        fset = cylag.EulerianFieldSet(
            triangular_mesh=cylagtri, dim=3, nlayers=nl, times=np.zeros(1),
            elevation=elev[None, :],
            velocity_x=np.zeros((1, npts*nl)),
            velocity_y=np.zeros((1, npts*nl)),
            velocity_z=np.zeros((1, npts*nl)),
            opt_fields=opt_fields,
            opt_fields_names=['TURBULENT ENERG.', 'DISSIPATION'],
            trifinder=tri.get_trifinder())
        x, y = 1500., -800.
        k = int(tri.get_trifinder()(x, y))
        for z in (0., 8., 15.):
            ll = fset.z_localize(k, x, y, z)
            self.assertGreaterEqual(ll, 0)
            kp, dkdx, dkdy, dkdz = fset._interpolate_diffusivity_gradient(
                k, ll, x, y, z, SIGC)
            self.assertAlmostEqual(kp, c + a*x + b*z, places=7)
            self.assertAlmostEqual(dkdx, a, places=9)
            self.assertAlmostEqual(dkdy, 0., places=9)
            self.assertAlmostEqual(dkdz, b, places=9)


class CheckDiffusivityGradientDrift(unittest.TestCase):
    """Mean displacement over one step must equal grad(K)*dt (Ito drift)."""

    X0 = 4800.
    ALPHA = 1.e-2
    DT = 1000.

    def kdiff_nodes(self, alpha):
        _, tri = mesh_and_tri()
        # all nodes of the release element lie beyond the kink at x=4000
        return 0.01 + alpha*np.maximum(tri.x - 4000., 0.)

    def mean_displacement(self, model, alpha, **extra):
        pcg32_seed(42, 0xda3e06d11bb2b203)
        fset, tri = fset_2d(self.kdiff_nodes(alpha))
        position = np.zeros((NPART, 2), dtype='d')
        position[:, 0] = self.X0
        pset = cylag.LagrangianParticleSet(
            position, dim=2, particle_density=1000., particle_diameter=1.e-4,
            drag_coefficient_model=1, added_mass_force=False)
        parameters = {
            'final_time': self.DT, 'time_step': self.DT,
            'time_scheme': 1, 'frozen_eulerian_fields': True,
            'model': model, 'diffusion_model': 2, 'schmidt_number': SIGC,
            'horizontal_diffusivity': 0., 'vertical_diffusivity': 0.,
            'rng_method': 2, 'water_density': 1000.,
            'boundary_conditions': False, 'listing': False,
            'output_file': False,
            }
        parameters.update(extra)
        if model == 2:
            parameters['particle_velocity_init'] = 0
        solver = cylag.Solver(fset, pset, parameters)
        solver.forward(self.DT)
        xp, _, _ = pset.get_state()
        dx = np.asarray(xp)[:, 0] - self.X0
        return dx.mean(), dx.std()/np.sqrt(NPART)

    def check_drift(self, model, **extra):
        mean, se = self.mean_displacement(model, self.ALPHA, **extra)
        self.assertLess(abs(mean - self.ALPHA*self.DT), 5.*se)
        mean0, se0 = self.mean_displacement(model, 0., **extra)
        self.assertLess(abs(mean0), 5.*se0)

    def test_lsm1_drift(self):
        self.check_drift(1)

    def test_lsm2_euler_position_diffusion_drift(self):
        self.check_drift(2, diffusion_lsm2_option=1)

    def test_lsm2_direct_position_diffusion_drift(self):
        self.check_drift(2, diffusion_lsm2_option=1, time_scheme=5)


if __name__ == '__main__':
    unittest.main()
