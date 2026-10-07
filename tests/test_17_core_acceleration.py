import numpy as np
import pytest
from matplotlib.tri import Triangulation
import cylag

L = 100.
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
XN = np.array([-L, L, L, -L])
YN = np.array([-L, -L, L, L])


def make_mesh():
    tri = Triangulation(XN, YN, triangles=TRIANGLES)
    return tri, cylag.TriangularMesh.from_matplotlib(tri)


def make_fset_2d(ux, uy, times=(0.,)):
    tri, mesh = make_mesh()
    nt = len(times)
    shape = (nt, 4)
    bottom = np.full((1, *shape), -100.)
    return cylag.EulerianFieldSet(
        triangular_mesh=mesh, dim=2, nlayers=1, times=np.array(times),
        elevation=np.full(shape, 100.),
        velocity_x=np.broadcast_to(ux, shape).copy(),
        velocity_y=np.broadcast_to(uy, shape).copy(),
        opt_fields=bottom, opt_fields_names=['BOTTOM'],
        trifinder=tri.get_trifinder()), tri


def make_fset_3d(slope, a, b, c, d):
    """ u = a*x + b*z and v = c*y + d*z on two layers with sloping bed/surface. """
    tri, mesh = make_mesh()
    zb = -100. + slope*XN
    zs = 100. + slope*XN
    z = np.concatenate([zb, zs])
    x = np.tile(XN, 2)
    y = np.tile(YN, 2)
    shape = (1, 8)
    fset = cylag.EulerianFieldSet(
        triangular_mesh=mesh, dim=3, nlayers=2, times=np.zeros(1),
        elevation=z[None, :], velocity_x=(a*x + b*z)[None, :],
        velocity_y=(c*y + d*z)[None, :], velocity_z=np.zeros(shape),
        opt_fields=np.full((1, *shape), -100.), opt_fields_names=['BOTTOM'],
        trifinder=tri.get_trifinder())
    return fset, tri


def test_acceleration_2d_convective_term():
    gamma, U0, V0 = 0.1, 1., 0.3
    fset, tri = make_fset_2d(U0 + gamma*XN, V0 + 0.2*YN)
    x, y = 10., 20.
    k = int(tri.get_trifinder()(x, y))
    ua, va = 0.7, -0.4
    ax, ay = fset._interpolate_acceleration_2d(k, x, y, 0.1, ua, va)
    assert ax == pytest.approx(gamma*ua, rel=1e-12)
    assert ay == pytest.approx(0.2*va, rel=1e-12)


def test_acceleration_2d_time_derivative():
    # u(x, t) = 1 + 0.1*x + 0.5*t is linear in time, so du/dt = 0.5 for any record > 0
    times = (0., 1., 2.)
    ux = np.array([[1. + 0.1*XN + 0.5*t] for t in times])[:, 0, :]
    fset, tri = make_fset_2d(ux, np.zeros_like(ux), times=times)
    x, y = 10., 20.
    k = int(tri.get_trifinder()(x, y))
    fset._set_record(1.1, 0)
    fset._time_interpolation(1.1)
    fset._time_interpolation(1.2)
    assert fset.record == 1
    ax, ay = fset._interpolate_acceleration_2d(k, x, y, 0.1, 0.7, 0.)
    assert ax == pytest.approx(0.5 + 0.1*0.7, rel=1e-9)
    assert ay == 0.


@pytest.mark.parametrize('slope', [0., 0.1])
def test_acceleration_3d_convective_term(slope):
    a, b, c, d = 0.1, 0.02, 0.05, 0.03
    fset, tri = make_fset_3d(slope, a, b, c, d)
    x, y = 10., 20.
    z = 5. + slope*x
    k = int(tri.get_trifinder()(x, y))
    ua, va, wa = 0.7, -0.4, 0.2
    ax, ay, az = fset._interpolate_acceleration_3d(
        k, 0, x, y, z, 0.1, ua, va, wa)
    assert ax == pytest.approx(a*ua + b*wa, rel=1e-10)
    assert ay == pytest.approx(c*va + d*wa, rel=1e-10)
    assert az == 0.


def make_solver(model, gamma, particle_kwargs, params, dim=2, npart=4):
    fset, _ = make_fset_2d(1. + gamma*XN, np.zeros(4))
    pset = cylag.LagrangianParticleSet(
        np.tile([[-50., 0.]], (npart, 1)) + np.arange(npart)[:, None]*[0., 1.],
        dim=2, particle_density=1000., drag_coefficient_model=1,
        **particle_kwargs)
    parameters = {
        'model': model, 'time_scheme': 5, 'time_step': 0.01,
        'final_time': 1., 'frozen_eulerian_fields': True,
        'particle_velocity_init': 1, 'boundary_conditions': False,
        'listing': False, 'output_file': False}
    parameters.update(params)
    return cylag.Solver(fset, pset, parameters), pset


def test_lsm3_seen_velocity_follows_mean_flow_along_particle_path():
    # steady accelerating flow: Us must keep tracking U(x_p) with no lag of order TL*dU/dt
    gamma = 0.1
    solver, pset = make_solver(
        3, gamma, dict(particle_diameter=1e-3),
        {'diffusion_model': 1, 'horizontal_diffusivity': 0.,
         'vertical_diffusivity': 0., 'diffusion_lsm3_tl_horizontal': 1.,
         'diffusion_lsm3_tl_vertical': 1.})
    solver.solve()
    xp, up, us = pset.get_state()
    assert pset.npart_active == 4
    mean_flow = 1. + gamma*xp[:, 0]
    # without the convective term the lag is about TL*gamma*U*(1-exp(-1)) = 0.06
    np.testing.assert_allclose(us[:, 0], mean_flow, atol=3e-3)


def test_lsm2_neutral_particles_follow_convective_acceleration():
    # A2 = 1: particles initially at rest relative to the fluid stay with the fluid
    gamma = 0.1
    solver, pset = make_solver(
        2, gamma, dict(particle_diameter=1e-2, added_mass_force=True,
                       added_mass_coef=0.5),
        {'diffusion_model': 0})
    assert pset.a2c == pytest.approx(1.)
    solver.solve()
    xp, up, us = pset.get_state()
    assert pset.npart_active == 4
    mean_flow = 1. + gamma*xp[:, 0]
    # without the convective term the lag is O(gamma*U/A1) ~ 0.5
    np.testing.assert_allclose(up[:, 0], mean_flow, atol=1e-2)
