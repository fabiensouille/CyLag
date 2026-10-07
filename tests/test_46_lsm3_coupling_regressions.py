import numpy as np
import pytest
from matplotlib.tri import Triangulation
from scipy.linalg import expm
import cylag
from cylag.lsm.random_utils import pcg32_seed
from cylag.lsm.update_state_lsm3 import lsm3_direct_coefficients

def make_solver(dim, field_dim, scheme, density=1000., added_mass=True,
                diffusivity=0.1, diffusion_model=1):
    tri = Triangulation(
        [-100., 100., 100., -100.], [-100., -100., 100., 100.],
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh = cylag.TriangularMesh.from_matplotlib(tri)
    nlayers = 2 if field_dim == 3 else 1
    shape = (1, 4*nlayers)
    elevation = (np.array([[-100.]*4 + [100.]*4]) if field_dim == 3
                 else np.full(shape, 100.))
    fields = np.empty((3, *shape))
    fields[0] = -100.
    fields[1] = (0.5 + 0.75*1.2)*0.7*0.2
    fields[2] = 0.2
    fset = cylag.EulerianFieldSet(
        triangular_mesh=mesh, dim=field_dim, nlayers=nlayers,
        times=np.zeros(1), elevation=elevation,
        velocity_x=np.zeros(shape), velocity_y=np.zeros(shape),
        velocity_z=np.zeros(shape) if field_dim == 3 else None,
        opt_fields=fields,
        opt_fields_names=['BOTTOM', 'TURBULENT ENERG.', 'DISSIPATION'],
        trifinder=tri.get_trifinder())
    pset = cylag.LagrangianParticleSet(
        np.zeros((4, dim)), dim=dim, particle_density=density,
        particle_diameter=0.001, drag_coefficient_model=1,
        added_mass_force=added_mass)
    solver = cylag.Solver(fset, pset, {
        'model': 3, 'time_scheme': scheme, 'time_step': 0.005,
        'final_time': 0.02, 'frozen_eulerian_fields': True,
        'particle_velocity_init': 0, 'diffusion_model': diffusion_model,
        'horizontal_diffusivity': diffusivity,
        'vertical_diffusivity': 0.5*diffusivity,
        'diffusion_lsm3_tl_horizontal': 0.7,
        'diffusion_lsm3_tl_vertical': 1.3,
        'rng_method': 2, 'water_density': 1000.,
        'boundary_conditions': False, 'listing': False,
        'output_file': False,
    })
    return solver, pset

@pytest.mark.parametrize('dim,field_dim,density,added_mass,diffusivity', [
    (2, 2, 1000., True, 0.1)])
def test_em_uses_initial_seen_velocity(dim, field_dim, density, added_mass,
                                       diffusivity):
    solver, pset = make_solver(dim, field_dim, 1, density, added_mass,
                               diffusivity)
    np.asarray(pset.velocity)[:4] = -0.1
    np.asarray(pset.fluid_velocity_seen)[:4] = 0.2
    xp0, up0, us0 = [v.copy() for v in pset.get_state()]
    slip = np.linalg.norm(us0[0] - up0[0])
    a1 = pset.a1c*1.5/0.001*slip*pset.drag_coefficient(slip, 0.001)
    dt = solver.simutime.time_step
    pcg32_seed(42, 7)
    solver.forward(dt)
    xp, up, us = pset.get_state()
    assert pset.npart_active == 4
    expected = up0 + a1*(us0 - up0)*dt + pset.a2c*(us - us0)
    if dim == 3:
        expected[:, 2] -= 9.80665*pset.a0c*dt
    np.testing.assert_allclose(up, expected, rtol=3e-14, atol=3e-16)
    np.testing.assert_allclose(xp, xp0 + expected*dt, rtol=3e-14, atol=3e-16)


@pytest.mark.parametrize('dim,field_dim,diffusion_model', [
    (2, 2, 1)])
@pytest.mark.parametrize('scheme', [5])
def test_neutral_particles_follow_seen_velocity(dim, field_dim, scheme,
                                                diffusion_model):
    solver, pset = make_solver(dim, field_dim, scheme,
                               diffusion_model=diffusion_model)
    np.asarray(pset.velocity)[:4] = 0.1
    np.asarray(pset.fluid_velocity_seen)[:4] = 0.1
    pcg32_seed(42, 7)
    assert pset.a2c == 1.
    for _ in range(4):
        solver.forward(solver.simutime.time_step)
        xp, up, us = pset.get_state()
        assert pset.npart_active == 4
        assert np.isfinite(xp).all()
        np.testing.assert_array_equal(up, us)


@pytest.mark.parametrize('a1,tl,a2,dt', [
    (0.3, 1., 0.5, 1e-18)])
def test_physical_transition_matches_expm(a1, tl, a2, dt):
    drift = np.array([[-1./tl, 0., 0.],
                      [a1 - a2/tl, -a1, 0.], [0., 1., 0.]])
    transition, covariance, response = lsm3_direct_coefficients(a1, tl, dt, a2)
    np.testing.assert_allclose(transition, expm(dt*drift), rtol=3e-13, atol=0.)
    force_drift = np.zeros((4, 4))
    force_drift[:3, :3] = drift
    force_drift[1, 3] = 1.
    np.testing.assert_allclose(response, expm(dt*force_drift)[:3, 3],
                               rtol=3e-13, atol=0.)
    np.testing.assert_array_equal(covariance, covariance.T)


@pytest.mark.parametrize('dt', [1e-6])
def test_zero_physical_drift_coupling(dt):
    transition, _, _ = lsm3_direct_coefficients(2., 0.25, dt, 0.5)
    assert transition[1, 0] == 0.
    assert transition[2, 0] == 0.


@pytest.mark.parametrize('a1,tl,a2', [
    (0.3, 1., 0.5)])
def test_covariance_matches_van_loan(a1, tl, a2):
    dt = 0.2
    drift = np.array([[-1./tl, 0., 0.],
                      [a1 - a2/tl, -a1, 0.], [0., 1., 0.]])
    noise = np.array([1., a2, 0.])
    augmented = np.zeros((6, 6))
    augmented[:3, :3] = drift
    augmented[:3, 3:] = np.outer(noise, noise)
    augmented[3:, 3:] = -drift.T
    reference = expm(dt*augmented)
    _, covariance, _ = lsm3_direct_coefficients(a1, tl, dt, a2)
    expected = reference[:3, 3:] @ reference[:3, :3].T
    np.testing.assert_allclose(covariance, expected, rtol=3e-13, atol=3e-16)

def van_loan(a1, tl, a2, dt):
    drift = np.array([[-1./tl, 0., 0.],
                      [a1 - a2/tl, -a1, 0.], [0., 1., 0.]])
    noise = np.array([1., a2, 0.])
    augmented = np.zeros((6, 6))
    augmented[:3, :3] = drift
    augmented[:3, 3:] = np.outer(noise, noise)
    augmented[3:, 3:] = -drift.T
    reference = expm(dt*augmented)
    return reference[:3, 3:] @ reference[:3, :3].T


@pytest.mark.parametrize('a1,tl,a2,dt', [
    (1., 1., 0.5, 0.3),            # a1 = 1/tl, singular closed form
    (50., 1e-2, 2.5, 1e-3)])
def test_clipped_option_matches_reference(a1, tl, a2, dt):
    drift = np.array([[-1./tl, 0., 0.],
                      [a1 - a2/tl, -a1, 0.], [0., 1., 0.]])
    transition, covariance, response = lsm3_direct_coefficients(
        a1, tl, dt, a2, 2)
    np.testing.assert_allclose(transition, expm(dt*drift),
                               rtol=1e-11, atol=1e-14)
    force_drift = np.zeros((4, 4))
    force_drift[:3, :3] = drift
    force_drift[1, 3] = 1.
    np.testing.assert_allclose(response, expm(dt*force_drift)[:3, 3],
                               rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(covariance, van_loan(a1, tl, a2, dt),
                               rtol=5e-6, atol=1e-12*np.abs(covariance).max())


@pytest.mark.parametrize('dt', [1e-6])
@pytest.mark.parametrize('a1,tl', [
    (1e8, 1e-8)])
@pytest.mark.parametrize('a2', [0.5])
def test_clipped_option_is_finite_and_positive(a1, tl, a2, dt):
    transition, covariance, response = lsm3_direct_coefficients(
        a1, tl, dt, a2, 2)
    assert np.isfinite(transition).all()
    assert np.isfinite(covariance).all()
    assert np.isfinite(response).all()
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues.min() >= -1e-10*max(eigenvalues.max(), 1e-300)


def test_invalid_edge_option():
    with pytest.raises(ValueError):
        lsm3_direct_coefficients(1., 1., 0.1, 0., 3)
    with pytest.raises(ValueError):
        cylag.Parameters({'lsm3_edge_opt': 3}).check_values()


@pytest.mark.parametrize('dim,field_dim,diffusion_model', [
    (3, 2, 3)])
def test_clipped_neutral_particles_follow_seen_velocity(dim, field_dim,
                                                        diffusion_model):
    solver, pset = make_solver(dim, field_dim, 5,
                               diffusion_model=diffusion_model)
    solver.parameters.lsm3_edge_opt = 2
    np.asarray(pset.velocity)[:4] = 0.1
    np.asarray(pset.fluid_velocity_seen)[:4] = 0.1
    pcg32_seed(42, 7)
    for _ in range(4):
        solver.forward(solver.simutime.time_step)
        xp, up, us = pset.get_state()
        assert pset.npart_active == 4
        assert np.isfinite(xp).all()
        np.testing.assert_allclose(up, us, rtol=1e-12, atol=1e-14)
