from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.geometry import SpecialEuclidean, SpecialOrthogonal
from geojax.optimization import ConjugateGradient, Minimize


def rotation(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])


def test_special_orthogonal_group_operations_and_local_maps():
    M = SpecialOrthogonal(size=3)
    R = M.random_point(jax.random.key(0))
    U = M.random_tangent(jax.random.key(1), R, scale=0.1)
    Q = M.exp(R, U)

    assert M.shape == (3, 3)
    assert M.dim == 3
    assert bool(M.belongs(R))
    assert bool(M.belongs(Q))
    assert jnp.allclose(M.log(R, Q), U, atol=2e-6)
    assert jnp.allclose(M.compose(R, M.inverse(R)), M.identity, atol=1e-6)


def test_special_orthogonal_log_is_differentiable_away_from_cut_locus():
    M = SpecialOrthogonal(size=2)
    target = rotation(0.8)

    def squared_distance(theta):
        return M.dist(rotation(theta), target) ** 2

    assert jnp.allclose(jax.grad(squared_distance)(0.4), -1.6, atol=1e-6)


def test_special_orthogonal_pi_rotation_marks_nonunique_log():
    M = SpecialOrthogonal(size=2)
    assert jnp.all(jnp.isnan(M.log(jnp.eye(2), -jnp.eye(2))))


def test_special_euclidean_group_and_riemannian_exponentials():
    M = SpecialEuclidean(size=2)
    omega = jnp.array([[0.0, -0.6], [0.6, 0.0]])
    velocity = jnp.array([1.0, -0.2])
    xi = M.tangent_from_components(omega, velocity)

    group_point = M.group_exp(xi)
    riemannian_point = M.exp(M.identity, xi)

    assert bool(M.belongs(group_point))
    assert bool(M.belongs(riemannian_point))
    assert jnp.allclose(M.group_log(group_point), xi, atol=1e-6)
    assert not jnp.allclose(M.translation(group_point), M.translation(riemannian_point))
    assert jnp.allclose(M.compose(group_point, M.inverse(group_point)), M.identity, atol=1e-6)


def test_special_euclidean_rigid_registration_smoke(dtype_atol):
    M = SpecialEuclidean(size=2)
    source = jnp.array([[-1.0, -0.4], [-0.2, 0.8], [0.7, 0.5], [1.0, -0.7]])
    truth = M.from_components(rotation(0.55), jnp.array([0.8, -0.35]))
    target = jax.vmap(lambda point: M.apply(truth, point))(source)

    def cost(G):
        aligned = jax.vmap(lambda point: M.apply(G, point))(source)
        return 0.5 * jnp.mean(jnp.sum((aligned - target) ** 2, axis=-1))

    estimate, final_cost, history = Minimize(
        M=M,
        cost=cost,
        x0=M.identity,
        solver=ConjugateGradient(maxiter=80, tolgradnorm=1e-8, verbosity=0),
    ).solve()

    assert final_cost < max(1e-10, dtype_atol)
    assert history[-1].gradnorm < max(1e-7, dtype_atol)
    assert jnp.allclose(estimate, truth, atol=max(2e-5, dtype_atol))
