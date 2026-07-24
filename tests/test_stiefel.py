from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import Stiefel, StiefelEuclidean
from geojax.optimization import Minimize, SteepestDescent


@pytest.mark.parametrize("geometry", [Stiefel, StiefelEuclidean])
def test_stiefel_constraints_projection_and_sampling(geometry):
    M = geometry(size=(5, 2))
    points = M.random_point(jax.random.key(0), sample_shape=(4,))
    tangents = M.random_tangent(jax.random.key(1), points)

    assert M.shape == (5, 2)
    assert M.dim == 7
    assert jnp.all(M.belongs(points))
    assert jnp.all(M.is_tangent(points, tangents))
    assert jnp.all(M.belongs(M.project(jax.random.normal(jax.random.key(2), (4, 5, 2)))))
    assert M.operation_kind("exp") == "exact"
    assert M.operation_kind("log") == "numerical-local"
    assert M.operation_kind("dist") == "numerical-local"


@pytest.mark.parametrize("geometry", [Stiefel, StiefelEuclidean])
def test_stiefel_local_exp_log_and_batch_roundtrip(geometry, dtype_atol):
    M = geometry(size=(4, 2), log_maxiter=24)
    X = M.random_point(jax.random.key(0))
    U = M.random_tangent(jax.random.key(1), X, scale=0.08)
    Y = M.exp(X, U)
    recovered, info = M.log_with_info(X, Y)

    assert bool(info.converged)
    assert info.residual_norm < max(1e-8, dtype_atol)
    assert jnp.allclose(recovered, U, atol=max(2e-8, dtype_atol))
    assert jnp.allclose(M.log(X, Y), U, atol=max(2e-8, dtype_atol))

    tangents = jnp.stack([0.25 * U, 0.5 * U, U])
    endpoints = M.exp_batch(X, tangents)
    assert jnp.all(M.belongs(endpoints))
    assert jnp.allclose(M.log_batch(X, endpoints), tangents, atol=max(2e-8, dtype_atol))


def test_canonical_and_euclidean_metrics_differ_on_vertical_tangents():
    canonical = Stiefel(size=(4, 2))
    euclidean = StiefelEuclidean(size=(4, 2))
    X = canonical.random_point(jax.random.key(0))
    A = jnp.array([[0.0, -0.7], [0.7, 0.0]])
    vertical = X @ A
    horizontal = canonical.tangent_project(
        X, jnp.array([[0.2, -0.4], [0.3, 0.1], [-0.5, 0.2], [0.1, 0.6]])
    )
    horizontal = horizontal - X @ (X.T @ horizontal)

    assert jnp.allclose(canonical.inner(X, vertical, vertical), 0.5 * jnp.sum(vertical**2))
    assert jnp.allclose(euclidean.inner(X, vertical, vertical), jnp.sum(vertical**2))
    assert jnp.allclose(canonical.inner(X, horizontal, horizontal), jnp.sum(horizontal**2))


def test_stiefel_euclidean_exponential_satisfies_geodesic_equation():
    M = StiefelEuclidean(size=(4, 2))
    X = M.random_point(jax.random.key(0))
    U = M.random_tangent(jax.random.key(1), X, scale=0.2)

    def curve(time):
        return M.exp(X, time * U)

    acceleration = jax.jacfwd(jax.jacfwd(curve))(0.0)

    assert jnp.allclose(acceleration, -X @ (U.T @ U), atol=2e-8)


@pytest.mark.parametrize("geometry", [Stiefel, StiefelEuclidean])
def test_stiefel_gradient_conversion_and_isometric_transport(geometry):
    M = geometry(size=(5, 3))
    X = M.random_point(jax.random.key(0))
    U = M.random_tangent(jax.random.key(1), X)
    ambient_gradient = jax.random.normal(jax.random.key(2), X.shape)
    gradient = M.egrad_to_rgrad(X, ambient_gradient)
    Y = M.exp(X, 0.1 * U)
    transported = M.transport(X, Y, U)

    assert bool(M.is_tangent(X, gradient))
    assert jnp.allclose(M.inner(X, gradient, U), jnp.sum(ambient_gradient * U), atol=2e-8)
    assert bool(M.is_tangent(Y, transported))
    assert jnp.allclose(M.norm(X, U), M.norm(Y, transported), atol=2e-8)


@pytest.mark.parametrize("geometry", [Stiefel, StiefelEuclidean])
def test_stiefel_optimization_smoke(geometry):
    M = geometry(size=(4, 2))
    target = M.random_point(jax.random.key(0))
    initial = M.exp(target, M.random_tangent(jax.random.key(1), target, scale=0.15))

    estimate, final_cost, history = Minimize(
        M=M,
        cost=lambda X: 0.5 * jnp.sum((X - target) ** 2),
        x0=initial,
        solver=SteepestDescent(maxiter=80, tolgradnorm=1e-8, verbosity=0),
    ).solve()

    assert final_cost < 1e-12
    assert history[-1].gradnorm < 1e-7
    assert jnp.allclose(estimate, target, atol=2e-6)
