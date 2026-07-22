from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import SPDAffineInvariant, SPDBuresWasserstein, SPDLogEuclidean


@pytest.mark.parametrize(
    "M",
    [
        SPDLogEuclidean(size=(3, 3)),
        SPDAffineInvariant(size=(3, 3)),
        SPDBuresWasserstein(size=(3, 3)),
    ],
)
def test_spd_constructors_and_random_points(M):
    P = M.random_point(jax.random.key(0))
    assert M.shape == (3, 3)
    assert M.dim == 6
    assert bool(M.belongs(P))
    assert jnp.all(jnp.linalg.eigvalsh(P) > 0.0)


def test_spd_requires_square_tuple_size():
    with pytest.raises(ValueError):
        SPDLogEuclidean(size=3)
    with pytest.raises(ValueError):
        SPDAffineInvariant(size=(2, 3))
    with pytest.raises(ValueError):
        SPDBuresWasserstein(size=3)


def test_bures_wasserstein_formulas_on_diagonal_matrices():
    M = SPDBuresWasserstein(size=(2, 2))
    P = jnp.diag(jnp.array([1.0, 4.0]))
    Q = jnp.diag(jnp.array([4.0, 9.0]))
    U = jnp.array([[0.4, 0.2], [0.2, -0.3]])

    expected_squared_dist = jnp.sum((jnp.sqrt(jnp.diag(P)) - jnp.sqrt(jnp.diag(Q))) ** 2)
    assert jnp.allclose(M.squared_dist(P, Q), expected_squared_dist, atol=1e-10)

    identity = jnp.eye(2)
    assert jnp.allclose(M.inner(identity, U, U), 0.25 * jnp.sum(U * U))
    assert jnp.allclose(M.optimal_transport_map(P, Q), jnp.diag(jnp.array([2.0, 1.5])))


def test_bures_wasserstein_transport_is_isometric():
    M = SPDBuresWasserstein(size=(3, 3))
    P = M.random_point(jax.random.key(20))
    Q = M.random_point(jax.random.key(21))
    U = M.random_tangent(jax.random.key(22), P)
    V = M.transport(P, Q, U)

    assert bool(M.is_tangent(Q, V))
    assert jnp.allclose(M.norm(P, U), M.norm(Q, V), atol=1e-8, rtol=1e-8)


def test_bures_wasserstein_distance_gradient_is_finite_at_repeated_eigenvalues():
    M = SPDBuresWasserstein(size=(2, 2))
    P = jnp.eye(2)
    Q = jnp.array([[2.0, 0.2], [0.2, 1.5]])

    gradient = jax.grad(lambda X: M.squared_dist(X, Q))(P)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, gradient.T, atol=1e-10)
