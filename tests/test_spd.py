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


@pytest.mark.parametrize(
    "manifold",
    [
        SPDLogEuclidean((2, 2)),
        SPDAffineInvariant((2, 2)),
        SPDBuresWasserstein((2, 2)),
    ],
)
def test_spd_operations_do_not_clip_valid_small_positive_eigenvalues(manifold):
    point = jnp.diag(jnp.array([1e-12, 3e-12]))
    other = jnp.diag(jnp.array([2e-12, 6e-12]))

    assert bool(manifold.belongs(point))
    assert jnp.allclose(manifold.project(point), point, rtol=2e-6, atol=0.0)
    assert bool(jnp.isfinite(manifold.dist(point, other)))
    assert jnp.allclose(
        manifold.exp(point, manifold.log(point, other)),
        other,
        rtol=2e-5,
        atol=1e-20,
    )


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


def test_bures_sylvester_uses_batched_spd_eigenbasis_solve(dtype_atol):
    M = SPDBuresWasserstein(size=(3, 3))
    P = jnp.stack(
        [
            jnp.eye(3),
            jnp.array(
                [
                    [2.0, 0.2, 0.0],
                    [0.2, 1.5, 0.1],
                    [0.0, 0.1, 1.0],
                ]
            ),
        ]
    )
    raw = jax.random.normal(jax.random.key(5), P.shape)
    U = 0.5 * (raw + jnp.swapaxes(raw, -1, -2))

    solution = M.sylvester(P, U)
    residual = P @ solution + solution @ P - U

    assert solution.shape == P.shape
    assert jnp.allclose(
        residual,
        jnp.zeros_like(residual),
        atol=max(2e-6, 5.0 * dtype_atol),
        rtol=2e-6,
    )


def test_bures_sylvester_jvp_is_finite_at_repeated_eigenvalues():
    M = SPDBuresWasserstein(size=(2, 2))
    P = jnp.eye(2)
    U = jnp.array([[0.5, 0.2], [0.2, -0.3]])
    P_dot = jnp.array([[0.1, -0.05], [-0.05, 0.2]])

    solution, solution_dot = jax.jvp(
        lambda base: M.sylvester(base, U),
        (P,),
        (P_dot,),
    )
    expected_dot = -0.5 * (P_dot @ solution + solution @ P_dot)

    assert jnp.all(jnp.isfinite(solution_dot))
    assert jnp.allclose(solution_dot, expected_dot, atol=1e-8, rtol=1e-8)


def test_bures_wasserstein_nearby_distance_avoids_trace_cancellation():
    M = SPDBuresWasserstein(size=(2, 2))
    delta = 1e-8 if jax.config.x64_enabled else 1e-4
    P = jnp.eye(2)
    Q = jnp.diag(jnp.array([1.0 + delta, 1.0 - delta]))
    expected = jnp.sum((jnp.sqrt(jnp.diag(P)) - jnp.sqrt(jnp.diag(Q))) ** 2)

    actual = M.squared_dist(P, Q)

    assert actual > 0.0
    assert jnp.allclose(actual, expected, rtol=2e-4, atol=0.0)


def test_bures_wasserstein_transport_is_isometric(dtype_atol):
    M = SPDBuresWasserstein(size=(3, 3))
    P = M.random_point(jax.random.key(20))
    Q = M.random_point(jax.random.key(21))
    U = M.random_tangent(jax.random.key(22), P)
    V = M.transport(P, Q, U)

    assert bool(M.is_tangent(Q, V))
    assert jnp.allclose(
        M.norm(P, U),
        M.norm(Q, V),
        atol=max(1e-8, dtype_atol),
        rtol=max(1e-8, dtype_atol),
    )


def test_bures_wasserstein_distance_gradient_is_finite_at_repeated_eigenvalues():
    M = SPDBuresWasserstein(size=(2, 2))
    P = jnp.eye(2)
    Q = jnp.array([[2.0, 0.2], [0.2, 1.5]])

    gradient = jax.grad(lambda X: M.squared_dist(X, Q))(P)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, gradient.T, atol=1e-10)


def test_bures_exponential_rejects_leave_and_reenter_path():
    M = SPDBuresWasserstein(size=(2, 2))
    P = jnp.eye(2)
    # L_P(U) = -2 I, so I + t L_P(U) is singular at t=1/2 but
    # invertible again at t=1. Endpoint-only checks would incorrectly return P.
    U = -4.0 * P

    endpoint = M.exp(P, U)

    assert bool(M.is_tangent(P, U))
    assert bool(jnp.all(~jnp.isfinite(endpoint)))
