from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.geometry import KendallShape


def test_kendall_shape_constraints_dimension_and_rotation_invariance(dtype_atol):
    M = KendallShape(size=(5, 2))
    X = M.random_point(jax.random.key(0))
    angle = 0.7
    rotation = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ]
    )

    assert M.dim == 6
    assert bool(M.belongs(X))
    assert jnp.allclose(jnp.mean(X, axis=0), 0.0, atol=max(1e-12, dtype_atol))
    assert jnp.allclose(jnp.linalg.norm(X), 1.0, atol=max(1e-12, dtype_atol))
    assert jnp.allclose(M.dist(X, X @ rotation), 0.0, atol=max(1e-10, dtype_atol))


def test_kendall_horizontal_tangent_and_local_roundtrip(dtype_atol):
    M = KendallShape(size=(5, 2))
    X = M.random_point(jax.random.key(1))
    U = M.random_tangent(jax.random.key(2), X, scale=0.05)
    Y = M.exp(X, U)

    assert bool(M.is_tangent(X, U))
    assert jnp.allclose(X.T @ U, U.T @ X, atol=max(1e-10, dtype_atol))
    assert bool(M.belongs(Y))
    assert jnp.allclose(M.log(X, Y), U, atol=max(2e-8, dtype_atol))


def test_kendall_planar_collinear_shapes_are_regular(dtype_atol):
    M = KendallShape(size=(3, 2))
    centered_basis = jnp.array(
        [
            [1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(6.0)],
            [-1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(6.0)],
            [0.0, -2.0 / jnp.sqrt(6.0)],
        ]
    )
    X = centered_basis / jnp.sqrt(2.0)
    singular = jnp.column_stack((centered_basis[:, 0], jnp.zeros(3)))
    cosine = jnp.sum(X * singular)
    direction = singular - cosine * X
    angle = jnp.arccos(cosine)
    U = angle * direction / jnp.linalg.norm(direction)

    endpoint = M.exp(X, U)

    assert bool(M.is_tangent(X, U))
    assert bool(M.belongs(singular))
    assert bool(M.belongs(endpoint))
    assert jnp.allclose(endpoint, singular, atol=max(2e-8, 5.0 * dtype_atol))


def test_kendall_three_dimensional_rank_one_shape_is_singular():
    M = KendallShape(size=(4, 3))
    centered_line = jnp.array([-3.0, -1.0, 1.0, 3.0])
    centered_line /= jnp.linalg.norm(centered_line)
    singular = jnp.column_stack((centered_line, jnp.zeros((4, 2))))

    assert not bool(M.belongs(singular))


def test_kendall_log_rejects_nonunique_procrustes_alignment(dtype_atol):
    M = KendallShape(size=(5, 2))
    centering = jnp.eye(5) - jnp.ones((5, 5)) / 5.0
    basis, _ = jnp.linalg.qr(centering[:, :4])
    X = basis[:, :2] / jnp.sqrt(2.0)
    Y = basis[:, 2:4] / jnp.sqrt(2.0)

    tangent = M.log(X, Y)

    assert bool(M.belongs(X))
    assert bool(M.belongs(Y))
    assert jnp.all(~jnp.isfinite(tangent))
    assert jnp.allclose(M.dist(X, Y), jnp.pi / 2.0, atol=max(1e-8, dtype_atol))


def test_kendall_squared_distance_agrees_with_distance(dtype_atol):
    M = KendallShape(size=(5, 2))
    X = M.random_point(jax.random.key(10))
    Y = M.random_point(jax.random.key(11))
    distance = M.dist(X, Y)

    assert jnp.allclose(
        M.squared_dist(X, Y),
        distance * distance,
        atol=max(1e-10, dtype_atol),
    )
