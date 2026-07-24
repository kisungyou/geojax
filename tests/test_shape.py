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
