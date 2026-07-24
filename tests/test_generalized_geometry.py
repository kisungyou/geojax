from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import GeneralizedGrassmann, GeneralizedStiefel


@pytest.mark.parametrize("geometry", [GeneralizedStiefel, GeneralizedGrassmann])
def test_generalized_orthogonality_and_pullback_metric(geometry, dtype_atol):
    B = jnp.array(
        [
            [2.0, 0.2, 0.0, 0.0],
            [0.2, 1.5, 0.1, 0.0],
            [0.0, 0.1, 1.2, 0.1],
            [0.0, 0.0, 0.1, 1.0],
        ]
    )
    M = geometry(size=(4, 2), metric=B)
    X = M.random_point(jax.random.key(0))
    U = M.random_tangent(jax.random.key(1), X, scale=0.05)
    Y = M.exp(X, U)

    assert bool(M.belongs(X))
    assert bool(M.is_tangent(X, U))
    assert jnp.allclose(X.T @ B @ X, jnp.eye(2), atol=max(1e-10, dtype_atol))
    assert jnp.allclose(M.inner(X, U, U), jnp.sum(U * (B @ U)), atol=max(1e-12, dtype_atol))
    assert jnp.allclose(M.log(X, Y), U, atol=max(2e-8, dtype_atol))
    if geometry is GeneralizedStiefel:
        recovered, info = M.log_with_info(X, Y)
        assert bool(info.converged)
        assert jnp.allclose(recovered, U, atol=max(2e-8, dtype_atol))
        assert M.operation_kind("log") == "numerical-local"


@pytest.mark.parametrize("geometry", [GeneralizedStiefel, GeneralizedGrassmann])
def test_generalized_gradient_conversion_is_metric_dual(geometry, dtype_atol):
    B = jnp.diag(jnp.array([1.0, 1.5, 2.0, 2.5]))
    M = geometry(size=(4, 2), metric=B)
    X = M.random_point(jax.random.key(2))
    U = M.random_tangent(jax.random.key(3), X)
    ambient_gradient = jax.random.normal(jax.random.key(4), X.shape)
    gradient = M.egrad_to_rgrad(X, ambient_gradient)

    assert bool(M.is_tangent(X, gradient))
    assert jnp.allclose(
        M.inner(X, gradient, U),
        jnp.sum(ambient_gradient * U),
        atol=max(1e-10, dtype_atol),
    )
