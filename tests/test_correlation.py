from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import CorrelationAffineQuotient, CorrelationECM, CorrelationLEC


@pytest.mark.parametrize(
    "M",
    [
        CorrelationECM(size=(3, 3)),
        CorrelationLEC(size=(3, 3)),
    ],
)
def test_correlation_random_points_have_unit_diagonal(M):
    C = M.random_point(jax.random.key(0))
    assert bool(M.belongs(C))
    assert jnp.allclose(jnp.diag(C), 1.0, atol=1e-8)


def test_correlation_closed_form_mean_belongs():
    M = CorrelationECM(size=(3, 3))
    Cs = M.random_point(jax.random.key(1), sample_shape=(4,))
    mean = M.frechet_mean_closed_form(Cs)
    assert bool(M.belongs(mean))


def test_correlation_geometries_are_distinct():
    ecm = CorrelationECM(size=(3, 3))
    lec = CorrelationLEC(size=(3, 3))
    samples = ecm.random_point(jax.random.key(2), sample_shape=(6,), scale=0.7)

    mean_ecm = ecm.frechet_mean_closed_form(samples)
    mean_lec = lec.frechet_mean_closed_form(samples)

    assert type(ecm) is not type(lec)
    assert bool(ecm.belongs(mean_ecm))
    assert bool(lec.belongs(mean_lec))
    assert not jnp.allclose(mean_ecm, mean_lec, atol=1e-7, rtol=1e-7)


def test_correlation_requires_square_tuple_size():
    with pytest.raises(ValueError):
        CorrelationECM(size=3)
    with pytest.raises(ValueError):
        CorrelationLEC(size=(2, 3))


def test_affine_quotient_horizontal_lift_and_gradient_duality(dtype_atol):
    M = CorrelationAffineQuotient(size=(3, 3))
    C = M.random_point(jax.random.key(4))
    U = M.random_tangent(jax.random.key(5), C)
    V = M.random_tangent(jax.random.key(6), C)
    lift = M.horizontal_lift(C, U)
    inverse = jnp.linalg.inv(C)
    ambient_gradient = jax.random.normal(jax.random.key(7), C.shape)
    gradient = M.egrad_to_rgrad(C, ambient_gradient)

    assert M.operation_kind("exp") == "proxy"
    assert jnp.allclose(jnp.diag(inverse @ lift), 0.0, atol=max(1e-10, dtype_atol))
    assert bool(M.is_tangent(C, gradient))
    assert jnp.allclose(
        M.inner(C, gradient, V),
        jnp.sum(ambient_gradient * V),
        atol=max(1e-10, dtype_atol),
    )


def test_affine_quotient_gradient_supports_batched_points(dtype_atol):
    M = CorrelationAffineQuotient(size=(3, 3))
    C = M.random_point(jax.random.key(20), sample_shape=(4,))
    egrad = jax.random.normal(jax.random.key(21), shape=(4, 3, 3))
    tangent = M.random_tangent(jax.random.key(22), C)

    rgrad = jax.jit(M.egrad_to_rgrad)(C, egrad)

    assert rgrad.shape == C.shape
    assert jnp.all(M.is_tangent(C, rgrad))
    assert jnp.allclose(
        M.inner(C, rgrad, tangent),
        jnp.sum(egrad * tangent, axis=(-2, -1)),
        atol=max(2e-8, 10.0 * dtype_atol),
        rtol=max(2e-8, 10.0 * dtype_atol),
    )


@pytest.mark.parametrize(
    "M",
    [
        CorrelationECM(size=(2, 2), eps=1e-3),
        CorrelationLEC(size=(2, 2), eps=1e-3),
        CorrelationAffineQuotient(size=(2, 2), eps=1e-3),
    ],
)
def test_correlation_projection_preserves_valid_near_boundary_points(M, dtype_atol):
    C = jnp.array([[1.0, 0.9999], [0.9999, 1.0]])

    repaired = M.project(C)

    assert bool(M.belongs(C))
    assert bool(M.belongs(repaired))
    assert jnp.allclose(repaired, C, atol=max(2e-7, 5.0 * dtype_atol))
