"""Analytic regression tests for smooth Kendall quotient derivatives.

Repeated spectra and rank-(m-1) pre-shapes are intentional: neither is a
singularity of the regular SO(m) quotient or its unique local alignment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import KendallShape
from geojax.learning import frechet_mean


_CASES = [(1, False), (2, False), (2, True), (3, False), (3, True), (4, True)]


def _centered_basis(n):
    """Helmert columns, constructed without a numerical decomposition."""
    basis = np.zeros((n, n - 1))
    for column in range(n - 1):
        scale = np.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return jnp.asarray(basis)


def _balanced_shapes(m, rank_deficient=False):
    weights = jnp.ones(m)
    rank = m - int(rank_deficient)
    if rank_deficient:
        weights = weights.at[-1].set(0.0)
    # Disjoint +/- coordinate pairs give exactly equal positive Gram entries,
    # without a QR/eigenbasis whose roundoff could split the repeated spectrum.
    pairs = jnp.concatenate([jnp.eye(m), -jnp.eye(m)])
    zeros = jnp.zeros_like(pairs)
    x = jnp.concatenate([pairs, zeros]) * weights / jnp.sqrt(float(2 * rank))
    # All columns remain active in u, including the null column of x. Thus
    # rank-(m-1) cases exercise curves which immediately acquire full rank.
    u = jnp.concatenate([zeros, pairs]) / jnp.sqrt(float(2 * m))
    return KendallShape(x.shape), x, u


def _rotation(m, t):
    generator = jnp.zeros((m, m)).at[0, -1].set(-1.0).at[-1, 0].set(1.0)
    rotation = jnp.eye(m) + jnp.sin(t) * generator + (1 - jnp.cos(t)) * (generator @ generator)
    return rotation, generator


def _tolerance():
    return 2e-9 if jax.config.jax_enable_x64 else 4e-5


@pytest.mark.parametrize("m,rank_deficient", _CASES)
def test_balanced_shape_squared_distance_derivatives(m, rank_deficient):
    manifold, x, u = _balanced_shapes(m, rank_deficient)
    assert bool(manifold.belongs(x))
    assert bool(manifold.is_tangent(x, u))

    def objective(t):
        y = jnp.cos(t) * x + jnp.sin(t) * u
        return manifold.squared_dist(x, y)

    times = jnp.array([0.0, 1e-7, 0.1])
    value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(objective)))
    values, derivatives = value_and_grad(times)
    second = jax.jit(jax.vmap(jax.jacfwd(jax.grad(objective))))(times)
    np.testing.assert_allclose(values, times**2, rtol=_tolerance(), atol=_tolerance() * 1e-3)
    np.testing.assert_allclose(derivatives, 2 * times, rtol=_tolerance(), atol=_tolerance())
    np.testing.assert_allclose(second, 2.0, rtol=_tolerance(), atol=_tolerance())

    # Check the complete gradient, not only one directional contraction.
    t = 0.1
    y = jnp.cos(t) * x + jnp.sin(t) * u
    rgrad = manifold.egrad_to_rgrad(x, jax.grad(lambda p: manifold.squared_dist(p, y))(x))
    np.testing.assert_allclose(rgrad, -2 * t * u, rtol=_tolerance(), atol=_tolerance())


@pytest.mark.parametrize("m,rank_deficient", _CASES)
def test_balanced_shape_log_forward_reverse_and_jit(m, rank_deficient):
    manifold, x, u = _balanced_shapes(m, rank_deficient)

    def function(t):
        return manifold.log(x, jnp.cos(t) * x + jnp.sin(t) * u)

    for t in [0.0, 0.2]:
        forward = jax.jit(jax.jacfwd(function))(t)
        reverse = jax.jacrev(function)(t)
        np.testing.assert_allclose(forward, u, rtol=_tolerance(), atol=_tolerance())
        np.testing.assert_allclose(reverse, u, rtol=_tolerance(), atol=_tolerance())


@pytest.mark.parametrize("m,rank_deficient", [(2, False), (3, False), (3, True), (4, True)])
def test_procrustes_rotation_and_horizontal_projection_equivariance_derivatives(m, rank_deficient):
    manifold, x, u = _balanced_shapes(m, rank_deficient)
    _, generator = _rotation(m, 0.0)
    ambient = x @ generator + u
    cotangent = jnp.arange(x.size, dtype=x.dtype).reshape(x.shape) / x.size

    def rotation_curve(t):
        rotation, _ = _rotation(m, t)
        return manifold.align(x @ rotation, x)[1]

    def projection_curve(t):
        rotation, _ = _rotation(m, t)
        return manifold.tangent_project(x @ rotation, ambient @ rotation)

    t = 0.2
    rotation, _ = _rotation(m, t)
    expected_rotation_derivative = -generator @ rotation.T
    expected_projection_derivative = u @ generator @ rotation
    np.testing.assert_allclose(rotation_curve(t), rotation.T, rtol=_tolerance(), atol=_tolerance())
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(rotation_curve))(t),
        expected_rotation_derivative,
        rtol=_tolerance(),
        atol=_tolerance(),
    )
    np.testing.assert_allclose(
        jax.jacrev(rotation_curve)(t),
        expected_rotation_derivative,
        rtol=_tolerance(),
        atol=_tolerance(),
    )
    np.testing.assert_allclose(
        jax.jacfwd(jax.jacrev(rotation_curve))(t),
        generator @ generator @ rotation.T,
        rtol=_tolerance(),
        atol=_tolerance(),
    )
    np.testing.assert_allclose(
        projection_curve(t), u @ rotation, rtol=_tolerance(), atol=_tolerance()
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(projection_curve))(t),
        expected_projection_derivative,
        rtol=_tolerance(),
        atol=_tolerance(),
    )

    def scalar_projection(s):
        return jnp.sum(cotangent * projection_curve(s))

    np.testing.assert_allclose(
        jax.grad(scalar_projection)(t),
        jnp.sum(cotangent * expected_projection_derivative),
        rtol=_tolerance(),
        atol=_tolerance(),
    )
    np.testing.assert_allclose(
        jax.jacfwd(jax.grad(scalar_projection))(t),
        jnp.sum(cotangent * (u @ generator @ generator @ rotation)),
        rtol=_tolerance(),
        atol=_tolerance(),
    )


def test_unique_reflected_alignment_repeated_positive_singular_values():
    # det(cross)<0, with repeated two largest singular values but a unique
    # smallest one: the SO(3) optimizer is smooth, despite the reflection.
    basis = _centered_basis(7)[:, :3]
    x = basis * jnp.sqrt(jnp.array([3.0, 3.0, 1.0]) / 7)
    reflected = x * jnp.array([1.0, 1.0, -1.0])
    manifold = KendallShape(x.shape)

    def alignment(t):
        rotation, _ = _rotation(3, t)
        return manifold.align(reflected @ rotation.T, x)[1]

    times = jnp.array([-0.2, 0.0, 0.3])
    actual = jax.jit(jax.vmap(jax.jacfwd(alignment)))(times)
    expected = jax.vmap(lambda t: _rotation(3, t)[1] @ _rotation(3, t)[0])(times)
    np.testing.assert_allclose(actual, expected, rtol=_tolerance(), atol=_tolerance())
    np.testing.assert_allclose(
        jax.jacrev(alignment)(0.3), expected[-1], rtol=_tolerance(), atol=_tolerance()
    )


@pytest.mark.parametrize("m,rank_deficient", [(2, False), (3, True), (4, True)])
def test_default_mean_of_balanced_shapes_is_geodesic_midpoint(m, rank_deficient):
    manifold, x, u = _balanced_shapes(m, rank_deficient)
    y = jnp.cos(0.1) * x + jnp.sin(0.1) * u
    expected = jnp.cos(0.05) * x + jnp.sin(0.05) * u
    fitted = frechet_mean(manifold, jnp.stack([x, y]))
    assert fitted.converged, fitted.reason
    assert float(manifold.dist(fitted.point, expected)) < max(_tolerance(), 5e-7)
    np.testing.assert_allclose(
        fitted.objective, 0.0025, rtol=_tolerance(), atol=_tolerance() * 1e-3
    )


@pytest.mark.parametrize(
    "m,angles,weights",
    [
        (3, [0.0, 0.07, 0.2], [1.0, 2.0, 5.0]),
        (4, [0.0, 0.01, 0.1, 0.2, 0.4], [1.0, 2.0, 3.0, 5.0, 8.0]),
    ],
)
def test_weighted_shape_mean_matches_short_geodesic_barycenter(m, angles, weights):
    manifold, x, u = _balanced_shapes(m, rank_deficient=True)
    angles = jnp.asarray(angles)
    weights = jnp.asarray(weights)
    data = jnp.cos(angles)[:, None, None] * x + jnp.sin(angles)[:, None, None] * u

    # Along this arc, cross covariances are positive semidefinite with at least
    # m-1 positive eigenvalues: x.T @ u is zero, and both Gram matrices are
    # diagonal. Thus the unique proper alignment is the identity and distances
    # are angle differences. The exact weighted mean and minimum objective are
    # the weighted angle average and variance, including the rank-(m-1) endpoint.
    mean_angle = jnp.sum(weights * angles) / jnp.sum(weights)
    expected = jnp.cos(mean_angle) * x + jnp.sin(mean_angle) * u
    expected_objective = jnp.sum(weights * (angles - mean_angle) ** 2) / jnp.sum(weights)
    assert bool(jnp.all(manifold.belongs(data)))

    fitted = frechet_mean(manifold, data, sample_weight=weights)
    assert fitted.converged, fitted.reason
    assert float(fitted.gradient_norm) <= fitted.diagnostics["effective_tolerance"]
    assert float(manifold.dist(fitted.point, expected)) < max(_tolerance(), 5e-7)
    np.testing.assert_allclose(
        fitted.objective,
        expected_objective,
        rtol=_tolerance(),
        atol=_tolerance() * 1e-3,
    )


def test_original_balanced_planar_audit_mean():
    manifold = KendallShape((4, 2))
    x = 0.5 * jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    u = 0.5 * jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0]])
    y = jnp.cos(0.1) * x + jnp.sin(0.1) * u
    expected = jnp.cos(0.05) * x + jnp.sin(0.05) * u
    fitted = frechet_mean(manifold, jnp.stack([x, y]))
    assert fitted.converged, fitted.reason
    assert float(manifold.dist(fitted.point, expected)) < max(_tolerance(), 5e-7)
    np.testing.assert_allclose(
        fitted.objective, 0.0025, rtol=_tolerance(), atol=_tolerance() * 1e-3
    )


def test_nonunique_reflection_and_singular_shapes_remain_outside_log_domain():
    manifold, x, u = _balanced_shapes(3)
    reflected = x * jnp.array([1.0, 1.0, -1.0])
    assert bool(manifold.belongs(x))
    assert bool(manifold.belongs(reflected))
    assert bool(jnp.all(jnp.isnan(manifold.log(x, reflected))))
    assert bool(jnp.all(jnp.isnan(manifold.log(x, u))))
    singular = jnp.zeros_like(x).at[:, 0].set(x[:, 0] * jnp.sqrt(3.0))
    assert not bool(manifold.belongs(singular))


def test_reflection_uniqueness_certificate_survives_determinant_underflow():
    m = 32
    pairs = jnp.concatenate([jnp.eye(m, dtype=jnp.float32), -jnp.eye(m, dtype=jnp.float32)])
    x = pairs / jnp.sqrt(jnp.float32(2 * m))
    reflected = x.at[:, -1].multiply(-1.0)
    manifold = KendallShape(x.shape)
    # The full-rank cross covariance has determinant -(1/32)**32, below the
    # smallest float32 subnormal. Its determinant *sign* remains meaningful.
    _, _, unique = manifold._alignment(reflected, x)
    assert not bool(unique)
