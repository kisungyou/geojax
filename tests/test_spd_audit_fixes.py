"""Analytic matrix-calculus checks for repeated-spectrum audit corrections."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import SPDAffineInvariant, SPDBuresWasserstein, SPDLogEuclidean
from geojax.geometry.spd import (
    _solve_spd_sylvester,
    _spd_expm,
    _spd_invsqrtm_differentiable,
    _spd_logm_differentiable,
    _spd_sqrtm_differentiable,
)


def _spectral_case():
    q, _ = np.linalg.qr(np.array([[1.0, 2.0, -1.0], [2.0, 1.0, 3.0], [1.0, -2.0, 2.0]]))
    lam = np.array([1.0, 1.0, 3.0])
    e = np.array([[0.3, 0.2, -0.1], [0.2, -0.1, 0.4], [-0.1, 0.4, 0.2]])
    return jnp.asarray((q * lam) @ q.T), jnp.asarray(e), q, lam


@pytest.mark.parametrize("kind", ["log", "sqrt", "invsqrt", "exp"])
def test_spectral_trace_hessians_match_analytic_frechet_derivatives(kind):
    p, e, q, lam = _spectral_case()
    functions = {
        "log": lambda x: _spd_logm_differentiable(x, 1e-10),
        "sqrt": _spd_sqrtm_differentiable,
        "invsqrt": _spd_invsqrtm_differentiable,
        "exp": _spd_expm,
    }
    derivative, second = {
        "log": (1 / lam, -1 / lam**2),
        "sqrt": (0.5 / np.sqrt(lam), -0.25 / lam**1.5),
        "invsqrt": (-0.5 / lam**1.5, 0.75 / lam**2.5),
        "exp": (np.exp(lam), np.exp(lam)),
    }[kind]
    loewner = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            loewner[i, j] = (
                second[i]
                if lam[i] == lam[j]
                else (derivative[i] - derivative[j]) / (lam[i] - lam[j])
            )
    expected_grad = (q * derivative) @ q.T
    expected_hessian = q @ (loewner * (q.T @ np.asarray(e) @ q)) @ q.T

    def loss(x):
        return jnp.trace(functions[kind](x))

    grad = jax.grad(loss)
    np.testing.assert_allclose(grad(p), expected_grad, rtol=3e-5, atol=3e-6)
    actual = jax.jit(lambda x, v: jax.jvp(grad, (x,), (v,))[1])(p, e)
    np.testing.assert_allclose(actual, expected_hessian, rtol=5e-5, atol=5e-6)
    # Reverse-over-reverse transposes the implicit solve; this was a separate
    # failure path from forward differentiation of the custom matrix JVP.
    reverse = jax.grad(lambda x: jnp.sum(grad(x) * e))(p)
    np.testing.assert_allclose(reverse, expected_hessian, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize(
    "geometry,coefficient",
    [
        (SPDLogEuclidean, 2 * (1 + np.log(2))),
        (SPDAffineInvariant, 2 * (1 + np.log(2))),
        (SPDBuresWasserstein, 1 / np.sqrt(2)),
    ],
)
def test_spd_squared_distance_hessian_at_repeated_spectrum(geometry, coefficient):
    m = geometry((2, 2))
    p, q = jnp.eye(2), 2 * jnp.eye(2)
    e = jnp.array([[0.3, 0.2], [0.2, -0.1]])
    gradient = jax.grad(lambda x: m.squared_dist(x, q))
    hessian = jax.jit(lambda x: jax.jvp(gradient, (x,), (e,))[1])(p)
    np.testing.assert_allclose(hessian, coefficient * e, rtol=5e-5, atol=5e-6)


def test_log_euclidean_maps_and_metric_have_finite_second_derivatives():
    m = SPDLogEuclidean((2, 2))
    p = jnp.eye(2)
    e = jnp.array([[0.3, 0.2], [0.2, -0.1]])

    # At a scalar multiple of I, g_aI(E,E) = tr(E²)/a².
    def metric(t):
        return m.inner((1 + t) * p, e, e)

    actual = jax.grad(jax.grad(metric))(0.0)
    np.testing.assert_allclose(actual, 6 * jnp.sum(e * e), rtol=5e-5)

    # Dlog and Dexp are inverse maps, including their parameter derivatives.
    def identity(t):
        return m.dexp(m.logm(p + t * e), m.dlog(p + t * e, e))

    np.testing.assert_allclose(identity(0.0), e, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(jax.jacfwd(jax.jacrev(identity))(0.0), 0.0, atol=5e-6)


def test_matrix_functions_support_third_derivatives_at_identity():
    p = jnp.eye(2)
    e = jnp.array([[0.3, 0.2], [0.2, -0.1]])

    def value(t):
        return jnp.trace(_spd_logm_differentiable(p + t * e, 1e-10))

    actual = jax.jit(jax.grad(jax.grad(jax.grad(value))))(0.0)
    np.testing.assert_allclose(actual, 2 * jnp.trace(e @ e @ e), rtol=5e-5, atol=5e-6)


def test_spd_sylvester_implicit_derivative_satisfies_differentiated_equation():
    p, e, _, _ = _spectral_case()
    u = jnp.eye(3)
    a, da = jax.jvp(lambda x: _solve_spd_sylvester(x, u), (p,), (e,))
    np.testing.assert_allclose(p @ a + a @ p, u, rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(p @ da + da @ p + e @ a + a @ e, 0.0, atol=5e-6)
    actual = jax.jvp(jax.grad(lambda x: jnp.trace(_solve_spd_sylvester(x, u))), (p,), (e,))[1]
    inverse = jnp.linalg.inv(p)
    expected = 0.5 * (inverse @ e @ inverse @ inverse + inverse @ inverse @ e @ inverse)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-6)


def test_spd_logarithm_nested_batch_derivatives():
    points = jnp.broadcast_to(jnp.eye(2), (2, 3, 2, 2))
    directions = jnp.ones_like(points) * 0.1

    def log(p):
        return _spd_logm_differentiable(p, 1e-10)

    result = jax.jit(lambda p, e: jax.jvp(log, (p,), (e,))[1])(points, directions)
    np.testing.assert_allclose(result, directions, rtol=3e-5, atol=3e-6)
    gradient = jax.grad(lambda p: jnp.sum(log(p) ** 2))
    hessian = jax.jvp(gradient, (points,), (directions,))[1]
    np.testing.assert_allclose(hessian, 2 * directions, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("exponent", [-70.0, 0.0, 70.0])
def test_log_frechet_resolves_nearby_eigenvalues_at_large_and_small_scales(exponent):
    dtype = jnp.asarray(1.0).dtype
    gap = 1e-12 if jax.config.jax_enable_x64 else 1e-4
    scale = jnp.exp(jnp.asarray(exponent, dtype=dtype))
    p = scale * jnp.diag(jnp.array([1.0, 1.0 + gap], dtype=dtype))
    e = scale * jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    m = SPDLogEuclidean((2, 2))
    # The common scale cancels exactly: log1p(gap)/gap is close to one.
    actual = m.dlog(p, e)
    expected = np.log1p(gap) / gap * np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=0.0)


@pytest.mark.parametrize("batched_base", [False, True])
def test_spd_frechet_maps_and_sylvester_preserve_broadcasting(batched_base):
    p = jnp.eye(2)
    u = jnp.stack((p, 2 * p))
    if batched_base:
        p, u = u, p
    m = SPDLogEuclidean((2, 2))
    np.testing.assert_allclose(m.dlog(p, u), u / p[..., 0, 0, None, None], atol=2e-6)
    np.testing.assert_allclose(
        m.dexp(jnp.zeros_like(p), u), jnp.broadcast_to(u, (2, 2, 2)), atol=2e-6
    )
    expected = jnp.broadcast_to(u / (2 * p[..., 0, 0, None, None]), (2, 2, 2))
    np.testing.assert_allclose(_solve_spd_sylvester(p, u), expected, atol=2e-6)


def test_spd_sylvester_and_symmetrization_avoid_large_scale_overflow():
    from geojax.geometry.spd import _sym

    largest = jnp.finfo(jnp.asarray(1.0).dtype).max
    p = largest * jnp.eye(2)
    np.testing.assert_array_equal(_sym(p), p)
    np.testing.assert_array_equal(jax.jit(_sym)(p), p)
    np.testing.assert_allclose(_solve_spd_sylvester(p, p), 0.5 * jnp.eye(2), atol=2e-6)
    np.testing.assert_allclose(jax.jit(_solve_spd_sylvester)(p, p), 0.5 * jnp.eye(2), atol=2e-6)


def test_power_of_two_rescaling_has_the_correct_derivative_at_zero_entries():
    from geojax.geometry._numerics import power_of_two_rescale

    point = jnp.array([2.0, 0.0])
    direction = jnp.array([0.0, 1.0])
    derivative = jax.jvp(lambda x: power_of_two_rescale(x, axis=0), (point,), (direction,))[1]
    np.testing.assert_allclose(derivative, [0.0, 0.25], atol=0.0, rtol=0.0)


def test_spd_logarithm_is_finite_at_the_maximum_diagonal_scale():
    largest = jnp.finfo(jnp.asarray(1.0).dtype).max
    point = largest * jnp.eye(2)
    expected = np.log(float(largest)) * np.eye(2)
    actual = jax.jit(SPDLogEuclidean((2, 2)).logm)(point)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.0)


@pytest.mark.parametrize("end", ["small", "large"])
def test_spd_membership_and_projection_preserve_normal_range_endpoints(end):
    limits = np.finfo(jnp.asarray(1.0).dtype)
    scale = limits.tiny if end == "small" else 0.75 * limits.max
    point = jnp.asarray(scale) * jnp.eye(2)
    for geometry in (SPDLogEuclidean, SPDAffineInvariant, SPDBuresWasserstein):
        manifold = geometry((2, 2))
        assert bool(jax.jit(manifold.belongs)(point))
        np.testing.assert_array_equal(jax.jit(manifold.project)(point), point)
    logarithm = jax.jit(SPDLogEuclidean((2, 2)).logm)(point)
    np.testing.assert_allclose(logarithm, np.log(scale) * np.eye(2), rtol=2e-6)
    root = jax.jit(_spd_sqrtm_differentiable)(point)
    np.testing.assert_allclose(np.asarray(root) / np.sqrt(scale), np.eye(2), rtol=2e-6)


def test_smallest_normal_covariance_keeps_sylvester_and_logarithm_derivatives():
    scale = np.finfo(jnp.asarray(1.0).dtype).tiny
    point = jnp.asarray(scale) * jnp.eye(2)
    manifold = SPDLogEuclidean((2, 2))

    def solve(t):
        return jnp.trace(_solve_spd_sylvester(t * point, point))

    def logarithm(t):
        return jnp.trace(manifold.logm(t * point))

    np.testing.assert_allclose(jax.jit(solve)(1.0), 1.0, rtol=2e-6)
    np.testing.assert_allclose(jax.jit(jax.grad(solve))(1.0), -1.0, rtol=2e-6)
    np.testing.assert_allclose(jax.jit(jax.grad(logarithm))(1.0), 2.0, rtol=2e-6)
    np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(logarithm)))(1.0), -2.0, rtol=5e-5)


def test_safe_symmetrization_keeps_linear_forward_reverse_and_higher_derivatives():
    from geojax.geometry._numerics import symmetric_part

    point = jnp.array([[2.0, 0.3], [0.1, 1.0]])
    direction = jnp.array([[0.1, 0.7], [-0.2, 0.3]])
    cotangent = jnp.array([[0.3, -0.4], [0.1, 0.6]])
    expected = 0.5 * (direction + direction.T)
    _, forward = jax.jvp(symmetric_part, (point,), (direction,))
    np.testing.assert_allclose(forward, expected, atol=1e-7)
    _, pullback = jax.vjp(symmetric_part, point)
    np.testing.assert_allclose(pullback(cotangent)[0], 0.5 * (cotangent + cotangent.T), atol=1e-7)

    def quadratic(t):
        return jnp.sum(symmetric_part(point + t * direction) ** 2)

    for hessian in (jax.grad(jax.grad(quadratic)), jax.jacfwd(jax.grad(quadratic))):
        np.testing.assert_allclose(jax.jit(hessian)(0.0), 2 * jnp.sum(expected**2), rtol=2e-6)
