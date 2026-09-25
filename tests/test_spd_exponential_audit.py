"""Independent scale, first-variation, and higher-AD checks for symmetric exp."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import SPDLogEuclidean
from geojax.geometry.spd import _frechet_spectral


def _dexp(matrix, tangent):
    return _frechet_spectral(matrix, tangent, "exp", 0.0)


@pytest.mark.parametrize("spectrum", ["small_covariance", "wide", "large", "smallest_normal"])
def test_exp_frechet_matches_diagonal_divided_differences(spectrum):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    limit = 700.0 if jax.config.x64_enabled else 80.0
    if spectrum == "small_covariance":
        values = [-27.631021, -26.532409]
    elif spectrum == "wide":
        values = [-limit, limit]
    elif spectrum == "large":
        values = [limit - 1, limit]
    else:
        # The next float toward zero has an exponential above the smallest
        # normal value, avoiding assumptions about subnormal backend support.
        boundary = np.nextafter(np.log(np.finfo(dtype).tiny), np.inf, dtype=dtype)
        values = [boundary, boundary]
    matrix = jnp.diag(jnp.asarray(values, dtype=dtype))
    tangent = jnp.ones_like(matrix)
    eigenvalues = np.asarray(jnp.diag(matrix)).astype(np.float64)
    exponentials = np.exp(eigenvalues)
    oracle = np.empty((2, 2))
    for i in range(2):
        for k in range(2):
            gap = eigenvalues[i] - eigenvalues[k]
            oracle[i, k] = (
                exponentials[i] if gap == 0 else (exponentials[i] - exponentials[k]) / gap
            )
    result = jax.jit(_dexp)(matrix, tangent)
    np.testing.assert_allclose(result, oracle, rtol=16 * np.finfo(dtype).eps, atol=0.0)
    assert np.all(np.asarray(result) > 0.0)


@pytest.mark.parametrize("sign", [-1.0, 0.0, 1.0])
def test_scaled_repeated_exp_preserves_second_and_third_derivatives(sign):
    limit = 700.0 if jax.config.x64_enabled else 80.0
    shift = sign * limit
    matrix = shift * jnp.eye(2)
    direction = jnp.array([[0.2, 0.3], [0.3, -0.1]])
    manifold = SPDLogEuclidean((2, 2))

    def objective(time):
        return jnp.trace(manifold.expm(matrix + time * direction)) / jnp.exp(shift)

    expected_second = np.trace(np.asarray(direction) @ np.asarray(direction))
    expected_third = np.trace(np.linalg.matrix_power(np.asarray(direction), 3))
    forward = jax.jit(jax.jacfwd(jax.jacfwd(objective)))(0.0)
    reverse = jax.jit(jax.grad(jax.grad(objective)))(0.0)
    third = jax.jit(jax.grad(jax.grad(jax.grad(objective))))(0.0)
    np.testing.assert_allclose([forward, reverse], expected_second, rtol=8e-6, atol=0.0)
    np.testing.assert_allclose(third, expected_third, rtol=8e-6, atol=0.0)


def test_exp_frechet_keeps_small_tangents_before_amplification():
    shift = 700.0 if jax.config.x64_enabled else 80.0
    coefficient = 1e-300 if jax.config.x64_enabled else 1e-36
    matrix = shift * jnp.eye(2)
    tangent = coefficient * jnp.eye(2)
    expected = float(np.asarray(tangent)[0, 0]) * np.exp(shift)
    result = jax.jit(_dexp)(matrix, tangent)
    np.testing.assert_allclose(jnp.diag(result), expected, rtol=3e-6, atol=0.0)


def test_exp_frechet_broadcasting_vmap_and_transpose():
    bases = jnp.stack((-27.0 * jnp.eye(2), 26.0 * jnp.eye(2)))
    tangent = jnp.array([[0.2, 0.3], [0.3, 0.4]])
    dual = jnp.array([[0.5, -0.1], [-0.1, 0.7]])
    expected = jnp.exp(jnp.array([-27.0, 26.0]))[:, None, None] * tangent
    broadcast = jax.jit(_dexp)(bases, tangent)
    mapped = jax.jit(jax.vmap(_dexp, in_axes=(0, None)))(bases, tangent)
    np.testing.assert_allclose(broadcast, expected, rtol=3e-6, atol=0.0)
    np.testing.assert_allclose(mapped, expected, rtol=3e-6, atol=0.0)
    transpose = jax.jit(jax.grad(lambda vector: jnp.sum(_dexp(bases, vector) * dual)))(tangent)
    expected_transpose = jnp.sum(jnp.exp(jnp.array([-27.0, 26.0]))) * dual
    np.testing.assert_allclose(transpose, expected_transpose, rtol=3e-6, atol=0.0)
    empty = jax.jit(_dexp)(jnp.empty((0, 2, 2)), tangent)
    assert empty.shape == (0, 2, 2)


@pytest.mark.parametrize("shifts", [(0.1, 26.0), (-80.0, 80.0)])
def test_gradients_through_vmapped_exp_keep_inactive_scaling_steps_finite(shifts):
    bases = jnp.asarray(shifts)[:, None, None] * jnp.eye(2)
    direction = jnp.array([[0.2, 0.3], [0.3, -0.1]])
    manifold = SPDLogEuclidean((2, 2))

    def single(base, time):
        return jnp.trace(manifold.expm(base + time * direction)) / jnp.exp(jnp.trace(base) / 2)

    def objective(time):
        return jnp.sum(jax.vmap(single, in_axes=(0, None))(bases, time))

    gradient = jax.jit(jax.grad(objective))(0.0)
    hessian = jax.jit(jax.grad(jax.grad(objective)))(0.0)
    np.testing.assert_allclose(gradient, 2 * jnp.trace(direction), rtol=8e-6, atol=0.0)
    np.testing.assert_allclose(hessian, 2 * jnp.trace(direction @ direction), rtol=8e-6, atol=0.0)
