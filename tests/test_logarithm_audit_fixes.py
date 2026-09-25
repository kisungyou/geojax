"""Independent regression oracles for the O3/O5 logarithm audit findings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import Grassmann, SpecialOrthogonal
from geojax.geometry._numerics import matrix_expm
from geojax.geometry.grassmann import _matrix_arctan_polar_single


def _tol():
    return 2e-8 if jax.config.jax_enable_x64 else 8e-5


def _plane_generator(n, first=0, second=1):
    return jnp.zeros((n, n)).at[first, second].set(-1.0).at[second, first].set(1.0)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_grassmann_log_repeated_angles_forward_reverse_and_jit(rank):
    manifold = Grassmann((2 * rank + 1, rank))
    x = jnp.eye(2 * rank + 1)[:, :rank]
    z = jnp.eye(2 * rank + 1)[:, rank : 2 * rank]

    def logarithm(angle):
        return manifold.log(x, jnp.cos(angle) * x + jnp.sin(angle) * z)

    angles = jnp.array([0.0, 1e-9, 0.2, 1.4])
    values = jax.jit(jax.vmap(logarithm))(angles)
    forward = jax.jit(jax.vmap(jax.jacfwd(logarithm)))(angles)
    reverse = jax.jit(jax.vmap(jax.jacrev(logarithm)))(angles)
    np.testing.assert_allclose(values, angles[:, None, None] * z, atol=_tol(), rtol=_tol())
    expected = jnp.broadcast_to(z, forward.shape)
    np.testing.assert_allclose(forward, expected, atol=_tol(), rtol=_tol())
    np.testing.assert_allclose(reverse, expected, atol=_tol(), rtol=_tol())


def test_grassmann_log_noncommuting_feature_hessian_at_repeated_angles():
    manifold = Grassmann((5, 2))
    x = jnp.eye(5)[:, :2]
    u = 0.2 * jnp.eye(5)[:, 2:4]
    v = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.3, -0.7], [0.2, 0.1], [-0.2, 0.4]])
    coefficient = jnp.arange(10.0).reshape(5, 2) / 10.0

    def features(t):
        log = manifold.log(x, manifold.exp(x, u + t * v))
        return jnp.sum(coefficient * log) + 0.3 * jnp.sum(log * log)

    expected_first = jnp.sum(coefficient * v) + 0.6 * jnp.sum(u * v)
    expected_second = 0.6 * jnp.sum(v * v)
    np.testing.assert_allclose(jax.grad(features)(0.0), expected_first, atol=_tol(), rtol=_tol())
    for second in (jax.jacfwd(jax.jacrev(features)), jax.jacrev(jax.jacfwd(features))):
        np.testing.assert_allclose(jax.jit(second)(0.0), expected_second, atol=_tol(), rtol=_tol())

    def squared_distance(t):
        return manifold.squared_dist(x, manifold.exp(x, u + t * v))

    np.testing.assert_allclose(
        jax.jit(jax.hessian(squared_distance))(0.0),
        2 * jnp.sum(v * v),
        atol=_tol(),
        rtol=_tol(),
    )


def test_grassmann_arctan_derivative_is_basis_equivariant_and_adjoint():
    # Two equal nonzero singular values and a structural zero row.
    matrix = jnp.array([[0.3, 0.0], [0.0, 0.3], [0.0, 0.0]])
    direction = jnp.array([[0.1, 0.2], [-0.3, 0.4], [0.2, -0.1]])
    cotangent = jnp.array([[0.2, -0.5], [0.7, 0.3], [-0.4, 0.6]])
    left = matrix_expm(_plane_generator(3, 0, 2) * 0.7)
    right = matrix_expm(_plane_generator(2) * 0.4)

    def transform(value):
        return left @ value @ right

    derivative = jax.jvp(_matrix_arctan_polar_single, (matrix,), (direction,))[1]
    rotated_derivative = jax.jvp(
        _matrix_arctan_polar_single, (transform(matrix),), (transform(direction),)
    )[1]
    np.testing.assert_allclose(rotated_derivative, transform(derivative), atol=_tol(), rtol=_tol())
    _, pullback = jax.vjp(_matrix_arctan_polar_single, matrix)
    np.testing.assert_allclose(
        jnp.sum(cotangent * derivative),
        jnp.sum(pullback(cotangent)[0] * direction),
        atol=_tol(),
        rtol=_tol(),
    )
    step = 1e-4 if jax.config.jax_enable_x64 else 2e-3
    numerical = (
        _matrix_arctan_polar_single(matrix + step * direction)
        - _matrix_arctan_polar_single(matrix - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(derivative, numerical, atol=3 * _tol(), rtol=3 * _tol())


def test_grassmann_log_near_cut_and_at_cut():
    manifold = Grassmann((4, 2))
    x = jnp.eye(4)[:, :2]
    z = jnp.eye(4)[:, 2:]
    gap = 1e-5 if jax.config.jax_enable_x64 else 2e-3
    angle = jnp.pi / 2 - gap

    def curve(t):
        return manifold.log(x, jnp.cos(t) * x + jnp.sin(t) * z)

    np.testing.assert_allclose(curve(angle), angle * z, atol=_tol(), rtol=_tol())
    np.testing.assert_allclose(jax.jacfwd(curve)(angle), z, atol=5 * _tol(), rtol=5 * _tol())
    assert bool(jnp.all(jnp.isnan(manifold.log(x, z))))
    np.testing.assert_allclose(manifold.squared_dist(x, z), jnp.pi**2 / 2, atol=_tol())


def test_grassmann_mixed_near_cut_angles_do_not_square_conditioning():
    manifold = Grassmann((5, 2))
    frame = jnp.eye(5)[:, :2]
    normal = jnp.eye(5)[:, 2:4]
    ambient = matrix_expm(0.6 * (_plane_generator(5, 0, 4) + _plane_generator(5, 1, 3)))
    gauge = matrix_expm(0.4 * _plane_generator(2))
    x = ambient @ frame @ gauge
    gap = 1e-8 if jax.config.jax_enable_x64 else 1e-4

    def curve(angle):
        angles = jnp.array([angle, 0.2])
        point = ambient @ (frame * jnp.cos(angles) + normal * jnp.sin(angles)) @ gauge
        return manifold.log(x, point)

    angle = jnp.pi / 2 - gap
    expected = ambient @ normal @ jnp.diag(jnp.array([1.0, 0.0])) @ gauge
    # Conditioning of XtY itself limits precision near the cut, but its
    # condition number must not be squared by forming a tangent Gram matrix.
    tolerance = 3e-7 if jax.config.jax_enable_x64 else 5e-3
    derivative = jax.jacfwd(curve)(angle)
    np.testing.assert_allclose(derivative, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_rotation_squared_distance_hessians_general_dimension_and_repeated_angles(n):
    manifold = SpecialOrthogonal(n)
    identity = jnp.eye(n)
    generator = sum((_plane_generator(n, index, index + 1) for index in range(0, n - 1, 2)))
    expected_second = 2 * jnp.sum(generator * generator)

    def distance(t):
        return manifold.squared_dist(identity, manifold.exp(identity, t * generator))

    angles = jnp.array([0.0, 0.2, 1.8, 3.0])
    second = jax.jit(jax.vmap(jax.grad(jax.grad(distance))))(angles)
    np.testing.assert_allclose(second, expected_second, atol=5 * _tol(), rtol=5 * _tol())
    first = jax.jit(jax.vmap(jax.grad(distance)))(angles)
    np.testing.assert_allclose(first, expected_second * angles, atol=_tol(), rtol=_tol())


@pytest.mark.parametrize("angle", [0.7, 2.4])
def test_rotation_squared_distance_noncommuting_hessian_has_analytic_value(angle):
    # Quaternion composition gives cos(relative_angle/2) = cos(angle/2)cos(t/2).
    # With the Frobenius metric d² = 2 relative_angle², its second derivative
    # at t=0 is 2 angle cot(angle/2), rather than the flat-space value 4.
    manifold = SpecialOrthogonal(3)
    target = matrix_expm(angle * _plane_generator(3, 0, 1))
    direction = _plane_generator(3, 1, 2)

    def cost(t):
        return manifold.squared_dist(matrix_expm(t * direction), target)

    expected = 2 * angle / jnp.tan(angle / 2)
    for second in (jax.jacfwd(jax.jacrev(cost)), jax.jacrev(jax.jacfwd(cost))):
        np.testing.assert_allclose(jax.jit(second)(0.0), expected, atol=_tol(), rtol=_tol())


def test_rotation_log_inverse_differential_and_basis_equivariance():
    manifold = SpecialOrthogonal(4)
    generator = 0.8 * (_plane_generator(4) + _plane_generator(4, 2, 3))
    direction = _plane_generator(4, 0, 2) + 0.4 * _plane_generator(4, 1, 3)
    point = matrix_expm(generator)
    tangent = point @ direction
    derivative = jax.jit(lambda p, u: jax.jvp(manifold.group_log, (p,), (u,))[1])(point, tangent)
    recovered = jax.jvp(matrix_expm, (generator,), (derivative,))[1]
    np.testing.assert_allclose(recovered, tangent, atol=_tol(), rtol=_tol())

    basis = matrix_expm(0.3 * _plane_generator(4, 0, 3))

    def transform(value):
        return basis @ value @ basis.T

    changed = jax.jvp(manifold.group_log, (transform(point),), (transform(tangent),))[1]
    np.testing.assert_allclose(changed, transform(derivative), atol=_tol(), rtol=_tol())
    _, pullback = jax.vjp(manifold.group_log, point)
    np.testing.assert_allclose(
        jnp.sum(direction * derivative),
        jnp.sum(pullback(direction)[0] * tangent),
        atol=_tol(),
        rtol=_tol(),
    )


def test_rotation_log_native_batch_and_vmap_hessians():
    manifold = SpecialOrthogonal(3)
    generator = _plane_generator(3)

    def curve(ts):
        return manifold.group_log(matrix_expm(ts[:, None, None] * generator))

    angles = jnp.array([0.0, 0.2, 0.2])
    log, differential = jax.jit(lambda ts: jax.jvp(curve, (ts,), (jnp.ones_like(ts),)))(angles)
    np.testing.assert_allclose(log, angles[:, None, None] * generator, atol=_tol(), rtol=_tol())
    np.testing.assert_allclose(
        differential, jnp.broadcast_to(generator, log.shape), atol=_tol(), rtol=_tol()
    )

    def loss(ts):
        return jnp.sum(curve(ts) ** 2)

    np.testing.assert_allclose(
        jax.jit(jax.hessian(loss))(angles), 4 * jnp.eye(3), atol=_tol(), rtol=_tol()
    )


def test_rotation_cut_has_finite_distance_but_undefined_log_and_gradient():
    manifold = SpecialOrthogonal(4)
    identity = jnp.eye(4)
    cut = -identity
    assert bool(jnp.all(jnp.isnan(manifold.group_log(cut))))
    np.testing.assert_allclose(manifold.squared_dist(identity, cut), 4 * jnp.pi**2, atol=_tol())
    assert bool(jnp.all(jnp.isnan(jax.grad(lambda q: manifold.squared_dist(identity, q))(cut))))


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rotation_projection_repeated_singular_values_have_analytic_two_derivatives(scale):
    manifold = SpecialOrthogonal(2)
    identity = jnp.eye(2)
    generator = _plane_generator(2)

    def curve(time):
        return manifold.project(scale * identity + time * generator)

    # The polar factor is (scale I+tG)/sqrt(scale²+t²), independent of SVD.
    np.testing.assert_allclose(jax.jit(jax.jacfwd(curve))(0.0), generator / scale, atol=_tol())
    np.testing.assert_allclose(jax.jit(jax.jacrev(curve))(0.0), generator / scale, atol=_tol())
    for hessian in (jax.jacfwd(jax.jacrev(curve)), jax.jacrev(jax.jacfwd(curve))):
        np.testing.assert_allclose(jax.jit(hessian)(0.0), -identity / scale**2, atol=_tol())


def test_rotation_projection_unique_reflection_and_rank_deficiency():
    manifold = SpecialOrthogonal(3)
    generator = _plane_generator(3, 1, 2)
    for diagonal in (jnp.array([3.0, 2.0, -1.0]), jnp.array([2.0, 2.0, 0.0])):
        point = jnp.diag(diagonal)

        def curve(time):
            return manifold.project(point + time * generator)

        # The affected 2x2 proper polar block has rotation angle
        # atan(2t/(d1+d2)); this also covers a unique rank-(n-1) factor.
        rate = 2.0 / (diagonal[1] + diagonal[2])
        np.testing.assert_allclose(curve(0.0), jnp.eye(3), atol=_tol())
        np.testing.assert_allclose(jax.jit(jax.jacfwd(curve))(0.0), rate * generator, atol=_tol())
        np.testing.assert_allclose(jax.jit(jax.jacrev(curve))(0.0), rate * generator, atol=_tol())
        np.testing.assert_allclose(
            jax.jit(jax.jacfwd(jax.jacrev(curve)))(0.0),
            rate**2 * generator @ generator,
            atol=4 * _tol(),
        )

    # Equal smallest singular values on the reversed-orientation branch
    # have multiple closest rotations. A smooth derivative is not asserted.
    nonunique = jnp.diag(jnp.array([2.0, 1.0, -1.0]))
    assert bool(manifold.belongs(manifold.project(nonunique)))
    derivative = jax.jvp(manifold.project, (nonunique,), (generator,))[1]
    assert bool(jnp.any(~jnp.isfinite(derivative)))


def test_rotation_projection_native_batch_and_adjoint_at_repeated_spectra():
    manifold = SpecialOrthogonal(3)
    points = jnp.array([1.0, 2.0, 3.0])[:, None, None] * jnp.eye(3)
    directions = jax.random.normal(jax.random.key(2401), points.shape)
    cotangent = jax.random.normal(jax.random.key(2402), points.shape)
    expected = 0.5 * (directions - jnp.swapaxes(directions, -1, -2))
    expected = expected / jnp.array([1.0, 2.0, 3.0])[:, None, None]
    derivative = jax.jit(lambda p, u: jax.jvp(manifold.project, (p,), (u,))[1])(points, directions)
    np.testing.assert_allclose(derivative, expected, atol=_tol(), rtol=_tol())
    _, pullback = jax.vjp(manifold.project, points)
    np.testing.assert_allclose(
        jnp.sum(cotangent * derivative),
        jnp.sum(pullback(cotangent)[0] * directions),
        atol=4 * _tol(),
    )
