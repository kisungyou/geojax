"""Analytic and invariant checks for the fixed-rank Bures audit correction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import Elliptope, FixedRank, RankKPSD, RankKPSDBuresWasserstein, Spectrahedron
from geojax.learning import frechet_mean


def _tolerance():
    return 2e-8 if jax.config.jax_enable_x64 else 3e-4


def _assert_close(actual, expected, *, multiplier=1.0):
    tolerance = multiplier * _tolerance()
    assert np.all(np.isfinite(np.asarray(actual)))
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def _example(rank):
    n = 2 * rank + 1
    manifold = RankKPSDBuresWasserstein((n, n), rank=rank)
    basis = jnp.eye(n)[:, :rank]
    point = basis @ basis.T
    cross = jnp.zeros((n, n)).at[rank : 2 * rank, :rank].set(jnp.eye(rank))
    tangent = 0.2 * point + 0.15 * (cross + cross.T)
    return manifold, point, tangent


@pytest.mark.parametrize("rank", [1, 2])
def test_repeated_positive_and_zero_distance_gradient_and_hessian(rank):
    manifold, point, _ = _example(rank)

    def objective(scale):
        return manifold.squared_dist(scale * point, 2.0 * point)

    _assert_close(jax.jit(jax.grad(objective))(1.0), rank * (1.0 - jnp.sqrt(2.0)))
    _assert_close(jax.jit(jax.grad(jax.grad(objective)))(1.0), rank / jnp.sqrt(2.0))
    gradient = jax.jit(jax.grad(lambda p: manifold.squared_dist(p, 2.0 * point)))(point)
    _assert_close(gradient, (1.0 - jnp.sqrt(2.0)) * point)


@pytest.mark.parametrize("rank", [1, 2])
def test_projector_and_projection_derivatives_are_basis_independent(rank):
    manifold, point, tangent = _example(rank)
    ambient = tangent + 0.3 * (jnp.eye(manifold.n) - point)
    _, derivative = jax.jit(lambda p, u: jax.jvp(manifold.project, (p,), (u,)))(point, ambient)
    _assert_close(derivative, tangent)
    _, support_derivative = jax.jvp(manifold._support_projector, (point,), (ambient,))
    _assert_close(support_derivative, tangent - point @ tangent @ point)
    # Rotating both a repeated positive eigenspace and its null complement
    # changes the eigensolver's basis but cannot change this linear map.
    rotation, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(314 + rank), point.shape))
    rotated_point = rotation @ point @ rotation.T
    rotated_ambient = rotation @ ambient @ rotation.T
    _, rotated_derivative = jax.jvp(manifold.project, (rotated_point,), (rotated_ambient,))
    _assert_close(rotated_derivative, rotation @ tangent @ rotation.T)


@pytest.mark.parametrize("rank", [1, 2])
def test_log_jvp_vjp_and_hessian_of_distance_along_horizontal_geodesic(rank):
    manifold, point, tangent = _example(rank)

    def curve(time):
        return manifold.exp(point, time * tangent)

    def log_curve(time):
        return manifold.log(point, curve(time))

    _assert_close(jax.jit(jax.jacfwd(log_curve))(0.3), tangent)
    _assert_close(jax.jit(jax.jacrev(log_curve))(0.3), tangent)
    expected = 2.0 * manifold.inner(point, tangent, tangent)

    def objective(time):
        return manifold.squared_dist(point, curve(time))

    for time in (0.0, 0.3):
        _assert_close(jax.jit(jax.grad(jax.grad(objective)))(time), expected)


@pytest.mark.parametrize("rank", [1, 2])
def test_bures_distance_directional_derivative_and_adjoint_identity(rank):
    manifold, point, tangent = _example(rank)
    target = manifold.exp(point, 0.7 * tangent)

    def curve(time):
        return manifold.exp(point, time * tangent)

    def log_at_time(time):
        return manifold.log(curve(time), target)

    time = jnp.asarray(0.2)
    _, jvp = jax.jvp(log_at_time, (time,), (jnp.asarray(1.0),))
    cotangent = jax.random.normal(jax.random.key(71 + rank), point.shape)
    _, pullback = jax.vjp(log_at_time, time)
    _assert_close(jnp.sum(jvp * cotangent), pullback(cotangent)[0])
    step = 1e-5 if jax.config.jax_enable_x64 else 2e-3
    finite_difference = (log_at_time(time + step) - log_at_time(time - step)) / (2.0 * step)
    _assert_close(jvp, finite_difference, multiplier=4.0)

    def objective(t):
        return manifold.squared_dist(curve(t), target)

    expected = -2.0 * (0.7 - time) * manifold.inner(point, tangent, tangent)
    _assert_close(jax.grad(objective)(time), expected)

    def distance(t):
        return manifold.dist(curve(t), target)

    _assert_close(jax.grad(distance)(time), -manifold.norm(point, tangent))


@pytest.mark.parametrize("rank", [1, 2])
def test_sylvester_differentiation_preserves_equation_and_null_constraint(rank):
    manifold, point, tangent = _example(rank)
    ambient = jnp.arange(manifold.n**2, dtype=point.dtype).reshape(point.shape) / 30.0

    def equations(time):
        p = manifold.exp(point, time * tangent)
        u = manifold.tangent_project(p, ambient)
        solution = manifold.sylvester(p, u)
        null = jnp.eye(manifold.n) - manifold._support_projector(p)
        return p @ solution + solution @ p - u, null @ solution @ null

    values, derivatives = jax.jit(lambda t: jax.jvp(equations, (t,), (jnp.ones_like(t),)))(
        jnp.asarray(0.0)
    )
    for residual in (*values, *derivatives):
        _assert_close(residual, jnp.zeros_like(point), multiplier=3.0)


@pytest.mark.parametrize("rank", [1, 2])
def test_ordinary_frechet_mean_at_repeated_spectra(rank):
    manifold, point, _ = _example(rank)
    result = frechet_mean(manifold, jnp.stack([point, 2.0 * point]))
    _assert_close(result.point, ((1.0 + jnp.sqrt(2.0)) / 2.0) ** 2 * point)
    assert result.converged


def test_jit_vmap_and_direct_batch_agree_for_regular_log_and_gradient():
    manifold, point, tangent = _example(2)
    scales = jnp.array([0.8, 1.0, 1.4])
    points = scales[:, None, None] * point
    targets = jax.vmap(lambda p: manifold.exp(p, 0.3 * tangent))(points)
    batched_logs = jax.jit(manifold.log)(points, targets)
    mapped_logs = jax.jit(jax.vmap(manifold.log))(points, targets)
    _assert_close(batched_logs, mapped_logs)
    expected = 2.0 * (jnp.sqrt(scales) - jnp.sqrt(2.0)) / (2.0 * jnp.sqrt(scales))
    gradients = jax.jit(
        jax.vmap(jax.grad(lambda scale: manifold.squared_dist(scale * point, 2.0 * point)))
    )(scales)
    _assert_close(gradients, 2.0 * expected)
    direct = jax.grad(lambda factor: jnp.sum(manifold.squared_dist(factor * points, targets)))(1.0)
    mapped = jax.grad(
        lambda factor: jnp.sum(jax.vmap(manifold.squared_dist)(factor * points, targets))
    )(1.0)
    _assert_close(direct, mapped)


def test_rotated_support_hessian_matches_finite_difference_and_analytic_geodesic():
    manifold, point, tangent = _example(2)
    rotation, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(401), point.shape))
    point, tangent = rotation @ point @ rotation.T, rotation @ tangent @ rotation.T

    def curve(time):
        return manifold.exp(point, time * tangent)

    target = manifold.exp(point, 0.4 * tangent)
    gradient = jax.grad(lambda p: manifold.squared_dist(p, target))
    _, hessian_action = jax.jit(lambda p, u: jax.jvp(gradient, (p,), (u,)))(point, tangent)
    step = 1e-5 if jax.config.jax_enable_x64 else 2e-3
    finite_difference = (gradient(curve(step)) - gradient(curve(-step))) / (2.0 * step)
    _assert_close(hessian_action, finite_difference, multiplier=15.0)

    def objective(time):
        return manifold.squared_dist(curve(time), 2.0 * point)

    expected = 2.0 * manifold.inner(point, tangent, tangent)
    _assert_close(jax.jit(jax.grad(jax.grad(objective)))(0.0), expected, multiplier=3.0)


@pytest.mark.parametrize("rank", [1, 2])
def test_exp_accepts_regular_horizontal_path_beyond_positive_overlap(rank):
    n = 2 * rank
    manifold = RankKPSDBuresWasserstein((n, n), rank=rank)
    basis = jnp.eye(n)[:, :rank]
    point = basis @ basis.T
    horizontal = jnp.concatenate([-2.0 * jnp.eye(rank), jnp.eye(rank)], axis=0)
    tangent = horizontal @ basis.T + basis @ horizontal.T
    expected_factor = basis + horizontal
    _assert_close(jax.jit(manifold.exp)(point, tangent), expected_factor @ expected_factor.T)


@pytest.mark.parametrize("endpoint", [False, True])
def test_exp_rejects_rank_loss_in_repeated_compressed_eigenspace(endpoint):
    manifold = RankKPSDBuresWasserstein((4, 4), rank=2)
    basis = jnp.eye(4)[:, :2]
    point = basis @ basis.T
    speed = -1.0 if endpoint else -2.0
    horizontal = jnp.concatenate([speed * jnp.eye(2), jnp.diag(jnp.array([0.0, 0.5]))])
    tangent = horizontal @ basis.T + basis @ horizontal.T
    assert np.all(np.isnan(jax.jit(manifold.exp)(point, tangent)))


def test_exp_domain_certificate_is_rotation_invariant_and_batched():
    manifold = RankKPSDBuresWasserstein((3, 3), rank=1)
    point = jnp.diag(jnp.array([1.0, 0.0, 0.0]))
    valid = jnp.array([[-4.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    invalid = -4.0 * point
    rotation, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(982), point.shape))
    p = rotation @ point @ rotation.T
    tangents = jnp.stack([rotation @ valid @ rotation.T, rotation @ invalid @ rotation.T])
    endpoints = jax.jit(jax.vmap(lambda u: manifold.exp(p, u)))(tangents)
    expected = rotation @ manifold.exp(point, valid) @ rotation.T
    _assert_close(endpoints[0], expected)
    assert np.all(np.isnan(endpoints[1]))


def test_exp_positive_horizontal_generator_with_large_dynamic_range():
    manifold = RankKPSDBuresWasserstein((3, 3), rank=2)
    point = jnp.diag(jnp.array([1.0, 1.0, 0.0]))
    tangent = jnp.diag(jnp.array([2.0, 2e10, 0.0]))
    expected = jnp.diag(jnp.array([4.0, (1.0 + 1e10) ** 2, 0.0]))
    _assert_close(manifold.exp(point, tangent), expected)


def _projection_example(kind):
    if kind == "rectangular":
        manifold = FixedRank((4, 5), rank=2)
        point = jnp.zeros((4, 5)).at[0, 0].set(1.0).at[1, 1].set(1.0)
        tangent = jnp.zeros_like(point).at[0, 0].set(0.2).at[1, 1].set(-0.1)
        tangent = tangent.at[2, 0].set(0.4).at[0, 3].set(-0.3)
    elif kind == "elliptope":
        manifold = Elliptope((4, 4), rank=2)
        factor = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        horizontal = jnp.array([[0.0, 1.0], [-1.0, 0.0], [0.0, 0.3], [0.7, 0.0]])
        point = factor @ factor.T
        tangent = horizontal @ factor.T + factor @ horizontal.T
    else:
        manifold = RankKPSD((4, 4), rank=2)
        point = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0]))
        tangent = jnp.diag(jnp.array([0.2, -0.2, 0.0, 0.0]))
        tangent = tangent.at[2, 0].set(0.1).at[0, 2].set(0.1)
        if kind == "spectrahedron":
            manifold = Spectrahedron((4, 4), rank=2)
            point = point / 2.0
    return manifold, point, tangent


@pytest.mark.parametrize("kind", ["rectangular", "psd", "elliptope", "spectrahedron"])
def test_related_rank_projections_and_retractions_differentiate_tangents_as_identity(kind):
    manifold, point, tangent = _projection_example(kind)
    _, derivative = jax.jit(lambda p, u: jax.jvp(manifold.project, (p,), (u,)))(point, tangent)
    _assert_close(derivative, tangent)
    derivative = jax.jacfwd(lambda t: manifold.retr(point, t * tangent))(0.0)
    _assert_close(derivative, tangent)


@pytest.mark.parametrize("kind", ["rectangular", "psd", "elliptope", "spectrahedron"])
def test_related_rank_projection_jvp_vjp_and_finite_difference(kind):
    manifold, point, _ = _projection_example(kind)
    ambient = jax.random.normal(jax.random.key(889), point.shape)
    cotangent = jax.random.normal(jax.random.key(890), point.shape)
    _, forward = jax.jvp(manifold.project, (point,), (ambient,))
    _, pullback = jax.vjp(manifold.project, point)
    _assert_close(jnp.sum(forward * cotangent), jnp.sum(pullback(cotangent)[0] * ambient))
    step = 1e-5 if jax.config.jax_enable_x64 else 2e-3
    finite_difference = (
        manifold.project(point + step * ambient) - manifold.project(point - step * ambient)
    ) / (2.0 * step)
    _assert_close(forward, finite_difference, multiplier=4.0)
    batch = jnp.stack([point, 1.2 * point])
    direct = jax.grad(lambda t: jnp.sum(manifold.project(batch + t * ambient)))(0.0)
    mapped = jax.grad(lambda t: jnp.sum(jax.vmap(manifold.project)(batch + t * ambient)))(0.0)
    _assert_close(direct, mapped)


@pytest.mark.parametrize("wide", [False, True])
def test_rectangular_projection_nested_derivative_with_discarded_spectrum(wide):
    point = jnp.zeros((5, 3)).at[:3, :].set(jnp.diag(jnp.array([2.0, 2.0, 0.5])))
    if wide:
        point = point.T
    manifold = FixedRank(point.shape, rank=2)
    direction = jax.random.normal(jax.random.key(927), point.shape) / 3.0
    cotangent = jax.random.normal(jax.random.key(928), point.shape)
    _, derivative = jax.jvp(manifold.project, (point,), (direction,))
    step = 1e-5 if jax.config.jax_enable_x64 else 2e-3
    difference = (
        manifold.project(point + step * direction) - manifold.project(point - step * direction)
    ) / (2.0 * step)
    _assert_close(derivative, difference, multiplier=4.0)
    gradient = jax.grad(lambda p: jnp.sum(manifold.project(p) * cotangent))
    _, hessian = jax.jit(lambda p, u: jax.jvp(gradient, (p,), (u,)))(point, direction)
    difference = (gradient(point + step * direction) - gradient(point - step * direction)) / (
        2.0 * step
    )
    _assert_close(hessian, difference, multiplier=8.0)


def test_tall_rectangular_projection_keeps_thin_matrix_working_shapes():
    manifold = FixedRank((10_000, 3), rank=2)
    point = jnp.zeros(manifold.shape).at[0, 0].set(1.0).at[1, 1].set(1.0)
    ambient = jnp.ones_like(point)
    expected = ambient.at[2:, 2].set(0.0)
    _assert_close(manifold.project(point), point)
    _assert_close(manifold.tangent_project(point, ambient), expected)
    _, derivative = jax.jvp(manifold.project, (point,), (ambient,))
    _assert_close(derivative, expected)


def test_bures_sylvester_and_metric_broadcast_base_and_tangent_batches():
    manifold = RankKPSDBuresWasserstein((3, 3), rank=1)
    point = jnp.diag(jnp.array([1.0, 0.0, 0.0]))
    scales = jnp.array([0.5, 1.0, 2.0])
    points = scales[:, None, None] * point
    _assert_close(manifold.sylvester(points, point), point / (2.0 * scales[:, None, None]))
    _assert_close(manifold.sylvester(point, points), 0.5 * points)

    def objective(scale):
        return jnp.sum(manifold.inner(scale * points, point, point))

    _assert_close(jax.jit(jax.grad(objective))(1.0), -0.25 * jnp.sum(1.0 / scales))


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("unit_direction", [-1, 1])
def test_bures_log_derivatives_are_invariant_under_extreme_unit_changes(rank, unit_direction):
    manifold, point, _ = _example(rank)
    exponent = 200 if jax.config.jax_enable_x64 else 20
    scale = 10.0 ** (unit_direction * exponent)

    def logarithm(time):
        return manifold.log(time * scale * point, 2.0 * scale * point) / scale

    def objective(time):
        return jnp.trace(logarithm(time)) / rank

    # On this fixed support, log_(tP)(2P) = (2 sqrt(2t) - 2t) P.
    _assert_close(logarithm(1.0), 2.0 * (jnp.sqrt(2.0) - 1.0) * point)
    _assert_close(jax.jit(jax.jacfwd(logarithm))(1.0), (jnp.sqrt(2.0) - 2.0) * point)
    _assert_close(jax.jit(jax.grad(objective))(1.0), jnp.sqrt(2.0) - 2.0)
    expected_hessian = -1.0 / jnp.sqrt(2.0)
    _assert_close(jax.jit(jax.grad(jax.grad(objective)))(1.0), expected_hessian)
    _assert_close(jax.jit(jax.jacfwd(jax.grad(objective)))(1.0), expected_hessian)


def test_bures_log_scaling_preserves_inputs_at_opposite_extreme_scales():
    manifold = RankKPSDBuresWasserstein((3, 3), rank=1)
    point = jnp.diag(jnp.array([1.0, 0.0, 0.0]))
    scale = 1e200 if jax.config.jax_enable_x64 else 1e20

    def objective(time):
        return manifold.log(scale * point, (time / scale) * point)[0, 0]

    # The exact scalar is 2*sqrt(t) - 2*scale. Its small derivative must
    # survive even though finite differences of that large primal cannot.
    _assert_close(jax.jit(jax.grad(objective))(1.0), 1.0)
    _assert_close(jax.jit(jax.grad(jax.grad(objective)))(1.0), -0.5)


@pytest.mark.parametrize("rank", [1, 2])
def test_bures_finite_range_endpoints_preserve_points_metric_and_distance_gradient(rank):
    manifold = RankKPSDBuresWasserstein((rank + 1, rank + 1), rank=rank)
    unit = jnp.diag(jnp.concatenate([jnp.ones(rank), jnp.zeros(1)]))
    scale = 1e308 if jax.config.jax_enable_x64 else 2e38
    point = scale * unit
    target = 0.5 * point
    # Normalize on the host: forming a device reciprocal of scale itself
    # would introduce a subnormal unrelated to these geometric operations.
    projected = jax.jit(manifold.project)(point)
    _assert_close(np.asarray(projected) / scale, unit)
    assert bool(jax.jit(manifold.belongs)(point))
    tangent = jax.jit(manifold.tangent_project)(point, point)
    _assert_close(np.asarray(tangent) / scale, unit)
    _assert_close(jax.jit(manifold.sylvester)(point, point), 0.5 * unit)
    distance = jax.jit(manifold.squared_dist)(point, target)
    _assert_close(np.asarray(distance) / scale, rank * (1.0 - np.sqrt(0.5)) ** 2)
    gradient = jax.jit(jax.grad(lambda p: manifold.squared_dist(p, target)))(point)
    _assert_close(gradient, (1.0 - np.sqrt(0.5)) * unit)
    log = jax.jit(manifold.log)(point, target)
    _assert_close(np.asarray(log) / scale, 2.0 * (np.sqrt(0.5) - 1.0) * unit)


def test_low_rank_projection_extreme_spectrum_keeps_rotating_support_derivative():
    manifold = RankKPSD((3, 3), rank=1)
    scale = 1e308 if jax.config.jax_enable_x64 else 2e38
    point = jnp.diag(jnp.array([scale, 0.0, 0.0]))
    direction = jnp.array([[0.0, 0.2 * scale, 0.0], [0.2 * scale, 0.0, 0.0], [0.0, 0.0, 0.0]])
    _, derivative = jax.jit(lambda p, u: jax.jvp(manifold.project, (p,), (u,)))(point, direction)
    _assert_close(np.asarray(derivative) / scale, np.asarray(direction) / scale)


def test_low_rank_projection_preserves_the_smallest_normal_positive_spectrum():
    manifold = RankKPSD((2, 2), rank=1)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    scale = np.finfo(dtype).tiny
    point = jnp.diag(jnp.array([scale, 0.0], dtype=dtype))
    projected = jax.jit(manifold.project)(point)
    _assert_close(np.asarray(projected) / scale, jnp.diag(jnp.array([1.0, 0.0])))
    assert bool(jax.jit(manifold.belongs)(point))
