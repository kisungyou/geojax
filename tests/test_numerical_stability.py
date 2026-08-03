from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import (
    CorrelationAffineQuotient,
    CorrelationECM,
    CorrelationLEC,
    Elliptope,
    Euclidean,
    FixedRank,
    GeneralizedGrassmann,
    GeneralizedStiefel,
    Grassmann,
    GrassmannProjection,
    Hyperboloid,
    KendallShape,
    Oblique,
    PoincareBall,
    ProbabilitySimplex,
    Product,
    RankKPSD,
    RankKPSDBuresWasserstein,
    SPDAffineInvariant,
    SPDBuresWasserstein,
    SPDLogEuclidean,
    SpecialEuclidean,
    SpecialOrthogonal,
    Sphere,
    SphereExtrinsic,
    Spectrahedron,
    Stiefel,
    StiefelEuclidean,
    Torus,
)


def _geometries() -> list[Any]:
    metric = jnp.diag(jnp.array([1.0, 1.5, 2.0]))
    return [
        Euclidean(size=3),
        Oblique(size=(3, 2)),
        ProbabilitySimplex(size=3),
        PoincareBall(size=2),
        Sphere(size=3),
        SphereExtrinsic(size=3),
        Grassmann(size=(4, 2)),
        GrassmannProjection(size=(4, 2)),
        GeneralizedStiefel(size=(3, 2), metric=metric, log_maxiter=8),
        GeneralizedGrassmann(size=(3, 2), metric=metric),
        SPDLogEuclidean(size=(2, 2)),
        SPDAffineInvariant(size=(2, 2)),
        SPDBuresWasserstein(size=(2, 2)),
        FixedRank(size=(3, 2), rank=1),
        RankKPSD(size=(3, 3), rank=2),
        RankKPSDBuresWasserstein(size=(3, 3), rank=2),
        Elliptope(size=(3, 3), rank=2),
        Spectrahedron(size=(3, 3), rank=2),
        CorrelationECM(size=(3, 3)),
        CorrelationLEC(size=(3, 3)),
        CorrelationAffineQuotient(size=(3, 3)),
        Product({"direction": Sphere(size=2), "phase": Torus(size=2)}),
        Hyperboloid(size=3),
        Torus(size=2),
        SpecialOrthogonal(size=3),
        SpecialEuclidean(size=2),
        Stiefel(size=(3, 2), log_maxiter=8),
        StiefelEuclidean(size=(3, 2), log_maxiter=8),
        KendallShape(size=(4, 2)),
    ]


def _geometry_id(manifold: Any) -> str:
    return type(manifold).__name__


def _assert_finite(tree: Any) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    assert leaves
    for leaf in leaves:
        assert bool(jnp.all(jnp.isfinite(leaf)))


def _assert_tree_allclose(left: Any, right: Any, *, atol: float, rtol: float) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=rtol)


def _tree_sum(tree: Any) -> Any:
    return sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(tree))


@pytest.mark.parametrize("M", _geometries(), ids=_geometry_id)
def test_zero_and_coincident_operations_have_finite_first_derivatives(M, dtype_atol):
    key_x, key_u = jax.random.split(jax.random.key(610))
    x = M.random_point(key_x)
    u = M.random_tangent(key_u, x, scale=1e-4)
    zero = jax.tree_util.tree_map(jnp.zeros_like, u)

    exp_value, exp_jvp = jax.jvp(lambda tangent: M.exp(x, tangent), (zero,), (u,))
    log_value, log_jvp = jax.jvp(lambda endpoint: M.log(x, endpoint), (x,), (u,))
    squared_distance = M.squared_dist(x, x)
    squared_distance_gradient = jax.grad(lambda endpoint: M.squared_dist(x, endpoint))(x)
    transported = M.transport(x, x, u)

    _assert_finite(
        (
            exp_value,
            exp_jvp,
            log_value,
            log_jvp,
            squared_distance,
            squared_distance_gradient,
            transported,
        )
    )
    tolerance = max(2e-5, 10.0 * dtype_atol)
    _assert_tree_allclose(exp_value, x, atol=tolerance, rtol=tolerance)
    _assert_tree_allclose(log_value, zero, atol=tolerance, rtol=tolerance)
    _assert_tree_allclose(transported, u, atol=max(2e-4, 20.0 * dtype_atol), rtol=2e-4)
    assert jnp.allclose(squared_distance, 0.0, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "M",
    [
        Oblique(size=(3, 2)),
        ProbabilitySimplex(size=3),
        Sphere(size=3),
        Hyperboloid(size=3),
    ],
    ids=_geometry_id,
)
def test_coincident_logarithms_have_finite_reverse_mode_derivatives(M):
    x = M.random_point(jax.random.key(615))
    endpoint_gradient = jax.grad(lambda endpoint: _tree_sum(M.log(x, endpoint)))(x)
    joint_gradient = jax.grad(lambda point: _tree_sum(M.log(point, point)))(x)

    _assert_finite((endpoint_gradient, joint_gradient))


@pytest.mark.parametrize("M", _geometries(), ids=_geometry_id)
def test_nested_batches_and_broadcast_bases_preserve_event_axes(M):
    key_x, key_u = jax.random.split(jax.random.key(620))
    x = M.random_point(key_x)
    points = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (2, 2) + leaf.shape),
        x,
    )
    tangents = M.random_tangent(key_u, points, scale=1e-4)

    endpoints = M.exp(x, tangents)
    logarithms = M.log(x, endpoints)
    scalars = (
        M.belongs(endpoints),
        M.is_tangent(x, tangents),
        M.inner(x, tangents, tangents),
        M.norm(x, tangents),
        M.dist(x, endpoints),
        M.squared_dist(x, endpoints),
    )

    _assert_finite((endpoints, logarithms, scalars))
    for value in scalars:
        assert jnp.shape(value) == (2, 2)
    for leaf in jax.tree_util.tree_leaves((endpoints, logarithms)):
        assert leaf.shape[:2] == (2, 2)


@pytest.mark.parametrize("M", _geometries(), ids=_geometry_id)
def test_normalized_random_tangents_have_requested_riemannian_scale(M, dtype_atol):
    points = M.random_point(jax.random.key(625), sample_shape=(2, 2))
    tangents = M.random_tangent(
        jax.random.key(626),
        points,
        scale=0.3,
        normalize=True,
    )
    lengths = M.norm(points, tangents)

    _assert_finite(tangents)
    assert lengths.shape == (2, 2)
    assert jnp.allclose(
        lengths,
        0.3,
        atol=max(2e-5, 10.0 * dtype_atol),
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    "M",
    [
        Sphere(size=3),
        Hyperboloid(size=3),
        PoincareBall(size=2),
        Grassmann(size=(4, 2)),
        SPDAffineInvariant(size=(2, 2)),
        SpecialOrthogonal(size=3),
    ],
    ids=_geometry_id,
)
def test_exponential_jvp_matches_directional_finite_difference(M):
    key_x, key_u, key_v = jax.random.split(jax.random.key(630), 3)
    x = M.random_point(key_x)
    u = M.random_tangent(key_u, x, scale=0.03)
    v = M.random_tangent(key_v, x, scale=0.02)
    _, derivative = jax.jvp(lambda tangent: M.exp(x, tangent), (u,), (v,))

    leaves = jax.tree_util.tree_leaves(x)
    dtype = leaves[0].dtype
    step = 2e-4 if dtype == jnp.float32 else 2e-6
    forward = M.exp(x, M.lincomb(x, 1.0, u, step, v))
    backward = M.exp(x, M.lincomb(x, 1.0, u, -step, v))
    finite_difference = jax.tree_util.tree_map(
        lambda upper, lower: (upper - lower) / (2.0 * step),
        forward,
        backward,
    )

    _assert_finite((derivative, finite_difference))
    _assert_tree_allclose(
        derivative,
        finite_difference,
        atol=2e-3 if dtype == jnp.float32 else 2e-5,
        rtol=2e-3 if dtype == jnp.float32 else 2e-5,
    )


@pytest.mark.parametrize(
    "M",
    [
        SPDLogEuclidean(size=(3, 3)),
        SPDAffineInvariant(size=(3, 3)),
        SPDBuresWasserstein(size=(3, 3)),
    ],
    ids=_geometry_id,
)
def test_spd_repeated_spectrum_has_finite_values_and_gradients(M):
    identity = jnp.eye(3)
    direction = jnp.array(
        [
            [0.2, 0.1, 0.0],
            [0.1, -0.1, 0.05],
            [0.0, 0.05, 0.3],
        ]
    )
    zero = jnp.zeros_like(identity)

    projected, projection_jvp = jax.jvp(M.project, (identity,), (direction,))
    endpoint, exp_jvp = jax.jvp(lambda tangent: M.exp(identity, tangent), (zero,), (direction,))
    logarithm, log_jvp = jax.jvp(lambda point: M.log(identity, point), (identity,), (direction,))
    gradient = jax.grad(lambda point: M.squared_dist(identity, point))(identity)

    _assert_finite((projected, projection_jvp, endpoint, exp_jvp, logarithm, log_jvp, gradient))
    assert bool(M.belongs(projected))
    assert bool(M.belongs(endpoint))


def test_genuine_sphere_cut_locus_remains_explicit():
    M = Sphere(size=3)
    x = jnp.array([1.0, 0.0, 0.0])
    antipode = -x
    tangent = jnp.array([0.0, 1.0, 0.0])

    assert jnp.allclose(M.dist(x, antipode), jnp.pi)
    assert bool(jnp.any(~jnp.isfinite(M.log(x, antipode))))
    assert bool(jnp.any(~jnp.isfinite(M.transport(x, antipode, tangent))))


def test_sphere_log_remains_accurate_near_but_off_the_cut_locus():
    M = Sphere(size=3)
    x = jnp.array([1.0, 0.0, 0.0])
    angle = jnp.asarray(jnp.pi - 0.001, dtype=x.dtype)
    y = jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
    direction = jnp.array([0.0, 0.0, 1.0])

    tangent = M.log(x, y)
    transported = M.transport(x, y, direction)

    _assert_finite((tangent, transported))
    assert jnp.allclose(M.norm(x, tangent), angle, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(M.exp(x, tangent), y, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(M.norm(x, direction), M.norm(y, transported), atol=2e-5)

    oblique = Oblique(size=(3, 1))
    frame_x = x[:, None]
    frame_y = y[:, None]
    frame_direction = direction[:, None]
    frame_tangent = oblique.log(frame_x, frame_y)
    frame_transported = oblique.transport(frame_x, frame_y, frame_direction)
    _assert_finite((frame_tangent, frame_transported))
    assert jnp.allclose(
        oblique.norm(frame_x, frame_tangent),
        angle,
        atol=2e-5,
        rtol=2e-5,
    )
    assert jnp.allclose(
        oblique.norm(frame_x, frame_direction),
        oblique.norm(frame_y, frame_transported),
        atol=2e-5,
    )


def test_grassmann_cut_locus_has_finite_distance_but_no_selected_logarithm():
    M = Grassmann(size=(4, 2))
    x = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    y = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    assert jnp.allclose(M.dist(x, y), 0.5 * jnp.pi, atol=1e-6, rtol=1e-6)
    assert bool(jnp.any(~jnp.isfinite(M.log(x, y))))


def test_hyperboloid_poincare_roundtrip_and_distance_agree():
    M = Hyperboloid(size=3)
    points = M.random_point(jax.random.key(640), sample_shape=(8,))
    disk = M.to_poincare(points)
    recovered = M.from_poincare(disk)

    _assert_finite((disk, recovered))
    assert bool(jnp.all(jnp.linalg.norm(disk, axis=-1) < 1.0))
    assert bool(jnp.all(M.belongs(recovered)))
    assert jnp.allclose(recovered, points, atol=2e-6, rtol=2e-6)
