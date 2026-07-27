from __future__ import annotations

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
    GeometryMixin,
    GeometryProtocol,
    Grassmann,
    GrassmannProjection,
    Hyperboloid,
    KendallShape,
    ManifoldProtocol,
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
from geojax.optimization.minimize import tree_zeros_like


def assert_tree_allclose(a, b, *, atol=1e-5):
    leaves_a, treedef_a = jax.tree_util.tree_flatten(a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(b)
    assert treedef_a == treedef_b
    for xa, xb in zip(leaves_a, leaves_b):
        assert jnp.allclose(xa, xb, atol=atol, rtol=atol)


def geometries():
    metric = jnp.diag(jnp.array([1.0, 1.5, 2.0, 3.0]))
    return [
        Oblique(size=(4, 2)),
        ProbabilitySimplex(size=4),
        PoincareBall(size=3),
        Sphere(size=4),
        SphereExtrinsic(size=4),
        Grassmann(size=(5, 2)),
        GrassmannProjection(size=(5, 2)),
        GeneralizedStiefel(size=(4, 2), metric=metric),
        GeneralizedGrassmann(size=(4, 2), metric=metric),
        SPDLogEuclidean(size=(3, 3)),
        SPDAffineInvariant(size=(3, 3)),
        SPDBuresWasserstein(size=(3, 3)),
        FixedRank(size=(4, 3), rank=2),
        RankKPSD(size=(4, 4), rank=2),
        RankKPSDBuresWasserstein(size=(4, 4), rank=2),
        Elliptope(size=(4, 4), rank=2),
        Spectrahedron(size=(4, 4), rank=2),
        CorrelationECM(size=(3, 3)),
        CorrelationLEC(size=(3, 3)),
        CorrelationAffineQuotient(size=(3, 3)),
        Product({"sphere": Sphere(size=3), "torus": Torus(size=2)}),
        Hyperboloid(size=3),
        Torus(size=3),
        SpecialOrthogonal(size=3),
        SpecialEuclidean(size=2),
        Stiefel(size=(4, 2)),
        StiefelEuclidean(size=(4, 2)),
        KendallShape(size=(4, 2)),
    ]


@pytest.mark.parametrize("M", geometries())
def test_shared_geometry_invariants(M):
    key = jax.random.key(0)
    key_x, key_y, key_u = jax.random.split(key, 3)
    x = M.random_point(key_x)
    y = M.random_point(key_y)
    tangent_scale = (
        0.05 if M.operation_kind("exp") == "exact" and M.operation_kind("log") == "exact" else 1e-3
    )
    u = M.random_tangent(key_u, x, scale=tangent_scale)
    zero = tree_zeros_like(u)

    assert bool(jnp.all(M.belongs(M.project(x))))
    assert bool(jnp.all(M.is_tangent(x, M.tangent_project(x, u))))
    assert_tree_allclose(M.exp(x, zero), x, atol=1e-5)

    y_local = M.exp(x, u)
    assert bool(jnp.all(M.belongs(y_local)))
    recovered = M.log(x, y_local)
    assert bool(jnp.all(M.is_tangent(x, recovered)))
    if M.operation_kind("log") == "exact":
        assert_tree_allclose(recovered, u, atol=5e-4)
    else:
        error = M.norm(x, M.lincomb(x, 1.0, recovered, -1.0, u))
        assert bool(jnp.all(jnp.isfinite(error)))
        assert bool(jnp.all(error <= 0.1 * jnp.maximum(M.norm(x, u), 1e-10)))

    distance = M.dist(x, y)
    assert bool(jnp.all(jnp.isfinite(distance)))
    assert bool(jnp.all(distance >= 0.0))
    if M.operation_kind("dist") == "exact":
        assert jnp.allclose(distance, M.dist(y, x), atol=1e-5, rtol=1e-5)
    else:
        assert M.operation_kind("dist") in {"proxy", "numerical-local"}

    v = M.random_tangent(jax.random.key(3), x, scale=0.05)
    transported = M.transport(x, y_local, v)
    assert bool(jnp.all(M.is_tangent(y_local, transported)))
    if M.transport_is_isometric:
        assert jnp.allclose(M.norm(x, v), M.norm(y_local, transported), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("M", geometries())
def test_optimizer_protocol_and_operation_metadata(M):
    assert isinstance(M, ManifoldProtocol)
    assert isinstance(M, GeometryProtocol)
    for name in ("exp", "log", "dist"):
        kind = M.operation_kind(name)
        if getattr(M, f"{name}_is_exact"):
            assert kind == "exact"
        else:
            assert kind in {"proxy", "numerical-local"}

    transport_kind = M.operation_kind("transport")
    assert transport_kind in {"parallel", "isometric", "vector"}
    assert M.operation_kind("ehess_to_rhess") in {"exact", "projection"}
    assert M.operation_kind("rgrad_jvp") in {"exact", "projection"}


@pytest.mark.parametrize("M", geometries())
def test_batch_helpers_and_random_sample_shape(M):
    key = jax.random.key(10)
    x = M.random_point(key)
    ys = M.random_point(jax.random.key(11), sample_shape=(3,))
    xs = jax.tree_util.tree_map(lambda z: jnp.broadcast_to(z, (3,) + jnp.shape(z)), x)
    us = M.random_tangent(jax.random.key(12), xs, scale=0.01)

    dists = M.dist_batch(x, ys)
    logs = M.log_batch(x, ys)
    exps = M.exp_batch(x, us)

    assert jnp.shape(dists) == (3,)
    assert len(jax.tree_util.tree_leaves(logs)) > 0
    assert bool(jnp.all(M.belongs(exps)))


@pytest.mark.parametrize("M", geometries())
def test_project_repairs_zero_and_noisy_ambient_inputs(M):
    point = M.random_point(jax.random.key(20))
    leaves, treedef = jax.tree_util.tree_flatten(point)
    keys = jax.random.split(jax.random.key(21), len(leaves))
    zero = jax.tree_util.tree_unflatten(treedef, [jnp.zeros_like(leaf) for leaf in leaves])
    noisy = jax.tree_util.tree_unflatten(
        treedef,
        [
            10.0 * jax.random.normal(key, shape=leaf.shape, dtype=leaf.dtype)
            for key, leaf in zip(keys, leaves)
        ],
    )

    for ambient in (zero, noisy):
        projected = M.project(ambient)
        assert bool(jnp.all(M.belongs(projected))), type(M).__name__
        assert all(
            bool(jnp.all(jnp.isfinite(leaf)))
            for leaf in jax.tree_util.tree_leaves(projected)
        )


@pytest.mark.parametrize("M", [M for M in geometries() if not isinstance(M, Product)])
def test_array_geometries_enforce_declared_event_shape(M):
    wrong_shape = M.shape[:-1] + (M.shape[-1] + 1,)
    point = jnp.zeros(wrong_shape)
    tangent = jnp.zeros(wrong_shape)

    assert not bool(jnp.any(M.belongs(point)))
    assert not bool(jnp.any(M.is_tangent(point, tangent)))
    with pytest.raises(ValueError, match="trailing event shape"):
        M.project(point)
    with pytest.raises(ValueError, match="trailing event shape"):
        M.tangent_project(point, tangent)


def test_product_rejects_non_geometry_factors_and_uncertified_metadata():
    with pytest.raises(TypeError, match="GeometryProtocol"):
        Product({"valid": Sphere(3), "invalid": object()})

    class UncertifiedEuclidean(Euclidean):
        exp_is_exact = False
        log_is_exact = False
        dist_is_exact = False
        transport_is_isometric = False
        transport_is_parallel = False

    factor = UncertifiedEuclidean(2)
    assert isinstance(factor, GeometryMixin)
    product = Product((Sphere(3), factor))
    assert not product.exp_is_exact
    assert not product.log_is_exact
    assert not product.dist_is_exact
    assert not product.transport_is_isometric
    assert not product.transport_is_parallel
