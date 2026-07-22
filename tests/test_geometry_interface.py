from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import (
    CorrelationAffineQuotient,
    CorrelationECM,
    CorrelationLEC,
    Elliptope,
    FixedRank,
    GeneralizedGrassmann,
    GeneralizedStiefel,
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
    tangent_scale = 0.05 if M.exp_is_exact and M.log_is_exact else 1e-3
    u = M.random_tangent(key_u, x, scale=tangent_scale)
    zero = tree_zeros_like(u)

    assert bool(jnp.all(M.belongs(M.project(x))))
    assert bool(jnp.all(M.is_tangent(x, M.tangent_project(x, u))))
    assert_tree_allclose(M.exp(x, zero), x, atol=1e-5)

    y_local = M.exp(x, u)
    assert bool(jnp.all(M.belongs(y_local)))
    recovered = M.log(x, y_local)
    assert bool(jnp.all(M.is_tangent(x, recovered)))
    if M.log_is_exact:
        assert_tree_allclose(recovered, u, atol=5e-4)
    else:
        error = M.norm(x, M.lincomb(x, 1.0, recovered, -1.0, u))
        assert bool(jnp.all(jnp.isfinite(error)))
        assert bool(jnp.all(error <= 0.1 * jnp.maximum(M.norm(x, u), 1e-10)))

    distance = M.dist(x, y)
    assert bool(jnp.all(jnp.isfinite(distance)))
    assert bool(jnp.all(distance >= 0.0))
    if M.dist_is_exact:
        assert jnp.allclose(distance, M.dist(y, x), atol=1e-5, rtol=1e-5)
    else:
        assert M.operation_kind("dist") == "proxy"

    v = M.random_tangent(jax.random.key(3), x, scale=0.05)
    transported = M.transport(x, y_local, v)
    assert bool(jnp.all(M.is_tangent(y_local, transported)))
    if M.transport_is_isometric:
        assert jnp.allclose(M.norm(x, v), M.norm(y_local, transported), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("M", geometries())
def test_optimizer_protocol_and_operation_metadata(M):
    assert isinstance(M, ManifoldProtocol)
    for name in ("exp", "log", "dist"):
        expected = "exact" if getattr(M, f"{name}_is_exact") else "proxy"
        assert M.operation_kind(name) == expected

    transport_kind = M.operation_kind("transport")
    assert transport_kind in {"parallel", "isometric", "vector"}


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
