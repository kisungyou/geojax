from __future__ import annotations

from collections.abc import Callable
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
from geojax.optimization import Minimize


def _geometry_cases() -> list[Any]:
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


def _geometry_id(geometry: Any) -> str:
    return type(geometry).__name__


def _assert_tree_finite(tree: Any) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    assert leaves
    for leaf in leaves:
        assert bool(jnp.all(jnp.isfinite(leaf)))


@pytest.mark.parametrize("M", _geometry_cases(), ids=_geometry_id)
def test_complete_geometry_protocol_is_jittable(M):
    """Compile all numerical protocol methods with keys and arrays as tracers."""

    def transformed(key):
        key_x, key_u = jax.random.split(key)
        x = M.random_point(key_x)
        u = M.random_tangent(key_u, x, scale=1e-3)
        projected = M.project(x)
        tangent = M.tangent_project(x, u)
        endpoint = M.exp(x, tangent)
        recovered = M.log(x, endpoint)
        moved = M.transport(x, endpoint, tangent)
        return {
            "projected": projected,
            "tangent": tangent,
            "endpoint": endpoint,
            "recovered": recovered,
            "moved": moved,
            "inner": M.inner(x, tangent, tangent),
            "norm": M.norm(x, tangent),
            "distance": M.dist(x, endpoint),
            "belongs": M.belongs(endpoint),
            "is_tangent": M.is_tangent(endpoint, moved),
        }

    result = jax.jit(transformed)(jax.random.key(100))

    _assert_tree_finite(result)
    assert bool(jnp.all(result["belongs"]))
    assert bool(jnp.all(result["is_tangent"]))
    assert bool(jnp.all(result["norm"] >= 0.0))
    assert bool(jnp.all(result["distance"] >= 0.0))


@pytest.mark.parametrize(
    "M",
    [
        Sphere(size=3),
        Grassmann(size=(4, 2)),
        SPDAffineInvariant(size=(2, 2)),
        FixedRank(size=(3, 2), rank=1),
        SpecialOrthogonal(size=3),
        Stiefel(size=(3, 2), log_maxiter=8),
        KendallShape(size=(4, 2)),
        Product({"direction": Sphere(size=2), "phase": Torus(size=2)}),
    ],
    ids=_geometry_id,
)
def test_batch_helpers_compose_with_jit_and_vmap(M):
    key_x, key_y, key_u = jax.random.split(jax.random.key(200), 3)
    x = M.random_point(key_x)
    ys = M.random_point(key_y, sample_shape=(3,))
    xs = jax.tree_util.tree_map(lambda leaf: jnp.broadcast_to(leaf, (3,) + leaf.shape), x)
    us = M.random_tangent(key_u, xs, scale=1e-3)

    @jax.jit
    def transformed(point, endpoints, tangents):
        return (
            M.dist_batch(point, endpoints),
            M.log_batch(point, endpoints),
            M.exp_batch(point, tangents),
        )

    distances, logarithms, exponentials = transformed(x, ys, us)

    _assert_tree_finite((distances, logarithms, exponentials))
    assert distances.shape == (3,)
    assert bool(jnp.all(M.belongs(exponentials)))


def _sphere_objective(target: Any) -> Callable[[Any], Any]:
    return lambda x: 0.5 * jnp.sum((x - target) ** 2)


def _spd_objective(target: Any) -> Callable[[Any], Any]:
    return lambda x: 0.5 * jnp.sum((x - target) ** 2)


def _product_objective(target: Any) -> Callable[[Any], Any]:
    return lambda x: (
        0.5
        * (
            jnp.sum((x["direction"] - target["direction"]) ** 2)
            + jnp.sum((x["phase"] - target["phase"]) ** 2)
        )
    )


@pytest.mark.parametrize(
    ("M", "objective_factory"),
    [
        (Sphere(size=3), _sphere_objective),
        (
            Product({"direction": Sphere(size=2), "phase": Torus(size=2)}),
            _product_objective,
        ),
    ],
    ids=["Sphere", "Product"],
)
def test_autodiff_gradient_and_hessian_vector_products_are_jittable(M, objective_factory):
    key_x, key_target, key_u = jax.random.split(jax.random.key(300), 3)
    x = M.random_point(key_x)
    target = M.random_point(key_target)
    u = M.random_tangent(key_u, x, scale=1e-3)
    cost = objective_factory(target)
    problem = Minimize(M=M, cost=cost)

    @jax.jit
    def transformed(point, tangent):
        value, egrad = jax.value_and_grad(cost)(point)
        rgrad = M.egrad_to_rgrad(point, egrad)
        ehess = problem.ehess_vec(point, tangent)
        rhess = problem.rhess_vec(point, tangent)
        return value, egrad, rgrad, ehess, rhess

    value, egrad, rgrad, ehess, rhess = transformed(x, u)

    _assert_tree_finite((value, egrad, rgrad, ehess, rhess))
    assert bool(jnp.all(M.is_tangent(x, rgrad)))
    assert bool(jnp.all(M.is_tangent(x, rhess)))


def test_unsupported_automatic_spd_hessian_is_reported_explicitly():
    M = SPDAffineInvariant(size=(2, 2))
    x = M.random_point(jax.random.key(310))
    u = M.random_tangent(jax.random.key(311), x, scale=1e-3)
    problem = Minimize(M=M, cost=_spd_objective(x))

    assert M.operation_kind("ehess_to_rhess") == "projection"
    with pytest.raises(ValueError, match="Supply rhess_vec explicitly"):
        problem.rhess_vec(x, u)
