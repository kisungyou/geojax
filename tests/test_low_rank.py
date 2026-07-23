from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import (
    Elliptope,
    FixedRank,
    RankKPSD,
    RankKPSDBuresWasserstein,
    Spectrahedron,
)


@pytest.mark.parametrize(
    "M",
    [
        FixedRank(size=(5, 4), rank=2),
        RankKPSD(size=(5, 5), rank=2),
        Elliptope(size=(5, 5), rank=2),
        Spectrahedron(size=(5, 5), rank=2),
    ],
)
def test_svd_based_geometries_advertise_retraction_proxies(M):
    assert M.operation_kind("exp") == "proxy"
    assert M.operation_kind("log") == "proxy"
    assert M.operation_kind("dist") == "proxy"


def test_fixed_rank_projection_and_tangent_constraint():
    M = FixedRank(size=(5, 4), rank=2)
    X = M.project(jax.random.normal(jax.random.key(0), M.shape))
    U = M.tangent_project(X, jax.random.normal(jax.random.key(1), M.shape))
    singular_values = jnp.linalg.svd(X, compute_uv=False)

    assert M.dim == 14
    assert bool(M.belongs(X))
    assert bool(M.is_tangent(X, U))
    assert singular_values[1] > M.atol
    assert singular_values[2] <= M.atol


def test_fixed_rank_psd_constraints():
    M = RankKPSD(size=(4, 4), rank=2)
    P = M.random_point(jax.random.key(0), sample_shape=(3,))
    eigenvalues = jnp.linalg.eigvalsh(P)

    assert jnp.all(M.belongs(P))
    assert jnp.all(eigenvalues[..., :2] <= M.atol)
    assert jnp.all(eigenvalues[..., 2:] > M.atol)


def test_rank_one_bures_distance_matches_factor_procrustes_distance():
    M = RankKPSDBuresWasserstein(size=(3, 3), rank=1)
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([0.6, 0.8, 0.0])
    P = jnp.outer(a, a)
    Q = jnp.outer(b, b)

    expected = jnp.minimum(jnp.linalg.norm(a - b), jnp.linalg.norm(a + b))
    assert M.operation_kind("dist") == "exact"
    assert jnp.allclose(M.dist(P, Q), expected, atol=1e-12)


def test_elliptope_and_spectrahedron_constraints_and_tangents(dtype_atol):
    elliptope = Elliptope(size=(5, 5), rank=2)
    correlation = elliptope.random_point(jax.random.key(0))
    corr_tangent = elliptope.random_tangent(jax.random.key(1), correlation)

    spectrahedron = Spectrahedron(size=(5, 5), rank=2)
    density = spectrahedron.random_point(jax.random.key(2))
    density_tangent = spectrahedron.random_tangent(jax.random.key(3), density)

    assert bool(elliptope.belongs(correlation))
    assert jnp.allclose(
        jnp.diag(correlation), 1.0, atol=max(1e-10, dtype_atol)
    )
    assert jnp.allclose(
        jnp.diag(corr_tangent), 0.0, atol=max(1e-8, dtype_atol)
    )
    assert bool(elliptope.is_tangent(correlation, corr_tangent))

    assert bool(spectrahedron.belongs(density))
    assert jnp.allclose(jnp.trace(density), 1.0, atol=max(1e-10, dtype_atol))
    assert jnp.allclose(
        jnp.trace(density_tangent), 0.0, atol=max(1e-10, dtype_atol)
    )
    assert bool(spectrahedron.is_tangent(density, density_tangent))
