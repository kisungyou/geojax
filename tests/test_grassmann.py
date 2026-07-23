from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.geometry import Grassmann, GrassmannProjection
from geojax.optimization import ConjugateGradient, Minimize


def test_grassmann_constructor_projector():
    M = Grassmann(size=(5, 2))
    X = M.random_point(jax.random.key(0))
    P = M.projector(X)
    assert M.shape == (5, 2)
    assert M.dim == 6
    assert jnp.allclose(P, P.T, atol=1e-8)
    assert jnp.allclose(P @ P, P, atol=1e-8)


def test_grassmann_rejects_invalid_size():
    try:
        Grassmann(size=(2, 3))
    except ValueError:
        return
    raise AssertionError("Grassmann should reject rank > ambient dimension")


def test_projection_embedding_is_basis_invariant():
    canonical = Grassmann(size=(5, 2))
    embedded = GrassmannProjection(size=(5, 2))
    X = canonical.random_point(jax.random.key(1))
    rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]])

    P = embedded.embed(X)
    P_rotated = embedded.embed(X @ rotation)

    assert embedded.shape == (5, 2)
    assert bool(embedded.belongs(X))
    assert jnp.allclose(P, P_rotated, atol=1e-6)


def test_projection_embedding_returns_frame_from_symmetric_matrix():
    M = GrassmannProjection(size=(4, 2))
    A = jax.random.normal(jax.random.key(2), shape=(4, 4))
    X = M.project_embedding(A)
    P = M.embed(X)

    assert X.shape == (4, 2)
    assert bool(M.belongs(X))
    assert jnp.allclose(P @ P, P, atol=1e-6)
    assert jnp.allclose(jnp.trace(P), 2.0, atol=1e-6)


def test_projection_geometry_matches_canonical_intrinsic_geometry():
    canonical = Grassmann(size=(5, 2))
    embedded = GrassmannProjection(size=(5, 2))
    X = canonical.random_point(jax.random.key(3))
    U = canonical.random_tangent(jax.random.key(4), X, scale=0.2)
    Y = canonical.exp(X, U)
    H = embedded.embed_tangent(X, U)

    assert H.shape == (5, 5)
    assert jnp.allclose(embedded.inner(X, U, U), canonical.inner(X, U, U), atol=1e-6)
    assert jnp.allclose(embedded.dist(X, Y), canonical.dist(X, Y), atol=1e-6)
    assert jnp.allclose(embedded.embed(embedded.exp(X, U)), embedded.embed(Y), atol=1e-5)


def test_projection_gradient_conversion_matches_canonical_frame_gradient():
    canonical = Grassmann(size=(5, 2))
    embedded = GrassmannProjection(size=(5, 2))
    X = canonical.random_point(jax.random.key(11))
    egrad = jax.random.normal(jax.random.key(12), shape=X.shape)

    expected = canonical.egrad_to_rgrad(X, egrad)
    actual = embedded.egrad_to_rgrad(X, egrad)

    assert bool(embedded.is_tangent(X, actual))
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_chordal_distance_matches_sines_of_principal_angles():
    canonical = Grassmann(size=(6, 2))
    embedded = GrassmannProjection(size=(6, 2))
    X = canonical.random_point(jax.random.key(5))
    Y = canonical.random_point(jax.random.key(6))
    singular_values = jnp.linalg.svd(X.T @ Y, compute_uv=False)
    expected = jnp.sqrt(jnp.sum(1.0 - jnp.clip(singular_values, 0.0, 1.0) ** 2))

    chordal = embedded.chordal_dist(X, Y)

    assert jnp.allclose(chordal, expected, atol=1e-6)
    assert jnp.allclose(chordal, embedded.projection_dist(X, Y), atol=1e-7)
    assert chordal <= embedded.dist(X, Y) + 1e-6


def test_nearby_geodesic_distance_is_numerically_stable(dtype_atol):
    canonical = Grassmann(size=(5, 2))
    embedded = GrassmannProjection(size=(5, 2))
    X = canonical.random_point(jax.random.key(7))
    scale = 1e-7 if jax.config.x64_enabled else 1e-3
    U = canonical.random_tangent(jax.random.key(8), X, scale=scale)
    Y = canonical.exp(X, U)

    expected = canonical.norm(X, U)
    geodesic = embedded.dist(X, Y)
    chordal = embedded.chordal_dist(X, Y)

    assert jnp.allclose(
        geodesic,
        expected,
        atol=max(1e-12, dtype_atol),
        rtol=max(1e-7, dtype_atol),
    )
    assert chordal <= geodesic + max(1e-12, dtype_atol)


def test_projector_decomposition_aligns_to_reference_frame():
    M = GrassmannProjection(size=(6, 2))
    X = M.random_point(jax.random.key(9))
    recovered = M.to_frame(M.embed(X), reference=X)

    assert bool(M.belongs(recovered))
    assert jnp.allclose(recovered, X, atol=1e-6)


def test_projection_extrinsic_mean_is_frame_valued_and_basis_invariant():
    M = GrassmannProjection(size=(5, 2))
    points = M.random_point(jax.random.key(10), sample_shape=(5,))
    rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]])

    mean = M.extrinsic_mean(points)
    rotated_mean = M.extrinsic_mean(points @ rotation)

    assert mean.shape == (5, 2)
    assert bool(M.belongs(mean))
    assert jnp.allclose(M.embed(mean), M.embed(rotated_mean), atol=1e-6)


def test_projection_geometry_optimizes_with_frame_valued_points():
    M = GrassmannProjection(size=(4, 1))
    matrix = jnp.diag(jnp.array([4.0, 2.0, 1.0, 0.5]))
    problem = Minimize(
        M=M,
        cost=lambda X: -jnp.trace(X.T @ matrix @ X),
        solver=ConjugateGradient(maxiter=80, tolgradnorm=1e-8, verbosity=0),
        key=4,
    )

    estimate, final_cost, _ = problem.solve()

    assert estimate.shape == (4, 1)
    assert bool(M.belongs(estimate))
    assert final_cost < -3.999999
