from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

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


def test_projection_extrinsic_mean_marks_a_nonunique_eigenspace():
    manifold = GrassmannProjection(size=(2, 1))
    points = jnp.array([[[1.0], [0.0]], [[0.0], [1.0]]])

    assert bool(jnp.all(jnp.isnan(manifold.extrinsic_mean(points))))


def test_projection_extrinsic_mean_validates_points_and_weights():
    M = GrassmannProjection(size=(5, 2))
    points = M.random_point(jax.random.key(18), sample_shape=(3,))

    with pytest.raises(ValueError, match="points must have shape"):
        M.extrinsic_mean(points[0])
    with pytest.raises(ValueError, match="finite Grassmann frames"):
        M.extrinsic_mean(points.at[0, 0, 0].set(2.0))
    with pytest.raises(ValueError, match="weights must have shape"):
        M.extrinsic_mean(points, jnp.ones(2))
    with pytest.raises(ValueError, match="nonnegative"):
        M.extrinsic_mean(points, jnp.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="positive total mass"):
        M.extrinsic_mean(points, jnp.zeros(3))
    with pytest.raises(TypeError, match="real-valued"):
        M.extrinsic_mean(points.astype(jnp.complex64))
    with pytest.raises(TypeError, match="real-valued"):
        M.extrinsic_mean(points, jnp.ones(3, dtype=jnp.complex64))


def test_grassmann_exact_ambient_hessian_conversion():
    M = Grassmann(size=(4, 2))
    X = jnp.eye(4)[:, :2]
    matrix = jnp.diag(jnp.array([1.0, 2.0, 4.0, 5.0]))
    U = jnp.zeros_like(X).at[2, 0].set(1.0)
    egrad = -2.0 * matrix @ X
    ehess_u = -2.0 * matrix @ U

    converted = M.ehess_to_rhess(X, egrad, ehess_u, U)

    assert M.operation_kind("ehess_to_rhess") == "exact"
    assert bool(M.is_tangent(X, converted))
    assert jnp.allclose(converted, -6.0 * U)


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


def test_grassmann_transport_is_equivariant_to_endpoint_frame_gauge(dtype_atol):
    M = Grassmann(size=(6, 3))
    key_x, key_y, key_u, key_r = jax.random.split(jax.random.key(31), 4)
    X = M.random_point(key_x)
    Y = M.random_point(key_y)
    U = M.random_tangent(key_u, X)
    R, _ = jnp.linalg.qr(jax.random.normal(key_r, shape=(3, 3)))

    transported = M.transport(X, Y, U)
    gauge_transport = M.transport(X, Y @ R, U)

    assert jnp.allclose(
        gauge_transport,
        transported @ R,
        atol=max(2e-8, 10.0 * dtype_atol),
        rtol=max(2e-8, 10.0 * dtype_atol),
    )


def test_grassmann_log_detects_numerically_orthogonal_subspaces(dtype_atol):
    M = Grassmann(size=(4, 2))
    orthogonal, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(40), shape=(4, 4)))
    X = orthogonal[:, :2]
    Y = orthogonal[:, 2:]

    tangent = M.log(X, Y)

    assert jnp.all(~jnp.isfinite(tangent))
    assert jnp.allclose(
        M.dist(X, Y),
        jnp.pi / jnp.sqrt(2.0),
        atol=max(2e-8, 5.0 * dtype_atol),
    )
