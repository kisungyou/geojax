from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import Sphere, SphereExtrinsic


def test_sphere_constructor_and_geodesic_flow():
    M = Sphere(size=3)
    x = jnp.array([1.0, 0.0, 0.0])
    u = jnp.array([0.0, 0.2, 0.0])
    y, v = M.geodesic_flow(x, u, t=1.0)
    assert M.shape == (3,)
    assert M.dim == 2
    assert bool(M.belongs(y))
    assert bool(M.is_tangent(y, v))
    assert jnp.allclose(M.norm(x, u), M.norm(y, v), atol=1e-8)


def test_sphere_random_shapes():
    M = Sphere(size=5)
    xs = M.random_point(jax.random.key(0), sample_shape=(4,))
    assert xs.shape == (4, 5)
    assert jnp.all(M.belongs(xs))


def test_sphere_projection_preserves_direction_of_tiny_nonzero_vector(dtype_atol):
    M = Sphere(size=3, eps=1e-3)
    vector = jnp.array([1e-12, -2e-12, 2e-12])
    expected = vector / jnp.linalg.norm(vector)

    projected = M.project(vector)

    assert bool(M.belongs(projected))
    assert jnp.allclose(projected, expected, atol=max(1e-10, dtype_atol))


def test_sphere_extrinsic_identity_embedding_and_chordal_distance():
    M = SphereExtrinsic(size=3)
    x = jnp.array([1.0, 0.0, 0.0])
    y = jnp.array([0.0, 1.0, 0.0])

    assert jnp.array_equal(M.embed(x), x)
    assert jnp.allclose(M.chordal_dist(x, y), jnp.sqrt(2.0), atol=1e-8)
    assert M.chordal_dist(x, y) <= M.dist(x, y)


def test_sphere_extrinsic_mean_projects_ambient_mean():
    M = SphereExtrinsic(size=2)
    points = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    expected = jnp.array([1.0, 1.0]) / jnp.sqrt(2.0)

    mean = M.extrinsic_mean(points)

    assert bool(M.belongs(mean))
    assert jnp.allclose(mean, expected, atol=1e-8)


def test_sphere_extrinsic_mean_is_undefined_for_zero_ambient_mean():
    M = SphereExtrinsic(size=2)
    points = jnp.array([[1.0, 0.0], [-1.0, 0.0]])

    assert jnp.all(jnp.isnan(M.extrinsic_mean(points)))


def test_sphere_extrinsic_mean_validates_points_and_weights():
    M = SphereExtrinsic(3)
    points = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        M.extrinsic_mean(points[0])
    with pytest.raises(ValueError, match="sphere points"):
        M.extrinsic_mean(points.at[0, 0].set(2.0))
    with pytest.raises(ValueError, match="shape"):
        M.extrinsic_mean(points, jnp.ones(3))
    with pytest.raises(ValueError, match="nonnegative"):
        M.extrinsic_mean(points, jnp.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="positive total"):
        M.extrinsic_mean(points, jnp.zeros(2))
    with pytest.raises(TypeError, match="real-valued"):
        M.extrinsic_mean(points.astype(jnp.complex64))
    with pytest.raises(TypeError, match="real-valued"):
        M.extrinsic_mean(points, jnp.ones(2, dtype=jnp.complex64))
