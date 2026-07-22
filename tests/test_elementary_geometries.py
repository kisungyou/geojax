from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.geometry import Oblique, PoincareBall, ProbabilitySimplex


def test_oblique_is_a_product_of_column_spheres():
    M = Oblique(size=(4, 3))
    X = M.random_point(jax.random.key(0), sample_shape=(5,))
    U = M.random_tangent(jax.random.key(1), X)

    assert M.dim == 9
    assert jnp.allclose(jnp.linalg.norm(X, axis=-2), 1.0, atol=1e-10)
    assert jnp.allclose(jnp.sum(X * U, axis=-2), 0.0, atol=1e-10)
    assert jnp.all(M.belongs(X))
    assert jnp.all(M.is_tangent(X, U))


def test_simplex_fisher_rao_distance_and_gradient_duality():
    M = ProbabilitySimplex(size=3)
    p = jnp.array([0.2, 0.3, 0.5])
    q = jnp.array([0.1, 0.6, 0.3])
    U = M.tangent_project(p, jnp.array([0.3, -0.2, 0.1]))
    ambient_gradient = jnp.array([0.2, -0.7, 0.4])
    gradient = M.egrad_to_rgrad(p, ambient_gradient)

    expected = 2.0 * jnp.arccos(jnp.sum(jnp.sqrt(p * q)))
    assert jnp.allclose(M.dist(p, q), expected, atol=1e-12)
    assert jnp.allclose(M.inner(p, gradient, U), jnp.sum(ambient_gradient * U), atol=1e-12)


def test_poincare_origin_formulas_and_isometric_transport():
    M = PoincareBall(size=2)
    origin = jnp.zeros(2)
    point = jnp.array([0.3, -0.2])
    tangent = jnp.array([0.1, 0.25])
    transported = M.transport(origin, point, tangent)

    expected_distance = 2.0 * jnp.arctanh(jnp.linalg.norm(point))
    assert jnp.allclose(M.dist(origin, point), expected_distance, atol=1e-12)
    assert jnp.allclose(M.exp(origin, M.log(origin, point)), point, atol=1e-12)
    assert jnp.allclose(M.norm(origin, tangent), M.norm(point, transported), atol=1e-12)
