from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.geometry import Oblique, PoincareBall, ProbabilitySimplex


def test_oblique_is_a_product_of_column_spheres(dtype_atol):
    M = Oblique(size=(4, 3))
    X = M.random_point(jax.random.key(0), sample_shape=(5,))
    U = M.random_tangent(jax.random.key(1), X)

    assert M.dim == 9
    assert jnp.allclose(jnp.linalg.norm(X, axis=-2), 1.0, atol=max(1e-10, dtype_atol))
    assert jnp.allclose(jnp.sum(X * U, axis=-2), 0.0, atol=max(1e-10, dtype_atol))
    assert jnp.all(M.belongs(X))
    assert jnp.all(M.is_tangent(X, U))


def test_oblique_hessian_conversion_satisfies_quadratic_sphere_formula(dtype_atol):
    M = Oblique(size=(3, 2))
    X = M.random_point(jax.random.key(30))
    U = M.random_tangent(jax.random.key(31), X)
    weights = jnp.array([[2.0, 1.0], [0.5, 3.0], [1.5, 0.25]])
    egrad = weights * X
    ehess_u = weights * U

    converted = M.ehess_to_rhess(X, egrad, ehess_u, U)
    expected = M.tangent_project(
        X,
        ehess_u - U * jnp.sum(X * egrad, axis=0, keepdims=True),
    )

    assert M.operation_kind("hessian") == "exact"
    assert bool(M.is_tangent(X, converted))
    assert jnp.allclose(converted, expected, atol=max(1e-10, dtype_atol))


def test_simplex_fisher_rao_distance_and_gradient_duality(dtype_atol):
    M = ProbabilitySimplex(size=3)
    p = jnp.array([0.2, 0.3, 0.5])
    q = jnp.array([0.1, 0.6, 0.3])
    U = M.tangent_project(p, jnp.array([0.3, -0.2, 0.1]))
    ambient_gradient = jnp.array([0.2, -0.7, 0.4])
    gradient = M.egrad_to_rgrad(p, ambient_gradient)

    expected = 2.0 * jnp.arccos(jnp.sum(jnp.sqrt(p * q)))
    assert jnp.allclose(M.dist(p, q), expected, atol=1e-12)
    assert jnp.allclose(M.inner(p, gradient, U), jnp.sum(ambient_gradient * U), atol=1e-12)

    raw = jnp.array([0.7, -0.4, 0.2])
    projected = M.tangent_project(p, raw)
    residual = raw - projected
    test_tangent = jnp.array([0.3, -0.1, -0.2])
    tolerance = max(1e-12, dtype_atol)
    assert jnp.allclose(jnp.sum(projected), 0.0, atol=tolerance)
    assert jnp.allclose(M.inner(p, residual, test_tangent), 0.0, atol=tolerance)


def test_simplex_exponential_rejects_a_path_that_leaves_and_reenters_the_interior():
    M = ProbabilitySimplex(size=2)
    p = jnp.array([0.5, 0.5])
    # In square-root coordinates this traverses one complete great circle.
    # Its endpoint equals p, but it crosses the simplex boundary twice.
    u = jnp.array([-2.0 * jnp.pi, 2.0 * jnp.pi])

    endpoint = M.exp(p, u)

    assert bool(M.is_tangent(p, u))
    assert bool(jnp.all(~jnp.isfinite(endpoint)))
    assert bool(jnp.all(jnp.isfinite(M.exp(p, 0.01 * u))))


def test_simplex_operations_preserve_valid_small_probabilities(dtype_atol):
    manifold = ProbabilitySimplex(size=3, eps=1e-3)
    point = jnp.array([1e-8, 0.4, 0.6 - 1e-8])
    other = jnp.array([2e-8, 0.3, 0.7 - 2e-8])

    assert bool(manifold.belongs(point))
    assert jnp.allclose(manifold.project(point), point, rtol=1e-10, atol=1e-15)
    assert jnp.allclose(
        manifold.exp(point, manifold.log(point, other)),
        other,
        rtol=max(2e-7, dtype_atol),
        atol=2e-12,
    )


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
