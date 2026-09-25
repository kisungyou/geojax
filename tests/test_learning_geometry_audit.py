"""JAX boundary and exact-cost checks for learning geometry kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import Euclidean, Product
from geojax.learning import (
    as_manifold_data,
    geodesic_barycentric_coding,
    pairwise_distances,
    sinkhorn_divergence,
)
from geojax.learning import _transport


@pytest.mark.parametrize("adapted", [False, True])
def test_pairwise_kernels_accept_closed_constants_and_traced_data(adapted):
    m = Euclidean(1)
    points = jnp.array([[-1.0], [0.0], [1.0]])
    left = as_manifold_data(m, points, check="shape") if adapted else points

    def objective(shift):
        right = as_manifold_data(m, points + shift, check="shape") if adapted else points + shift
        return jnp.sum(pairwise_distances(m, left, right, squared=True))

    value, grad = jax.jit(jax.value_and_grad(objective))(0.4)
    np.testing.assert_allclose(value, 12 + 9 * 0.4**2, rtol=2e-6)
    np.testing.assert_allclose(grad, 18 * 0.4, rtol=2e-6)


def test_sinkhorn_quadratic_cost_preserves_hessian_at_coincidence(monkeypatch):
    def independent_cost(costs, a, b, epsilon):
        return jnp.sum(costs * a[:, None] * b[None, :])

    monkeypatch.setattr(_transport, "_ott_cost", independent_cost)
    points = jnp.array([[0.0]])

    def objective(shift):
        return sinkhorn_divergence(Euclidean(1), points, points + shift)

    np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(objective)))(0.0), 2.0, rtol=2e-6)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_barycentric_objective_resolves_near_exact_reconstructions(scale):
    points = jnp.array([[0.3]]) * scale
    atoms = jnp.array([[-np.sqrt(2.0)], [np.sqrt(3.0)]]) * scale
    result = geodesic_barycentric_coding(
        Euclidean(1), points, atoms, ridge=0.0, maxiter=200, reconstruction_maxiter=20
    )
    # Use host double arithmetic on the actual represented tangent vectors.
    # The error bound accounts for rounding the small weighted residual,
    # before squaring; it does not allow cancellation of a Gram quadratic.
    logs = np.asarray(atoms - points[0]).astype(np.float64)
    residual = np.asarray(result.codes).astype(np.float64) @ logs
    expected = np.mean(np.sum(residual**2, axis=-1))
    eps = np.finfo(np.asarray(result.codes).dtype).eps
    residual_roundoff = 4 * eps * np.max(np.abs(logs))
    np.testing.assert_allclose(
        result.objective,
        expected,
        rtol=8 * eps,
        atol=residual_roundoff**2 + 2 * residual_roundoff * np.sqrt(expected),
    )
    assert float(result.objective) >= 0.0
    assert np.all(np.asarray(result.diagnostics["objective_histories"][0]) >= 0.0)


@pytest.mark.parametrize("product", [False, True])
def test_barycentric_ridge_survives_large_canceling_tangent_vectors(product):
    locations = jnp.array([[-1e8], [1e8]])
    if product:
        manifold = Product({"location": Euclidean(1), "other": [Euclidean(1)]})
        atoms = {"location": locations, "other": [2 * locations]}
        points = {"location": jnp.zeros((1, 1)), "other": [jnp.zeros((1, 1))]}
    else:
        manifold = Euclidean(1)
        atoms = locations
        points = jnp.zeros((1, 1))
    ridge = 1e-6
    result = geodesic_barycentric_coding(manifold, points, atoms, ridge=ridge, maxiter=2)
    np.testing.assert_allclose(result.codes, [[0.5, 0.5]], rtol=0.0, atol=0.0)
    # The residual vanishes exactly. The entire objective is the ridge
    # penalty, although ridge is below one ulp of the Gram diagonal.
    np.testing.assert_allclose(result.objective, ridge / 2, rtol=2e-7, atol=0.0)
