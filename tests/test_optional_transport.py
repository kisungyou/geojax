"""Integration checks against the real optional OTT transport backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ott")

from geojax.geometry import Euclidean
from geojax.learning import sinkhorn_divergence


@pytest.mark.parametrize("epsilon", [0.05, 0.5, 2.0])
def test_sinkhorn_translated_measure_has_exact_squared_translation_cost(epsilon):
    points = jnp.array([[-1.0], [0.0], [1.0]])
    weights = jnp.array([0.2, 0.3, 0.5])
    # Quadratic translation changes the regularized cross cost by d² after
    # the two fixed marginals cancel the linear terms; debiasing removes
    # the self-cost. This identity holds for every entropy regularization.
    value = sinkhorn_divergence(
        Euclidean(1),
        points,
        points + 0.4,
        epsilon=epsilon,
        weights_x=weights,
        weights_y=weights,
    )
    np.testing.assert_allclose(value, 0.4**2, rtol=1e-3, atol=2e-5)


def test_sinkhorn_translation_has_the_analytic_compiled_gradient():
    points = jnp.array([[-1.0], [0.0], [1.0]])

    def objective(shift):
        return sinkhorn_divergence(Euclidean(1), points, points + shift, epsilon=0.5)

    value, derivative = jax.jit(jax.value_and_grad(objective))(0.4)
    np.testing.assert_allclose(value, 0.4**2, rtol=1e-3, atol=2e-5)
    np.testing.assert_allclose(derivative, 0.8, rtol=1e-3, atol=2e-5)


def test_sinkhorn_ignores_zero_weight_support_points():
    points = jnp.array([[-1.0], [0.0], [1.0]])
    weights = jnp.array([0.5, 0.0, 0.5])
    value = sinkhorn_divergence(
        Euclidean(1),
        points,
        points + 0.4,
        epsilon=0.5,
        weights_x=weights,
        weights_y=weights,
    )
    np.testing.assert_allclose(value, 0.4**2, rtol=1e-3, atol=2e-5)
