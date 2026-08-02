from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import Euclidean, Sphere
from geojax.learning import (
    as_manifold_data,
    biswas_ghosh_two_sample_test,
    empirical_wasserstein_distance,
    frechet_anova,
    sinkhorn_divergence,
    wasserstein_two_sample_test,
)
import geojax.learning._transport as transport_module


def test_exact_weighted_wasserstein_has_feasible_known_plan():
    manifold = Euclidean(1)
    x = jnp.array([[0.0], [1.0], [2.0]])
    y = jnp.array([[0.0], [3.0]])
    result = empirical_wasserstein_distance(
        manifold,
        x,
        y,
        weights_x=jnp.array([0.25, 0.5, 0.25]),
        weights_y=jnp.array([0.5, 0.5]),
    )

    assert result.converged
    assert jnp.allclose(jnp.sum(result.plan, axis=1), jnp.array([0.25, 0.5, 0.25]), atol=1e-6)
    assert jnp.allclose(jnp.sum(result.plan, axis=0), jnp.array([0.5, 0.5]), atol=1e-6)
    assert abs(float(result.diagnostics["duality_gap"])) < 1e-6
    assert jnp.isclose(result.cost, 1.5, atol=1e-6)


def test_transportation_simplex_matches_scipy_oracle_on_random_tiny_problems():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(208)
    manifold = Euclidean(1)
    for n, m in ((2, 3), (3, 3), (3, 4)):
        x = jnp.asarray(rng.normal(size=(n, 1)))
        y = jnp.asarray(rng.normal(size=(m, 1)))
        a = rng.random(n)
        b = rng.random(m)
        a /= a.sum()
        b /= b.sum()
        result = empirical_wasserstein_distance(
            manifold, x, y, weights_x=jnp.asarray(a), weights_y=jnp.asarray(b)
        )
        costs = np.asarray((x - y.T) ** 2)
        equalities, rhs = [], []
        for row in range(n):
            constraint = np.zeros((n, m))
            constraint[row] = 1.0
            equalities.append(constraint.ravel())
            rhs.append(a[row])
        for column in range(m - 1):
            constraint = np.zeros((n, m))
            constraint[:, column] = 1.0
            equalities.append(constraint.ravel())
            rhs.append(b[column])
        oracle = scipy_optimize.linprog(
            costs.ravel(),
            A_eq=np.asarray(equalities),
            b_eq=np.asarray(rhs),
            bounds=(0.0, None),
            method="highs",
        )
        assert oracle.success
        assert jnp.isclose(result.cost, oracle.fun, atol=2e-5, rtol=2e-5)


def test_biswas_ghosh_test_is_seed_reproducible_and_detects_separation():
    manifold = Euclidean(1)
    left = jnp.linspace(-0.2, 0.2, 8)[:, None]
    right = jnp.linspace(2.8, 3.2, 8)[:, None]
    first = biswas_ghosh_two_sample_test(
        manifold, left, right, n_permutations=39, key=209
    )
    second = biswas_ghosh_two_sample_test(
        manifold, left, right, n_permutations=39, key=209
    )
    assert jnp.array_equal(first.null_distribution, second.null_distribution)
    assert first.pvalue <= 0.05


def test_frechet_anova_asymptotic_and_permutation_paths_are_finite():
    manifold = Sphere(3)
    base_a = jnp.array([1.0, 0.0, 0.0])
    base_b = jnp.array([0.0, 1.0, 0.0])
    tangent_a = jnp.c_[jnp.zeros(6), jnp.linspace(-0.08, 0.08, 6), jnp.zeros(6)]
    tangent_b = jnp.c_[jnp.linspace(-0.08, 0.08, 6), jnp.zeros(6), jnp.zeros(6)]
    values = jnp.concatenate([manifold.exp(base_a, tangent_a), manifold.exp(base_b, tangent_b)])
    groups = jnp.repeat(jnp.arange(2), 6)
    asymptotic = frechet_anova(manifold, values, groups, maxiter=50)
    permutation = frechet_anova(
        manifold,
        values,
        groups,
        method="permutation",
        n_permutations=9,
        key=210,
        maxiter=30,
    )
    assert bool(jnp.isfinite(asymptotic.statistic))
    assert 0.0 <= float(asymptotic.pvalue) <= 1.0
    assert permutation.null_distribution.shape == (9,)


def test_wasserstein_permutation_test_uses_exact_transport_statistic():
    manifold = Euclidean(1)
    left = jnp.array([[0.0], [0.2], [0.4]])
    right = jnp.array([[2.0], [2.2], [2.4]])
    result = wasserstein_two_sample_test(
        manifold, left, right, n_permutations=5, key=211
    )
    exact = empirical_wasserstein_distance(manifold, left, right)
    assert jnp.isclose(result.statistic, exact.distance)
    assert result.null_distribution.shape == (5,)


def test_transport_handles_zero_weights_ties_and_explicit_data_objects():
    manifold = Euclidean(1)
    left = as_manifold_data(manifold, jnp.array([[0.0], [1.0], [2.0]]))
    right = as_manifold_data(manifold, jnp.array([[0.0], [1.0], [3.0]]))
    result = empirical_wasserstein_distance(
        manifold,
        left,
        right,
        p=1.0,
        weights_x=jnp.array([0.5, 0.5, 0.0]),
        weights_y=jnp.array([0.5, 0.0, 0.5]),
    )
    assert result.converged
    assert jnp.allclose(jnp.sum(result.plan, axis=1), jnp.array([0.5, 0.5, 0.0]))
    assert jnp.allclose(jnp.sum(result.plan, axis=0), jnp.array([0.5, 0.0, 0.5]))

    with pytest.raises(ValueError, match="at least 1"):
        empirical_wasserstein_distance(manifold, left, right, p=0.5)


def test_sinkhorn_contract_without_requiring_optional_ott(monkeypatch):
    manifold = Euclidean(1)
    left = jnp.array([[0.0], [1.0]])
    right = jnp.array([[1.0], [2.0]])

    def fake_cost(costs, a, b, epsilon):
        return jnp.sum(costs * a[:, None] * b[None, :]) + 0.0 * epsilon

    monkeypatch.setattr(transport_module, "_ott_cost", fake_cost)
    value = sinkhorn_divergence(manifold, left, right, epsilon=0.2)
    assert bool(jnp.isfinite(value))
    assert jnp.isclose(sinkhorn_divergence(manifold, left, left, epsilon=0.2), 0.0)
    with pytest.raises(ValueError, match="epsilon"):
        sinkhorn_divergence(manifold, left, right, epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        sinkhorn_divergence(manifold, left, right, p=0.5)


def test_inference_rejects_small_samples_bad_groups_and_bad_permutations():
    manifold = Euclidean(1)
    sample = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="at least two"):
        biswas_ghosh_two_sample_test(
            manifold, sample[:1], sample[1:], n_permutations=3, key=221
        )
    with pytest.raises(ValueError, match="positive"):
        biswas_ghosh_two_sample_test(
            manifold, sample[:2], sample[2:], n_permutations=0, key=221
        )
    with pytest.raises(ValueError, match="groups must have shape"):
        frechet_anova(manifold, sample, jnp.array([0, 0, 1]))
    with pytest.raises(ValueError, match="at least two groups"):
        frechet_anova(manifold, sample, jnp.array([0, 0, 0, 1]))
    with pytest.raises(ValueError, match="method must"):
        frechet_anova(manifold, sample, jnp.array([0, 0, 1, 1]), method="bootstrap")


def test_two_sample_null_case_is_seeded_and_calibrated_at_identity():
    manifold = Euclidean(1)
    sample = jnp.linspace(-1.0, 1.0, 6)[:, None]
    result = biswas_ghosh_two_sample_test(
        manifold,
        sample,
        sample,
        n_permutations=19,
        key=222,
    )
    assert 0.0 <= float(result.pvalue) <= 1.0
    assert result.null_distribution.shape == (19,)
