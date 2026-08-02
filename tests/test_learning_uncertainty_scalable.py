from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import Euclidean, FixedRank, Product, Sphere, Torus
from geojax.learning import (
    LearningCapabilityError,
    bootstrap_frechet_mean,
    energy_two_sample_test,
    kernel_mmd_two_sample_test,
    minibatch_frechet_mean,
    minibatch_kmeans,
    paired_frechet_test,
    streaming_frechet_mean,
)


def test_bootstrap_mean_is_reproducible_and_returns_a_geodesic_ball():
    manifold = Euclidean(1)
    values = jnp.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    first = bootstrap_frechet_mean(
        manifold, values, n_bootstrap=12, confidence_level=0.8, key=310, maxiter=30
    )
    second = bootstrap_frechet_mean(
        manifold, values, n_bootstrap=12, confidence_level=0.8, key=310, maxiter=30
    )

    assert first.replicates.shape == (12, 1)
    assert jnp.array_equal(first.replicates, second.replicates)
    assert first.confidence_radius >= 0.0
    assert jnp.allclose(first.estimate, jnp.array([0.0]), atol=1e-5)


def test_energy_mmd_and_paired_tests_detect_large_location_changes():
    manifold = Euclidean(1)
    left = jnp.linspace(-0.3, 0.3, 8)[:, None]
    right = left + 3.0
    energy = energy_two_sample_test(manifold, left, right, n_permutations=63, key=311)
    mmd = kernel_mmd_two_sample_test(manifold, left, right, n_permutations=63, key=312)
    paired = paired_frechet_test(manifold, left, right, n_permutations=63, key=313)

    assert energy.statistic > 1.0
    assert mmd.statistic > 0.5
    assert paired.statistic > 2.9
    assert energy.pvalue <= 0.05
    assert mmd.pvalue <= 0.05
    assert paired.pvalue <= 0.05
    assert mmd.diagnostics["effective_psd_tolerance"] >= 1e-8
    assert (
        jnp.min(mmd.diagnostics["kernel_eigenvalues"])
        >= -mmd.diagnostics["effective_psd_tolerance"]
    )


def test_uncertainty_methods_validate_keys_sizes_and_bandwidths():
    manifold = Euclidean(1)
    values = jnp.arange(4.0)[:, None]
    with pytest.raises(ValueError, match="explicit JAX random key"):
        bootstrap_frechet_mean(manifold, values, n_bootstrap=2, key=None)
    with pytest.raises(ValueError, match="positive"):
        bootstrap_frechet_mean(manifold, values, n_bootstrap=0, key=314)
    with pytest.raises(ValueError, match="strictly"):
        bootstrap_frechet_mean(
            manifold, values, n_bootstrap=2, confidence_level=1.0, key=314
        )
    with pytest.raises(ValueError, match="bandwidth"):
        kernel_mmd_two_sample_test(
            manifold, values, values, bandwidth=0.0, n_permutations=2, key=314
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        kernel_mmd_two_sample_test(
            manifold,
            values,
            values,
            kernel=lambda distances: -jnp.eye(distances.shape[0]),
            n_permutations=2,
            key=314,
        )
    with pytest.raises(ValueError, match="same size"):
        paired_frechet_test(
            manifold, values, values[:-1], n_permutations=2, key=314
        )


def test_streaming_mean_is_exact_for_weighted_euclidean_data():
    manifold = Euclidean(2)
    values = jnp.array([[0.0, 1.0], [2.0, 3.0], [8.0, -1.0]])
    weights = jnp.array([0.2, 0.3, 0.5])
    result = streaming_frechet_mean(manifold, values, sample_weight=weights)

    assert jnp.allclose(result.point, jnp.sum(weights[:, None] * values, axis=0))
    assert result.reason == "single pass completed"
    assert result.diagnostics["step_sizes"].shape == (3,)


def test_minibatch_mean_is_reproducible_and_improves_a_bad_initial_point():
    manifold = Euclidean(1)
    values = jnp.linspace(-1.0, 1.0, 20)[:, None]
    initial = jnp.array([20.0])
    first = minibatch_frechet_mean(
        manifold,
        values,
        batch_size=5,
        epochs=8,
        key=315,
        initial_point=initial,
        learning_rate=0.6,
        decay=0.2,
    )
    second = minibatch_frechet_mean(
        manifold,
        values,
        batch_size=5,
        epochs=8,
        key=315,
        initial_point=initial,
        learning_rate=0.6,
        decay=0.2,
    )
    assert jnp.array_equal(first.point, second.point)
    assert abs(float(first.point[0])) < 1.0
    assert first.objective < jnp.mean((values[:, 0] - 20.0) ** 2)


def test_minibatch_kmeans_separates_clusters_and_supports_product_points():
    manifold = Euclidean(1)
    values = jnp.array([[-3.0], [-2.8], [-2.6], [2.6], [2.8], [3.0]])
    result = minibatch_kmeans(
        manifold,
        values,
        n_clusters=2,
        batch_size=3,
        epochs=10,
        key=316,
    )
    assert jnp.all(result.labels[:3] == result.labels[0])
    assert jnp.all(result.labels[3:] == result.labels[3])
    assert result.labels[0] != result.labels[3]

    product = Product({"direction": Sphere(3), "phase": Torus(1)})
    product_values = product.random_point(jax.random.key(317), sample_shape=(6,))
    product_result = minibatch_kmeans(
        product,
        product_values,
        n_clusters=2,
        batch_size=3,
        epochs=2,
        key=318,
    )
    assert bool(jnp.all(product.belongs(product_result.centers)))

    zero_mass = minibatch_kmeans(
        Euclidean(1),
        jnp.array([[0.0], [10.0], [11.0], [12.0]]),
        n_clusters=2,
        batch_size=2,
        epochs=3,
        key=318,
        sample_weight=jnp.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert bool(jnp.all(jnp.isfinite(zero_mass.centers)))
    assert bool(jnp.isfinite(zero_mass.objective))


def test_scalable_methods_validate_controls_and_exact_capabilities():
    manifold = Euclidean(1)
    values = jnp.arange(4.0)[:, None]
    with pytest.raises(ValueError, match="batch_size"):
        minibatch_frechet_mean(manifold, values, batch_size=0, epochs=2, key=319)
    with pytest.raises(ValueError, match="epochs"):
        minibatch_kmeans(
            manifold, values, n_clusters=2, batch_size=2, epochs=0, key=319
        )
    proxy = FixedRank((3, 3), rank=1)
    proxy_values = proxy.random_point(jax.random.key(320), sample_shape=(3,))
    with pytest.raises(LearningCapabilityError, match="dist=proxy"):
        streaming_frechet_mean(proxy, proxy_values)
