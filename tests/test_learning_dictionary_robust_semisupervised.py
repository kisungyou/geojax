from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import Euclidean, Product, Sphere, Torus
from geojax.learning import (
    geodesic_barycentric_coding,
    geodesic_m_estimator,
    geodesic_spatial_depth,
    label_propagation,
    manifold_dictionary_learning,
    manifold_regularized_regression,
    metric_distance_ranks,
    trimmed_frechet_mean,
)


def test_barycentric_codes_reconstruct_points_between_euclidean_atoms():
    manifold = Euclidean(1)
    data = jnp.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    atoms = jnp.array([[-2.0], [2.0]])
    result = geodesic_barycentric_coding(
        manifold, data, atoms, ridge=0.0, maxiter=200, reconstruction_maxiter=30
    )

    assert jnp.allclose(jnp.sum(result.codes, axis=1), 1.0)
    assert bool(jnp.all(result.codes >= 0.0))
    assert jnp.allclose(result.reconstructions, data, atol=2e-4)
    assert result.objective < 1e-6


def test_barycentric_coding_supports_nested_product_atoms():
    manifold = Product({"direction": Sphere(3), "phase": [Torus(1)]})
    atoms = manifold.random_point(jax.random.key(330), sample_shape=(3,))
    data = manifold.random_point(jax.random.key(331), sample_shape=(4,))
    result = geodesic_barycentric_coding(
        manifold, data, atoms, maxiter=20, reconstruction_maxiter=20
    )
    assert result.codes.shape == (4, 3)
    assert bool(jnp.all(manifold.belongs(result.reconstructions)))
    assert bool(jnp.isfinite(result.objective))


def test_dictionary_learning_has_nonincreasing_objective_and_valid_atoms():
    manifold = Euclidean(1)
    data = jnp.array([[-3.0], [-2.5], [-2.0], [2.0], [2.5], [3.0]])
    weights = jnp.array([0.40, 0.30, 0.20, 0.05, 0.03, 0.02])
    result = manifold_dictionary_learning(
        manifold,
        data,
        n_atoms=2,
        key=332,
        sample_weight=weights,
        maxiter=3,
        coding_maxiter=60,
        center_maxiter=30,
    )
    history = result.diagnostics["objective_history"]
    assert bool(jnp.all(jnp.diff(history) <= 1e-7))
    assert result.atoms.shape == (2, 1)
    assert jnp.allclose(jnp.sum(result.codes, axis=1), 1.0)
    coding = result.diagnostics["coding_result"]
    penalties = coding.diagnostics["reconstruction_errors"] + 1e-6 * jnp.sum(
        result.codes**2, axis=1
    )
    assert jnp.allclose(result.objective, jnp.sum(weights * penalties))


def test_dictionary_methods_validate_atom_and_iteration_contracts():
    manifold = Euclidean(1)
    data = jnp.arange(4.0)[:, None]
    with pytest.raises(ValueError, match="iteration"):
        geodesic_barycentric_coding(manifold, data, data[:2], maxiter=0)
    with pytest.raises(ValueError, match="between 1"):
        manifold_dictionary_learning(manifold, data, n_atoms=5, key=333)
    with pytest.raises(ValueError, match="explicit JAX random key"):
        manifold_dictionary_learning(manifold, data, n_atoms=2)
    with pytest.raises(ValueError, match="exactly n_atoms"):
        manifold_dictionary_learning(
            manifold, data, n_atoms=2, initial_atoms=data[:1]
        )


def test_trimmed_mean_and_m_estimators_resist_a_large_outlier():
    manifold = Euclidean(1)
    clean = jnp.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    contaminated = jnp.concatenate([clean, jnp.array([[50.0]])])
    trimmed = trimmed_frechet_mean(
        manifold, contaminated, trim_fraction=1 / 6, maxiter=20
    )
    huber = geodesic_m_estimator(manifold, contaminated, loss="huber", maxiter=30)
    cauchy = geodesic_m_estimator(manifold, contaminated, loss="cauchy", maxiter=30)
    tukey = geodesic_m_estimator(manifold, contaminated, loss="tukey", maxiter=30)

    assert abs(float(trimmed.point[0])) < 0.05
    assert abs(float(huber.point[0])) < 1.0
    assert abs(float(cauchy.point[0])) < 0.5
    assert abs(float(tukey.point[0])) < 0.5
    assert 5 not in trimmed.diagnostics["retained_indices"].tolist()

    weighted = trimmed_frechet_mean(
        manifold,
        contaminated,
        trim_fraction=1 / 6,
        sample_weight=jnp.array([0.40, 0.25, 0.15, 0.10, 0.09, 0.01]),
        maxiter=20,
    )
    retained = weighted.diagnostics["retained_indices"]
    expected = jnp.sum(
        weighted.diagnostics["retained_weights"]
        * weighted.diagnostics["distances"][retained] ** 2
    )
    assert jnp.allclose(weighted.objective, expected)


def test_spatial_depth_and_metric_ranks_have_expected_ordering():
    manifold = Euclidean(1)
    data = jnp.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    depths = geodesic_spatial_depth(manifold, jnp.array([[0.0], [5.0]]), data)
    ranks = metric_distance_ranks(manifold, data)

    assert depths[0] > depths[1]
    assert jnp.argmax(ranks.ranks) in jnp.array([0, 4])
    assert jnp.argmin(ranks.ranks) == 2
    assert jnp.allclose(jnp.sort(ranks.ranks), jnp.array([0.2, 0.5, 0.5, 0.9, 0.9]))


def test_robust_methods_validate_controls():
    manifold = Euclidean(1)
    data = jnp.arange(5.0)[:, None]
    with pytest.raises(ValueError, match="trim_fraction"):
        trimmed_frechet_mean(manifold, data, trim_fraction=1.0)
    with pytest.raises(ValueError, match="loss"):
        geodesic_m_estimator(manifold, data, loss="absolute")
    with pytest.raises(ValueError, match="scale"):
        geodesic_m_estimator(manifold, data, scale=0.0)


def test_label_propagation_recovers_two_geodesic_clusters():
    manifold = Euclidean(1)
    data = jnp.array([[-3.0], [-2.7], [-2.4], [2.4], [2.7], [3.0]])
    labels = jnp.array([0, -1, 0, 1, -1, 1])
    result = label_propagation(
        manifold,
        data,
        labels,
        bandwidth=0.7,
        n_neighbors=2,
        maxiter=200,
    )
    assert jnp.array_equal(result.predictions, jnp.array([0, 0, 0, 1, 1, 1]))
    assert jnp.allclose(jnp.sum(result.scores, axis=1), 1.0, atol=5e-3)


def test_graph_neighbors_exclude_self_even_with_duplicate_points():
    manifold = Euclidean(1)
    data = jnp.array([[0.0], [0.0], [0.0], [10.0]])
    result = label_propagation(
        manifold,
        data,
        jnp.array([0, -1, -1, 1]),
        n_neighbors=1,
        maxiter=20,
    )
    affinity = result.diagnostics["affinity"]
    assert jnp.allclose(jnp.diag(affinity), 0.0)
    assert affinity[0, 2] > 0.0
    assert affinity[1, 2] == 0.0


def test_manifold_regularized_regression_interpolates_unlabeled_targets():
    manifold = Euclidean(1)
    data = jnp.linspace(-1.0, 1.0, 9)[:, None]
    truth = 2.0 * data[:, 0]
    targets = truth.at[1::2].set(jnp.nan)
    result = manifold_regularized_regression(
        manifold,
        data,
        targets,
        bandwidth=0.35,
        n_neighbors=3,
        ambient_regularization=1e-4,
        intrinsic_regularization=0.1,
    )
    assert bool(jnp.all(jnp.isfinite(result.predictions)))
    assert jnp.mean((result.predictions[1::2] - truth[1::2]) ** 2) < 0.15


def test_semisupervised_methods_validate_labels_masks_and_graph_controls():
    manifold = Euclidean(1)
    data = jnp.arange(5.0)[:, None]
    with pytest.raises(ValueError, match="both labeled and unlabeled"):
        label_propagation(manifold, data, jnp.arange(5), bandwidth=1.0)
    with pytest.raises(ValueError, match="alpha"):
        label_propagation(
            manifold, data, jnp.array([0, -1, 0, 1, 1]), alpha=1.0
        )
    with pytest.raises(ValueError, match="n_neighbors"):
        label_propagation(
            manifold,
            data,
            jnp.array([0, -1, 0, 1, 1]),
            n_neighbors=5,
        )
    with pytest.raises(ValueError, match="both labeled and unlabeled"):
        manifold_regularized_regression(manifold, data, jnp.arange(5.0))
    with pytest.raises(ValueError, match="ambient_regularization"):
        manifold_regularized_regression(
            manifold,
            data,
            jnp.array([0.0, jnp.nan, 2.0, jnp.nan, 4.0]),
            ambient_regularization=0.0,
        )
