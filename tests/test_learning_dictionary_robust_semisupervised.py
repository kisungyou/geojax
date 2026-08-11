from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
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
from geojax.learning import _dictionary as dictionary_module
from geojax.learning import _robust as robust_module
from geojax.learning import _semisupervised as semisupervised_module


class _FiniteDistanceNonfiniteLogEuclidean(Euclidean):
    def dist(self, x, y):
        return jnp.linalg.norm(jnp.asarray(y) - jnp.asarray(x), axis=-1)

    def squared_dist(self, x, y):
        return jnp.sum((jnp.asarray(y) - jnp.asarray(x)) ** 2, axis=-1)

    def log(self, x, y):
        return jnp.full_like(super().log(x, y), jnp.nan)


class _NonfiniteExpEuclidean(Euclidean):
    def exp(self, x, u):
        return jnp.full_like(super().exp(x, u), jnp.nan)


class _NonfiniteInnerEuclidean(Euclidean):
    def inner(self, x, u, v):
        del x, u, v
        return jnp.asarray(jnp.nan)


class _IndefiniteInnerEuclidean(Euclidean):
    def inner(self, x, u, v):
        return -super().inner(x, u, v)


class _FiniteDistanceNonfiniteNormEuclidean(Euclidean):
    def dist(self, x, y):
        return jnp.linalg.norm(jnp.asarray(y) - jnp.asarray(x), axis=-1)

    def squared_dist(self, x, y):
        return jnp.sum((jnp.asarray(y) - jnp.asarray(x)) ** 2, axis=-1)

    def norm(self, x, u):
        del x, u
        return jnp.asarray(jnp.nan)


class _NonfiniteDistanceEuclidean(Euclidean):
    def dist(self, x, y):
        return jnp.full_like(super().dist(x, y), jnp.nan)


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
    assert jnp.allclose(
        result.objective,
        jnp.sum(weights * coding.diagnostics["coding_objectives"]),
    )
    assert bool(jnp.all(jnp.isfinite(coding.diagnostics["reconstruction_errors"])))


def test_dictionary_learning_optimizes_nested_product_atoms_end_to_end():
    manifold = Product(
        {
            "location": Euclidean(1),
            "state": [Euclidean(1)],
        }
    )
    data = {
        "location": jnp.array([[-2.0], [-1.0], [1.0], [2.0]]),
        "state": [jnp.array([[0.0], [1.0], [2.0], [3.0]])],
    }
    atom_indices = jnp.array([0, 3])
    initial_atoms = jax.tree_util.tree_map(lambda leaf: leaf[atom_indices], data)
    result = manifold_dictionary_learning(
        manifold,
        data,
        n_atoms=2,
        initial_atoms=initial_atoms,
        maxiter=1,
        coding_maxiter=20,
        center_maxiter=5,
        tol=1e-5,
    )

    assert result.codes.shape == (4, 2)
    assert bool(jnp.all(manifold.belongs(result.atoms)))
    assert bool(jnp.all(manifold.belongs(result.reconstructions)))
    assert bool(jnp.isfinite(result.objective))


def test_dictionary_methods_validate_atom_and_iteration_contracts():
    manifold = Euclidean(1)
    data = jnp.arange(4.0)[:, None]
    with pytest.raises(ValueError, match="maxiter"):
        geodesic_barycentric_coding(manifold, data, data[:2], maxiter=0)
    with pytest.raises(ValueError, match="between 1"):
        manifold_dictionary_learning(manifold, data, n_atoms=5, key=333)
    with pytest.raises(ValueError, match="explicit JAX random key"):
        manifold_dictionary_learning(manifold, data, n_atoms=2)
    with pytest.raises(ValueError, match="exactly n_atoms"):
        manifold_dictionary_learning(manifold, data, n_atoms=2, initial_atoms=data[:1])


def test_trimmed_mean_and_m_estimators_resist_a_large_outlier():
    manifold = Euclidean(1)
    clean = jnp.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    contaminated = jnp.concatenate([clean, jnp.array([[50.0]])])
    trimmed = trimmed_frechet_mean(manifold, contaminated, trim_fraction=1 / 6, maxiter=20)
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
        weighted.diagnostics["retained_weights"] * weighted.diagnostics["distances"][retained] ** 2
    )
    assert jnp.allclose(weighted.objective, expected)

    nearly_complete_trim = trimmed_frechet_mean(
        manifold,
        clean,
        trim_fraction=1.0 - 1e-6,
        initial_point=jnp.array([0.0]),
        maxiter=2,
    )
    assert nearly_complete_trim.diagnostics["retained_indices"].size >= 1
    assert bool(jnp.isfinite(nearly_complete_trim.objective))


def test_m_estimator_diagnostics_match_the_returned_point():
    manifold = Euclidean(1)
    values = jnp.array([[-1.0], [0.0], [1.0], [20.0]])
    result = geodesic_m_estimator(
        manifold,
        values,
        loss="huber",
        initial_point=jnp.array([5.0]),
        maxiter=1,
        center_maxiter=10,
        tol=0.0,
    )
    logs = manifold.log(result.point, values)
    expected = jnp.sum(
        result.diagnostics["unnormalized_effective_weights"][:, None] * logs,
        axis=0,
    )
    assert jnp.allclose(result.gradient_norm, manifold.norm(result.point, expected))


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
        label_propagation(manifold, data, jnp.array([0, -1, 0, 1, 1]), alpha=1.0)
    with pytest.raises(ValueError, match="n_neighbors"):
        label_propagation(
            manifold,
            data,
            jnp.array([0, -1, 0, 1, 1]),
            n_neighbors=5,
        )
    with pytest.raises(ValueError, match="component must contain at least one labeled"):
        label_propagation(
            manifold,
            jnp.array([[0.0], [0.1], [10.0], [10.1]]),
            jnp.array([0, -1, -1, -1]),
            bandwidth=0.2,
            n_neighbors=1,
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


def test_label_propagation_validates_sentinels_label_types_and_decodes_strings():
    manifold = Euclidean(1)
    data = jnp.arange(4.0)[:, None]

    with pytest.raises(ValueError, match="scalar sentinel"):
        label_propagation(
            manifold,
            data,
            np.array([0, -1, 1, -1]),
            unlabeled=np.array([-1]),
        )
    with pytest.raises(ValueError, match="finite"):
        label_propagation(manifold, data, np.array([0.0, -1.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="missing"):
        label_propagation(manifold, data, np.array([0, -1, None, 1], dtype=object))
    with pytest.raises(TypeError, match="comparable"):
        label_propagation(manifold, data, np.array([0, -1, "right", 1], dtype=object))

    result = label_propagation(
        manifold,
        data,
        np.array(["left", "?", "right", "?"], dtype=object),
        unlabeled="?",
        maxiter=10,
    )
    assert isinstance(result.predictions, np.ndarray)
    assert set(result.predictions.tolist()) <= {"left", "right"}


def test_regularized_regression_validates_masks_and_solver_certificates(monkeypatch):
    manifold = Euclidean(1)
    data = jnp.arange(4.0)[:, None]
    targets = jnp.array([0.0, 1.0, 2.0, 3.0])
    mask = jnp.array([True, False, True, False])

    with pytest.raises(ValueError, match="targets must have shape"):
        manifold_regularized_regression(manifold, data, targets[:-1], labeled_mask=mask)
    with pytest.raises(TypeError, match="booleans"):
        manifold_regularized_regression(
            manifold,
            data,
            targets,
            labeled_mask=jnp.array([1, 0, 1, 0]),
        )
    with pytest.raises(ValueError, match="both labeled and unlabeled"):
        manifold_regularized_regression(manifold, data, jnp.full((4,), jnp.nan))

    monkeypatch.setattr(
        semisupervised_module.jnp.linalg,
        "solve",
        lambda system, observed: jnp.full_like(observed, jnp.nan),
    )
    with pytest.raises(FloatingPointError, match="nonfinite linear-system solution"):
        manifold_regularized_regression(manifold, data, targets, labeled_mask=mask)

    monkeypatch.setattr(
        semisupervised_module.jnp.linalg,
        "solve",
        lambda system, observed: jnp.zeros_like(observed),
    )
    with pytest.raises(FloatingPointError, match="backward-error"):
        manifold_regularized_regression(manifold, data, targets, labeled_mask=mask)


def test_semisupervised_nonfinite_affinity_and_score_guards(monkeypatch):
    manifold = Euclidean(1)
    data = jnp.arange(4.0)[:, None]
    labels = np.array([0, -1, 1, -1])

    monkeypatch.setattr(
        semisupervised_module,
        "pairwise_distances",
        lambda *args, **kwargs: jnp.full((4, 4), jnp.nan),
    )
    with pytest.raises(FloatingPointError, match="affinity matrix"):
        label_propagation(manifold, data, labels, bandwidth=1.0)

    adapted = semisupervised_module.as_manifold_data(manifold, data)
    monkeypatch.setattr(
        semisupervised_module,
        "_prepare_graph",
        lambda *args, **kwargs: (
            adapted,
            jnp.zeros((4, 4)),
            jnp.full((4, 4), jnp.inf),
            1.0,
        ),
    )
    with pytest.raises(FloatingPointError, match="class scores"):
        label_propagation(manifold, data, labels, maxiter=1)


def test_barycentric_coding_rejects_nonfinite_and_indefinite_geometry(monkeypatch):
    values = jnp.array([[0.0], [1.0]])
    atoms = jnp.array([[0.0], [2.0]])

    with pytest.raises(ValueError, match="undefined atom logarithm"):
        geodesic_barycentric_coding(
            _FiniteDistanceNonfiniteLogEuclidean(1),
            values,
            atoms,
            maxiter=1,
        )
    with pytest.raises(FloatingPointError, match="Gram matrix is nonfinite"):
        geodesic_barycentric_coding(
            _NonfiniteInnerEuclidean(1),
            values,
            atoms,
            maxiter=1,
        )
    with pytest.raises(FloatingPointError, match="positive semidefinite"):
        geodesic_barycentric_coding(
            _IndefiniteInnerEuclidean(1),
            values,
            atoms,
            maxiter=1,
        )

    failed_fit = SimpleNamespace(point=jnp.array([jnp.nan]), converged=True)
    monkeypatch.setattr(dictionary_module, "frechet_mean", lambda *a, **k: failed_fit)
    with pytest.raises(FloatingPointError, match="nonfinite result"):
        geodesic_barycentric_coding(Euclidean(1), values, atoms, maxiter=1)


def test_dictionary_interpolation_checks_logarithm_and_exponential_domains():
    old = jnp.array([[0.0], [1.0]])
    candidate = jnp.array([[1.0], [2.0]])
    with pytest.raises(FloatingPointError, match="undefined logarithm"):
        dictionary_module._interpolate_atoms(
            _FiniteDistanceNonfiniteLogEuclidean(1),
            old,
            candidate,
            0.5,
        )
    with pytest.raises(FloatingPointError, match="exponential-map domain"):
        dictionary_module._interpolate_atoms(
            _NonfiniteExpEuclidean(1),
            old,
            candidate,
            0.5,
        )


def test_robust_summaries_reject_nonfinite_geometry_and_unconverged_centers(monkeypatch):
    values = jnp.array([[0.0], [1.0], [2.0]])
    fixed_fit = SimpleNamespace(point=jnp.array([0.0]), converged=True)
    monkeypatch.setattr(robust_module, "frechet_mean", lambda *a, **k: fixed_fit)
    with pytest.raises(FloatingPointError, match="stationarity vector"):
        trimmed_frechet_mean(
            _FiniteDistanceNonfiniteLogEuclidean(1),
            values,
            initial_point=jnp.array([0.0]),
            maxiter=1,
        )

    with pytest.raises(FloatingPointError, match="undefined logarithm"):
        geodesic_spatial_depth(
            _FiniteDistanceNonfiniteLogEuclidean(1),
            values[:1],
            values,
        )
    with pytest.raises(FloatingPointError, match="metric norm bound"):
        geodesic_spatial_depth(
            _FiniteDistanceNonfiniteNormEuclidean(1),
            values[:1],
            values,
        )
    with pytest.raises(FloatingPointError, match="nonfinite distances"):
        metric_distance_ranks(
            _NonfiniteDistanceEuclidean(1),
            values,
            center=jnp.array([0.0]),
        )

    with pytest.raises(ValueError, match="loss"):
        robust_module._robust_weights_and_loss(jnp.ones(2), 1.0, "unsupported")

    incomplete_fit = SimpleNamespace(point=jnp.array([0.0]), converged=False)
    monkeypatch.setattr(robust_module, "frechet_mean", lambda *a, **k: incomplete_fit)
    incomplete_trimmed = trimmed_frechet_mean(
        Euclidean(1),
        values,
        initial_point=jnp.array([0.0]),
        maxiter=1,
        tol=0.0,
    )
    assert incomplete_trimmed.reason == "an inner Frechet-mean solve was incomplete"
    incomplete_m = geodesic_m_estimator(
        Euclidean(1),
        values,
        initial_point=jnp.array([0.0]),
        maxiter=1,
        tol=0.0,
    )
    assert incomplete_m.reason == "an inner Frechet solve was incomplete"

    monkeypatch.setattr(robust_module, "frechet_mean", lambda *a, **k: fixed_fit)
    stalled_trimmed = trimmed_frechet_mean(
        Euclidean(1),
        values,
        initial_point=jnp.array([0.0]),
        maxiter=1,
        tol=0.0,
    )
    assert stalled_trimmed.reason == "updates stopped before stationarity was reached"
    stalled_m = geodesic_m_estimator(
        Euclidean(1),
        values,
        initial_point=jnp.array([0.0]),
        maxiter=1,
        tol=0.0,
    )
    assert stalled_m.reason == "updates stopped before stationarity was reached"

    monkeypatch.setattr(
        robust_module,
        "frechet_median",
        lambda *a, **k: SimpleNamespace(converged=False, point=jnp.array([0.0])),
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        metric_distance_ranks(Euclidean(1), values)
