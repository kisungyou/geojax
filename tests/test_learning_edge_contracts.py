from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import pytest

from geojax.geometry import Euclidean
from geojax.learning import (
    energy_two_sample_test,
    geodesic_m_estimator,
    geodesic_regression,
    kernel_mmd_two_sample_test,
    label_propagation,
    local_polynomial_regression,
    manifold_dictionary_learning,
    manifold_regularized_regression,
    metric_distance_ranks,
    minibatch_frechet_mean,
    minibatch_kmeans,
    nearest_centroid_classifier,
    streaming_frechet_mean,
    tangent_space_discriminant_analysis,
    tangent_space_logistic_regression,
    trimmed_frechet_mean,
)
from geojax.learning._capabilities import operation_kind
from geojax.learning._features import fit_tangent_feature_map


def _classification_data():
    values = jnp.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = jnp.array([0, 0, 1, 1])
    return values, labels


def test_classifier_weight_prior_and_optimizer_contracts():
    manifold = Euclidean(1)
    values, labels = _classification_data()

    with pytest.raises(ValueError, match="sample_weight"):
        nearest_centroid_classifier(
            manifold,
            values,
            labels,
            sample_weight=jnp.array([1.0, 1.0, -1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="regularization"):
        tangent_space_logistic_regression(
            manifold, values, labels, regularization=-1.0
        )
    with pytest.raises(ValueError, match="priors"):
        tangent_space_discriminant_analysis(
            manifold, values, labels, priors=jnp.array([1.0, 0.0])
        )

    model = tangent_space_logistic_regression(
        manifold,
        values,
        labels,
        tol=1e6,
        maxiter=2,
    )
    assert model.converged
    assert model.reason == "gradient tolerance reached"


def test_tangent_feature_controls_empty_basis_and_covariance_guard():
    manifold = Euclidean(2)
    values = jnp.zeros((4, 2))
    labels = jnp.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="rank_tolerance"):
        fit_tangent_feature_map(manifold, values, rank_tolerance=-1.0)
    with pytest.raises(ValueError, match="n_components"):
        fit_tangent_feature_map(manifold, values, n_components=0)

    feature_map, features = fit_tangent_feature_map(
        manifold,
        values,
        base_point=jnp.array([1.0, 1.0]),
    )
    assert features.shape == (4, 1)
    assert feature_map.diagnostics["mean_result"] is None

    empty_map, empty_features = fit_tangent_feature_map(manifold, values)
    assert empty_features.shape == (4, 0)
    assert empty_map.basis == ()

    empty_model = tangent_space_discriminant_analysis(manifold, values, labels)
    assert empty_model.predict(values).shape == (4,)

    separated = jnp.array([[-2.0, -0.2], [-1.0, 0.2], [1.0, -0.2], [2.0, 0.2]])
    fitted = tangent_space_discriminant_analysis(manifold, separated, labels)
    bad_covariance = jnp.eye(fitted.covariances.shape[-1]).at[0, 0].set(-1.0)
    invalid = replace(fitted, covariances=bad_covariance[None, ...])
    with pytest.raises(FloatingPointError, match="positive definite"):
        invalid.predict(separated)


def test_response_regression_alternate_paths_and_validation():
    manifold = Euclidean(1)
    predictors = jnp.linspace(-1.0, 1.0, 7)
    responses = (2.0 + predictors)[:, None]

    with pytest.raises(ValueError, match="shape"):
        geodesic_regression(manifold, predictors[:-1], responses)
    with pytest.raises(ValueError, match="finite"):
        geodesic_regression(manifold, predictors.at[0].set(jnp.nan), responses)
    with pytest.raises(ValueError, match="maxiter"):
        geodesic_regression(manifold, predictors, responses, maxiter=0)
    with pytest.raises(ValueError, match="maxiter"):
        local_polynomial_regression(
            manifold, predictors, responses, bandwidth=0.4, maxiter=0
        )

    geodesic = geodesic_regression(
        manifold,
        predictors,
        responses,
        initial_point=jnp.array([2.0]),
        maxiter=20,
    )
    with pytest.raises(ValueError, match="finite scalar"):
        geodesic.predict(jnp.array([jnp.nan]))

    compact = local_polynomial_regression(
        manifold,
        predictors,
        responses,
        bandwidth=0.1,
        degree=1,
        kernel=lambda distances, bandwidth: (distances <= bandwidth).astype(float),
        maxiter=20,
    )
    assert compact.predict(4.0).shape == (1,)

    local_linear = local_polynomial_regression(
        manifold,
        predictors,
        responses,
        bandwidth=0.8,
        degree=1,
        maxiter=20,
    )
    assert jnp.allclose(local_linear.predict(1.5), jnp.array([3.5]), atol=2e-2)
    with pytest.raises(ValueError, match="finite scalar"):
        local_linear.predict(jnp.array([[0.0]]))


def test_uncertainty_tests_cover_small_sample_and_kernel_contracts():
    manifold = Euclidean(1)
    one = jnp.array([[0.0]])
    two = jnp.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="at least two"):
        energy_two_sample_test(manifold, one, two, n_permutations=2, key=401)
    with pytest.raises(ValueError, match="at least two"):
        kernel_mmd_two_sample_test(manifold, two, one, n_permutations=2, key=401)
    with pytest.raises(ValueError, match="psd_tolerance"):
        kernel_mmd_two_sample_test(
            manifold, two, two + 2.0, psd_tolerance=-1.0, n_permutations=2, key=401
        )
    with pytest.raises(ValueError, match="square matrix"):
        kernel_mmd_two_sample_test(
            manifold,
            two,
            two + 2.0,
            kernel=lambda distances: distances[:, 0],
            n_permutations=2,
            key=401,
        )

    exploratory = kernel_mmd_two_sample_test(
        manifold,
        two,
        two + 2.0,
        kernel=lambda distances: -jnp.eye(distances.shape[0]),
        check_psd=False,
        n_permutations=2,
        key=401,
    )
    assert jnp.isfinite(exploratory.statistic)


def test_scalable_methods_cover_initialization_zero_weights_and_convergence():
    manifold = Euclidean(1)
    values = jnp.zeros((4, 1))

    streaming = streaming_frechet_mean(
        manifold,
        values,
        sample_weight=jnp.array([1.0, 0.0, 0.0, 0.0]),
        initial_point=jnp.array([2.0]),
    )
    assert jnp.allclose(streaming.point, jnp.array([1.0]))

    minibatch = minibatch_frechet_mean(
        manifold,
        values,
        batch_size=1,
        epochs=2,
        key=402,
        sample_weight=jnp.array([1.0, 0.0, 0.0, 0.0]),
        initial_point=jnp.array([0.0]),
    )
    assert minibatch.converged

    clustering = minibatch_kmeans(
        manifold,
        values,
        n_clusters=1,
        batch_size=2,
        epochs=3,
        key=403,
    )
    assert clustering.converged
    assert clustering.reason == "objective tolerance reached"

    with pytest.raises(ValueError, match="positive total mass"):
        streaming_frechet_mean(manifold, values, sample_weight=jnp.zeros((4,)))
    with pytest.raises(ValueError, match="epochs"):
        minibatch_frechet_mean(
            manifold, values, batch_size=2, epochs=0, key=402
        )
    with pytest.raises(ValueError, match="n_clusters"):
        minibatch_kmeans(
            manifold, values, n_clusters=0, batch_size=2, epochs=2, key=403
        )
    with pytest.raises(ValueError, match="batch_size"):
        minibatch_kmeans(
            manifold, values, n_clusters=1, batch_size=5, epochs=2, key=403
        )


def test_robust_location_explicit_initialization_scale_and_center():
    manifold = Euclidean(1)
    values = jnp.array([[-1.0], [0.0], [1.0], [20.0]])

    trimmed = trimmed_frechet_mean(
        manifold,
        values,
        trim_fraction=0.25,
        initial_point=jnp.array([0.0]),
        maxiter=5,
    )
    robust = geodesic_m_estimator(
        manifold,
        values,
        loss="tukey",
        scale=1e-3,
        initial_point=jnp.array([10.0]),
        maxiter=2,
    )
    ranks = metric_distance_ranks(manifold, values, center=jnp.array([0.0]))

    assert bool(jnp.isfinite(trimmed.objective))
    assert bool(jnp.isfinite(robust.objective))
    assert ranks.diagnostics["center_fit"] is None

    with pytest.raises(ValueError, match="iteration limits"):
        trimmed_frechet_mean(manifold, values, maxiter=0)
    with pytest.raises(ValueError, match="iteration limits"):
        geodesic_m_estimator(manifold, values, center_maxiter=0)


def test_dictionary_and_semisupervised_alternate_contracts():
    manifold = Euclidean(1)
    values = jnp.array([[-2.0], [-1.0], [1.0], [2.0]])
    initial_atoms = values[jnp.array([0, 3])]
    dictionary = manifold_dictionary_learning(
        manifold,
        values,
        n_atoms=2,
        initial_atoms=initial_atoms,
        maxiter=1,
        coding_maxiter=20,
        center_maxiter=10,
        tol=0.0,
    )
    assert dictionary.diagnostics["initial_indices"] is None
    assert dictionary.reason in {
        "maximum iterations reached",
        "no decreasing atom update found",
        "objective tolerance reached",
    }
    with pytest.raises(ValueError, match="nonnegative"):
        manifold_dictionary_learning(
            manifold, values, n_atoms=2, initial_atoms=initial_atoms, ridge=-1.0
        )
    with pytest.raises(ValueError, match="iteration limits"):
        manifold_dictionary_learning(
            manifold, values, n_atoms=2, initial_atoms=initial_atoms, maxiter=0
        )

    string_labels = jnp.array([0, -1, -1, 1])
    propagated = label_propagation(manifold, values, string_labels, maxiter=20)
    assert propagated.predictions.shape == (4,)
    with pytest.raises(ValueError, match="labels must have shape"):
        label_propagation(manifold, values, string_labels[:-1])
    with pytest.raises(ValueError, match="bandwidth"):
        label_propagation(manifold, values, string_labels, bandwidth=-1.0)
    with pytest.raises(ValueError, match="maxiter"):
        label_propagation(manifold, values, string_labels, maxiter=0)

    targets = jnp.array([0.0, 1.0, 2.0, 3.0])
    mask = jnp.array([True, False, True, False])
    regression = manifold_regularized_regression(
        manifold,
        values,
        targets,
        labeled_mask=mask,
    )
    assert bool(jnp.all(jnp.isfinite(regression.predictions)))
    with pytest.raises(ValueError, match="labeled_mask"):
        manifold_regularized_regression(
            manifold, values, targets, labeled_mask=mask[:-1]
        )
    with pytest.raises(ValueError, match="labeled targets"):
        manifold_regularized_regression(
            manifold,
            values,
            targets.at[0].set(jnp.nan),
            labeled_mask=mask,
        )


def test_operation_kind_handles_declared_unknown_and_attribute_fallbacks():
    class RaisesForUnknown:
        def operation_kind(self, operation):
            raise ValueError(operation)

    class ExactDistance:
        dist_is_exact = True

    assert operation_kind(RaisesForUnknown(), "dist") == "unknown"
    assert operation_kind(ExactDistance(), "dist") == "exact"
    assert operation_kind(object(), "dist") == "unknown"
