from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import Euclidean, Product, Sphere, Torus
from geojax.learning import (
    geodesic_regression,
    knn_classifier,
    local_polynomial_regression,
    nearest_centroid_classifier,
    tangent_space_discriminant_analysis,
    tangent_space_logistic_regression,
)


def separated_data():
    values = jnp.array(
        [
            [-2.0, -0.2],
            [-1.7, 0.1],
            [-1.3, -0.1],
            [1.3, 0.1],
            [1.7, -0.1],
            [2.0, 0.2],
        ]
    )
    return values, jnp.array([0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize(
    "factory",
    [
        lambda manifold, data, labels: nearest_centroid_classifier(manifold, data, labels),
        lambda manifold, data, labels: knn_classifier(
            manifold, data, labels, n_neighbors=3, weights="distance"
        ),
        lambda manifold, data, labels: tangent_space_logistic_regression(
            manifold, data, labels, maxiter=200
        ),
        lambda manifold, data, labels: tangent_space_discriminant_analysis(
            manifold, data, labels, method="lda"
        ),
        lambda manifold, data, labels: tangent_space_discriminant_analysis(
            manifold, data, labels, method="qda"
        ),
    ],
)
def test_supervised_classifiers_recover_separated_euclidean_classes(factory):
    manifold = Euclidean(2)
    data, labels = separated_data()
    model = factory(manifold, data, labels)
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    assert np.array_equal(np.asarray(predictions), np.asarray(labels))
    assert probabilities.shape == (6, 2)
    assert jnp.allclose(jnp.sum(probabilities, axis=1), 1.0)


def test_classifiers_preserve_string_labels_and_validate_controls():
    manifold = Euclidean(2)
    data, _ = separated_data()
    labels = np.array(["left"] * 3 + ["right"] * 3)
    model = nearest_centroid_classifier(manifold, data, labels)
    assert np.array_equal(model.predict(data), labels)

    with pytest.raises(ValueError, match="at least two"):
        nearest_centroid_classifier(manifold, data, jnp.zeros((6,), dtype=int))
    with pytest.raises(ValueError, match="shape"):
        knn_classifier(manifold, data, labels[:-1])
    with pytest.raises(ValueError, match="between 1"):
        knn_classifier(manifold, data, labels, n_neighbors=0)
    with pytest.raises(ValueError, match="uniform"):
        knn_classifier(manifold, data, labels, weights="rank")
    with pytest.raises(ValueError, match="method"):
        tangent_space_discriminant_analysis(manifold, data, labels, method="rlDA")
    with pytest.raises(ValueError, match="positive"):
        tangent_space_discriminant_analysis(
            manifold, data, labels, regularization=0.0
        )


def test_tangent_classification_uses_metric_orthonormal_coordinates():
    manifold = Sphere(3)
    angles = jnp.array([-0.5, -0.3, -0.15, 0.15, 0.3, 0.5])
    data = jnp.stack([jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)], axis=1)
    labels = jnp.array([0, 0, 0, 1, 1, 1])
    model = tangent_space_logistic_regression(manifold, data, labels, maxiter=300)
    features = model.feature_map.transform(data)

    assert jnp.array_equal(model.predict(data), labels)
    assert features.shape[1] == 1
    assert model.feature_map.diagnostics["effective_rank_threshold"] >= 0.0
    for left, basis_left in enumerate(model.feature_map.basis):
        for right, basis_right in enumerate(model.feature_map.basis):
            expected = 1.0 if left == right else 0.0
            assert jnp.allclose(
                manifold.inner(model.feature_map.base_point, basis_left, basis_right),
                expected,
                atol=2e-4,
            )


def test_center_and_knn_classifiers_support_nested_product_data():
    manifold = Product({"direction": Sphere(3), "phase": [Torus(1)]})
    data = manifold.random_point(jax.random.key(301), sample_shape=(8,))
    labels = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    center_model = nearest_centroid_classifier(manifold, data, labels, maxiter=30)
    neighbor_model = knn_classifier(manifold, data, labels, n_neighbors=1)

    assert center_model.predict(data).shape == (8,)
    assert jnp.array_equal(neighbor_model.predict(data), labels)
    assert bool(jnp.all(manifold.belongs(center_model.centers)))


def test_geodesic_regression_matches_an_exact_euclidean_line():
    manifold = Euclidean(2)
    predictors = jnp.linspace(-1.0, 1.0, 9)
    responses = jnp.stack([1.0 + 2.0 * predictors, -0.5 + 0.75 * predictors], axis=1)
    model = geodesic_regression(manifold, predictors, responses, maxiter=100)
    queries = jnp.array([-0.75, 0.25, 1.25])
    expected = jnp.stack([1.0 + 2.0 * queries, -0.5 + 0.75 * queries], axis=1)

    assert jnp.allclose(model.predict(queries), expected, atol=2e-4)
    assert jnp.allclose(model.predict(0.25), expected[1], atol=2e-4)
    assert jnp.allclose(model.slope, jnp.array([2.0, 0.75]), atol=2e-4)


def test_geodesic_regression_recovers_a_short_spherical_geodesic():
    manifold = Sphere(3)
    intercept = jnp.array([1.0, 0.0, 0.0])
    slope = jnp.array([0.0, 0.45, 0.0])
    predictors = jnp.linspace(-0.8, 0.8, 7)
    responses = manifold.exp(intercept, predictors[:, None] * slope)
    model = geodesic_regression(manifold, predictors, responses, maxiter=100, tol=1e-6)
    predictions = model.predict(predictors)

    assert jnp.max(manifold.dist(predictions, responses)) < 2e-3
    assert bool(jnp.all(manifold.belongs(predictions)))


@pytest.mark.parametrize("degree", [0, 1])
def test_local_polynomial_frechet_regression_predicts_manifold_responses(degree):
    manifold = Euclidean(1)
    predictors = jnp.linspace(-1.0, 1.0, 11)
    responses = (1.0 + 1.5 * predictors)[:, None]
    model = local_polynomial_regression(
        manifold,
        predictors,
        responses,
        bandwidth=0.4,
        degree=degree,
        maxiter=50,
    )
    predictions = model.predict(jnp.array([-0.5, 0.0, 0.5]))[:, 0]
    expected = 1.0 + 1.5 * jnp.array([-0.5, 0.0, 0.5])
    tolerance = 0.18 if degree == 0 else 2e-3
    assert jnp.allclose(predictions, expected, atol=tolerance)


def test_response_regression_rejects_degenerate_designs_and_controls():
    manifold = Euclidean(1)
    responses = jnp.arange(5.0)[:, None]
    with pytest.raises(ValueError, match="positive weighted variance"):
        geodesic_regression(manifold, jnp.ones((5,)), responses)
    with pytest.raises(ValueError, match="one-dimensional"):
        geodesic_regression(manifold, jnp.ones((5, 1)), responses)
    with pytest.raises(ValueError, match="bandwidth"):
        local_polynomial_regression(manifold, jnp.arange(5.0), responses, bandwidth=0.0)
    with pytest.raises(ValueError, match="degree"):
        local_polynomial_regression(
            manifold, jnp.arange(5.0), responses, bandwidth=1.0, degree=2
        )
