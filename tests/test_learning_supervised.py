from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import geojax.learning._classification as classification_module
import geojax.learning._response as response_module
from geojax.geometry import Euclidean, Product, Sphere, Torus
from geojax.learning import (
    geodesic_regression,
    knn_classifier,
    local_polynomial_regression,
    nearest_centroid_classifier,
    tangent_space_discriminant_analysis,
    tangent_space_logistic_regression,
)
from geojax.learning._features import fit_tangent_feature_map
from geojax.learning._results import FrechetMeanResult, GeodesicRegressionModel
from geojax.optimization import InfoEntry


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
    query = jnp.array([[0.25, 0.0]])
    probability = model.predict_proba(query)[0]
    batched_probability = model.predict_proba(
        jnp.concatenate([query, jnp.array([[1000.0, 1000.0]])])
    )[0]
    assert jnp.allclose(probability, batched_probability)

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
        tangent_space_discriminant_analysis(manifold, data, labels, regularization=0.0)
    with pytest.raises(ValueError, match="priors"):
        tangent_space_discriminant_analysis(
            manifold, data, labels, priors=jnp.array([jnp.nan, 1.0])
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
    assert center_model.converged == all(
        fit.converged for fit in center_model.diagnostics["class_fits"]
    )


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


def test_geodesic_regression_supports_product_responses_with_joint_fit():
    manifold = Product({"location": Euclidean(1), "phase": Torus(1)})
    predictors = jnp.linspace(-0.5, 0.5, 7)
    intercept = {"location": jnp.array([1.0]), "phase": jnp.array([0.2])}
    slope = {"location": jnp.array([2.0]), "phase": jnp.array([0.4])}
    responses = manifold.exp(
        intercept,
        jax.tree_util.tree_map(lambda leaf: predictors[:, None] * leaf, slope),
    )
    model = geodesic_regression(manifold, predictors, responses, maxiter=100, tol=1e-6)
    predictions = model.predict(predictors)
    assert jnp.max(manifold.dist(predictions, responses)) < 2e-3
    assert bool(jnp.all(manifold.is_tangent(model.intercept, model.slope)))


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
        local_polynomial_regression(manifold, jnp.arange(5.0), responses, bandwidth=1.0, degree=2)


@pytest.mark.parametrize(
    ("labels", "exception", "message"),
    [
        (np.array([0.0, 0.0, 0.0, 1.0, 1.0, np.nan]), ValueError, "NaN or infinite"),
        (
            np.array([0, 0, None, 1, 1, 1], dtype=object),
            ValueError,
            "missing values",
        ),
        (
            np.array([0, 0, "right", 1, 1, "right"], dtype=object),
            TypeError,
            "mutually comparable",
        ),
    ],
)
def test_classifiers_reject_nonfinite_missing_and_incomparable_labels(labels, exception, message):
    data, _ = separated_data()
    with pytest.raises(exception, match=message):
        nearest_centroid_classifier(Euclidean(2), data, labels)


class _NonfiniteLogEuclidean(Euclidean):
    def log(self, x, y):
        return jnp.full_like(super().log(x, y), jnp.nan)


class _NonfiniteInnerEuclidean(Euclidean):
    def inner(self, x, u, v):
        return jnp.full_like(super().inner(x, u, v), jnp.nan)


class _NegativeInnerEuclidean(Euclidean):
    def inner(self, x, u, v):
        return -super().inner(x, u, v)


class _NonfiniteExpEuclidean(Euclidean):
    def exp(self, x, u):
        return jnp.full_like(super().exp(x, u), jnp.nan)


@pytest.mark.parametrize(
    ("manifold", "exception", "message"),
    [
        (_NonfiniteLogEuclidean(2), ValueError, "undefined logarithm"),
        (_NonfiniteInnerEuclidean(2), FloatingPointError, "Gram matrix is nonfinite"),
        (_NegativeInnerEuclidean(2), FloatingPointError, "not positive semidefinite"),
    ],
)
def test_tangent_feature_fit_rejects_invalid_logs_and_metric_grams(manifold, exception, message):
    data, _ = separated_data()
    with pytest.raises(exception, match=message):
        fit_tangent_feature_map(
            manifold,
            data,
            base_point=jnp.zeros(2),
            n_components=1,
        )


@pytest.mark.parametrize(
    ("manifold", "exception", "message"),
    [
        (_NonfiniteLogEuclidean(2), ValueError, "undefined logarithm"),
        (_NonfiniteInnerEuclidean(2), FloatingPointError, "coordinates are nonfinite"),
    ],
)
def test_tangent_feature_transform_rejects_invalid_logs_and_coordinates(
    manifold, exception, message
):
    data, _ = separated_data()
    feature_map, _ = fit_tangent_feature_map(
        Euclidean(2), data, base_point=jnp.zeros(2), n_components=1
    )
    invalid = replace(feature_map, manifold=manifold)
    with pytest.raises(exception, match=message):
        invalid.transform(data)


def test_nearest_centroid_rejects_a_class_with_zero_weight_mass():
    data, labels = separated_data()
    with pytest.raises(ValueError, match="zero total mass"):
        nearest_centroid_classifier(
            Euclidean(2),
            data,
            labels,
            sample_weight=jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        )


@pytest.mark.parametrize(
    ("operation", "exception", "message"),
    [
        (
            lambda manifold, data, labels: knn_classifier(manifold, data, labels, n_neighbors=True),
            TypeError,
            "integer",
        ),
        (
            lambda manifold, data, labels: tangent_space_logistic_regression(
                manifold, data, labels, maxiter=0
            ),
            ValueError,
            "maxiter",
        ),
        (
            lambda manifold, data, labels: tangent_space_logistic_regression(
                manifold, data, labels, tol=-1.0
            ),
            ValueError,
            "tol",
        ),
        (
            lambda manifold, data, labels: tangent_space_logistic_regression(
                manifold, data, labels, learning_rate=0.0
            ),
            ValueError,
            "learning_rate",
        ),
        (
            lambda manifold, data, labels: tangent_space_discriminant_analysis(
                manifold, data, labels, priors=jnp.ones(3)
            ),
            ValueError,
            "priors",
        ),
        (
            lambda manifold, data, labels: tangent_space_discriminant_analysis(
                manifold, data, labels, priors=jnp.array([1.0, 0.0])
            ),
            ValueError,
            "priors",
        ),
        (
            lambda manifold, data, labels: tangent_space_discriminant_analysis(
                manifold,
                data,
                labels,
                priors=jnp.array([1.0 + 0.0j, 1.0 + 0.0j]),
            ),
            ValueError,
            "real-valued",
        ),
    ],
)
def test_classifier_alternate_control_validation(operation, exception, message):
    data, labels = separated_data()
    with pytest.raises(exception, match=message):
        operation(Euclidean(2), data, labels)


def test_discriminant_analysis_normalizes_valid_user_priors():
    data, labels = separated_data()
    model = tangent_space_discriminant_analysis(
        Euclidean(2), data, labels, priors=jnp.array([3.0, 1.0])
    )
    assert jnp.allclose(model.priors, jnp.array([0.75, 0.25]))


def test_classifier_status_reflects_incomplete_feature_and_centroid_fits(monkeypatch):
    manifold = Euclidean(2)
    data, labels = separated_data()
    feature_map, features = fit_tangent_feature_map(
        manifold, data, base_point=jnp.zeros(2), n_components=1
    )
    incomplete_map = replace(feature_map, converged=False)
    monkeypatch.setattr(
        classification_module,
        "fit_tangent_feature_map",
        lambda *args, **kwargs: (incomplete_map, features),
    )

    logistic = tangent_space_logistic_regression(manifold, data, labels, maxiter=1, tol=1e6)
    discriminant = tangent_space_discriminant_analysis(manifold, data, labels)
    assert not logistic.converged
    assert logistic.reason == "classifier converged after an incomplete tangent-reference fit"
    assert not discriminant.converged
    assert discriminant.reason == "closed-form fit used an incomplete tangent-reference fit"

    original_mean = classification_module.frechet_mean

    def incomplete_mean(*args, **kwargs):
        return replace(original_mean(*args, **kwargs), converged=False)

    monkeypatch.setattr(classification_module, "frechet_mean", incomplete_mean)
    centroid = nearest_centroid_classifier(manifold, data, labels)
    assert not centroid.converged
    assert centroid.reason == "at least one class centroid fit was incomplete"


def test_distance_weighted_knn_covers_exact_and_inverse_distance_votes():
    data, labels = separated_data()
    model = knn_classifier(Euclidean(2), data, labels, n_neighbors=3, weights="distance")
    exact = model.predict_proba(data[:1])
    remote = model.predict_proba(jnp.array([[10.0, 10.0]]))
    assert jnp.allclose(jnp.sum(exact, axis=1), 1.0)
    assert jnp.allclose(jnp.sum(remote, axis=1), 1.0)
    assert bool(jnp.all(jnp.isfinite(remote)))


class _StaticRegressionSolver:
    def __init__(self, *, value, gradnorm):
        self.value = value
        self.gradnorm = gradnorm

    def solve(self, problem):
        return (
            problem.x0,
            self.value,
            [
                InfoEntry(
                    iter=1,
                    cost=self.value,
                    gradnorm=self.gradnorm,
                    stepsize=0.0,
                    time=0.0,
                    reason="stub solver stopped",
                )
            ],
        )


def test_geodesic_regression_checks_solver_finiteness_and_resolution_reason():
    predictors = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float32)
    responses = predictors[:, None]
    manifold = Euclidean(1)

    with pytest.raises(FloatingPointError, match="nonfinite fit"):
        geodesic_regression(
            manifold,
            predictors,
            responses,
            initial_point=jnp.zeros(1, dtype=jnp.float32),
            solver=_StaticRegressionSolver(value=jnp.nan, gradnorm=1.0),
        )

    fitted = geodesic_regression(
        manifold,
        predictors,
        responses,
        initial_point=jnp.zeros(1, dtype=jnp.float32),
        solver=_StaticRegressionSolver(value=0.0, gradnorm=1e-5),
        tol=0.0,
    )
    assert fitted.converged
    assert fitted.reason == "gradient tolerance reached at floating-point resolution"


@pytest.mark.parametrize(
    ("predictors", "message"),
    [
        (jnp.arange(4.0), "shape"),
        (jnp.array([0.0, 1.0, 2.0, 3.0, jnp.nan]), "finite"),
        (jnp.arange(5.0, dtype=jnp.complex64), "real-valued"),
    ],
)
def test_geodesic_regression_rejects_malformed_predictor_vectors(predictors, message):
    with pytest.raises(ValueError, match=message):
        geodesic_regression(Euclidean(1), predictors, jnp.arange(5.0)[:, None])


@pytest.mark.parametrize(
    ("keyword", "value", "exception", "message"),
    [
        ("kernel", 3, TypeError, "callable"),
        ("maxiter", 0, ValueError, "maxiter"),
        ("tol", -1.0, ValueError, "tol"),
    ],
)
def test_local_polynomial_regression_validates_solver_and_kernel_controls(
    keyword, value, exception, message
):
    with pytest.raises(exception, match=message):
        local_polynomial_regression(
            Euclidean(1),
            jnp.arange(5.0),
            jnp.arange(5.0)[:, None],
            bandwidth=1.0,
            **{keyword: value},
        )


def test_geodesic_model_prediction_rejects_bad_inputs_and_nonfinite_exponentials():
    valid = GeodesicRegressionModel(
        manifold=Euclidean(1),
        intercept=jnp.zeros(1),
        slope=jnp.ones(1),
        predictor_mean=0.0,
        objective=0.0,
        iterations=1,
        converged=True,
        reason="test model",
    )
    for predictors in (
        jnp.ones((2, 2)),
        jnp.array([0.0, jnp.nan]),
    ):
        with pytest.raises(ValueError, match="finite scalar or one-dimensional"):
            valid.predict(predictors)
    with pytest.raises(ValueError, match="real-valued"):
        valid.predict(jnp.array([1.0 + 0.0j]))

    invalid = replace(valid, manifold=_NonfiniteExpEuclidean(1))
    with pytest.raises(FloatingPointError, match="exponential-map domain"):
        invalid.predict(jnp.array([0.0, 1.0]))


def _mean_result(point, *, converged):
    return FrechetMeanResult(
        point=jnp.asarray(point),
        objective=jnp.asarray(0.0),
        gradient_norm=jnp.asarray(0.0),
        iterations=1,
        converged=converged,
        reason="test mean",
    )


@pytest.mark.parametrize(
    ("statuses", "message"),
    [
        ((False,), "initialization mean did not converge"),
        ((True, False), "local Frechet-mean solve did not converge"),
    ],
)
def test_local_polynomial_model_reports_unconverged_mean_stages(monkeypatch, statuses, message):
    model = local_polynomial_regression(
        Euclidean(1),
        jnp.arange(3.0),
        jnp.arange(3.0)[:, None],
        bandwidth=1.0,
        degree=0,
    )
    results = iter(_mean_result([1.0], converged=status) for status in statuses)
    monkeypatch.setattr(response_module, "frechet_mean", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match=message):
        model.predict(1.0)


@pytest.mark.parametrize(
    ("value", "gradnorm", "exception", "message"),
    [
        (jnp.nan, 0.0, FloatingPointError, "nonfinite minimizer"),
        (0.0, 1.0, RuntimeError, "signed local-linear Frechet solve"),
    ],
)
def test_signed_local_linear_prediction_checks_terminal_solver_state(
    monkeypatch, value, gradnorm, exception, message
):
    model = local_polynomial_regression(
        Euclidean(1),
        jnp.arange(3.0),
        jnp.arange(3.0)[:, None],
        bandwidth=1.0,
        degree=1,
    )
    monkeypatch.setattr(
        response_module,
        "_local_weights",
        lambda model, query: (
            jnp.array([-1.0, 1.0, 1.0]),
            jnp.full((3,), 1.0 / 3.0),
        ),
    )
    monkeypatch.setattr(
        response_module,
        "frechet_mean",
        lambda *args, **kwargs: _mean_result([1.0], converged=True),
    )

    class StaticMinimize:
        def __init__(self, *, x0, **kwargs):
            self.x0 = x0

        def solve(self):
            return (
                self.x0,
                value,
                [
                    InfoEntry(
                        iter=1,
                        cost=value,
                        gradnorm=gradnorm,
                        stepsize=0.0,
                        time=0.0,
                    )
                ],
            )

    monkeypatch.setattr(response_module, "Minimize", StaticMinimize)
    with pytest.raises(exception, match=message):
        model.predict(1.0)
