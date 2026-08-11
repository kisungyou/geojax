"""Supervised classifiers for manifold-valued predictors."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._capabilities import require_exact_operations
from ._data import as_manifold_data
from ._features import fit_tangent_feature_map
from ._geometry import pairwise_distances
from ._results import (
    KNearestNeighborsModel,
    NearestCentroidModel,
    TangentSpaceClassifierModel,
)
from ._statistics import frechet_mean
from ._utils import (
    as_real_array,
    integer_control,
    nonnegative_control,
    normalize_weights,
    positive_control,
    require_unbatched,
    stack_points,
    take_samples,
)


def _encode_labels(labels: Any, n_samples: int) -> tuple[Any, Any]:
    values = np.asarray(labels)
    if values.shape != (n_samples,):
        raise ValueError(f"labels must have shape ({n_samples},); received {values.shape}.")
    if values.dtype.kind in {"f", "c"} and not np.all(np.isfinite(values)):
        raise ValueError("labels must not contain NaN or infinite values.")
    if values.dtype.kind in {"O", "U", "S"}:
        missing = [
            value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
            for value in values.tolist()
        ]
        if any(missing):
            raise ValueError("labels must not contain missing values.")
    try:
        classes, encoded = np.unique(values, return_inverse=True)
    except TypeError as exc:
        raise TypeError("labels must be mutually comparable scalar values.") from exc
    if classes.size < 2:
        raise ValueError("classification requires at least two observed classes.")
    return classes, jnp.asarray(encoded, dtype=int)


def _decode_labels(classes: Any, encoded: Any) -> Any:
    decoded = np.asarray(classes)[np.asarray(encoded)]
    try:
        return jnp.asarray(decoded)
    except TypeError:
        return decoded


def nearest_centroid_classifier(
    manifold: Any,
    data: Any,
    labels: Any,
    *,
    sample_weight: Any | None = None,
    maxiter: int = 200,
    tol: float = 1e-7,
) -> NearestCentroidModel:
    """Fit one intrinsic Fréchet centroid per class."""
    require_exact_operations(manifold, "nearest_centroid_classifier", "dist", "log", "exp")
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "nearest_centroid_classifier")
    classes, encoded = _encode_labels(labels, adapted.n_samples)
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    centers = []
    fits = []
    for class_index in range(len(classes)):
        indices = jnp.flatnonzero(encoded == class_index)
        if float(jnp.sum(weights[indices])) <= 0.0:
            raise ValueError(
                f"sample_weight assigns zero total mass to class {classes[class_index]!r}."
            )
        fit = frechet_mean(
            manifold,
            take_samples(manifold, adapted.values, indices),
            sample_weight=weights[indices],
            maxiter=maxiter,
            tol=tol,
        )
        centers.append(fit.point)
        fits.append(fit)
    all_converged = all(fit.converged for fit in fits)
    return NearestCentroidModel(
        manifold=manifold,
        classes=classes,
        centers=stack_points(manifold, centers),
        converged=all_converged,
        reason=(
            "all class centroids converged"
            if all_converged
            else "at least one class centroid fit was incomplete"
        ),
        diagnostics={"class_fits": tuple(fits), "encoded_labels": encoded},
    )


def _nearest_centroid_distances(model: NearestCentroidModel, data: Any) -> Any:
    queries = as_manifold_data(model.manifold, data)
    require_unbatched(queries, "NearestCentroidModel.predict")
    centers = as_manifold_data(model.manifold, model.centers)
    return pairwise_distances(model.manifold, queries, centers, squared=True)


def _nearest_centroid_probabilities(model: NearestCentroidModel, data: Any) -> Any:
    distances = _nearest_centroid_distances(model, data)
    positive = jnp.where(distances > 0.0, distances, jnp.nan)
    scale = jnp.nanmedian(positive, axis=-1, keepdims=True)
    scale = jnp.where(jnp.isfinite(scale), scale, 1.0)
    scale = jnp.maximum(scale, jnp.finfo(distances.dtype).tiny)
    return jax.nn.softmax(-distances / scale, axis=-1)


def _predict_nearest_centroid(model: NearestCentroidModel, data: Any) -> Any:
    encoded = jnp.argmin(_nearest_centroid_distances(model, data), axis=-1)
    return _decode_labels(model.classes, encoded)


def knn_classifier(
    manifold: Any,
    data: Any,
    labels: Any,
    *,
    n_neighbors: int = 5,
    weights: str = "uniform",
) -> KNearestNeighborsModel:
    """Fit a geodesic-distance k-nearest-neighbors classifier."""
    require_exact_operations(manifold, "knn_classifier", "dist")
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "knn_classifier")
    classes, encoded = _encode_labels(labels, adapted.n_samples)
    n_neighbors = integer_control(n_neighbors, name="n_neighbors")
    if not 1 <= n_neighbors <= adapted.n_samples:
        raise ValueError("n_neighbors must be between 1 and n_samples.")
    if weights not in {"uniform", "distance"}:
        raise ValueError("weights must be 'uniform' or 'distance'.")
    return KNearestNeighborsModel(
        manifold=manifold,
        training_data=adapted,
        classes=classes,
        encoded_labels=encoded,
        n_neighbors=n_neighbors,
        weights=weights,
    )


def _knn_probabilities(model: KNearestNeighborsModel, data: Any) -> Any:
    queries = as_manifold_data(model.manifold, data)
    require_unbatched(queries, "KNearestNeighborsModel.predict")
    distances = pairwise_distances(model.manifold, queries, model.training_data)
    order = jnp.argsort(distances, axis=-1)[:, : model.n_neighbors]
    neighbor_distances = jnp.take_along_axis(distances, order, axis=-1)
    neighbor_labels = model.encoded_labels[order]
    if model.weights == "distance":
        exact = neighbor_distances == 0.0
        has_exact = jnp.any(exact, axis=-1, keepdims=True)
        vote_weights = jnp.where(
            has_exact,
            exact.astype(distances.dtype),
            1.0 / jnp.maximum(neighbor_distances, jnp.finfo(distances.dtype).tiny),
        )
    else:
        vote_weights = jnp.ones_like(neighbor_distances)
    votes = jax.nn.one_hot(neighbor_labels, len(model.classes)) * vote_weights[..., None]
    scores = jnp.sum(votes, axis=1)
    return scores / jnp.sum(scores, axis=-1, keepdims=True)


def _predict_knn(model: KNearestNeighborsModel, data: Any) -> Any:
    return _decode_labels(model.classes, jnp.argmax(_knn_probabilities(model, data), axis=-1))


def tangent_space_logistic_regression(
    manifold: Any,
    data: Any,
    labels: Any,
    *,
    base_point: Any | None = None,
    n_components: int | None = None,
    regularization: float = 1e-3,
    maxiter: int = 500,
    tol: float = 1e-7,
    learning_rate: float = 1.0,
) -> TangentSpaceClassifierModel:
    """Fit multinomial logistic regression in intrinsic tangent coordinates."""
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "tangent_space_logistic_regression")
    classes, encoded = _encode_labels(labels, adapted.n_samples)
    regularization = nonnegative_control(regularization, name="regularization")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    learning_rate = positive_control(learning_rate, name="learning_rate")
    feature_map, raw_features = fit_tangent_feature_map(
        manifold,
        adapted,
        base_point=base_point,
        n_components=n_components,
    )
    location = jnp.mean(raw_features, axis=0)
    raw_scale = jnp.std(raw_features, axis=0)
    scale = jnp.where(raw_scale > 0.0, raw_scale, 1.0)
    features = (raw_features - location) / scale
    n_classes = len(classes)
    coefficients = jnp.zeros((features.shape[1], n_classes))
    intercept = jnp.zeros((n_classes,))

    def loss(params: tuple[Any, Any]) -> Any:
        matrix, offset = params
        logits = features @ matrix + offset
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        likelihood = -jnp.mean(log_probabilities[jnp.arange(adapted.n_samples), encoded])
        return likelihood + 0.5 * regularization * jnp.sum(matrix * matrix)

    value_and_grad = jax.value_and_grad(loss)
    history = []
    converged = False
    reason = "maximum iterations reached"
    for iteration in range(1, maxiter + 1):
        value, (matrix_grad, intercept_grad) = value_and_grad((coefficients, intercept))
        gradient_norm = jnp.sqrt(jnp.sum(matrix_grad**2) + jnp.sum(intercept_grad**2))
        history.append(value)
        if not bool(jnp.isfinite(value)) or not bool(jnp.isfinite(gradient_norm)):
            raise FloatingPointError(
                "Logistic regression produced a nonfinite objective or gradient."
            )
        if float(gradient_norm) <= tol:
            converged = True
            reason = "gradient tolerance reached"
            break
        step = learning_rate
        accepted = False
        directional = jnp.sum(matrix_grad**2) + jnp.sum(intercept_grad**2)
        minimum_step = float(jnp.finfo(features.dtype).eps) * learning_rate
        while step >= minimum_step:
            candidate_matrix = coefficients - step * matrix_grad
            candidate_intercept = intercept - step * intercept_grad
            candidate = loss((candidate_matrix, candidate_intercept))
            if float(candidate) <= float(value - 1e-4 * step * directional):
                coefficients, intercept = candidate_matrix, candidate_intercept
                accepted = True
                break
            step *= 0.5
        if not accepted:
            reason = "line search failed"
            break
    final_objective = loss((coefficients, intercept))
    (_, (matrix_grad, intercept_grad)) = value_and_grad((coefficients, intercept))
    final_gradient_norm = jnp.sqrt(jnp.sum(matrix_grad**2) + jnp.sum(intercept_grad**2))
    if not bool(jnp.isfinite(final_objective)) or not bool(jnp.isfinite(final_gradient_norm)):
        raise FloatingPointError("Logistic regression produced a nonfinite final fit.")
    if float(final_gradient_norm) <= tol:
        converged = True
        reason = "gradient tolerance reached"
    model_converged = converged and feature_map.converged
    if converged and not feature_map.converged:
        reason = "classifier converged after an incomplete tangent-reference fit"
    return TangentSpaceClassifierModel(
        manifold=manifold,
        classes=classes,
        feature_map=feature_map,
        method="logistic",
        coefficients=coefficients,
        intercept=intercept,
        location=location,
        scale=scale,
        objective=final_objective,
        iterations=iteration,
        converged=model_converged,
        reason=reason,
        diagnostics={
            "objective_history": jnp.asarray(history),
            "regularization": regularization,
            "encoded_labels": encoded,
            "gradient_norm": final_gradient_norm,
        },
    )


def tangent_space_discriminant_analysis(
    manifold: Any,
    data: Any,
    labels: Any,
    *,
    method: str = "lda",
    base_point: Any | None = None,
    n_components: int | None = None,
    regularization: float = 1e-4,
    priors: Any | None = None,
) -> TangentSpaceClassifierModel:
    """Fit LDA or QDA in intrinsic metric-orthonormal tangent coordinates."""
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "tangent_space_discriminant_analysis")
    classes, encoded = _encode_labels(labels, adapted.n_samples)
    if not isinstance(method, str) or method not in {"lda", "qda"}:
        raise ValueError("method must be 'lda' or 'qda'.")
    regularization = positive_control(regularization, name="regularization")
    feature_map, features = fit_tangent_feature_map(
        manifold,
        adapted,
        base_point=base_point,
        n_components=n_components,
    )
    n_classes = len(classes)
    counts = jnp.asarray([jnp.sum(encoded == index) for index in range(n_classes)])
    if priors is None:
        prior_values = counts / adapted.n_samples
    else:
        prior_values = as_real_array(priors, name="priors")
        if (
            prior_values.shape != (n_classes,)
            or not bool(jnp.all(jnp.isfinite(prior_values)))
            or bool(jnp.any(prior_values <= 0.0))
        ):
            raise ValueError("priors must contain one positive value per class.")
        prior_values = prior_values / jnp.max(prior_values)
        prior_values = prior_values / jnp.sum(prior_values)
    means = jnp.stack([jnp.mean(features[encoded == index], axis=0) for index in range(n_classes)])
    identity = jnp.eye(features.shape[1])
    class_covariances = []
    for index in range(n_classes):
        centered = features[encoded == index] - means[index]
        denominator = max(int(counts[index]) - 1, 1)
        covariance = centered.T @ centered / denominator
        class_covariances.append(covariance + regularization * identity)
    if method == "lda":
        scatter = sum(
            (max(int(counts[index]) - 1, 0))
            * (class_covariances[index] - regularization * identity)
            for index in range(n_classes)
        )
        denominator = max(adapted.n_samples - n_classes, 1)
        covariances = (scatter / denominator + regularization * identity)[None, ...]
    else:
        covariances = jnp.stack(class_covariances)
    return TangentSpaceClassifierModel(
        manifold=manifold,
        classes=classes,
        feature_map=feature_map,
        method=method,
        class_means=means,
        covariances=covariances,
        priors=prior_values,
        objective=jnp.asarray(0.0),
        converged=feature_map.converged,
        reason=(
            "closed-form discriminant fit completed"
            if feature_map.converged
            else "closed-form fit used an incomplete tangent-reference fit"
        ),
        diagnostics={"regularization": regularization, "encoded_labels": encoded},
    )


def _tangent_classifier_scores(model: TangentSpaceClassifierModel, data: Any) -> Any:
    features = model.feature_map.transform(data)
    if model.method == "logistic":
        standardized = (features - model.location) / model.scale
        return standardized @ model.coefficients + model.intercept
    if features.shape[1] == 0:
        return jnp.broadcast_to(jnp.log(model.priors), (features.shape[0], len(model.classes)))
    scores = []
    for index in range(len(model.classes)):
        covariance = model.covariances[0] if model.method == "lda" else model.covariances[index]
        difference = features - model.class_means[index]
        sign, logdet = jnp.linalg.slogdet(covariance)
        if not bool(sign > 0.0) or not bool(jnp.isfinite(logdet)):
            raise FloatingPointError("discriminant covariance is not positive definite.")
        inverse_difference = jnp.linalg.solve(covariance, difference.T).T
        quadratic = jnp.sum(difference * inverse_difference, axis=-1)
        scores.append(-0.5 * quadratic - 0.5 * logdet + jnp.log(model.priors[index]))
    return jnp.stack(scores, axis=-1)


def _tangent_classifier_probabilities(model: TangentSpaceClassifierModel, data: Any) -> Any:
    return jax.nn.softmax(_tangent_classifier_scores(model, data), axis=-1)


def _predict_tangent_classifier(model: TangentSpaceClassifierModel, data: Any) -> Any:
    encoded = jnp.argmax(_tangent_classifier_scores(model, data), axis=-1)
    return _decode_labels(model.classes, encoded)


__all__ = [
    "knn_classifier",
    "nearest_centroid_classifier",
    "tangent_space_discriminant_analysis",
    "tangent_space_logistic_regression",
]
