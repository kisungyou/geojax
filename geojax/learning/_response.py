"""Regression methods with manifold-valued responses."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from geojax.optimization import ConjugateGradient, Minimize

from ._capabilities import require_exact_operations
from ._data import as_manifold_data
from ._regression import _kernel_weights
from ._results import GeodesicRegressionModel, LocalPolynomialRegressionModel
from ._statistics import frechet_mean
from ._utils import (
    normalize_weights,
    require_unbatched,
    scale_tangent,
    stack_points,
    weighted_tangent_sum,
)


def _validate_predictors(predictors: Any, n_samples: int | None = None) -> Any:
    values = jnp.asarray(predictors, dtype=float)
    if values.ndim != 1:
        raise ValueError("predictors must be a one-dimensional vector.")
    if n_samples is not None and values.shape != (n_samples,):
        raise ValueError(f"predictors must have shape ({n_samples},); received {values.shape}.")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("predictors must contain only finite values.")
    return values


def _profile_slope(
    manifold: Any,
    point: Any,
    values: Any,
    centered_predictors: Any,
    weights: Any,
) -> Any:
    logs = manifold.log(point, values)
    denominator = jnp.sum(weights * centered_predictors**2)
    coefficients = weights * centered_predictors / jnp.maximum(denominator, 1e-15)
    return weighted_tangent_sum(manifold, logs, coefficients)


def geodesic_regression(
    manifold: Any,
    predictors: Any,
    responses: Any,
    *,
    sample_weight: Any | None = None,
    initial_point: Any | None = None,
    solver: Any | None = None,
    maxiter: int = 200,
    tol: float = 1e-7,
) -> GeodesicRegressionModel:
    r"""Fit ``Y(t) = Exp_p((t - t_bar) v)`` by intrinsic least squares."""
    require_exact_operations(manifold, "geodesic_regression", "dist", "log", "exp")
    adapted = as_manifold_data(manifold, responses)
    require_unbatched(adapted, "geodesic_regression")
    predictor_values = _validate_predictors(predictors, adapted.n_samples)
    weights = normalize_weights(adapted.n_samples, sample_weight)
    predictor_mean = jnp.sum(weights * predictor_values)
    centered = predictor_values - predictor_mean
    if float(jnp.sum(weights * centered**2)) <= 1e-15:
        raise ValueError("predictors must have positive weighted variance.")
    if int(maxiter) < 1 or float(tol) < 0.0:
        raise ValueError("maxiter must be positive and tol must be nonnegative.")
    mean_fit = frechet_mean(
        manifold,
        adapted,
        sample_weight=weights,
        maxiter=min(int(maxiter), 200),
        tol=float(tol),
    )
    x0 = mean_fit.point if initial_point is None else manifold.project(initial_point)

    def objective(point: Any) -> Any:
        slope = _profile_slope(manifold, point, adapted.values, centered, weights)
        predictions = manifold.exp(point, scale_tangent(manifold, slope, centered))
        return jnp.sum(weights * manifold.squared_dist(predictions, adapted.values))

    selected_solver = solver or ConjugateGradient(
        maxiter=int(maxiter),
        tolgradnorm=float(tol),
        verbosity=0,
    )
    point, value, history = Minimize(
        M=manifold,
        cost=objective,
        x0=x0,
        solver=selected_solver,
    ).solve()
    slope = _profile_slope(manifold, point, adapted.values, centered, weights)
    final = history[-1]
    converged = bool(final.gradnorm <= float(tol))
    return GeodesicRegressionModel(
        manifold=manifold,
        intercept=point,
        slope=slope,
        predictor_mean=predictor_mean,
        objective=jnp.asarray(value),
        iterations=int(final.iter),
        converged=converged,
        reason="gradient tolerance reached" if converged else final.reason,
        diagnostics={"weights": weights, "history": tuple(history), "mean_fit": mean_fit},
    )


def _predict_geodesic_regression(model: GeodesicRegressionModel, predictors: Any) -> Any:
    values = jnp.asarray(predictors, dtype=float)
    if values.ndim > 1 or not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("predictors must be a finite scalar or one-dimensional vector.")
    centered = values - model.predictor_mean
    return model.manifold.exp(
        model.intercept,
        scale_tangent(model.manifold, model.slope, centered),
    )


def local_polynomial_regression(
    manifold: Any,
    predictors: Any,
    responses: Any,
    *,
    bandwidth: float,
    degree: int = 1,
    kernel: Callable[[Any, float], Any] | None = None,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> LocalPolynomialRegressionModel:
    """Fit local-constant or local-linear Fréchet regression."""
    require_exact_operations(manifold, "local_polynomial_regression", "dist", "log", "exp")
    adapted = as_manifold_data(manifold, responses)
    require_unbatched(adapted, "local_polynomial_regression")
    predictor_values = _validate_predictors(predictors, adapted.n_samples)
    if float(bandwidth) <= 0.0:
        raise ValueError("bandwidth must be positive.")
    if int(degree) not in {0, 1}:
        raise ValueError("degree must be 0 or 1.")
    if int(maxiter) < 1 or float(tol) < 0.0:
        raise ValueError("maxiter must be positive and tol must be nonnegative.")
    return LocalPolynomialRegressionModel(
        manifold=manifold,
        predictors=predictor_values,
        training_data=adapted,
        bandwidth=float(bandwidth),
        degree=int(degree),
        kernel=kernel,
        maxiter=int(maxiter),
        tol=float(tol),
    )


def _local_weights(model: LocalPolynomialRegressionModel, query: Any) -> tuple[Any, Any]:
    offsets = model.predictors - query
    kernel_weights = _kernel_weights(jnp.abs(offsets), model.bandwidth, model.kernel)
    if float(jnp.sum(kernel_weights)) <= 1e-15:
        nearest = jnp.argmin(jnp.abs(offsets))
        kernel_weights = jnp.zeros_like(offsets).at[nearest].set(1.0)
    positive = kernel_weights / jnp.sum(kernel_weights)
    if model.degree == 0:
        return positive, positive
    moment0 = jnp.sum(kernel_weights)
    moment1 = jnp.sum(kernel_weights * offsets)
    moment2 = jnp.sum(kernel_weights * offsets**2)
    denominator = moment0 * moment2 - moment1**2
    if abs(float(denominator)) <= 1e-12 * max(1.0, float(moment0 * moment2)):
        return positive, positive
    local_linear = kernel_weights * (moment2 - offsets * moment1) / denominator
    return local_linear, positive


def _predict_one_local_polynomial(model: LocalPolynomialRegressionModel, query: Any) -> Any:
    weights, initialization_weights = _local_weights(model, query)
    initial = frechet_mean(
        model.manifold,
        model.training_data,
        sample_weight=initialization_weights,
        maxiter=model.maxiter,
        tol=model.tol,
    ).point
    if model.degree == 0 or bool(jnp.all(weights >= 0.0)):
        return frechet_mean(
            model.manifold,
            model.training_data,
            sample_weight=weights,
            initial_point=initial,
            maxiter=model.maxiter,
            tol=model.tol,
        ).point

    def objective(point: Any) -> Any:
        return jnp.sum(weights * model.manifold.squared_dist(point, model.training_data.values))

    point, _, _ = Minimize(
        M=model.manifold,
        cost=objective,
        x0=initial,
        solver=ConjugateGradient(
            maxiter=model.maxiter,
            tolgradnorm=model.tol,
            verbosity=0,
        ),
    ).solve()
    return point


def _predict_local_polynomial_regression(
    model: LocalPolynomialRegressionModel,
    predictors: Any,
) -> Any:
    values = jnp.asarray(predictors, dtype=float)
    scalar = values.ndim == 0
    if values.ndim > 1 or not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("predictors must be a finite scalar or one-dimensional vector.")
    vector = values.reshape(-1)
    points = [_predict_one_local_polynomial(model, query) for query in vector]
    stacked = stack_points(model.manifold, points)
    if not scalar:
        return stacked
    return jax.tree_util.tree_map(lambda leaf: leaf[0], stacked)


__all__ = ["geodesic_regression", "local_polynomial_regression"]
