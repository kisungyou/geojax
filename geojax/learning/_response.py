"""Regression methods with manifold-valued responses."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from geojax.geometry import Euclidean, Product
from geojax.optimization import AdaptiveArmijo, ConjugateGradient, Minimize

from ._capabilities import require_exact_operations
from ._data import as_manifold_data
from ._regression import _kernel_weights
from ._results import GeodesicRegressionModel, LocalPolynomialRegressionModel
from ._statistics import _gradient_tolerances, frechet_mean
from ._utils import (
    as_real_array,
    integer_control,
    nonnegative_control,
    normalize_weights,
    positive_control,
    require_unbatched,
    scale_tangent,
    stack_points,
    take_point,
    tree_all_finite,
    validate_manifold_point,
    validate_tangent_vector,
    weighted_tangent_sum,
)


def _validate_predictors(predictors: Any, n_samples: int | None = None) -> Any:
    values = as_real_array(predictors, name="predictors")
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
    coefficients = weights * centered_predictors / denominator
    return weighted_tangent_sum(manifold, logs, coefficients)


def _positive_predictor_variance(centered: Any, weights: Any) -> Any:
    variance = jnp.sum(weights * centered**2)
    dtype = jnp.result_type(centered, float)
    scale_squared = jnp.max(centered**2)
    resolution = 32.0 * jnp.finfo(dtype).eps * jnp.maximum(scale_squared, jnp.finfo(dtype).tiny)
    if not bool(jnp.isfinite(variance)) or float(variance) <= float(resolution):
        raise ValueError("predictors must have positive weighted variance.")
    return variance


def _joint_regression_geometry(manifold: Any) -> Product:
    if isinstance(manifold, Product):
        ambient = jax.tree_util.tree_map(
            lambda factor: Euclidean(size=factor.shape),
            manifold.factors,
        )
        factors = {"intercept": manifold.factors, "slope": ambient}
    else:
        factors = {
            "intercept": manifold,
            "slope": Euclidean(size=manifold.shape),
        }
    return Product(factors)


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
    r"""Fit ``Y(t) = Exp_p((t - t_bar) v)`` by joint intrinsic least squares.

    The intercept and an ambient parameterization of its tangent slope are
    optimized jointly. The slope is projected into ``T_p M`` inside the
    objective, so this minimizes the stated nonlinear residual rather than a
    flat-space profile approximation.
    """
    require_exact_operations(manifold, "geodesic_regression", "dist", "log", "exp")
    adapted = as_manifold_data(manifold, responses)
    require_unbatched(adapted, "geodesic_regression")
    predictor_values = _validate_predictors(predictors, adapted.n_samples)
    weights = normalize_weights(adapted.n_samples, sample_weight)
    predictor_mean = jnp.sum(weights * predictor_values)
    centered = predictor_values - predictor_mean
    _positive_predictor_variance(centered, weights)
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    if initial_point is None:
        mean_fit = frechet_mean(
            manifold,
            adapted,
            sample_weight=weights,
            maxiter=min(maxiter, 200),
            tol=tol,
        )
        x0 = mean_fit.point
    else:
        mean_fit = None
        initial_data = as_manifold_data(
            manifold,
            stack_points(manifold, [initial_point]),
        )
        x0 = take_point(manifold, initial_data.values, 0)
    initial_radius = jnp.sqrt(
        jnp.maximum(
            jnp.sum(weights * manifold.squared_dist(x0, adapted.values)),
            0.0,
        )
    )
    requested_gradient_tol, effective_tol = _gradient_tolerances(
        x0,
        initial_radius,
        tol,
    )

    initial_slope = _profile_slope(manifold, x0, adapted.values, centered, weights)
    joint_manifold = _joint_regression_geometry(manifold)
    initial_state = {"intercept": x0, "slope": initial_slope}

    def objective(state: Any) -> Any:
        point = state["intercept"]
        slope = manifold.tangent_project(point, state["slope"])
        predictions = manifold.exp(point, scale_tangent(manifold, slope, centered))
        return jnp.sum(weights * manifold.squared_dist(predictions, adapted.values))

    selected_solver = (
        solver
        if solver is not None
        else ConjugateGradient(
            maxiter=maxiter,
            tolgradnorm=effective_tol,
            verbosity=0,
            line_search=AdaptiveArmijo(normalize_step=False),
        )
    )
    state, value, history = Minimize(
        M=joint_manifold,
        cost=objective,
        x0=initial_state,
        solver=selected_solver,
    ).solve()
    point = state["intercept"]
    slope = manifold.tangent_project(point, state["slope"])
    point = validate_manifold_point(manifold, point, name="fitted intercept")
    slope = validate_tangent_vector(
        manifold,
        point,
        slope,
        name="fitted slope",
    )
    final = history[-1]
    if (
        not tree_all_finite(point)
        or not tree_all_finite(slope)
        or not bool(jnp.isfinite(value))
        or not bool(jnp.isfinite(final.gradnorm))
    ):
        raise FloatingPointError(
            "geodesic_regression produced a nonfinite fit; the observations may "
            "cross a cut locus or the selected solver may be unstable."
        )
    converged = bool(final.gradnorm <= effective_tol)
    return GeodesicRegressionModel(
        manifold=manifold,
        intercept=point,
        slope=slope,
        predictor_mean=predictor_mean,
        objective=jnp.asarray(value),
        iterations=int(final.iter),
        converged=converged,
        reason=(
            "gradient tolerance reached"
            if converged and final.gradnorm <= requested_gradient_tol
            else (
                "gradient tolerance reached at floating-point resolution"
                if converged
                else final.reason
            )
        ),
        diagnostics={
            "weights": weights,
            "history": tuple(history),
            "mean_fit": mean_fit,
            "joint_state": state,
            "requested_tolerance": tol,
            "requested_gradient_tolerance": requested_gradient_tol,
            "effective_tolerance": effective_tol,
            "initial_rms_radius": initial_radius,
        },
    )


def _predict_geodesic_regression(model: GeodesicRegressionModel, predictors: Any) -> Any:
    values = as_real_array(predictors, name="predictors")
    if values.ndim > 1 or not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("predictors must be a finite scalar or one-dimensional vector.")
    centered = values - model.predictor_mean
    predictions = model.manifold.exp(
        model.intercept,
        scale_tangent(model.manifold, model.slope, centered),
    )
    if not tree_all_finite(predictions):
        raise FloatingPointError("Geodesic predictions left the certified exponential-map domain.")
    return validate_manifold_point(
        model.manifold,
        predictions,
        name="geodesic predictions",
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
    bandwidth = positive_control(bandwidth, name="bandwidth")
    degree = integer_control(degree, name="degree")
    if degree not in {0, 1}:
        raise ValueError("degree must be 0 or 1.")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    if kernel is not None and not callable(kernel):
        raise TypeError("kernel must be callable or None.")
    return LocalPolynomialRegressionModel(
        manifold=manifold,
        predictors=predictor_values,
        training_data=adapted,
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
        maxiter=maxiter,
        tol=tol,
    )


def _local_weights(model: LocalPolynomialRegressionModel, query: Any) -> tuple[Any, Any]:
    offsets = model.predictors - query
    kernel_weights = _kernel_weights(jnp.abs(offsets), model.bandwidth, model.kernel)
    if float(jnp.sum(kernel_weights)) <= 0.0:
        nearest = jnp.argmin(jnp.abs(offsets))
        kernel_weights = jnp.zeros_like(offsets).at[nearest].set(1.0)
    positive = kernel_weights / jnp.sum(kernel_weights)
    if model.degree == 0:
        return positive, positive
    moment0 = jnp.sum(kernel_weights)
    moment1 = jnp.sum(kernel_weights * offsets)
    moment2 = jnp.sum(kernel_weights * offsets**2)
    denominator = moment0 * moment2 - moment1**2
    dtype = jnp.result_type(kernel_weights, float)
    determinant_scale = jnp.maximum(
        jnp.maximum(jnp.abs(moment0 * moment2), jnp.abs(moment1**2)),
        jnp.finfo(dtype).tiny,
    )
    if abs(float(denominator)) <= float(100.0 * jnp.finfo(dtype).eps * determinant_scale):
        return positive, positive
    local_linear = kernel_weights * (moment2 - offsets * moment1) / denominator
    return local_linear, positive


def _predict_one_local_polynomial(model: LocalPolynomialRegressionModel, query: Any) -> Any:
    weights, initialization_weights = _local_weights(model, query)
    initial_fit = frechet_mean(
        model.manifold,
        model.training_data,
        sample_weight=initialization_weights,
        maxiter=model.maxiter,
        tol=model.tol,
    )
    if not initial_fit.converged:
        raise RuntimeError(
            "The local-regression initialization mean did not converge; "
            "increase maxiter or bandwidth."
        )
    initial = initial_fit.point
    if model.degree == 0 or bool(jnp.all(weights >= 0.0)):
        fit = frechet_mean(
            model.manifold,
            model.training_data,
            sample_weight=weights,
            initial_point=initial,
            maxiter=model.maxiter,
            tol=model.tol,
        )
        if not fit.converged:
            raise RuntimeError(
                "The local Frechet-mean solve did not converge; increase maxiter or bandwidth."
            )
        return fit.point

    def objective(point: Any) -> Any:
        return jnp.sum(weights * model.manifold.squared_dist(point, model.training_data.values))

    initial_radius = jnp.sqrt(
        jnp.maximum(
            jnp.sum(
                initialization_weights
                * model.manifold.squared_dist(initial, model.training_data.values)
            ),
            0.0,
        )
    )
    _, effective_tol = _gradient_tolerances(
        initial,
        initial_radius,
        model.tol,
    )
    point, value, history = Minimize(
        M=model.manifold,
        cost=objective,
        x0=initial,
        solver=ConjugateGradient(
            maxiter=model.maxiter,
            tolgradnorm=effective_tol,
            verbosity=0,
        ),
    ).solve()
    final = history[-1]
    if (
        not tree_all_finite(point)
        or not bool(jnp.isfinite(value))
        or not bool(jnp.isfinite(final.gradnorm))
    ):
        raise FloatingPointError("Local-linear Frechet regression produced a nonfinite minimizer.")
    if final.gradnorm > effective_tol:
        raise RuntimeError(
            "The signed local-linear Frechet solve did not converge; its "
            "objective may be ill-conditioned or unbounded at this query."
        )
    return validate_manifold_point(
        model.manifold,
        point,
        name="local-polynomial prediction",
    )


def _predict_local_polynomial_regression(
    model: LocalPolynomialRegressionModel,
    predictors: Any,
) -> Any:
    values = as_real_array(predictors, name="predictors")
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
