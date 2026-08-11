"""Robust intrinsic summaries, depths, and distance ranks."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._results import MetricRanksResult, RobustLocationResult
from ._statistics import (
    _gradient_tolerances,
    _validated_point,
    frechet_mean,
    frechet_median,
)
from ._utils import (
    integer_control,
    interval_control,
    nonnegative_control,
    normalize_weights,
    positive_control,
    require_unbatched,
    scale_tangent_samples,
    take_point,
    take_samples,
    tree_all_finite,
    weighted_tangent_sum,
)


def _prepare(manifold: Any, data: Any, method: str) -> ManifoldData:
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, method)
    return adapted


def trimmed_frechet_mean(
    manifold: Any,
    data: Any,
    *,
    trim_fraction: float = 0.1,
    sample_weight: Any | None = None,
    initial_point: Any | None = None,
    maxiter: int = 100,
    center_maxiter: int = 100,
    tol: float = 1e-6,
) -> RobustLocationResult:
    """Compute a least-trimmed-squares intrinsic location estimate."""
    require_exact_operations(manifold, "trimmed_frechet_mean", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "trimmed_frechet_mean")
    trim_fraction = interval_control(
        trim_fraction,
        name="trim_fraction",
        lower=0.0,
        upper=1.0,
        upper_closed=False,
    )
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    center_maxiter = integer_control(center_maxiter, name="center_maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    target_mass = 1.0 - trim_fraction

    def retained_measure(distances_sq: Any) -> tuple[Any, Any]:
        order = jnp.argsort(distances_sq)
        ordered_weights = weights[order]
        mass_before = jnp.cumsum(ordered_weights) - ordered_weights
        allocated = jnp.clip(target_mass - mass_before, 0.0, ordered_weights)
        allocation_tolerance = 64.0 * jnp.finfo(allocated.dtype).eps * target_mass
        retained_mask = allocated > allocation_tolerance
        retained_indices = order[retained_mask]
        retained_weights = allocated[retained_mask]
        retained_weights = retained_weights / jnp.sum(retained_weights)
        return retained_indices, retained_weights

    initial_fit = None
    if initial_point is None:
        initial_fit = frechet_mean(
            manifold,
            adapted,
            sample_weight=weights,
            maxiter=center_maxiter,
            tol=tol,
        )
        point = initial_fit.point
    else:
        point = _validated_point(manifold, initial_point, name="initial_point")
    initial_radius = jnp.sqrt(
        jnp.maximum(
            jnp.sum(weights * manifold.squared_dist(point, adapted.values)),
            0.0,
        )
    )
    requested_tol, effective_tol = _gradient_tolerances(
        point,
        initial_radius,
        tol,
    )
    inner_fits = []
    inner_converged = initial_fit is None or initial_fit.converged
    retained_history = []
    objective_history = []
    movement_stopped = False
    retained = jnp.arange(adapted.n_samples)
    retained_weights = weights
    for iteration in range(1, maxiter + 1):
        distances_sq = manifold.squared_dist(point, adapted.values)
        retained, retained_weights = retained_measure(distances_sq)
        retained_history.append(retained)
        objective = jnp.sum(retained_weights * distances_sq[retained])
        objective_history.append(objective)
        subset = as_manifold_data(manifold, take_samples(manifold, adapted.values, retained))
        candidate_fit = frechet_mean(
            manifold,
            subset,
            sample_weight=retained_weights,
            initial_point=point,
            maxiter=center_maxiter,
            tol=tol,
        )
        inner_fits.append(candidate_fit)
        inner_converged = inner_converged and candidate_fit.converged
        candidate = candidate_fit.point
        movement = manifold.dist(point, candidate)
        point = candidate
        if float(movement) <= float(effective_tol):
            movement_stopped = True
            break
    final_distances = manifold.dist(point, adapted.values)
    retained, retained_weights = retained_measure(final_distances**2)
    gradient = weighted_tangent_sum(
        manifold,
        manifold.log(point, take_samples(manifold, adapted.values, retained)),
        retained_weights,
    )
    if not tree_all_finite(gradient):
        raise FloatingPointError("trimmed_frechet_mean produced a nonfinite stationarity vector.")
    gradient_norm = 2.0 * manifold.norm(point, gradient)
    converged = bool(movement_stopped and inner_converged and gradient_norm <= effective_tol)
    if converged:
        reason = (
            "location tolerance reached"
            if gradient_norm <= requested_tol
            else "location tolerance reached at floating-point resolution"
        )
    elif not inner_converged:
        reason = "an inner Frechet-mean solve was incomplete"
    elif movement_stopped:
        reason = "updates stopped before stationarity was reached"
    else:
        reason = "maximum iterations reached"
    return RobustLocationResult(
        point=point,
        objective=jnp.sum(retained_weights * final_distances[retained] ** 2),
        gradient_norm=gradient_norm,
        iterations=iteration,
        converged=converged,
        reason=reason,
        diagnostics={
            "retained_indices": retained,
            "retained_history": tuple(retained_history),
            "objective_history": jnp.asarray(objective_history),
            "distances": final_distances,
            "weights": weights,
            "retained_weights": retained_weights,
            "trim_fraction": trim_fraction,
            "initial_fit": initial_fit,
            "inner_fits": tuple(inner_fits),
            "requested_tolerance": tol,
            "requested_stationarity_tolerance": requested_tol,
            "effective_stationarity_tolerance": effective_tol,
            "initial_rms_radius": initial_radius,
        },
    )


def _robust_weights_and_loss(residuals: Any, scale: float, loss: str) -> tuple[Any, Any]:
    normalized = residuals / float(scale)
    absolute = jnp.abs(normalized)
    tiny = jnp.finfo(normalized.dtype).tiny
    if loss == "huber":
        tuning = 1.345
        weights = jnp.where(
            absolute <= tuning,
            1.0,
            tuning / jnp.maximum(absolute, tiny),
        )
        rho = jnp.where(
            absolute <= tuning,
            0.5 * normalized**2,
            tuning * absolute - 0.5 * tuning**2,
        )
    elif loss == "cauchy":
        tuning = 2.385
        ratio = normalized / tuning
        weights = 1.0 / (1.0 + ratio**2)
        rho = 0.5 * tuning**2 * jnp.log1p(ratio**2)
    elif loss == "tukey":
        tuning = 4.685
        ratio = normalized / tuning
        inside = absolute < tuning
        weights = jnp.where(inside, (1.0 - ratio**2) ** 2, 0.0)
        rho = jnp.where(
            inside,
            tuning**2 / 6.0 * (1.0 - (1.0 - ratio**2) ** 3),
            tuning**2 / 6.0,
        )
    else:
        raise ValueError("loss must be 'huber', 'cauchy', or 'tukey'.")
    return weights, rho * scale**2


def geodesic_m_estimator(
    manifold: Any,
    data: Any,
    *,
    loss: str = "huber",
    scale: float | None = None,
    sample_weight: Any | None = None,
    initial_point: Any | None = None,
    maxiter: int = 100,
    center_maxiter: int = 100,
    tol: float = 1e-6,
) -> RobustLocationResult:
    """Compute a geodesic M-location by iteratively reweighted Fréchet means."""
    require_exact_operations(manifold, "geodesic_m_estimator", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "geodesic_m_estimator")
    if not isinstance(loss, str) or loss not in {"huber", "cauchy", "tukey"}:
        raise ValueError("loss must be 'huber', 'cauchy', or 'tukey'.")
    if scale is not None:
        scale = positive_control(scale, name="scale")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    center_maxiter = integer_control(center_maxiter, name="center_maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    base_weights = normalize_weights(adapted.n_samples, sample_weight)
    initial_fit = None
    if initial_point is None:
        initial_fit = frechet_median(
            manifold,
            adapted,
            sample_weight=base_weights,
            maxiter=center_maxiter,
            tol=tol,
        )
        point = initial_fit.point
    else:
        point = _validated_point(manifold, initial_point, name="initial_point")
    initial_residuals = manifold.dist(point, adapted.values)
    initial_radius = jnp.sqrt(jnp.sum(base_weights * initial_residuals**2))
    requested_tol, effective_tol = _gradient_tolerances(
        point,
        initial_radius,
        tol,
    )
    if scale is None:
        residual_dtype = jnp.asarray(initial_residuals).dtype
        residual_maximum = float(jnp.max(initial_residuals))
        numerical_floor = max(
            float(jnp.sqrt(jnp.finfo(residual_dtype).eps)) * residual_maximum,
            float(jnp.finfo(residual_dtype).tiny),
        )
        selected_scale = max(
            float(jnp.median(initial_residuals)) / 0.67448975,
            numerical_floor,
        )
    else:
        selected_scale = scale
    objective_history = []
    weight_history = []
    movement_stopped = False
    inner_fits = []
    inner_converged = initial_fit is None or initial_fit.converged
    effective_weights = base_weights
    for iteration in range(1, maxiter + 1):
        residuals = manifold.dist(point, adapted.values)
        robust_weights, losses = _robust_weights_and_loss(residuals, selected_scale, loss)
        effective_weights = base_weights * robust_weights
        if float(jnp.sum(effective_weights)) <= float(jnp.finfo(effective_weights.dtype).tiny):
            effective_weights = base_weights
        effective_weights = effective_weights / jnp.sum(effective_weights)
        objective_history.append(jnp.sum(base_weights * losses))
        weight_history.append(effective_weights)
        candidate_fit = frechet_mean(
            manifold,
            adapted,
            sample_weight=effective_weights,
            initial_point=point,
            maxiter=center_maxiter,
            tol=tol,
        )
        inner_fits.append(candidate_fit)
        inner_converged = inner_converged and candidate_fit.converged
        candidate = candidate_fit.point
        movement = manifold.dist(point, candidate)
        point = candidate
        if float(movement) <= float(effective_tol):
            movement_stopped = True
            break
    residuals = manifold.dist(point, adapted.values)
    robust_weights, losses = _robust_weights_and_loss(residuals, selected_scale, loss)
    unnormalized_effective_weights = base_weights * robust_weights
    effective_mass = jnp.sum(unnormalized_effective_weights)
    mass_floor = jnp.finfo(unnormalized_effective_weights.dtype).tiny
    used_weight_fallback = float(effective_mass) <= float(mass_floor)
    effective_weights = jnp.where(
        used_weight_fallback,
        base_weights,
        unnormalized_effective_weights / jnp.maximum(effective_mass, mass_floor),
    )
    gradient = weighted_tangent_sum(
        manifold,
        manifold.log(point, adapted.values),
        unnormalized_effective_weights,
    )
    gradient_norm = manifold.norm(point, gradient)
    converged = bool(movement_stopped and inner_converged and gradient_norm <= effective_tol)
    if converged:
        reason = (
            "location tolerance reached"
            if gradient_norm <= requested_tol
            else "location tolerance reached at floating-point resolution"
        )
    elif not inner_converged:
        reason = "an inner Frechet solve was incomplete"
    elif movement_stopped:
        reason = "updates stopped before stationarity was reached"
    else:
        reason = "maximum iterations reached"
    return RobustLocationResult(
        point=point,
        objective=jnp.sum(base_weights * losses),
        gradient_norm=gradient_norm,
        iterations=iteration,
        converged=converged,
        reason=reason,
        diagnostics={
            "loss": loss,
            "scale": selected_scale,
            "effective_weights": effective_weights,
            "unnormalized_effective_weights": unnormalized_effective_weights,
            "used_weight_fallback": used_weight_fallback,
            "weight_history": tuple(weight_history),
            "objective_history": jnp.asarray(objective_history),
            "residuals": residuals,
            "initial_fit": initial_fit,
            "inner_fits": tuple(inner_fits),
            "requested_tolerance": tol,
            "requested_stationarity_tolerance": requested_tol,
            "effective_stationarity_tolerance": effective_tol,
            "initial_rms_radius": initial_radius,
        },
    )


def geodesic_spatial_depth(
    manifold: Any,
    points: Any,
    reference_data: Any,
    *,
    sample_weight: Any | None = None,
) -> Any:
    """Evaluate intrinsic spatial depth relative to a reference sample."""
    require_exact_operations(manifold, "geodesic_spatial_depth", "dist", "log")
    queries = _prepare(manifold, points, "geodesic_spatial_depth")
    reference = _prepare(manifold, reference_data, "geodesic_spatial_depth")
    weights = normalize_weights(reference.n_samples, sample_weight)
    depths = []
    for index in range(queries.n_samples):
        point = take_point(manifold, queries.values, index)
        distances = manifold.dist(point, reference.values)
        logs = manifold.log(point, reference.values)
        if not tree_all_finite(logs):
            raise FloatingPointError("geodesic_spatial_depth encountered an undefined logarithm.")
        inverse = jnp.where(
            distances > 0.0,
            1.0 / jnp.maximum(distances, jnp.finfo(distances.dtype).tiny),
            0.0,
        )
        unit_logs = scale_tangent_samples(manifold, logs, inverse)
        average_sign = weighted_tangent_sum(manifold, unit_logs, weights)
        raw_depth = 1.0 - manifold.norm(point, average_sign)
        bound_tolerance = (
            100.0 * float(jnp.finfo(jnp.asarray(raw_depth).dtype).eps) * reference.n_samples
        )
        if not bool(jnp.isfinite(raw_depth)) or float(raw_depth) < -bound_tolerance:
            raise FloatingPointError("geodesic_spatial_depth violated its metric norm bound.")
        depths.append(jnp.clip(raw_depth, 0.0, 1.0))
    return jnp.asarray(depths)


def metric_distance_ranks(
    manifold: Any,
    data: Any,
    *,
    center: Any | None = None,
    sample_weight: Any | None = None,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> MetricRanksResult:
    """Rank observations by geodesic distance from an intrinsic median."""
    require_exact_operations(manifold, "metric_distance_ranks", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "metric_distance_ranks")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    center_fit = None
    if center is None:
        center_fit = frechet_median(
            manifold,
            adapted,
            sample_weight=sample_weight,
            maxiter=maxiter,
            tol=tol,
        )
        if not center_fit.converged:
            raise RuntimeError(
                "The intrinsic-median center did not converge; distance ranks "
                "would not use the requested center."
            )
        location = center_fit.point
    else:
        location = _validated_point(manifold, center, name="center")
    scores = manifold.dist(location, adapted.values)
    if not bool(jnp.all(jnp.isfinite(scores))):
        raise FloatingPointError("metric_distance_ranks produced nonfinite distances.")
    order = jnp.argsort(scores, stable=True)
    lower = jnp.sum(scores[None, :] < scores[:, None], axis=1)
    equal = jnp.sum(scores[None, :] == scores[:, None], axis=1)
    ranks = (lower + 0.5 * (equal + 1.0)) / adapted.n_samples
    return MetricRanksResult(
        ranks=ranks,
        scores=scores,
        center=location,
        diagnostics={"order": order, "center_fit": center_fit},
    )


__all__ = [
    "geodesic_m_estimator",
    "geodesic_spatial_depth",
    "metric_distance_ranks",
    "trimmed_frechet_mean",
]
