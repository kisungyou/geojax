"""Robust intrinsic summaries, depths, and distance ranks."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._results import MetricRanksResult, RobustLocationResult
from ._statistics import frechet_mean, frechet_median
from ._utils import (
    normalize_weights,
    require_unbatched,
    scale_tangent_samples,
    take_point,
    take_samples,
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
    if not 0.0 <= float(trim_fraction) < 1.0:
        raise ValueError("trim_fraction must lie in [0, 1).")
    if int(maxiter) < 1 or int(center_maxiter) < 1 or tol < 0.0:
        raise ValueError("iteration limits must be positive and tol nonnegative.")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    target_mass = 1.0 - float(trim_fraction)

    def retained_measure(distances_sq: Any) -> tuple[Any, Any]:
        order = jnp.argsort(distances_sq)
        ordered_weights = weights[order]
        mass_before = jnp.cumsum(ordered_weights) - ordered_weights
        allocated = jnp.clip(target_mass - mass_before, 0.0, ordered_weights)
        allocation_tolerance = 64.0 * jnp.finfo(allocated.dtype).eps * max(
            target_mass, 1.0
        )
        retained_mask = allocated > allocation_tolerance
        retained_indices = order[retained_mask]
        retained_weights = allocated[retained_mask]
        retained_weights = retained_weights / jnp.sum(retained_weights)
        return retained_indices, retained_weights

    point = (
        frechet_mean(
            manifold,
            adapted,
            sample_weight=weights,
            maxiter=int(center_maxiter),
            tol=float(tol),
        ).point
        if initial_point is None
        else manifold.project(initial_point)
    )
    retained_history = []
    objective_history = []
    converged = False
    retained = jnp.arange(adapted.n_samples)
    retained_weights = weights
    for iteration in range(1, int(maxiter) + 1):
        distances_sq = manifold.squared_dist(point, adapted.values)
        retained, retained_weights = retained_measure(distances_sq)
        retained_history.append(retained)
        objective = jnp.sum(retained_weights * distances_sq[retained])
        objective_history.append(objective)
        subset = as_manifold_data(manifold, take_samples(manifold, adapted.values, retained))
        candidate = frechet_mean(
            manifold,
            subset,
            sample_weight=retained_weights,
            initial_point=point,
            maxiter=int(center_maxiter),
            tol=float(tol),
        ).point
        movement = manifold.dist(point, candidate)
        point = candidate
        if float(movement) <= float(tol):
            converged = True
            break
    final_distances = manifold.dist(point, adapted.values)
    retained, retained_weights = retained_measure(final_distances**2)
    gradient = weighted_tangent_sum(
        manifold,
        manifold.log(point, take_samples(manifold, adapted.values, retained)),
        retained_weights,
    )
    return RobustLocationResult(
        point=point,
        objective=jnp.sum(retained_weights * final_distances[retained] ** 2),
        gradient_norm=manifold.norm(point, gradient),
        iterations=iteration,
        converged=converged,
        reason="location tolerance reached" if converged else "maximum iterations reached",
        diagnostics={
            "retained_indices": retained,
            "retained_history": tuple(retained_history),
            "objective_history": jnp.asarray(objective_history),
            "distances": final_distances,
            "weights": weights,
            "retained_weights": retained_weights,
            "trim_fraction": float(trim_fraction),
        },
    )


def _robust_weights_and_loss(residuals: Any, scale: float, loss: str) -> tuple[Any, Any]:
    normalized = residuals / max(float(scale), 1e-15)
    absolute = jnp.abs(normalized)
    if loss == "huber":
        tuning = 1.345
        weights = jnp.where(absolute <= tuning, 1.0, tuning / jnp.maximum(absolute, 1e-15))
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
    if loss not in {"huber", "cauchy", "tukey"}:
        raise ValueError("loss must be 'huber', 'cauchy', or 'tukey'.")
    if scale is not None and float(scale) <= 0.0:
        raise ValueError("scale must be positive when supplied.")
    if int(maxiter) < 1 or int(center_maxiter) < 1 or tol < 0.0:
        raise ValueError("iteration limits must be positive and tol nonnegative.")
    base_weights = normalize_weights(adapted.n_samples, sample_weight)
    point = (
        frechet_median(
            manifold,
            adapted,
            sample_weight=base_weights,
            maxiter=int(center_maxiter),
            tol=float(tol),
        ).point
        if initial_point is None
        else manifold.project(initial_point)
    )
    initial_residuals = manifold.dist(point, adapted.values)
    selected_scale = (
        max(float(jnp.median(initial_residuals)) / 0.67448975, 1e-8)
        if scale is None
        else float(scale)
    )
    objective_history = []
    weight_history = []
    converged = False
    effective_weights = base_weights
    for iteration in range(1, int(maxiter) + 1):
        residuals = manifold.dist(point, adapted.values)
        robust_weights, losses = _robust_weights_and_loss(residuals, selected_scale, loss)
        effective_weights = base_weights * robust_weights
        if float(jnp.sum(effective_weights)) <= 1e-15:
            effective_weights = base_weights
        effective_weights = effective_weights / jnp.sum(effective_weights)
        objective_history.append(jnp.sum(base_weights * losses))
        weight_history.append(effective_weights)
        candidate = frechet_mean(
            manifold,
            adapted,
            sample_weight=effective_weights,
            initial_point=point,
            maxiter=int(center_maxiter),
            tol=float(tol),
        ).point
        movement = manifold.dist(point, candidate)
        point = candidate
        if float(movement) <= float(tol):
            converged = True
            break
    residuals = manifold.dist(point, adapted.values)
    _, losses = _robust_weights_and_loss(residuals, selected_scale, loss)
    gradient = weighted_tangent_sum(
        manifold,
        manifold.log(point, adapted.values),
        effective_weights,
    )
    return RobustLocationResult(
        point=point,
        objective=jnp.sum(base_weights * losses),
        gradient_norm=manifold.norm(point, gradient),
        iterations=iteration,
        converged=converged,
        reason="location tolerance reached" if converged else "maximum iterations reached",
        diagnostics={
            "loss": loss,
            "scale": selected_scale,
            "effective_weights": effective_weights,
            "weight_history": tuple(weight_history),
            "objective_history": jnp.asarray(objective_history),
            "residuals": residuals,
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
        inverse = jnp.where(distances > 1e-12, 1.0 / distances, 0.0)
        unit_logs = scale_tangent_samples(manifold, logs, inverse)
        average_sign = weighted_tangent_sum(manifold, unit_logs, weights)
        depths.append(jnp.clip(1.0 - manifold.norm(point, average_sign), 0.0, 1.0))
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
    center_fit = None
    if center is None:
        center_fit = frechet_median(
            manifold,
            adapted,
            sample_weight=sample_weight,
            maxiter=int(maxiter),
            tol=float(tol),
        )
        location = center_fit.point
    else:
        location = manifold.project(center)
    scores = manifold.dist(location, adapted.values)
    order = jnp.argsort(scores, stable=True)
    lower = jnp.sum(scores[None, :] < scores[:, None], axis=1)
    equal = jnp.sum(jnp.isclose(scores[None, :], scores[:, None]), axis=1)
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
