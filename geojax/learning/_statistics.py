"""Intrinsic descriptive statistics for manifold-valued samples."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from geojax.optimization import AdaptiveArmijo, ConjugateGradient, Minimize

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import EnclosingBallResult, FrechetMeanResult, FrechetMedianResult
from ._utils import (
    integer_control,
    nonnegative_control,
    normalize_weights,
    positive_control,
    require_unbatched,
    stack_points,
    take_point,
    tree_all_finite,
    weighted_tangent_sum,
)


def _prepare(manifold: Any, data: Any, method: str) -> ManifoldData:
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, method)
    return adapted


def _weighted_medoid(manifold: Any, data: ManifoldData, weights: Any, *, squared: bool) -> Any:
    distances = pairwise_distances(manifold, data, squared=squared)
    index = int(jnp.argmin(distances @ weights))
    return take_point(manifold, data.values, index)


def _validated_point(manifold: Any, point: Any, *, name: str) -> Any:
    try:
        adapted = as_manifold_data(manifold, stack_points(manifold, [point]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid manifold point: {exc}") from exc
    return take_point(manifold, adapted.values, 0)


def _gradient_tolerances(
    point: Any,
    scale: Any,
    requested_relative_tolerance: float,
) -> tuple[float, float]:
    """Return requested and attainable gradient tolerances at a data scale."""
    epsilons = []
    tiny_values = []
    for leaf in jax.tree_util.tree_leaves(point):
        value = jnp.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            limits = jnp.finfo(jnp.real(value).dtype)
            epsilons.append(float(jnp.sqrt(limits.eps)))
            tiny_values.append(float(limits.tiny))
    dtype_factor = max(epsilons, default=0.0)
    tiny = max(tiny_values, default=0.0)
    scale_value = abs(float(jnp.asarray(scale)))
    if not math.isfinite(scale_value):
        raise FloatingPointError("The data scale is nonfinite.")
    absolute_scale = max(scale_value, tiny)
    requested = requested_relative_tolerance * absolute_scale
    attainable = max(requested_relative_tolerance, dtype_factor) * absolute_scale
    return requested, attainable


def frechet_mean(
    manifold: Any,
    data: Any,
    *,
    sample_weight: Any | None = None,
    initial_point: Any | None = None,
    solver: Any | None = None,
    maxiter: int = 200,
    tol: float = 1e-7,
) -> FrechetMeanResult:
    r"""Compute a local weighted Fréchet-mean minimizer.

    The optimized objective is ``sum_i w_i d(x, x_i)^2``. On a general
    manifold it need not be geodesically convex, so convergence certifies a
    stationary local solution rather than a unique global mean.
    """
    require_exact_operations(manifold, "frechet_mean", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "frechet_mean")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    x0 = (
        _weighted_medoid(manifold, adapted, weights, squared=True)
        if initial_point is None
        else _validated_point(manifold, initial_point, name="initial_point")
    )
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

    def objective(point: Any) -> Any:
        return jnp.sum(weights * manifold.squared_dist(point, adapted.values))

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
    point, value, history = Minimize(
        M=manifold,
        cost=objective,
        x0=x0,
        solver=selected_solver,
    ).solve()
    final = history[-1]
    if (
        not tree_all_finite(point)
        or not bool(jnp.isfinite(value))
        or not bool(jnp.isfinite(final.gradnorm))
    ):
        raise FloatingPointError(
            "frechet_mean produced a nonfinite result; the objective may meet a "
            "cut locus or the selected solver may be unstable."
        )
    _validated_point(manifold, point, name="computed mean")
    converged = final.gradnorm <= effective_tol
    if converged:
        reason = (
            "gradient tolerance reached"
            if final.gradnorm <= requested_gradient_tol
            else "gradient tolerance reached at floating-point resolution"
        )
    else:
        reason = final.reason or "maximum iterations reached"
    return FrechetMeanResult(
        point=point,
        objective=jnp.asarray(value),
        gradient_norm=jnp.asarray(final.gradnorm),
        iterations=int(final.iter),
        converged=bool(converged),
        reason=reason,
        diagnostics={
            "weights": weights,
            "history": tuple(history),
            "requested_tolerance": tol,
            "requested_gradient_tolerance": requested_gradient_tol,
            "effective_tolerance": effective_tol,
            "initial_rms_radius": initial_radius,
        },
    )


def frechet_median(
    manifold: Any,
    data: Any,
    *,
    sample_weight: Any | None = None,
    initial_point: Any | None = None,
    smoothing: float = 1e-8,
    maxiter: int = 200,
    tol: float = 1e-7,
) -> FrechetMedianResult:
    r"""Compute a Huber-smoothed weighted geometric median.

    The guarded Weiszfeld iteration minimizes ``sum_i w_i rho_s(d(x, x_i))``,
    where ``rho_s(r)=r`` for ``r >= s`` and
    ``rho_s(r)=r^2/(2s)+s/2`` otherwise. Thus ``smoothing`` controls the local
    approximation to the nonsmooth Fréchet-median objective.
    """
    require_exact_operations(manifold, "frechet_median", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "frechet_median")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    smoothing = positive_control(smoothing, name="smoothing")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    initialization_fit = None
    if initial_point is None:
        initialization_fit = frechet_mean(
            manifold,
            adapted,
            sample_weight=weights,
            maxiter=min(maxiter, 100),
            tol=tol,
        )
        point = initialization_fit.point
    else:
        point = _validated_point(manifold, initial_point, name="initial_point")
    initial_distances = manifold.dist(point, adapted.values)
    if not bool(jnp.all(jnp.isfinite(initial_distances))):
        raise FloatingPointError("frechet_median has nonfinite initial distances.")
    distance_scale = float(jnp.max(initial_distances))
    dtype = jnp.asarray(initial_distances).dtype
    requested_gradient_tol = tol
    effective_gradient_tol = max(
        tol,
        100.0 * float(jnp.finfo(dtype).eps) * adapted.n_samples,
    )
    movement_scale = max(distance_scale, float(jnp.finfo(dtype).tiny))
    requested_movement_tol = tol * movement_scale
    effective_movement_tol = (
        max(
            tol,
            100.0 * float(jnp.finfo(dtype).eps) * adapted.n_samples,
        )
        * movement_scale
    )
    converged = False
    gradient_norm = jnp.inf
    update_norm = jnp.inf
    reason = "maximum iterations reached"
    objective_history: list[Any] = []
    step_history: list[float] = []
    for iteration in range(1, maxiter + 1):
        distances = manifold.dist(point, adapted.values)
        if not bool(jnp.all(jnp.isfinite(distances))):
            raise FloatingPointError("frechet_median encountered a nonfinite distance.")
        losses = jnp.where(
            distances >= smoothing,
            distances,
            0.5 * (distances**2 / smoothing + smoothing),
        )
        objective = jnp.sum(weights * losses)
        objective_history.append(objective)
        inverse = weights / jnp.maximum(distances, smoothing)
        logs = manifold.log(point, adapted.values)
        if not tree_all_finite(logs):
            raise FloatingPointError("frechet_median encountered an undefined logarithm.")
        stationarity = weighted_tangent_sum(manifold, logs, inverse)
        gradient_norm = manifold.norm(point, stationarity)
        if not bool(jnp.isfinite(gradient_norm)):
            raise FloatingPointError("frechet_median encountered a nonfinite gradient.")
        if float(gradient_norm) <= effective_gradient_tol:
            converged = True
            update_norm = jnp.asarray(0.0, dtype=dtype)
            reason = (
                "gradient tolerance reached"
                if float(gradient_norm) <= requested_gradient_tol
                else "gradient tolerance reached at floating-point resolution"
            )
            break
        inverse_sum = jnp.sum(inverse)
        direction = manifold.lincomb(point, 1.0 / inverse_sum, stationarity)
        update_norm = manifold.norm(point, direction)
        if not bool(jnp.isfinite(update_norm)):
            raise FloatingPointError("frechet_median encountered a nonfinite update.")
        directional_derivative = -(gradient_norm**2) / inverse_sum
        fraction = 1.0
        accepted = False
        rounding = (
            100.0
            * float(jnp.finfo(dtype).eps)
            * max(abs(float(objective)), float(jnp.finfo(dtype).tiny))
        )
        candidate = point
        for _ in range(60):
            candidate = manifold.exp(
                point,
                manifold.lincomb(point, fraction, direction),
            )
            if tree_all_finite(candidate):
                candidate_distances = manifold.dist(candidate, adapted.values)
                candidate_losses = jnp.where(
                    candidate_distances >= smoothing,
                    candidate_distances,
                    0.5 * (candidate_distances**2 / smoothing + smoothing),
                )
                candidate_objective = jnp.sum(weights * candidate_losses)
                if bool(jnp.isfinite(candidate_objective)) and float(candidate_objective) <= float(
                    objective + 1e-4 * fraction * directional_derivative + rounding
                ):
                    accepted = True
                    break
            fraction *= 0.5
        step_history.append(fraction if accepted else 0.0)
        if not accepted:
            reason = "descent line search failed"
            break
        point = candidate
        if float(fraction * update_norm) <= effective_movement_tol:
            reason = "updates stopped before gradient stationarity was reached"
            break
    final_distances = manifold.dist(point, adapted.values)
    final_losses = jnp.where(
        final_distances >= smoothing,
        final_distances,
        0.5 * (final_distances**2 / smoothing + smoothing),
    )
    final_objective = jnp.sum(weights * final_losses)
    if not objective_history or not bool(
        jnp.array_equal(jnp.asarray(objective_history[-1]), jnp.asarray(final_objective))
    ):
        objective_history.append(final_objective)
    final_inverse = weights / jnp.maximum(final_distances, smoothing)
    final_stationarity = weighted_tangent_sum(
        manifold,
        manifold.log(point, adapted.values),
        final_inverse,
    )
    final_direction = manifold.lincomb(
        point,
        1.0 / jnp.sum(final_inverse),
        final_stationarity,
    )
    update_norm = manifold.norm(point, final_direction)
    gradient_norm = manifold.norm(point, final_stationarity)
    if not bool(jnp.all(jnp.isfinite(update_norm))) or not bool(
        jnp.all(jnp.isfinite(gradient_norm))
    ):
        raise FloatingPointError("frechet_median encountered a nonfinite final update.")
    point = _validated_point(manifold, point, name="computed median")
    converged = float(gradient_norm) <= effective_gradient_tol
    if converged:
        reason = (
            "gradient tolerance reached"
            if float(gradient_norm) <= requested_gradient_tol
            else "gradient tolerance reached at floating-point resolution"
        )
    return FrechetMedianResult(
        point=point,
        objective=final_objective,
        gradient_norm=gradient_norm,
        iterations=iteration,
        converged=converged,
        reason=reason,
        diagnostics={
            "weights": weights,
            "smoothing": float(smoothing),
            "objective_history": jnp.asarray(objective_history),
            "step_fractions": jnp.asarray(step_history),
            "final_distances": final_distances,
            "unsmoothed_objective": jnp.sum(weights * final_distances),
            "update_norm": update_norm,
            "initialization_fit": initialization_fit,
            "distance_scale": distance_scale,
            "requested_gradient_tolerance": requested_gradient_tol,
            "effective_gradient_tolerance": effective_gradient_tol,
            "requested_movement_tolerance": requested_movement_tol,
            "effective_movement_tolerance": effective_movement_tol,
        },
    )


def minimum_enclosing_ball(
    manifold: Any,
    data: Any,
    *,
    initial_point: Any | None = None,
    maxiter: int = 500,
    tol: float = 1e-7,
) -> EnclosingBallResult:
    """Approximate the smallest enclosing geodesic ball by farthest-point updates."""
    require_exact_operations(manifold, "minimum_enclosing_ball", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "minimum_enclosing_ball")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    point = (
        take_point(manifold, adapted.values, 0)
        if initial_point is None
        else _validated_point(manifold, initial_point, name="initial_point")
    )
    radii: list[Any] = []
    movements: list[Any] = []
    converged = False
    previous = jnp.inf
    for iteration in range(1, maxiter + 1):
        distances = manifold.dist(point, adapted.values)
        if not bool(jnp.all(jnp.isfinite(distances))):
            raise FloatingPointError("minimum_enclosing_ball produced a nonfinite distance.")
        farthest_index = int(jnp.argmax(distances))
        radius = distances[farthest_index]
        radii.append(radius)
        farthest = take_point(manifold, adapted.values, farthest_index)
        direction = manifold.log(point, farthest)
        if not all(
            bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(direction)
        ):
            raise FloatingPointError("minimum_enclosing_ball encountered an undefined logarithm.")
        step = manifold.lincomb(point, 1.0 / (iteration + 1.0), direction)
        movement = manifold.norm(point, step)
        movements.append(movement)
        if (
            iteration > 1
            and abs(float(previous - radius))
            <= tol
            * max(
                abs(float(previous)),
                abs(float(radius)),
                float(jnp.finfo(jnp.asarray(radius).dtype).tiny),
            )
            and float(movement)
            <= tol
            * max(
                abs(float(radius)),
                float(jnp.finfo(jnp.asarray(radius).dtype).tiny),
            )
        ):
            converged = True
            break
        point = manifold.exp(point, step)
        if not tree_all_finite(point):
            raise FloatingPointError(
                "minimum_enclosing_ball left the certified exponential-map domain."
            )
        previous = radius
    final_distances = manifold.dist(point, adapted.values)
    radius = jnp.max(final_distances)
    if not bool(jnp.isfinite(radius)):
        raise FloatingPointError("minimum_enclosing_ball produced a nonfinite radius.")
    point = _validated_point(manifold, point, name="computed enclosing-ball center")
    reason = (
        "farthest-point update tolerance reached" if converged else "maximum iterations reached"
    )
    return EnclosingBallResult(
        center=point,
        radius=radius,
        objective=radius,
        iterations=iteration,
        converged=converged,
        reason=reason,
        diagnostics={
            "radius_history": jnp.asarray(radii),
            "movement_history": jnp.asarray(movements),
            "distances": final_distances,
        },
    )


__all__ = ["frechet_mean", "frechet_median", "minimum_enclosing_ball"]
