"""Intrinsic descriptive statistics for manifold-valued samples."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from geojax.optimization import ConjugateGradient, Minimize

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import EnclosingBallResult, FrechetMeanResult, FrechetMedianResult
from ._utils import (
    normalize_weights,
    require_unbatched,
    take_point,
    weighted_tangent_sum,
)


def _prepare(manifold: Any, data: Any, method: str) -> ManifoldData:
    adapted = data if isinstance(data, ManifoldData) else as_manifold_data(manifold, data)
    require_unbatched(adapted, method)
    return adapted


def _weighted_medoid(manifold: Any, data: ManifoldData, weights: Any, *, squared: bool) -> Any:
    distances = pairwise_distances(manifold, data, squared=squared)
    index = int(jnp.argmin(distances @ weights))
    return take_point(manifold, data.values, index)


def _floating_point_tolerance(point: Any) -> float:
    """Return the gradient resolution implied by the point's least precise leaf."""
    floors = []
    for leaf in jax.tree_util.tree_leaves(point):
        value = jnp.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            floors.append(float(jnp.sqrt(jnp.finfo(jnp.real(value).dtype).eps)))
    return max(floors, default=0.0)


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
    r"""Compute a weighted Fréchet mean minimizing ``sum_i w_i d(x, x_i)^2``."""
    require_exact_operations(manifold, "frechet_mean", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "frechet_mean")
    if int(maxiter) < 1 or float(tol) < 0.0:
        raise ValueError("maxiter must be positive and tol must be nonnegative.")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    x0 = (
        _weighted_medoid(manifold, adapted, weights, squared=True)
        if initial_point is None
        else manifold.project(initial_point)
    )
    effective_tol = max(float(tol), _floating_point_tolerance(x0))

    def objective(point: Any) -> Any:
        return jnp.sum(weights * manifold.squared_dist(point, adapted.values))

    selected_solver = solver or ConjugateGradient(
        maxiter=int(maxiter),
        tolgradnorm=effective_tol,
        verbosity=0,
    )
    point, value, history = Minimize(
        M=manifold,
        cost=objective,
        x0=x0,
        solver=selected_solver,
    ).solve()
    final = history[-1]
    converged = final.gradnorm <= effective_tol
    if converged:
        reason = (
            "gradient tolerance reached"
            if final.gradnorm <= tol
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
            "requested_tolerance": float(tol),
            "effective_tolerance": effective_tol,
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
    r"""Compute a weighted geometric median with a guarded Weiszfeld iteration."""
    require_exact_operations(manifold, "frechet_median", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "frechet_median")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    if smoothing <= 0.0 or int(maxiter) < 1 or float(tol) < 0.0:
        raise ValueError(
            "smoothing and maxiter must be positive and tol must be nonnegative."
        )
    point = (
        frechet_mean(
            manifold,
            adapted,
            sample_weight=weights,
            maxiter=min(int(maxiter), 100),
            tol=max(float(tol), 1e-6),
        ).point
        if initial_point is None
        else manifold.project(initial_point)
    )
    converged = False
    gradient_norm = jnp.inf
    objective_history: list[Any] = []
    for iteration in range(1, int(maxiter) + 1):
        distances = manifold.dist(point, adapted.values)
        objective = jnp.sum(weights * distances)
        objective_history.append(objective)
        inverse = weights / jnp.maximum(distances, float(smoothing))
        logs = manifold.log(point, adapted.values)
        direction = weighted_tangent_sum(manifold, logs, inverse / jnp.sum(inverse))
        gradient_norm = manifold.norm(point, direction)
        if not bool(jnp.all(jnp.isfinite(gradient_norm))):
            raise FloatingPointError("frechet_median encountered a nonfinite update.")
        if float(jnp.max(gradient_norm)) <= tol:
            converged = True
            break
        point = manifold.exp(point, direction)
    reason = "update tolerance reached" if converged else "maximum iterations reached"
    final_objective = jnp.sum(weights * manifold.dist(point, adapted.values))
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
    if int(maxiter) < 1 or float(tol) < 0.0:
        raise ValueError("maxiter must be positive and tol must be nonnegative.")
    point = (
        take_point(manifold, adapted.values, 0)
        if initial_point is None
        else manifold.project(initial_point)
    )
    radii: list[Any] = []
    converged = False
    previous = jnp.inf
    for iteration in range(1, int(maxiter) + 1):
        distances = manifold.dist(point, adapted.values)
        farthest_index = int(jnp.argmax(distances))
        radius = distances[farthest_index]
        radii.append(radius)
        if iteration > 1 and abs(float(previous - radius)) <= tol * max(1.0, float(radius)):
            converged = True
            break
        farthest = take_point(manifold, adapted.values, farthest_index)
        direction = manifold.log(point, farthest)
        if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(direction)):
            raise FloatingPointError("minimum_enclosing_ball encountered an undefined logarithm.")
        point = manifold.exp(point, manifold.lincomb(point, 1.0 / (iteration + 1.0), direction))
        previous = radius
    final_distances = manifold.dist(point, adapted.values)
    radius = jnp.max(final_distances)
    reason = "radius tolerance reached" if converged else "maximum iterations reached"
    return EnclosingBallResult(
        center=point,
        radius=radius,
        objective=radius,
        iterations=iteration,
        converged=converged,
        reason=reason,
        diagnostics={"radius_history": jnp.asarray(radii), "distances": final_distances},
    )


__all__ = ["frechet_mean", "frechet_median", "minimum_enclosing_ball"]
