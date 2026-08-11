"""Single-pass and mini-batch manifold summaries."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._clustering import _initial_indices, _kmeans_stationarity
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import ClusteringResult, FrechetMeanResult
from ._statistics import _gradient_tolerances
from ._utils import (
    as_key,
    integer_control,
    nonnegative_control,
    normalize_weights,
    positive_control,
    require_unbatched,
    stack_points,
    take_point,
    take_samples,
    tree_all_finite,
    weighted_tangent_sum,
)


def _prepare(manifold: Any, data: Any, method: str) -> ManifoldData:
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, method)
    return adapted


def _mean_diagnostics(
    manifold: Any, point: Any, data: ManifoldData, weights: Any
) -> tuple[Any, Any]:
    objective = jnp.sum(weights * manifold.squared_dist(point, data.values))
    logs = manifold.log(point, data.values)
    if not tree_all_finite(logs):
        raise FloatingPointError("Mean diagnostics encountered an undefined logarithm.")
    gradient = weighted_tangent_sum(manifold, logs, weights)
    gradient_norm = 2.0 * manifold.norm(point, gradient)
    if not bool(jnp.isfinite(objective)) or not bool(jnp.isfinite(gradient_norm)):
        raise FloatingPointError("Mean diagnostics are nonfinite.")
    return objective, gradient_norm


def _validated_point(manifold: Any, point: Any, *, name: str) -> Any:
    try:
        adapted = as_manifold_data(manifold, stack_points(manifold, [point]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid manifold point: {exc}") from exc
    return take_point(manifold, adapted.values, 0)


def streaming_frechet_mean(
    manifold: Any,
    data: Any,
    *,
    sample_weight: Any | None = None,
    initial_point: Any | None = None,
    initial_weight: float = 0.0,
) -> FrechetMeanResult:
    """Compute the one-pass inductive Fréchet mean in observation order.

    ``initial_point`` influences the estimate only when ``initial_weight`` is
    positive, making any prior mass explicit rather than silently counting the
    starting point as an extra observation.
    """
    require_exact_operations(manifold, "streaming_frechet_mean", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "streaming_frechet_mean")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    positive = [index for index in range(adapted.n_samples) if float(weights[index]) > 0.0]
    if not positive:
        raise ValueError("sample_weight must contain positive mass.")
    initial_weight = nonnegative_control(initial_weight, name="initial_weight")
    if initial_weight > 0.0 and initial_point is None:
        raise ValueError("initial_point is required when initial_weight is positive.")
    first = positive[0]
    point = (
        take_point(manifold, adapted.values, first)
        if initial_point is None
        else _validated_point(manifold, initial_point, name="initial_point")
    )
    cumulative = initial_weight
    updates = []
    for index in positive:
        weight = float(weights[index])
        if cumulative == 0.0:
            point = take_point(manifold, adapted.values, index)
            cumulative = weight
            updates.append(0.0)
            continue
        cumulative += weight
        step = weight / cumulative
        sample = take_point(manifold, adapted.values, index)
        direction = manifold.log(point, sample)
        if not tree_all_finite(direction):
            raise FloatingPointError("streaming_frechet_mean encountered an undefined logarithm.")
        point = manifold.exp(point, manifold.lincomb(point, step, direction))
        if not tree_all_finite(point):
            raise FloatingPointError(
                "streaming_frechet_mean left the certified exponential-map domain."
            )
        updates.append(step)
    objective, gradient_norm = _mean_diagnostics(manifold, point, adapted, weights)
    return FrechetMeanResult(
        point=point,
        objective=objective,
        gradient_norm=gradient_norm,
        iterations=len(positive),
        converged=True,
        reason="single pass completed",
        diagnostics={
            "weights": weights,
            "step_sizes": jnp.asarray(updates),
            "initial_weight": initial_weight,
        },
    )


def minibatch_frechet_mean(
    manifold: Any,
    data: Any,
    *,
    batch_size: int = 32,
    epochs: int = 10,
    key: Any | int | None,
    sample_weight: Any | None = None,
    initial_point: Any | None = None,
    learning_rate: float = 1.0,
    decay: float = 0.1,
    tol: float = 1e-6,
) -> FrechetMeanResult:
    """Approximate a Fréchet mean by shuffled mini-batch log-map updates."""
    require_exact_operations(manifold, "minibatch_frechet_mean", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "minibatch_frechet_mean")
    batch_size = integer_control(batch_size, name="batch_size")
    if not 1 <= batch_size <= adapted.n_samples:
        raise ValueError("batch_size must be between 1 and n_samples.")
    epochs = integer_control(epochs, name="epochs", minimum=1)
    learning_rate = positive_control(learning_rate, name="learning_rate")
    decay = nonnegative_control(decay, name="decay")
    tol = nonnegative_control(tol, name="tol")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    point = (
        streaming_frechet_mean(manifold, adapted, sample_weight=weights).point
        if initial_point is None
        else _validated_point(manifold, initial_point, name="initial_point")
    )
    initial_objective, _ = _mean_diagnostics(manifold, point, adapted, weights)
    movement_scale = jnp.sqrt(jnp.maximum(initial_objective, 0.0))
    requested_movement_tol, effective_movement_tol = _gradient_tolerances(
        point,
        movement_scale,
        tol,
    )
    random_key = as_key(key, "minibatch_frechet_mean")
    objective_history = []
    movement_history = []
    gradient_history = []
    update = 0
    converged = False
    for epoch in range(1, epochs + 1):
        random_key, permutation_key = jax.random.split(random_key)
        order = jax.random.permutation(permutation_key, adapted.n_samples)
        maximum_movement = 0.0
        for start in range(0, adapted.n_samples, batch_size):
            indices = order[start : start + batch_size]
            batch = take_samples(manifold, adapted.values, indices)
            batch_weights = weights[indices]
            if float(jnp.sum(batch_weights)) <= 0.0:
                continue
            # This Horvitz-Thompson scaling is unbiased for the full weighted
            # log-map direction under a uniformly shuffled mini-batch. Merely
            # normalizing within each batch overweights batches carrying little
            # probability mass when sample weights are unequal.
            logs = manifold.log(point, batch)
            if not tree_all_finite(logs):
                raise FloatingPointError(
                    "minibatch_frechet_mean encountered an undefined logarithm."
                )
            direction = weighted_tangent_sum(
                manifold,
                logs,
                batch_weights * adapted.n_samples / int(indices.shape[0]),
            )
            step = learning_rate / (1.0 + decay * update)
            movement = manifold.lincomb(point, step, direction)
            maximum_movement = max(maximum_movement, float(manifold.norm(point, movement)))
            point = manifold.exp(point, movement)
            if not tree_all_finite(point):
                raise FloatingPointError(
                    "minibatch_frechet_mean left the certified exponential-map domain."
                )
            update += 1
        objective, epoch_gradient_norm = _mean_diagnostics(
            manifold,
            point,
            adapted,
            weights,
        )
        objective_history.append(objective)
        movement_history.append(maximum_movement)
        gradient_history.append(epoch_gradient_norm)
        if (
            maximum_movement <= effective_movement_tol
            and epoch_gradient_norm <= effective_movement_tol
        ):
            converged = True
            break
    objective, gradient_norm = _mean_diagnostics(manifold, point, adapted, weights)
    return FrechetMeanResult(
        point=point,
        objective=objective,
        gradient_norm=gradient_norm,
        iterations=epoch,
        converged=converged,
        reason=(
            "movement and stationarity tolerances reached"
            if converged and maximum_movement <= requested_movement_tol
            else (
                "movement and stationarity tolerances reached at floating-point resolution"
                if converged
                else "requested epochs completed"
            )
        ),
        diagnostics={
            "weights": weights,
            "objective_history": jnp.asarray(objective_history),
            "maximum_movement": jnp.asarray(movement_history),
            "gradient_norm_history": jnp.asarray(gradient_history),
            "updates": update,
            "requested_tolerance": tol,
            "requested_movement_tolerance": requested_movement_tol,
            "effective_movement_tolerance": effective_movement_tol,
            "initial_rms_radius": movement_scale,
        },
    )


def minibatch_kmeans(
    manifold: Any,
    data: Any,
    *,
    n_clusters: int,
    batch_size: int = 32,
    epochs: int = 10,
    key: Any | int | None,
    sample_weight: Any | None = None,
    learning_rate: float = 0.5,
    decay: float = 0.01,
    tol: float = 1e-6,
) -> ClusteringResult:
    """Run shuffled mini-batch intrinsic k-means center updates."""
    require_exact_operations(manifold, "minibatch_kmeans", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "minibatch_kmeans")
    n_clusters = integer_control(n_clusters, name="n_clusters")
    if not 1 <= n_clusters <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    batch_size = integer_control(batch_size, name="batch_size")
    if not 1 <= batch_size <= adapted.n_samples:
        raise ValueError("batch_size must be between 1 and n_samples.")
    epochs = integer_control(epochs, name="epochs", minimum=1)
    learning_rate = positive_control(learning_rate, name="learning_rate")
    decay = nonnegative_control(decay, name="decay")
    tol = nonnegative_control(tol, name="tol")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    random_key = as_key(key, "minibatch_kmeans")
    initialization_key, random_key = jax.random.split(random_key)
    indices = _initial_indices(
        manifold,
        adapted,
        n_clusters,
        initialization_key,
        "kmeans++",
        weights,
    )
    centers = [take_point(manifold, adapted.values, int(index)) for index in indices]
    initial_center_tree = stack_points(manifold, centers)
    initial_distances = pairwise_distances(
        manifold,
        adapted.values,
        initial_center_tree,
        squared=True,
    )
    initial_objective = jnp.sum(weights * jnp.min(initial_distances, axis=1))
    if not bool(jnp.isfinite(initial_objective)):
        raise FloatingPointError("minibatch_kmeans has a nonfinite initial objective.")
    movement_scale = jnp.sqrt(jnp.maximum(initial_objective, 0.0))
    requested_movement_tol, effective_movement_tol = _gradient_tolerances(
        initial_center_tree,
        movement_scale,
        tol,
    )
    objective_dtype = jnp.asarray(initial_objective).dtype
    relative_objective_tolerance = max(
        tol,
        100.0 * float(jnp.finfo(objective_dtype).eps),
    )
    update_counts = jnp.zeros((n_clusters,), dtype=int)
    objective_history = []
    movement_history = []
    stationarity_history = []
    converged = False
    previous = jnp.inf
    for epoch in range(1, epochs + 1):
        random_key, permutation_key = jax.random.split(random_key)
        order = jax.random.permutation(permutation_key, adapted.n_samples)
        maximum_movement = 0.0
        for start in range(0, adapted.n_samples, batch_size):
            batch_indices = order[start : start + batch_size]
            batch = take_samples(manifold, adapted.values, batch_indices)
            center_tree = stack_points(manifold, centers)
            distances = pairwise_distances(manifold, batch, center_tree, squared=True)
            assignments = jnp.argmin(distances, axis=1)
            for cluster in range(n_clusters):
                positions = jnp.flatnonzero(assignments == cluster)
                if positions.size == 0:
                    continue
                cluster_points = take_samples(manifold, batch, positions)
                local_weights = weights[batch_indices[positions]]
                local_mass = jnp.sum(local_weights)
                if float(local_mass) <= 0.0:
                    continue
                logs = manifold.log(centers[cluster], cluster_points)
                if not tree_all_finite(logs):
                    raise FloatingPointError("minibatch_kmeans encountered an undefined logarithm.")
                direction = weighted_tangent_sum(
                    manifold,
                    logs,
                    local_weights * adapted.n_samples / int(batch_indices.shape[0]),
                )
                count = int(update_counts[cluster])
                step = learning_rate / (1.0 + decay * count)
                movement = manifold.lincomb(centers[cluster], step, direction)
                maximum_movement = max(
                    maximum_movement,
                    float(manifold.norm(centers[cluster], movement)),
                )
                centers[cluster] = manifold.exp(
                    centers[cluster],
                    movement,
                )
                if not tree_all_finite(centers[cluster]):
                    raise FloatingPointError(
                        "minibatch_kmeans left the certified exponential-map domain."
                    )
                update_counts = update_counts.at[cluster].add(1)
        center_tree = stack_points(manifold, centers)
        distances = pairwise_distances(manifold, adapted.values, center_tree, squared=True)
        epoch_labels = jnp.argmin(distances, axis=1)
        objective = jnp.sum(weights * jnp.min(distances, axis=1))
        stationarity = _kmeans_stationarity(
            manifold,
            adapted,
            centers,
            epoch_labels,
            weights,
        )
        maximum_stationarity = float(jnp.max(stationarity))
        if not bool(jnp.isfinite(objective)):
            raise FloatingPointError("minibatch_kmeans produced a nonfinite objective.")
        objective_history.append(objective)
        movement_history.append(maximum_movement)
        stationarity_history.append(stationarity)
        if (
            math.isfinite(float(previous))
            and abs(float(previous - objective))
            <= relative_objective_tolerance
            * max(
                abs(float(previous)),
                abs(float(objective)),
                float(jnp.finfo(objective_dtype).tiny),
            )
            and maximum_movement <= effective_movement_tol
            and maximum_stationarity <= effective_movement_tol
        ):
            converged = True
            break
        previous = objective
    centers_tree = stack_points(manifold, centers)
    final_distances = pairwise_distances(manifold, adapted.values, centers_tree, squared=True)
    labels = jnp.argmin(final_distances, axis=1)
    objective = jnp.sum(weights * jnp.min(final_distances, axis=1))
    return ClusteringResult(
        labels=labels,
        centers=centers_tree,
        objective=objective,
        iterations=epoch,
        converged=converged,
        reason=(
            "objective, movement, and stationarity tolerances reached"
            if converged
            else "requested epochs completed"
        ),
        diagnostics={
            "objective_history": jnp.asarray(objective_history),
            "maximum_movement": jnp.asarray(movement_history),
            "stationarity_norms": jnp.asarray(stationarity_history),
            "update_counts": update_counts,
            "weights": weights,
            "requested_tolerance": tol,
            "requested_movement_tolerance": requested_movement_tol,
            "effective_movement_tolerance": effective_movement_tol,
            "relative_objective_tolerance": relative_objective_tolerance,
            "initial_rms_radius": movement_scale,
        },
    )


__all__ = ["minibatch_frechet_mean", "minibatch_kmeans", "streaming_frechet_mean"]
