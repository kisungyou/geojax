"""Single-pass and mini-batch manifold summaries."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._clustering import _initial_indices
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import ClusteringResult, FrechetMeanResult
from ._utils import (
    as_key,
    normalize_weights,
    require_unbatched,
    stack_points,
    take_point,
    take_samples,
    weighted_tangent_sum,
)


def _prepare(manifold: Any, data: Any, method: str) -> ManifoldData:
    adapted = data if isinstance(data, ManifoldData) else as_manifold_data(manifold, data)
    require_unbatched(adapted, method)
    return adapted


def _mean_diagnostics(manifold: Any, point: Any, data: ManifoldData, weights: Any) -> tuple[Any, Any]:
    objective = jnp.sum(weights * manifold.squared_dist(point, data.values))
    gradient = weighted_tangent_sum(manifold, manifold.log(point, data.values), weights)
    return objective, manifold.norm(point, gradient)


def streaming_frechet_mean(
    manifold: Any,
    data: Any,
    *,
    sample_weight: Any | None = None,
    initial_point: Any | None = None,
) -> FrechetMeanResult:
    """Compute the one-pass inductive Fréchet mean in observation order."""
    require_exact_operations(manifold, "streaming_frechet_mean", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "streaming_frechet_mean")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    positive = [index for index in range(adapted.n_samples) if float(weights[index]) > 0.0]
    if not positive:
        raise ValueError("sample_weight must contain positive mass.")
    first = positive[0]
    point = (
        take_point(manifold, adapted.values, first)
        if initial_point is None
        else manifold.project(initial_point)
    )
    cumulative = 0.0 if initial_point is None else 1.0
    updates = []
    for index in positive:
        weight = float(weights[index])
        if initial_point is None and index == first:
            cumulative = weight
            updates.append(0.0)
            continue
        cumulative += weight
        step = weight / cumulative
        sample = take_point(manifold, adapted.values, index)
        direction = manifold.log(point, sample)
        point = manifold.exp(point, manifold.lincomb(point, step, direction))
        updates.append(step)
    objective, gradient_norm = _mean_diagnostics(manifold, point, adapted, weights)
    return FrechetMeanResult(
        point=point,
        objective=objective,
        gradient_norm=gradient_norm,
        iterations=len(positive),
        converged=True,
        reason="single pass completed",
        diagnostics={"weights": weights, "step_sizes": jnp.asarray(updates)},
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
    if not 1 <= int(batch_size) <= adapted.n_samples:
        raise ValueError("batch_size must be between 1 and n_samples.")
    if int(epochs) < 1 or learning_rate <= 0.0 or decay < 0.0 or tol < 0.0:
        raise ValueError(
            "epochs and learning_rate must be positive; decay and tol must be nonnegative."
        )
    weights = normalize_weights(adapted.n_samples, sample_weight)
    point = (
        streaming_frechet_mean(manifold, adapted, sample_weight=weights).point
        if initial_point is None
        else manifold.project(initial_point)
    )
    random_key = as_key(key, "minibatch_frechet_mean")
    objective_history = []
    movement_history = []
    update = 0
    converged = False
    for epoch in range(1, int(epochs) + 1):
        random_key, permutation_key = jax.random.split(random_key)
        order = jax.random.permutation(permutation_key, adapted.n_samples)
        maximum_movement = 0.0
        for start in range(0, adapted.n_samples, int(batch_size)):
            indices = order[start : start + int(batch_size)]
            batch = take_samples(manifold, adapted.values, indices)
            batch_weights = weights[indices]
            if float(jnp.sum(batch_weights)) <= 0.0:
                continue
            batch_weights = batch_weights / jnp.sum(batch_weights)
            direction = weighted_tangent_sum(manifold, manifold.log(point, batch), batch_weights)
            step = float(learning_rate) / (1.0 + float(decay) * update)
            movement = manifold.lincomb(point, step, direction)
            maximum_movement = max(maximum_movement, float(manifold.norm(point, movement)))
            point = manifold.exp(point, movement)
            update += 1
        objective, _ = _mean_diagnostics(manifold, point, adapted, weights)
        objective_history.append(objective)
        movement_history.append(maximum_movement)
        if maximum_movement <= float(tol):
            converged = True
            break
    objective, gradient_norm = _mean_diagnostics(manifold, point, adapted, weights)
    return FrechetMeanResult(
        point=point,
        objective=objective,
        gradient_norm=gradient_norm,
        iterations=epoch,
        converged=converged,
        reason="update tolerance reached" if converged else "requested epochs completed",
        diagnostics={
            "weights": weights,
            "objective_history": jnp.asarray(objective_history),
            "maximum_movement": jnp.asarray(movement_history),
            "updates": update,
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
    n_clusters = int(n_clusters)
    if not 1 <= n_clusters <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    if not 1 <= int(batch_size) <= adapted.n_samples:
        raise ValueError("batch_size must be between 1 and n_samples.")
    if int(epochs) < 1 or learning_rate <= 0.0 or decay < 0.0 or tol < 0.0:
        raise ValueError(
            "epochs and learning_rate must be positive; decay and tol must be nonnegative."
        )
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
    update_counts = jnp.zeros((n_clusters,), dtype=int)
    objective_history = []
    converged = False
    previous = jnp.inf
    for epoch in range(1, int(epochs) + 1):
        random_key, permutation_key = jax.random.split(random_key)
        order = jax.random.permutation(permutation_key, adapted.n_samples)
        for start in range(0, adapted.n_samples, int(batch_size)):
            batch_indices = order[start : start + int(batch_size)]
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
                local_weights = local_weights / local_mass
                direction = weighted_tangent_sum(
                    manifold,
                    manifold.log(centers[cluster], cluster_points),
                    local_weights,
                )
                count = int(update_counts[cluster])
                step = float(learning_rate) / (1.0 + float(decay) * count)
                centers[cluster] = manifold.exp(
                    centers[cluster],
                    manifold.lincomb(centers[cluster], step, direction),
                )
                update_counts = update_counts.at[cluster].add(1)
        center_tree = stack_points(manifold, centers)
        distances = pairwise_distances(manifold, adapted.values, center_tree, squared=True)
        objective = jnp.sum(weights * jnp.min(distances, axis=1))
        objective_history.append(objective)
        if abs(float(previous - objective)) <= float(tol) * max(1.0, float(objective)):
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
        reason="objective tolerance reached" if converged else "requested epochs completed",
        diagnostics={
            "objective_history": jnp.asarray(objective_history),
            "update_counts": update_counts,
            "weights": weights,
        },
    )


__all__ = ["minibatch_frechet_mean", "minibatch_kmeans", "streaming_frechet_mean"]
