"""Clustering algorithms for general exact-distance manifolds."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from geojax.geometry import Euclidean

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import (
    ClusteringResult,
    CoresetResult,
    HierarchicalClusteringResult,
)
from ._statistics import frechet_mean
from ._utils import (
    as_key,
    normalize_weights,
    require_unbatched,
    stack_points,
    take_point,
    take_samples,
)


def _prepare(manifold: Any, data: Any, method: str) -> ManifoldData:
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, method)
    return adapted


def _initial_indices(
    manifold: Any,
    data: ManifoldData,
    n_clusters: int,
    key: Any,
    method: str,
    weights: Any,
) -> Any:
    if method == "random":
        return jax.random.choice(key, data.n_samples, (n_clusters,), replace=False, p=weights)
    if method != "kmeans++":
        raise ValueError("init must be 'kmeans++', 'random', or explicit centers.")
    keys = jax.random.split(key, n_clusters)
    selected = [int(jax.random.choice(keys[0], data.n_samples, p=weights))]
    closest = manifold.squared_dist(take_point(manifold, data.values, selected[0]), data.values)
    for index in range(1, n_clusters):
        probabilities = weights * jnp.maximum(closest, 0.0)
        total = float(jnp.sum(probabilities))
        if total <= 1e-15:
            available = [candidate for candidate in range(data.n_samples) if candidate not in selected]
            chosen = available[0]
        else:
            probabilities = probabilities / total
            chosen = int(jax.random.choice(keys[index], data.n_samples, p=probabilities))
            if chosen in selected:
                available = [candidate for candidate in range(data.n_samples) if candidate not in selected]
                chosen = max(available, key=lambda candidate: float(probabilities[candidate]))
        selected.append(chosen)
        distance = manifold.squared_dist(take_point(manifold, data.values, chosen), data.values)
        closest = jnp.minimum(closest, distance)
    return jnp.asarray(selected, dtype=int)


def _centers_from_labels(
    manifold: Any,
    data: ManifoldData,
    labels: Any,
    n_clusters: int,
    weights: Any,
    *,
    center_maxiter: int,
    center_tol: float,
    fallback_distances: Any,
) -> tuple[Any, int]:
    centers = []
    empty_count = 0
    min_distances = jnp.min(fallback_distances, axis=1)
    used_fallback: set[int] = set()
    for cluster in range(n_clusters):
        indices = jnp.nonzero(labels == cluster, size=data.n_samples, fill_value=-1)[0]
        indices = indices[indices >= 0]
        if indices.size == 0 or float(jnp.sum(weights[indices])) <= 0.0:
            empty_count += 1
            order = jnp.argsort(weights * min_distances)[::-1].tolist()
            chosen = next(index for index in order if index not in used_fallback)
            used_fallback.add(chosen)
            centers.append(take_point(manifold, data.values, chosen))
            continue
        subset = as_manifold_data(manifold, take_samples(manifold, data.values, indices))
        center = frechet_mean(
            manifold,
            subset,
            sample_weight=weights[indices],
            maxiter=center_maxiter,
            tol=center_tol,
        ).point
        centers.append(center)
    return stack_points(manifold, centers), empty_count


def kmeans(
    manifold: Any,
    data: Any,
    *,
    n_clusters: int,
    key: Any | int | None = None,
    sample_weight: Any | None = None,
    init: str | Any = "kmeans++",
    n_init: int = 1,
    maxiter: int = 100,
    tol: float = 1e-6,
    center_maxiter: int = 100,
) -> ClusteringResult:
    """Run weighted intrinsic Lloyd clustering with deterministic-key initialization."""
    require_exact_operations(manifold, "kmeans", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "kmeans")
    n_clusters = int(n_clusters)
    if not 1 <= n_clusters <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    if int(maxiter) < 1 or int(center_maxiter) < 1 or float(tol) < 0.0:
        raise ValueError(
            "maxiter and center_maxiter must be positive and tol must be nonnegative."
        )
    weights = normalize_weights(adapted.n_samples, sample_weight)
    if isinstance(init, str):
        random_key = as_key(key, "kmeans")
        if int(n_init) < 1:
            raise ValueError("n_init must be positive.")
        run_keys = jax.random.split(random_key, int(n_init))
    else:
        if int(n_init) != 1:
            raise ValueError("Explicit initial centers require n_init=1.")
        run_keys = [None]

    best: ClusteringResult | None = None
    for run, run_key in enumerate(run_keys):
        if isinstance(init, str):
            indices = _initial_indices(manifold, adapted, n_clusters, run_key, init, weights)
            centers = take_samples(manifold, adapted.values, indices)
        else:
            center_data = as_manifold_data(manifold, init)
            if center_data.n_samples != n_clusters or center_data.batch_shape:
                raise ValueError("Explicit initial centers must contain exactly n_clusters points.")
            centers = center_data.values
        previous = jnp.inf
        history = []
        empty_total = 0
        converged = False
        for iteration in range(1, int(maxiter) + 1):
            distances_sq = pairwise_distances(manifold, adapted.values, centers, squared=True)
            labels = jnp.argmin(distances_sq, axis=1)
            objective = jnp.sum(weights * jnp.min(distances_sq, axis=1))
            history.append(objective)
            if abs(float(previous - objective)) <= tol * max(1.0, float(objective)):
                converged = True
                break
            centers, empty_count = _centers_from_labels(
                manifold,
                adapted,
                labels,
                n_clusters,
                weights,
                center_maxiter=int(center_maxiter),
                center_tol=float(tol),
                fallback_distances=distances_sq,
            )
            empty_total += empty_count
            previous = objective
        final_distances = pairwise_distances(manifold, adapted.values, centers, squared=True)
        labels = jnp.argmin(final_distances, axis=1)
        objective = jnp.sum(weights * jnp.min(final_distances, axis=1))
        result = ClusteringResult(
            labels=labels,
            centers=centers,
            objective=objective,
            iterations=iteration,
            converged=converged,
            reason="objective tolerance reached" if converged else "maximum iterations reached",
            diagnostics={
                "objective_history": jnp.asarray(history),
                "empty_cluster_recoveries": empty_total,
                "run": run,
                "weights": weights,
            },
        )
        if best is None or float(result.objective) < float(best.objective):
            best = result
    assert best is not None
    return best


def lightweight_coreset(
    manifold: Any,
    data: Any,
    *,
    size: int,
    key: Any | int | None,
    sample_weight: Any | None = None,
) -> CoresetResult:
    """Sample the lightweight-coreset sensitivity heuristic on a manifold."""
    require_exact_operations(manifold, "lightweight_coreset", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "lightweight_coreset")
    size = int(size)
    if size < 1:
        raise ValueError("size must be positive.")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    center = frechet_mean(manifold, adapted, sample_weight=weights, maxiter=100).point
    costs = manifold.squared_dist(center, adapted.values)
    total = jnp.sum(weights * costs)
    sensitivity = 0.5 * weights + 0.5 * weights * costs / jnp.maximum(total, 1e-15)
    sensitivity = sensitivity / jnp.sum(sensitivity)
    indices = jax.random.choice(
        as_key(key, "lightweight_coreset"),
        adapted.n_samples,
        shape=(size,),
        replace=True,
        p=sensitivity,
    )
    coreset_weights = weights[indices] / (size * sensitivity[indices])
    coreset_weights = coreset_weights / jnp.sum(coreset_weights)
    return CoresetResult(
        indices=indices,
        points=take_samples(manifold, adapted.values, indices),
        weights=coreset_weights,
        diagnostics={"sampling_probabilities": sensitivity, "reference_center": center},
    )


def kmedoids(
    manifold: Any,
    data: Any,
    *,
    n_clusters: int,
    key: Any | int | None,
    sample_weight: Any | None = None,
    maxiter: int = 100,
) -> ClusteringResult:
    """Cluster using exact sample medoids and arbitrary manifold distances."""
    require_exact_operations(manifold, "kmedoids", "dist")
    adapted = _prepare(manifold, data, "kmedoids")
    n_clusters = int(n_clusters)
    if not 1 <= n_clusters <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    if int(maxiter) < 1:
        raise ValueError("maxiter must be positive.")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    distances = pairwise_distances(manifold, adapted, squared=False)
    medoid_indices = _initial_indices(
        manifold,
        adapted,
        n_clusters,
        as_key(key, "kmedoids"),
        "kmeans++",
        weights,
    )
    history = []
    converged = False
    for iteration in range(1, int(maxiter) + 1):
        labels = jnp.argmin(distances[:, medoid_indices], axis=1)
        objective = jnp.sum(weights * distances[jnp.arange(adapted.n_samples), medoid_indices[labels]])
        history.append(objective)
        updated = []
        for cluster in range(n_clusters):
            members = jnp.nonzero(labels == cluster, size=adapted.n_samples, fill_value=-1)[0]
            members = members[members >= 0]
            if members.size == 0:
                closest = jnp.min(distances[:, medoid_indices], axis=1)
                updated.append(int(jnp.argmax(weights * closest)))
            else:
                submatrix = distances[members[:, None], members[None, :]]
                costs = submatrix @ weights[members]
                updated.append(int(members[int(jnp.argmin(costs))]))
        new_indices = jnp.asarray(updated)
        if bool(jnp.array_equal(new_indices, medoid_indices)):
            converged = True
            medoid_indices = new_indices
            break
        medoid_indices = new_indices
    labels = jnp.argmin(distances[:, medoid_indices], axis=1)
    objective = jnp.sum(weights * distances[jnp.arange(adapted.n_samples), medoid_indices[labels]])
    return ClusteringResult(
        labels=labels,
        centers=take_samples(manifold, adapted.values, medoid_indices),
        objective=objective,
        iterations=iteration,
        converged=converged,
        reason="medoids unchanged" if converged else "maximum iterations reached",
        diagnostics={
            "medoid_indices": medoid_indices,
            "objective_history": jnp.asarray(history),
        },
    )


def agglomerative_clustering(
    manifold: Any,
    data: Any,
    *,
    n_clusters: int = 2,
    linkage: str = "average",
) -> HierarchicalClusteringResult:
    """Perform dense single, complete, or average-linkage clustering."""
    require_exact_operations(manifold, "agglomerative_clustering", "dist")
    adapted = _prepare(manifold, data, "agglomerative_clustering")
    if linkage not in {"single", "complete", "average"}:
        raise ValueError("linkage must be 'single', 'complete', or 'average'.")
    if not 1 <= int(n_clusters) <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    distances = pairwise_distances(manifold, adapted)
    active: dict[int, list[int]] = {index: [index] for index in range(adapted.n_samples)}
    linkage_rows = []
    target_labels = None
    next_id = adapted.n_samples
    while len(active) > 1:
        keys = sorted(active)
        best_pair = None
        best_distance = float("inf")
        for offset, left in enumerate(keys):
            for right in keys[offset + 1 :]:
                block = distances[jnp.asarray(active[left])[:, None], jnp.asarray(active[right])[None, :]]
                if linkage == "single":
                    value = float(jnp.min(block))
                elif linkage == "complete":
                    value = float(jnp.max(block))
                else:
                    value = float(jnp.mean(block))
                if (value, left, right) < (best_distance, *(best_pair or (10**9, 10**9))):
                    best_distance, best_pair = value, (left, right)
        assert best_pair is not None
        left, right = best_pair
        members = active.pop(left) + active.pop(right)
        active[next_id] = members
        linkage_rows.append([left, right, best_distance, len(members)])
        next_id += 1
        if len(active) == int(n_clusters):
            labels = [0] * adapted.n_samples
            for label, cluster_id in enumerate(sorted(active)):
                for member in active[cluster_id]:
                    labels[member] = label
            target_labels = jnp.asarray(labels, dtype=int)
    if target_labels is None:
        target_labels = jnp.zeros((adapted.n_samples,), dtype=int)
    linkage_matrix = (
        jnp.asarray(linkage_rows, dtype=float)
        if linkage_rows
        else jnp.empty((0, 4), dtype=float)
    )
    return HierarchicalClusteringResult(
        labels=target_labels,
        linkage=linkage_matrix,
        objective=jnp.sum(linkage_matrix[:, 2]) if linkage_rows else jnp.asarray(0.0),
        iterations=len(linkage_rows),
        converged=True,
        reason="complete dendrogram constructed",
        diagnostics={"method": linkage, "pairwise_distances": distances},
    )


def spectral_clustering(
    manifold: Any,
    data: Any,
    *,
    n_clusters: int,
    key: Any | int | None,
    affinity: str = "rbf",
    bandwidth: float | None = None,
    n_neighbors: int = 7,
    laplacian: str = "symmetric",
    maxiter: int = 100,
) -> ClusteringResult:
    """Cluster an exact-distance affinity graph through a Laplacian embedding."""
    require_exact_operations(manifold, "spectral_clustering", "dist")
    adapted = _prepare(manifold, data, "spectral_clustering")
    if not 1 <= int(n_clusters) <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    if int(maxiter) < 1:
        raise ValueError("maxiter must be positive.")
    distances = pairwise_distances(manifold, adapted)
    if affinity == "rbf":
        positive = distances[distances > 0.0]
        scale = float(jnp.median(positive)) if bandwidth is None and positive.size else 1.0
        scale = float(bandwidth) if bandwidth is not None else scale
        if scale <= 0.0:
            raise ValueError("bandwidth must be positive.")
        weights = jnp.exp(-(distances**2) / (2.0 * scale**2))
    elif affinity == "self_tuning":
        if not 1 <= int(n_neighbors) < adapted.n_samples:
            raise ValueError("n_neighbors must be between 1 and n_samples - 1.")
        masked = jnp.where(jnp.eye(adapted.n_samples, dtype=bool), jnp.inf, distances)
        sorted_distances = jnp.sort(masked, axis=1)
        scales = sorted_distances[:, int(n_neighbors) - 1]
        weights = jnp.exp(-(distances**2) / jnp.maximum(scales[:, None] * scales[None, :], 1e-15))
    else:
        raise ValueError("affinity must be 'rbf' or 'self_tuning'.")
    weights = weights.at[jnp.diag_indices(adapted.n_samples)].set(0.0)
    degrees = jnp.sum(weights, axis=1)
    if laplacian == "unnormalized":
        matrix = jnp.diag(degrees) - weights
    elif laplacian == "symmetric":
        inverse = 1.0 / jnp.sqrt(jnp.maximum(degrees, 1e-15))
        matrix = jnp.eye(adapted.n_samples) - inverse[:, None] * weights * inverse[None, :]
    elif laplacian == "random_walk":
        transition = weights / jnp.maximum(degrees[:, None], 1e-15)
        matrix = 0.5 * ((jnp.eye(adapted.n_samples) - transition) + (jnp.eye(adapted.n_samples) - transition).T)
    else:
        raise ValueError("laplacian must be 'unnormalized', 'symmetric', or 'random_walk'.")
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    coordinates = eigenvectors[:, : int(n_clusters)]
    if laplacian == "symmetric":
        norm = jnp.linalg.norm(coordinates, axis=1, keepdims=True)
        coordinates = coordinates / jnp.maximum(norm, 1e-15)
    embedded = kmeans(
        Euclidean(size=(int(n_clusters),)),
        coordinates,
        n_clusters=int(n_clusters),
        key=key,
        maxiter=maxiter,
        center_maxiter=20,
    )
    labels = embedded.labels
    intrinsic_distances = pairwise_distances(manifold, adapted, squared=True)
    centers, _ = _centers_from_labels(
        manifold,
        adapted,
        labels,
        int(n_clusters),
        normalize_weights(adapted.n_samples, None),
        center_maxiter=100,
        center_tol=1e-6,
        fallback_distances=intrinsic_distances[:, : int(n_clusters)],
    ) if all(getattr(manifold, f"{name}_is_exact", False) for name in ("log", "exp")) else (
        take_samples(manifold, adapted.values, jnp.arange(int(n_clusters))),
        0,
    )
    center_distances = pairwise_distances(manifold, adapted.values, centers, squared=True)
    objective = jnp.mean(jnp.min(center_distances, axis=1))
    return ClusteringResult(
        labels=labels,
        centers=centers,
        objective=objective,
        iterations=embedded.iterations,
        converged=embedded.converged,
        reason=embedded.reason,
        diagnostics={
            "embedding": coordinates,
            "eigenvalues": eigenvalues,
            "affinity": weights,
            "laplacian": laplacian,
        },
    )


def mean_shift(
    manifold: Any,
    data: Any,
    *,
    bandwidth: float,
    sample_weight: Any | None = None,
    maxiter: int = 100,
    tol: float = 1e-6,
    merge_tol: float | None = None,
) -> ClusteringResult:
    """Find modes by Gaussian-kernel geodesic mean-shift updates."""
    require_exact_operations(manifold, "mean_shift", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "mean_shift")
    if bandwidth <= 0.0 or int(maxiter) < 1 or float(tol) < 0.0:
        raise ValueError("bandwidth and maxiter must be positive and tol must be nonnegative.")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    modes = []
    iteration_counts = []
    for index in range(adapted.n_samples):
        point = take_point(manifold, adapted.values, index)
        for iteration in range(1, int(maxiter) + 1):
            distances = manifold.dist(point, adapted.values)
            local = weights * jnp.exp(-0.5 * (distances / bandwidth) ** 2)
            local = local / jnp.sum(local)
            direction = manifold.lincomb(point, 1.0, _weighted_logs(manifold, point, adapted.values, local))
            if float(manifold.norm(point, direction)) <= tol:
                break
            point = manifold.exp(point, direction)
        modes.append(point)
        iteration_counts.append(iteration)
    threshold = float(merge_tol) if merge_tol is not None else 0.5 * float(bandwidth)
    unique = []
    for mode in modes:
        if not unique or min(float(manifold.dist(mode, candidate)) for candidate in unique) > threshold:
            unique.append(mode)
    centers = stack_points(manifold, unique)
    distances = pairwise_distances(manifold, adapted.values, centers)
    labels = jnp.argmin(distances, axis=1)
    objective = jnp.sum(weights * jnp.min(distances**2, axis=1))
    return ClusteringResult(
        labels=labels,
        centers=centers,
        objective=objective,
        iterations=max(iteration_counts),
        converged=max(iteration_counts) < int(maxiter),
        reason="all mode updates converged" if max(iteration_counts) < int(maxiter) else "maximum iterations reached",
        diagnostics={"mode_iterations": jnp.asarray(iteration_counts), "bandwidth": bandwidth},
    )


def _weighted_logs(manifold: Any, point: Any, values: Any, weights: Any) -> Any:
    from ._utils import weighted_tangent_sum

    return weighted_tangent_sum(manifold, manifold.log(point, values), weights)


def competitive_quantization(
    manifold: Any,
    data: Any,
    *,
    n_clusters: int,
    key: Any | int | None,
    epochs: int = 10,
    initial_gain: float = 0.5,
    decay: float = 0.01,
) -> ClusteringResult:
    """Run competitive learning Riemannian quantization (CLRQ)."""
    require_exact_operations(manifold, "competitive_quantization", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "competitive_quantization")
    if not 1 <= int(n_clusters) <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    if int(epochs) < 1 or float(initial_gain) <= 0.0 or float(decay) < 0.0:
        raise ValueError("epochs and initial_gain must be positive and decay nonnegative.")
    random_key = as_key(key, "competitive_quantization")
    key_init, random_key = jax.random.split(random_key)
    indices = _initial_indices(
        manifold,
        adapted,
        int(n_clusters),
        key_init,
        "kmeans++",
        normalize_weights(adapted.n_samples, None),
    )
    centers = [take_point(manifold, adapted.values, int(index)) for index in indices]
    objective_history = []
    step = 0
    for epoch in range(int(epochs)):
        random_key, subkey = jax.random.split(random_key)
        order = jax.random.permutation(subkey, adapted.n_samples)
        for sample_index in order.tolist():
            point = take_point(manifold, adapted.values, sample_index)
            winner = min(range(len(centers)), key=lambda index: float(manifold.dist(point, centers[index])))
            gain = float(initial_gain) / (1.0 + float(decay) * jnp.sqrt(float(step + 1)))
            direction = manifold.log(centers[winner], point)
            centers[winner] = manifold.exp(
                centers[winner],
                manifold.lincomb(centers[winner], gain, direction),
            )
            step += 1
        center_tree = stack_points(manifold, centers)
        distances = pairwise_distances(manifold, adapted.values, center_tree, squared=True)
        objective_history.append(jnp.mean(jnp.min(distances, axis=1)))
    centers_tree = stack_points(manifold, centers)
    distances = pairwise_distances(manifold, adapted.values, centers_tree, squared=True)
    labels = jnp.argmin(distances, axis=1)
    return ClusteringResult(
        labels=labels,
        centers=centers_tree,
        objective=jnp.mean(jnp.min(distances, axis=1)),
        iterations=int(epochs),
        converged=True,
        reason="requested epochs completed",
        diagnostics={"objective_history": jnp.asarray(objective_history), "updates": step},
    )


__all__ = [
    "agglomerative_clustering",
    "competitive_quantization",
    "kmeans",
    "kmedoids",
    "lightweight_coreset",
    "mean_shift",
    "spectral_clustering",
]
