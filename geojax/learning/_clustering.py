"""Clustering algorithms for general exact-distance manifolds."""

from __future__ import annotations

import math
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
from ._statistics import _gradient_tolerances, frechet_mean
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


def _kmeans_stationarity(
    manifold: Any,
    data: ManifoldData,
    centers: list[Any],
    labels: Any,
    weights: Any,
) -> Any:
    """Return full-data gradient norms for fixed cluster assignments."""
    residuals = []
    for cluster, center in enumerate(centers):
        positions = jnp.flatnonzero(labels == cluster)
        if positions.size == 0 or float(jnp.sum(weights[positions])) <= 0.0:
            residuals.append(jnp.asarray(0.0, dtype=weights.dtype))
            continue
        logs = manifold.log(center, take_samples(manifold, data.values, positions))
        if not tree_all_finite(logs):
            raise FloatingPointError(
                "A clustering stationarity check encountered an undefined logarithm."
            )
        gradient = weighted_tangent_sum(manifold, logs, weights[positions])
        residuals.append(2.0 * manifold.norm(center, gradient))
    return jnp.asarray(residuals)


def _initial_indices(
    manifold: Any,
    data: ManifoldData,
    n_clusters: int,
    key: Any,
    method: str,
    weights: Any,
) -> Any:
    if method == "random":
        positive = jnp.flatnonzero(weights > 0.0)
        if int(positive.size) >= n_clusters:
            return jax.random.choice(
                key,
                data.n_samples,
                (n_clusters,),
                replace=False,
                p=weights,
            )
        # Weighted sampling without replacement is undefined once every
        # positive-mass observation has been selected. Include that support and
        # fill the remaining slots uniformly from zero-mass observations.
        key_positive, key_zero = jax.random.split(key)
        selected_positive = jax.random.permutation(key_positive, positive)
        zero = jnp.flatnonzero(weights == 0.0)
        selected_zero = jax.random.choice(
            key_zero,
            zero,
            (n_clusters - int(positive.size),),
            replace=False,
        )
        return jnp.concatenate((selected_positive, selected_zero))
    if method != "kmeans++":
        raise ValueError("init must be 'kmeans++', 'random', or explicit centers.")
    keys = jax.random.split(key, n_clusters)
    selected = [int(jax.random.choice(keys[0], data.n_samples, p=weights))]
    closest = manifold.squared_dist(take_point(manifold, data.values, selected[0]), data.values)
    for index in range(1, n_clusters):
        probabilities = weights * jnp.maximum(closest, 0.0)
        total = float(jnp.sum(probabilities))
        if total <= 0.0:
            available = [
                candidate for candidate in range(data.n_samples) if candidate not in selected
            ]
            chosen = available[0]
        else:
            probabilities = probabilities / total
            chosen = int(jax.random.choice(keys[index], data.n_samples, p=probabilities))
            if chosen in selected:
                available = [
                    candidate for candidate in range(data.n_samples) if candidate not in selected
                ]
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
) -> tuple[Any, int, int]:
    centers: list[Any | None] = [None] * n_clusters
    empty_clusters = []
    empty_count = 0
    unconverged_count = 0
    for cluster in range(n_clusters):
        indices = jnp.nonzero(labels == cluster, size=data.n_samples, fill_value=-1)[0]
        indices = indices[indices >= 0]
        if indices.size == 0 or float(jnp.sum(weights[indices])) <= 0.0:
            empty_count += 1
            empty_clusters.append(cluster)
            continue
        subset = as_manifold_data(manifold, take_samples(manifold, data.values, indices))
        fit = frechet_mean(
            manifold,
            subset,
            sample_weight=weights[indices],
            maxiter=center_maxiter,
            tol=center_tol,
        )
        if not fit.converged:
            unconverged_count += 1
        centers[cluster] = fit.point

    used_fallback: set[int] = set()
    for cluster in empty_clusters:
        present = [center for center in centers if center is not None]
        if not present:
            raise RuntimeError("Unable to recover a center before any nonempty center exists.")
        reference = stack_points(manifold, present)
        min_distances = jnp.min(
            pairwise_distances(manifold, data.values, reference, squared=True),
            axis=1,
        )
        order = jnp.argsort(weights * min_distances, stable=True)[::-1].tolist()
        chosen = next(index for index in order if index not in used_fallback)
        used_fallback.add(chosen)
        centers[cluster] = take_point(manifold, data.values, chosen)

    result = stack_points(manifold, [center for center in centers if center is not None])
    if not tree_all_finite(result):
        raise FloatingPointError("A cluster-center fit produced a nonfinite point.")
    return result, empty_count, unconverged_count


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
    n_clusters = integer_control(n_clusters, name="n_clusters")
    if not 1 <= n_clusters <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    center_maxiter = integer_control(center_maxiter, name="center_maxiter", minimum=1)
    n_init = integer_control(n_init, name="n_init", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    if isinstance(init, str):
        if init not in {"kmeans++", "random"}:
            raise ValueError("init must be 'kmeans++', 'random', or explicit centers.")
        random_key = as_key(key, "kmeans")
        run_keys = jax.random.split(random_key, n_init)
    else:
        if n_init != 1:
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
        previous_labels = None
        history = []
        assignment_changes = []
        empty_total = 0
        latest_center_nonconvergence = 0
        converged = False
        for iteration in range(1, maxiter + 1):
            distances_sq = pairwise_distances(manifold, adapted.values, centers, squared=True)
            labels = jnp.argmin(distances_sq, axis=1)
            centers, empty_count, latest_center_nonconvergence = _centers_from_labels(
                manifold,
                adapted,
                labels,
                n_clusters,
                weights,
                center_maxiter=center_maxiter,
                center_tol=tol,
            )
            empty_total += empty_count
            updated_distances = pairwise_distances(
                manifold,
                adapted.values,
                centers,
                squared=True,
            )
            updated_labels = jnp.argmin(updated_distances, axis=1)
            objective = jnp.sum(
                weights * updated_distances[jnp.arange(adapted.n_samples), updated_labels]
            )
            if not bool(jnp.isfinite(objective)):
                raise FloatingPointError("kmeans produced a nonfinite objective.")
            history.append(objective)
            changes = (
                adapted.n_samples
                if previous_labels is None
                else int(jnp.sum(updated_labels != previous_labels))
            )
            assignment_changes.append(changes)
            objective_scale = max(
                abs(float(previous)),
                abs(float(objective)),
                float(jnp.finfo(jnp.asarray(objective).dtype).tiny),
            )
            if (
                math.isfinite(float(previous))
                and abs(float(previous - objective)) <= tol * objective_scale
                and latest_center_nonconvergence == 0
                and bool(jnp.array_equal(updated_labels, labels))
            ):
                converged = True
                break
            previous = objective
            previous_labels = updated_labels
        final_distances = pairwise_distances(manifold, adapted.values, centers, squared=True)
        labels = jnp.argmin(final_distances, axis=1)
        objective = jnp.sum(weights * jnp.min(final_distances, axis=1))
        if not bool(jnp.isfinite(objective)):
            raise FloatingPointError("kmeans produced a nonfinite final objective.")
        result = ClusteringResult(
            labels=labels,
            centers=centers,
            objective=objective,
            iterations=iteration,
            converged=converged,
            reason="objective tolerance reached" if converged else "maximum iterations reached",
            diagnostics={
                "objective_history": jnp.asarray(history),
                "assignment_changes": jnp.asarray(assignment_changes),
                "empty_cluster_recoveries": empty_total,
                "unconverged_center_fits": latest_center_nonconvergence,
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
    size = integer_control(size, name="size")
    if size < 1:
        raise ValueError("size must be positive.")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    center_fit = frechet_mean(manifold, adapted, sample_weight=weights, maxiter=100)
    center = center_fit.point
    if not tree_all_finite(center) or not bool(jnp.isfinite(center_fit.objective)):
        raise FloatingPointError("The lightweight-coreset reference mean is nonfinite.")
    costs = manifold.squared_dist(center, adapted.values)
    total = jnp.sum(weights * costs)
    sensitivity = jnp.where(
        total > 0.0,
        0.5 * weights + 0.5 * weights * costs / jnp.maximum(total, jnp.finfo(total.dtype).tiny),
        weights,
    )
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
        diagnostics={
            "sampling_probabilities": sensitivity,
            "reference_center": center,
            "reference_mean_result": center_fit,
        },
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
    n_clusters = integer_control(n_clusters, name="n_clusters")
    if not 1 <= n_clusters <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
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
    for iteration in range(1, maxiter + 1):
        labels = jnp.argmin(distances[:, medoid_indices], axis=1)
        objective = jnp.sum(
            weights * distances[jnp.arange(adapted.n_samples), medoid_indices[labels]]
        )
        if not bool(jnp.isfinite(objective)):
            raise FloatingPointError("kmedoids produced a nonfinite objective.")
        history.append(objective)
        updated = []
        for cluster in range(n_clusters):
            members = jnp.nonzero(labels == cluster, size=adapted.n_samples, fill_value=-1)[0]
            members = members[members >= 0]
            if members.size == 0:
                closest = jnp.min(distances[:, medoid_indices], axis=1)
                unavailable = set(int(index) for index in medoid_indices.tolist())
                unavailable.update(updated)
                available = [
                    index for index in range(adapted.n_samples) if index not in unavailable
                ]
                if not available:
                    raise RuntimeError("Unable to recover a distinct empty-cluster medoid.")
                updated.append(
                    max(available, key=lambda index: float(weights[index] * closest[index]))
                )
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
    if not bool(jnp.isfinite(objective)):
        raise FloatingPointError("kmedoids produced a nonfinite final objective.")
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
    if not isinstance(linkage, str) or linkage not in {"single", "complete", "average"}:
        raise ValueError("linkage must be 'single', 'complete', or 'average'.")
    n_clusters = integer_control(n_clusters, name="n_clusters")
    if not 1 <= n_clusters <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    distances = pairwise_distances(manifold, adapted)
    active: dict[int, list[int]] = {index: [index] for index in range(adapted.n_samples)}
    linkage_rows = []
    target_labels = (
        jnp.arange(adapted.n_samples, dtype=int) if n_clusters == adapted.n_samples else None
    )
    next_id = adapted.n_samples
    while len(active) > 1:
        keys = sorted(active)
        best_pair = None
        best_distance = float("inf")
        for offset, left in enumerate(keys):
            for right in keys[offset + 1 :]:
                block = distances[
                    jnp.asarray(active[left])[:, None], jnp.asarray(active[right])[None, :]
                ]
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
        if len(active) == n_clusters:
            labels = [0] * adapted.n_samples
            for label, cluster_id in enumerate(sorted(active)):
                for member in active[cluster_id]:
                    labels[member] = label
            target_labels = jnp.asarray(labels, dtype=int)
    if target_labels is None:
        target_labels = jnp.zeros((adapted.n_samples,), dtype=int)
    linkage_matrix = (
        jnp.asarray(linkage_rows, dtype=float) if linkage_rows else jnp.empty((0, 4), dtype=float)
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
    n_clusters = integer_control(n_clusters, name="n_clusters")
    if not 1 <= n_clusters <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    if not isinstance(affinity, str):
        raise TypeError("affinity must be a string.")
    if not isinstance(laplacian, str):
        raise TypeError("laplacian must be a string.")
    distances = pairwise_distances(manifold, adapted)
    if affinity == "rbf":
        positive = distances[distances > 0.0]
        scale = float(jnp.median(positive)) if bandwidth is None and positive.size else 1.0
        scale = positive_control(
            bandwidth if bandwidth is not None else scale,
            name="bandwidth",
        )
        weights = jnp.exp(-(distances**2) / (2.0 * scale**2))
    elif affinity == "self_tuning":
        if bandwidth is not None:
            raise ValueError("bandwidth is only defined for affinity='rbf'.")
        n_neighbors = integer_control(n_neighbors, name="n_neighbors", minimum=1)
        if n_neighbors >= adapted.n_samples:
            raise ValueError("n_neighbors must be between 1 and n_samples - 1.")
        masked = jnp.where(jnp.eye(adapted.n_samples, dtype=bool), jnp.inf, distances)
        sorted_distances = jnp.sort(masked, axis=1)
        scales = sorted_distances[:, n_neighbors - 1]
        if bool(jnp.any(scales <= 0.0)):
            raise ValueError(
                "self_tuning affinity requires a positive kth-neighbor distance for every sample."
            )
        weights = jnp.exp(-(distances**2) / (scales[:, None] * scales[None, :]))
    else:
        raise ValueError("affinity must be 'rbf' or 'self_tuning'.")
    weights = weights.at[jnp.diag_indices(adapted.n_samples)].set(0.0)
    degrees = jnp.sum(weights, axis=1)
    if not bool(jnp.all(jnp.isfinite(weights))):
        raise FloatingPointError("spectral_clustering produced a nonfinite affinity matrix.")
    if bool(jnp.any(degrees <= 0.0)):
        raise ValueError(
            "The affinity graph contains an isolated sample; increase the bandwidth "
            "or use a less degenerate affinity."
        )
    if laplacian == "unnormalized":
        matrix = jnp.diag(degrees) - weights
        eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
        coordinates = eigenvectors[:, :n_clusters]
    elif laplacian == "symmetric":
        inverse = 1.0 / jnp.sqrt(degrees)
        matrix = jnp.eye(adapted.n_samples) - inverse[:, None] * weights * inverse[None, :]
        eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
        coordinates = eigenvectors[:, :n_clusters]
        norm = jnp.linalg.norm(coordinates, axis=1, keepdims=True)
        if bool(jnp.any(norm <= 0.0)):
            raise FloatingPointError("The symmetric spectral embedding contains a zero row.")
        coordinates = coordinates / norm
    elif laplacian == "random_walk":
        inverse_sqrt = 1.0 / jnp.sqrt(degrees)
        symmetric_matrix = (
            jnp.eye(adapted.n_samples) - inverse_sqrt[:, None] * weights * inverse_sqrt[None, :]
        )
        eigenvalues, symmetric_vectors = jnp.linalg.eigh(symmetric_matrix)
        coordinates = inverse_sqrt[:, None] * symmetric_vectors[:, :n_clusters]
        matrix = jnp.eye(adapted.n_samples) - weights / degrees[:, None]
    else:
        raise ValueError("laplacian must be 'unnormalized', 'symmetric', or 'random_walk'.")
    embedded = kmeans(
        Euclidean(size=(n_clusters,)),
        coordinates,
        n_clusters=n_clusters,
        key=key,
        maxiter=maxiter,
        center_maxiter=20,
    )
    labels = embedded.labels
    intrinsic_distances = pairwise_distances(manifold, adapted, squared=True)
    center_nonconvergence = 0
    if all(getattr(manifold, f"{name}_is_exact", False) for name in ("log", "exp")):
        centers, _, center_nonconvergence = _centers_from_labels(
            manifold,
            adapted,
            labels,
            n_clusters,
            normalize_weights(adapted.n_samples, None),
            center_maxiter=100,
            center_tol=1e-6,
        )
    else:
        medoid_indices = []
        for cluster in range(n_clusters):
            members = jnp.nonzero(labels == cluster, size=adapted.n_samples, fill_value=-1)[0]
            members = members[members >= 0]
            if members.size:
                within = intrinsic_distances[members[:, None], members[None, :]]
                medoid_indices.append(int(members[int(jnp.argmin(jnp.sum(within, axis=1)))]))
            else:
                medoid_indices.append(
                    next(index for index in range(adapted.n_samples) if index not in medoid_indices)
                )
        centers = take_samples(manifold, adapted.values, jnp.asarray(medoid_indices))
    center_distances = pairwise_distances(manifold, adapted.values, centers, squared=True)
    objective = jnp.mean(center_distances[jnp.arange(adapted.n_samples), labels])
    if not bool(jnp.isfinite(objective)):
        raise FloatingPointError("spectral_clustering produced a nonfinite objective.")
    fully_converged = embedded.converged and center_nonconvergence == 0
    return ClusteringResult(
        labels=labels,
        centers=centers,
        objective=objective,
        iterations=embedded.iterations,
        converged=fully_converged,
        reason=(
            embedded.reason
            if fully_converged
            else (
                f"{center_nonconvergence} intrinsic center fit(s) did not converge"
                if center_nonconvergence
                else embedded.reason
            )
        ),
        diagnostics={
            "embedding": coordinates,
            "eigenvalues": eigenvalues,
            "affinity": weights,
            "laplacian": laplacian,
            "laplacian_matrix": matrix,
            "degrees": degrees,
            "unconverged_center_fits": center_nonconvergence,
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
    bandwidth = positive_control(bandwidth, name="bandwidth")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    if merge_tol is not None:
        merge_tol = nonnegative_control(merge_tol, name="merge_tol")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    modes = []
    iteration_counts = []
    mode_converged = []
    for index in range(adapted.n_samples):
        point = take_point(manifold, adapted.values, index)
        converged = False
        for iteration in range(1, maxiter + 1):
            distances = manifold.dist(point, adapted.values)
            log_local = jnp.where(
                weights > 0.0,
                jnp.log(jnp.maximum(weights, jnp.finfo(weights.dtype).tiny))
                - 0.5 * (distances / bandwidth) ** 2,
                -jnp.inf,
            )
            maximum = jnp.max(log_local)
            if bool(jnp.isfinite(maximum)):
                local = jnp.exp(log_local - maximum)
                local = local / jnp.sum(local)
            else:
                eligible = jnp.where(weights > 0.0, distances, jnp.inf)
                local = jax.nn.one_hot(jnp.argmin(eligible), adapted.n_samples)
            direction = _weighted_logs(manifold, point, adapted.values, local)
            if not tree_all_finite(direction):
                raise FloatingPointError(
                    "mean_shift encountered an undefined logarithm, likely at a cut locus."
                )
            if float(manifold.norm(point, direction)) <= tol:
                converged = True
                break
            point = manifold.exp(point, direction)
            if not tree_all_finite(point):
                raise FloatingPointError("mean_shift left the certified exponential-map domain.")
        modes.append(point)
        iteration_counts.append(iteration)
        mode_converged.append(converged)
    threshold = merge_tol if merge_tol is not None else 0.5 * bandwidth
    unique = []
    for mode in modes:
        if (
            not unique
            or min(float(manifold.dist(mode, candidate)) for candidate in unique) > threshold
        ):
            unique.append(mode)
    centers = stack_points(manifold, unique)
    distances = pairwise_distances(manifold, adapted.values, centers)
    labels = jnp.argmin(distances, axis=1)
    objective = jnp.sum(weights * jnp.min(distances**2, axis=1))
    if not bool(jnp.isfinite(objective)):
        raise FloatingPointError("mean_shift produced a nonfinite objective.")
    return ClusteringResult(
        labels=labels,
        centers=centers,
        objective=objective,
        iterations=max(iteration_counts),
        converged=all(mode_converged),
        reason="all mode updates converged"
        if all(mode_converged)
        else "maximum iterations reached",
        diagnostics={
            "mode_iterations": jnp.asarray(iteration_counts),
            "mode_converged": jnp.asarray(mode_converged),
            "bandwidth": bandwidth,
        },
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
    tol: float = 1e-6,
) -> ClusteringResult:
    """Run competitive learning Riemannian quantization (CLRQ)."""
    require_exact_operations(manifold, "competitive_quantization", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "competitive_quantization")
    n_clusters = integer_control(n_clusters, name="n_clusters")
    if not 1 <= n_clusters <= adapted.n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    epochs = integer_control(epochs, name="epochs", minimum=1)
    initial_gain = positive_control(initial_gain, name="initial_gain")
    if initial_gain > 1.0:
        raise ValueError("initial_gain must lie in (0, 1].")
    decay = nonnegative_control(decay, name="decay")
    tol = nonnegative_control(tol, name="tol")
    random_key = as_key(key, "competitive_quantization")
    key_init, random_key = jax.random.split(random_key)
    indices = _initial_indices(
        manifold,
        adapted,
        n_clusters,
        key_init,
        "kmeans++",
        normalize_weights(adapted.n_samples, None),
    )
    centers = [take_point(manifold, adapted.values, int(index)) for index in indices]
    weights = normalize_weights(adapted.n_samples, None)
    initial_center_tree = stack_points(manifold, centers)
    initial_distances = pairwise_distances(
        manifold,
        adapted.values,
        initial_center_tree,
        squared=True,
    )
    initial_objective = jnp.mean(jnp.min(initial_distances, axis=1))
    if not bool(jnp.isfinite(initial_objective)):
        raise FloatingPointError("competitive_quantization has a nonfinite initial objective.")
    movement_scale = jnp.sqrt(jnp.maximum(initial_objective, 0.0))
    requested_stationarity_tol, effective_stationarity_tol = _gradient_tolerances(
        initial_center_tree,
        movement_scale,
        tol,
    )
    objective_dtype = jnp.asarray(initial_objective).dtype
    relative_objective_tolerance = max(
        tol,
        100.0 * float(jnp.finfo(objective_dtype).eps),
    )
    objective_history = []
    stationarity_history = []
    step = 0
    converged = False
    previous_objective = math.inf
    for epoch in range(epochs):
        random_key, subkey = jax.random.split(random_key)
        order = jax.random.permutation(subkey, adapted.n_samples)
        for sample_index in order.tolist():
            point = take_point(manifold, adapted.values, sample_index)
            winner = min(
                range(len(centers)), key=lambda index: float(manifold.dist(point, centers[index]))
            )
            gain = initial_gain / (1.0 + decay * math.sqrt(step + 1.0))
            direction = manifold.log(centers[winner], point)
            if not tree_all_finite(direction):
                raise FloatingPointError(
                    "competitive_quantization encountered an undefined logarithm."
                )
            centers[winner] = manifold.exp(
                centers[winner],
                manifold.lincomb(centers[winner], gain, direction),
            )
            if not tree_all_finite(centers[winner]):
                raise FloatingPointError(
                    "competitive_quantization left the certified exponential-map domain."
                )
            step += 1
        center_tree = stack_points(manifold, centers)
        distances = pairwise_distances(manifold, adapted.values, center_tree, squared=True)
        objective = jnp.mean(jnp.min(distances, axis=1))
        if not bool(jnp.isfinite(objective)):
            raise FloatingPointError("competitive_quantization produced a nonfinite objective.")
        labels = jnp.argmin(distances, axis=1)
        stationarity = _kmeans_stationarity(
            manifold,
            adapted,
            centers,
            labels,
            weights,
        )
        maximum_stationarity = float(jnp.max(stationarity))
        objective_history.append(objective)
        stationarity_history.append(stationarity)
        objective_scale = max(
            abs(previous_objective),
            abs(float(objective)),
            float(jnp.finfo(jnp.asarray(objective).dtype).tiny),
        )
        if (
            math.isfinite(previous_objective)
            and abs(previous_objective - float(objective))
            <= relative_objective_tolerance * objective_scale
            and maximum_stationarity <= effective_stationarity_tol
        ):
            converged = True
            break
        previous_objective = float(objective)
    centers_tree = stack_points(manifold, centers)
    distances = pairwise_distances(manifold, adapted.values, centers_tree, squared=True)
    labels = jnp.argmin(distances, axis=1)
    final_objective = jnp.mean(jnp.min(distances, axis=1))
    if not bool(jnp.isfinite(final_objective)):
        raise FloatingPointError("competitive_quantization produced a nonfinite final objective.")
    return ClusteringResult(
        labels=labels,
        centers=centers_tree,
        objective=final_objective,
        iterations=epoch + 1,
        converged=converged,
        reason=(
            "objective and stationarity tolerances reached"
            if converged
            else "requested epochs completed"
        ),
        diagnostics={
            "objective_history": jnp.asarray(objective_history),
            "stationarity_norms": jnp.asarray(stationarity_history),
            "updates": step,
            "requested_tolerance": tol,
            "requested_stationarity_tolerance": requested_stationarity_tol,
            "effective_stationarity_tolerance": effective_stationarity_tol,
            "relative_objective_tolerance": relative_objective_tolerance,
            "initial_rms_radius": movement_scale,
        },
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
