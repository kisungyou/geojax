"""Graph-based semi-supervised learning from manifold distances."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import SemiSupervisedResult
from ._utils import require_unbatched


def _prepare_graph(
    manifold: Any,
    data: Any,
    method: str,
    *,
    bandwidth: float | None,
    n_neighbors: int | None,
) -> tuple[ManifoldData, Any, Any, float]:
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, method)
    distances = pairwise_distances(manifold, adapted)
    positive = distances[distances > 0.0]
    scale = float(jnp.median(positive)) if bandwidth is None and positive.size else bandwidth
    scale = 1.0 if scale is None else float(scale)
    if scale <= 0.0:
        raise ValueError("bandwidth must be positive.")
    affinity = jnp.exp(-(distances**2) / (2.0 * scale**2))
    affinity = affinity.at[jnp.diag_indices(adapted.n_samples)].set(0.0)
    if n_neighbors is not None:
        neighbors = int(n_neighbors)
        if not 1 <= neighbors < adapted.n_samples:
            raise ValueError("n_neighbors must be between 1 and n_samples - 1.")
        candidate_distances = distances.at[jnp.diag_indices(adapted.n_samples)].set(jnp.inf)
        order = jnp.argsort(candidate_distances, axis=1)[:, :neighbors]
        mask = jnp.zeros_like(affinity, dtype=bool)
        mask = mask.at[jnp.arange(adapted.n_samples)[:, None], order].set(True)
        mask = mask | mask.T
        affinity = jnp.where(mask, affinity, 0.0)
    return adapted, distances, affinity, scale


def _decode(classes: Any, indices: Any) -> Any:
    decoded = np.asarray(classes)[np.asarray(indices)]
    try:
        return jnp.asarray(decoded)
    except TypeError:
        return decoded


def label_propagation(
    manifold: Any,
    data: Any,
    labels: Any,
    *,
    unlabeled: Any = -1,
    bandwidth: float | None = None,
    n_neighbors: int | None = None,
    alpha: float = 0.95,
    maxiter: int = 1000,
    tol: float = 1e-7,
) -> SemiSupervisedResult:
    """Propagate categorical labels over a geodesic-distance affinity graph."""
    require_exact_operations(manifold, "label_propagation", "dist")
    adapted, distances, affinity, scale = _prepare_graph(
        manifold,
        data,
        "label_propagation",
        bandwidth=bandwidth,
        n_neighbors=n_neighbors,
    )
    label_values = np.asarray(labels)
    if label_values.shape != (adapted.n_samples,):
        raise ValueError(f"labels must have shape ({adapted.n_samples},).")
    labeled = label_values != unlabeled
    if not np.any(labeled) or np.all(labeled):
        raise ValueError("labels must contain both labeled and unlabeled observations.")
    classes = np.unique(label_values[labeled])
    encoded = np.searchsorted(classes, label_values[labeled])
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("alpha must lie in [0, 1).")
    if int(maxiter) < 1 or tol < 0.0:
        raise ValueError("maxiter must be positive and tol nonnegative.")
    initial = jnp.zeros((adapted.n_samples, len(classes)))
    labeled_indices = jnp.asarray(np.flatnonzero(labeled))
    initial = initial.at[labeled_indices, jnp.asarray(encoded)].set(1.0)
    degrees = jnp.sum(affinity, axis=1)
    inverse = 1.0 / jnp.sqrt(jnp.maximum(degrees, 1e-15))
    transition = inverse[:, None] * affinity * inverse[None, :]
    scores = initial
    converged = False
    history = []
    for iteration in range(1, int(maxiter) + 1):
        candidate = float(alpha) * transition @ scores + (1.0 - float(alpha)) * initial
        candidate = candidate.at[labeled_indices].set(initial[labeled_indices])
        change = jnp.max(jnp.abs(candidate - scores))
        history.append(change)
        scores = candidate
        if float(change) <= float(tol):
            converged = True
            break
    raw_scores = scores
    row_sums = jnp.sum(raw_scores, axis=1, keepdims=True)
    scores = jnp.where(
        row_sums > 1e-15,
        raw_scores / row_sums,
        jnp.full_like(raw_scores, 1.0 / len(classes)),
    )
    encoded_predictions = jnp.argmax(scores, axis=1)
    predictions = _decode(classes, encoded_predictions)
    differences = scores[:, None, :] - scores[None, :, :]
    objective = 0.5 * jnp.sum(affinity[..., None] * differences**2)
    return SemiSupervisedResult(
        predictions=predictions,
        scores=scores,
        objective=objective,
        iterations=iteration,
        converged=converged,
        reason="score tolerance reached" if converged else "maximum iterations reached",
        diagnostics={
            "classes": classes,
            "labeled_mask": jnp.asarray(labeled),
            "affinity": affinity,
            "distances": distances,
            "bandwidth": scale,
            "change_history": jnp.asarray(history),
            "raw_scores": raw_scores,
        },
    )


def manifold_regularized_regression(
    manifold: Any,
    data: Any,
    targets: Any,
    *,
    labeled_mask: Any | None = None,
    bandwidth: float | None = None,
    n_neighbors: int | None = None,
    ambient_regularization: float = 1e-3,
    intrinsic_regularization: float = 1.0,
) -> SemiSupervisedResult:
    """Fit transductive squared-loss regression with graph-Laplacian regularization."""
    require_exact_operations(manifold, "manifold_regularized_regression", "dist")
    adapted, distances, affinity, scale = _prepare_graph(
        manifold,
        data,
        "manifold_regularized_regression",
        bandwidth=bandwidth,
        n_neighbors=n_neighbors,
    )
    target_values = jnp.asarray(targets, dtype=float)
    if target_values.shape != (adapted.n_samples,):
        raise ValueError(f"targets must have shape ({adapted.n_samples},).")
    if labeled_mask is None:
        mask = jnp.isfinite(target_values)
    else:
        mask = jnp.asarray(labeled_mask, dtype=bool)
        if mask.shape != (adapted.n_samples,):
            raise ValueError(f"labeled_mask must have shape ({adapted.n_samples},).")
        if not bool(jnp.all(jnp.isfinite(target_values[mask]))):
            raise ValueError("labeled targets must be finite.")
    if not bool(jnp.any(mask)) or bool(jnp.all(mask)):
        raise ValueError("targets must identify both labeled and unlabeled observations.")
    if ambient_regularization <= 0.0 or intrinsic_regularization < 0.0:
        raise ValueError(
            "ambient_regularization must be positive and intrinsic_regularization nonnegative."
        )
    observed = jnp.where(mask, target_values, 0.0)
    label_matrix = jnp.diag(mask.astype(float))
    degree = jnp.diag(jnp.sum(affinity, axis=1))
    laplacian = degree - affinity
    system = (
        label_matrix
        + float(ambient_regularization) * jnp.eye(adapted.n_samples)
        + float(intrinsic_regularization) * laplacian
    )
    predictions = jnp.linalg.solve(system, observed)
    residual = jnp.where(mask, predictions - observed, 0.0)
    objective = (
        jnp.sum(residual**2)
        + float(ambient_regularization) * jnp.sum(predictions**2)
        + float(intrinsic_regularization) * predictions @ laplacian @ predictions
    )
    return SemiSupervisedResult(
        predictions=predictions,
        scores=predictions,
        objective=objective,
        iterations=1,
        converged=bool(jnp.all(jnp.isfinite(predictions))),
        reason="linear system solved",
        diagnostics={
            "labeled_mask": mask,
            "affinity": affinity,
            "laplacian": laplacian,
            "distances": distances,
            "bandwidth": scale,
            "system_matrix": system,
        },
    )


__all__ = ["label_propagation", "manifold_regularized_regression"]
