"""Graph-based semi-supervised learning from manifold distances."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import SemiSupervisedResult
from ._utils import (
    as_real_array,
    integer_control,
    interval_control,
    nonnegative_control,
    positive_control,
    require_unbatched,
)


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
    scale = positive_control(1.0 if scale is None else scale, name="bandwidth")
    affinity = jnp.exp(-(distances**2) / (2.0 * scale**2))
    affinity = affinity.at[jnp.diag_indices(adapted.n_samples)].set(0.0)
    if n_neighbors is not None:
        neighbors = integer_control(n_neighbors, name="n_neighbors")
        if not 1 <= neighbors < adapted.n_samples:
            raise ValueError("n_neighbors must be between 1 and n_samples - 1.")
        candidate_distances = distances.at[jnp.diag_indices(adapted.n_samples)].set(jnp.inf)
        order = jnp.argsort(candidate_distances, axis=1)[:, :neighbors]
        mask = jnp.zeros_like(affinity, dtype=bool)
        mask = mask.at[jnp.arange(adapted.n_samples)[:, None], order].set(True)
        mask = mask | mask.T
        affinity = jnp.where(mask, affinity, 0.0)
    if not bool(jnp.all(jnp.isfinite(affinity))):
        raise FloatingPointError("The geodesic affinity matrix is nonfinite.")
    return adapted, distances, affinity, scale


def _require_labeled_component(affinity: Any, labeled: Any) -> None:
    n_samples = affinity.shape[0]
    reachable = (affinity > 0.0) | jnp.eye(n_samples, dtype=bool)
    for pivot in range(n_samples):
        reachable = reachable | (reachable[:, pivot, None] & reachable[pivot, None, :])
    informed = jnp.any(reachable[:, jnp.asarray(labeled, dtype=bool)], axis=1)
    if not bool(jnp.all(informed)):
        raise ValueError(
            "Every affinity-graph component must contain at least one labeled observation."
        )


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
    unlabeled_value = np.asarray(unlabeled)
    if unlabeled_value.ndim != 0:
        raise ValueError("unlabeled must be a scalar sentinel.")
    labeled = label_values != unlabeled_value.item()
    if not np.any(labeled) or np.all(labeled):
        raise ValueError("labels must contain both labeled and unlabeled observations.")
    labeled_values = label_values[labeled]
    if labeled_values.dtype.kind in {"f", "c"} and not np.all(np.isfinite(labeled_values)):
        raise ValueError("Observed labels must be finite.")
    if labeled_values.dtype.kind in {"O", "U", "S"} and any(
        value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        for value in labeled_values.tolist()
    ):
        raise ValueError("Observed labels must not be missing.")
    try:
        classes = np.unique(labeled_values)
    except TypeError as exc:
        raise TypeError("Observed labels must be mutually comparable scalars.") from exc
    encoded = np.searchsorted(classes, label_values[labeled])
    alpha = interval_control(
        alpha,
        name="alpha",
        lower=0.0,
        upper=1.0,
        upper_closed=False,
    )
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    _require_labeled_component(affinity, labeled)
    initial = jnp.zeros((adapted.n_samples, len(classes)))
    labeled_indices = jnp.asarray(np.flatnonzero(labeled))
    initial = initial.at[labeled_indices, jnp.asarray(encoded)].set(1.0)
    degrees = jnp.sum(affinity, axis=1)
    inverse = jnp.where(degrees > 0.0, 1.0 / jnp.sqrt(degrees), 0.0)
    transition = inverse[:, None] * affinity * inverse[None, :]
    scores = initial
    converged = False
    history = []
    for iteration in range(1, maxiter + 1):
        candidate = alpha * transition @ scores + (1.0 - alpha) * initial
        candidate = candidate.at[labeled_indices].set(initial[labeled_indices])
        change = jnp.max(jnp.abs(candidate - scores))
        history.append(change)
        scores = candidate
        if not bool(jnp.all(jnp.isfinite(scores))):
            raise FloatingPointError("label_propagation produced nonfinite class scores.")
        if float(change) <= tol:
            converged = True
            break
    raw_scores = scores
    row_sums = jnp.sum(raw_scores, axis=1, keepdims=True)
    scores = jnp.where(
        row_sums > 0.0,
        raw_scores / jnp.where(row_sums > 0.0, row_sums, 1.0),
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
            "alpha": alpha,
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
    target_values = as_real_array(targets, name="targets")
    if target_values.shape != (adapted.n_samples,):
        raise ValueError(f"targets must have shape ({adapted.n_samples},).")
    if labeled_mask is None:
        mask = jnp.isfinite(target_values)
    else:
        raw_mask = jnp.asarray(labeled_mask)
        if raw_mask.dtype != jnp.bool_:
            raise TypeError("labeled_mask must contain booleans.")
        mask = raw_mask
        if mask.shape != (adapted.n_samples,):
            raise ValueError(f"labeled_mask must have shape ({adapted.n_samples},).")
        if not bool(jnp.all(jnp.isfinite(target_values[mask]))):
            raise ValueError("labeled targets must be finite.")
    if not bool(jnp.any(mask)) or bool(jnp.all(mask)):
        raise ValueError("targets must identify both labeled and unlabeled observations.")
    ambient_regularization = positive_control(
        ambient_regularization,
        name="ambient_regularization",
    )
    intrinsic_regularization = nonnegative_control(
        intrinsic_regularization,
        name="intrinsic_regularization",
    )
    observed = jnp.where(mask, target_values, 0.0)
    label_matrix = jnp.diag(mask.astype(float))
    degree = jnp.diag(jnp.sum(affinity, axis=1))
    laplacian = degree - affinity
    system = (
        label_matrix
        + ambient_regularization * jnp.eye(adapted.n_samples)
        + intrinsic_regularization * laplacian
    )
    predictions = jnp.linalg.solve(system, observed)
    if not bool(jnp.all(jnp.isfinite(predictions))):
        raise FloatingPointError(
            "manifold_regularized_regression produced a nonfinite linear-system solution."
        )
    linear_residual = system @ predictions - observed
    denominator = jnp.linalg.norm(system, ord=jnp.inf) * jnp.linalg.norm(
        predictions, ord=jnp.inf
    ) + jnp.linalg.norm(observed, ord=jnp.inf)
    denominator = jnp.maximum(denominator, jnp.finfo(system.dtype).tiny)
    relative_backward_error = (
        jnp.linalg.norm(
            linear_residual,
            ord=jnp.inf,
        )
        / denominator
    )
    backward_tolerance = 100.0 * jnp.finfo(system.dtype).eps * adapted.n_samples
    if not bool(jnp.isfinite(relative_backward_error)) or float(relative_backward_error) > float(
        backward_tolerance
    ):
        raise FloatingPointError(
            "manifold_regularized_regression failed its linear-system backward-error check."
        )
    residual = jnp.where(mask, predictions - observed, 0.0)
    objective = (
        jnp.sum(residual**2)
        + ambient_regularization * jnp.sum(predictions**2)
        + intrinsic_regularization * predictions @ laplacian @ predictions
    )
    return SemiSupervisedResult(
        predictions=predictions,
        scores=predictions,
        objective=objective,
        iterations=1,
        converged=True,
        reason="linear system solved",
        diagnostics={
            "labeled_mask": mask,
            "affinity": affinity,
            "laplacian": laplacian,
            "distances": distances,
            "bandwidth": scale,
            "system_matrix": system,
            "linear_residual": linear_residual,
            "relative_backward_error": relative_backward_error,
            "backward_error_tolerance": backward_tolerance,
        },
    )


__all__ = ["label_propagation", "manifold_regularized_regression"]
