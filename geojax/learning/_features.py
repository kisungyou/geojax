"""Metric-orthonormal tangent coordinates for manifold-valued learning."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._results import TangentFeatureMap
from ._statistics import frechet_mean
from ._utils import require_unbatched, stack_points, take_point, weighted_tangent_sum


def _tangent_gram(manifold: Any, base_point: Any, tangents: list[Any]) -> Any:
    rows = []
    for left in tangents:
        rows.append(jnp.stack([manifold.inner(base_point, left, right) for right in tangents]))
    return jnp.stack(rows)


def fit_tangent_feature_map(
    manifold: Any,
    data: Any,
    *,
    base_point: Any | None = None,
    n_components: int | None = None,
    rank_tolerance: float = 1e-8,
    mean_maxiter: int = 200,
    mean_tol: float = 1e-7,
) -> tuple[TangentFeatureMap, Any]:
    """Fit intrinsic, metric-orthonormal coordinates at one reference point."""
    require_exact_operations(manifold, "fit_tangent_feature_map", "log")
    adapted = data if isinstance(data, ManifoldData) else as_manifold_data(manifold, data)
    require_unbatched(adapted, "fit_tangent_feature_map")
    if float(rank_tolerance) < 0.0:
        raise ValueError("rank_tolerance must be nonnegative.")
    if n_components is not None and int(n_components) < 1:
        raise ValueError("n_components must be positive when supplied.")

    if base_point is None:
        require_exact_operations(manifold, "fit_tangent_feature_map", "dist", "exp")
        mean_result = frechet_mean(
            manifold,
            adapted,
            maxiter=int(mean_maxiter),
            tol=float(mean_tol),
        )
        base = mean_result.point
    else:
        mean_result = None
        base = manifold.project(base_point)

    batched_logs = manifold.log(base, adapted.values)
    logs = [take_point(manifold, batched_logs, index) for index in range(adapted.n_samples)]
    raw_gram = _tangent_gram(manifold, base, logs)
    gram = 0.5 * (raw_gram + raw_gram.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = jnp.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    leading = float(eigenvalues[0]) if eigenvalues.size else 0.0
    spectral_scale = max(leading, 1.0)
    backward_error = (
        float(jnp.finfo(gram.dtype).eps) * max(gram.shape, default=1) * spectral_scale
    )
    threshold = max(float(rank_tolerance) * spectral_scale, backward_error)
    rank = int(jnp.sum(eigenvalues > threshold))
    requested = rank if n_components is None else min(int(n_components), rank)

    stacked_logs = stack_points(manifold, logs)
    basis = []
    for component in range(requested):
        weights = eigenvectors[:, component] / jnp.sqrt(eigenvalues[component])
        vector = weighted_tangent_sum(manifold, stacked_logs, weights)
        basis.append(manifold.tangent_project(base, vector))
    feature_map = TangentFeatureMap(
        manifold=manifold,
        base_point=base,
        basis=tuple(basis),
        eigenvalues=eigenvalues[:requested],
        diagnostics={
            "gram_matrix": gram,
            "numerical_rank": rank,
            "rank_tolerance": float(rank_tolerance),
            "effective_rank_threshold": threshold,
            "mean_result": mean_result,
        },
    )
    return feature_map, transform_tangent_features(feature_map, adapted)


def transform_tangent_features(feature_map: TangentFeatureMap, data: Any) -> Any:
    """Map manifold observations to metric-orthonormal tangent coordinates."""
    manifold = feature_map.manifold
    adapted = data if isinstance(data, ManifoldData) else as_manifold_data(manifold, data)
    require_unbatched(adapted, "TangentFeatureMap.transform")
    if not feature_map.basis:
        return jnp.zeros((adapted.n_samples, 0))
    logs = manifold.log(feature_map.base_point, adapted.values)
    columns = [manifold.inner(feature_map.base_point, logs, basis) for basis in feature_map.basis]
    return jnp.stack(columns, axis=-1)


__all__ = ["fit_tangent_feature_map", "transform_tangent_features"]
