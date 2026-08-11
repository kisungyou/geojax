"""Metric-orthonormal tangent coordinates for manifold-valued learning."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._data import as_manifold_data
from ._results import TangentFeatureMap
from ._statistics import frechet_mean
from ._utils import (
    deterministic_sign_columns,
    integer_control,
    nonnegative_control,
    require_unbatched,
    stack_points,
    take_point,
    tree_all_finite,
    weighted_tangent_sum,
)


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
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "fit_tangent_feature_map")
    rank_tolerance = nonnegative_control(rank_tolerance, name="rank_tolerance")
    mean_maxiter = integer_control(mean_maxiter, name="mean_maxiter", minimum=1)
    mean_tol = nonnegative_control(mean_tol, name="mean_tol")
    if n_components is not None:
        n_components = integer_control(n_components, name="n_components", minimum=1)

    if base_point is None:
        require_exact_operations(manifold, "fit_tangent_feature_map", "dist", "exp")
        mean_result = frechet_mean(
            manifold,
            adapted,
            maxiter=mean_maxiter,
            tol=mean_tol,
        )
        base = mean_result.point
    else:
        mean_result = None
        base_data = as_manifold_data(
            manifold,
            stack_points(manifold, [base_point]),
        )
        base = take_point(manifold, base_data.values, 0)

    batched_logs = manifold.log(base, adapted.values)
    if not all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(batched_logs)
    ):
        raise ValueError(
            "fit_tangent_feature_map encountered an undefined logarithm; "
            "the data may meet the cut locus of the selected base point."
        )
    logs = [take_point(manifold, batched_logs, index) for index in range(adapted.n_samples)]
    raw_gram = _tangent_gram(manifold, base, logs)
    gram = 0.5 * (raw_gram + raw_gram.T)
    if not bool(jnp.all(jnp.isfinite(gram))):
        raise FloatingPointError("The tangent metric Gram matrix is nonfinite.")
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    order = jnp.argsort(eigenvalues)[::-1]
    raw_eigenvalues = eigenvalues[order]
    eigenvectors = deterministic_sign_columns(eigenvectors[:, order])
    spectral_scale = max(
        float(jnp.max(jnp.abs(raw_eigenvalues))),
        float(jnp.finfo(gram.dtype).tiny),
    )
    backward_error = (
        100.0 * float(jnp.finfo(gram.dtype).eps) * max(gram.shape, default=1) * spectral_scale
    )
    if float(jnp.min(raw_eigenvalues)) < -backward_error:
        raise FloatingPointError(
            "The tangent metric Gram matrix is not positive semidefinite within "
            "floating-point backward error."
        )
    eigenvalues = jnp.maximum(raw_eigenvalues, 0.0)
    leading = float(eigenvalues[0]) if eigenvalues.size else 0.0
    threshold = max(rank_tolerance * leading, backward_error)
    rank = int(jnp.sum(eigenvalues > threshold))
    if n_components is not None and n_components > rank:
        raise ValueError(
            f"n_components={n_components} exceeds the tangent-data numerical rank {rank}."
        )
    requested = rank if n_components is None else n_components

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
        converged=mean_result is None or mean_result.converged,
        reason=(
            "user-supplied base point used"
            if mean_result is None
            else (
                "reference mean converged"
                if mean_result.converged
                else "reference mean fit was incomplete"
            )
        ),
        diagnostics={
            "gram_matrix": gram,
            "numerical_rank": rank,
            "rank_tolerance": rank_tolerance,
            "effective_rank_threshold": threshold,
            "mean_result": mean_result,
        },
    )
    return feature_map, transform_tangent_features(feature_map, adapted)


def transform_tangent_features(feature_map: TangentFeatureMap, data: Any) -> Any:
    """Map manifold observations to metric-orthonormal tangent coordinates."""
    manifold = feature_map.manifold
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "TangentFeatureMap.transform")
    if not feature_map.basis:
        return jnp.zeros((adapted.n_samples, 0))
    logs = manifold.log(feature_map.base_point, adapted.values)
    if not tree_all_finite(logs):
        raise ValueError(
            "TangentFeatureMap.transform encountered an undefined logarithm; "
            "the data may meet the cut locus of the fitted base point."
        )
    columns = [manifold.inner(feature_map.base_point, logs, basis) for basis in feature_map.basis]
    features = jnp.stack(columns, axis=-1)
    if not bool(jnp.all(jnp.isfinite(features))):
        raise FloatingPointError("Tangent feature coordinates are nonfinite.")
    return features


__all__ = ["fit_tangent_feature_map", "transform_tangent_features"]
