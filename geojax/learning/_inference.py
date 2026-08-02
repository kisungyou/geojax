"""Permutation and asymptotic inference for manifold-valued samples."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaincc

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import HypothesisTestResult
from ._statistics import frechet_mean
from ._transport import empirical_wasserstein_distance
from ._utils import as_key, require_unbatched, stack_points, take_point, take_samples


def _combine(manifold: Any, left: ManifoldData, right: ManifoldData) -> ManifoldData:
    points = [take_point(manifold, left.values, index) for index in range(left.n_samples)]
    points.extend(take_point(manifold, right.values, index) for index in range(right.n_samples))
    return as_manifold_data(manifold, stack_points(manifold, points))


def _validate_permutations(n_permutations: int) -> int:
    count = int(n_permutations)
    if count < 1:
        raise ValueError("n_permutations must be positive.")
    return count


def _bg_statistic(distances: Any, left_indices: Any, right_indices: Any) -> Any:
    left_size, right_size = left_indices.size, right_indices.size
    left_block = distances[left_indices[:, None], left_indices[None, :]]
    right_block = distances[right_indices[:, None], right_indices[None, :]]
    cross = distances[left_indices[:, None], right_indices[None, :]]
    left_mean = jnp.sum(jnp.triu(left_block, k=1)) / (left_size * (left_size - 1) / 2)
    right_mean = jnp.sum(jnp.triu(right_block, k=1)) / (right_size * (right_size - 1) / 2)
    cross_mean = jnp.mean(cross)
    return (left_mean - cross_mean) ** 2 + (cross_mean - right_mean) ** 2


def biswas_ghosh_two_sample_test(
    manifold: Any,
    x: Any,
    y: Any,
    *,
    n_permutations: int = 999,
    key: Any | int | None,
) -> HypothesisTestResult:
    """Run the metric-space modification of the Biswas-Ghosh two-sample test."""
    require_exact_operations(manifold, "biswas_ghosh_two_sample_test", "dist")
    left = x if isinstance(x, ManifoldData) else as_manifold_data(manifold, x)
    right = y if isinstance(y, ManifoldData) else as_manifold_data(manifold, y)
    require_unbatched(left, "biswas_ghosh_two_sample_test")
    require_unbatched(right, "biswas_ghosh_two_sample_test")
    if left.n_samples < 2 or right.n_samples < 2:
        raise ValueError("Each sample must contain at least two observations.")
    count = _validate_permutations(n_permutations)
    pooled = _combine(manifold, left, right)
    distances = pairwise_distances(manifold, pooled)
    observed_left = jnp.arange(left.n_samples)
    observed_right = jnp.arange(left.n_samples, pooled.n_samples)
    observed = _bg_statistic(distances, observed_left, observed_right)
    keys = jax.random.split(as_key(key, "biswas_ghosh_two_sample_test"), count)
    null = []
    for permutation_key in keys:
        order = jax.random.permutation(permutation_key, pooled.n_samples)
        null.append(_bg_statistic(distances, order[: left.n_samples], order[left.n_samples :]))
    null_distribution = jnp.asarray(null)
    pvalue = (1.0 + jnp.sum(null_distribution >= observed)) / (count + 1.0)
    return HypothesisTestResult(
        statistic=observed,
        pvalue=pvalue,
        null_distribution=null_distribution,
        method="Biswas-Ghosh metric two-sample permutation test",
        diagnostics={"pairwise_distances": distances},
    )


def _fanova_statistic(
    manifold: Any,
    data: ManifoldData,
    groups: Any,
    *,
    maxiter: int,
    tol: float,
    variance_floor: float,
    pooled_variance: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    labels = jnp.unique(groups)
    n_samples = data.n_samples
    if pooled_variance is None:
        pooled_mean = frechet_mean(manifold, data, maxiter=maxiter, tol=tol)
        pooled_distances = manifold.dist(pooled_mean.point, data.values)
        pooled_variance = jnp.mean(pooled_distances**2)
    sizes = []
    variances = []
    variance_estimators = []
    means = []
    for label in labels.tolist():
        indices = jnp.nonzero(groups == label, size=n_samples, fill_value=-1)[0]
        indices = indices[indices >= 0]
        subset = as_manifold_data(manifold, take_samples(manifold, data.values, indices))
        mean_result = frechet_mean(manifold, subset, maxiter=maxiter, tol=tol)
        distances = manifold.dist(mean_result.point, subset.values)
        variance = jnp.mean(distances**2)
        sigma2 = jnp.maximum(jnp.mean(distances**4) - variance**2, variance_floor)
        sizes.append(indices.size)
        variances.append(variance)
        variance_estimators.append(sigma2)
        means.append(mean_result.point)
    sizes_array = jnp.asarray(sizes, dtype=float)
    variances_array = jnp.asarray(variances)
    sigma_array = jnp.asarray(variance_estimators)
    proportions = sizes_array / n_samples
    mean_component = pooled_variance - jnp.sum(proportions * variances_array)
    variance_component = 0.0
    for left in range(labels.size - 1):
        for right in range(left + 1, labels.size):
            variance_component = variance_component + (
                proportions[left]
                * proportions[right]
                / (sigma_array[left] * sigma_array[right])
                * (variances_array[left] - variances_array[right]) ** 2
            )
    term_variance = n_samples * variance_component / jnp.sum(sizes_array / sigma_array)
    term_mean = n_samples * mean_component**2 / jnp.sum(sizes_array**2 * sigma_array)
    statistic = term_variance + term_mean
    return statistic, {
        "labels": labels,
        "sizes": sizes_array,
        "group_variances": variances_array,
        "variance_estimators": sigma_array,
        "pooled_variance": pooled_variance,
        "mean_component": mean_component,
        "variance_component": variance_component,
        "group_means": tuple(means),
    }


def frechet_anova(
    manifold: Any,
    data: Any,
    groups: Any,
    *,
    method: str = "asymptotic",
    n_permutations: int = 999,
    key: Any | int | None = None,
    maxiter: int = 100,
    tol: float = 1e-6,
    variance_floor: float = 1e-12,
) -> HypothesisTestResult:
    """Test equality of metric-space populations using Dubey-Mueller FANOVA."""
    require_exact_operations(manifold, "frechet_anova", "dist", "log", "exp")
    adapted = data if isinstance(data, ManifoldData) else as_manifold_data(manifold, data)
    require_unbatched(adapted, "frechet_anova")
    group_values = jnp.asarray(groups)
    if group_values.shape != (adapted.n_samples,):
        raise ValueError(f"groups must have shape ({adapted.n_samples},).")
    labels, counts = jnp.unique(group_values, return_counts=True)
    if labels.size < 2 or bool(jnp.any(counts < 2)):
        raise ValueError("FANOVA requires at least two groups with at least two observations each.")
    if method not in {"asymptotic", "permutation"}:
        raise ValueError("method must be 'asymptotic' or 'permutation'.")
    observed, diagnostics = _fanova_statistic(
        manifold,
        adapted,
        group_values,
        maxiter=int(maxiter),
        tol=float(tol),
        variance_floor=float(variance_floor),
    )
    if method == "asymptotic":
        pvalue = gammaincc(0.5 * (labels.size - 1), 0.5 * observed)
        null_distribution = jnp.empty((0,))
    else:
        count = _validate_permutations(n_permutations)
        keys = jax.random.split(as_key(key, "frechet_anova"), count)
        null = []
        for permutation_key in keys:
            permuted = group_values[jax.random.permutation(permutation_key, adapted.n_samples)]
            statistic, _ = _fanova_statistic(
                manifold,
                adapted,
                permuted,
                maxiter=int(maxiter),
                tol=float(tol),
                variance_floor=float(variance_floor),
                pooled_variance=diagnostics["pooled_variance"],
            )
            null.append(statistic)
        null_distribution = jnp.asarray(null)
        pvalue = (1.0 + jnp.sum(null_distribution >= observed)) / (count + 1.0)
    return HypothesisTestResult(
        statistic=observed,
        pvalue=pvalue,
        null_distribution=null_distribution,
        method=f"Dubey-Mueller Frechet ANOVA ({method})",
        diagnostics=diagnostics,
    )


def wasserstein_two_sample_test(
    manifold: Any,
    x: Any,
    y: Any,
    *,
    p: float = 2.0,
    n_permutations: int = 999,
    key: Any | int | None,
    tolerance: float = 1e-10,
) -> HypothesisTestResult:
    """Permutation test using exact empirical Wasserstein distance."""
    require_exact_operations(manifold, "wasserstein_two_sample_test", "dist")
    left = x if isinstance(x, ManifoldData) else as_manifold_data(manifold, x)
    right = y if isinstance(y, ManifoldData) else as_manifold_data(manifold, y)
    require_unbatched(left, "wasserstein_two_sample_test")
    require_unbatched(right, "wasserstein_two_sample_test")
    count = _validate_permutations(n_permutations)
    observed_result = empirical_wasserstein_distance(
        manifold, left, right, p=p, tolerance=tolerance
    )
    pooled = _combine(manifold, left, right)
    keys = jax.random.split(as_key(key, "wasserstein_two_sample_test"), count)
    null = []
    for permutation_key in keys:
        order = jax.random.permutation(permutation_key, pooled.n_samples)
        permuted_left = as_manifold_data(
            manifold, take_samples(manifold, pooled.values, order[: left.n_samples])
        )
        permuted_right = as_manifold_data(
            manifold, take_samples(manifold, pooled.values, order[left.n_samples :])
        )
        null.append(
            empirical_wasserstein_distance(
                manifold,
                permuted_left,
                permuted_right,
                p=p,
                tolerance=tolerance,
            ).distance
        )
    null_distribution = jnp.asarray(null)
    pvalue = (1.0 + jnp.sum(null_distribution >= observed_result.distance)) / (count + 1.0)
    return HypothesisTestResult(
        statistic=observed_result.distance,
        pvalue=pvalue,
        null_distribution=null_distribution,
        method="Exact Wasserstein two-sample permutation test",
        diagnostics={"observed_transport": observed_result, "p": float(p)},
    )


__all__ = [
    "biswas_ghosh_two_sample_test",
    "frechet_anova",
    "wasserstein_two_sample_test",
]
