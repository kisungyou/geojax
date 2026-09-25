"""Permutation and asymptotic inference for manifold-valued samples."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaincc

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import HypothesisTestResult
from ._statistics import frechet_mean
from ._transport import empirical_wasserstein_distance
from ._utils import (
    as_key,
    integer_control,
    nonnegative_control,
    positive_control,
    require_unbatched,
    stack_points,
    take_point,
    take_samples,
)


def _combine(manifold: Any, left: ManifoldData, right: ManifoldData) -> ManifoldData:
    points = [take_point(manifold, left.values, index) for index in range(left.n_samples)]
    points.extend(take_point(manifold, right.values, index) for index in range(right.n_samples))
    return as_manifold_data(manifold, stack_points(manifold, points))


def _validate_permutations(n_permutations: int) -> int:
    count = integer_control(n_permutations, name="n_permutations")
    if count < 1:
        raise ValueError("n_permutations must be positive.")
    return count


def _require_converged_mean(result: Any, *, context: str) -> None:
    if not result.converged:
        raise RuntimeError(
            f"The {context} Frechet-mean fit did not converge; the resulting "
            "inference statistic is not certified. Increase maxiter or inspect "
            "the sample's geometric concentration."
        )


def _encode_groups(groups: Any, n_samples: int) -> tuple[Any, Any, Any]:
    values = np.asarray(groups)
    if values.shape != (n_samples,):
        raise ValueError(f"groups must have shape ({n_samples},); received {values.shape}.")
    if values.dtype.kind in {"f", "c"} and not np.all(np.isfinite(values)):
        raise ValueError("groups must not contain NaN or infinite values.")
    if values.dtype.kind in {"O", "U", "S"}:
        if any(
            value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
            for value in values.tolist()
        ):
            raise ValueError("groups must not contain missing values.")
    try:
        labels, encoded, counts = np.unique(
            values,
            return_inverse=True,
            return_counts=True,
        )
    except TypeError as exc:
        raise TypeError("groups must contain mutually comparable scalar labels.") from exc
    return labels, jnp.asarray(encoded, dtype=int), jnp.asarray(counts, dtype=int)


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
    left = as_manifold_data(manifold, x)
    right = as_manifold_data(manifold, y)
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
    if not bool(jnp.isfinite(observed)) or not bool(jnp.all(jnp.isfinite(null_distribution))):
        raise FloatingPointError(
            "biswas_ghosh_two_sample_test produced a nonfinite statistic; rescale the observations."
        )
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
    pooled_distance_scale: Any | None = None,
    pooled_variance_normalized: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    labels = jnp.unique(groups)
    n_samples = data.n_samples
    pooled_mean_result = None
    if pooled_distance_scale is None or pooled_variance_normalized is None:
        pooled_mean_result = frechet_mean(manifold, data, maxiter=maxiter, tol=tol)
        _require_converged_mean(pooled_mean_result, context="pooled")
        pooled_distances = manifold.dist(pooled_mean_result.point, data.values)
        if not bool(jnp.all(jnp.isfinite(pooled_distances))):
            raise FloatingPointError("The pooled Frechet distances are nonfinite.")
        pooled_distance_scale = jnp.max(pooled_distances)
        safe_distance_scale = jnp.where(
            pooled_distance_scale > 0.0,
            pooled_distance_scale,
            jnp.ones_like(pooled_distance_scale),
        )
        pooled_variance_normalized = jnp.mean((pooled_distances / safe_distance_scale) ** 2)
    safe_distance_scale = jnp.where(
        pooled_distance_scale > 0.0,
        pooled_distance_scale,
        jnp.ones_like(pooled_distance_scale),
    )
    variance_dtype = jnp.asarray(pooled_variance_normalized).dtype
    effective_variance_floor_normalized = jnp.maximum(
        variance_floor * pooled_variance_normalized**2,
        jnp.finfo(variance_dtype).tiny,
    )
    sizes = []
    variances = []
    variance_estimators = []
    means = []
    mean_results = []
    for label in labels.tolist():
        indices = jnp.nonzero(groups == label, size=n_samples, fill_value=-1)[0]
        indices = indices[indices >= 0]
        subset = as_manifold_data(manifold, take_samples(manifold, data.values, indices))
        mean_result = frechet_mean(manifold, subset, maxiter=maxiter, tol=tol)
        _require_converged_mean(
            mean_result,
            context=f"group {label!r}",
        )
        distances = manifold.dist(mean_result.point, subset.values)
        if not bool(jnp.all(jnp.isfinite(distances))):
            raise FloatingPointError(f"The Frechet distances for group {label!r} are nonfinite.")
        normalized_squared_distances = (distances / safe_distance_scale) ** 2
        variance_normalized = jnp.mean(normalized_squared_distances)
        sigma2_normalized = jnp.maximum(
            jnp.mean(normalized_squared_distances**2) - variance_normalized**2,
            effective_variance_floor_normalized,
        )
        sizes.append(indices.size)
        variances.append(variance_normalized)
        variance_estimators.append(sigma2_normalized)
        means.append(mean_result.point)
        mean_results.append(mean_result)
    sizes_array = jnp.asarray(sizes, dtype=float)
    variances_array = jnp.asarray(variances)
    sigma_array = jnp.asarray(variance_estimators)
    proportions = sizes_array / n_samples
    mean_component_normalized = pooled_variance_normalized - jnp.sum(proportions * variances_array)
    sigma_scale = jnp.max(sigma_array)
    scaled_sigma = sigma_array / sigma_scale
    scaled_variance_component = 0.0
    for left in range(labels.size - 1):
        for right in range(left + 1, labels.size):
            scaled_variance_component = scaled_variance_component + (
                proportions[left]
                * proportions[right]
                / (scaled_sigma[left] * scaled_sigma[right])
                * (variances_array[left] - variances_array[right]) ** 2
                / sigma_scale
            )
    variance_component_normalized = scaled_variance_component / sigma_scale
    # Dubey--Mueller's statistic is
    # n U / sum_j(gamma_j / sigma_j^2)
    #   + n F^2 / sum_j(gamma_j^2 sigma_j^2).
    # Keep the proportions explicit: using raw group sizes in either
    # denominator silently removes powers of n and destroys the chi-square
    # calibration of the asymptotic test.
    term_variance = n_samples * scaled_variance_component / jnp.sum(proportions / scaled_sigma)
    term_mean = (
        n_samples
        * (mean_component_normalized**2 / sigma_scale)
        / jnp.sum(proportions**2 * scaled_sigma)
    )
    statistic = term_variance + term_mean
    if not bool(jnp.isfinite(statistic)):
        raise FloatingPointError("Fréchet ANOVA produced a nonfinite statistic.")

    distance_scale_squared = pooled_distance_scale**2
    variance_scale = distance_scale_squared**2
    pooled_variance = distance_scale_squared * pooled_variance_normalized
    group_variances = distance_scale_squared * variances_array
    variance_estimators_physical = variance_scale * sigma_array
    mean_component = distance_scale_squared * mean_component_normalized
    variance_component = jnp.where(
        variance_scale > 0.0,
        variance_component_normalized / variance_scale,
        0.0,
    )
    return statistic, {
        "labels": labels,
        "sizes": sizes_array,
        "group_variances": group_variances,
        "variance_estimators": variance_estimators_physical,
        "normalized_group_variances": variances_array,
        "normalized_variance_estimators": sigma_array,
        "relative_variance_floor": variance_floor,
        "effective_variance_floor": variance_scale * effective_variance_floor_normalized,
        "effective_variance_floor_normalized": effective_variance_floor_normalized,
        "distance_scale": pooled_distance_scale,
        "pooled_variance": pooled_variance,
        "pooled_variance_normalized": pooled_variance_normalized,
        "pooled_mean_result": pooled_mean_result,
        "mean_component": mean_component,
        "mean_component_normalized": mean_component_normalized,
        "variance_component": variance_component,
        "variance_component_normalized": variance_component_normalized,
        "term_variance": term_variance,
        "term_mean": term_mean,
        "group_means": tuple(means),
        "group_mean_results": tuple(mean_results),
        "unconverged_mean_fits": sum(not result.converged for result in mean_results)
        + (pooled_mean_result is not None and not pooled_mean_result.converged),
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
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "frechet_anova")
    labels, group_values, counts = _encode_groups(groups, adapted.n_samples)
    if labels.size < 2 or bool(jnp.any(counts < 2)):
        raise ValueError("FANOVA requires at least two groups with at least two observations each.")
    if not isinstance(method, str) or method not in {"asymptotic", "permutation"}:
        raise ValueError("method must be 'asymptotic' or 'permutation'.")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    variance_floor = positive_control(variance_floor, name="variance_floor")
    observed, diagnostics = _fanova_statistic(
        manifold,
        adapted,
        group_values,
        maxiter=maxiter,
        tol=tol,
        variance_floor=variance_floor,
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
                maxiter=maxiter,
                tol=tol,
                variance_floor=variance_floor,
                pooled_distance_scale=diagnostics["distance_scale"],
                pooled_variance_normalized=diagnostics["pooled_variance_normalized"],
            )
            null.append(statistic)
        null_distribution = jnp.asarray(null)
        pvalue = (1.0 + jnp.sum(null_distribution >= observed)) / (count + 1.0)
    diagnostics = {**diagnostics, "labels": labels, "encoded_groups": group_values}
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
    left = as_manifold_data(manifold, x)
    right = as_manifold_data(manifold, y)
    require_unbatched(left, "wasserstein_two_sample_test")
    require_unbatched(right, "wasserstein_two_sample_test")
    p = positive_control(p, name="p")
    if p < 1.0:
        raise ValueError("p must be at least 1.")
    tolerance = positive_control(tolerance, name="tolerance")
    count = _validate_permutations(n_permutations)
    observed_result = empirical_wasserstein_distance(
        manifold, left, right, p=p, tolerance=tolerance
    )
    if not observed_result.converged:
        raise RuntimeError("The observed exact transport problem did not converge.")
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
        permuted_result = empirical_wasserstein_distance(
            manifold,
            permuted_left,
            permuted_right,
            p=p,
            tolerance=tolerance,
        )
        if not permuted_result.converged:
            raise RuntimeError("An exact transport permutation problem did not converge.")
        null.append(permuted_result.distance)
    null_distribution = jnp.asarray(null)
    pvalue = (1.0 + jnp.sum(null_distribution >= observed_result.distance)) / (count + 1.0)
    return HypothesisTestResult(
        statistic=observed_result.distance,
        pvalue=pvalue,
        null_distribution=null_distribution,
        method="Exact Wasserstein two-sample permutation test",
        diagnostics={"observed_transport": observed_result, "p": p},
    )


__all__ = [
    "biswas_ghosh_two_sample_test",
    "frechet_anova",
    "wasserstein_two_sample_test",
]
