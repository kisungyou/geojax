"""Bootstrap and permutation procedures for manifold-valued samples."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._inference import _combine, _validate_permutations
from ._results import BootstrapResult, HypothesisTestResult
from ._statistics import frechet_mean
from ._utils import (
    as_key,
    normalize_weights,
    require_unbatched,
    stack_points,
    take_samples,
    weighted_tangent_sum,
)


def bootstrap_frechet_mean(
    manifold: Any,
    data: Any,
    *,
    sample_weight: Any | None = None,
    n_bootstrap: int = 999,
    confidence_level: float = 0.95,
    key: Any | int | None,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> BootstrapResult:
    """Bootstrap an intrinsic mean and return a geodesic confidence ball."""
    require_exact_operations(manifold, "bootstrap_frechet_mean", "dist", "log", "exp")
    adapted = data if isinstance(data, ManifoldData) else as_manifold_data(manifold, data)
    require_unbatched(adapted, "bootstrap_frechet_mean")
    count = int(n_bootstrap)
    if count < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    estimate = frechet_mean(
        manifold,
        adapted,
        sample_weight=weights,
        maxiter=int(maxiter),
        tol=float(tol),
    )
    keys = jax.random.split(as_key(key, "bootstrap_frechet_mean"), count)
    replicates = []
    indices_history = []
    for bootstrap_key in keys:
        indices = jax.random.choice(
            bootstrap_key,
            adapted.n_samples,
            shape=(adapted.n_samples,),
            replace=True,
            p=weights,
        )
        resample = as_manifold_data(
            manifold,
            take_samples(manifold, adapted.values, indices),
        )
        replicate = frechet_mean(
            manifold,
            resample,
            initial_point=estimate.point,
            maxiter=int(maxiter),
            tol=float(tol),
        ).point
        replicates.append(replicate)
        indices_history.append(indices)
    replicate_tree = stack_points(manifold, replicates)
    distances = manifold.dist(estimate.point, replicate_tree)
    radius = jnp.quantile(distances, float(confidence_level))
    return BootstrapResult(
        estimate=estimate.point,
        replicates=replicate_tree,
        confidence_radius=radius,
        confidence_level=float(confidence_level),
        diagnostics={
            "bootstrap_indices": jnp.stack(indices_history),
            "replicate_distances": distances,
            "original_fit": estimate,
        },
    )


def _energy_statistic(distances: Any, left: Any, right: Any) -> Any:
    cross = distances[left[:, None], right[None, :]]
    within_left = distances[left[:, None], left[None, :]]
    within_right = distances[right[:, None], right[None, :]]
    return jnp.maximum(
        2.0 * jnp.mean(cross) - jnp.mean(within_left) - jnp.mean(within_right),
        0.0,
    )


def energy_two_sample_test(
    manifold: Any,
    x: Any,
    y: Any,
    *,
    n_permutations: int = 999,
    key: Any | int | None,
) -> HypothesisTestResult:
    """Run the metric energy-distance two-sample permutation test."""
    require_exact_operations(manifold, "energy_two_sample_test", "dist")
    left = x if isinstance(x, ManifoldData) else as_manifold_data(manifold, x)
    right = y if isinstance(y, ManifoldData) else as_manifold_data(manifold, y)
    require_unbatched(left, "energy_two_sample_test")
    require_unbatched(right, "energy_two_sample_test")
    if left.n_samples < 2 or right.n_samples < 2:
        raise ValueError("Each sample must contain at least two observations.")
    count = _validate_permutations(n_permutations)
    pooled = _combine(manifold, left, right)
    distances = pairwise_distances(manifold, pooled)
    observed_left = jnp.arange(left.n_samples)
    observed_right = jnp.arange(left.n_samples, pooled.n_samples)
    observed = _energy_statistic(distances, observed_left, observed_right)
    keys = jax.random.split(as_key(key, "energy_two_sample_test"), count)
    null = []
    for permutation_key in keys:
        order = jax.random.permutation(permutation_key, pooled.n_samples)
        null.append(
            _energy_statistic(
                distances,
                order[: left.n_samples],
                order[left.n_samples :],
            )
        )
    null_distribution = jnp.asarray(null)
    pvalue = (1.0 + jnp.sum(null_distribution >= observed)) / (count + 1.0)
    return HypothesisTestResult(
        statistic=observed,
        pvalue=pvalue,
        null_distribution=null_distribution,
        method="Metric energy-distance two-sample permutation test",
        diagnostics={"pairwise_distances": distances},
    )


def _mmd_statistic(kernel_matrix: Any, left: Any, right: Any) -> Any:
    left_kernel = kernel_matrix[left[:, None], left[None, :]]
    right_kernel = kernel_matrix[right[:, None], right[None, :]]
    cross_kernel = kernel_matrix[left[:, None], right[None, :]]
    return jnp.maximum(
        jnp.mean(left_kernel) + jnp.mean(right_kernel) - 2.0 * jnp.mean(cross_kernel),
        0.0,
    )


def kernel_mmd_two_sample_test(
    manifold: Any,
    x: Any,
    y: Any,
    *,
    bandwidth: float | None = None,
    kernel: Callable[[Any], Any] | None = None,
    check_psd: bool = True,
    psd_tolerance: float = 1e-8,
    n_permutations: int = 999,
    key: Any | int | None,
) -> HypothesisTestResult:
    """Run a finite-sample PSD-kernel maximum mean discrepancy test."""
    require_exact_operations(manifold, "kernel_mmd_two_sample_test", "dist")
    left = x if isinstance(x, ManifoldData) else as_manifold_data(manifold, x)
    right = y if isinstance(y, ManifoldData) else as_manifold_data(manifold, y)
    require_unbatched(left, "kernel_mmd_two_sample_test")
    require_unbatched(right, "kernel_mmd_two_sample_test")
    if left.n_samples < 2 or right.n_samples < 2:
        raise ValueError("Each sample must contain at least two observations.")
    count = _validate_permutations(n_permutations)
    pooled = _combine(manifold, left, right)
    distances = pairwise_distances(manifold, pooled)
    if psd_tolerance < 0.0:
        raise ValueError("psd_tolerance must be nonnegative.")
    if kernel is None:
        positive = distances[distances > 0.0]
        scale = float(jnp.median(positive)) if bandwidth is None and positive.size else bandwidth
        scale = 1.0 if scale is None else float(scale)
        if scale <= 0.0:
            raise ValueError("bandwidth must be positive.")
        kernel_matrix = jnp.exp(-(distances**2) / (2.0 * scale**2))
    else:
        scale = None
        kernel_matrix = jnp.asarray(kernel(distances))
        if kernel_matrix.shape != distances.shape:
            raise ValueError("kernel must return a square matrix matching pairwise distances.")
    kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.T)
    eigenvalues = jnp.linalg.eigvalsh(kernel_matrix)
    spectral_scale = max(float(jnp.max(jnp.abs(eigenvalues))), 1.0)
    backward_error = (
        float(jnp.finfo(kernel_matrix.dtype).eps)
        * kernel_matrix.shape[0]
        * spectral_scale
    )
    effective_psd_tolerance = max(float(psd_tolerance), backward_error)
    if check_psd and float(jnp.min(eigenvalues)) < -effective_psd_tolerance:
        raise ValueError(
            "The observed kernel Gram matrix is not positive semidefinite; "
            "supply a valid kernel or set check_psd=False for exploratory use."
        )
    observed_left = jnp.arange(left.n_samples)
    observed_right = jnp.arange(left.n_samples, pooled.n_samples)
    observed = _mmd_statistic(kernel_matrix, observed_left, observed_right)
    keys = jax.random.split(as_key(key, "kernel_mmd_two_sample_test"), count)
    null = []
    for permutation_key in keys:
        order = jax.random.permutation(permutation_key, pooled.n_samples)
        null.append(
            _mmd_statistic(
                kernel_matrix,
                order[: left.n_samples],
                order[left.n_samples :],
            )
        )
    null_distribution = jnp.asarray(null)
    pvalue = (1.0 + jnp.sum(null_distribution >= observed)) / (count + 1.0)
    return HypothesisTestResult(
        statistic=observed,
        pvalue=pvalue,
        null_distribution=null_distribution,
        method="RBF-distance-kernel MMD permutation test",
        diagnostics={
            "bandwidth": scale,
            "kernel_eigenvalues": eigenvalues,
            "requested_psd_tolerance": float(psd_tolerance),
            "effective_psd_tolerance": effective_psd_tolerance,
            "kernel_matrix": kernel_matrix,
            "pairwise_distances": distances,
        },
    )


def paired_frechet_test(
    manifold: Any,
    x: Any,
    y: Any,
    *,
    n_permutations: int = 999,
    key: Any | int | None,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> HypothesisTestResult:
    """Test a zero mean paired displacement by within-pair random sign flips."""
    require_exact_operations(manifold, "paired_frechet_test", "dist", "log", "exp")
    left = x if isinstance(x, ManifoldData) else as_manifold_data(manifold, x)
    right = y if isinstance(y, ManifoldData) else as_manifold_data(manifold, y)
    require_unbatched(left, "paired_frechet_test")
    require_unbatched(right, "paired_frechet_test")
    if left.n_samples != right.n_samples or left.n_samples < 2:
        raise ValueError("paired samples must have the same size of at least two.")
    count = _validate_permutations(n_permutations)
    pooled = _combine(manifold, left, right)
    base_fit = frechet_mean(manifold, pooled, maxiter=int(maxiter), tol=float(tol))
    left_logs = manifold.log(base_fit.point, left.values)
    right_logs = manifold.log(base_fit.point, right.values)
    differences = manifold.lincomb(base_fit.point, 1.0, right_logs, -1.0, left_logs)
    mean_difference = weighted_tangent_sum(
        manifold,
        differences,
        jnp.full((left.n_samples,), 1.0 / left.n_samples),
    )
    observed = manifold.norm(base_fit.point, mean_difference)
    keys = jax.random.split(as_key(key, "paired_frechet_test"), count)
    null = []
    for permutation_key in keys:
        signs = 2.0 * jax.random.bernoulli(
            permutation_key,
            shape=(left.n_samples,),
        ).astype(float) - 1.0
        permuted_mean = weighted_tangent_sum(
            manifold,
            differences,
            signs / left.n_samples,
        )
        null.append(manifold.norm(base_fit.point, permuted_mean))
    null_distribution = jnp.asarray(null)
    pvalue = (1.0 + jnp.sum(null_distribution >= observed)) / (count + 1.0)
    return HypothesisTestResult(
        statistic=observed,
        pvalue=pvalue,
        null_distribution=null_distribution,
        method="Paired tangent-displacement sign-flip test",
        diagnostics={
            "base_point": base_fit.point,
            "paired_differences": differences,
            "base_fit": base_fit,
        },
    )


__all__ = [
    "bootstrap_frechet_mean",
    "energy_two_sample_test",
    "kernel_mmd_two_sample_test",
    "paired_frechet_test",
]
