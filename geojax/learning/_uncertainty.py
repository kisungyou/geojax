"""Bootstrap and permutation procedures for manifold-valued samples."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._data import as_manifold_data
from ._geometry import pairwise_distances
from ._inference import _combine, _validate_permutations
from ._results import BootstrapResult, HypothesisTestResult
from ._statistics import frechet_mean
from ._utils import (
    as_key,
    integer_control,
    interval_control,
    nonnegative_control,
    normalize_weights,
    positive_control,
    require_unbatched,
    stack_points,
    take_samples,
    tree_all_finite,
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
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "bootstrap_frechet_mean")
    count = integer_control(n_bootstrap, name="n_bootstrap")
    if count < 1:
        raise ValueError("n_bootstrap must be positive.")
    confidence_level = interval_control(
        confidence_level,
        name="confidence_level",
        lower=0.0,
        upper=1.0,
        lower_closed=False,
        upper_closed=False,
    )
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    estimate = frechet_mean(
        manifold,
        adapted,
        sample_weight=weights,
        maxiter=maxiter,
        tol=tol,
    )
    if not estimate.converged:
        raise RuntimeError(
            "The original Fréchet-mean fit did not converge; a bootstrap "
            "confidence radius would not be trustworthy."
        )
    keys = jax.random.split(as_key(key, "bootstrap_frechet_mean"), count)
    replicates = []
    replicate_fits = []
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
        replicate_fit = frechet_mean(
            manifold,
            resample,
            initial_point=estimate.point,
            maxiter=maxiter,
            tol=tol,
        )
        if not replicate_fit.converged:
            raise RuntimeError(
                "A bootstrap Fréchet-mean fit did not converge; increase maxiter "
                "or inspect the sample's geometric concentration."
            )
        replicates.append(replicate_fit.point)
        replicate_fits.append(replicate_fit)
        indices_history.append(indices)
    replicate_tree = stack_points(manifold, replicates)
    distances = manifold.dist(estimate.point, replicate_tree)
    if not tree_all_finite(replicate_tree) or not bool(jnp.all(jnp.isfinite(distances))):
        raise FloatingPointError("bootstrap_frechet_mean produced nonfinite replicates.")
    radius = jnp.quantile(distances, confidence_level)
    return BootstrapResult(
        estimate=estimate.point,
        replicates=replicate_tree,
        confidence_radius=radius,
        confidence_level=confidence_level,
        diagnostics={
            "bootstrap_indices": jnp.stack(indices_history),
            "replicate_distances": distances,
            "original_fit": estimate,
            "replicate_fits": tuple(replicate_fits),
            "unconverged_replicates": sum(not fit.converged for fit in replicate_fits),
        },
    )


def _energy_statistic(distances: Any, left: Any, right: Any) -> Any:
    cross = distances[left[:, None], right[None, :]]
    within_left = distances[left[:, None], left[None, :]]
    within_right = distances[right[:, None], right[None, :]]
    return 2.0 * jnp.mean(cross) - jnp.mean(within_left) - jnp.mean(within_right)


def energy_two_sample_test(
    manifold: Any,
    x: Any,
    y: Any,
    *,
    n_permutations: int = 999,
    key: Any | int | None,
) -> HypothesisTestResult:
    """Run a biased metric energy-statistic permutation test.

    The statistic is guaranteed nonnegative at the population level only when
    the metric is of negative type. GeoJAX therefore preserves the signed
    finite-sample V-statistic instead of clipping it, which also preserves the
    exact permutation ordering for arbitrary manifold metrics.
    """
    require_exact_operations(manifold, "energy_two_sample_test", "dist")
    left = as_manifold_data(manifold, x)
    right = as_manifold_data(manifold, y)
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
    if not bool(jnp.isfinite(observed)) or not bool(jnp.all(jnp.isfinite(null_distribution))):
        raise FloatingPointError("energy_two_sample_test produced a nonfinite statistic.")
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
    return jnp.mean(left_kernel) + jnp.mean(right_kernel) - 2.0 * jnp.mean(cross_kernel)


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
    left = as_manifold_data(manifold, x)
    right = as_manifold_data(manifold, y)
    require_unbatched(left, "kernel_mmd_two_sample_test")
    require_unbatched(right, "kernel_mmd_two_sample_test")
    if left.n_samples < 2 or right.n_samples < 2:
        raise ValueError("Each sample must contain at least two observations.")
    count = _validate_permutations(n_permutations)
    pooled = _combine(manifold, left, right)
    distances = pairwise_distances(manifold, pooled)
    psd_tolerance = nonnegative_control(psd_tolerance, name="psd_tolerance")
    if not isinstance(check_psd, bool):
        raise TypeError("check_psd must be a boolean.")
    if kernel is None:
        positive = distances[distances > 0.0]
        scale = float(jnp.median(positive)) if bandwidth is None and positive.size else bandwidth
        scale = positive_control(1.0 if scale is None else scale, name="bandwidth")
        kernel_matrix = jnp.exp(-(distances**2) / (2.0 * scale**2))
    else:
        if not callable(kernel):
            raise TypeError("kernel must be callable or None.")
        if bandwidth is not None:
            raise ValueError("bandwidth is only defined for the built-in RBF kernel.")
        scale = None
        kernel_matrix = jnp.asarray(kernel(distances))
        if jnp.iscomplexobj(kernel_matrix):
            raise ValueError("kernel must return a real-valued Gram matrix.")
        if kernel_matrix.shape != distances.shape:
            raise ValueError("kernel must return a square matrix matching pairwise distances.")
        if not bool(jnp.all(jnp.isfinite(kernel_matrix))):
            raise ValueError("kernel must return only finite values.")
        asymmetry = jnp.linalg.norm(kernel_matrix - kernel_matrix.T)
        symmetry_scale = max(
            float(jnp.linalg.norm(kernel_matrix)),
            float(jnp.finfo(kernel_matrix.dtype).tiny),
        )
        symmetry_tolerance = 100.0 * float(jnp.finfo(kernel_matrix.dtype).eps) * symmetry_scale
        if float(asymmetry) > symmetry_tolerance:
            raise ValueError("kernel must return a symmetric Gram matrix.")
    kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.T)
    eigenvalues = jnp.linalg.eigvalsh(kernel_matrix)
    spectral_scale = max(
        float(jnp.max(jnp.abs(eigenvalues))),
        float(jnp.finfo(kernel_matrix.dtype).tiny),
    )
    backward_error = (
        float(jnp.finfo(kernel_matrix.dtype).eps) * kernel_matrix.shape[0] * spectral_scale
    )
    effective_psd_tolerance = max(psd_tolerance, backward_error)
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
    if not bool(jnp.isfinite(observed)) or not bool(jnp.all(jnp.isfinite(null_distribution))):
        raise FloatingPointError("kernel_mmd_two_sample_test produced a nonfinite statistic.")
    pvalue = (1.0 + jnp.sum(null_distribution >= observed)) / (count + 1.0)
    return HypothesisTestResult(
        statistic=observed,
        pvalue=pvalue,
        null_distribution=null_distribution,
        method=(
            "RBF-distance-kernel MMD permutation test"
            if kernel is None
            else "Custom-kernel MMD permutation test"
        ),
        diagnostics={
            "bandwidth": scale,
            "kernel_eigenvalues": eigenvalues,
            "requested_psd_tolerance": psd_tolerance,
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
    """Test a zero mean paired displacement by within-pair random sign flips.

    Pairwise exchangeability of ``(x_i, y_i)`` under the null gives exact
    randomization calibration: swapping a pair leaves the pooled base fit
    unchanged and negates that pair's tangent displacement. Central symmetry
    of the tangent displacements is a weaker modeling route to the same
    sign-flip invariance.
    """
    require_exact_operations(manifold, "paired_frechet_test", "dist", "log", "exp")
    left = as_manifold_data(manifold, x)
    right = as_manifold_data(manifold, y)
    require_unbatched(left, "paired_frechet_test")
    require_unbatched(right, "paired_frechet_test")
    if left.n_samples != right.n_samples or left.n_samples < 2:
        raise ValueError("paired samples must have the same size of at least two.")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    count = _validate_permutations(n_permutations)
    pooled = _combine(manifold, left, right)
    base_fit = frechet_mean(manifold, pooled, maxiter=maxiter, tol=tol)
    if not base_fit.converged:
        raise RuntimeError(
            "The pooled Fréchet-mean fit did not converge; paired tangent "
            "displacements are not certified."
        )
    left_logs = manifold.log(base_fit.point, left.values)
    right_logs = manifold.log(base_fit.point, right.values)
    differences = manifold.lincomb(base_fit.point, 1.0, right_logs, -1.0, left_logs)
    if not all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(differences)
    ):
        raise ValueError(
            "paired_frechet_test encountered an undefined logarithm at the pooled base point."
        )
    mean_difference = weighted_tangent_sum(
        manifold,
        differences,
        jnp.full((left.n_samples,), 1.0 / left.n_samples),
    )
    observed = manifold.norm(base_fit.point, mean_difference)
    keys = jax.random.split(as_key(key, "paired_frechet_test"), count)
    null = []
    for permutation_key in keys:
        signs = (
            2.0
            * jax.random.bernoulli(
                permutation_key,
                shape=(left.n_samples,),
            ).astype(float)
            - 1.0
        )
        permuted_mean = weighted_tangent_sum(
            manifold,
            differences,
            signs / left.n_samples,
        )
        null.append(manifold.norm(base_fit.point, permuted_mean))
    null_distribution = jnp.asarray(null)
    if not bool(jnp.isfinite(observed)) or not bool(jnp.all(jnp.isfinite(null_distribution))):
        raise FloatingPointError("paired_frechet_test produced a nonfinite statistic.")
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
