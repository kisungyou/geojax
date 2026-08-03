"""Distance-kernel regression with manifold-valued predictors."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._data import as_manifold_data
from ._geometry import pairwise_distances
from ._results import KernelCVResult, KernelRegressionModel
from ._utils import as_key, require_unbatched, take_samples


def _validate_targets(targets: Any, n_samples: int) -> Any:
    values = jnp.asarray(targets)
    if values.ndim < 1 or values.shape[0] != n_samples:
        raise ValueError(f"targets must have leading sample dimension {n_samples}; received {values.shape}.")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("targets must contain only finite values.")
    return values


def _kernel_weights(
    distances: Any,
    bandwidth: float,
    kernel: Callable[[Any, float], Any] | None,
) -> Any:
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive.")
    weights = (
        jnp.exp(-0.5 * (distances / bandwidth) ** 2)
        if kernel is None
        else jnp.asarray(kernel(distances, bandwidth))
    )
    if weights.shape != distances.shape:
        raise ValueError("kernel must return one weight per supplied distance.")
    return jnp.maximum(weights, 0.0)


def kernel_regression(
    manifold: Any,
    data: Any,
    targets: Any,
    *,
    bandwidth: float,
    kernel: Callable[[Any, float], Any] | None = None,
) -> KernelRegressionModel:
    """Fit Nadaraya-Watson regression with manifold-valued predictors."""
    require_exact_operations(manifold, "kernel_regression", "dist")
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "kernel_regression")
    target_values = _validate_targets(targets, adapted.n_samples)
    if float(bandwidth) <= 0.0:
        raise ValueError("bandwidth must be positive.")
    return KernelRegressionModel(
        manifold=manifold,
        training_data=adapted,
        targets=target_values,
        bandwidth=float(bandwidth),
        kernel=kernel,
    )


def _predict_kernel_regression(model: KernelRegressionModel, data: Any) -> Any:
    queries = as_manifold_data(model.manifold, data)
    require_unbatched(queries, "KernelRegressionModel.predict")
    distances = pairwise_distances(
        model.manifold,
        queries,
        model.training_data,
    )
    weights = _kernel_weights(distances, model.bandwidth, model.kernel)
    denominator = jnp.sum(weights, axis=-1, keepdims=True)
    nearest = jax.nn.one_hot(jnp.argmin(distances, axis=-1), distances.shape[-1])
    normalized = jnp.where(denominator > 1e-15, weights / denominator, nearest)
    return jnp.tensordot(normalized, model.targets, axes=((-1,), (0,)))


def select_kernel_bandwidth(
    manifold: Any,
    data: Any,
    targets: Any,
    bandwidths: Any,
    *,
    n_folds: int = 5,
    key: Any | int | None,
    kernel: Callable[[Any, float], Any] | None = None,
) -> KernelCVResult:
    """Select a kernel bandwidth by deterministic-key K-fold mean squared error."""
    require_exact_operations(manifold, "select_kernel_bandwidth", "dist")
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "select_kernel_bandwidth")
    target_values = _validate_targets(targets, adapted.n_samples)
    candidates = jnp.asarray(bandwidths, dtype=float)
    if candidates.ndim != 1 or candidates.size == 0 or not bool(jnp.all(candidates > 0.0)):
        raise ValueError("bandwidths must be a nonempty vector of positive values.")
    if not 2 <= int(n_folds) <= adapted.n_samples:
        raise ValueError("n_folds must be between 2 and n_samples.")
    permutation = jax.random.permutation(as_key(key, "select_kernel_bandwidth"), adapted.n_samples)
    fold_ids = jnp.arange(adapted.n_samples) % int(n_folds)
    scores: list[Any] = []
    for bandwidth in candidates.tolist():
        losses = []
        for fold in range(int(n_folds)):
            test_positions = jnp.where(fold_ids == fold, size=adapted.n_samples, fill_value=-1)[0]
            test_positions = test_positions[test_positions >= 0]
            test_indices = permutation[test_positions]
            train_positions = jnp.where(fold_ids != fold, size=adapted.n_samples, fill_value=-1)[0]
            train_positions = train_positions[train_positions >= 0]
            train_indices = permutation[train_positions]
            train_data = as_manifold_data(manifold, take_samples(manifold, adapted.values, train_indices))
            test_data = as_manifold_data(manifold, take_samples(manifold, adapted.values, test_indices))
            model = kernel_regression(
                manifold,
                train_data,
                target_values[train_indices],
                bandwidth=float(bandwidth),
                kernel=kernel,
            )
            residual = model.predict(test_data) - target_values[test_indices]
            losses.append(jnp.mean(residual * residual))
        scores.append(jnp.mean(jnp.asarray(losses)))
    score_array = jnp.asarray(scores)
    best = int(jnp.argmin(score_array))
    # JAX exposes float32 scalars as their exact binary value through ``float``.
    # Its string form is the shortest round-trippable decimal and is the stable
    # public hyperparameter representation across x32 and x64 modes.
    selected = float(str(candidates[best]))
    model = kernel_regression(
        manifold,
        adapted,
        target_values,
        bandwidth=selected,
        kernel=kernel,
    )
    return KernelCVResult(
        model=model,
        bandwidth=selected,
        scores=score_array,
        diagnostics={"bandwidths": candidates, "permutation": permutation},
    )


__all__ = ["kernel_regression", "select_kernel_bandwidth"]
