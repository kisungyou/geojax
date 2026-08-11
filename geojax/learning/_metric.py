"""Equivariant-embedding metric learning."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from geojax.geometry import Euclidean, Product
from geojax.geometry._numerics import stable_norm

from ._capabilities import LearningCapabilityError
from ._data import as_manifold_data
from ._results import MetricLearningModel
from ._utils import (
    flatten_embedding,
    interval_control,
    nonnegative_control,
    positive_control,
    require_unbatched,
)


def _spd_log(matrix: Any, relative_floor: float) -> tuple[Any, Any]:
    values, vectors = jnp.linalg.eigh(0.5 * (matrix + matrix.T))
    tiny = jnp.finfo(values.dtype).tiny
    scale = jnp.maximum(jnp.max(jnp.abs(values)), tiny)
    floor = jnp.maximum(
        jnp.asarray(relative_floor, dtype=values.dtype) * scale,
        tiny,
    )
    logged = jnp.log(jnp.maximum(values, floor))
    return (vectors * logged[None, :]) @ vectors.T, floor


def _spd_exp(matrix: Any) -> Any:
    values, vectors = jnp.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * jnp.exp(values)[None, :]) @ vectors.T


def _default_embedding(manifold: Any) -> Callable[[Any], Any]:
    if isinstance(manifold, Euclidean):
        return lambda values: values
    if isinstance(manifold, Product):
        factors, factor_tree = jax.tree_util.tree_flatten(manifold.factors)
        if not all(hasattr(factor, "embed") for factor in factors):
            raise LearningCapabilityError(
                "riemannian_metric_learning requires every Product factor to provide embed(x), "
                "or an explicit embedding callable."
            )

        def embed_product(values: Any) -> Any:
            leaves, value_tree = jax.tree_util.tree_flatten(values)
            if value_tree != factor_tree:
                raise ValueError("Embedding input must match the Product factor pytree.")
            return tuple(factor.embed(leaf) for factor, leaf in zip(factors, leaves))

        return embed_product
    if hasattr(manifold, "embed") and callable(manifold.embed):
        return manifold.embed
    raise LearningCapabilityError(
        "riemannian_metric_learning requires an equivariant embedding; supply embedding=... "
        f"for {type(manifold).__name__}."
    )


def riemannian_metric_learning(
    manifold: Any,
    data: Any,
    labels: Any,
    *,
    regularization: float = 0.1,
    balance: float = 0.5,
    embedding: Callable[[Any], Any] | None = None,
    eigenvalue_floor: float = 1e-10,
) -> MetricLearningModel:
    r"""Fit the regularized log-Euclidean RMML closed form.

    For similar- and dissimilar-pair scatter matrices ``S`` and ``D``, this
    implements Equation (21) of Zhu et al. (2018),

    ``A = exp((-balance * log(S) + (1 - balance) * log(D)) / 2)``.

    The derivation is invariant only when ``embedding`` is an appropriate
    equivariant embedding for the supplied geometry.  An arbitrary callable
    still defines a valid Euclidean-feature Mahalanobis model, but it does not
    inherit that Riemannian invariance automatically.
    """
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "riemannian_metric_learning")
    raw_labels = np.asarray(labels)
    if raw_labels.shape != (adapted.n_samples,):
        raise ValueError(f"labels must have shape ({adapted.n_samples},).")
    if raw_labels.dtype.kind in {"f", "c"} and not np.all(np.isfinite(raw_labels)):
        raise ValueError("labels must not contain NaN or infinite values.")
    if raw_labels.dtype.kind in {"O", "U", "S"} and any(
        value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        for value in raw_labels.tolist()
    ):
        raise ValueError("labels must not contain missing values.")
    try:
        classes, encoded = np.unique(raw_labels, return_inverse=True)
    except TypeError as exc:
        raise TypeError("labels must contain mutually comparable scalar values.") from exc
    if classes.size < 2:
        raise ValueError("labels must contain at least two classes.")
    label_values = jnp.asarray(encoded, dtype=int)
    regularization = nonnegative_control(regularization, name="regularization")
    eigenvalue_floor = positive_control(eigenvalue_floor, name="eigenvalue_floor")
    balance = interval_control(balance, name="balance", lower=0.0, upper=1.0)
    if embedding is not None and not callable(embedding):
        raise TypeError("embedding must be callable.")
    embedding_function = _default_embedding(manifold) if embedding is None else embedding
    coordinates = flatten_embedding(embedding_function(adapted.values))
    if coordinates.shape[0] != adapted.n_samples:
        raise ValueError("embedding must preserve the leading sample dimension.")
    if not bool(jnp.all(jnp.isfinite(coordinates))):
        raise ValueError("embedding must return only finite coordinates.")
    dimension = coordinates.shape[1]
    if dimension < 1:
        raise ValueError("embedding must return at least one feature per observation.")
    similar = jnp.zeros((dimension, dimension), dtype=coordinates.dtype)
    dissimilar = jnp.zeros_like(similar)
    similar_count = dissimilar_count = 0
    for left in range(adapted.n_samples - 1):
        for right in range(left + 1, adapted.n_samples):
            difference = coordinates[left] - coordinates[right]
            scatter = jnp.outer(difference, difference)
            if bool(label_values[left] == label_values[right]):
                similar = similar + scatter
                similar_count += 1
            else:
                dissimilar = dissimilar + scatter
                dissimilar_count += 1
    if similar_count == 0 or dissimilar_count == 0:
        raise ValueError("RMML needs at least one similar and one dissimilar pair.")
    identity = jnp.eye(dimension, dtype=coordinates.dtype)
    if not bool(jnp.all(jnp.isfinite(similar))) or not bool(jnp.all(jnp.isfinite(dissimilar))):
        raise FloatingPointError("RMML pair-scatter matrices are nonfinite.")
    regularized_similar = similar + regularization * identity
    regularized_dissimilar = dissimilar + regularization * identity
    log_similar, similar_floor = _spd_log(
        regularized_similar,
        eigenvalue_floor,
    )
    log_dissimilar, dissimilar_floor = _spd_log(
        regularized_dissimilar,
        eigenvalue_floor,
    )
    metric = _spd_exp(0.5 * (-balance * log_similar + (1.0 - balance) * log_dissimilar))
    metric = 0.5 * (metric + metric.T)
    metric_eigenvalues = jnp.linalg.eigvalsh(metric)
    if not bool(jnp.all(jnp.isfinite(metric))) or not bool(jnp.all(metric_eigenvalues > 0.0)):
        raise FloatingPointError("RMML failed to construct a finite positive-definite metric.")
    transformed = coordinates @ jnp.linalg.cholesky(metric)
    learned_distances = stable_norm(
        transformed[:, None, :] - transformed[None, :, :],
        axis=-1,
    )
    model = MetricLearningModel(
        manifold=manifold,
        metric=metric,
        embedding=embedding_function,
        regularization=regularization,
        diagnostics={
            "similar_scatter": similar,
            "dissimilar_scatter": dissimilar,
            "similar_pairs": similar_count,
            "dissimilar_pairs": dissimilar_count,
            "balance": balance,
            "classes": classes,
            "encoded_labels": label_values,
            "eigenvalue_floor": eigenvalue_floor,
            "effective_similar_eigenvalue_floor": similar_floor,
            "effective_dissimilar_eigenvalue_floor": dissimilar_floor,
            "metric_eigenvalues": metric_eigenvalues,
            "embedded_data": coordinates,
            "pairwise_distances": learned_distances,
        },
    )
    return model


__all__ = ["riemannian_metric_learning"]
