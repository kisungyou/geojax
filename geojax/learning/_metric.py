"""Equivariant-embedding metric learning."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from geojax.geometry import Euclidean, Product

from ._capabilities import LearningCapabilityError
from ._data import as_manifold_data
from ._results import MetricLearningModel
from ._utils import flatten_embedding, require_unbatched


def _spd_log(matrix: Any, floor: float) -> Any:
    values, vectors = jnp.linalg.eigh(0.5 * (matrix + matrix.T))
    logged = jnp.log(jnp.maximum(values, floor))
    return (vectors * logged[None, :]) @ vectors.T


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
    """Fit regularized log-Euclidean RMML from embedded labeled pairs."""
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "riemannian_metric_learning")
    label_values = jnp.asarray(labels)
    if label_values.shape != (adapted.n_samples,):
        raise ValueError(f"labels must have shape ({adapted.n_samples},).")
    if jnp.unique(label_values).size < 2:
        raise ValueError("labels must contain at least two classes.")
    if regularization < 0.0 or eigenvalue_floor <= 0.0:
        raise ValueError("regularization must be nonnegative and eigenvalue_floor positive.")
    if not 0.0 <= float(balance) <= 1.0:
        raise ValueError("balance must lie between 0 and 1.")
    if embedding is not None and not callable(embedding):
        raise TypeError("embedding must be callable.")
    embedding_function = embedding or _default_embedding(manifold)
    coordinates = flatten_embedding(embedding_function(adapted.values))
    if coordinates.shape[0] != adapted.n_samples:
        raise ValueError("embedding must preserve the leading sample dimension.")
    dimension = coordinates.shape[1]
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
    regularized_similar = similar + float(regularization) * identity
    regularized_dissimilar = dissimilar + float(regularization) * identity
    floor = float(eigenvalue_floor)
    metric = _spd_exp(
        -float(balance) * _spd_log(regularized_similar, floor)
        + (1.0 - float(balance)) * _spd_log(regularized_dissimilar, floor)
    )
    metric = 0.5 * (metric + metric.T)
    transformed = coordinates @ jnp.linalg.cholesky(metric)
    learned_distances = jnp.linalg.norm(
        transformed[:, None, :] - transformed[None, :, :], axis=-1
    )
    model = MetricLearningModel(
        metric=metric,
        embedding=embedding_function,
        regularization=float(regularization),
        diagnostics={
            "similar_scatter": similar,
            "dissimilar_scatter": dissimilar,
            "similar_pairs": similar_count,
            "dissimilar_pairs": dissimilar_count,
            "balance": float(balance),
            "embedded_data": coordinates,
            "pairwise_distances": learned_distances,
        },
    )
    return model


__all__ = ["riemannian_metric_learning"]
