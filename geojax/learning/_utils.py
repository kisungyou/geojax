"""Tree and sample-axis utilities for manifold learning."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from geojax.geometry import Product


def event_shapes(manifold: Any) -> Any:
    if isinstance(manifold, Product):
        return jax.tree_util.tree_map(lambda factor: tuple(factor.shape), manifold.factors)
    return tuple(manifold.shape)


def flatten_geometry_values(manifold: Any, values: Any, *, name: str) -> tuple[list[Any], list[Any]]:
    if isinstance(manifold, Product):
        factors, factor_tree = jax.tree_util.tree_flatten(manifold.factors)
        leaves, value_tree = jax.tree_util.tree_flatten(values)
        if factor_tree != value_tree:
            raise ValueError(f"{name} must match the Product factor pytree.")
        return list(factors), list(leaves)
    return [manifold], [values]


def unflatten_geometry(manifold: Any, leaves: list[Any]) -> Any:
    if isinstance(manifold, Product):
        return jax.tree_util.tree_unflatten(manifold._treedef, leaves)
    return leaves[0]


def take_samples(manifold: Any, values: Any, indices: Any) -> Any:
    factors, leaves = flatten_geometry_values(manifold, values, name="data")
    out = [
        jnp.take(jnp.asarray(leaf), indices, axis=-(len(factor.shape) + 1))
        for factor, leaf in zip(factors, leaves)
    ]
    return unflatten_geometry(manifold, out)


def take_point(manifold: Any, values: Any, index: int | Any) -> Any:
    factors, leaves = flatten_geometry_values(manifold, values, name="data")
    out = [
        jnp.take(jnp.asarray(leaf), index, axis=-(len(factor.shape) + 1))
        for factor, leaf in zip(factors, leaves)
    ]
    return unflatten_geometry(manifold, out)


def stack_points(manifold: Any, points: list[Any]) -> Any:
    if not points:
        raise ValueError("points must contain at least one manifold point.")
    factors, first = flatten_geometry_values(manifold, points[0], name="point")
    collected = [[leaf] for leaf in first]
    for point in points[1:]:
        _, leaves = flatten_geometry_values(manifold, point, name="point")
        for bucket, leaf in zip(collected, leaves):
            bucket.append(leaf)
    out = [jnp.stack(bucket, axis=0) for bucket in collected]
    return unflatten_geometry(manifold, out)


def scale_tangent(manifold: Any, tangent: Any, coefficient: Any) -> Any:
    factors, leaves = flatten_geometry_values(manifold, tangent, name="tangent vector")
    coefficient = jnp.asarray(coefficient)
    out = []
    for factor, leaf in zip(factors, leaves):
        array = jnp.asarray(leaf)
        event_ndim = len(factor.shape)
        batch_ndim = array.ndim - event_ndim
        coefficient_shape = coefficient.shape + (1,) * array.ndim
        array_shape = (1,) * coefficient.ndim + array.shape
        del batch_ndim
        out.append(coefficient.reshape(coefficient_shape) * array.reshape(array_shape))
    return unflatten_geometry(manifold, out)


def scale_tangent_samples(manifold: Any, tangents: Any, coefficients: Any) -> Any:
    """Scale each canonical sample tangent by its corresponding coefficient."""
    factors, leaves = flatten_geometry_values(manifold, tangents, name="tangent vectors")
    coefficients = jnp.asarray(coefficients)
    out = []
    for factor, leaf in zip(factors, leaves):
        array = jnp.asarray(leaf)
        event_ndim = len(factor.shape)
        sample_axis = array.ndim - event_ndim - 1
        expected = array.shape[sample_axis]
        if coefficients.shape != (expected,):
            raise ValueError(
                f"coefficients must have shape ({expected},); received {coefficients.shape}."
            )
        shape = (1,) * sample_axis + coefficients.shape + (1,) * event_ndim
        out.append(coefficients.reshape(shape) * array)
    return unflatten_geometry(manifold, out)


def weighted_tangent_sum(manifold: Any, tangents: Any, weights: Any) -> Any:
    factors, leaves = flatten_geometry_values(manifold, tangents, name="tangent vectors")
    weights = jnp.asarray(weights)
    out = []
    for factor, leaf in zip(factors, leaves):
        axis = jnp.asarray(leaf).ndim - len(factor.shape) - 1
        out.append(jnp.tensordot(weights, leaf, axes=((0,), (axis,))))
    return unflatten_geometry(manifold, out)


def normalize_weights(n_samples: int, sample_weight: Any | None) -> Any:
    if sample_weight is None:
        return jnp.full((n_samples,), 1.0 / n_samples)
    weights = jnp.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,):
        raise ValueError(f"sample_weight must have shape ({n_samples},); received {weights.shape}.")
    if not bool(jnp.all(jnp.isfinite(weights))):
        raise ValueError("sample_weight must contain only finite values.")
    if not bool(jnp.all(weights >= 0.0)):
        raise ValueError("sample_weight must be nonnegative.")
    total = float(jnp.sum(weights))
    if total <= 0.0:
        raise ValueError("sample_weight must have positive total mass.")
    return weights / total


def require_unbatched(data: Any, method: str) -> None:
    if tuple(data.batch_shape):
        raise ValueError(
            f"{method} currently expects one unbatched dataset; received batch shape "
            f"{data.batch_shape}. Use jax.vmap over independent datasets."
        )


def as_key(key: Any | int | None, method: str) -> Any:
    if key is None:
        raise ValueError(f"{method} requires an explicit JAX random key.")
    return jax.random.key(key) if isinstance(key, int) else key


def tree_all_finite(values: Any) -> bool:
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in jax.tree_util.tree_leaves(values))


def deterministic_sign_columns(matrix: Any) -> Any:
    """Choose deterministic signs for frame/eigenvector columns."""
    matrix = jnp.asarray(matrix)
    pivot = jnp.argmax(jnp.abs(matrix), axis=-2)
    selected = jnp.take_along_axis(matrix, pivot[..., None, :], axis=-2)[..., 0, :]
    signs = jnp.where(selected < 0.0, -1.0, 1.0)
    return matrix * signs[..., None, :]


def flatten_embedding(values: Any) -> Any:
    leaves = [jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(values)]
    if not leaves:
        raise ValueError("embedding returned an empty pytree.")
    n_samples = leaves[0].shape[0]
    if any(leaf.shape[0] != n_samples for leaf in leaves):
        raise ValueError("embedding leaves must share their leading sample dimension.")
    return jnp.concatenate([leaf.reshape((n_samples, -1)) for leaf in leaves], axis=-1)


__all__ = [
    "as_key",
    "deterministic_sign_columns",
    "event_shapes",
    "flatten_embedding",
    "flatten_geometry_values",
    "normalize_weights",
    "require_unbatched",
    "scale_tangent",
    "scale_tangent_samples",
    "stack_points",
    "take_point",
    "take_samples",
    "tree_all_finite",
    "unflatten_geometry",
    "weighted_tangent_sum",
]
