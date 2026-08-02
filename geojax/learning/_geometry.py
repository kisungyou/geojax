"""Differentiable geometry-aware learning primitives."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from geojax.geometry import Product

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._results import NeighborsResult
from ._utils import event_shapes, scale_tangent, take_samples

Array = Any


def _values(manifold: Any, data: Any, *, name: str) -> tuple[Any, int]:
    if isinstance(data, ManifoldData):
        return data.values, data.n_samples
    adapted = as_manifold_data(manifold, data, check="shape")
    return adapted.values, adapted.n_samples


def _map_points(
    manifold: Any,
    points: Any,
    transform: Callable[[Array, tuple[int, ...]], Array],
    *,
    name: str,
) -> Any:
    if isinstance(manifold, Product):
        point_leaves, point_tree = jax.tree_util.tree_flatten(points)
        factors, factor_tree = jax.tree_util.tree_flatten(manifold.factors)
        if point_tree != factor_tree:
            raise ValueError(f"{name} must match the Product factor pytree.")
        return jax.tree_util.tree_unflatten(
            point_tree,
            [
                transform(jnp.asarray(point), tuple(factor.shape))
                for point, factor in zip(point_leaves, factors)
            ],
        )
    return transform(jnp.asarray(points), tuple(event_shapes(manifold)))


def pairwise_distances(
    manifold: Any,
    x: Any,
    y: Any | None = None,
    *,
    squared: bool = False,
    block_size: int | None = None,
) -> Array:
    """Return all pairwise exact distances between two point collections.

    Collections use ``batch_shape + (n_samples,) + event_shape``. Product
    collections use the factor pytree and share their sample and batch axes.
    ``block_size`` limits the number of right-hand samples materialized in one
    geometry call while retaining a dense result.
    """
    require_exact_operations(manifold, "pairwise_distances", "dist")
    left, _ = _values(manifold, x, name="x")
    right, n_right = _values(manifold, x if y is None else y, name="y")

    def expand_left(point: Array, event_shape: tuple[int, ...]) -> Array:
        return jnp.expand_dims(point, axis=-(len(event_shape) + 1))

    def expand_right(point: Array, event_shape: tuple[int, ...]) -> Array:
        return jnp.expand_dims(point, axis=-(len(event_shape) + 2))

    paired_left = _map_points(manifold, left, expand_left, name="x")

    def evaluate(right_values: Any) -> Array:
        paired_right = _map_points(manifold, right_values, expand_right, name="y")
        values = manifold.squared_dist(paired_left, paired_right)
        return values if squared else jnp.sqrt(jnp.maximum(values, 0.0))

    if block_size is None:
        return evaluate(right)
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    blocks = [
        evaluate(take_samples(manifold, right, jnp.arange(start, min(start + block_size, n_right))))
        for start in range(0, n_right, block_size)
    ]
    return jnp.concatenate(blocks, axis=-1)


def geodesic_interpolation(manifold: Any, x: Any, y: Any, t: Array) -> Any:
    """Evaluate ``Exp_x(t Log_x(y))`` on the selected exact geodesic."""
    require_exact_operations(manifold, "geodesic_interpolation", "log", "exp")
    tangent = manifold.log(x, y)
    return manifold.exp(x, scale_tangent(manifold, tangent, t))


def tangent_space_map(
    source: Any,
    target: Any,
    x: Any,
    *,
    source_base: Any,
    target_base: Any,
    transform: Callable[[Any], Any],
) -> Any:
    """Apply a user transform between exact source and target tangent spaces."""
    require_exact_operations(source, "tangent_space_map source", "log")
    require_exact_operations(target, "tangent_space_map target", "exp")
    source_tangent = source.log(source_base, x)
    transformed = transform(source_tangent)
    target_tangent = target.tangent_project(target_base, transformed)
    return target.exp(target_base, target_tangent)


def nearest_neighbors(
    manifold: Any,
    data: Any,
    queries: Any | None = None,
    *,
    n_neighbors: int = 5,
    exclude_self: bool = True,
    block_size: int | None = None,
) -> NeighborsResult:
    """Find exact-distance nearest neighbors in a dense manifold dataset."""
    adapted = data if isinstance(data, ManifoldData) else as_manifold_data(manifold, data)
    query_data = adapted if queries is None else (
        queries if isinstance(queries, ManifoldData) else as_manifold_data(manifold, queries)
    )
    if adapted.batch_shape or query_data.batch_shape:
        raise ValueError("nearest_neighbors currently expects unbatched datasets.")
    maximum = adapted.n_samples - (1 if queries is None and exclude_self else 0)
    if not 1 <= int(n_neighbors) <= maximum:
        raise ValueError(f"n_neighbors must be between 1 and {maximum}.")
    distances = pairwise_distances(
        manifold,
        query_data,
        adapted,
        block_size=block_size,
    )
    if queries is None and exclude_self:
        distances = distances.at[jnp.diag_indices(adapted.n_samples)].set(jnp.inf)
    indices = jnp.argsort(distances, axis=-1)[..., : int(n_neighbors)]
    selected = jnp.take_along_axis(distances, indices, axis=-1)
    return NeighborsResult(selected, indices)


__all__ = [
    "geodesic_interpolation",
    "nearest_neighbors",
    "pairwise_distances",
    "tangent_space_map",
]
