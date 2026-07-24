"""Geometry-aware building blocks for JAX learning systems."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from geojax.geometry import Product

Array = Any


def _require_exact_operations(manifold: Any, helper: str, *operations: str) -> None:
    unavailable = []
    for operation in operations:
        if hasattr(manifold, "operation_kind"):
            try:
                kind = manifold.operation_kind(operation)
            except ValueError:
                kind = "unknown"
        else:
            kind = "exact" if bool(getattr(manifold, f"{operation}_is_exact", False)) else "unknown"
        if kind != "exact":
            unavailable.append(f"{operation}={kind}")
    if unavailable:
        details = ", ".join(unavailable)
        raise ValueError(
            f"{helper} requires exact geodesic operations; received {details}. "
            "Call the manifold's retraction operations directly when proxy or "
            "numerical-local behavior is intended."
        )


def _event_shapes(manifold: Any) -> Any:
    if isinstance(manifold, Product):
        return jax.tree_util.tree_map(lambda factor: tuple(factor.shape), manifold.factors)
    return tuple(manifold.shape)


def _map_points(
    manifold: Any,
    points: Any,
    transform: Callable[[Array, tuple[int, ...]], Array],
    *,
    name: str,
) -> Any:
    shapes = _event_shapes(manifold)
    if isinstance(manifold, Product):
        point_leaves, point_tree = jax.tree_util.tree_flatten(points)
        shape_leaves, shape_tree = jax.tree_util.tree_flatten(
            shapes,
            is_leaf=lambda value: isinstance(value, tuple),
        )
        if point_tree != shape_tree:
            raise ValueError(f"{name} must match the Product factor pytree.")
        return jax.tree_util.tree_unflatten(
            point_tree,
            [
                transform(jnp.asarray(point), tuple(shape))
                for point, shape in zip(point_leaves, shape_leaves)
            ],
        )
    return transform(jnp.asarray(points), shapes)


def _validate_event_shape(point: Array, event_shape: tuple[int, ...], name: str) -> None:
    event_ndim = len(event_shape)
    if point.ndim <= event_ndim:
        raise ValueError(f"{name} must include a collection axis before {event_shape}.")
    if tuple(point.shape[-event_ndim:]) != event_shape:
        raise ValueError(
            f"{name} must end in manifold shape {event_shape}; received {point.shape}."
        )


def pairwise_squared_dist(manifold: Any, x: Any, y: Any) -> Array:
    """Return all squared distances between two collections of points.

    ``x`` and ``y`` have shapes ``batch_x + (n,) + manifold.shape`` and
    ``batch_y + (m,) + manifold.shape``. Their leading batch shapes follow
    standard NumPy broadcasting and the result has shape
    ``broadcast(batch_x, batch_y) + (n, m)``.

    Product-manifold collections use the same pytree as ``manifold.factors``;
    every leaf has its own factor event shape and shares the collection axes.
    The helper rejects geometries whose distance is a retraction proxy or only
    a numerical-local candidate.
    """
    _require_exact_operations(manifold, "pairwise_squared_dist", "dist")

    def expand_x(point: Array, event_shape: tuple[int, ...]) -> Array:
        _validate_event_shape(point, event_shape, "x")
        return jnp.expand_dims(point, axis=-(len(event_shape) + 1))

    def expand_y(point: Array, event_shape: tuple[int, ...]) -> Array:
        _validate_event_shape(point, event_shape, "y")
        return jnp.expand_dims(point, axis=-(len(event_shape) + 2))

    paired_x = _map_points(manifold, x, expand_x, name="x")
    paired_y = _map_points(manifold, y, expand_y, name="y")
    return manifold.squared_dist(paired_x, paired_y)


def _scale_tangent(manifold: Any, tangent: Any, coefficient: Array) -> Any:
    coefficient = jnp.asarray(coefficient)

    def scale(leaf: Array, event_shape: tuple[int, ...]) -> Array:
        del event_shape
        coefficient_shape = coefficient.shape + (1,) * leaf.ndim
        leaf_shape = (1,) * coefficient.ndim + leaf.shape
        return coefficient.reshape(coefficient_shape) * leaf.reshape(leaf_shape)

    return _map_points(
        manifold,
        tangent,
        scale,
        name="tangent vector",
    )


def geodesic_interpolate(manifold: Any, x: Any, y: Any, t: Array) -> Any:
    """Interpolate along the selected shortest geodesic from ``x`` to ``y``.

    Scalar ``t`` preserves the broadcast endpoint batch shape. An array of
    interpolation parameters adds its shape before all endpoint batch and
    manifold event dimensions. Values in ``[0, 1]`` trace the segment, while
    values outside that interval extrapolate where the exponential map is
    defined. Proxy and numerical-local logarithms are rejected.
    """
    _require_exact_operations(manifold, "geodesic_interpolate", "log", "exp")
    tangent = manifold.log(x, y)
    return manifold.exp(x, _scale_tangent(manifold, tangent, t))


def tangent_map(
    source: Any,
    target: Any,
    x: Any,
    *,
    source_base: Any,
    target_base: Any,
    transform: Callable[[Any], Any],
) -> Any:
    """Map points through user-defined tangent-space computation.

    The operation is

    ``x -> Log_source_base(x) -> transform -> tangent projection -> Exp_target_base``.

    ``transform`` owns any trainable parameters through its closure, keeping
    this primitive independent of Flax, Equinox, and other neural frameworks.
    Both manifolds must advertise the corresponding geodesic operation as
    exact.
    """
    _require_exact_operations(source, "tangent_map source", "log")
    _require_exact_operations(target, "tangent_map target", "exp")
    source_tangent = source.log(source_base, x)
    target_tangent = target.tangent_project(target_base, transform(source_tangent))
    return target.exp(target_base, target_tangent)


__all__ = ["geodesic_interpolate", "pairwise_squared_dist", "tangent_map"]
