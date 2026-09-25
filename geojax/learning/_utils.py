"""Tree and sample-axis utilities for manifold learning."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp

from geojax.geometry import Product
from geojax.geometry._numerics import power_of_two_rescale
from geojax.geometry.base import validate_integer, validate_nonnegative, validate_positive


def event_shapes(manifold: Any) -> Any:
    if isinstance(manifold, Product):
        return jax.tree_util.tree_map(lambda factor: tuple(factor.shape), manifold.factors)
    return tuple(manifold.shape)


def flatten_geometry_values(
    manifold: Any, values: Any, *, name: str
) -> tuple[list[Any], list[Any]]:
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


def as_real_array(values: Any, *, name: str) -> Any:
    """Convert array-like input without silently discarding imaginary parts."""
    try:
        array = jnp.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric array.") from exc
    if jnp.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued.")
    return jnp.asarray(array, dtype=float)


def normalize_weights(n_samples: int, sample_weight: Any | None) -> Any:
    if sample_weight is None:
        return jnp.full((n_samples,), 1.0 / n_samples)
    weights = as_real_array(sample_weight, name="sample_weight")
    if weights.shape != (n_samples,):
        raise ValueError(f"sample_weight must have shape ({n_samples},); received {weights.shape}.")
    if not bool(jnp.all(jnp.isfinite(weights))):
        raise ValueError("sample_weight must contain only finite values.")
    if not bool(jnp.all(weights >= 0.0)):
        raise ValueError("sample_weight must be nonnegative.")
    maximum = float(jnp.max(weights))
    if maximum <= 0.0:
        raise ValueError("sample_weight must have positive total mass.")
    # Scaling before summation avoids overflow while preserving every relative
    # weight that is representable in the input dtype.
    scaled = power_of_two_rescale(weights, axis=0)
    total = jnp.sum(scaled)
    if not bool(jnp.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("sample_weight could not be normalized safely.")
    return scaled / total


def require_unbatched(data: Any, method: str) -> None:
    if tuple(data.batch_shape):
        raise ValueError(
            f"{method} currently expects one unbatched dataset; received batch shape "
            f"{data.batch_shape}. Adapt and process independent datasets separately."
        )


def as_key(key: Any | int | None, method: str) -> Any:
    if key is None:
        raise ValueError(f"{method} requires an explicit JAX random key.")
    if isinstance(key, bool):
        raise TypeError(f"{method} key must be an integer seed or JAX random key.")
    if isinstance(key, Integral):
        return jax.random.key(int(key))
    try:
        key_data = jax.random.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{method} key must be an integer seed or JAX random key.") from exc
    if jnp.shape(key_data) != (2,):
        raise TypeError(f"{method} key must be a scalar JAX random key, not a key batch.")
    return key


def integer_control(value: Any, *, name: str, minimum: int | None = None) -> int:
    """Validate an integer-valued public algorithm control."""
    return validate_integer(value, name=name, minimum=minimum)


def nonnegative_control(value: Any, *, name: str) -> float:
    """Validate a finite nonnegative public algorithm control."""
    return validate_nonnegative(value, name=name)


def positive_control(value: Any, *, name: str) -> float:
    """Validate a finite positive public algorithm control."""
    return validate_positive(value, name=name)


def interval_control(
    value: Any,
    *,
    name: str,
    lower: float,
    upper: float,
    lower_closed: bool = True,
    upper_closed: bool = True,
) -> float:
    """Validate a finite scalar in a declared interval."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not a boolean.")
    result = float(value)
    if not bool(jnp.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    lower_ok = result >= lower if lower_closed else result > lower
    upper_ok = result <= upper if upper_closed else result < upper
    if not lower_ok or not upper_ok:
        left = "[" if lower_closed else "("
        right = "]" if upper_closed else ")"
        raise ValueError(f"{name} must lie in {left}{lower}, {upper}{right}.")
    return result


def tree_all_finite(values: Any) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in jax.tree_util.tree_leaves(values)
    )


def tree_contains_tracer(values: Any) -> bool:
    """Return whether a pytree contains a value currently traced by JAX."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(values))


def validate_manifold_point(manifold: Any, point: Any, *, name: str) -> Any:
    """Eagerly validate a point while leaving traced numerical kernels composable."""
    if tree_contains_tracer(point):
        return point
    if not tree_all_finite(point):
        raise ValueError(f"{name} must contain only finite values.")
    try:
        membership = jnp.asarray(manifold.belongs(point), dtype=bool)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} has an invalid manifold-point structure: {exc}") from exc
    if not bool(jnp.all(membership)):
        raise ValueError(f"{name} must belong to {type(manifold).__name__}.")
    return point


def validate_tangent_vector(
    manifold: Any,
    base_point: Any,
    tangent: Any,
    *,
    name: str,
) -> Any:
    """Eagerly validate a tangent vector and its Product pytree structure."""
    if tree_contains_tracer((base_point, tangent)):
        return tangent
    if not tree_all_finite(tangent):
        raise ValueError(f"{name} must contain only finite values.")
    try:
        tangent_check = jnp.asarray(
            manifold.is_tangent(base_point, tangent),
            dtype=bool,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} has an invalid tangent-vector structure: {exc}") from exc
    if not bool(jnp.all(tangent_check)):
        raise ValueError(f"{name} must lie in the tangent space of {type(manifold).__name__}.")
    return tangent


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
    if any(leaf.ndim < 1 for leaf in leaves):
        raise ValueError("embedding leaves must have a leading sample dimension.")
    if any(jnp.iscomplexobj(leaf) for leaf in leaves):
        raise ValueError("embedding leaves must be real-valued.")
    if not tree_contains_tracer(leaves) and any(
        not bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves
    ):
        raise ValueError("embedding must contain finite coordinates only.")
    n_samples = leaves[0].shape[0]
    if any(leaf.shape[0] != n_samples for leaf in leaves):
        raise ValueError("embedding leaves must share their leading sample dimension.")
    return jnp.concatenate([leaf.reshape((n_samples, -1)) for leaf in leaves], axis=-1)


__all__ = [
    "as_real_array",
    "as_key",
    "deterministic_sign_columns",
    "event_shapes",
    "flatten_embedding",
    "flatten_geometry_values",
    "integer_control",
    "interval_control",
    "nonnegative_control",
    "normalize_weights",
    "positive_control",
    "require_unbatched",
    "scale_tangent",
    "scale_tangent_samples",
    "stack_points",
    "take_point",
    "take_samples",
    "tree_all_finite",
    "tree_contains_tracer",
    "unflatten_geometry",
    "validate_manifold_point",
    "validate_tangent_vector",
    "weighted_tangent_sum",
]
