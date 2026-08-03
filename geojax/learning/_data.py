"""Canonical data adaptation for manifold-valued learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from geojax.geometry import (
    CorrelationAffineQuotient,
    CorrelationECM,
    CorrelationLEC,
    Elliptope,
    FixedRank,
    GeneralizedGrassmann,
    Grassmann,
    GrassmannProjection,
    Hyperboloid,
    KendallShape,
    PoincareBall,
    ProbabilitySimplex,
    Product,
    RankKPSD,
    RankKPSDBuresWasserstein,
    SPDAffineInvariant,
    SPDBuresWasserstein,
    SPDLogEuclidean,
    SpecialEuclidean,
    SpecialOrthogonal,
    Spectrahedron,
    Sphere,
    SphereExtrinsic,
    Torus,
)

from ._utils import deterministic_sign_columns, event_shapes, stack_points, tree_all_finite

Array = Any
Adapter = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class DataValidationReport:
    """Eager validation summary for one adapted manifold dataset."""

    valid: bool
    check: str
    n_samples: int
    batch_shape: tuple[int, ...]
    invalid_count: int = 0
    repaired_count: int = 0
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManifoldData:
    """Canonical observations bound to the geometry that validated them.

    Binding prevents a validated dataset from being silently reused under a
    different metric or representation with the same event shape. Reusing the
    object with its original geometry skips checks that are already at least as
    strong as the requested validation level.
    """

    manifold: Any
    values: Any
    n_samples: int
    batch_shape: tuple[int, ...]
    event_shapes: Any
    report: DataValidationReport


@runtime_checkable
class ManifoldDataAdapterProtocol(Protocol):
    """Callable protocol for user-registered representation adapters."""

    def __call__(self, manifold: Any, values: Any) -> Any:
        """Convert ``values`` to the geometry's canonical point representation."""
        ...


_ADAPTERS: dict[tuple[type, str], Adapter] = {}


def register_manifold_data_adapter(
    geometry_type: type,
    representation: str,
    adapter: ManifoldDataAdapterProtocol,
    *,
    overwrite: bool = False,
) -> None:
    """Register an explicit representation converter for a geometry class."""
    if not isinstance(geometry_type, type):
        raise TypeError("geometry_type must be a class.")
    if not representation or representation == "canonical":
        raise ValueError("representation must be a nonempty, non-canonical name.")
    if not callable(adapter):
        raise TypeError("adapter must be callable.")
    key = (geometry_type, str(representation))
    if key in _ADAPTERS and not overwrite:
        raise ValueError(f"An adapter is already registered for {key!r}.")
    _ADAPTERS[key] = adapter


def _registered_adapter(manifold: Any, representation: str) -> Adapter | None:
    for cls in type(manifold).__mro__:
        adapter = _ADAPTERS.get((cls, representation))
        if adapter is not None:
            return adapter
    return None


def _sym(matrix: Array) -> Array:
    matrix = jnp.asarray(matrix)
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _matrix_expm_symmetric(matrix: Array) -> Array:
    values, vectors = jnp.linalg.eigh(_sym(matrix))
    return (vectors * jnp.exp(values)[..., None, :]) @ jnp.swapaxes(vectors, -1, -2)


def _hyperspherical_to_cartesian(angles: Array, size: int) -> Array:
    angles = jnp.asarray(angles)
    if angles.shape[-1:] != (size - 1,):
        raise ValueError(f"hyperspherical angles must end in ({size - 1},).")
    coordinates = []
    sine_product = jnp.ones(angles.shape[:-1], dtype=angles.dtype)
    for index in range(size - 1):
        coordinates.append(sine_product * jnp.cos(angles[..., index]))
        sine_product = sine_product * jnp.sin(angles[..., index])
    coordinates.append(sine_product)
    return jnp.stack(coordinates, axis=-1)


def _orthonormalize(matrix: Array) -> Array:
    q, r = jnp.linalg.qr(jnp.asarray(matrix), mode="reduced")
    diagonal = jnp.diagonal(r, axis1=-2, axis2=-1)
    signs = jnp.where(diagonal < 0.0, -1.0, 1.0)
    return deterministic_sign_columns(q * signs[..., None, :])


def _projector_to_frame(matrix: Array, k: int) -> Array:
    values, vectors = jnp.linalg.eigh(_sym(matrix))
    del values
    selected = vectors[..., :, ::-1][..., :k]
    return deterministic_sign_columns(selected)


def _rotation_from_axis_angle(vector: Array) -> Array:
    vector = jnp.asarray(vector)
    if vector.shape[-1:] != (3,):
        raise ValueError("axis-angle coordinates must end in (3,).")
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zeros = jnp.zeros_like(x)
    skew = jnp.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1
    ).reshape(vector.shape[:-1] + (3, 3))
    theta2 = jnp.sum(vector * vector, axis=-1, keepdims=True)[..., None]
    theta = jnp.sqrt(theta2)
    a = jnp.where(theta2 > 1e-12, jnp.sin(theta) / theta, 1.0 - theta2 / 6.0)
    b = jnp.where(
        theta2 > 1e-12,
        (1.0 - jnp.cos(theta)) / theta2,
        0.5 - theta2 / 24.0,
    )
    identity = jnp.eye(3, dtype=vector.dtype)
    return identity + a * skew + b * (skew @ skew)


def _rotation_from_quaternion(quaternion: Array) -> Array:
    quaternion = jnp.asarray(quaternion)
    if quaternion.shape[-1:] != (4,):
        raise ValueError("quaternions must use trailing shape (4,) in (w, x, y, z) order.")
    norm = jnp.linalg.norm(quaternion, axis=-1, keepdims=True)
    if bool(jnp.any(norm <= 0.0)):
        raise ValueError("quaternions must be nonzero.")
    w, x, y, z = jnp.moveaxis(quaternion / norm, -1, 0)
    return jnp.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _convert_representation(manifold: Any, values: Any, representation: str) -> Any:
    custom = _registered_adapter(manifold, representation)
    if custom is not None:
        return custom(manifold, values)
    if representation == "canonical":
        return jnp.asarray(values)

    if isinstance(manifold, (Sphere, SphereExtrinsic)):
        if representation in {"hyperspherical", "angles"}:
            return _hyperspherical_to_cartesian(values, manifold.size)

    if isinstance(manifold, Torus) and representation in {"unit_circle", "circle_pairs"}:
        pairs = jnp.asarray(values)
        if pairs.shape[-2:] != (manifold.size, 2):
            raise ValueError(f"unit-circle pairs must end in ({manifold.size}, 2).")
        return jnp.arctan2(pairs[..., 1], pairs[..., 0])

    if isinstance(manifold, Hyperboloid) and representation in {"poincare", "poincare_ball"}:
        return manifold.from_poincare(values)
    if isinstance(manifold, PoincareBall) and representation == "hyperboloid":
        point = jnp.asarray(values)
        if point.shape[-1:] != (manifold.size + 1,):
            raise ValueError(f"hyperboloid coordinates must end in ({manifold.size + 1},).")
        return point[..., 1:] / (point[..., :1] + 1.0)

    if isinstance(manifold, ProbabilitySimplex):
        array = jnp.asarray(values)
        if representation in {"positive", "positive_weights"}:
            if bool(jnp.any(array < 0.0)):
                raise ValueError("positive weights must be nonnegative.")
            total = jnp.sum(array, axis=-1, keepdims=True)
            if bool(jnp.any(total <= 0.0)):
                raise ValueError("positive weights must have positive row sums.")
            return array / total
        if representation == "logits":
            return jax.nn.softmax(array, axis=-1)

    if isinstance(manifold, (Grassmann, GrassmannProjection, GeneralizedGrassmann)):
        if representation in {"basis", "spanning_basis"}:
            return manifold.project(values) if isinstance(manifold, GeneralizedGrassmann) else _orthonormalize(values)
        if representation == "projector":
            return _projector_to_frame(values, manifold.rank)

    if isinstance(manifold, (SPDLogEuclidean, SPDAffineInvariant, SPDBuresWasserstein)):
        array = jnp.asarray(values)
        if representation in {"cholesky", "cholesky_factor"}:
            return array @ jnp.swapaxes(array, -1, -2)
        if representation in {"log", "log_matrix"}:
            return _matrix_expm_symmetric(array)

    if isinstance(manifold, (CorrelationECM, CorrelationLEC, CorrelationAffineQuotient)):
        array = jnp.asarray(values)
        if representation in {"cholesky", "cholesky_factor"}:
            array = array @ jnp.swapaxes(array, -1, -2)
        if representation in {"cholesky", "cholesky_factor", "covariance"}:
            diagonal = jnp.sqrt(jnp.maximum(jnp.diagonal(array, axis1=-2, axis2=-1), manifold.eps))
            return array / (diagonal[..., :, None] * diagonal[..., None, :])

    if isinstance(manifold, SpecialOrthogonal):
        if representation == "angle" and manifold.n == 2:
            angle = jnp.asarray(values)
            if angle.shape[-1:] == (1,):
                angle = angle[..., 0]
            cosine, sine = jnp.cos(angle), jnp.sin(angle)
            return jnp.stack([cosine, -sine, sine, cosine], axis=-1).reshape(angle.shape + (2, 2))
        if representation in {"axis_angle", "rotation_vector"} and manifold.n == 3:
            return _rotation_from_axis_angle(values)
        if representation == "quaternion" and manifold.n == 3:
            return _rotation_from_quaternion(values)

    if isinstance(manifold, SpecialEuclidean):
        if representation in {"components", "rotation_translation"}:
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                raise ValueError("rotation_translation input must be (rotation, translation).")
            return manifold.from_components(values[0], values[1])
        if representation == "twist":
            twist = jnp.asarray(values)
            if twist.shape[-1:] != (manifold.dim,):
                raise ValueError(f"twists must end in ({manifold.dim},).")
            if manifold.n == 2:
                omega = twist[..., 0]
                skew = jnp.zeros(twist.shape[:-1] + (2, 2), dtype=twist.dtype)
                skew = skew.at[..., 0, 1].set(-omega)
                skew = skew.at[..., 1, 0].set(omega)
                tangent = manifold.tangent_from_components(skew, twist[..., 1:])
            elif manifold.n == 3:
                omega = twist[..., :3]
                x, y, z = omega[..., 0], omega[..., 1], omega[..., 2]
                zeros = jnp.zeros_like(x)
                skew = jnp.stack(
                    [zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1
                ).reshape(omega.shape[:-1] + (3, 3))
                tangent = manifold.tangent_from_components(skew, twist[..., 3:])
            else:
                raise ValueError("twist coordinates are built in only for SE(2) and SE(3).")
            return manifold.group_exp(tangent)

    if isinstance(manifold, KendallShape) and representation in {"landmarks", "raw_landmarks"}:
        return manifold.project(values)

    if isinstance(manifold, FixedRank) and representation in {"svd", "svd_factors"}:
        if not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError("SVD factors must be supplied as (U, singular_values, Vh).")
        left, singular_values, right = map(jnp.asarray, values)
        return (left * singular_values[..., None, :]) @ right

    if isinstance(manifold, (RankKPSD, RankKPSDBuresWasserstein, Elliptope, Spectrahedron)):
        if representation in {"factor", "low_rank_factor"}:
            factor = jnp.asarray(values)
            matrix = factor @ jnp.swapaxes(factor, -1, -2)
            return manifold.project(matrix) if isinstance(manifold, (Elliptope, Spectrahedron)) else matrix

    valid = "canonical"
    raise ValueError(
        f"Unsupported representation {representation!r} for {type(manifold).__name__}; "
        f"use {valid!r} or register an adapter."
    )


def _tree_option(manifold: Product, option: Any, default: Any) -> list[Any]:
    factors, factor_tree = jax.tree_util.tree_flatten(manifold.factors)
    if option is default or isinstance(option, (str, int)) or option is None:
        return [option] * len(factors)
    leaves, option_tree = jax.tree_util.tree_flatten(option)
    if option_tree != factor_tree:
        raise ValueError("Product representation/sample_axis must match the factor pytree.")
    return list(leaves)


def _canonicalize_product(
    manifold: Product,
    values: Any,
    representation: Any,
    sample_axis: Any,
) -> Any:
    factors, factor_tree = jax.tree_util.tree_flatten(manifold.factors)
    leaves, value_tree = jax.tree_util.tree_flatten(values)
    if value_tree != factor_tree:
        raise ValueError("values must match the Product factor pytree.")
    representations = _tree_option(manifold, representation, "canonical")
    axes = _tree_option(manifold, sample_axis, None)
    out = []
    for factor, leaf, rep, axis in zip(factors, leaves, representations, axes):
        converted = _convert_representation(factor, leaf, str(rep))
        event_ndim = len(factor.shape)
        if axis is not None:
            converted = jnp.moveaxis(converted, int(axis), -(event_ndim + 1))
        out.append(converted)
    return jax.tree_util.tree_unflatten(factor_tree, out)


def _layout_metadata(manifold: Any, values: Any) -> tuple[int, tuple[int, ...]]:
    if isinstance(manifold, Product):
        factors, factor_tree = jax.tree_util.tree_flatten(manifold.factors)
        leaves, value_tree = jax.tree_util.tree_flatten(values)
        if factor_tree != value_tree:
            raise ValueError("values must match the Product factor pytree.")
    else:
        factors, leaves = [manifold], [values]
    sample_counts: list[int] = []
    batch_shapes: list[tuple[int, ...]] = []
    for factor, leaf in zip(factors, leaves):
        array = jnp.asarray(leaf)
        shape = tuple(factor.shape)
        event_ndim = len(shape)
        if array.ndim <= event_ndim or tuple(array.shape[-event_ndim:]) != shape:
            raise ValueError(
                f"Data for {type(factor).__name__} must end in event shape {shape} "
                f"and include a sample axis; received {array.shape}."
            )
        sample_counts.append(int(array.shape[-event_ndim - 1]))
        batch_shapes.append(tuple(array.shape[: -event_ndim - 1]))
    if len(set(sample_counts)) != 1:
        raise ValueError(f"Product leaves have inconsistent sample counts: {sample_counts}.")
    if len(set(batch_shapes)) != 1:
        raise ValueError(f"Product leaves have inconsistent batch shapes: {batch_shapes}.")
    return sample_counts[0], batch_shapes[0]


def as_manifold_data(
    manifold: Any,
    values: Any,
    *,
    sample_axis: Any = None,
    representation: Any = "canonical",
    check: str = "belongs",
    repair: bool = False,
) -> ManifoldData:
    """Convert manifold observations to the canonical learning-data layout.

    ``sample_axis=None`` denotes the axis immediately before each geometry's
    event dimensions. Product representations and axes may be pytrees matching
    ``manifold.factors``. Python sequences of complete points require
    ``representation='point_sequence'`` so their interpretation is explicit.
    """
    if isinstance(values, ManifoldData):
        if values.manifold is not manifold:
            raise ValueError(
                "ManifoldData is bound to a different geometry instance; "
                "adapt values.values explicitly for the new geometry."
            )
        if representation != "canonical" or sample_axis is not None or repair:
            raise ValueError("Already-adapted ManifoldData cannot be converted again.")
        validation_order = {"shape": 0, "finite": 1, "belongs": 2}
        if check not in validation_order:
            raise ValueError("check must be 'shape', 'finite', or 'belongs'.")
        if validation_order[values.report.check] >= validation_order[check]:
            return values
        return as_manifold_data(manifold, values.values, check=check)
    if check not in {"shape", "finite", "belongs"}:
        raise ValueError("check must be 'shape', 'finite', or 'belongs'.")
    if representation == "point_sequence":
        if not isinstance(values, (tuple, list)):
            raise TypeError("point_sequence representation requires a Python sequence.")
        values = stack_points(manifold, list(values))
        representation = "canonical"
        sample_axis = None

    if isinstance(manifold, Product):
        canonical = _canonicalize_product(manifold, values, representation, sample_axis)
    else:
        canonical = _convert_representation(manifold, values, str(representation))
        if sample_axis is not None:
            canonical = jnp.moveaxis(canonical, int(sample_axis), -(len(manifold.shape) + 1))
    n_samples, batch_shape = _layout_metadata(manifold, canonical)

    messages: list[str] = []
    invalid_count = 0
    repaired_count = 0
    if check in {"finite", "belongs"} and not tree_all_finite(canonical):
        raise ValueError("Manifold data contain NaN or infinite values.")
    if check == "belongs":
        membership = jnp.asarray(manifold.belongs(canonical), dtype=bool)
        invalid_count = int(jnp.sum(~membership))
        if invalid_count and repair:
            canonical = manifold.project(canonical)
            repaired_count = invalid_count
            membership = jnp.asarray(manifold.belongs(canonical), dtype=bool)
            invalid_count = int(jnp.sum(~membership))
            messages.append(f"Repaired {repaired_count} invalid point(s) with project().")
        if invalid_count:
            raise ValueError(
                f"{invalid_count} observation(s) do not belong to {type(manifold).__name__}; "
                "pass repair=True to project them explicitly."
            )
    report = DataValidationReport(
        valid=True,
        check=check,
        n_samples=n_samples,
        batch_shape=batch_shape,
        invalid_count=invalid_count,
        repaired_count=repaired_count,
        messages=tuple(messages),
    )
    return ManifoldData(
        manifold,
        canonical,
        n_samples,
        batch_shape,
        event_shapes(manifold),
        report,
    )


def check_manifold_data(
    manifold: Any,
    values: Any,
    *,
    sample_axis: Any = None,
    representation: Any = "canonical",
    check: str = "belongs",
) -> DataValidationReport:
    """Return a validation report without propagating data-validation errors."""
    try:
        return as_manifold_data(
            manifold,
            values,
            sample_axis=sample_axis,
            representation=representation,
            check=check,
        ).report
    except (TypeError, ValueError) as exc:
        return DataValidationReport(
            valid=False,
            check=check,
            n_samples=0,
            batch_shape=(),
            messages=(str(exc),),
        )


__all__ = [
    "DataValidationReport",
    "ManifoldData",
    "ManifoldDataAdapterProtocol",
    "as_manifold_data",
    "check_manifold_data",
    "register_manifold_data_adapter",
]
