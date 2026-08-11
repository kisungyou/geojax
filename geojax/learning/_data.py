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
from geojax.geometry._numerics import stable_norm

from ._utils import (
    as_real_array,
    deterministic_sign_columns,
    event_shapes,
    integer_control,
    stack_points,
    tree_all_finite,
)

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
    if not isinstance(representation, str):
        raise TypeError("representation must be a string.")
    if not representation or representation == "canonical":
        raise ValueError("representation must be a nonempty, non-canonical name.")
    if not callable(adapter):
        raise TypeError("adapter must be callable.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
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
    matrix = as_real_array(matrix, name="matrix")
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _as_real_floating_tree(values: Any) -> Any:
    """Convert canonical data leaves to floating JAX arrays without data loss."""

    def convert(value: Any) -> Any:
        array = jnp.asarray(value)
        if jnp.iscomplexobj(array):
            raise ValueError("Manifold data must be real-valued; complex data are unsupported.")
        return array.astype(jnp.result_type(array, float))

    return jax.tree_util.tree_map(convert, values)


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


def _require_full_column_rank(matrix: Array, shape: tuple[int, int], name: str) -> Array:
    matrix = as_real_array(matrix, name=name)
    if matrix.shape[-2:] != shape:
        raise ValueError(f"{name} must end in {shape}.")
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError(f"{name} must contain only finite values.")
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    largest = jnp.max(singular_values, axis=-1)
    tolerance = jnp.finfo(matrix.dtype).eps * max(shape) * largest
    if bool(jnp.any(singular_values[..., -1] <= tolerance)):
        raise ValueError(f"{name} must have full column rank.")
    return matrix


def _require_cholesky_factor(matrix: Array, shape: tuple[int, int]) -> Array:
    matrix = as_real_array(matrix, name="Cholesky factors")
    if matrix.shape[-2:] != shape:
        raise ValueError(f"Cholesky factors must end in {shape}.")
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError("Cholesky factors must contain only finite values.")
    scale = jnp.maximum(
        jnp.max(jnp.abs(matrix), axis=(-2, -1)),
        jnp.finfo(matrix.dtype).tiny,
    )
    tolerance = 100.0 * jnp.finfo(matrix.dtype).eps * scale
    upper = jnp.triu(matrix, k=1)
    if bool(jnp.any(jnp.max(jnp.abs(upper), axis=(-2, -1)) > tolerance)):
        raise ValueError("Cholesky factors must be lower triangular.")
    if bool(jnp.any(jnp.diagonal(matrix, axis1=-2, axis2=-1) <= 0.0)):
        raise ValueError("Cholesky factors must have a positive diagonal.")
    return matrix


def _require_symmetric(matrix: Array, shape: tuple[int, int], name: str) -> Array:
    matrix = as_real_array(matrix, name=name)
    if matrix.shape[-2:] != shape:
        raise ValueError(f"{name} must end in {shape}.")
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError(f"{name} must contain only finite values.")
    scale = jnp.maximum(
        jnp.max(jnp.abs(matrix), axis=(-2, -1)),
        jnp.finfo(matrix.dtype).tiny,
    )
    tolerance = 100.0 * jnp.finfo(matrix.dtype).eps * scale
    residual = jnp.max(jnp.abs(matrix - jnp.swapaxes(matrix, -1, -2)), axis=(-2, -1))
    if bool(jnp.any(residual > tolerance)):
        raise ValueError(f"{name} must be symmetric.")
    return _sym(matrix)


def _require_positive_definite(matrix: Array, name: str) -> Array:
    """Require batched symmetric matrices to lie in the open SPD cone."""
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    if not bool(jnp.all(jnp.isfinite(eigenvalues))) or bool(jnp.any(eigenvalues[..., 0] <= 0.0)):
        raise ValueError(f"{name} must be positive definite.")
    return matrix


def _rotation_from_axis_angle(vector: Array) -> Array:
    vector = jnp.asarray(vector)
    if vector.shape[-1:] != (3,):
        raise ValueError("axis-angle coordinates must end in (3,).")
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zeros = jnp.zeros_like(x)
    skew = jnp.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )
    theta2 = jnp.sum(vector * vector, axis=-1, keepdims=True)[..., None]
    cutoff = jnp.sqrt(jnp.finfo(jnp.result_type(vector, float)).eps)
    regular = theta2 > cutoff
    # Keep the inactive Rodrigues branch finite at zero so reverse-mode AD does
    # not encounter the undefined derivative of sqrt(0) or a hidden 0 / 0.
    theta = jnp.sqrt(jnp.where(regular, theta2, 1.0))
    safe_theta2 = jnp.where(regular, theta2, 1.0)
    a = jnp.where(
        regular,
        jnp.sin(theta) / theta,
        1.0 - theta2 / 6.0 + theta2**2 / 120.0,
    )
    b = jnp.where(
        regular,
        (1.0 - jnp.cos(theta)) / safe_theta2,
        0.5 - theta2 / 24.0 + theta2**2 / 720.0,
    )
    identity = jnp.eye(3, dtype=vector.dtype)
    return identity + a * skew + b * (skew @ skew)


def _rotation_from_quaternion(quaternion: Array) -> Array:
    quaternion = jnp.asarray(quaternion)
    if quaternion.shape[-1:] != (4,):
        raise ValueError("quaternions must use trailing shape (4,) in (w, x, y, z) order.")
    norm = stable_norm(quaternion, axis=-1, keepdims=True)
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
        norms = jnp.linalg.norm(pairs, axis=-1)
        if not bool(jnp.all(jnp.isfinite(norms))) or bool(jnp.any(norms <= 0.0)):
            raise ValueError("unit-circle pairs must be finite and nonzero.")
        return manifold.wrap(jnp.arctan2(pairs[..., 1], pairs[..., 0]))

    if isinstance(manifold, Hyperboloid) and representation in {"poincare", "poincare_ball"}:
        point = jnp.asarray(values)
        if point.shape[-1:] != (manifold.dim,):
            raise ValueError(f"Poincare coordinates must end in ({manifold.dim},).")
        squared_radius = jnp.sum(point * point, axis=-1)
        if not bool(jnp.all(jnp.isfinite(squared_radius))) or bool(jnp.any(squared_radius >= 1.0)):
            raise ValueError("Poincare coordinates must lie in the open unit ball.")
        return manifold.from_poincare(point)
    if isinstance(manifold, PoincareBall) and representation == "hyperboloid":
        point = jnp.asarray(values)
        if point.shape[-1:] != (manifold.size + 1,):
            raise ValueError(f"hyperboloid coordinates must end in ({manifold.size + 1},).")
        source = Hyperboloid(manifold.size + 1)
        if not bool(jnp.all(source.belongs(point))):
            raise ValueError("hyperboloid coordinates must lie on the upper unit hyperboloid.")
        return point[..., 1:] / (point[..., :1] + 1.0)

    if isinstance(manifold, ProbabilitySimplex):
        array = as_real_array(values, name="simplex representation")
        if representation in {"positive", "positive_weights"}:
            if not bool(jnp.all(jnp.isfinite(array))) or bool(jnp.any(array <= 0.0)):
                raise ValueError("positive weights must be finite and strictly positive.")
            total = jnp.sum(array, axis=-1, keepdims=True)
            if not bool(jnp.all(jnp.isfinite(total))) or bool(jnp.any(total <= 0.0)):
                scale = jnp.max(array, axis=-1, keepdims=True)
                scaled = array / scale
                total = jnp.sum(scaled, axis=-1, keepdims=True)
                return scaled / total
            return array / total
        if representation == "logits":
            if not bool(jnp.all(jnp.isfinite(array))):
                raise ValueError("logits must contain only finite values.")
            probabilities = jax.nn.softmax(array, axis=-1)
            floor = jnp.finfo(probabilities.dtype).tiny
            probabilities = jnp.maximum(probabilities, floor)
            return probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)

    if isinstance(manifold, (Grassmann, GrassmannProjection, GeneralizedGrassmann)):
        if representation in {"basis", "spanning_basis"}:
            basis = _require_full_column_rank(values, manifold.shape, "spanning bases")
            return (
                manifold.project(basis)
                if isinstance(manifold, GeneralizedGrassmann)
                else _orthonormalize(basis)
            )
        if representation == "projector" and not isinstance(manifold, GeneralizedGrassmann):
            projector_shape = (manifold.shape[0], manifold.shape[0])
            projector = _require_symmetric(values, projector_shape, "projectors")
            residual = projector @ projector - projector
            scale = jnp.maximum(jnp.max(jnp.abs(projector), axis=(-2, -1)), 1.0)
            tolerance = 200.0 * jnp.finfo(projector.dtype).eps * projector_shape[0] * scale
            idempotence = jnp.max(jnp.abs(residual), axis=(-2, -1))
            trace_error = jnp.abs(jnp.trace(projector, axis1=-2, axis2=-1) - manifold.rank)
            if bool(jnp.any(idempotence > tolerance)) or bool(
                jnp.any(trace_error > tolerance * projector_shape[0])
            ):
                raise ValueError("projectors must be rank-k symmetric idempotent matrices.")
            return _projector_to_frame(projector, manifold.rank)

    if isinstance(manifold, (SPDLogEuclidean, SPDAffineInvariant, SPDBuresWasserstein)):
        array = jnp.asarray(values)
        if representation in {"cholesky", "cholesky_factor"}:
            array = _require_cholesky_factor(array, manifold.shape)
            return array @ jnp.swapaxes(array, -1, -2)
        if representation in {"log", "log_matrix"}:
            array = _require_symmetric(array, manifold.shape, "log-matrices")
            return _matrix_expm_symmetric(array)

    if isinstance(manifold, (CorrelationECM, CorrelationLEC, CorrelationAffineQuotient)):
        array = jnp.asarray(values)
        if representation in {"cholesky", "cholesky_factor"}:
            array = _require_cholesky_factor(array, manifold.shape)
            array = array @ jnp.swapaxes(array, -1, -2)
        if representation in {"cholesky", "cholesky_factor", "covariance"}:
            array = _require_symmetric(array, manifold.shape, "covariance matrices")
            array = _require_positive_definite(array, "covariance matrices")
            raw_diagonal = jnp.diagonal(array, axis1=-2, axis2=-1)
            diagonal = jnp.sqrt(raw_diagonal)
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
                skew = jnp.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1).reshape(
                    omega.shape[:-1] + (3, 3)
                )
                tangent = manifold.tangent_from_components(skew, twist[..., 3:])
            else:
                raise ValueError("twist coordinates are built in only for SE(2) and SE(3).")
            return manifold.group_exp(tangent)

    if isinstance(manifold, KendallShape) and representation in {"landmarks", "raw_landmarks"}:
        return manifold.project(values)

    if isinstance(manifold, FixedRank) and representation in {"svd", "svd_factors"}:
        if not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError("SVD factors must be supplied as (U, singular_values, Vh).")
        left = as_real_array(values[0], name="SVD left factors")
        singular_values = as_real_array(values[1], name="SVD singular values")
        right = as_real_array(values[2], name="SVD right factors")
        if left.shape[-2:] != (manifold.m, manifold.rank):
            raise ValueError(f"SVD left factors must end in ({manifold.m}, {manifold.rank}).")
        if singular_values.shape[-1:] != (manifold.rank,):
            raise ValueError(f"SVD singular values must end in ({manifold.rank},).")
        if right.shape[-2:] != (manifold.rank, manifold.n):
            raise ValueError(f"SVD right factors must end in ({manifold.rank}, {manifold.n}).")
        batch_shapes = (
            left.shape[:-2],
            singular_values.shape[:-1],
            right.shape[:-2],
        )
        if len(set(batch_shapes)) != 1:
            raise ValueError("SVD factors must have identical leading batch shapes.")
        if not tree_all_finite((left, singular_values, right)):
            raise ValueError("SVD factors must contain only finite values.")
        if bool(jnp.any(singular_values <= 0.0)):
            raise ValueError("SVD singular values must be strictly positive.")
        left_gram = jnp.swapaxes(left, -1, -2) @ left
        right_gram = right @ jnp.swapaxes(right, -1, -2)
        identity = jnp.eye(manifold.rank, dtype=left.dtype)
        tolerance = 200.0 * jnp.finfo(left.dtype).eps * max(manifold.shape)
        if bool(jnp.any(jnp.max(jnp.abs(left_gram - identity), axis=(-2, -1)) > tolerance)):
            raise ValueError("SVD left factors must have orthonormal columns.")
        if bool(jnp.any(jnp.max(jnp.abs(right_gram - identity), axis=(-2, -1)) > tolerance)):
            raise ValueError("SVD right factors must have orthonormal rows.")
        return (left * singular_values[..., None, :]) @ right

    if isinstance(manifold, (RankKPSD, RankKPSDBuresWasserstein, Elliptope, Spectrahedron)):
        if representation in {"factor", "low_rank_factor"}:
            factor = _require_full_column_rank(
                values,
                (manifold.n, manifold.rank),
                "low-rank factors",
            )
            tolerance = 200.0 * jnp.finfo(factor.dtype).eps * manifold.n
            if isinstance(manifold, Elliptope):
                row_norms_squared = jnp.sum(factor * factor, axis=-1)
                if bool(jnp.any(jnp.abs(row_norms_squared - 1.0) > tolerance)):
                    raise ValueError("Elliptope factors must have unit-norm rows.")
            if isinstance(manifold, Spectrahedron):
                trace = jnp.sum(factor * factor, axis=(-2, -1))
                if bool(jnp.any(jnp.abs(trace - 1.0) > tolerance)):
                    raise ValueError("Spectrahedron factors must have squared Frobenius norm one.")
            matrix = factor @ jnp.swapaxes(factor, -1, -2)
            return matrix

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
        if not isinstance(rep, str):
            raise TypeError("Each Product representation must be a string.")
        converted = _convert_representation(factor, leaf, str(rep))
        event_ndim = len(factor.shape)
        if axis is not None:
            axis = integer_control(axis, name="sample_axis")
            converted = jnp.moveaxis(converted, axis, -(event_ndim + 1))
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
    if sample_counts[0] < 1:
        raise ValueError("Manifold data must contain at least one observation.")
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
    if not isinstance(repair, bool):
        raise TypeError("repair must be a boolean.")
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
    if repair and check != "belongs":
        raise ValueError("repair=True requires check='belongs'.")
    if representation == "point_sequence":
        if not isinstance(values, (tuple, list)):
            raise TypeError("point_sequence representation requires a Python sequence.")
        values = stack_points(manifold, list(values))
        representation = "canonical"
        sample_axis = None

    if isinstance(manifold, Product):
        canonical = _canonicalize_product(manifold, values, representation, sample_axis)
    else:
        if not isinstance(representation, str):
            raise TypeError("representation must be a string.")
        canonical = _convert_representation(manifold, values, representation)
        if sample_axis is not None:
            sample_axis = integer_control(sample_axis, name="sample_axis")
            canonical = jnp.moveaxis(
                canonical,
                sample_axis,
                -(len(manifold.shape) + 1),
            )
    canonical = _as_real_floating_tree(canonical)
    n_samples, batch_shape = _layout_metadata(manifold, canonical)

    messages: list[str] = []
    invalid_count = 0
    repaired_count = 0
    if check in {"finite", "belongs"} and not tree_all_finite(canonical):
        raise ValueError("Manifold data contain NaN or infinite values.")
    if check == "belongs":
        membership = jnp.asarray(manifold.belongs(canonical), dtype=bool)
        expected_membership_shape = batch_shape + (n_samples,)
        if membership.shape != expected_membership_shape:
            raise ValueError(
                f"{type(manifold).__name__}.belongs returned shape {membership.shape}; "
                f"expected {expected_membership_shape} for the adapted data layout."
            )
        invalid_count = int(jnp.sum(~membership))
        if invalid_count and repair:
            canonical = manifold.project(canonical)
            if not tree_all_finite(canonical):
                raise ValueError("project() produced NaN or infinite repaired values.")
            repaired_count = invalid_count
            membership = jnp.asarray(manifold.belongs(canonical), dtype=bool)
            if membership.shape != expected_membership_shape:
                raise ValueError(
                    f"{type(manifold).__name__}.belongs returned shape {membership.shape} "
                    f"after repair; expected {expected_membership_shape}."
                )
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
    except (TypeError, ValueError, FloatingPointError) as exc:
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
