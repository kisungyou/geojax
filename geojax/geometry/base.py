"""Shared geometry protocol and mixin helpers."""

from __future__ import annotations

import math
import operator
from typing import Any, Protocol, Sequence, Tuple, Union, runtime_checkable

import jax.numpy as jnp

from ._numerics import nonnegative

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


def as_sample_shape(sample_shape: Shape = ()) -> tuple[int, ...]:
    """Normalize an integer or tuple-like sample shape."""
    values = (sample_shape,) if isinstance(sample_shape, int) else tuple(sample_shape)
    parsed = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("sample_shape entries must be integers, not booleans.")
        try:
            dimension = operator.index(value)
        except TypeError as exc:
            raise TypeError("sample_shape entries must be integers.") from exc
        if dimension < 0:
            raise ValueError("sample_shape entries must be nonnegative.")
        parsed.append(int(dimension))
    return tuple(parsed)


def validate_nonnegative(value: float, *, name: str) -> float:
    """Return a finite nonnegative scalar or raise a precise error."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not a boolean.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def validate_positive(value: float, *, name: str) -> float:
    """Return a finite positive scalar or raise a precise error."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not a boolean.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def validate_integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    """Return an integer-valued dimension/control without truncating floats."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    result = int(result)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def validate_boolean(value: Any, *, name: str) -> bool:
    """Return a genuine boolean without accepting integer lookalikes."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def event_shape_matches(x: Array, event_shape: Sequence[int]) -> bool:
    """Return whether ``x`` ends in the declared manifold event shape.

    JAX array shapes are static while a function is traced, so this check is
    compatible with ``jax.jit``. Leading dimensions are intentionally ignored:
    they are batch dimensions and may be broadcast by individual operations.
    """
    array = jnp.asarray(x)
    shape = tuple(int(value) for value in event_shape)
    return (
        not jnp.iscomplexobj(array)
        and array.ndim >= len(shape)
        and tuple(array.shape[-len(shape) :]) == shape
    )


def false_for_event_shape(x: Array, event_shape: Sequence[int]) -> Array:
    """Return a false membership result with the inferable batch shape."""
    array = jnp.asarray(x)
    event_ndim = len(tuple(event_shape))
    batch_shape = array.shape[:-event_ndim] if array.ndim >= event_ndim else ()
    return jnp.zeros(batch_shape, dtype=bool)


def check_event_shape(x: Array, event_shape: Sequence[int], *, name: str = "array") -> Array:
    """Convert ``x`` to a JAX array and reject an incompatible event shape."""
    array = jnp.asarray(x)
    if jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued; complex arrays are unsupported.")
    shape = tuple(int(value) for value in event_shape)
    if not event_shape_matches(array, shape):
        raise ValueError(
            f"{name} must have trailing event shape {shape}; received shape {array.shape}."
        )
    return array


def scale_tangent(vector: Array, coefficient: float | Array, *, event_ndim: int) -> Array:
    """Scale an event-shaped tangent by scalar or leading-batch coefficients.

    ``coefficient`` follows the leading batch dimensions of ``vector``.  Event
    dimensions are never considered for coefficient broadcasting, which
    prevents a batch of steps from accidentally scaling event coordinates when
    a batch dimension happens to equal an event dimension.

    Trailing singleton dimensions beyond the available batch rank are accepted
    for compatibility with explicitly expanded coefficients.
    """
    value = jnp.asarray(vector)
    if isinstance(event_ndim, bool) or not isinstance(event_ndim, int):
        raise TypeError("event_ndim must be an integer.")
    if event_ndim < 0 or event_ndim > value.ndim:
        raise ValueError("event_ndim must be between zero and vector.ndim.")

    batch_shape = value.shape[:-event_ndim] if event_ndim else value.shape
    factor = jnp.asarray(coefficient)
    while factor.ndim > len(batch_shape) and factor.shape[-1] == 1:
        factor = jnp.squeeze(factor, axis=-1)
    if factor.ndim > len(batch_shape):
        raise ValueError(
            "coefficient must be scalar or broadcast over the tangent's "
            f"leading batch shape {batch_shape}; received shape {factor.shape}."
        )
    try:
        jnp.broadcast_shapes(batch_shape, factor.shape)
    except ValueError as exc:
        raise ValueError(
            "coefficient must be scalar or broadcast over the tangent's "
            f"leading batch shape {batch_shape}; received shape {factor.shape}."
        ) from exc
    return value * factor.reshape(factor.shape + (1,) * event_ndim)


def dtype_margin(
    x: Array,
    *,
    configured: float = 0.0,
    atol: float = 0.0,
    ulps: float = 32.0,
) -> float:
    """Return a positive, dtype-aware margin for open-set numerical repairs.

    ``configured`` preserves an explicitly requested floor, ``atol`` keeps a
    repaired point on the accepted side of strict membership checks, and the
    machine-precision term prevents margins from rounding away in float32.
    """
    configured = validate_nonnegative(configured, name="configured margin")
    atol = validate_nonnegative(atol, name="absolute tolerance")
    ulps = validate_nonnegative(ulps, name="ULP multiplier")
    array = jnp.asarray(x)
    dtype = jnp.result_type(array, float)
    return max(
        float(configured),
        2.0 * float(atol),
        float(ulps) * float(jnp.finfo(dtype).eps),
    )


@runtime_checkable
class ManifoldProtocol(Protocol):
    """Retraction-based interface consumed by manifold optimizers.

    This is the GeoJAX counterpart of Manopt's core manifold structure. Exact
    geodesic maps are deliberately not required.
    """

    size: Any
    shape: tuple[int, ...] | tuple[Any, ...]
    dim: int

    def belongs(self, x: Array, atol: float | None = None) -> Array:
        """Return one membership boolean per broadcast batch element."""
        ...

    def project(self, x: Array) -> Array:
        """Repair an ambient array with event shape ``shape`` into the manifold."""
        ...

    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array:
        """Return whether ``u`` satisfies the tangent constraints at ``x``."""
        ...

    def tangent_project(self, x: Array, u: Array) -> Array:
        """Project an ambient vector onto the represented tangent space at ``x``."""
        ...

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Evaluate the Riemannian inner product at ``x``."""
        ...

    def norm(self, x: Array, u: Array) -> Array:
        """Evaluate the Riemannian norm induced by ``inner``."""
        ...

    def lincomb(self, x: Array, *terms: Any) -> Array:
        """Form coefficient/vector pairs in the tangent space at ``x``."""
        ...

    def retr(self, x: Array, u: Array, t: float | Array = 1.0) -> Array:
        """Retract ``t * u`` from ``x`` to a manifold point."""
        ...

    def invretr(self, x: Array, y: Array) -> Array:
        """Return the documented local inverse-retraction displacement."""
        ...

    def transport(self, x: Array, y: Array, u: Array) -> Array:
        """Move ``u`` from ``T_x M`` to ``T_y M`` using the advertised transport."""
        ...

    def egrad_to_rgrad(self, x: Array, egrad: Array) -> Array:
        """Convert an ambient Euclidean gradient to the metric-dual tangent gradient."""
        ...

    def ehess_to_rhess(
        self,
        x: Array,
        egrad: Array,
        ehess_vec: Array,
        u: Array,
    ) -> Array:
        """Convert an ambient Hessian-vector product using the advertised capability."""
        ...

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        """Sample points with shape ``sample_shape + shape``."""
        ...

    def random_tangent(
        self,
        key: Array,
        x: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        """Sample tangent vectors, optionally normalizing before applying ``scale``."""
        ...


@runtime_checkable
class GeometryProtocol(ManifoldProtocol, Protocol):
    """Uniform geometry interface with capability-qualified named operations.

    Retraction-only geometries may satisfy this structural protocol through
    documented compatibility aliases. Use ``operation_kind`` to distinguish
    exact geodesic operations from numerical-local or retraction proxies.
    """

    exp_is_exact: bool
    log_is_exact: bool
    dist_is_exact: bool
    transport_is_isometric: bool
    transport_is_parallel: bool
    hessian_conversion_is_exact: bool
    riemannian_gradient_jvp_is_exact: bool

    def operation_kind(self, name: str) -> str:
        """Return the certified status of a named geometric operation."""
        ...

    def exp(self, x: Array, u: Array) -> Array:
        """Apply the capability-qualified exponential or retraction proxy."""
        ...

    def log(self, x: Array, y: Array) -> Array:
        """Apply the capability-qualified logarithm or inverse-retraction proxy."""
        ...

    def squared_dist(self, x: Array, y: Array) -> Array:
        """Return the squared capability-qualified point distance."""
        ...

    def dist(self, x: Array, y: Array) -> Array:
        """Return the capability-qualified point distance."""
        ...

    def pair_mean(self, x: Array, y: Array) -> Array:
        """Return the midpoint-like construction supported by the geometry."""
        ...


class GeometryMixin:
    """Default helpers for geometries with the GeoJAX protocol."""

    exp_is_exact = False
    log_is_exact = False
    dist_is_exact = False
    transport_is_isometric = False
    transport_is_parallel = False
    hessian_conversion_is_exact = False
    riemannian_gradient_jvp_is_exact = False

    def _event_shape(self) -> tuple[int, ...]:
        """Return the array event shape used by non-Product geometries."""
        shape = tuple(self.shape)
        if not shape or not all(isinstance(value, int) for value in shape):
            raise TypeError("Array geometry shape must be a nonempty tuple of integers.")
        return shape

    def _shape_matches(self, *arrays: Array) -> bool:
        """Return whether event shapes match and leading batch axes broadcast."""
        event_shape = self._event_shape()
        if not all(event_shape_matches(array, event_shape) for array in arrays):
            return False
        event_ndim = len(event_shape)
        try:
            jnp.broadcast_shapes(*(jnp.shape(array)[:-event_ndim] for array in arrays))
        except ValueError:
            return False
        return True

    def _shape_failure(self, x: Array) -> Array:
        """Return a false validation result for malformed array input."""
        return false_for_event_shape(x, self._event_shape())

    def _check_shape(self, x: Array, *, name: str = "array") -> Array:
        """Validate one array's event shape and return it as a JAX array."""
        return check_event_shape(x, self._event_shape(), name=name)

    def _check_shapes(self, *named_arrays: tuple[str, Array]) -> tuple[Array, ...]:
        """Validate event shapes and broadcast-compatible leading dimensions."""
        arrays = tuple(self._check_shape(array, name=name) for name, array in named_arrays)
        event_ndim = len(self._event_shape())
        try:
            jnp.broadcast_shapes(*(array.shape[:-event_ndim] for array in arrays))
        except ValueError as exc:
            shapes = ", ".join(
                f"{name}={array.shape}" for (name, _), array in zip(named_arrays, arrays)
            )
            raise ValueError(
                f"Leading batch dimensions are not broadcast-compatible: {shapes}."
            ) from exc
        return arrays

    def _scale_tangent(self, vector: Array, coefficient: float | Array) -> Array:
        """Apply a scalar or leading-batch coefficient to an array tangent."""
        return scale_tangent(
            vector,
            coefficient,
            event_ndim=len(self._event_shape()),
        )

    def normalize(self, x: Array) -> Array:
        """Compatibility alias that dispatches to :meth:`project`."""
        return self.project(x)

    def projection(self, x: Array, u: Array) -> Array:
        """Compatibility alias that dispatches to :meth:`tangent_project`."""
        return self.tangent_project(x, u)

    def proj(self, x: Array, u: Array) -> Array:
        """Compatibility alias that dispatches to :meth:`tangent_project`."""
        return self.tangent_project(x, u)

    def to_tangent(self, x: Array, u: Array) -> Array:
        """Compatibility alias that dispatches to :meth:`tangent_project`."""
        return self.tangent_project(x, u)

    def transp(self, x: Array, y: Array, u: Array) -> Array:
        """Compatibility alias that dispatches to :meth:`transport`."""
        return self.transport(x, y, u)

    def egrad2rgrad(self, x: Array, egrad: Array) -> Array:
        """Compatibility alias that dispatches to :meth:`egrad_to_rgrad`."""
        return self.egrad_to_rgrad(x, egrad)

    def operation_kind(self, name: str) -> str:
        """Describe the mathematical status of a geometric operation."""
        if name == "transport":
            if self.transport_is_parallel:
                return "parallel"
            if self.transport_is_isometric:
                return "isometric"
            return "vector"
        if name in {"hessian", "ehess_to_rhess"}:
            return "exact" if self.hessian_conversion_is_exact else "projection"
        if name == "rgrad_jvp":
            return "exact" if self.riemannian_gradient_jvp_is_exact else "projection"
        if name == "squared_dist":
            explicit_kind = getattr(self, "squared_dist_kind", None)
            return str(explicit_kind) if explicit_kind is not None else self.operation_kind("dist")
        if name not in {"exp", "log", "dist"}:
            raise ValueError(f"Unknown geometric operation: {name!r}.")
        explicit_kind = getattr(self, f"{name}_kind", None)
        if explicit_kind is not None:
            return str(explicit_kind)
        return "exact" if bool(getattr(self, f"{name}_is_exact")) else "proxy"

    def exp_batch(self, x: Array, us: Array) -> Array:
        """Compatibility wrapper for natively batched ``exp``."""
        return self.exp(x, us)

    def log_batch(self, x: Array, ys: Array) -> Array:
        """Compatibility wrapper for natively batched ``log``."""
        return self.log(x, ys)

    def dist_batch(self, x: Array, ys: Array) -> Array:
        """Compatibility wrapper for natively batched ``dist``."""
        return self.dist(x, ys)

    def squared_dist(self, x: Array, y: Array) -> Array:
        """Squared geodesic distance, evaluated without differentiating ``sqrt``.

        Geometries with a more direct or more stable formula should override
        this method.  The logarithm-based default is also meaningful for
        retraction geometries, where it inherits the documented proxy
        semantics of ``log`` and ``dist``.
        """
        tangent = self.log(x, y)
        return nonnegative(self.inner(x, tangent, tangent))

    def retr(self, x: Array, u: Array, t: float | Array = 1.0) -> Array:
        """Default retraction: use the exponential map."""
        return self.exp(x, self._scale_tangent(u, t))

    def invretr(self, x: Array, y: Array) -> Array:
        """Default inverse retraction: use the logarithmic map."""
        return self.log(x, y)

    def lincomb(self, x: Array, *terms: Any) -> Array:
        """Linear combination of tangent vectors, projected back to ``T_x M``."""
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = self._scale_tangent(vec, coeff)
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(x, out)

    def pair_mean(self, x: Array, y: Array) -> Array:
        """Midpoint-like construction from the available ``exp`` and ``log``.

        This is a geodesic midpoint only when both operations are exact and the
        selected logarithm is the unique minimizing one.
        """
        return self.exp(x, 0.5 * self.log(x, y))

    def ehess_to_rhess(self, x: Array, egrad: Array, ehess_vec: Array, u: Array) -> Array:
        """Default Hessian conversion: tangent-project the ambient Hessian-vector product."""
        del egrad, u
        return self.tangent_project(x, ehess_vec)


class RetractionGeometryMixin(GeometryMixin):
    """Compatibility maps for manifolds known only through a retraction.

    Subclasses implement ``retr`` and ``invretr``. ``exp``, ``log`` and ``dist``
    remain available for algorithms that need point differences, but their
    metadata identifies them as proxies rather than genuine geodesic maps.
    """

    exp_is_exact = False
    log_is_exact = False
    dist_is_exact = False
    transport_is_isometric = False
    transport_is_parallel = False

    def exp(self, x: Array, u: Array) -> Array:
        """Retraction proxy for the exponential map."""
        return self.retr(x, u)

    def log(self, x: Array, y: Array) -> Array:
        """Inverse-retraction proxy for the logarithmic map."""
        return self.invretr(x, y)

    def dist(self, x: Array, y: Array) -> Array:
        """Local inverse-retraction norm; not a geodesic distance."""
        return self.norm(x, self.invretr(x, y))

    def transport(self, x: Array, y: Array, u: Array) -> Array:
        """Projection vector transport associated with the retraction."""
        del x
        return self.tangent_project(y, u)


class ExactGeometryMixin(GeometryMixin):
    """Opt-in defaults for geometries with certified exact geodesic operations."""

    exp_is_exact = True
    log_is_exact = True
    dist_is_exact = True
    transport_is_isometric = True
    transport_is_parallel = True


__all__ = [
    "Array",
    "Shape",
    "ManifoldProtocol",
    "GeometryProtocol",
    "GeometryMixin",
    "ExactGeometryMixin",
    "RetractionGeometryMixin",
    "as_sample_shape",
    "check_event_shape",
    "dtype_margin",
    "event_shape_matches",
    "false_for_event_shape",
    "scale_tangent",
    "validate_boolean",
    "validate_nonnegative",
    "validate_positive",
    "validate_integer",
]
