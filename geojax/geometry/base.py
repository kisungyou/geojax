"""Shared geometry protocol and mixin helpers."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Tuple, Union, runtime_checkable

import jax.numpy as jnp

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


def as_sample_shape(sample_shape: Shape = ()) -> tuple[int, ...]:
    """Normalize an integer or tuple-like sample shape."""
    if isinstance(sample_shape, int):
        return (sample_shape,)
    return tuple(sample_shape)


@runtime_checkable
class ManifoldProtocol(Protocol):
    """Retraction-based interface consumed by manifold optimizers.

    This is the GeoJAX counterpart of Manopt's core manifold structure. Exact
    geodesic maps are deliberately not required.
    """

    size: Any
    shape: tuple[int, ...] | tuple[Any, ...]
    dim: int

    def belongs(self, x: Array, atol: float | None = None) -> Array: ...
    def project(self, x: Array) -> Array: ...
    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array: ...
    def tangent_project(self, x: Array, u: Array) -> Array: ...
    def inner(self, x: Array, u: Array, v: Array) -> Array: ...
    def norm(self, x: Array, u: Array) -> Array: ...
    def lincomb(self, x: Array, *terms: Any) -> Array: ...
    def retr(self, x: Array, u: Array, t: float | Array = 1.0) -> Array: ...
    def invretr(self, x: Array, y: Array) -> Array: ...
    def transport(self, x: Array, y: Array, u: Array) -> Array: ...
    def egrad_to_rgrad(self, x: Array, egrad: Array) -> Array: ...
    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array: ...
    def random_tangent(self, key: Array, x: Array, scale: float | Array = 1.0) -> Array: ...


@runtime_checkable
class GeometryProtocol(ManifoldProtocol, Protocol):
    """Uniform geometry interface with capability-qualified named operations.

    Retraction-only geometries may satisfy this structural protocol through
    documented compatibility aliases. Use ``operation_kind`` to distinguish
    exact geodesic operations from numerical-local or retraction proxies.
    """

    def exp(self, x: Array, u: Array) -> Array: ...
    def log(self, x: Array, y: Array) -> Array: ...
    def squared_dist(self, x: Array, y: Array) -> Array: ...
    def dist(self, x: Array, y: Array) -> Array: ...


class GeometryMixin:
    """Default helpers for geometries with the GeoJAX protocol."""

    exp_is_exact = True
    log_is_exact = True
    dist_is_exact = True
    transport_is_isometric = True
    transport_is_parallel = True
    hessian_conversion_is_exact = False
    riemannian_gradient_jvp_is_exact = False

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
        return jnp.maximum(self.inner(x, tangent, tangent), 0.0)

    def retr(self, x: Array, u: Array, t: float | Array = 1.0) -> Array:
        """Default retraction: use the exponential map."""
        return self.exp(x, t * u)

    def invretr(self, x: Array, y: Array) -> Array:
        """Default inverse retraction: use the logarithmic map."""
        return self.log(x, y)

    def lincomb(self, x: Array, *terms: Any) -> Array:
        """Linear combination of tangent vectors, projected back to ``T_x M``."""
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = coeff * vec
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

    transp = transport


__all__ = [
    "Array",
    "Shape",
    "ManifoldProtocol",
    "GeometryProtocol",
    "GeometryMixin",
    "RetractionGeometryMixin",
    "as_sample_shape",
]
