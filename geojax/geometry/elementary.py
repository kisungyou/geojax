"""Elementary matrix, probability, and hyperbolic geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from ._numerics import nonnegative

from .base import (
    ExactGeometryMixin,
    Shape,
    as_sample_shape,
    dtype_margin,
    validate_integer,
    validate_nonnegative,
    validate_positive,
)
from ._numerics import (
    acos_over_sin,
    spherical_squared_dist,
    atanhc_from_squared_norm,
    cos_from_squared_norm,
    sinc_from_squared_norm,
    stable_metric_norm,
    stable_norm,
    sqrt_nonnegative,
    squared_norm,
    tanhc_from_squared_norm,
)

Array = Any


def _parse_matrix_size(size: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError(f"{name} size must be a pair.")
    parsed = tuple(validate_integer(value, name=f"{name} size entry", minimum=1) for value in size)
    if len(parsed) != 2 or min(parsed) < 1:
        raise ValueError(f"{name} size must be a pair of positive integers.")
    return parsed


@dataclass(frozen=True, init=False)
class Oblique(ExactGeometryMixin):
    """Matrices whose columns have unit Euclidean norm.

    ``Oblique(size=(n, m))`` is the efficient matrix representation of a
    product of ``m`` copies of the sphere ``S^(n-1)``.
    """

    size: tuple[int, int]
    atol: float
    eps: float

    hessian_conversion_is_exact = True
    riemannian_gradient_jvp_is_exact = True

    def __init__(self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-12):
        object.__setattr__(self, "size", _parse_matrix_size(size, "Oblique"))
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="Oblique atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="Oblique eps"))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def m(self) -> int:
        return self.size[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    @property
    def dim(self) -> int:
        return self.m * (self.n - 1)

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(X):
            return self._shape_failure(X)
        norms = stable_norm(jnp.asarray(X), axis=-2)
        return jnp.all(jnp.abs(norms - 1.0) <= tol, axis=-1)

    def project(self, A: Array) -> Array:
        A = self._check_shape(A, name="A")
        norms = stable_norm(A, axis=-2, keepdims=True)
        fallback = jnp.zeros_like(A).at[..., 0, :].set(1.0)
        safe_norms = jnp.where(norms > 0.0, norms, jnp.ones_like(norms))
        return jnp.where(norms > 0.0, A / safe_norms, fallback)

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(X, U):
            return self._shape_failure(X)
        X, U = self._check_shapes(("X", X), ("U", U))
        radial = jnp.sum(jnp.asarray(X) * jnp.asarray(U), axis=-2)
        return jnp.all(jnp.abs(radial) <= tol, axis=-1)

    def tangent_project(self, X: Array, U: Array) -> Array:
        X, U = self._check_shapes(("X", X), ("U", U))
        return U - X * jnp.sum(X * U, axis=-2, keepdims=True)

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        _, U, V = self._check_shapes(("X", X), ("U", U), ("V", V))
        return jnp.sum(U * V, axis=(-2, -1))

    def norm(self, X: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(X, normalized, normalized),
            axis=(-2, -1),
        )

    def exp(self, X: Array, U: Array) -> Array:
        X = self._check_shape(X, name="X")
        U = self.tangent_project(X, U)
        lengths_squared = squared_norm(U, axis=-2, keepdims=True)
        return X * cos_from_squared_norm(lengths_squared) + U * sinc_from_squared_norm(
            lengths_squared
        )

    def log(self, X: Array, Y: Array) -> Array:
        X, Y = self._check_shapes(("X", X), ("Y", Y))
        dots = jnp.clip(jnp.sum(X * Y, axis=-2, keepdims=True), -1.0, 1.0)
        direction = Y - dots * X
        sine_squared = squared_norm(direction, axis=-2, keepdims=True)
        result = acos_over_sin(dots, sine_squared) * direction
        dtype = jnp.result_type(X, float)
        sine_cutoff = 64.0 * jnp.finfo(dtype).eps
        at_cut = (dots < 0.0) & (sine_squared <= sine_cutoff**2)
        return jnp.where(at_cut, jnp.full_like(result, jnp.nan), result)

    def squared_dist(self, X: Array, Y: Array) -> Array:
        X, Y = self._check_shapes(("X", X), ("Y", Y))
        return jnp.sum(spherical_squared_dist(X, Y, axis=-2), axis=-1)

    def dist(self, X: Array, Y: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(X, Y))

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        X, Y = self._check_shapes(("X", X), ("Y", Y))
        U = self.tangent_project(X, U)
        dots = jnp.clip(jnp.sum(X * Y, axis=-2, keepdims=True), -1.0, 1.0)
        direction = Y - dots * X
        sine_squared = squared_norm(direction, axis=-2, keepdims=True)
        sine = jnp.sqrt(sine_squared)
        dtype = jnp.result_type(X, float)
        sine_cutoff = 64.0 * jnp.finfo(dtype).eps
        safe_sine = jnp.where(sine > sine_cutoff, sine, 1.0)
        unit_direction = direction / safe_sine
        component = jnp.sum(U * unit_direction, axis=-2, keepdims=True)
        terminal_direction = -sine * X + dots * unit_direction
        result = U + component * (terminal_direction - unit_direction)
        at_cut = (dots < 0.0) & (sine <= sine_cutoff)
        return jnp.where(at_cut, jnp.full_like(result, jnp.nan), result)

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        return self.tangent_project(X, egrad)

    def ehess_to_rhess(self, X: Array, egrad: Array, ehess_vec: Array, U: Array) -> Array:
        X, egrad, ehess_vec, U = self._check_shapes(
            ("X", X),
            ("egrad", egrad),
            ("ehess_vec", ehess_vec),
            ("U", U),
        )
        curvature = U * jnp.sum(X * egrad, axis=-2, keepdims=True)
        return self.tangent_project(X, ehess_vec - curvature)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        normal = jax.random.normal(key, shape=as_sample_shape(sample_shape) + self.shape)
        return self.project(normal)

    def random_tangent(
        self,
        key: Array,
        X: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        tangent = self.tangent_project(X, jax.random.normal(key, shape=jnp.shape(X)))
        if normalize:
            length = self.norm(X, tangent)[..., None, None]
            safe_length = jnp.where(length > 0.0, length, jnp.ones_like(length))
            tangent = jnp.where(length > 0.0, tangent / safe_length, tangent)
        return self._scale_tangent(tangent, scale)


@dataclass(frozen=True, init=False)
class ProbabilitySimplex(ExactGeometryMixin):
    """Interior probability simplex with the Fisher--Rao metric."""

    size: int
    atol: float
    eps: float

    def __init__(self, size: int, *, atol: float = 1e-6, eps: float = 1e-10):
        size = validate_integer(size, name="ProbabilitySimplex size", minimum=2)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="ProbabilitySimplex atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="ProbabilitySimplex eps"))

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    @property
    def dim(self) -> int:
        return self.size - 1

    def belongs(self, p: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        p = jnp.asarray(p)
        if not self._shape_matches(p):
            return self._shape_failure(p)
        return (jnp.min(p, axis=-1) > 0.0) & (jnp.abs(jnp.sum(p, axis=-1) - 1.0) <= tol)

    def project(self, p: Array) -> Array:
        p = self._check_shape(p, name="p")
        floor = dtype_margin(p, configured=self.eps)
        p = jnp.where(p > 0.0, p, floor)
        return p / jnp.sum(p, axis=-1, keepdims=True)

    def is_tangent(self, p: Array, u: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(p, u):
            return self._shape_failure(p)
        _, u = self._check_shapes(("p", p), ("u", u))
        return jnp.abs(jnp.sum(u, axis=-1)) <= tol

    def tangent_project(self, p: Array, u: Array) -> Array:
        p, u = self._check_shapes(("p", p), ("u", u))
        p = self.project(p)
        # The Fisher--Rao normal is span{p}, since
        # g_p(p, v) = sum_i v_i for every ambient vector v.
        return u - p * jnp.sum(u, axis=-1, keepdims=True)

    def inner(self, p: Array, u: Array, v: Array) -> Array:
        p, u, v = self._check_shapes(("p", p), ("u", u), ("v", v))
        p = self.project(p)
        return jnp.sum(u * v / p, axis=-1)

    def norm(self, p: Array, u: Array) -> Array:
        return stable_metric_norm(
            u,
            lambda normalized: self.inner(p, normalized, normalized),
            axis=-1,
        )

    def exp(self, p: Array, u: Array) -> Array:
        p = self.project(p)
        u = self.tangent_project(p, u)
        length_squared = nonnegative(self.inner(p, u, u))[..., None]
        root = jnp.sqrt(p)
        half_length_squared = 0.25 * length_squared
        next_root = cos_from_squared_norm(
            half_length_squared
        ) * root + 0.5 * sinc_from_squared_norm(half_length_squared) * (u / root)
        result = next_root**2
        result = result / jnp.sum(result, axis=-1, keepdims=True)
        # The interior simplex is the positive orthant of the radius-two
        # sphere under p -> 2 sqrt(p). Checking only the endpoint is
        # insufficient: a long great circle can cross a coordinate hyperplane
        # and later re-enter the orthant. Compute the first positive zero of
        # every square-root coordinate and certify the entire segment.
        half_length = jnp.sqrt(half_length_squared)
        safe_half_length = jnp.where(half_length > 0.0, half_length, 1.0)
        root_velocity = 0.5 * u / root
        unit_velocity = root_velocity / safe_half_length
        first_boundary = jnp.min(
            jnp.arctan2(root, -unit_velocity),
            axis=-1,
            keepdims=True,
        )
        valid = half_length < first_boundary
        return jnp.where(valid, result, jnp.full_like(result, jnp.nan))

    def retr(self, p: Array, u: Array, t: float | Array = 1.0) -> Array:
        """Positive normalized-addition retraction used by optimizers."""
        return self.project(jnp.asarray(p) + self._scale_tangent(self.tangent_project(p, u), t))

    def log(self, p: Array, q: Array) -> Array:
        p = self.project(p)
        q = self.project(q)
        root_p = jnp.sqrt(p)
        root_q = jnp.sqrt(q)
        cosine = jnp.clip(jnp.sum(root_p * root_q, axis=-1, keepdims=True), -1.0, 1.0)
        scale = 2.0 * acos_over_sin(cosine)
        return self.tangent_project(p, scale * root_p * (root_q - cosine * root_p))

    def squared_dist(self, p: Array, q: Array) -> Array:
        return 4.0 * spherical_squared_dist(
            jnp.sqrt(self.project(p)), jnp.sqrt(self.project(q)), axis=-1
        )

    def dist(self, p: Array, q: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(p, q))

    def transport(self, p: Array, q: Array, u: Array) -> Array:
        p = self.project(p)
        q = self.project(q)
        root_p = jnp.sqrt(p)
        root_q = jnp.sqrt(q)
        sphere_p = 2.0 * root_p
        sphere_q = 2.0 * root_q
        sphere_u = self.tangent_project(p, u) / root_p
        denominator = 4.0 + jnp.sum(sphere_p * sphere_q, axis=-1, keepdims=True)
        coefficient = jnp.sum(sphere_u * sphere_q, axis=-1, keepdims=True) / denominator
        sphere_v = sphere_u - coefficient * (sphere_p + sphere_q)
        return self.tangent_project(q, root_q * sphere_v)

    def egrad_to_rgrad(self, p: Array, egrad: Array) -> Array:
        p, egrad = self._check_shapes(("p", p), ("egrad", egrad))
        p = self.project(p)
        return p * (egrad - jnp.sum(p * egrad, axis=-1, keepdims=True))

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        values = jax.random.exponential(key, shape=as_sample_shape(sample_shape) + self.shape)
        return self.project(values)

    def random_tangent(
        self,
        key: Array,
        p: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        tangent = self.tangent_project(p, jax.random.normal(key, shape=jnp.shape(p)))
        if normalize:
            length = self.norm(p, tangent)[..., None]
            safe_length = jnp.where(length > 0.0, length, jnp.ones_like(length))
            tangent = jnp.where(length > 0.0, tangent / safe_length, tangent)
        return self._scale_tangent(tangent, scale)


@dataclass(frozen=True, init=False)
class PoincareBall(ExactGeometryMixin):
    """Poincaré ball model of curvature-minus-one hyperbolic space."""

    size: int
    atol: float
    eps: float

    def __init__(self, size: int, *, atol: float = 1e-6, eps: float = 1e-10):
        size = validate_integer(size, name="PoincareBall size", minimum=1)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="PoincareBall atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="PoincareBall eps"))

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    @property
    def dim(self) -> int:
        return self.size

    def belongs(self, x: Array, atol: float | None = None) -> Array:
        del atol
        if not self._shape_matches(x):
            return self._shape_failure(x)
        return stable_norm(jnp.asarray(x), axis=-1) < 1.0

    def project(self, x: Array) -> Array:
        x = self._check_shape(x, name="x")
        norm = stable_norm(x, axis=-1, keepdims=True)
        margin = dtype_margin(x, configured=self.eps, atol=self.atol)
        radius = 1.0 - margin
        denominator = jnp.where(norm > 0.0, norm, jnp.ones_like(norm))
        return jnp.where(norm < 1.0, x, radius * x / denominator)

    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array:
        del atol
        if not self._shape_matches(x, u):
            return self._shape_failure(x)
        _, u = self._check_shapes(("x", x), ("u", u))
        return jnp.all(jnp.isfinite(u), axis=-1)

    def tangent_project(self, x: Array, u: Array) -> Array:
        _, u = self._check_shapes(("x", x), ("u", u))
        return u

    def conformal_factor(self, x: Array) -> Array:
        x = self._check_shape(x, name="x")
        squared_norm = jnp.sum(x**2, axis=-1)
        denominator = 1.0 - squared_norm
        safe_denominator = jnp.where(
            denominator > 0.0,
            denominator,
            jnp.ones_like(denominator),
        )
        factor = 2.0 / safe_denominator
        return jnp.where(denominator > 0.0, factor, jnp.full_like(factor, jnp.nan))

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        x, u, v = self._check_shapes(("x", x), ("u", u), ("v", v))
        factor = self.conformal_factor(x)
        return factor**2 * jnp.sum(u * v, axis=-1)

    def norm(self, x: Array, u: Array) -> Array:
        return stable_metric_norm(
            u,
            lambda normalized: self.inner(x, normalized, normalized),
            axis=-1,
        )

    def mobius_add(self, x: Array, y: Array) -> Array:
        x, y = self._check_shapes(("x", x), ("y", y))
        x2 = jnp.sum(x * x, axis=-1, keepdims=True)
        y2 = jnp.sum(y * y, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        numerator = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
        denominator = 1.0 + 2.0 * xy + x2 * y2
        safe_denominator = jnp.where(
            denominator > 0.0,
            denominator,
            jnp.ones_like(denominator),
        )
        result = numerator / safe_denominator
        return jnp.where(
            denominator > 0.0,
            result,
            jnp.full_like(result, jnp.nan),
        )

    def exp(self, x: Array, u: Array) -> Array:
        x = self.project(x)
        u = self.tangent_project(x, u)
        factor = self.conformal_factor(x)[..., None]
        squared_length = squared_norm(u, axis=-1, keepdims=True)
        half_factor = 0.5 * factor
        scaled_squared_length = half_factor**2 * squared_length
        step = half_factor * tanhc_from_squared_norm(scaled_squared_length) * u
        return self.project(self.mobius_add(x, step))

    def log(self, x: Array, y: Array) -> Array:
        x = self.project(x)
        y = self.project(y)
        displacement = self.mobius_add(-x, y)
        squared_length = squared_norm(displacement, axis=-1, keepdims=True)
        factor = self.conformal_factor(x)[..., None]
        scale = 2.0 * atanhc_from_squared_norm(squared_length, self.eps) / factor
        return scale * displacement

    def squared_dist(self, x: Array, y: Array) -> Array:
        displacement = self.mobius_add(-self.project(x), self.project(y))
        squared_length = squared_norm(displacement, axis=-1)
        ratio = atanhc_from_squared_norm(squared_length, self.eps)
        return 4.0 * squared_length * ratio * ratio

    def dist(self, x: Array, y: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(x, y))

    def _gyration(self, u: Array, v: Array, w: Array) -> Array:
        u2 = jnp.sum(u * u, axis=-1, keepdims=True)
        v2 = jnp.sum(v * v, axis=-1, keepdims=True)
        uv = jnp.sum(u * v, axis=-1, keepdims=True)
        uw = jnp.sum(u * w, axis=-1, keepdims=True)
        vw = jnp.sum(v * w, axis=-1, keepdims=True)
        a = -uw * v2 + vw + 2.0 * uv * vw
        b = -vw * u2 - uw
        denominator = 1.0 + 2.0 * uv + u2 * v2
        safe_denominator = jnp.where(
            denominator > 0.0,
            denominator,
            jnp.ones_like(denominator),
        )
        result = w + 2.0 * (a * u + b * v) / safe_denominator
        return jnp.where(
            denominator > 0.0,
            result,
            jnp.full_like(result, jnp.nan),
        )

    def transport(self, x: Array, y: Array, u: Array) -> Array:
        x = self.project(x)
        y = self.project(y)
        u = self.tangent_project(x, u)
        rotated = self._gyration(y, -x, u)
        return (self.conformal_factor(x) / self.conformal_factor(y))[..., None] * rotated

    def egrad_to_rgrad(self, x: Array, egrad: Array) -> Array:
        x, egrad = self._check_shapes(("x", x), ("egrad", egrad))
        factor = self.conformal_factor(x)[..., None]
        return egrad / factor**2

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        key_direction, key_radius = jax.random.split(key)
        direction = jax.random.normal(key_direction, shape=sample_shape + self.shape)
        direction_norm = stable_norm(direction, axis=-1, keepdims=True)
        direction /= jnp.where(
            direction_norm > 0.0,
            direction_norm,
            jnp.ones_like(direction_norm),
        )
        radius = 0.8 * jax.random.uniform(key_radius, shape=sample_shape + (1,)) ** (
            1.0 / self.size
        )
        return radius * direction

    def random_tangent(
        self,
        key: Array,
        x: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        self._check_shape(x, name="x")
        tangent = jax.random.normal(key, shape=jnp.shape(x))
        if normalize:
            length = self.norm(x, tangent)[..., None]
            safe_length = jnp.where(length > 0.0, length, jnp.ones_like(length))
            tangent = jnp.where(length > 0.0, tangent / safe_length, tangent)
        return self._scale_tangent(tangent, scale)


__all__ = ["Oblique", "ProbabilitySimplex", "PoincareBall"]
