"""Elementary matrix, probability, and hyperbolic geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from .base import ExactGeometryMixin, Shape, as_sample_shape, dtype_margin
from ._numerics import (
    acos_over_sin,
    acos_squared,
    atanhc_from_squared_norm,
    cos_from_squared_norm,
    sinc_from_squared_norm,
    squared_norm,
    tanhc_from_squared_norm,
)

Array = Any


def _safe_divide(numerator: Array, denominator: Array, eps: float) -> Array:
    return numerator / jnp.where(jnp.abs(denominator) > eps, denominator, 1.0)


def _parse_matrix_size(size: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError(f"{name} size must be a pair.")
    parsed = tuple(int(value) for value in size)
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

    def __init__(self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-12):
        object.__setattr__(self, "size", _parse_matrix_size(size, "Oblique"))
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

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
        norms = jnp.linalg.norm(jnp.asarray(X), axis=-2)
        return jnp.all(jnp.abs(norms - 1.0) <= tol, axis=-1)

    def project(self, A: Array) -> Array:
        A = self._check_shape(A, name="A")
        norms = jnp.linalg.norm(A, axis=-2, keepdims=True)
        fallback = jnp.zeros_like(A).at[..., 0, :].set(1.0)
        return jnp.where(norms > self.eps, A / jnp.maximum(norms, self.eps), fallback)

    normalize = project

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

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        _, U, V = self._check_shapes(("X", X), ("U", U), ("V", V))
        return jnp.sum(U * V, axis=(-2, -1))

    def norm(self, X: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(X, U, U), 0.0))

    def exp(self, X: Array, U: Array) -> Array:
        X = jnp.asarray(X)
        U = self.tangent_project(X, U)
        lengths_squared = squared_norm(U, axis=-2, keepdims=True)
        return X * cos_from_squared_norm(lengths_squared) + U * sinc_from_squared_norm(
            lengths_squared
        )

    def log(self, X: Array, Y: Array) -> Array:
        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
        dots = jnp.clip(jnp.sum(X * Y, axis=-2, keepdims=True), -1.0, 1.0)
        direction = Y - dots * X
        sine_squared = squared_norm(direction, axis=-2, keepdims=True)
        result = acos_over_sin(dots, sine_squared) * direction
        dtype = jnp.result_type(X, float)
        sine_cutoff = 64.0 * jnp.finfo(dtype).eps
        at_cut = (dots < 0.0) & (sine_squared <= sine_cutoff**2)
        return jnp.where(at_cut, jnp.full_like(result, jnp.nan), result)

    def squared_dist(self, X: Array, Y: Array) -> Array:
        dots = jnp.clip(jnp.sum(jnp.asarray(X) * jnp.asarray(Y), axis=-2), -1.0, 1.0)
        return jnp.sum(acos_squared(dots), axis=-1)

    def dist(self, X: Array, Y: Array) -> Array:
        return jnp.sqrt(self.squared_dist(X, Y))

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
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

    transp = transport

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        return self.tangent_project(X, egrad)

    egrad2rgrad = egrad_to_rgrad

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
            tangent = jnp.where(length > self.eps, tangent / length, tangent)
        return scale * tangent


@dataclass(frozen=True, init=False)
class ProbabilitySimplex(ExactGeometryMixin):
    """Interior probability simplex with the Fisher--Rao metric."""

    size: int
    atol: float
    eps: float

    def __init__(self, size: int, *, atol: float = 1e-6, eps: float = 1e-10):
        size = int(size)
        if size < 2:
            raise ValueError("ProbabilitySimplex size must be at least 2.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

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
        p = jnp.maximum(p, floor)
        return p / jnp.sum(p, axis=-1, keepdims=True)

    normalize = project

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

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, p: Array, u: Array, v: Array) -> Array:
        p, u, v = self._check_shapes(("p", p), ("u", u), ("v", v))
        p = self.project(p)
        return jnp.sum(u * v / p, axis=-1)

    def norm(self, p: Array, u: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(p, u, u), 0.0))

    def exp(self, p: Array, u: Array) -> Array:
        p = self.project(p)
        u = self.tangent_project(p, u)
        length_squared = jnp.maximum(self.inner(p, u, u), 0.0)[..., None]
        root = jnp.sqrt(p)
        half_length_squared = 0.25 * length_squared
        next_root = cos_from_squared_norm(
            half_length_squared
        ) * root + 0.5 * sinc_from_squared_norm(half_length_squared) * (u / root)
        result = next_root**2
        result = result / jnp.sum(result, axis=-1, keepdims=True)
        valid = jnp.all(next_root > 0.0, axis=-1, keepdims=True) | (length_squared <= self.eps**2)
        return jnp.where(valid, result, jnp.full_like(result, jnp.nan))

    def retr(self, p: Array, u: Array, t: float | Array = 1.0) -> Array:
        """Positive normalized-addition retraction used by optimizers."""
        return self.project(jnp.asarray(p) + t * self.tangent_project(p, u))

    def log(self, p: Array, q: Array) -> Array:
        p = self.project(p)
        q = self.project(q)
        root_p = jnp.sqrt(p)
        root_q = jnp.sqrt(q)
        cosine = jnp.clip(jnp.sum(root_p * root_q, axis=-1, keepdims=True), -1.0, 1.0)
        scale = 2.0 * acos_over_sin(cosine)
        return self.tangent_project(p, scale * root_p * (root_q - cosine * root_p))

    def squared_dist(self, p: Array, q: Array) -> Array:
        affinity = jnp.sum(jnp.sqrt(self.project(p) * self.project(q)), axis=-1)
        return 4.0 * acos_squared(jnp.clip(affinity, -1.0, 1.0))

    def dist(self, p: Array, q: Array) -> Array:
        return jnp.sqrt(self.squared_dist(p, q))

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

    transp = transport

    def egrad_to_rgrad(self, p: Array, egrad: Array) -> Array:
        p, egrad = self._check_shapes(("p", p), ("egrad", egrad))
        p = self.project(p)
        return p * (egrad - jnp.sum(p * egrad, axis=-1, keepdims=True))

    egrad2rgrad = egrad_to_rgrad

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
            tangent = jnp.where(length > self.eps, tangent / length, tangent)
        return scale * tangent


@dataclass(frozen=True, init=False)
class PoincareBall(ExactGeometryMixin):
    """Poincaré ball model of curvature-minus-one hyperbolic space."""

    size: int
    atol: float
    eps: float

    def __init__(self, size: int, *, atol: float = 1e-6, eps: float = 1e-10):
        size = int(size)
        if size < 1:
            raise ValueError("PoincareBall size must be positive.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

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
        return jnp.linalg.norm(jnp.asarray(x), axis=-1) < 1.0

    def project(self, x: Array) -> Array:
        x = self._check_shape(x, name="x")
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        margin = dtype_margin(x, configured=self.eps, atol=self.atol)
        radius = 1.0 - margin
        denominator = jnp.maximum(norm, margin)
        return jnp.where(norm < radius, x, radius * x / denominator)

    normalize = project

    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array:
        del atol
        if not self._shape_matches(x, u):
            return self._shape_failure(x)
        _, u = self._check_shapes(("x", x), ("u", u))
        return jnp.all(jnp.isfinite(u), axis=-1)

    def tangent_project(self, x: Array, u: Array) -> Array:
        _, u = self._check_shapes(("x", x), ("u", u))
        return u

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def conformal_factor(self, x: Array) -> Array:
        x = self._check_shape(x, name="x")
        squared_norm = jnp.sum(x**2, axis=-1)
        floor = dtype_margin(x, configured=self.eps)
        return 2.0 / jnp.maximum(1.0 - squared_norm, floor)

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        x, u, v = self._check_shapes(("x", x), ("u", u), ("v", v))
        factor = self.conformal_factor(x)
        return factor**2 * jnp.sum(u * v, axis=-1)

    def norm(self, x: Array, u: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(x, u, u), 0.0))

    def mobius_add(self, x: Array, y: Array) -> Array:
        x, y = self._check_shapes(("x", x), ("y", y))
        x2 = jnp.sum(x * x, axis=-1, keepdims=True)
        y2 = jnp.sum(y * y, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        numerator = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
        denominator = 1.0 + 2.0 * xy + x2 * y2
        floor = dtype_margin(x, configured=self.eps)
        return numerator / jnp.maximum(denominator, floor)

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
        return jnp.sqrt(self.squared_dist(x, y))

    def _gyration(self, u: Array, v: Array, w: Array) -> Array:
        u2 = jnp.sum(u * u, axis=-1, keepdims=True)
        v2 = jnp.sum(v * v, axis=-1, keepdims=True)
        uv = jnp.sum(u * v, axis=-1, keepdims=True)
        uw = jnp.sum(u * w, axis=-1, keepdims=True)
        vw = jnp.sum(v * w, axis=-1, keepdims=True)
        a = -uw * v2 + vw + 2.0 * uv * vw
        b = -vw * u2 - uw
        denominator = 1.0 + 2.0 * uv + u2 * v2
        return w + 2.0 * (a * u + b * v) / jnp.maximum(denominator, self.eps)

    def transport(self, x: Array, y: Array, u: Array) -> Array:
        x = self.project(x)
        y = self.project(y)
        u = self.tangent_project(x, u)
        rotated = self._gyration(y, -x, u)
        return (self.conformal_factor(x) / self.conformal_factor(y))[..., None] * rotated

    transp = transport

    def egrad_to_rgrad(self, x: Array, egrad: Array) -> Array:
        x, egrad = self._check_shapes(("x", x), ("egrad", egrad))
        factor = self.conformal_factor(x)[..., None]
        return egrad / factor**2

    egrad2rgrad = egrad_to_rgrad

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        key_direction, key_radius = jax.random.split(key)
        direction = jax.random.normal(key_direction, shape=sample_shape + self.shape)
        direction /= jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), self.eps)
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
            tangent = jnp.where(length > self.eps, tangent / length, tangent)
        return scale * tangent


__all__ = ["Oblique", "ProbabilitySimplex", "PoincareBall"]
