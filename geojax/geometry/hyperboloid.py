"""Hyperboloid model of hyperbolic space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import GeometryMixin, as_sample_shape
from ._numerics import (
    acosh_over_sqrt,
    acosh_squared,
    cosh_from_squared_norm,
    sinhc_from_squared_norm,
)

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


@dataclass(frozen=True, init=False)
class Hyperboloid(GeometryMixin):
    """Upper-sheet hyperboloid in ambient Minkowski space ``R^size``."""

    size: int
    atol: float
    eps: float

    def __init__(self, size: int, *, atol: float = 1e-6, eps: float = 1e-12) -> None:
        size = int(size)
        if size < 2:
            raise ValueError("Hyperboloid size must be at least 2.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def dim(self) -> int:
        return self.size - 1

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    def lorentz_inner(self, u: Array, v: Array, keepdims: bool = False) -> Array:
        u = jnp.asarray(u)
        v = jnp.asarray(v)
        out = -u[..., 0] * v[..., 0] + jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)
        return out[..., None] if keepdims else out

    def belongs(self, x: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        x = jnp.asarray(x)
        sheet = x[..., 0] > 0.0
        quad = self.lorentz_inner(x, x)
        cancellation_scale = x[..., 0] ** 2 + jnp.sum(x[..., 1:] ** 2, axis=-1) + 1.0
        rounding = 16.0 * jnp.finfo(x.dtype).eps * cancellation_scale
        return sheet & (jnp.abs(quad + 1.0) <= tol + rounding)

    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        x = jnp.asarray(x)
        u = jnp.asarray(u)
        cancellation_scale = (
            jnp.abs(x[..., 0] * u[..., 0])
            + jnp.sum(jnp.abs(x[..., 1:] * u[..., 1:]), axis=-1)
            + 1.0
        )
        rounding = 16.0 * jnp.finfo(jnp.result_type(x, u)).eps * cancellation_scale
        return jnp.abs(self.lorentz_inner(x, u)) <= tol + rounding

    def project(self, x: Array) -> Array:
        x = jnp.asarray(x)
        spatial = x[..., 1:]
        time = jnp.sqrt(1.0 + jnp.sum(spatial * spatial, axis=-1, keepdims=True))
        return jnp.concatenate([time, spatial], axis=-1)

    normalize = project

    def tangent_project(self, x: Array, u: Array) -> Array:
        x = self.project(x)
        u = jnp.asarray(u)
        return u + self.lorentz_inner(x, u, keepdims=True) * x

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        del x
        return self.lorentz_inner(u, v)

    def norm(self, x: Array, u: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(x, u, u), 0.0))

    def exp(self, x: Array, u: Array) -> Array:
        x = self.project(x)
        u = self.tangent_project(x, u)
        r2 = jnp.maximum(self.inner(x, u, u), 0.0)[..., None]
        result = cosh_from_squared_norm(r2) * x + sinhc_from_squared_norm(r2) * u
        return self.project(result)

    def log(self, x: Array, y: Array) -> Array:
        x = self.project(x)
        y = self.project(y)
        alpha = jnp.maximum(-self.lorentz_inner(x, y, keepdims=True), 1.0)
        coef = acosh_over_sqrt(alpha)
        return self.tangent_project(x, coef * (y - alpha * x))

    def squared_dist(self, x: Array, y: Array) -> Array:
        """Squared hyperbolic distance with a finite derivative at coincidence."""
        x = self.project(x)
        y = self.project(y)
        alpha = jnp.maximum(-self.lorentz_inner(x, y), 1.0)
        return acosh_squared(alpha)

    def dist(self, x: Array, y: Array) -> Array:
        return jnp.sqrt(self.squared_dist(x, y))

    def transport(self, x: Array, y: Array, u: Array) -> Array:
        x = self.project(x)
        y = self.project(y)
        u = self.tangent_project(x, u)
        denom = 1.0 - self.lorentz_inner(x, y, keepdims=True)
        denom_safe = jnp.where(denom > self.eps, denom, 1.0)
        coef = self.lorentz_inner(y, u, keepdims=True) / denom_safe
        return self.tangent_project(y, u + coef * (x + y))

    transp = transport

    def geodesic_flow(self, x: Array, v: Array, t: float | Array = 1.0) -> tuple[Array, Array]:
        x = self.project(x)
        v = self.tangent_project(x, v)
        r2 = jnp.maximum(self.inner(x, v, v), 0.0)[..., None]
        t_array = jnp.asarray(t)
        t_array = jnp.reshape(t_array, t_array.shape + (1,))
        tr2 = t_array * t_array * r2
        cosine = cosh_from_squared_norm(tr2)
        sinhc = sinhc_from_squared_norm(tr2)
        x_t = self.project(cosine * x + t_array * sinhc * v)
        v_t = t_array * r2 * sinhc * x + cosine * v
        return x_t, self.tangent_project(x_t, v_t)

    def to_poincare(self, x: Array) -> Array:
        """Convert hyperboloid points to the Poincare ball model."""
        x = self.project(x)
        return x[..., 1:] / (x[..., :1] + 1.0)

    def from_poincare(self, point: Array) -> Array:
        """Convert points in the open Poincare ball to the hyperboloid."""
        point = jnp.asarray(point)
        squared_radius = jnp.sum(point * point, axis=-1, keepdims=True)
        radius = jnp.sqrt(jnp.maximum(squared_radius, 0.0))
        max_radius = 1.0 - self.eps
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        point = jnp.where(radius < max_radius, point, max_radius * point / safe_radius)
        squared_radius = jnp.sum(point * point, axis=-1, keepdims=True)
        denominator = jnp.maximum(1.0 - squared_radius, self.eps)
        time = (1.0 + squared_radius) / denominator
        spatial = 2.0 * point / denominator
        return jnp.concatenate([time, spatial], axis=-1)

    def egrad_to_rgrad(self, x: Array, egrad: Array) -> Array:
        egrad = jnp.asarray(egrad)
        minkowski_grad = egrad.at[..., 0].multiply(-1.0)
        return self.tangent_project(x, minkowski_grad)

    egrad2rgrad = egrad_to_rgrad

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        spatial = jax.random.normal(key, shape=sample_shape + (self.dim,))
        time = jnp.sqrt(1.0 + jnp.sum(spatial * spatial, axis=-1, keepdims=True))
        return jnp.concatenate([time, spatial], axis=-1)

    def random_tangent(
        self,
        key: Array,
        x: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        z = jax.random.normal(key, shape=jnp.shape(x))
        u = self.tangent_project(x, z)
        if normalize:
            n = self.norm(x, u)[..., None]
            u = jnp.where(n > self.eps, u / n, u)
        return scale * u


__all__ = ["Hyperboloid"]
