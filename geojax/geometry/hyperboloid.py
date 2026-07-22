"""Hyperboloid model of hyperbolic space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import GeometryMixin, as_sample_shape

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
        return sheet & (jnp.abs(quad + 1.0) <= tol)

    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        return jnp.abs(self.lorentz_inner(x, u)) <= tol

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
        r = self.norm(x, u)[..., None]
        r2 = r * r
        r_safe = jnp.where(r > self.eps, r, 1.0)
        sinh_over_r = jnp.where(
            r > self.eps,
            jnp.sinh(r) / r_safe,
            1.0 + r2 / 6.0 + (r2 * r2) / 120.0,
        )
        return self.project(jnp.cosh(r) * x + sinh_over_r * u)

    def log(self, x: Array, y: Array) -> Array:
        x = self.project(x)
        y = self.project(y)
        alpha = jnp.maximum(-self.lorentz_inner(x, y, keepdims=True), 1.0)
        d = jnp.arccosh(alpha)
        sinh_d = jnp.sqrt(jnp.maximum(alpha * alpha - 1.0, 0.0))
        coef = jnp.where(sinh_d > self.eps, d / sinh_d, 1.0)
        return self.tangent_project(x, coef * (y - alpha * x))

    def dist(self, x: Array, y: Array) -> Array:
        x = self.project(x)
        y = self.project(y)
        alpha = jnp.maximum(-self.lorentz_inner(x, y), 1.0)
        return jnp.arccosh(alpha)

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
        r = self.norm(x, v)[..., None]
        tr = t * r
        r2 = r * r
        r_safe = jnp.where(r > self.eps, r, 1.0)
        sinh_tr_over_r = jnp.where(
            r > self.eps,
            jnp.sinh(tr) / r_safe,
            t + (t**3) * r2 / 6.0 + (t**5) * r2 * r2 / 120.0,
        )
        x_t = self.project(jnp.cosh(tr) * x + sinh_tr_over_r * v)
        v_t = r * jnp.sinh(tr) * x + jnp.cosh(tr) * v
        return x_t, self.tangent_project(x_t, v_t)

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
