"""Flat torus geometry for angular coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import GeometryMixin, as_sample_shape

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


def wrap_angles(x: Array) -> Array:
    """Wrap angles to ``[-pi, pi)``."""
    return (jnp.asarray(x) + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@dataclass(frozen=True, init=False)
class Torus(GeometryMixin):
    """Flat ``d``-torus represented by angles in ``[-pi, pi)``."""

    size: int
    atol: float

    def __init__(self, size: int, *, atol: float = 1e-6) -> None:
        size = int(size)
        if size < 1:
            raise ValueError("Torus size must be positive.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", float(atol))

    @property
    def dim(self) -> int:
        return self.size

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    def wrap(self, x: Array) -> Array:
        return wrap_angles(x)

    def belongs(self, x: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        x = jnp.asarray(x)
        return jnp.all((x >= -jnp.pi - tol) & (x < jnp.pi + tol), axis=-1)

    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array:
        del x, atol
        return jnp.asarray(jnp.shape(u)[-1] == self.size)

    def project(self, x: Array) -> Array:
        return self.wrap(x)

    normalize = project

    def tangent_project(self, x: Array, u: Array) -> Array:
        del x
        return jnp.asarray(u)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        del x
        return jnp.sum(u * v, axis=-1)

    def norm(self, x: Array, u: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(x, u, u), 0.0))

    def exp(self, x: Array, u: Array) -> Array:
        return self.wrap(jnp.asarray(x) + jnp.asarray(u))

    def retr(self, x: Array, u: Array, t: float | Array = 1.0) -> Array:
        return self.wrap(jnp.asarray(x) + t * jnp.asarray(u))

    def log(self, x: Array, y: Array) -> Array:
        return self.wrap(jnp.asarray(y) - jnp.asarray(x))

    def dist(self, x: Array, y: Array) -> Array:
        return self.norm(x, self.log(x, y))

    def transport(self, x: Array, y: Array, u: Array) -> Array:
        del x, y
        return jnp.asarray(u)

    transp = transport

    def pair_mean(self, x: Array, y: Array) -> Array:
        return self.exp(x, 0.5 * self.log(x, y))

    def egrad_to_rgrad(self, x: Array, egrad: Array) -> Array:
        del x
        return jnp.asarray(egrad)

    egrad2rgrad = egrad_to_rgrad

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        return jax.random.uniform(
            key,
            shape=sample_shape + self.shape,
            minval=-jnp.pi,
            maxval=jnp.pi,
        )

    def random_tangent(
        self,
        key: Array,
        x: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        u = jax.random.normal(key, shape=jnp.shape(x))
        if normalize:
            n = jnp.linalg.norm(u, axis=-1, keepdims=True)
            u = jnp.where(n > 0.0, u / n, u)
        return scale * u


__all__ = ["Torus", "wrap_angles"]
