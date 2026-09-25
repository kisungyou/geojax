"""Hyperboloid model of hyperbolic space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from ._numerics import nonnegative

from .base import (
    ExactGeometryMixin,
    as_sample_shape,
    check_event_shape,
    dtype_margin,
    validate_integer,
    validate_nonnegative,
    validate_positive,
)
from ._numerics import (
    acosh_over_sqrt,
    asinh_squared_from_squared_chord,
    cosh_from_squared_norm,
    sinhc_from_squared_norm,
    stable_metric_norm,
    stable_norm,
    sqrt_nonnegative,
    squared_norm,
)

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


@dataclass(frozen=True, init=False)
class Hyperboloid(ExactGeometryMixin):
    """Upper-sheet hyperboloid in ambient Minkowski space ``R^size``."""

    size: int
    atol: float
    eps: float

    def __init__(self, size: int, *, atol: float = 1e-6, eps: float = 1e-12) -> None:
        size = validate_integer(size, name="Hyperboloid size", minimum=2)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="Hyperboloid atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="Hyperboloid eps"))

    @property
    def dim(self) -> int:
        return self.size - 1

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    def lorentz_inner(self, u: Array, v: Array, keepdims: bool = False) -> Array:
        u, v = self._check_shapes(("u", u), ("v", v))
        out = -u[..., 0] * v[..., 0] + jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)
        return out[..., None] if keepdims else out

    def belongs(self, x: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        x = jnp.asarray(x)
        if not self._shape_matches(x):
            return self._shape_failure(x)
        sheet = x[..., 0] > 0.0
        expected_time = jnp.hypot(
            jnp.ones_like(x[..., 0]),
            stable_norm(x[..., 1:], axis=-1),
        )
        dtype = jnp.result_type(x, float)
        rounding = 16.0 * jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(x[..., 0]), expected_time)
        return sheet & (jnp.abs(x[..., 0] - expected_time) <= tol + rounding)

    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(x, u):
            return self._shape_failure(x)
        x, u = self._check_shapes(("x", x), ("u", u))
        cancellation_scale = (
            jnp.abs(x[..., 0] * u[..., 0])
            + jnp.sum(jnp.abs(x[..., 1:] * u[..., 1:]), axis=-1)
            + 1.0
        )
        dtype = jnp.result_type(x, u, float)
        rounding = 16.0 * jnp.finfo(dtype).eps * cancellation_scale
        return jnp.abs(self.lorentz_inner(x, u)) <= tol + rounding

    def project(self, x: Array) -> Array:
        x = self._check_shape(x, name="x")
        spatial = x[..., 1:]
        # Including the constant coordinate keeps the norm away from its
        # nonsmooth origin, retaining the correct curvature at spatial zero.
        time = stable_norm(
            jnp.concatenate([jnp.ones_like(x[..., :1]), spatial], axis=-1),
            axis=-1,
            keepdims=True,
        )
        return jnp.concatenate([time, spatial], axis=-1)

    def tangent_project(self, x: Array, u: Array) -> Array:
        x = self.project(x)
        _, u = self._check_shapes(("x", x), ("u", u))
        return u + self.lorentz_inner(x, u, keepdims=True) * x

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        self._check_shapes(("x", x), ("u", u), ("v", v))
        return self.lorentz_inner(u, v)

    def norm(self, x: Array, u: Array) -> Array:
        return stable_metric_norm(
            u,
            lambda normalized: self.inner(x, normalized, normalized),
            axis=-1,
        )

    def exp(self, x: Array, u: Array) -> Array:
        x = self.project(x)
        u = self.tangent_project(x, u)
        r2 = nonnegative(self.inner(x, u, u))[..., None]
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
        # In Poincare coordinates p = x_spatial / (x_time + 1), the Lorentz
        # chord square is
        #
        #   4 ||p-q||^2 / ((1-||p||^2)(1-||q||^2))
        #   = ||p-q||^2 (x_time+1)(y_time+1).
        #
        # This positive-products form avoids subtracting nearly equal, huge
        # temporal and spatial chord squares near the ideal boundary.
        x_scale = x[..., :1] + 1.0
        y_scale = y[..., :1] + 1.0
        poincare_x = x[..., 1:] / x_scale
        poincare_y = y[..., 1:] / y_scale
        squared_chord = squared_norm(poincare_y - poincare_x, axis=-1)
        squared_chord = squared_chord * x_scale[..., 0] * y_scale[..., 0]
        return asinh_squared_from_squared_chord(squared_chord)

    def dist(self, x: Array, y: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(x, y))

    def transport(self, x: Array, y: Array, u: Array) -> Array:
        x = self.project(x)
        y = self.project(y)
        u = self.tangent_project(x, u)
        denom = 1.0 - self.lorentz_inner(x, y, keepdims=True)
        denom_safe = jnp.where(denom > self.eps, denom, 1.0)
        coef = self.lorentz_inner(y, u, keepdims=True) / denom_safe
        return self.tangent_project(y, u + coef * (x + y))

    def geodesic_flow(self, x: Array, v: Array, t: float | Array = 1.0) -> tuple[Array, Array]:
        x = self.project(x)
        v = self.tangent_project(x, v)
        r2 = nonnegative(self.inner(x, v, v))[..., None]
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
        point = check_event_shape(point, (self.dim,), name="point")
        radius = stable_norm(point, axis=-1, keepdims=True)
        margin = dtype_margin(point, configured=self.eps, atol=self.atol)
        max_radius = 1.0 - margin
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        inside = radius < 1.0
        point = jnp.where(inside, point, max_radius * point / safe_radius)
        squared_radius = jnp.sum(point * point, axis=-1, keepdims=True)
        denominator = 1.0 - squared_radius
        time = (1.0 + squared_radius) / denominator
        spatial = 2.0 * point / denominator
        return jnp.concatenate([time, spatial], axis=-1)

    def egrad_to_rgrad(self, x: Array, egrad: Array) -> Array:
        egrad = jnp.asarray(egrad)
        minkowski_grad = egrad.at[..., 0].multiply(-1.0)
        return self.tangent_project(x, minkowski_grad)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        spatial = jax.random.normal(key, shape=sample_shape + (self.dim,))
        spatial_norm = stable_norm(spatial, axis=-1, keepdims=True)
        time = jnp.hypot(jnp.ones_like(spatial_norm), spatial_norm)
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
            safe_n = jnp.where(n > 0.0, n, jnp.ones_like(n))
            u = jnp.where(n > 0.0, u / safe_n, u)
        return self._scale_tangent(u, scale)


__all__ = ["Hyperboloid"]
