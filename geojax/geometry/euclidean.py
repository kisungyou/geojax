"""Euclidean geometry in JAX.

This is the flat Euclidean manifold with the same small Manopt-style interface
used by the other GeoJAX geometries.  A point is a JAX array with shape
``size``.  For example, ``Euclidean(size=5)`` represents ``R^5`` and
``Euclidean(size=(3, 2))`` represents the vector space of ``3 x 2`` matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import math
from operator import mul
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import ExactGeometryMixin, as_sample_shape, validate_integer
from ._numerics import stable_metric_norm, stable_norm

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


def _parse_size(size: Shape) -> tuple[int, ...]:
    if isinstance(size, int):
        return (validate_integer(size, name="Euclidean size", minimum=1),)
    shape = tuple(validate_integer(v, name="Euclidean size entry", minimum=1) for v in size)
    if not shape:
        raise ValueError(
            "Euclidean size must be a positive integer or a nonempty positive shape tuple."
        )
    return shape


@dataclass(frozen=True, init=False)
class Euclidean(ExactGeometryMixin):
    """Flat Euclidean geometry.

    Parameters
    ----------
    size:
        Shape of one unbatched point. ``size=5`` is interpreted as ``(5,)``.
    atol:
        Tolerance used in shape/tangency checks.
    """

    hessian_conversion_is_exact = True
    riemannian_gradient_jvp_is_exact = True

    size: tuple[int, ...]
    atol: float

    def __init__(self, size: Shape, *, atol: float = 1e-6) -> None:
        if not math.isfinite(float(atol)) or float(atol) < 0.0:
            raise ValueError("Euclidean atol must be finite and nonnegative.")
        object.__setattr__(self, "size", _parse_size(size))
        object.__setattr__(self, "atol", float(atol))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size

    @property
    def dim(self) -> int:
        return int(reduce(mul, self.shape, 1))

    def belongs(self, x: Array, atol: float | None = None) -> Array:
        del atol
        x = jnp.asarray(x)
        if not self._shape_matches(x):
            return self._shape_failure(x)
        axes = tuple(range(-len(self.shape), 0))
        return jnp.all(jnp.isfinite(x), axis=axes)

    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array:
        del atol
        if not self._shape_matches(x, u):
            return self._shape_failure(x)
        x, u = self._check_shapes(("x", x), ("u", u))
        event_ndim = len(self.shape)
        x, u = jnp.broadcast_arrays(x, u)
        axes = tuple(range(-event_ndim, 0))
        return jnp.all(jnp.isfinite(x) & jnp.isfinite(u), axis=axes)

    def project(self, x: Array) -> Array:
        return self._check_shape(x, name="x")

    def tangent_project(self, x: Array, u: Array) -> Array:
        _, u = self._check_shapes(("x", x), ("u", u))
        return u

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        _, u, v = self._check_shapes(("x", x), ("u", u), ("v", v))
        axes = tuple(range(-len(self.shape), 0))
        return jnp.sum(u * v, axis=axes)

    def norm(self, x: Array, u: Array) -> Array:
        return stable_metric_norm(
            u,
            lambda normalized: self.inner(x, normalized, normalized),
            axis=tuple(range(-len(self.shape), 0)),
        )

    def lincomb(self, x: Array, *terms: Any) -> Array:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = self._scale_tangent(vec, coeff)
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(x, out)

    def exp(self, x: Array, u: Array) -> Array:
        x, u = self._check_shapes(("x", x), ("u", u))
        return x + u

    def retr(self, x: Array, u: Array, t: float | Array = 1.0) -> Array:
        x, u = self._check_shapes(("x", x), ("u", u))
        return x + self._scale_tangent(u, t)

    def log(self, x: Array, y: Array) -> Array:
        x, y = self._check_shapes(("x", x), ("y", y))
        return y - x

    def dist(self, x: Array, y: Array) -> Array:
        return self.norm(x, self.log(x, y))

    def transport(self, x: Array, y: Array, u: Array) -> Array:
        _, _, u = self._check_shapes(("x", x), ("y", y), ("u", u))
        return u

    def pair_mean(self, x: Array, y: Array) -> Array:
        x, y = self._check_shapes(("x", x), ("y", y))
        return 0.5 * (x + y)

    def egrad_to_rgrad(self, x: Array, egrad: Array) -> Array:
        _, egrad = self._check_shapes(("x", x), ("egrad", egrad))
        return egrad

    def ehess_to_rhess(self, x: Array, egrad: Array, ehess_vec: Array, u: Array) -> Array:
        _, _, ehess_vec, _ = self._check_shapes(
            ("x", x),
            ("egrad", egrad),
            ("ehess_vec", ehess_vec),
            ("u", u),
        )
        return ehess_vec

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        return jax.random.normal(key, shape=sample_shape + self.shape)

    def random_tangent(
        self,
        key: Array,
        x: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        self._check_shape(x, name="x")
        u = jax.random.normal(key, shape=jnp.shape(x))
        if normalize:
            axes = tuple(range(-len(self.shape), 0))
            nrm = stable_norm(u, axis=axes, keepdims=True)
            u = jnp.where(nrm > 0.0, u / nrm, u)
        return self._scale_tangent(u, scale)


__all__ = ["Euclidean"]
