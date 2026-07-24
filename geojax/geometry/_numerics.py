"""Numerically stable kernels shared by GeoJAX geometries.

The helpers in this module remove *removable* singularities from formulas such
as ``sin(r) / r``.  They do not hide genuine geometric singularities such as
the non-unique logarithm at the antipode of a sphere.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

Array = Any


def _series_cutoff(x: Array) -> Array:
    """Return a dtype-aware cutoff for arguments represented as squared radii."""
    dtype = jnp.result_type(jnp.asarray(x), float)
    return jnp.sqrt(jnp.finfo(dtype).eps)


def squared_norm(x: Array, *, axis: int | tuple[int, ...], keepdims: bool = False) -> Array:
    """Squared Euclidean norm with negative roundoff clipped to zero."""
    value = jnp.sum(jnp.asarray(x) * jnp.asarray(x), axis=axis, keepdims=keepdims)
    return jnp.maximum(value, 0.0)


def cos_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``cos(sqrt(s))`` with a finite derivative at ``s = 0``."""
    s = jnp.maximum(jnp.asarray(squared_radius), 0.0)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    regular = jnp.cos(jnp.sqrt(regular_s))
    series = 1.0 - 0.5 * s + s * s / 24.0 - s * s * s / 720.0
    return jnp.where(s > cutoff, regular, series)


def sinc_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``sin(sqrt(s)) / sqrt(s)`` at and near zero."""
    s = jnp.maximum(jnp.asarray(squared_radius), 0.0)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    radius = jnp.sqrt(regular_s)
    regular = jnp.sin(radius) / radius
    series = 1.0 - s / 6.0 + s * s / 120.0 - s * s * s / 5040.0
    return jnp.where(s > cutoff, regular, series)


def cosh_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``cosh(sqrt(s))`` with a finite derivative at ``s = 0``."""
    s = jnp.maximum(jnp.asarray(squared_radius), 0.0)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    regular = jnp.cosh(jnp.sqrt(regular_s))
    series = 1.0 + 0.5 * s + s * s / 24.0 + s * s * s / 720.0
    return jnp.where(s > cutoff, regular, series)


def sinhc_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``sinh(sqrt(s)) / sqrt(s)`` at and near zero."""
    s = jnp.maximum(jnp.asarray(squared_radius), 0.0)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    radius = jnp.sqrt(regular_s)
    regular = jnp.sinh(radius) / radius
    series = 1.0 + s / 6.0 + s * s / 120.0 + s * s * s / 5040.0
    return jnp.where(s > cutoff, regular, series)


def tanhc_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``tanh(sqrt(s)) / sqrt(s)`` at and near zero."""
    s = jnp.maximum(jnp.asarray(squared_radius), 0.0)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    radius = jnp.sqrt(regular_s)
    regular = jnp.tanh(radius) / radius
    series = 1.0 - s / 3.0 + 2.0 * s * s / 15.0
    return jnp.where(s > cutoff, regular, series)


def atanhc_from_squared_norm(squared_radius: Array, eps: float = 0.0) -> Array:
    """Evaluate ``atanh(sqrt(s)) / sqrt(s)`` at zero and inside the unit ball."""
    s = jnp.maximum(jnp.asarray(squared_radius), 0.0)
    dtype = s.dtype
    cutoff = _series_cutoff(s)
    upper = (1.0 - jnp.maximum(jnp.asarray(eps, dtype=dtype), jnp.finfo(dtype).eps)) ** 2
    clipped = jnp.minimum(s, upper)
    regular_s = jnp.where(clipped > cutoff, clipped, jnp.ones_like(clipped))
    radius = jnp.sqrt(regular_s)
    regular = jnp.arctanh(radius) / radius
    series = 1.0 + clipped / 3.0 + clipped * clipped / 5.0
    return jnp.where(clipped > cutoff, regular, series)


@jax.custom_jvp
def _acos_over_sin_from_cosine(cosine: Array) -> Array:
    c = jnp.clip(jnp.asarray(cosine), -1.0, 1.0)
    delta = 1.0 - c
    cutoff = _series_cutoff(delta)
    sine_squared = jnp.maximum(1.0 - c * c, 0.0)
    safe_sine_squared = jnp.where(sine_squared > 0.0, sine_squared, jnp.ones_like(c))
    regular = jnp.arccos(c) / jnp.sqrt(safe_sine_squared)
    series = 1.0 + delta / 3.0 + 2.0 * delta**2 / 15.0 + 2.0 * delta**3 / 35.0
    return jnp.where(delta <= cutoff, series, regular)


@_acos_over_sin_from_cosine.defjvp
def _acos_over_sin_from_cosine_jvp(primals, tangents):
    (cosine,), (cosine_dot,) = primals, tangents
    c = jnp.clip(jnp.asarray(cosine), -1.0, 1.0)
    delta = 1.0 - c
    cutoff = _series_cutoff(delta)
    sine_squared = jnp.maximum(1.0 - c * c, 0.0)
    safe_sine_squared = jnp.where(
        sine_squared > 0.0,
        sine_squared,
        jnp.ones_like(c),
    )
    sine = jnp.sqrt(safe_sine_squared)
    regular_derivative = (c * jnp.arccos(c) - sine) / (sine * safe_sine_squared)
    series_derivative = -(1.0 / 3.0 + 4.0 * delta / 15.0 + 6.0 * delta**2 / 35.0)
    derivative = jnp.where(delta <= cutoff, series_derivative, regular_derivative)
    return _acos_over_sin_from_cosine(c), derivative * cosine_dot


def acos_over_sin(cosine: Array, sine_squared: Array | None = None) -> Array:
    """Evaluate ``acos(c) / sqrt(1 - c**2)`` near ``c = 1``.

    When available, ``sine_squared`` should be computed from the orthogonal
    component of the endpoint rather than by subtracting ``c**2`` from one.
    This permits the equivalent and more accurate
    ``atan2(sin(theta), cos(theta)) / sin(theta)`` evaluation near ``c = -1``.

    The value diverges at ``c = -1``.  Callers that represent spherical
    logarithms must continue to mark that genuine cut locus explicitly.
    """
    if sine_squared is None:
        return _acos_over_sin_from_cosine(cosine)

    c = jnp.clip(jnp.asarray(cosine), -1.0, 1.0)
    delta = 1.0 - c
    cutoff = _series_cutoff(delta)
    sine_squared = jnp.maximum(jnp.asarray(sine_squared), 0.0)
    safe_sine_squared = jnp.where(
        sine_squared > 0.0,
        sine_squared,
        jnp.ones_like(sine_squared),
    )
    sine = jnp.sqrt(safe_sine_squared)
    regular = jnp.arctan2(sine, c) / sine
    series = 1.0 + delta / 3.0 + 2.0 * delta**2 / 15.0 + 2.0 * delta**3 / 35.0
    return jnp.where(delta <= cutoff, series, regular)


@jax.custom_jvp
def acosh_over_sqrt(alpha: Array) -> Array:
    """Evaluate ``acosh(a) / sqrt(a**2 - 1)`` near ``a = 1``."""
    a = jnp.maximum(jnp.asarray(alpha), 1.0)
    delta = a - 1.0
    cutoff = _series_cutoff(delta)
    denominator_squared = jnp.maximum(a * a - 1.0, 0.0)
    safe_denominator_squared = jnp.where(
        denominator_squared > cutoff,
        denominator_squared,
        jnp.ones_like(a),
    )
    regular = jnp.arccosh(a) / jnp.sqrt(safe_denominator_squared)
    series = 1.0 - delta / 3.0 + 2.0 * delta**2 / 15.0 - 2.0 * delta**3 / 35.0
    return jnp.where(delta <= cutoff, series, regular)


@acosh_over_sqrt.defjvp
def _acosh_over_sqrt_jvp(primals, tangents):
    (alpha,), (alpha_dot,) = primals, tangents
    raw = jnp.asarray(alpha)
    a = jnp.maximum(raw, 1.0)
    delta = a - 1.0
    cutoff = _series_cutoff(delta)
    denominator_squared = jnp.maximum(a * a - 1.0, 0.0)
    safe_denominator_squared = jnp.where(
        denominator_squared > 0.0,
        denominator_squared,
        jnp.ones_like(a),
    )
    denominator = jnp.sqrt(safe_denominator_squared)
    regular_derivative = (denominator - a * jnp.arccosh(a)) / (
        denominator * safe_denominator_squared
    )
    series_derivative = -1.0 / 3.0 + 4.0 * delta / 15.0 - 6.0 * delta**2 / 35.0
    derivative = jnp.where(delta <= cutoff, series_derivative, regular_derivative)
    tangent = jnp.where(raw >= 1.0, derivative * alpha_dot, 0.0)
    return acosh_over_sqrt(raw), tangent


@jax.custom_jvp
def acos_squared(cosine: Array) -> Array:
    """Squared arccosine with its removable derivative singularity filled."""
    c = jnp.clip(jnp.asarray(cosine), -1.0, 1.0)
    return jnp.arccos(c) ** 2


@acos_squared.defjvp
def _acos_squared_jvp(primals, tangents):
    (cosine,), (cosine_dot,) = primals, tangents
    c = jnp.clip(jnp.asarray(cosine), -1.0, 1.0)
    primal = acos_squared(c)
    derivative = -2.0 * acos_over_sin(c)
    dtype = jnp.result_type(c, float)
    at_cut = c <= -1.0 + 8.0 * jnp.finfo(dtype).eps
    tangent = jnp.where(at_cut, jnp.nan, derivative * cosine_dot)
    return primal, tangent


@jax.custom_jvp
def acosh_squared(alpha: Array) -> Array:
    """Squared inverse hyperbolic cosine with a finite derivative at one."""
    a = jnp.maximum(jnp.asarray(alpha), 1.0)
    return jnp.arccosh(a) ** 2


@acosh_squared.defjvp
def _acosh_squared_jvp(primals, tangents):
    (alpha,), (alpha_dot,) = primals, tangents
    a = jnp.maximum(jnp.asarray(alpha), 1.0)
    primal = acosh_squared(a)
    tangent = 2.0 * acosh_over_sqrt(a) * alpha_dot
    return primal, tangent


def matrix_expm(A: Array) -> Array:
    """Apply JAX's matrix exponential over optional leading batch axes."""
    A = jnp.asarray(A)
    if A.ndim == 2:
        return jsp_linalg.expm(A)
    shape = A.shape
    flat = A.reshape((-1, shape[-2], shape[-1]))
    return jax.vmap(jsp_linalg.expm)(flat).reshape(shape)


__all__ = [
    "acos_over_sin",
    "acos_squared",
    "acosh_over_sqrt",
    "acosh_squared",
    "atanhc_from_squared_norm",
    "cos_from_squared_norm",
    "cosh_from_squared_norm",
    "matrix_expm",
    "sinc_from_squared_norm",
    "sinhc_from_squared_norm",
    "squared_norm",
    "tanhc_from_squared_norm",
]
