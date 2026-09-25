"""Numerically stable kernels shared by GeoJAX geometries.

The helpers in this module remove *removable* singularities from formulas such
as ``sin(r) / r``.  They do not hide genuine geometric singularities such as
the non-unique logarithm at the antipode of a sphere.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

Array = Any


@jax.jit
def symmetric_part(A: Array) -> Array:
    """Symmetric part with linear AD and safe arithmetic at both range ends."""
    A = jnp.asarray(A, dtype=jnp.result_type(A, float))

    def half_sum(first, second):
        # Always halving first flushes the smallest normal entries to zero;
        # always adding first overflows the largest. Scale only large pairs.
        large = jnp.maximum(jnp.abs(first), jnp.abs(second)) > jnp.finfo(A.dtype).max / 2.0
        factor = jnp.where(large, 0.5, 1.0)
        first = jax.lax.optimization_barrier(first * factor)
        second = jax.lax.optimization_barrier(second * factor)
        return (first + second) / (2.0 * factor)

    # The constant, self-adjoint block system [[I,I],[I,-I]] has inverse
    # one-half itself. Its first solution is the desired average. Keeping
    # the arithmetic selection inside an opaque solve preserves linearity
    # under JVP transposition, including when A is itself a tangent tracer.
    def operator(pair):
        first, second = pair[0], pair[1]
        return jnp.stack((first + second, first - second))

    def solve(_, pair):
        first, second = pair[0], pair[1]
        return jnp.stack((half_sum(first, second), half_sum(first, -second)))

    # A single array also avoids partially symbolic-zero pytree leaves in
    # higher reverse-mode transformations of the unused skew component.
    pair = jnp.stack((A, jnp.swapaxes(A, -1, -2)))
    return jax.lax.custom_linear_solve(operator, pair, solve=solve, symmetric=True)[0]


def _series_cutoff(x: Array) -> Array:
    """Return a dtype-aware cutoff for arguments represented as squared radii."""
    dtype = jnp.result_type(jnp.asarray(x), float)
    return jnp.sqrt(jnp.finfo(dtype).eps)


def squared_norm(x: Array, *, axis: int | tuple[int, ...], keepdims: bool = False) -> Array:
    """Squared Euclidean norm, including its full Hessian at the origin."""
    return jnp.sum(jnp.asarray(x) * jnp.asarray(x), axis=axis, keepdims=keepdims)


def nonnegative(value: Array) -> Array:
    """Clip negative roundoff, retaining the interior derivative at zero.

    ``maximum(value, 0)`` assigns a half derivative at equality in JAX. That
    halves the Hessian when ``value`` is an already nonnegative quadratic.
    """
    value = jnp.asarray(value)
    return jnp.where(value >= 0.0, value, jnp.zeros_like(value))


def power_of_two_rescale(value: Array, *, axis: int | tuple[int, ...]) -> Array:
    """Rescale each slice to maximum magnitude in [1/2, 1), preserving zeros.

    Splitting the exponent avoids subnormal reciprocals and also supports JAX
    versions whose ldexp forms an intermediate power of two explicitly.
    """
    value = jnp.asarray(value)
    maximum = jnp.max(jnp.abs(value), axis=axis, keepdims=True)
    _, exponent = jnp.frexp(jax.lax.stop_gradient(maximum))
    first = -(exponent // 2)
    # ldexp itself selects derivative 1 at a zero entry in supported JAX
    # versions. Apply constant powers as multipliers so zero coordinates
    # receive the same linear derivative as every other coordinate.
    first_scale = jnp.ldexp(jnp.ones_like(maximum), first)
    second_scale = jnp.ldexp(jnp.ones_like(maximum), -exponent - first)
    return jax.lax.optimization_barrier(value * first_scale) * second_scale


def spherical_squared_dist(x: Array, y: Array, *, axis: int | tuple[int, ...]) -> Array:
    """Squared angle between unit vectors, accurate near coincidence.

    The half-angle formula uses both chords instead of subtracting a cosine
    from one. A series in the squared chord ratio fills the removable square
    root singularity at coincidence; the antipodal singularity remains.
    """
    chord = squared_norm(jnp.asarray(x) - jnp.asarray(y), axis=axis)
    antichord = squared_norm(jnp.asarray(x) + jnp.asarray(y), axis=axis)
    small = chord <= _series_cutoff(chord) * antichord
    safe_antichord = jnp.where(small, antichord, jnp.ones_like(antichord))
    ratio = jnp.where(small, chord, 0.0) / safe_antichord
    series = 4.0 * (ratio - 2.0 * ratio**2 / 3.0 + 23.0 * ratio**3 / 45.0)
    regular_chord = jnp.where(small, jnp.ones_like(chord), chord)
    angle = 2.0 * jnp.arctan2(jnp.sqrt(regular_chord), jnp.sqrt(antichord))
    return jnp.where(small, series, angle**2)


def stable_norm(
    x: Array,
    *,
    axis: int | tuple[int, ...],
    keepdims: bool = False,
) -> Array:
    """Euclidean norm evaluated after max-absolute-value rescaling.

    The rescaling prevents avoidable overflow and underflow in ``sum(x**2)``.
    At the origin, :func:`sqrt_nonnegative` selects the zero subgradient.
    """
    value = jnp.asarray(x)
    maximum = jnp.max(jnp.abs(value), axis=axis, keepdims=True)
    safe_maximum = jnp.where(maximum > 0.0, maximum, jnp.ones_like(maximum))
    normalized_squared = jnp.sum(
        (value / safe_maximum) ** 2,
        axis=axis,
        keepdims=True,
    )
    result = maximum * sqrt_nonnegative(normalized_squared)
    if keepdims:
        return result
    return jnp.squeeze(result, axis=axis)


def stable_metric_norm(
    vector: Array,
    quadratic_form: Callable[[Array], Array],
    *,
    axis: int | tuple[int, ...],
) -> Array:
    """Norm of a homogeneous positive quadratic form without squaring large data."""
    value = jnp.asarray(vector)
    maximum = jnp.max(jnp.abs(value), axis=axis, keepdims=True)
    safe_maximum = jnp.where(maximum > 0.0, maximum, jnp.ones_like(maximum))
    normalized = value / safe_maximum
    scale = jnp.squeeze(maximum, axis=axis)
    return scale * sqrt_nonnegative(jnp.maximum(quadratic_form(normalized), 0.0))


@jax.custom_jvp
def sqrt_nonnegative(value: Array) -> Array:
    """Square root on nonnegative inputs with a zero JVP at the origin.

    A norm, and therefore a metric distance, is not classically differentiable
    at zero.  Choosing the zero element of its subdifferential prevents the
    removable ``0 * inf`` produced when users differentiate expressions such
    as ``dist(x, y) ** 2``.  Away from zero this is the ordinary square root.
    """
    return jnp.sqrt(jnp.maximum(jnp.asarray(value), 0.0))


@sqrt_nonnegative.defjvp
def _sqrt_nonnegative_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    raw = jnp.asarray(value)
    positive = raw > 0.0
    safe_value = jnp.where(positive, raw, jnp.ones_like(raw))
    derivative = 0.5 / jnp.sqrt(safe_value)
    tangent = jnp.where(positive, derivative * value_dot, 0.0)
    return sqrt_nonnegative(raw), tangent


def cos_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``cos(sqrt(s))`` with a finite derivative at ``s = 0``."""
    s = nonnegative(squared_radius)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    regular = jnp.cos(jnp.sqrt(regular_s))
    series = 1.0 - 0.5 * s + s * s / 24.0 - s * s * s / 720.0
    return jnp.where(s > cutoff, regular, series)


def sinc_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``sin(sqrt(s)) / sqrt(s)`` at and near zero."""
    s = nonnegative(squared_radius)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    radius = jnp.sqrt(regular_s)
    regular = jnp.sin(radius) / radius
    series = 1.0 - s / 6.0 + s * s / 120.0 - s * s * s / 5040.0
    return jnp.where(s > cutoff, regular, series)


def cosh_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``cosh(sqrt(s))`` with a finite derivative at ``s = 0``."""
    s = nonnegative(squared_radius)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    regular = jnp.cosh(jnp.sqrt(regular_s))
    series = 1.0 + 0.5 * s + s * s / 24.0 + s * s * s / 720.0
    return jnp.where(s > cutoff, regular, series)


def sinhc_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``sinh(sqrt(s)) / sqrt(s)`` at and near zero."""
    s = nonnegative(squared_radius)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    radius = jnp.sqrt(regular_s)
    regular = jnp.sinh(radius) / radius
    series = 1.0 + s / 6.0 + s * s / 120.0 + s * s * s / 5040.0
    return jnp.where(s > cutoff, regular, series)


def tanhc_from_squared_norm(squared_radius: Array) -> Array:
    """Evaluate ``tanh(sqrt(s)) / sqrt(s)`` at and near zero."""
    s = nonnegative(squared_radius)
    cutoff = _series_cutoff(s)
    regular_s = jnp.where(s > cutoff, s, jnp.ones_like(s))
    radius = jnp.sqrt(regular_s)
    regular = jnp.tanh(radius) / radius
    series = 1.0 - s / 3.0 + 2.0 * s * s / 15.0
    return jnp.where(s > cutoff, regular, series)


def atanhc_from_squared_norm(squared_radius: Array, eps: float = 0.0) -> Array:
    """Evaluate ``atanh(sqrt(s)) / sqrt(s)`` at zero and inside the unit ball."""
    del eps
    s = nonnegative(squared_radius)
    cutoff = _series_cutoff(s)
    valid = s < 1.0
    regular_s = jnp.where(valid & (s > cutoff), s, 0.25 * jnp.ones_like(s))
    radius = jnp.sqrt(regular_s)
    regular = jnp.arctanh(radius) / radius
    series = 1.0 + s / 3.0 + s * s / 5.0
    interior = jnp.where(s > cutoff, regular, series)
    return jnp.where(valid, interior, jnp.full_like(interior, jnp.nan))


@jax.custom_jvp
def _acos_over_sin_from_cosine(cosine: Array) -> Array:
    c = jnp.clip(jnp.asarray(cosine), -1.0, 1.0)
    delta = 1.0 - c
    cutoff = _series_cutoff(delta)
    sine_squared = jnp.maximum(1.0 - c * c, 0.0)
    safe_sine_squared = jnp.where(sine_squared > 0.0, sine_squared, jnp.ones_like(c))
    regular = jnp.arccos(c) / jnp.sqrt(safe_sine_squared)
    series = 1.0 + delta / 3.0 + 2.0 * delta**2 / 15.0 + 2.0 * delta**3 / 35.0
    result = jnp.where(delta <= cutoff, series, regular)
    return jnp.where(c <= -1.0, jnp.full_like(result, jnp.inf), result)


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
    derivative = jnp.where(c <= -1.0, jnp.full_like(derivative, -jnp.inf), derivative)
    # The primal clips roundoff outside the cosine range. Its derivative must
    # consequently vanish there rather than extending the interior formula.
    tangent = jnp.where((cosine >= -1.0) & (cosine <= 1.0), derivative * cosine_dot, 0.0)
    return _acos_over_sin_from_cosine(c), tangent


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
    positive_sine = sine_squared > 0.0
    safe_sine = jnp.sqrt(jnp.where(positive_sine, sine_squared, jnp.ones_like(sine_squared)))
    sine = jnp.where(positive_sine, safe_sine, jnp.zeros_like(safe_sine))
    regular = jnp.arctan2(sine, c) / safe_sine
    series = 1.0 + delta / 3.0 + 2.0 * delta**2 / 15.0 + 2.0 * delta**3 / 35.0
    result = jnp.where(delta <= cutoff, series, regular)
    cut_locus = (c <= -1.0) & (sine_squared <= 0.0)
    return jnp.where(cut_locus, jnp.full_like(result, jnp.inf), result)


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
    tangent = jnp.where(
        at_cut,
        jnp.nan,
        jnp.where(cosine > 1.0, 0.0, derivative * cosine_dot),
    )
    return primal, tangent


@jax.custom_jvp
def acosh_squared(alpha: Array) -> Array:
    """Squared inverse hyperbolic cosine with a finite derivative at one."""
    a = jnp.maximum(jnp.asarray(alpha), 1.0)
    return jnp.arccosh(a) ** 2


@acosh_squared.defjvp
def _acosh_squared_jvp(primals, tangents):
    (alpha,), (alpha_dot,) = primals, tangents
    raw = jnp.asarray(alpha)
    a = jnp.maximum(raw, 1.0)
    primal = acosh_squared(a)
    tangent = jnp.where(raw >= 1.0, 2.0 * acosh_over_sqrt(a) * alpha_dot, 0.0)
    return primal, tangent


def asinh_squared_from_squared_chord(squared_chord: Array) -> Array:
    """Convert a nonnegative hyperbolic chord square to geodesic distance squared.

    For points on the unit hyperboloid, a Lorentzian chord with squared
    length ``c`` has geodesic distance ``2 * asinh(sqrt(c) / 2)``.  Evaluating
    the squared expression as a polynomial near zero avoids both the square
    root singularity and the loss of the small increment in ``acosh(1 + c/2)``.
    """
    c = nonnegative(squared_chord)
    cutoff = _series_cutoff(c)
    regular_c = jnp.where(c > cutoff, c, jnp.ones_like(c))
    regular = 4.0 * jnp.arcsinh(0.5 * jnp.sqrt(regular_c)) ** 2
    series = c - c * c / 12.0 + c * c * c / 90.0
    return jnp.where(c > cutoff, regular, series)


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
    "asinh_squared_from_squared_chord",
    "atanhc_from_squared_norm",
    "cos_from_squared_norm",
    "cosh_from_squared_norm",
    "matrix_expm",
    "nonnegative",
    "power_of_two_rescale",
    "sinc_from_squared_norm",
    "sinhc_from_squared_norm",
    "stable_metric_norm",
    "stable_norm",
    "sqrt_nonnegative",
    "squared_norm",
    "spherical_squared_dist",
    "tanhc_from_squared_norm",
]
