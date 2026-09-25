"""Symmetric positive-definite matrix geometries in JAX.

This module contains three Riemannian geometries on the same set

    SPD(n) = {P in Sym(n): P is positive definite}.

``SPDLogEuclidean`` uses the log-Euclidean metric, under which the matrix
logarithm is an isometry from SPD(n) to the vector space Sym(n).

``SPDAffineInvariant`` uses the affine-invariant metric

    g_P(U,V) = tr(P^{-1} U P^{-1} V).

``SPDBuresWasserstein`` uses the quotient metric induced by square-root
factors.  Its distance agrees with the 2-Wasserstein distance between
zero-mean Gaussian distributions.

All three geometries expose the same Manopt-style interface used by the
optimization module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from ._numerics import matrix_expm, nonnegative, symmetric_part as _sym

from .base import (
    ExactGeometryMixin,
    as_sample_shape,
    dtype_margin,
    validate_integer,
    validate_nonnegative,
    validate_positive,
)
from ._numerics import stable_metric_norm, stable_norm, sqrt_nonnegative

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


def _as_sample_shape(sample_shape: Shape = ()) -> tuple[int, ...]:
    return as_sample_shape(sample_shape)


def _parse_spd_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError("SPD size must be square, e.g. size=(3, 3).")
    shape = tuple(validate_integer(v, name="SPD size entry", minimum=1) for v in size)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("SPD size must be square, e.g. size=(3, 3).")
    if shape[0] < 1:
        raise ValueError("size must be positive for SPD geometry.")
    return shape


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(A * B, axis=(-2, -1))


def _matmul3(A: Array, B: Array, C: Array) -> Array:
    return A @ B @ C


def _eigh_sym(A: Array) -> tuple[Array, Array]:
    return jnp.linalg.eigh(_sym(A), symmetrize_input=False)


def _spd_from_eigh(Q: Array, vals: Array) -> Array:
    return (Q * vals[..., None, :]) @ jnp.swapaxes(Q, -1, -2)


def _sylvester_eigenbasis_impl(P: Array, U: Array) -> Array:
    """Solve ``P A + A P = U`` for an arbitrary matrix right-hand side."""
    eigenvalues, eigenvectors = _eigh_sym(P)
    rotated = jnp.swapaxes(eigenvectors, -1, -2) @ U @ eigenvectors
    denominator = eigenvalues[..., :, None] + eigenvalues[..., None, :]
    return eigenvectors @ (rotated / denominator) @ jnp.swapaxes(eigenvectors, -1, -2)


def _solve_spd_sylvester(P: Array, U: Array) -> Array:
    """SPD Sylvester solve with implicit derivatives at every order.

    Only the primal solve uses an eigenbasis. Differentiation acts on the
    defining equation, including reverse-mode residuals, so repeated
    eigenvalues never require differentiating a choice of eigenvectors.
    """
    P, U = jnp.broadcast_arrays(_sym(jnp.asarray(P)), _sym(jnp.asarray(U)))
    # A common exact power-of-two scale avoids overflow in eigenvalue sums
    # and in the defining operator, without changing the solution.
    maximum = jnp.max(jnp.abs(jax.lax.stop_gradient(P)), axis=(-2, -1), keepdims=True)
    _, exponent = jnp.frexp(maximum)
    first = -(exponent // 2)
    first_scale = jnp.ldexp(jnp.ones_like(maximum), first)
    second_scale = jnp.ldexp(jnp.ones_like(maximum), -exponent - first)
    P = jax.lax.optimization_barrier(P * first_scale) * second_scale
    U = jax.lax.optimization_barrier(U * first_scale) * second_scale
    solution = jax.lax.custom_linear_solve(
        lambda X: P @ X + X @ P,
        U,
        solve=lambda _, rhs: _sylvester_eigenbasis_impl(P, rhs),
        symmetric=True,
    )
    return _sym(solution)


def _spd_logm(P: Array, eps: float) -> Array:
    return _spd_logm_differentiable(P, eps)


def _spd_sqrtm(P: Array, eps: float) -> Array:
    return _spd_sqrtm_differentiable(_spd_project_differentiable(P, eps))


def _spd_invsqrtm(P: Array, eps: float) -> Array:
    return _spd_invsqrtm_differentiable(_spd_project_differentiable(P, eps))


def _spd_invm(P: Array, eps: float) -> Array:
    return jnp.linalg.inv(_spd_project_differentiable(P, eps))


def _spectral_divided_difference(
    eigenvalues: Array,
    function_values: Array,
    derivatives: Array,
    *,
    scale_floor: float = 1.0,
) -> Array:
    """Stable Loewner matrix for a scalar spectral function."""
    lam_i = eigenvalues[..., :, None]
    lam_j = eigenvalues[..., None, :]
    value_i = function_values[..., :, None]
    value_j = function_values[..., None, :]
    deriv_i = derivatives[..., :, None]
    deriv_j = derivatives[..., None, :]
    denominator = lam_i - lam_j
    dtype = jnp.result_type(eigenvalues, float)
    scale = jnp.maximum(
        jnp.asarray(scale_floor, dtype=dtype),
        jnp.maximum(jnp.abs(lam_i), jnp.abs(lam_j)),
    )
    scale = jnp.maximum(scale, jnp.finfo(dtype).tiny)
    separated = jnp.abs(denominator) > 32.0 * jnp.finfo(dtype).eps * scale
    safe_denominator = jnp.where(separated, denominator, jnp.ones_like(denominator))
    quotient = (value_i - value_j) / safe_denominator
    repeated = 0.5 * (deriv_i + deriv_j)
    return jnp.where(separated, quotient, repeated)


def _projection_eigh(P: Array) -> tuple[Array, Array]:
    """Resolve signed spectra after exact power scaling, including finite-range extremes."""
    maximum = jnp.max(jnp.abs(jax.lax.stop_gradient(P)), axis=(-2, -1), keepdims=True)
    _, exponent = jnp.frexp(maximum)
    first = -(exponent // 2)
    first_scale = jnp.ldexp(jnp.ones_like(maximum), first)
    second_scale = jnp.ldexp(jnp.ones_like(maximum), -exponent - first)
    normalized = jax.lax.optimization_barrier(P * first_scale) * second_scale
    values, vectors = _eigh_sym(normalized)
    undo_second = jnp.ldexp(jnp.ones_like(maximum), exponent + first)[..., 0]
    undo_first = jnp.ldexp(jnp.ones_like(maximum), -first)[..., 0]
    return jax.lax.optimization_barrier(values * undo_second) * undo_first, vectors


@jax.custom_jvp
def _symmetric_abs_invertible(P: Array) -> Array:
    """Absolute value of an invertible symmetric matrix, including derivatives."""
    values, vectors = _projection_eigh(P)
    return _spd_from_eigh(vectors, jnp.abs(values))


def _normalize_abs_equation(P: Array, absolute: Array) -> tuple[Array, Array]:
    """Apply a common, derivative-constant power scale to an absolute-value equation."""
    maximum = jnp.max(jnp.abs(jax.lax.stop_gradient(P)), axis=(-2, -1), keepdims=True)
    _, exponent = jnp.frexp(maximum)
    first = -(exponent // 2)
    first_scale = jnp.ldexp(jnp.ones_like(maximum), first)
    second_scale = jnp.ldexp(jnp.ones_like(maximum), -exponent - first)
    return (
        jax.lax.optimization_barrier(P * first_scale) * second_scale,
        jax.lax.optimization_barrier(absolute * first_scale) * second_scale,
    )


@_symmetric_abs_invertible.defjvp
def _symmetric_abs_invertible_jvp(primals, tangents):
    (P,), (E,) = primals, tangents
    P, E = _sym(P), _sym(E)
    absolute = _symmetric_abs_invertible(P)
    # Differentiate |P|^2=P^2. Unlike sqrt(P@P), this never forms a square
    # in the primal. Normalize the defining equation to avoid large sums of
    # eigenvalues in its Sylvester solve.
    normalized, normalized_absolute = _normalize_abs_equation(P, absolute)
    normalized = 0.5 * normalized
    rhs = normalized @ E + E @ normalized
    derivative = _solve_spd_sylvester(0.5 * normalized_absolute, rhs)
    return absolute, derivative


def _invertible_spd_repair(P: Array, eps: float) -> Array:
    absolute = _symmetric_abs_invertible(P)
    normalized, normalized_absolute = _normalize_abs_equation(P, absolute)
    sign = jnp.linalg.solve(normalized_absolute, normalized)
    identity = jnp.eye(P.shape[-1], dtype=P.dtype)
    half_P = jax.lax.optimization_barrier(0.5 * P)
    half_absolute = jax.lax.optimization_barrier(0.5 * absolute)
    return _sym(half_P + half_absolute + (0.5 * eps) * (identity - sign))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _spd_project_differentiable(P: Array, eps: float) -> Array:
    """Repair nonpositive eigenvalues while preserving every valid SPD point."""
    P = _sym(jnp.asarray(P))
    eigenvalues, eigenvectors = _projection_eigh(P)
    clipped = jnp.where(eigenvalues > 0.0, eigenvalues, eps)
    repaired = _spd_from_eigh(eigenvectors, clipped)
    valid = jnp.all(eigenvalues > 0.0, axis=-1)
    return jnp.where(valid[..., None, None], P, repaired)


@_spd_project_differentiable.defjvp
def _spd_project_differentiable_jvp(eps, primals, tangents):
    (P,), (E,) = primals, tangents
    P = _sym(jnp.asarray(P))
    E = _sym(jnp.asarray(E))
    # On the open SPD cone projection is exactly the identity. Keeping that
    # branch free of eigenvectors also permits higher derivatives there.
    eigenvalues, eigenvectors = _projection_eigh(jax.lax.stop_gradient(P))
    positive = jnp.all(eigenvalues > 0.0, axis=-1)
    invertible = jnp.all(eigenvalues != 0.0, axis=-1)

    def repaired_derivative(_):
        # Positive inputs retain the exact identity derivative. Every other
        # invertible input uses a smooth matrix absolute value; masking before
        # its evaluation also keeps vmap's inactive branches nonsingular.
        active = invertible & ~positive
        identity = jnp.eye(P.shape[-1], dtype=P.dtype)
        safe_P = jnp.where(active[..., None, None], P, identity)
        safe_E = jnp.where(active[..., None, None], E, jnp.zeros_like(E))
        _, smooth = jax.jvp(lambda A: _invertible_spd_repair(A, eps), (safe_P,), (safe_E,))
        smooth = jnp.where(positive[..., None, None], E, smooth)

        # Repair is not differentiable at a zero eigenvalue. Preserve its
        # previous selected first derivative there, with frozen coefficients:
        # a boundary item must not introduce eigenvector NaNs into derivatives
        # of separate, smooth entries of the same batch. No higher-order
        # mathematical derivative is asserted at this discontinuous boundary.
        clipped = jnp.where(eigenvalues > 0.0, eigenvalues, eps)
        derivatives = (eigenvalues > 0.0).astype(P.dtype)
        loewner = _spectral_divided_difference(eigenvalues, clipped, derivatives, scale_floor=0.0)
        rotated = jnp.swapaxes(eigenvectors, -1, -2) @ E @ eigenvectors
        boundary = _sym(eigenvectors @ (loewner * rotated) @ jnp.swapaxes(eigenvectors, -1, -2))
        return jnp.where(invertible[..., None, None], smooth, boundary)

    derivative = jax.lax.cond(jnp.all(positive), lambda _: E, repaired_derivative, operand=None)
    return _spd_project_differentiable(P, eps), derivative


# Host-driven optimizers repeatedly differentiate this operation. Reuse its
# compiled conditional and transpose instead of lowering the complete repair
# branch at every gradient evaluation. The static floor matches its custom-JVP
# nondifferentiable argument; nested JIT, vmap and higher AD remain supported.
_spd_project_differentiable = jax.jit(_spd_project_differentiable, static_argnums=(1,))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _spd_logm_differentiable(P: Array, eps: float) -> Array:
    """Principal symmetric matrix logarithm with a stable Frechet derivative."""
    P = _sym(jnp.asarray(P))
    eigenvalues, eigenvectors = _eigh_sym(P)
    del eps
    safe = jnp.maximum(eigenvalues, jnp.finfo(P.dtype).tiny)
    return _spd_from_eigh(eigenvectors, jnp.log(safe))


@_spd_logm_differentiable.defjvp
def _spd_logm_differentiable_jvp(eps, primals, tangents):
    (P,), (E,) = primals, tangents
    P, E = _sym(jnp.asarray(P)), _sym(jnp.asarray(E))
    logarithm = _spd_logm_differentiable(P, eps)
    return logarithm, _sym(_dlog_implicit(P, logarithm, E))


@jax.custom_jvp
def _spd_sqrtm_differentiable(P: Array) -> Array:
    """Principal SPD square root with a degeneracy-safe custom derivative."""
    P = _sym(jnp.asarray(P))
    vals, eigvecs = _eigh_sym(P)
    roots = jnp.sqrt(jnp.maximum(vals, jnp.finfo(P.dtype).tiny))
    return _spd_from_eigh(eigvecs, roots)


@_spd_sqrtm_differentiable.defjvp
def _spd_sqrtm_differentiable_jvp(primals, tangents):
    (P,), (E,) = primals, tangents
    root = _spd_sqrtm_differentiable(P)
    return root, _solve_spd_sylvester(root, E)


@jax.custom_jvp
def _spd_invsqrtm_differentiable(P: Array) -> Array:
    """Principal inverse SPD square root with a stable custom derivative."""
    P = _sym(jnp.asarray(P))
    vals, eigvecs = _eigh_sym(P)
    roots = jnp.sqrt(jnp.maximum(vals, jnp.finfo(P.dtype).tiny))
    return _spd_from_eigh(eigvecs, 1.0 / roots)


@_spd_invsqrtm_differentiable.defjvp
def _spd_invsqrtm_differentiable_jvp(primals, tangents):
    (P,), (E,) = primals, tangents
    root = _spd_sqrtm_differentiable(P)
    inverse_root = _spd_invsqrtm_differentiable(P)
    root_dot = _solve_spd_sylvester(root, E)
    return inverse_root, -_sym(inverse_root @ root_dot @ inverse_root)


def _dlog_eigenbasis_impl(P: Array, E: Array) -> Array:
    """Spectral Dlog solve, also valid for nonsymmetric right-hand sides."""
    vals, Q = _eigh_sym(P)
    safe = jnp.maximum(vals, jnp.finfo(P.dtype).tiny)
    high = jnp.maximum(safe[..., :, None], safe[..., None, :])
    low = jnp.minimum(safe[..., :, None], safe[..., None, :])
    # log(high)-log(low) loses digits for nearly equal eigenvalues whose
    # common scale is very large or small. Work with their relative gap,
    # and divide by high only after taking the dimensionless log mean.
    relative_gap = (high - low) / high
    close = relative_gap < 0.5
    gap = jnp.where(relative_gap > 0.0, relative_gap, 1.0)
    close_gap = jnp.where(close, relative_gap, 0.0)
    close_ratio = -jnp.log1p(-close_gap) / gap
    close_ratio = jnp.where(relative_gap > 0.0, close_ratio, 1.0)
    far_ratio = (jnp.log(high) - jnp.log(low)) / gap
    loewner = jnp.where(close, close_ratio, far_ratio) / high
    rotated = jnp.swapaxes(Q, -1, -2) @ E @ Q
    return Q @ (loewner * rotated) @ jnp.swapaxes(Q, -1, -2)


def _dlog_implicit(P: Array, logarithm: Array, E: Array) -> Array:
    """Invert Dexp at log(P) without differentiating its spectral solve.

    Dexp is self-adjoint and positive definite on the full matrix space
    when its base is symmetric. Its basis-independent JVP therefore gives
    an invertible defining operator for Dlog, even at repeated spectra.
    """
    P, logarithm, E = jnp.broadcast_arrays(P, logarithm, E)
    # Dlog_(cP)[cE] = Dlog_P[E]. Keep the defining exponential near unit
    # scale so higher derivatives do not form products at the underflow
    # boundary merely because the covariance units are very small.
    maximum = jnp.max(jnp.abs(jax.lax.stop_gradient(P)), axis=(-2, -1), keepdims=True)
    _, exponent = jnp.frexp(maximum)
    first = -(exponent // 2)
    first_scale = jnp.ldexp(jnp.ones_like(maximum), first)
    second_scale = jnp.ldexp(jnp.ones_like(maximum), -exponent - first)
    P = jax.lax.optimization_barrier(P * first_scale) * second_scale
    E = jax.lax.optimization_barrier(E * first_scale) * second_scale
    logarithm = logarithm - (exponent * jnp.log(jnp.asarray(2.0, dtype=P.dtype))) * jnp.eye(
        P.shape[-1], dtype=P.dtype
    )
    return jax.lax.custom_linear_solve(
        lambda X: jax.jvp(matrix_expm, (logarithm,), (X,))[1],
        E,
        solve=lambda _, rhs: _dlog_eigenbasis_impl(P, rhs),
        symmetric=True,
    )


@jax.jit
def _dexp_scaled(A: Array, E: Array) -> Array:
    """Symmetric Dexp with accurate scaling factors and smooth higher AD.

    Padé primal errors at a large negative shift are amplified in its JVP.
    Start with a fixed Padé approximant near zero, then recover larger arguments
    using Dexp_(2B)(E) = sym(exp(B) Dexp_B(E)).
    Each exponential factor has the accurate spectral primal and the same
    basis-independent custom derivative, rather than an accumulated Padé
    approximation. The fixed loop bound permits reverse differentiation.
    """
    norm = jnp.max(jnp.sum(jnp.abs(jax.lax.stop_gradient(A)), axis=-1), initial=0.0)
    steps = jnp.clip(jnp.ceil(jnp.log2(jnp.maximum(norm, 0.5))) + 1, 0, 16).astype(jnp.int32)
    one = jnp.asarray(1.0, dtype=A.dtype)
    scale = jnp.ldexp(one, -steps)

    def small_exponential(matrix):
        # Fixed Padé7 has ample accuracy at norm <= 1/2. Unlike native expm,
        # it has no conditional squaring JVP to transpose through vmap.
        # Keep the common coefficient scale: normalizing each coefficient
        # would flush needed intermediate products for tiny tangents.
        identity = jnp.broadcast_to(jnp.eye(A.shape[-1], dtype=A.dtype), A.shape)
        square = matrix @ matrix
        fourth = square @ square
        sixth = fourth @ square
        odd = matrix @ (sixth + 1512.0 * fourth + 277200.0 * square + 8648640.0 * identity)
        even = 56.0 * sixth + 25200.0 * fourth + 1995840.0 * square + 17297280.0 * identity
        return jnp.linalg.solve(even - odd, even + odd)

    derivative = jax.jvp(small_exponential, (A * scale,), (E,))[1]

    indices = jnp.arange(16, dtype=jnp.int32)
    leading_shape = (16,) + (1,) * A.ndim
    active = (indices < steps).reshape(leading_shape)
    powers = jnp.ldexp(one, indices - steps).reshape(leading_shape)
    arguments = jnp.where(active, A[None, ...] * powers, jnp.zeros_like(A)[None, ...])
    # Keep nonlinear factors outside scan so higher reverse-mode transforms
    # preserve their custom derivatives when differentiating scan residuals.
    factors = _spd_expm(arguments)

    def recover(index, current):
        # Average after multiplication: separately halved subnormal products
        # can both flush to zero despite a representable final derivative.
        return _sym(factors[index] @ current)

    return jax.lax.fori_loop(0, 16, recover, derivative)


def _frechet_spectral(A: Array, E: Array, func_name: str, eps: float) -> Array:
    """Frechet derivative of exp or log, including stable higher derivatives."""
    A, E = jnp.broadcast_arrays(_sym(jnp.asarray(A)), _sym(jnp.asarray(E)))
    if func_name == "exp":
        return _sym(_dexp_scaled(A, E))
    if func_name == "log":
        return _sym(_dlog_implicit(A, _spd_logm_differentiable(A, eps), E))
    raise ValueError("func_name must be 'exp' or 'log'.")


@jax.custom_jvp
def _spd_expm(A: Array) -> Array:
    """Symmetric matrix exponential with accurate small eigenvalues."""
    A = _sym(jnp.asarray(A))
    eigenvalues, eigenvectors = _eigh_sym(A)
    return _spd_from_eigh(eigenvectors, jnp.exp(eigenvalues))


@_spd_expm.defjvp
def _spd_expm_jvp(primals, tangents):
    (A,), (E,) = primals, tangents
    A = _sym(jnp.asarray(A))
    primal = _spd_expm(A)
    tangent = _frechet_spectral(A, E, "exp", 0.0)
    return primal, tangent


@dataclass(frozen=True, init=False)
class SPDLogEuclidean(ExactGeometryMixin):
    """Log-Euclidean geometry on SPD(size).

    The logarithm map ``log: SPD(n) -> Sym(n)`` is an isometry.  Therefore
    distances and geodesics are Euclidean after applying the matrix logarithm.
    Points are represented as SPD matrices of shape ``(size, size)``.
    Tangent vectors are symmetric matrices of the same shape.
    """

    size: tuple[int, int]
    atol: float
    eps: float

    def __init__(
        self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10
    ) -> None:
        object.__setattr__(self, "size", _parse_spd_size(size))
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="SPDLogEuclidean atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="SPDLogEuclidean eps"))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        P = jnp.asarray(P)
        if not self._shape_matches(P):
            return self._shape_failure(P)
        sym_ok = stable_norm(P - jnp.swapaxes(P, -1, -2), axis=(-2, -1)) <= tol
        vals, _ = _projection_eigh(_sym(P))
        pd_ok = jnp.min(vals, axis=-1) > 0.0
        return sym_ok & pd_ok

    def is_tangent(self, P: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(P, U):
            return self._shape_failure(P)
        _, U = self._check_shapes(("P", P), ("U", U))
        return stable_norm(U - jnp.swapaxes(U, -1, -2), axis=(-2, -1)) <= tol

    def project(self, P: Array) -> Array:
        P = self._check_shape(P, name="P")
        floor = dtype_margin(P, configured=self.eps)
        return _spd_project_differentiable(P, floor)

    def tangent_project(self, P: Array, U: Array) -> Array:
        _, U = self._check_shapes(("P", P), ("U", U))
        return _sym(U)

    def logm(self, P: Array) -> Array:
        return _spd_logm(self.project(P), self.eps)

    def expm(self, A: Array) -> Array:
        return _spd_expm(_sym(self._check_shape(A, name="A")))

    def dlog(self, P: Array, U: Array) -> Array:
        return _frechet_spectral(self.project(P), self.tangent_project(P, U), "log", self.eps)

    def dexp(self, A: Array, W: Array) -> Array:
        return _frechet_spectral(_sym(A), _sym(W), "exp", self.eps)

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        dU = self.dlog(P, U)
        dV = self.dlog(P, V)
        return _trace_inner(dU, dV)

    def norm(self, P: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(P, normalized, normalized),
            axis=(-2, -1),
        )

    def exp(self, P: Array, U: Array) -> Array:
        P = self.project(P)
        U = self.tangent_project(P, U)
        A = self.logm(P)
        W = self.dlog(P, U)
        return self.expm(A + W)

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(P, self._scale_tangent(U, t))

    def log(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        A = self.logm(P)
        B = self.logm(Q)
        return self.dexp(A, B - A)

    def dist(self, P: Array, Q: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(P, Q))

    def squared_dist(self, P: Array, Q: Array) -> Array:
        """Squared Euclidean distance between matrix-log coordinates."""
        D = self.logm(self.project(Q)) - self.logm(self.project(P))
        return _trace_inner(D, D)

    def transport(self, P: Array, Q: Array, U: Array) -> Array:
        B = self.logm(self.project(Q))
        W = self.dlog(P, U)
        return self.dexp(B, W)

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        P = self.project(P)
        E = self.tangent_project(P, egrad)
        A = self.logm(P)
        # Pull the covector to log coordinates and push the Euclidean gradient
        # back through the inverse chart.  Dexp_A is self-adjoint for symmetric A.
        grad_A = self.dexp(A, E)
        return self.dexp(A, grad_A)

    def lincomb(self, P: Array, *terms: Any) -> Array:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = self._scale_tangent(vec, coeff)
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(P, out)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = _as_sample_shape(sample_shape)
        A = jax.random.normal(key, shape=sample_shape + self.shape)
        A = _sym(A) / jnp.sqrt(float(self.n))
        return self.expm(A)

    def random_tangent(
        self,
        key: Array,
        P: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        Z = jax.random.normal(key, shape=jnp.shape(P))
        U = self.tangent_project(P, Z)
        if normalize:
            n = self.norm(P, U)[..., None, None]
            safe_n = jnp.where(n > 0.0, n, jnp.ones_like(n))
            U = jnp.where(n > 0.0, U / safe_n, U)
        return self._scale_tangent(U, scale)


@dataclass(frozen=True, init=False)
class SPDAffineInvariant(ExactGeometryMixin):
    """Affine-invariant geometry on SPD(size).

    The metric is

        g_P(U,V) = tr(P^{-1} U P^{-1} V).

    This is the canonical symmetric-space geometry of GL(n)/O(n).
    """

    size: tuple[int, int]
    atol: float
    eps: float

    def __init__(
        self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10
    ) -> None:
        object.__setattr__(self, "size", _parse_spd_size(size))
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="SPDAffineInvariant atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="SPDAffineInvariant eps"))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        P = jnp.asarray(P)
        if not self._shape_matches(P):
            return self._shape_failure(P)
        sym_ok = stable_norm(P - jnp.swapaxes(P, -1, -2), axis=(-2, -1)) <= tol
        vals, _ = _projection_eigh(_sym(P))
        pd_ok = jnp.min(vals, axis=-1) > 0.0
        return sym_ok & pd_ok

    def is_tangent(self, P: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(P, U):
            return self._shape_failure(P)
        _, U = self._check_shapes(("P", P), ("U", U))
        return stable_norm(U - jnp.swapaxes(U, -1, -2), axis=(-2, -1)) <= tol

    def project(self, P: Array) -> Array:
        P = self._check_shape(P, name="P")
        floor = dtype_margin(P, configured=self.eps)
        return _spd_project_differentiable(P, floor)

    def tangent_project(self, P: Array, U: Array) -> Array:
        _, U = self._check_shapes(("P", P), ("U", U))
        return _sym(U)

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        P = self.project(P)
        U = self.tangent_project(P, U)
        V = self.tangent_project(P, V)
        Pinv = _spd_invm(P, self.eps)
        return _trace_inner(Pinv @ U @ Pinv, V)

    def norm(self, P: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(P, normalized, normalized),
            axis=(-2, -1),
        )

    def exp(self, P: Array, U: Array) -> Array:
        P = self.project(P)
        U = self.tangent_project(P, U)
        Psqrt = _spd_sqrtm(P, self.eps)
        Pinvsqrt = _spd_invsqrtm(P, self.eps)
        A = Pinvsqrt @ U @ Pinvsqrt
        return self.project(Psqrt @ _spd_expm(A) @ Psqrt)

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(P, self._scale_tangent(U, t))

    def log(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        Psqrt = _spd_sqrtm(P, self.eps)
        Pinvsqrt = _spd_invsqrtm(P, self.eps)
        A = Pinvsqrt @ Q @ Pinvsqrt
        return _sym(Psqrt @ _spd_logm(A, self.eps) @ Psqrt)

    def squared_dist(self, P: Array, Q: Array) -> Array:
        """Squared affine-invariant distance."""
        P = self.project(P)
        Q = self.project(Q)
        Pinvsqrt = _spd_invsqrtm(P, self.eps)
        A = Pinvsqrt @ Q @ Pinvsqrt
        L = _spd_logm(A, self.eps)
        return _trace_inner(L, L)

    def dist(self, P: Array, Q: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(P, Q))

    def transport(self, P: Array, Q: Array, U: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        U = self.tangent_project(P, U)
        Psqrt = _spd_sqrtm(P, self.eps)
        Pinvsqrt = _spd_invsqrtm(P, self.eps)
        A = Pinvsqrt @ Q @ Pinvsqrt
        Asqrt = _spd_sqrtm(A, self.eps)
        E = Psqrt @ Asqrt @ Pinvsqrt
        return _sym(E @ U @ jnp.swapaxes(E, -1, -2))

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        P = self.project(P)
        E = self.tangent_project(P, egrad)
        return _sym(P @ E @ P)

    def lincomb(self, P: Array, *terms: Any) -> Array:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = self._scale_tangent(vec, coeff)
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(P, out)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = _as_sample_shape(sample_shape)
        A = jax.random.normal(key, shape=sample_shape + self.shape)
        A = _sym(A) / jnp.sqrt(float(self.n))
        return _spd_expm(A)

    def random_tangent(
        self,
        key: Array,
        P: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        Z = jax.random.normal(key, shape=jnp.shape(P))
        U = self.tangent_project(P, Z)
        if normalize:
            n = self.norm(P, U)[..., None, None]
            safe_n = jnp.where(n > 0.0, n, jnp.ones_like(n))
            U = jnp.where(n > 0.0, U / safe_n, U)
        return self._scale_tangent(U, scale)


@dataclass(frozen=True, init=False)
class SPDBuresWasserstein(ExactGeometryMixin):
    """Bures-Wasserstein geometry on SPD(size).

    If ``P = Q diag(d) Q.T`` and ``U_tilde = Q.T @ U @ Q``, the metric is

        g_P(U,V) = 1/2 sum_ij U_tilde_ij V_tilde_ij / (d_i + d_j).

    The exponential is defined only while its horizontal square-root lift is
    nonsingular.  ``transport`` is an exact isometric vector transport for
    optimization; it is not the Levi-Civita parallel transport, whose general
    evaluation requires integrating a differential equation.
    """

    transport_is_parallel = False

    size: tuple[int, int]
    atol: float
    eps: float

    def __init__(
        self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10
    ) -> None:
        object.__setattr__(self, "size", _parse_spd_size(size))
        object.__setattr__(
            self, "atol", validate_nonnegative(atol, name="SPDBuresWasserstein atol")
        )
        object.__setattr__(self, "eps", validate_positive(eps, name="SPDBuresWasserstein eps"))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        P = jnp.asarray(P)
        if not self._shape_matches(P):
            return self._shape_failure(P)
        sym_ok = stable_norm(P - jnp.swapaxes(P, -1, -2), axis=(-2, -1)) <= tol
        vals, _ = _projection_eigh(_sym(P))
        pd_ok = jnp.min(vals, axis=-1) > 0.0
        return sym_ok & pd_ok

    def project(self, P: Array) -> Array:
        P = self._check_shape(P, name="P")
        floor = dtype_margin(P, configured=self.eps)
        return _spd_project_differentiable(P, floor)

    def is_tangent(self, P: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(P, U):
            return self._shape_failure(P)
        _, U = self._check_shapes(("P", P), ("U", U))
        return stable_norm(U - jnp.swapaxes(U, -1, -2), axis=(-2, -1)) <= tol

    def tangent_project(self, P: Array, U: Array) -> Array:
        _, U = self._check_shapes(("P", P), ("U", U))
        return _sym(U)

    def sylvester(self, P: Array, U: Array) -> Array:
        """Solve ``P A + A P = U`` for symmetric ``A``."""
        P = self.project(P)
        U = self.tangent_project(P, U)
        return _solve_spd_sylvester(P, U)

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        P = self.project(P)
        V = self.tangent_project(P, V)
        return 0.5 * _trace_inner(self.sylvester(P, U), V)

    def norm(self, P: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(P, normalized, normalized),
            axis=(-2, -1),
        )

    def exp(self, P: Array, U: Array) -> Array:
        P = self.project(P)
        U = self.tangent_project(P, U)
        A = self.sylvester(P, U)
        # P + U + A P A = (I + A) P (I + A) on the valid branch.
        result = _sym(P + U + A @ P @ A)
        lift = jnp.eye(self.n, dtype=P.dtype) + A
        # The horizontal lift is (I + t A) P^(1/2). It remains full rank for
        # every t in [0, 1] exactly when I + A is positive definite. Endpoint
        # invertibility alone would accept paths that hit the PSD boundary and
        # later re-enter SPD.
        valid = jnp.min(jnp.linalg.eigvalsh(_sym(lift)), axis=-1) > 0.0
        return jnp.where(valid[..., None, None], result, jnp.full_like(result, jnp.nan))

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(P, self._scale_tangent(U, t))

    def optimal_transport_map(self, P: Array, Q: Array) -> Array:
        """Return the optimal Gaussian transport map from covariance P to Q."""
        P = self.project(P)
        Q = self.project(Q)
        Psqrt = _spd_sqrtm_differentiable(P)
        Pinvsqrt = _spd_invsqrtm_differentiable(P)
        middle = _spd_sqrtm_differentiable(Psqrt @ Q @ Psqrt)
        return _sym(Pinvsqrt @ middle @ Pinvsqrt)

    def log(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        transport_map = self.optimal_transport_map(P, Q)
        displacement = transport_map - jnp.eye(self.n, dtype=P.dtype)
        return _sym(displacement @ P + P @ displacement)

    def squared_dist(self, P: Array, Q: Array) -> Array:
        # The equivalent trace identity suffers catastrophic cancellation for
        # nearby matrices. The exact logarithm has norm equal to geodesic
        # distance and retains all resolvable first-order displacement.
        P = self.project(P)
        tangent = self.log(P, Q)
        return nonnegative(self.inner(P, tangent, tangent))

    def dist(self, P: Array, Q: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(P, Q))

    def _metric_coordinates(self, P: Array, U: Array) -> Array:
        """Map a tangent isometrically to the fixed Euclidean Sym(n) space."""
        P = self.project(P)
        U = self.tangent_project(P, U)
        vals, eigvecs = _eigh_sym(P)
        rotated = jnp.swapaxes(eigvecs, -1, -2) @ U @ eigvecs
        weights = jnp.sqrt(2.0 * (vals[..., :, None] + vals[..., None, :]))
        coordinates = rotated / jnp.maximum(weights, jnp.finfo(P.dtype).tiny)
        return _sym(eigvecs @ coordinates @ jnp.swapaxes(eigvecs, -1, -2))

    def _from_metric_coordinates(self, P: Array, coordinates: Array) -> Array:
        P, coordinates = self._check_shapes(
            ("P", P),
            ("coordinates", coordinates),
        )
        P = self.project(P)
        vals, eigvecs = _eigh_sym(P)
        rotated = jnp.swapaxes(eigvecs, -1, -2) @ _sym(coordinates) @ eigvecs
        weights = jnp.sqrt(2.0 * (vals[..., :, None] + vals[..., None, :]))
        return _sym(eigvecs @ (weights * rotated) @ jnp.swapaxes(eigvecs, -1, -2))

    def transport(self, P: Array, Q: Array, U: Array) -> Array:
        """Isometric vector transport between Bures-Wasserstein tangent spaces.

        This transport preserves the Bures-Wasserstein metric exactly but is
        not claimed to be Levi-Civita parallel transport.
        """
        return self._from_metric_coordinates(Q, self._metric_coordinates(P, U))

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        P = self.project(P)
        E = self.tangent_project(P, egrad)
        return _sym(2.0 * (P @ E + E @ P))

    def lincomb(self, P: Array, *terms: Any) -> Array:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = self._scale_tangent(vec, coeff)
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(P, out)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = _as_sample_shape(sample_shape)
        A = jax.random.normal(key, shape=sample_shape + self.shape)
        A = _sym(A) / jnp.sqrt(float(self.n))
        return _spd_expm(A)

    def random_tangent(
        self,
        key: Array,
        P: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        Z = jax.random.normal(key, shape=jnp.shape(P))
        U = self.tangent_project(P, Z)
        if normalize:
            norm = self.norm(P, U)[..., None, None]
            safe_norm = jnp.where(norm > 0.0, norm, jnp.ones_like(norm))
            U = jnp.where(norm > 0.0, U / safe_norm, U)
        return self._scale_tangent(U, scale)


__all__ = ["SPDLogEuclidean", "SPDAffineInvariant", "SPDBuresWasserstein"]
