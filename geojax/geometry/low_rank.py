"""Fixed-rank matrix and positive-semidefinite geometries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from .base import (
    RetractionGeometryMixin,
    Shape,
    as_sample_shape,
    dtype_margin,
    validate_integer,
    validate_nonnegative,
    validate_positive,
)
from ._numerics import stable_metric_norm, stable_norm, sqrt_nonnegative
from ._numerics import symmetric_part as _sym

Array = Any


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _normalize_matrix_scale(A: Array) -> tuple[Array, Array, Array]:
    """Apply two exact power factors without overflowing their reciprocal."""
    maximum = jnp.max(jnp.abs(jax.lax.stop_gradient(A)), axis=(-2, -1), keepdims=True)
    _, exponent = jnp.frexp(maximum)
    first = -(exponent // 2)
    first_scale = jnp.ldexp(jnp.ones_like(maximum), first)
    second_scale = jnp.ldexp(jnp.ones_like(maximum), -exponent - first)
    normalized = jax.lax.optimization_barrier(A * first_scale) * second_scale
    return normalized, first_scale, second_scale


def _eigh_sym(A: Array) -> tuple[Array, Array]:
    # The backend may symmetrize by adding A+A.T, even if A was already
    # symmetrized safely. Normalizing also avoids backend overflow in its
    # eigensolver on finite matrices near the end of the dtype range.
    normalized, first, second = _normalize_matrix_scale(_sym(A))
    values, vectors = jnp.linalg.eigh(normalized, symmetrize_input=False)
    values = jax.lax.optimization_barrier(values / second[..., 0]) / first[..., 0]
    return values, vectors


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(jnp.asarray(A) * jnp.asarray(B), axis=(-2, -1))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _spectral_support(A: Array, rank: int) -> Array:
    """Top-``rank`` spectral projector, smooth across internal multiplicities.

    Only the gap between the selected and discarded spectra is required.
    Individual eigenvectors are deliberately confined to primal evaluations.
    """
    _, vectors = _eigh_sym(A)
    support = vectors[..., :, -rank:]
    return _sym(support @ _transpose(support))


@_spectral_support.defjvp
def _spectral_support_jvp(rank, primals, tangents):
    (A,), (E,) = primals, tangents
    A, E = _sym(A), _sym(E)
    A, first, second = _normalize_matrix_scale(A)
    E = jax.lax.optimization_barrier(E * first) * second
    support = _spectral_support(A, rank)
    null = jnp.eye(A.shape[-1], dtype=A.dtype) - support
    active_A, null_A = support @ A @ support, null @ A @ null
    rhs = support @ E @ null + null @ E @ support

    def operator(X):
        return (
            active_A @ X @ null
            - support @ X @ null_A
            + null @ X @ active_A
            - null_A @ X @ support
            + support @ X @ support
            + null @ X @ null
        )

    # Implicit differentiation uses ``operator``, never this eigensolver.
    values, vectors = _eigh_sym(jax.lax.stop_gradient(A))
    active = jnp.arange(A.shape[-1]) >= A.shape[-1] - rank
    cross = active[:, None] != active[None, :]
    denominator = jnp.where(cross, jnp.abs(values[..., :, None] - values[..., None, :]), 1.0)

    def solve(_, B):
        rotated = _transpose(vectors) @ B @ vectors
        return vectors @ (rotated / denominator) @ _transpose(vectors)

    derivative = jax.lax.custom_linear_solve(operator, rhs, solve=solve, symmetric=True)
    return support, _sym(derivative)


def _rank_tangent(support: Array, E: Array) -> Array:
    E = _sym(E)
    null = jnp.eye(E.shape[-1], dtype=E.dtype) - support
    return _sym(E - null @ E @ null)


def _rank_sylvester(P: Array, B: Array, rank: int) -> Array:
    """Invert the rank-restricted Sylvester equation with a null-block extension.

    For tangent right-hand sides the solution has zero null-null block and
    is exactly the Moore--Penrose Sylvester solution. The complete operator
    also permits implicit differentiation when the support itself moves.
    """
    P, B = jnp.broadcast_arrays(_sym(P), _sym(B))
    # Scaling both sides leaves the tangent solution unchanged. The unit
    # null block belongs to this normalized equation; it remains a true
    # full-space inverse for implicit AD without an extreme eigenvalue sum.
    P, first, second = _normalize_matrix_scale(P)
    B = jax.lax.optimization_barrier(B * first) * second
    support = _spectral_support(P, rank)
    null = jnp.eye(P.shape[-1], dtype=P.dtype) - support
    active_P = support @ P @ support

    def operator(X):
        return active_P @ X + X @ active_P + null @ X @ null

    values, vectors = _eigh_sym(jax.lax.stop_gradient(P))
    active = jnp.arange(P.shape[-1]) >= P.shape[-1] - rank
    values = jnp.where(active, values, 0.0)
    denominator = values[..., :, None] + values[..., None, :]
    denominator = jnp.where(active[:, None] | active[None, :], denominator, 1.0)

    def solve(_, rhs):
        rotated = _transpose(vectors) @ rhs @ vectors
        return vectors @ (rotated / denominator) @ _transpose(vectors)

    return _sym(jax.lax.custom_linear_solve(operator, B, solve=solve, symmetric=True))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _rank_psd_root(P: Array, rank: int) -> Array:
    values, vectors = _eigh_sym(P)
    values = jnp.sqrt(values[..., -rank:])
    vectors = vectors[..., :, -rank:]
    return _sym((vectors * values[..., None, :]) @ _transpose(vectors))


@_rank_psd_root.defjvp
def _rank_psd_root_jvp(rank, primals, tangents):
    (P,), (E,) = primals, tangents
    root = _rank_psd_root(P, rank)
    support = _spectral_support(P, rank)
    return root, _rank_sylvester(root, _rank_tangent(support, E), rank)


def _rank_inverse_root(P: Array, rank: int) -> Array:
    root = _rank_psd_root(P, rank)
    support = _spectral_support(P, rank)
    null = jnp.eye(P.shape[-1], dtype=P.dtype) - support
    # Match the artificial null-block eigenvalue to the active scale. A
    # unit null eigenvalue can be lost to roundoff after a change of units.
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(root), axis=(-2, -1), keepdims=True))
    return _sym(jnp.linalg.solve(root / scale + null, support) / scale)


def _bures_factors(P: Array, Q: Array, rank: int) -> tuple[Array, Array]:
    def factor(A):
        values, vectors = _eigh_sym(A)
        return vectors[..., :, -rank:] * jnp.sqrt(values[..., None, -rank:])

    factor_p, factor_q = factor(P), factor(Q)
    left, _, right_t = jnp.linalg.svd(_transpose(factor_q) @ factor_p, full_matrices=False)
    return factor_p, factor_q @ (left @ right_t)


def _bures_log_regular(P: Array, Q: Array, rank: int) -> Array:
    # The cross covariance is quadratic in the units of P and Q. Normalize
    # both by the same exact power of two before forming it, then use
    # log_(cP)(cQ) = c log_P(Q). Split powers preserve the full exponent range
    # and the barriers prevent a compiler from reassociating them away.
    maximum_p = jnp.max(jnp.abs(P), axis=(-2, -1), keepdims=True)
    maximum_q = jnp.max(jnp.abs(Q), axis=(-2, -1), keepdims=True)
    maximum = jnp.maximum(maximum_p, maximum_q)
    _, exponent_p = jnp.frexp(jax.lax.stop_gradient(maximum_p))
    _, exponent_q = jnp.frexp(jax.lax.stop_gradient(maximum_q))
    exponent = (exponent_p + exponent_q) // 2
    # Center on the product's scale, not the larger input alone: the latter
    # could underflow a much smaller Q even when P*Q is moderate. Clipping
    # also prevents scaling either representable input out of its range.
    dtype = jnp.finfo(jnp.result_type(P, Q, float))
    smallest_exponent = jnp.minimum(exponent_p, exponent_q)
    lower = jnp.maximum(exponent_p, exponent_q) - dtype.maxexp
    upper = smallest_exponent - jnp.minimum(dtype.minexp + 1, smallest_exponent)
    exponent = jnp.clip(exponent, lower, upper)
    first = -(exponent // 2)
    first_scale = jnp.ldexp(jnp.ones_like(maximum), first)
    second_scale = jnp.ldexp(jnp.ones_like(maximum), -exponent - first)
    P = jax.lax.optimization_barrier(P * first_scale) * second_scale
    Q = jax.lax.optimization_barrier(Q * first_scale) * second_scale
    root_p = _rank_psd_root(P, rank)
    cross = _sym(root_p @ Q @ root_p)
    displacement = Q @ root_p @ _rank_inverse_root(cross, rank) @ root_p - P
    result = displacement + _transpose(displacement)
    undo_second = jnp.ldexp(jnp.ones_like(maximum), exponent + first)
    undo_first = jnp.ldexp(jnp.ones_like(maximum), -first)
    return jax.lax.optimization_barrier(result * undo_second) * undo_first


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _bures_log(P: Array, Q: Array, rank: int) -> Array:
    factor_p, aligned_q = _bures_factors(P, Q, rank)
    horizontal = aligned_q - factor_p
    return _sym(horizontal @ _transpose(factor_p) + factor_p @ _transpose(horizontal))


@_bures_log.defjvp
def _bures_log_jvp(rank, primals, tangents):
    P, Q = primals
    value = _bures_log(P, Q, rank)
    # The whole-matrix formula has the same value off the cut locus and
    # differentiates without choosing eigenvector or Procrustes gauges.
    _, derivative = jax.jvp(lambda p, q: _bures_log_regular(p, q, rank), primals, tangents)
    return value, derivative


def _bures_squared_distance_jvp(P, Q, dP, dQ, rank):
    gradient_p = -_rank_sylvester(P, _bures_log(P, Q, rank), rank)
    gradient_q = -_rank_sylvester(Q, _bures_log(Q, P, rank), rank)
    return _trace_inner(gradient_p, dP) + _trace_inner(gradient_q, dQ)


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _bures_squared_distance(P: Array, Q: Array, rank: int) -> Array:
    factor_p, aligned_q = _bures_factors(P, Q, rank)
    return jnp.sum(jnp.square(aligned_q - factor_p), axis=(-2, -1))


@_bures_squared_distance.defjvp
def _bures_squared_distance_rule(rank, primals, tangents):
    P, Q = primals
    dP, dQ = tangents
    return _bures_squared_distance(P, Q, rank), _bures_squared_distance_jvp(P, Q, dP, dQ, rank)


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _bures_distance(P: Array, Q: Array, rank: int) -> Array:
    factor_p, aligned_q = _bures_factors(P, Q, rank)
    return stable_norm(aligned_q - factor_p, axis=(-2, -1))


@_bures_distance.defjvp
def _bures_distance_rule(rank, primals, tangents):
    P, Q = primals
    dP, dQ = tangents
    value = _bures_distance(P, Q, rank)
    denominator = jnp.where(value > 0.0, 2.0 * value, 1.0)
    derivative = _bures_squared_distance_jvp(P, Q, dP, dQ, rank) / denominator
    return value, jnp.where(value > 0.0, derivative, 0.0)


def _repair_positive_spectrum(values: Array, reference: Array, configured: float) -> Array:
    """Preserve positive spectral values and repair only the closed boundary."""
    values = jnp.asarray(values)
    scale = jnp.max(jnp.abs(values), axis=-1, keepdims=True)
    scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    floor = dtype_margin(reference, configured=configured) * scale
    return jnp.where(values > 0.0, values, floor)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _rank_psd_project(A: Array, rank: int, eps: float) -> Array:
    values, vectors = _eigh_sym(A)
    values = _repair_positive_spectrum(values[..., -rank:], A, eps)
    vectors = vectors[..., :, -rank:]
    return _sym((vectors * values[..., None, :]) @ _transpose(vectors))


@_rank_psd_project.defjvp
def _rank_psd_project_jvp(rank, eps, primals, tangents):
    (A,), (E,) = primals, tangents
    A, E = _sym(A), _sym(E)
    values, vectors = _eigh_sym(jax.lax.stop_gradient(A))
    active_values = values[-rank:]

    def regular_derivative(_):
        def projection(B):
            support = _spectral_support(B, rank)
            return _sym(support @ B @ support)

        return jax.jvp(projection, (A,), (E,))[1]

    def repair_derivative(_):
        rotated = _transpose(vectors) @ E @ vectors
        repaired, d_repaired = jax.jvp(
            lambda spectrum: _repair_positive_spectrum(spectrum, A, eps),
            (active_values,),
            (jnp.diag(rotated)[-rank:],),
        )
        selected = jnp.arange(A.shape[-1]) >= A.shape[-1] - rank
        full_values = jnp.zeros_like(values).at[-rank:].set(repaired)
        difference = values[:, None] - values[None, :]
        same = difference == 0.0
        divided = (full_values[:, None] - full_values[None, :]) / jnp.where(same, 1.0, difference)
        interior = selected & (values > 0.0)
        divided = jnp.where(same, interior[:, None] & interior[None, :], divided)
        derivative = divided * rotated
        # At repeated repaired eigenvalues the common repair floor changes
        # identically on the whole eigenspace; its derivative is scalar.
        derivative = derivative.at[jnp.diag_indices(A.shape[-1])].set(
            jnp.zeros_like(values).at[-rank:].set(d_repaired), unique_indices=True
        )
        return _sym(vectors @ derivative @ _transpose(vectors))

    # Both expressions are linear in E. Selecting their values directly
    # preserves that linearity when vmap batches the custom JVP; batching a
    # conditional around E can insert a stop-gradient into its transpose.
    derivative = jnp.where(
        jnp.all(active_values > 0.0), regular_derivative(None), repair_derivative(None)
    )
    return _rank_psd_project(A, rank, eps), derivative


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _fixed_rank_pinv(A: Array, rank: int) -> Array:
    left, values, right_t = jnp.linalg.svd(A, full_matrices=False)
    return (_transpose(right_t[..., :rank, :]) / values[..., None, :rank]) @ _transpose(
        left[..., :, :rank]
    )


@_fixed_rank_pinv.defjvp
def _fixed_rank_pinv_jvp(rank, primals, tangents):
    (A,), (E,) = primals, tangents
    inverse = _fixed_rank_pinv(A, rank)
    left_normal = E - A @ (inverse @ E)
    right_normal = E - E @ (inverse @ A)
    derivative = -inverse @ E @ inverse
    derivative = derivative + (inverse @ _transpose(inverse)) @ _transpose(left_normal)
    derivative = derivative + (_transpose(right_normal) @ _transpose(inverse)) @ inverse
    return inverse, derivative


def _fixed_rank_tangent_from_pinv(A: Array, inverse: Array, E: Array) -> Array:
    right = E @ (inverse @ A)
    return A @ (inverse @ E) + right - A @ (inverse @ right)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _fixed_rank_project(A: Array, rank: int, eps: float, atol: float) -> Array:
    left, values, right_t = jnp.linalg.svd(A, full_matrices=False)
    values = values[..., :rank]
    floor = dtype_margin(A, configured=eps, atol=atol)
    values = jnp.where(values > 0.0, values, floor)
    return (left[..., :, :rank] * values[..., None, :]) @ right_t[..., :rank, :]


@_fixed_rank_project.defjvp
def _fixed_rank_project_jvp(rank, eps, atol, primals, tangents):
    (A,), (E,) = primals, tangents
    projected = _fixed_rank_project(A, rank, eps, atol)
    if rank == min(A.shape[-2:]):
        values = jnp.linalg.svd(jax.lax.stop_gradient(A), compute_uv=False)
        return projected, jnp.where(values[-1] > 0.0, E, jnp.nan)
    if A.shape[-2] < A.shape[-1]:
        _, derivative = jax.jvp(
            lambda B: _transpose(_fixed_rank_project(_transpose(B), rank, eps, atol)),
            (A,),
            (E,),
        )
        return projected, derivative

    inverse = _fixed_rank_pinv(projected, rank)
    residual = A - projected
    rhs = _fixed_rank_tangent_from_pinv(projected, inverse, E)

    def operator(X):
        # Differentiating normality of A - projected gives this self-adjoint
        # operator. On the normal-normal and selected-selected blocks it is
        # the identity; on cross blocks its eigenvalues are 1 +/- sigma_j /
        # sigma_i, strictly positive at a separated truncation boundary.
        return (
            X
            - _transpose(inverse) @ (_transpose(X) @ residual)
            - residual @ (_transpose(X) @ _transpose(inverse))
        )

    left, values, right_t = jnp.linalg.svd(jax.lax.stop_gradient(A), full_matrices=False)
    active = jnp.arange(values.shape[-1]) < rank
    cross = active[:, None] != active[None, :]
    numerator = jnp.where(active[:, None], values[None, :], values[:, None])
    denominator = jnp.where(active[:, None], values[:, None], values[None, :])
    ratio = jnp.where(cross, numerator / jnp.where(cross, denominator, 1.0), 0.0)

    def solve(_, B):
        rotated = _transpose(left) @ B @ _transpose(right_t)
        solution = (rotated + ratio * _transpose(rotated)) / (1.0 - ratio * ratio)
        return B + left @ (solution - rotated) @ right_t

    derivative = jax.lax.custom_linear_solve(operator, rhs, solve=solve, symmetric=True)
    # A rank-deficient repair has no continuous unique projection into the
    # prescribed stratum. Its original primal selection is preserved.
    derivative = jnp.where(values[rank - 1] > 0.0, derivative, jnp.nan)
    return projected, derivative


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _fixed_rank_tangent(A: Array, E: Array, rank: int) -> Array:
    if rank == min(A.shape[-2:]):
        return jnp.broadcast_to(E, jnp.broadcast_shapes(A.shape, E.shape))
    left, _, right_t = jnp.linalg.svd(A, full_matrices=False)
    left, right_t = left[..., :, :rank], right_t[..., :rank, :]
    right = E @ _transpose(right_t) @ right_t
    return left @ (_transpose(left) @ E) + right - left @ (_transpose(left) @ right)


@_fixed_rank_tangent.defjvp
def _fixed_rank_tangent_jvp(rank, primals, tangents):
    A, E = primals
    value = _fixed_rank_tangent(A, E, rank)
    if rank == min(A.shape[-2:]):
        return value, jnp.broadcast_to(tangents[1], value.shape)
    if A.shape[-2] < A.shape[-1]:
        _, derivative = jax.jvp(
            lambda B, F: _transpose(_fixed_rank_tangent(_transpose(B), _transpose(F), rank)),
            primals,
            tangents,
        )
        return value, derivative

    def projection(B, F):
        selected = _fixed_rank_project(B, rank, 1e-10, 0.0)
        inverse = _fixed_rank_pinv(selected, rank)
        return _fixed_rank_tangent_from_pinv(selected, inverse, F)

    _, derivative = jax.jvp(projection, primals, tangents)
    return value, derivative


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _elliptope_normalize(P: Array, rank: int) -> Array:
    values, vectors = _eigh_sym(P)
    factor = vectors[..., :, -rank:] * sqrt_nonnegative(values[..., -rank:])[..., None, :]
    row_norms = stable_norm(factor, axis=-1, keepdims=True)
    fallback = jnp.eye(rank, dtype=factor.dtype)[jnp.arange(P.shape[-1]) % rank]
    fallback = jnp.broadcast_to(fallback, factor.shape)
    safe_norms = jnp.where(row_norms > 0.0, row_norms, jnp.ones_like(row_norms))
    factor = jnp.where(row_norms > 0.0, factor / safe_norms, fallback)
    return _sym(factor @ _transpose(factor))


@_elliptope_normalize.defjvp
def _elliptope_normalize_jvp(rank, primals, tangents):
    (P,), (E,) = primals, tangents
    value = _elliptope_normalize(P, rank)
    diagonal = jnp.diagonal(P, axis1=-2, axis2=-1)
    diagonal_dot = jnp.diagonal(E, axis1=-2, axis2=-1)
    inverse_norm = 1.0 / jnp.sqrt(diagonal)
    scaled = E * inverse_norm[..., :, None] * inverse_norm[..., None, :]
    ratio = diagonal_dot / diagonal
    derivative = scaled - 0.5 * value * (ratio[..., :, None] + ratio[..., None, :])
    return value, _sym(derivative)


def _parse_matrix_size(size: int | Sequence[int], *, square: bool, name: str) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError(f"{name} size must be a matrix shape.")
    parsed = tuple(validate_integer(value, name=f"{name} size entry", minimum=1) for value in size)
    if len(parsed) != 2 or min(parsed) < 1 or (square and parsed[0] != parsed[1]):
        qualifier = "square " if square else ""
        raise ValueError(f"{name} size must be a {qualifier}matrix shape.")
    return parsed


def _validate_rank(rank: int, maximum: int, name: str) -> int:
    rank = validate_integer(rank, name=f"{name} rank", minimum=1)
    if rank > maximum:
        raise ValueError(f"{name} rank must satisfy 1 <= rank <= {maximum}.")
    return rank


@dataclass(frozen=True, init=False)
class FixedRank(RetractionGeometryMixin):
    """Embedded manifold of real matrices with fixed rank.

    The Frobenius metric is used. ``retr`` is the truncated-SVD retraction;
    compatibility ``exp``, ``log`` and ``dist`` calls are explicitly marked as
    proxies by :meth:`operation_kind`. Projection and tangent derivatives use
    whole singular subspaces, so repeated singular values within either
    spectral band are supported. A tie at the truncation boundary, or repair
    from rank below the requested rank, has no unique smooth projection.
    """

    size: tuple[int, int]
    rank: int
    atol: float
    eps: float

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        rank: int,
        atol: float = 1e-6,
        eps: float = 1e-10,
    ):
        parsed = _parse_matrix_size(size, square=False, name="FixedRank")
        object.__setattr__(self, "size", parsed)
        object.__setattr__(self, "rank", _validate_rank(rank, min(parsed), "FixedRank"))
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="FixedRank atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="FixedRank eps"))

    @property
    def m(self) -> int:
        return self.size[0]

    @property
    def n(self) -> int:
        return self.size[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    @property
    def dim(self) -> int:
        return self.rank * (self.m + self.n - self.rank)

    def _factors(self, X: Array) -> tuple[Array, Array, Array]:
        X = self._check_shape(X, name="X")
        U, singular_values, Vh = jnp.linalg.svd(X, full_matrices=False)
        return U[..., :, : self.rank], singular_values[..., : self.rank], Vh[..., : self.rank, :]

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(X):
            return self._shape_failure(X)
        X = self._check_shape(X, name="X")
        singular_values = jnp.linalg.svd(X, compute_uv=False)
        # Rank is an open condition: ``atol`` controls the zero tail, but it
        # must not remove a small, strictly positive active singular value.
        active = singular_values[..., self.rank - 1] > 0.0
        if self.rank == min(self.size):
            return active
        dtype = jnp.result_type(X, float)
        roundoff = 10.0 * max(self.size) * jnp.finfo(dtype).eps * stable_norm(X, axis=(-2, -1))
        inactive = singular_values[..., self.rank] <= tol + roundoff
        return active & inactive

    def project(self, A: Array) -> Array:
        A = self._check_shape(A, name="A")
        if A.ndim == 2:
            return _fixed_rank_project(A, self.rank, self.eps, self.atol)
        flat = A.reshape((-1,) + self.shape)
        projected = jax.vmap(
            lambda point: _fixed_rank_project(point, self.rank, self.eps, self.atol)
        )(flat)
        return projected.reshape(A.shape)

    def is_tangent(self, X: Array, Z: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(X, Z):
            return self._shape_failure(X)
        X, Z = self._check_shapes(("X", X), ("Z", Z))
        normal = Z - self.tangent_project(X, Z)
        return jnp.linalg.norm(normal, axis=(-2, -1)) <= tol

    def tangent_project(self, X: Array, Z: Array) -> Array:
        X, Z = self._check_shapes(("X", X), ("Z", Z))
        if X.ndim == 2 and Z.ndim == 2:
            return _fixed_rank_tangent(X, Z, self.rank)
        X, Z = jnp.broadcast_arrays(X, Z)
        flat_X, flat_Z = X.reshape((-1,) + self.shape), Z.reshape((-1,) + self.shape)
        projected = jax.vmap(lambda p, u: _fixed_rank_tangent(p, u, self.rank))(flat_X, flat_Z)
        return projected.reshape(X.shape)

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        _, U, V = self._check_shapes(("X", X), ("U", U), ("V", V))
        return _trace_inner(U, V)

    def norm(self, X: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(X, normalized, normalized),
            axis=(-2, -1),
        )

    def retr(self, X: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.project(jnp.asarray(X) + self._scale_tangent(self.tangent_project(X, U), t))

    def invretr(self, X: Array, Y: Array) -> Array:
        return self.tangent_project(X, jnp.asarray(Y) - jnp.asarray(X))

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        return self.tangent_project(X, egrad)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        key_u, key_v, key_s = jax.random.split(key, 3)
        U, _ = jnp.linalg.qr(
            jax.random.normal(key_u, shape=sample_shape + (self.m, self.rank)), mode="reduced"
        )
        V, _ = jnp.linalg.qr(
            jax.random.normal(key_v, shape=sample_shape + (self.n, self.rank)), mode="reduced"
        )
        singular_values = jnp.exp(0.2 * jax.random.normal(key_s, shape=sample_shape + (self.rank,)))
        return (U * singular_values[..., None, :]) @ _transpose(V)

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
class _RankKPSDBase(RetractionGeometryMixin):
    """Embedded fixed-rank PSD stratum with the Frobenius metric.

    Projection derivatives act on the complete selected subspace. Internal
    repeated eigenvalues are regular; a tie across the truncation boundary
    is a nonunique projection and has no differentiability guarantee.
    """

    size: tuple[int, int]
    rank: int
    atol: float
    eps: float

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        rank: int,
        atol: float = 1e-6,
        eps: float = 1e-10,
    ):
        parsed = _parse_matrix_size(size, square=True, name=type(self).__name__)
        object.__setattr__(self, "size", parsed)
        object.__setattr__(self, "rank", _validate_rank(rank, parsed[0], type(self).__name__))
        object.__setattr__(
            self,
            "atol",
            validate_nonnegative(atol, name=f"{type(self).__name__} atol"),
        )
        object.__setattr__(
            self,
            "eps",
            validate_positive(eps, name=f"{type(self).__name__} eps"),
        )

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    @property
    def dim(self) -> int:
        return self.n * self.rank - self.rank * (self.rank - 1) // 2

    def _eigen_factors(self, P: Array) -> tuple[Array, Array]:
        P = self._check_shape(P, name="P")
        eigenvalues, eigenvectors = _eigh_sym(P)
        return eigenvalues[..., -self.rank :], eigenvectors[..., :, -self.rank :]

    def _support_projector(self, P: Array) -> Array:
        return _spectral_support(P, self.rank)

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        P = jnp.asarray(P)
        if not self._shape_matches(P):
            return self._shape_failure(P)
        symmetric = jnp.linalg.norm(P - _transpose(P), axis=(-2, -1)) <= tol
        eigenvalues, _ = _eigh_sym(P)
        # The positive support is an open condition; preserve the complete
        # fixed-rank PSD stratum rather than imposing an absolute eigenvalue
        # floor through the membership tolerance.
        positive = jnp.min(eigenvalues[..., -self.rank :], axis=-1) > 0.0
        if self.rank == self.n:
            return symmetric & positive
        dtype = jnp.result_type(P, float)
        roundoff = 10.0 * self.n * jnp.finfo(dtype).eps * stable_norm(P, axis=(-2, -1))
        zero = jnp.max(jnp.abs(eigenvalues[..., : -self.rank]), axis=-1) <= tol + roundoff
        return symmetric & positive & zero

    def project(self, A: Array) -> Array:
        A = self._check_shape(A, name="A")
        if A.ndim == 2:
            return _rank_psd_project(A, self.rank, self.eps)
        flat = A.reshape((-1,) + self.shape)
        projected = jax.vmap(lambda point: _rank_psd_project(point, self.rank, self.eps))(flat)
        return projected.reshape(A.shape)

    def normalize(self, A: Array) -> Array:
        """Project through the most-derived manifold constraint."""
        return self.project(A)

    def tangent_project(self, P: Array, Z: Array) -> Array:
        P, Z = self._check_shapes(("P", P), ("Z", Z))
        support = self._support_projector(P)
        return _rank_tangent(support, Z)

    def projection(self, P: Array, Z: Array) -> Array:
        """Project through the most-derived tangent constraint."""
        return self.tangent_project(P, Z)

    def proj(self, P: Array, Z: Array) -> Array:
        """Project through the most-derived tangent constraint."""
        return self.tangent_project(P, Z)

    def to_tangent(self, P: Array, Z: Array) -> Array:
        """Project through the most-derived tangent constraint."""
        return self.tangent_project(P, Z)

    def is_tangent(self, P: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(P, U):
            return self._shape_failure(P)
        P, U = self._check_shapes(("P", P), ("U", U))
        residual = U - self.tangent_project(P, U)
        dtype = jnp.result_type(U, float)
        roundoff = 10.0 * self.n * jnp.finfo(dtype).eps * stable_norm(U, axis=(-2, -1))
        return jnp.linalg.norm(residual, axis=(-2, -1)) <= tol + roundoff

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        _, U, V = self._check_shapes(("P", P), ("U", U), ("V", V))
        return _trace_inner(U, V)

    def norm(self, P: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(P, normalized, normalized),
            axis=(-2, -1),
        )

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.project(jnp.asarray(P) + self._scale_tangent(self.tangent_project(P, U), t))

    def invretr(self, P: Array, Q: Array) -> Array:
        return self.tangent_project(P, jnp.asarray(Q) - jnp.asarray(P))

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        return self.tangent_project(P, egrad)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        factor = jax.random.normal(key, shape=sample_shape + (self.n, self.rank))
        return self.project(factor @ _transpose(factor))

    def random_tangent(
        self,
        key: Array,
        P: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        tangent = self.tangent_project(P, jax.random.normal(key, shape=jnp.shape(P)))
        if normalize:
            length = self.norm(P, tangent)[..., None, None]
            safe_length = jnp.where(length > 0.0, length, jnp.ones_like(length))
            tangent = jnp.where(length > 0.0, tangent / safe_length, tangent)
        return self._scale_tangent(tangent, scale)


class RankKPSD(_RankKPSDBase):
    """Fixed-rank positive-semidefinite matrices with embedded metric."""


class RankKPSDBuresWasserstein(_RankKPSDBase):
    """Fixed-rank PSD matrices with their Bures--Wasserstein quotient metric.

    Derivatives use invariant matrix equations, including at repeated positive
    and zero eigenvalues. Logarithms and squared distances are smooth when
    the factors' cross product is invertible. At a singular cross product
    (the quotient cut locus), ``log`` returns one Procrustes-selected tangent;
    its derivative and the distance gradient need not exist.

    ``exp`` follows the entire straight horizontal factor path and rejects
    steps that encounter rank loss, including paths that leave and reenter
    the regular stratum. Its numerical rank certificate resolves separation
    only to roundoff precision; near-boundary paths can be rejected.
    """

    exp_is_exact = True
    log_is_exact = True
    dist_is_exact = True
    transport_is_isometric = False
    transport_is_parallel = False

    def _factor(self, P: Array) -> Array:
        eigenvalues, eigenvectors = self._eigen_factors(P)
        return eigenvectors * sqrt_nonnegative(eigenvalues)[..., None, :]

    def sylvester(self, P: Array, U: Array) -> Array:
        P, U = self._check_shapes(("P", P), ("U", U))
        return _rank_sylvester(P, self.tangent_project(P, U), self.rank)

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        return 0.5 * _trace_inner(self.sylvester(P, U), self.tangent_project(P, V))

    def exp(self, P: Array, U: Array) -> Array:
        P = self.project(P)
        U = self.tangent_project(P, U)
        generator = self.sylvester(P, U)
        identity = jnp.eye(self.n, dtype=P.dtype)
        factor = identity + generator
        result = _sym(factor @ P @ factor)
        # Write the initial factor as V R with V orthonormal. A path loses
        # rank at t in (0,1] precisely when an eigenvector of V.T L V with
        # eigenvalue lambda <= -1 also satisfies L V v = lambda V v.
        # Testing the full residual handles repeated compressed eigenvalues
        # without selecting a basis inside their eigenspaces.
        _, vectors = _eigh_sym(jax.lax.stop_gradient(P))
        basis = vectors[..., :, -self.rank :]
        horizontal = jax.lax.stop_gradient(generator) @ basis
        compressed = _sym(_transpose(basis) @ horizontal)
        candidates = jnp.linalg.eigvalsh(compressed)
        dtype = jnp.result_type(P, float)
        scale = jnp.maximum(stable_norm(horizontal, axis=(-2, -1)), 1.0)
        roundoff = 32.0 * self.n * jnp.finfo(dtype).eps * scale
        possible_crossing = (candidates < 0.0) & (candidates <= -1.0 + roundoff[..., None])

        def check_crossings(_):
            residuals = horizontal[..., None, :, :] - (
                candidates[..., :, None, None] * basis[..., None, :, :]
            )
            residual_values = jnp.linalg.svd(residuals, compute_uv=False)
            rank_loss = possible_crossing & (
                jnp.min(residual_values, axis=-1) <= roundoff[..., None]
            )
            return ~jnp.any(rank_loss, axis=-1)

        # Most local optimizer steps have no candidate crossing. The cheap
        # compressed-spectrum certificate then avoids k additional SVDs.
        valid = jax.lax.cond(
            jnp.any(possible_crossing),
            check_crossings,
            lambda _: jnp.ones(candidates.shape[:-1], dtype=bool),
            operand=None,
        )
        return jnp.where(valid[..., None, None], result, jnp.full_like(result, jnp.nan))

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(P, self._scale_tangent(U, t))

    def log(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        return _bures_log(P, Q, self.rank)

    def invretr(self, P: Array, Q: Array) -> Array:
        return self.log(P, Q)

    def dist(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        return _bures_distance(P, Q, self.rank)

    def squared_dist(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        return _bures_squared_distance(P, Q, self.rank)

    def transport(self, P: Array, Q: Array, U: Array) -> Array:
        return self.tangent_project(Q, U)

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        P = _sym(P)
        E = _sym(egrad)
        return self.tangent_project(P, 2.0 * (P @ E + E @ P))


class Elliptope(_RankKPSDBase):
    """Rank-``rank`` PSD matrices with unit diagonal.

    Diagonal normalization has a basis-independent derivative when all rows
    have positive norm. Zero-row repairs preserve a chosen factor fallback;
    that noncontinuous repair has no ordinary derivative.
    """

    @property
    def dim(self) -> int:
        return self.n * (self.rank - 1) - self.rank * (self.rank - 1) // 2

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(P):
            return self._shape_failure(P)
        return super().belongs(P, atol=tol) & jnp.all(
            jnp.abs(jnp.diagonal(P, axis1=-2, axis2=-1) - 1.0) <= tol, axis=-1
        )

    def project(self, A: Array) -> Array:
        base = super().project(A)
        return _elliptope_normalize(base, self.rank)

    def tangent_project(self, P: Array, Z: Array) -> Array:
        rank_projected = super().tangent_project(P, Z)
        identity = jnp.eye(self.n, dtype=jnp.asarray(P).dtype)
        columns = []
        for index in range(self.n):
            diagonal_matrix = identity * identity[index][..., None, :]
            projected = super().tangent_project(P, diagonal_matrix)
            columns.append(jnp.diagonal(projected, axis1=-2, axis2=-1))
        operator = jnp.stack(columns, axis=-1)
        target = jnp.diagonal(rank_projected, axis1=-2, axis2=-1)
        multiplier = jnp.linalg.pinv(operator, rtol=self.eps) @ target[..., None]
        correction = identity * multiplier[..., 0][..., None, :]
        return super().tangent_project(P, _sym(Z) - correction)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        factor = jax.random.normal(key, shape=sample_shape + (self.n, self.rank))
        row_norms = stable_norm(factor, axis=-1, keepdims=True)
        fallback = jnp.eye(self.rank, dtype=factor.dtype)[jnp.arange(self.n) % self.rank]
        fallback = jnp.broadcast_to(fallback, factor.shape)
        factor = jnp.where(
            row_norms > 0.0,
            factor / jnp.where(row_norms > 0.0, row_norms, 1.0),
            fallback,
        )
        return _sym(factor @ _transpose(factor))


class Spectrahedron(_RankKPSDBase):
    """Rank-``rank`` PSD matrices with unit trace."""

    @property
    def dim(self) -> int:
        return super().dim - 1

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(P):
            return self._shape_failure(P)
        return super().belongs(P, atol=tol) & (
            jnp.abs(jnp.trace(P, axis1=-2, axis2=-1) - 1.0) <= tol
        )

    def project(self, A: Array) -> Array:
        base = super().project(A)
        return base / jnp.trace(base, axis1=-2, axis2=-1)[..., None, None]

    def tangent_project(self, P: Array, Z: Array) -> Array:
        projected = super().tangent_project(P, Z)
        support = self._support_projector(P)
        coefficient = jnp.trace(projected, axis1=-2, axis2=-1) / self.rank
        return projected - coefficient[..., None, None] * support


__all__ = [
    "FixedRank",
    "RankKPSD",
    "RankKPSDBuresWasserstein",
    "Elliptope",
    "Spectrahedron",
]
