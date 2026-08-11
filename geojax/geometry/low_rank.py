"""Fixed-rank matrix and positive-semidefinite geometries."""

from __future__ import annotations

from dataclasses import dataclass
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

Array = Any


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _sym(A: Array) -> Array:
    return 0.5 * (jnp.asarray(A) + _transpose(jnp.asarray(A)))


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(jnp.asarray(A) * jnp.asarray(B), axis=(-2, -1))


def _repair_positive_spectrum(values: Array, reference: Array, configured: float) -> Array:
    """Preserve positive spectral values and repair only the closed boundary."""
    values = jnp.asarray(values)
    scale = jnp.max(jnp.abs(values), axis=-1, keepdims=True)
    scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    floor = dtype_margin(reference, configured=configured) * scale
    return jnp.where(values > 0.0, values, floor)


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
    proxies by :meth:`operation_kind`.
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
        U, singular_values, Vh = self._factors(A)
        floor = dtype_margin(A, configured=self.eps, atol=self.atol)
        singular_values = jnp.where(singular_values > 0.0, singular_values, floor)
        return (U * singular_values[..., None, :]) @ Vh

    def is_tangent(self, X: Array, Z: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(X, Z):
            return self._shape_failure(X)
        X, Z = self._check_shapes(("X", X), ("Z", Z))
        U, _, Vh = self._factors(X)
        V = _transpose(Vh)
        normal = Z - U @ (_transpose(U) @ Z) - Z @ V @ _transpose(V)
        normal = normal + U @ (_transpose(U) @ Z @ V) @ _transpose(V)
        return jnp.linalg.norm(normal, axis=(-2, -1)) <= tol

    def tangent_project(self, X: Array, Z: Array) -> Array:
        X, Z = self._check_shapes(("X", X), ("Z", Z))
        U, _, Vh = self._factors(X)
        V = _transpose(Vh)
        projected = U @ (_transpose(U) @ Z) + Z @ V @ _transpose(V)
        return projected - U @ (_transpose(U) @ Z @ V) @ _transpose(V)

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
    """Embedded fixed-rank PSD stratum with the Frobenius metric."""

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
        eigenvalues, eigenvectors = jnp.linalg.eigh(_sym(P))
        return eigenvalues[..., -self.rank :], eigenvectors[..., :, -self.rank :]

    def _support_projector(self, P: Array) -> Array:
        _, eigenvectors = self._eigen_factors(P)
        return eigenvectors @ _transpose(eigenvectors)

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        P = jnp.asarray(P)
        if not self._shape_matches(P):
            return self._shape_failure(P)
        symmetric = jnp.linalg.norm(P - _transpose(P), axis=(-2, -1)) <= tol
        eigenvalues = jnp.linalg.eigvalsh(_sym(P))
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
        eigenvalues, eigenvectors = self._eigen_factors(A)
        eigenvalues = _repair_positive_spectrum(eigenvalues, A, self.eps)
        return (eigenvectors * eigenvalues[..., None, :]) @ _transpose(eigenvectors)

    def normalize(self, A: Array) -> Array:
        """Project through the most-derived manifold constraint."""
        return self.project(A)

    def tangent_project(self, P: Array, Z: Array) -> Array:
        P, Z = self._check_shapes(("P", P), ("Z", Z))
        support = self._support_projector(P)
        Z = _sym(Z)
        return _sym(support @ Z + Z @ support - support @ Z @ support)

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
    """Fixed-rank PSD matrices with their Bures--Wasserstein quotient metric."""

    exp_is_exact = True
    log_is_exact = True
    dist_is_exact = True
    transport_is_isometric = False
    transport_is_parallel = False

    def _factor(self, P: Array) -> Array:
        eigenvalues, eigenvectors = self._eigen_factors(P)
        return eigenvectors * sqrt_nonnegative(eigenvalues)[..., None, :]

    def sylvester(self, P: Array, U: Array) -> Array:
        eigenvalues, eigenvectors = jnp.linalg.eigh(_sym(P))
        rotated = _transpose(eigenvectors) @ self.tangent_project(P, U) @ eigenvectors
        # Eigensolvers can return tiny signed values on the null space. Build
        # the quotient operator from the known rank-k support instead: only
        # support-support and support-null blocks belong to the tangent space.
        support_values = eigenvalues[..., -self.rank :]
        spectrum = jnp.zeros_like(eigenvalues)
        spectrum = spectrum.at[..., -self.rank :].set(support_values)
        support = jnp.arange(self.n) >= self.n - self.rank
        active = support[:, None] | support[None, :]
        denominator = spectrum[..., :, None] + spectrum[..., None, :]
        safe_denominator = jnp.where(active & (denominator > 0.0), denominator, 1.0)
        solution = jnp.where(active, rotated / safe_denominator, 0.0)
        valid = jnp.min(support_values, axis=-1) > 0.0
        solution = jnp.where(
            valid[..., None, None],
            solution,
            jnp.full_like(solution, jnp.nan),
        )
        return _sym(eigenvectors @ solution @ _transpose(eigenvectors))

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        return 0.5 * _trace_inner(self.sylvester(P, U), self.tangent_project(P, V))

    def exp(self, P: Array, U: Array) -> Array:
        P = self.project(P)
        U = self.tangent_project(P, U)
        generator = self.sylvester(P, U)
        identity = jnp.eye(self.n, dtype=P.dtype)
        factor = identity + generator
        result = _sym(factor @ P @ factor)
        base_factor = self._factor(P)
        lifted_factor = factor @ base_factor
        # A positive-semidefinite overlap certifies that the straight
        # horizontal factor path remains full rank before the endpoint.  A
        # separate endpoint check admits legitimate orthogonal-support cases,
        # for which the overlap is singular but the endpoint is regular.
        overlap = _sym(_transpose(base_factor) @ lifted_factor)
        overlap_values = jnp.linalg.eigvalsh(overlap)
        overlap_scale = jnp.max(jnp.abs(overlap_values), axis=-1)
        dtype = jnp.result_type(P, float)
        roundoff = 32.0 * self.rank * jnp.finfo(dtype).eps * overlap_scale
        endpoint_values = jnp.linalg.svd(lifted_factor, compute_uv=False)
        valid = (jnp.min(overlap_values, axis=-1) >= -roundoff) & (
            jnp.min(endpoint_values, axis=-1) > 0.0
        )
        return jnp.where(valid[..., None, None], result, jnp.full_like(result, jnp.nan))

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(P, self._scale_tangent(U, t))

    def log(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        factor_p = self._factor(P)
        factor_q = self._factor(Q)
        left, _, right_t = jnp.linalg.svd(
            _transpose(factor_q) @ factor_p,
            full_matrices=False,
        )
        alignment = left @ right_t
        horizontal = factor_q @ alignment - factor_p
        return _sym(horizontal @ _transpose(factor_p) + factor_p @ _transpose(horizontal))

    def invretr(self, P: Array, Q: Array) -> Array:
        return self.log(P, Q)

    def dist(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        factor_p = self._factor(P)
        factor_q = self._factor(Q)
        left, _, right_t = jnp.linalg.svd(
            _transpose(factor_q) @ factor_p,
            full_matrices=False,
        )
        difference = factor_q @ (left @ right_t) - factor_p
        return stable_norm(difference, axis=(-2, -1))

    def transport(self, P: Array, Q: Array, U: Array) -> Array:
        return self.tangent_project(Q, U)

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        P = _sym(P)
        E = _sym(egrad)
        return self.tangent_project(P, 2.0 * (P @ E + E @ P))


class Elliptope(_RankKPSDBase):
    """Rank-``rank`` PSD matrices with unit diagonal."""

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
        eigenvalues, eigenvectors = self._eigen_factors(base)
        factor = eigenvectors * sqrt_nonnegative(eigenvalues)[..., None, :]
        row_norms = stable_norm(factor, axis=-1, keepdims=True)
        fallback = jnp.eye(self.rank, dtype=factor.dtype)[jnp.arange(self.n) % self.rank]
        fallback = jnp.broadcast_to(fallback, factor.shape)
        safe_norms = jnp.where(row_norms > 0.0, row_norms, jnp.ones_like(row_norms))
        factor = jnp.where(row_norms > 0.0, factor / safe_norms, fallback)
        return _sym(factor @ _transpose(factor))

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
