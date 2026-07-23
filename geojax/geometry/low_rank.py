"""Fixed-rank matrix and positive-semidefinite geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from .base import RetractionGeometryMixin, Shape, as_sample_shape

Array = Any


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _sym(A: Array) -> Array:
    return 0.5 * (jnp.asarray(A) + _transpose(jnp.asarray(A)))


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(jnp.asarray(A) * jnp.asarray(B), axis=(-2, -1))


def _parse_matrix_size(size: int | Sequence[int], *, square: bool, name: str) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError(f"{name} size must be a matrix shape.")
    parsed = tuple(int(value) for value in size)
    if len(parsed) != 2 or min(parsed) < 1 or (square and parsed[0] != parsed[1]):
        qualifier = "square " if square else ""
        raise ValueError(f"{name} size must be a {qualifier}matrix shape.")
    return parsed


def _validate_rank(rank: int, maximum: int, name: str) -> int:
    rank = int(rank)
    if rank < 1 or rank > maximum:
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
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

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
        U, singular_values, Vh = jnp.linalg.svd(jnp.asarray(X), full_matrices=False)
        return U[..., :, : self.rank], singular_values[..., : self.rank], Vh[..., : self.rank, :]

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        singular_values = jnp.linalg.svd(jnp.asarray(X), compute_uv=False)
        active = singular_values[..., self.rank - 1] > tol
        if self.rank == min(self.size):
            return active
        inactive = singular_values[..., self.rank] <= tol
        return active & inactive

    def project(self, A: Array) -> Array:
        U, singular_values, Vh = self._factors(A)
        singular_values = jnp.maximum(singular_values, self.eps)
        return (U * singular_values[..., None, :]) @ Vh

    normalize = project

    def is_tangent(self, X: Array, Z: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        U, _, Vh = self._factors(X)
        V = _transpose(Vh)
        normal = Z - U @ (_transpose(U) @ Z) - Z @ V @ _transpose(V)
        normal = normal + U @ (_transpose(U) @ Z @ V) @ _transpose(V)
        return jnp.linalg.norm(normal, axis=(-2, -1)) <= tol

    def tangent_project(self, X: Array, Z: Array) -> Array:
        U, _, Vh = self._factors(X)
        V = _transpose(Vh)
        projected = U @ (_transpose(U) @ Z) + Z @ V @ _transpose(V)
        return projected - U @ (_transpose(U) @ Z @ V) @ _transpose(V)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        del X
        return _trace_inner(U, V)

    def norm(self, X: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(X, U, U), 0.0))

    def retr(self, X: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.project(jnp.asarray(X) + t * self.tangent_project(X, U))

    def invretr(self, X: Array, Y: Array) -> Array:
        return self.tangent_project(X, jnp.asarray(Y) - jnp.asarray(X))

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        return self.tangent_project(X, egrad)

    egrad2rgrad = egrad_to_rgrad

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
            tangent = jnp.where(length > self.eps, tangent / length, tangent)
        return scale * tangent


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
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

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
        eigenvalues, eigenvectors = jnp.linalg.eigh(_sym(P))
        return eigenvalues[..., -self.rank :], eigenvectors[..., :, -self.rank :]

    def _support_projector(self, P: Array) -> Array:
        _, eigenvectors = self._eigen_factors(P)
        return eigenvectors @ _transpose(eigenvectors)

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        P = jnp.asarray(P)
        symmetric = jnp.linalg.norm(P - _transpose(P), axis=(-2, -1)) <= tol
        eigenvalues = jnp.linalg.eigvalsh(_sym(P))
        positive = jnp.min(eigenvalues[..., -self.rank :], axis=-1) > tol
        if self.rank == self.n:
            return symmetric & positive
        zero = jnp.max(jnp.abs(eigenvalues[..., : -self.rank]), axis=-1) <= tol
        return symmetric & positive & zero

    def project(self, A: Array) -> Array:
        eigenvalues, eigenvectors = self._eigen_factors(A)
        eigenvalues = jnp.maximum(eigenvalues, self.eps)
        return (eigenvectors * eigenvalues[..., None, :]) @ _transpose(eigenvectors)

    normalize = project

    def tangent_project(self, P: Array, Z: Array) -> Array:
        support = self._support_projector(P)
        Z = _sym(Z)
        return _sym(support @ Z + Z @ support - support @ Z @ support)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def is_tangent(self, P: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        U = jnp.asarray(U)
        residual = U - self.tangent_project(P, U)
        dtype = jnp.result_type(U, float)
        roundoff = (
            10.0
            * self.n
            * jnp.finfo(dtype).eps
            * jnp.maximum(jnp.linalg.norm(U, axis=(-2, -1)), 1.0)
        )
        return jnp.linalg.norm(residual, axis=(-2, -1)) <= tol + roundoff

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        del P
        return _trace_inner(U, V)

    def norm(self, P: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(P, U, U), 0.0))

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.project(jnp.asarray(P) + t * self.tangent_project(P, U))

    def invretr(self, P: Array, Q: Array) -> Array:
        return self.tangent_project(P, jnp.asarray(Q) - jnp.asarray(P))

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        return self.tangent_project(P, egrad)

    egrad2rgrad = egrad_to_rgrad

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
            tangent = jnp.where(length > self.eps, tangent / length, tangent)
        return scale * tangent


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
        return eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, self.eps))[..., None, :]

    def sylvester(self, P: Array, U: Array) -> Array:
        eigenvalues, eigenvectors = jnp.linalg.eigh(_sym(P))
        rotated = _transpose(eigenvectors) @ self.tangent_project(P, U) @ eigenvectors
        denominator = eigenvalues[..., :, None] + eigenvalues[..., None, :]
        solution = jnp.where(denominator > self.eps, rotated / denominator, 0.0)
        return _sym(eigenvectors @ solution @ _transpose(eigenvectors))

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        return 0.5 * _trace_inner(self.sylvester(P, U), self.tangent_project(P, V))

    def exp(self, P: Array, U: Array) -> Array:
        P = _sym(P)
        generator = self.sylvester(P, U)
        identity = jnp.eye(self.n, dtype=P.dtype)
        factor = identity + generator
        return _sym(factor @ P @ factor)

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(P, t * U)

    def log(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        factor_p = self._factor(P)
        factor_q = self._factor(Q)
        left, _, right_t = jnp.linalg.svd(_transpose(factor_q) @ factor_p)
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
        left, _, right_t = jnp.linalg.svd(_transpose(factor_q) @ factor_p)
        difference = factor_q @ (left @ right_t) - factor_p
        return jnp.linalg.norm(difference, axis=(-2, -1))

    def transport(self, P: Array, Q: Array, U: Array) -> Array:
        return self.tangent_project(Q, U)

    transp = transport

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        P = _sym(P)
        E = _sym(egrad)
        return self.tangent_project(P, 2.0 * (P @ E + E @ P))

    egrad2rgrad = egrad_to_rgrad


class Elliptope(_RankKPSDBase):
    """Rank-``rank`` PSD matrices with unit diagonal."""

    @property
    def dim(self) -> int:
        return self.n * (self.rank - 1) - self.rank * (self.rank - 1) // 2

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        return super().belongs(P, atol=tol) & jnp.all(
            jnp.abs(jnp.diagonal(P, axis1=-2, axis2=-1) - 1.0) <= tol, axis=-1
        )

    def project(self, A: Array) -> Array:
        base = super().project(A)
        eigenvalues, eigenvectors = self._eigen_factors(base)
        factor = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, self.eps))[..., None, :]
        row_norms = jnp.linalg.norm(factor, axis=-1, keepdims=True)
        factor = factor / jnp.maximum(row_norms, self.eps)
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
        factor /= jnp.maximum(jnp.linalg.norm(factor, axis=-1, keepdims=True), self.eps)
        return _sym(factor @ _transpose(factor))


class Spectrahedron(_RankKPSDBase):
    """Rank-``rank`` PSD matrices with unit trace."""

    @property
    def dim(self) -> int:
        return super().dim - 1

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
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
