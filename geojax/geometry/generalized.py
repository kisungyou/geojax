"""Generalized orthogonality geometries under an SPD metric matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from .base import GeometryMixin, Shape, as_sample_shape
from .grassmann import Grassmann
from .stiefel import StiefelEuclidean

Array = Any


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _sym(A: Array) -> Array:
    return 0.5 * (jnp.asarray(A) + _transpose(jnp.asarray(A)))


def _parse_size(size: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError(f"{name} size must be a pair (ambient_dim, rank).")
    parsed = tuple(int(value) for value in size)
    if len(parsed) != 2 or parsed[0] < 1 or parsed[1] < 1 or parsed[1] > parsed[0]:
        raise ValueError(f"{name} size must satisfy ambient_dim >= rank >= 1.")
    return parsed


def _metric_factors(
    metric: Array, n: int, eps: float, name: str
) -> tuple[Array, Array, Array, Array]:
    metric = _sym(metric)
    if metric.shape != (n, n):
        raise ValueError(f"{name} metric must have shape ({n}, {n}).")
    eigenvalues, eigenvectors = jnp.linalg.eigh(metric)
    if not bool(jnp.all(eigenvalues > eps)):
        raise ValueError(f"{name} metric must be symmetric positive definite.")
    sqrt = (eigenvectors * jnp.sqrt(eigenvalues)[None, :]) @ eigenvectors.T
    invsqrt = (eigenvectors * (1.0 / jnp.sqrt(eigenvalues))[None, :]) @ eigenvectors.T
    inverse = (eigenvectors * (1.0 / eigenvalues)[None, :]) @ eigenvectors.T
    return metric, sqrt, invsqrt, inverse


@dataclass(frozen=True, init=False)
class GeneralizedStiefel(GeometryMixin):
    """Frames satisfying ``X.T @ metric @ X = I``.

    The metric ``trace(U.T @ metric @ V)`` is the pullback of the embedded
    Euclidean Stiefel metric under ``X -> metric^(1/2) X``.
    """

    size: tuple[int, int]
    metric: Array
    atol: float
    eps: float
    _sqrt_metric: Array
    _invsqrt_metric: Array
    _inverse_metric: Array
    _log_maxiter: int
    _log_tol: float

    transport_is_parallel = False

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        metric: Array,
        atol: float = 1e-6,
        eps: float = 1e-10,
        log_maxiter: int = 32,
        log_tol: float = 1e-9,
    ):
        parsed = _parse_size(size, "GeneralizedStiefel")
        B, sqrt, invsqrt, inverse = _metric_factors(metric, parsed[0], eps, "GeneralizedStiefel")
        object.__setattr__(self, "size", parsed)
        object.__setattr__(self, "metric", B)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))
        object.__setattr__(self, "_sqrt_metric", sqrt)
        object.__setattr__(self, "_invsqrt_metric", invsqrt)
        object.__setattr__(self, "_inverse_metric", inverse)
        object.__setattr__(self, "_log_maxiter", int(log_maxiter))
        object.__setattr__(self, "_log_tol", float(log_tol))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def k(self) -> int:
        return self.size[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    @property
    def dim(self) -> int:
        return self.n * self.k - self.k * (self.k + 1) // 2

    @property
    def _ordinary(self) -> StiefelEuclidean:
        return StiefelEuclidean(
            self.size,
            atol=self.atol,
            eps=self.eps,
            log_maxiter=self._log_maxiter,
            log_tol=self._log_tol,
        )

    def _forward(self, X: Array) -> Array:
        return self._sqrt_metric @ jnp.asarray(X)

    def _inverse(self, X: Array) -> Array:
        return self._invsqrt_metric @ jnp.asarray(X)

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        gram = _transpose(X) @ self.metric @ X
        return jnp.linalg.norm(gram - jnp.eye(self.k, dtype=gram.dtype), axis=(-2, -1)) <= tol

    def project(self, A: Array) -> Array:
        return self._inverse(self._ordinary.project(self._forward(A)))

    normalize = project

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        constraint = _transpose(X) @ self.metric @ U
        return jnp.linalg.norm(constraint + _transpose(constraint), axis=(-2, -1)) <= tol

    def tangent_project(self, X: Array, U: Array) -> Array:
        return jnp.asarray(U) - X @ _sym(_transpose(X) @ self.metric @ U)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        del X
        return jnp.sum(jnp.asarray(U) * (self.metric @ jnp.asarray(V)), axis=(-2, -1))

    def norm(self, X: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(X, U, U), 0.0))

    def exp(self, X: Array, U: Array) -> Array:
        return self._inverse(self._ordinary.exp(self._forward(X), self._forward(U)))

    def log(self, X: Array, Y: Array) -> Array:
        return self._inverse(self._ordinary.log(self._forward(X), self._forward(Y)))

    def dist(self, X: Array, Y: Array) -> Array:
        return self._ordinary.dist(self._forward(X), self._forward(Y))

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        transported = self._ordinary.transport(self._forward(X), self._forward(Y), self._forward(U))
        return self._inverse(transported)

    transp = transport

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        ambient = self._inverse_metric @ jnp.asarray(egrad)
        return ambient - X @ _sym(_transpose(X) @ jnp.asarray(egrad))

    egrad2rgrad = egrad_to_rgrad

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        return self._inverse(
            self._ordinary.random_point(key, sample_shape=as_sample_shape(sample_shape))
        )

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
class GeneralizedGrassmann(GeometryMixin):
    """Generalized Grassmann geometry for ``metric``-orthonormal subspaces."""

    size: tuple[int, int]
    metric: Array
    atol: float
    eps: float
    _sqrt_metric: Array
    _invsqrt_metric: Array
    _inverse_metric: Array

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        metric: Array,
        atol: float = 1e-6,
        eps: float = 1e-10,
    ):
        parsed = _parse_size(size, "GeneralizedGrassmann")
        B, sqrt, invsqrt, inverse = _metric_factors(metric, parsed[0], eps, "GeneralizedGrassmann")
        object.__setattr__(self, "size", parsed)
        object.__setattr__(self, "metric", B)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))
        object.__setattr__(self, "_sqrt_metric", sqrt)
        object.__setattr__(self, "_invsqrt_metric", invsqrt)
        object.__setattr__(self, "_inverse_metric", inverse)

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def k(self) -> int:
        return self.size[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    @property
    def dim(self) -> int:
        return self.k * (self.n - self.k)

    @property
    def _ordinary(self) -> Grassmann:
        return Grassmann(self.size, atol=self.atol, eps=self.eps)

    def _forward(self, X: Array) -> Array:
        return self._sqrt_metric @ jnp.asarray(X)

    def _inverse(self, X: Array) -> Array:
        return self._invsqrt_metric @ jnp.asarray(X)

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        gram = _transpose(X) @ self.metric @ X
        return jnp.linalg.norm(gram - jnp.eye(self.k, dtype=gram.dtype), axis=(-2, -1)) <= tol

    def project(self, A: Array) -> Array:
        return self._inverse(self._ordinary.project(self._forward(A)))

    normalize = project

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        return jnp.linalg.norm(_transpose(X) @ self.metric @ U, axis=(-2, -1)) <= tol

    def tangent_project(self, X: Array, U: Array) -> Array:
        return jnp.asarray(U) - X @ (_transpose(X) @ self.metric @ U)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        del X
        return jnp.sum(jnp.asarray(U) * (self.metric @ jnp.asarray(V)), axis=(-2, -1))

    def norm(self, X: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(X, U, U), 0.0))

    def exp(self, X: Array, U: Array) -> Array:
        return self._inverse(self._ordinary.exp(self._forward(X), self._forward(U)))

    def log(self, X: Array, Y: Array) -> Array:
        return self._inverse(self._ordinary.log(self._forward(X), self._forward(Y)))

    def dist(self, X: Array, Y: Array) -> Array:
        return self._ordinary.dist(self._forward(X), self._forward(Y))

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        transported = self._ordinary.transport(self._forward(X), self._forward(Y), self._forward(U))
        return self._inverse(transported)

    transp = transport

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        ambient = self._inverse_metric @ jnp.asarray(egrad)
        return ambient - X @ (_transpose(X) @ jnp.asarray(egrad))

    egrad2rgrad = egrad_to_rgrad

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        return self._inverse(
            self._ordinary.random_point(key, sample_shape=as_sample_shape(sample_shape))
        )

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


__all__ = ["GeneralizedStiefel", "GeneralizedGrassmann"]
