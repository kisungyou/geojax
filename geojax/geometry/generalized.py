"""Generalized orthogonality geometries under an SPD metric matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from .base import (
    ExactGeometryMixin,
    Shape,
    as_sample_shape,
    validate_integer,
    validate_nonnegative,
    validate_positive,
)
from .grassmann import Grassmann
from ._numerics import stable_metric_norm, stable_norm, sqrt_nonnegative
from .stiefel import StiefelEuclidean, StiefelLogInfo

Array = Any


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _sym(A: Array) -> Array:
    return 0.5 * (jnp.asarray(A) + _transpose(jnp.asarray(A)))


def _parse_size(size: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError(f"{name} size must be a pair (ambient_dim, rank).")
    parsed = tuple(validate_integer(value, name=f"{name} size entry", minimum=1) for value in size)
    if len(parsed) != 2 or parsed[0] < 1 or parsed[1] < 1 or parsed[1] > parsed[0]:
        raise ValueError(f"{name} size must satisfy ambient_dim >= rank >= 1.")
    return parsed


def _metric_factors(
    metric: Array, n: int, eps: float, name: str
) -> tuple[Array, Array, Array, Array]:
    eps = validate_positive(eps, name=f"{name} eps")
    raw_metric = jnp.asarray(metric)
    if jnp.iscomplexobj(raw_metric):
        raise TypeError(f"{name} metric must be real-valued; complex arrays are unsupported.")
    raw_metric = jnp.asarray(raw_metric, dtype=float)
    if raw_metric.shape != (n, n):
        raise ValueError(f"{name} metric must have shape ({n}, {n}).")
    if not bool(jnp.all(jnp.isfinite(raw_metric))):
        raise ValueError(f"{name} metric must contain only finite values.")
    asymmetry = stable_norm(raw_metric - _transpose(raw_metric), axis=(-2, -1))
    scale = jnp.maximum(
        stable_norm(raw_metric, axis=(-2, -1)),
        jnp.finfo(raw_metric.dtype).tiny,
    )
    tolerance = 32.0 * jnp.finfo(raw_metric.dtype).eps * scale
    if float(asymmetry) > float(tolerance):
        raise ValueError(f"{name} metric must be symmetric positive definite.")
    metric = _sym(raw_metric)
    eigenvalues, eigenvectors = jnp.linalg.eigh(metric)
    spectral_scale = jnp.max(jnp.abs(eigenvalues))
    eigenvalue_floor = (
        max(
            eps,
            100.0 * float(jnp.finfo(metric.dtype).eps) * n,
        )
        * spectral_scale
    )
    if not bool(jnp.all(eigenvalues > eigenvalue_floor)):
        raise ValueError(f"{name} metric must be symmetric positive definite.")
    sqrt = (eigenvectors * jnp.sqrt(eigenvalues)[None, :]) @ eigenvectors.T
    invsqrt = (eigenvectors * (1.0 / jnp.sqrt(eigenvalues))[None, :]) @ eigenvectors.T
    inverse = (eigenvectors * (1.0 / eigenvalues)[None, :]) @ eigenvectors.T
    return metric, sqrt, invsqrt, inverse


@dataclass(frozen=True, init=False, eq=False)
class GeneralizedStiefel(ExactGeometryMixin):
    """Frames satisfying ``X.T @ metric @ X = I``.

    The metric ``trace(U.T @ metric @ V)`` is the pullback of the embedded
    Euclidean Stiefel metric under ``X -> metric^(1/2) X``. Its exact
    exponential and numerical-local logarithm are pulled back through this
    isometry.
    """

    log_is_exact = False
    dist_is_exact = False
    log_kind = "numerical-local"
    dist_kind = "numerical-local"
    hessian_conversion_is_exact = True
    riemannian_gradient_jvp_is_exact = True
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
        atol = validate_nonnegative(atol, name="GeneralizedStiefel atol")
        eps = validate_positive(eps, name="GeneralizedStiefel eps")
        log_maxiter = validate_integer(
            log_maxiter, name="GeneralizedStiefel log_maxiter", minimum=1
        )
        log_tol = validate_positive(log_tol, name="GeneralizedStiefel log_tol")
        B, sqrt, invsqrt, inverse = _metric_factors(metric, parsed[0], eps, "GeneralizedStiefel")
        object.__setattr__(self, "size", parsed)
        object.__setattr__(self, "metric", B)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "_sqrt_metric", sqrt)
        object.__setattr__(self, "_invsqrt_metric", invsqrt)
        object.__setattr__(self, "_inverse_metric", inverse)
        object.__setattr__(self, "_log_maxiter", log_maxiter)
        object.__setattr__(self, "_log_tol", log_tol)

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
        if not self._shape_matches(X):
            return self._shape_failure(X)
        X = self._check_shape(X, name="X")
        gram = _transpose(X) @ self.metric @ X
        return jnp.linalg.norm(gram - jnp.eye(self.k, dtype=gram.dtype), axis=(-2, -1)) <= tol

    def project(self, A: Array) -> Array:
        A = self._check_shape(A, name="A")
        return self._inverse(self._ordinary.project(self._forward(A)))

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(X, U):
            return self._shape_failure(X)
        X, U = self._check_shapes(("X", X), ("U", U))
        constraint = _transpose(X) @ self.metric @ U
        return jnp.linalg.norm(constraint + _transpose(constraint), axis=(-2, -1)) <= tol

    def tangent_project(self, X: Array, U: Array) -> Array:
        X, U = self._check_shapes(("X", X), ("U", U))
        return U - X @ _sym(_transpose(X) @ self.metric @ U)

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        _, U, V = self._check_shapes(("X", X), ("U", U), ("V", V))
        return jnp.sum(U * (self.metric @ V), axis=(-2, -1))

    def norm(self, X: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(X, normalized, normalized),
            axis=(-2, -1),
        )

    def exp(self, X: Array, U: Array) -> Array:
        X, U = self._check_shapes(("X", X), ("U", U))
        return self._inverse(self._ordinary.exp(self._forward(X), self._forward(U)))

    def log_with_info(self, X: Array, Y: Array) -> tuple[Array, StiefelLogInfo]:
        """Return the pulled-back local logarithm candidate and shooting diagnostics."""
        X, Y = self._check_shapes(("X", X), ("Y", Y))
        tangent, info = self._ordinary.log_with_info(self._forward(X), self._forward(Y))
        return self._inverse(tangent), info

    def log(self, X: Array, Y: Array) -> Array:
        tangent, info = self.log_with_info(X, Y)
        return jnp.where(
            info.converged[..., None, None],
            tangent,
            jnp.full_like(tangent, jnp.nan),
        )

    def dist(self, X: Array, Y: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(X, Y))

    def squared_dist(self, X: Array, Y: Array) -> Array:
        return self._ordinary.squared_dist(self._forward(X), self._forward(Y))

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        X, Y, U = self._check_shapes(("X", X), ("Y", Y), ("U", U))
        transported = self._ordinary.transport(self._forward(X), self._forward(Y), self._forward(U))
        return self._inverse(transported)

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        ambient = self._inverse_metric @ jnp.asarray(egrad)
        return ambient - X @ _sym(_transpose(X) @ jnp.asarray(egrad))

    def ehess_to_rhess(
        self,
        X: Array,
        egrad: Array,
        ehess_vec: Array,
        U: Array,
    ) -> Array:
        ordinary = self._ordinary
        converted = ordinary.ehess_to_rhess(
            self._forward(X),
            self._inverse(egrad),
            self._inverse(ehess_vec),
            self._forward(U),
        )
        return self._inverse(converted)

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
            safe_length = jnp.where(length > 0.0, length, jnp.ones_like(length))
            tangent = jnp.where(length > 0.0, tangent / safe_length, tangent)
        return self._scale_tangent(tangent, scale)


@dataclass(frozen=True, init=False, eq=False)
class GeneralizedGrassmann(ExactGeometryMixin):
    """Generalized Grassmann geometry for ``metric``-orthonormal subspaces."""

    hessian_conversion_is_exact = True
    riemannian_gradient_jvp_is_exact = True

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
        atol = validate_nonnegative(atol, name="GeneralizedGrassmann atol")
        eps = validate_positive(eps, name="GeneralizedGrassmann eps")
        B, sqrt, invsqrt, inverse = _metric_factors(metric, parsed[0], eps, "GeneralizedGrassmann")
        object.__setattr__(self, "size", parsed)
        object.__setattr__(self, "metric", B)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "eps", eps)
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
        if not self._shape_matches(X):
            return self._shape_failure(X)
        X = self._check_shape(X, name="X")
        gram = _transpose(X) @ self.metric @ X
        return jnp.linalg.norm(gram - jnp.eye(self.k, dtype=gram.dtype), axis=(-2, -1)) <= tol

    def project(self, A: Array) -> Array:
        A = self._check_shape(A, name="A")
        return self._inverse(self._ordinary.project(self._forward(A)))

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(X, U):
            return self._shape_failure(X)
        X, U = self._check_shapes(("X", X), ("U", U))
        return jnp.linalg.norm(_transpose(X) @ self.metric @ U, axis=(-2, -1)) <= tol

    def tangent_project(self, X: Array, U: Array) -> Array:
        X, U = self._check_shapes(("X", X), ("U", U))
        return U - X @ (_transpose(X) @ self.metric @ U)

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        _, U, V = self._check_shapes(("X", X), ("U", U), ("V", V))
        return jnp.sum(U * (self.metric @ V), axis=(-2, -1))

    def norm(self, X: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(X, normalized, normalized),
            axis=(-2, -1),
        )

    def exp(self, X: Array, U: Array) -> Array:
        X, U = self._check_shapes(("X", X), ("U", U))
        return self._inverse(self._ordinary.exp(self._forward(X), self._forward(U)))

    def log(self, X: Array, Y: Array) -> Array:
        X, Y = self._check_shapes(("X", X), ("Y", Y))
        return self._inverse(self._ordinary.log(self._forward(X), self._forward(Y)))

    def dist(self, X: Array, Y: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(X, Y))

    def squared_dist(self, X: Array, Y: Array) -> Array:
        return self._ordinary.squared_dist(self._forward(X), self._forward(Y))

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        X, Y, U = self._check_shapes(("X", X), ("Y", Y), ("U", U))
        transported = self._ordinary.transport(self._forward(X), self._forward(Y), self._forward(U))
        return self._inverse(transported)

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        ambient = self._inverse_metric @ jnp.asarray(egrad)
        return ambient - X @ (_transpose(X) @ jnp.asarray(egrad))

    def ehess_to_rhess(
        self,
        X: Array,
        egrad: Array,
        ehess_vec: Array,
        U: Array,
    ) -> Array:
        """Pull back the exact ordinary-Grassmann Hessian conversion."""
        converted = self._ordinary.ehess_to_rhess(
            self._forward(X),
            self._inverse(egrad),
            self._inverse(ehess_vec),
            self._forward(U),
        )
        return self._inverse(converted)

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
            safe_length = jnp.where(length > 0.0, length, jnp.ones_like(length))
            tangent = jnp.where(length > 0.0, tangent / safe_length, tangent)
        return self._scale_tangent(tangent, scale)


__all__ = ["GeneralizedStiefel", "GeneralizedGrassmann"]
