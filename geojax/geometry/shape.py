"""Kendall landmark-shape geometry."""

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
from ._numerics import (
    acos_over_sin,
    cos_from_squared_norm,
    nonnegative,
    spherical_squared_dist,
    sinc_from_squared_norm,
    stable_metric_norm,
    stable_norm,
)

Array = Any


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(jnp.asarray(A) * jnp.asarray(B), axis=(-2, -1))


def _skew(A: Array) -> Array:
    return 0.5 * (A - _transpose(A))


def _symmetric(A: Array) -> Array:
    return 0.5 * (A + _transpose(A))


def _skew_sylvester(gram: Array, right_hand_side: Array) -> Array:
    """Solve ``gram @ omega + omega @ gram = rhs`` on skew matrices.

    Only sums of *distinct* eigenvalue indices enter this equation. It is
    therefore nonsingular on the regular stratum even when ``gram`` has one
    zero eigenvalue. Extend the operator by the identity on symmetric matrices
    to give ``custom_linear_solve`` an invertible, self-adjoint ambient
    operator. Its implicit derivatives do not differentiate an eigenbasis and
    remain valid at repeated eigenvalues, including under nested autodiff.
    The same solve applies to a signed polar factor whenever its distinct-index
    eigenvalue sums are nonzero, as at a unique proper Procrustes optimum.
    """
    gram = _symmetric(gram)
    if gram.shape[-1] == 1:
        return jnp.zeros_like(right_hand_side)

    def operator(value):
        skew = _skew(value)
        return gram @ skew + skew @ gram + _symmetric(value)

    def solve(_operator, value):
        eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
        rotated = _transpose(eigenvectors) @ _skew(value) @ eigenvectors
        denominator = eigenvalues[..., :, None] + eigenvalues[..., None, :]
        off_diagonal = ~jnp.eye(gram.shape[-1], dtype=bool)
        # Diagonal entries are identically zero in the skew subspace; guarding
        # them preserves rank-(m-1) points without regularizing the equation.
        denominator = jnp.where(off_diagonal, denominator, 1.0)
        solution = jnp.where(off_diagonal, rotated / denominator, 0.0)
        return _skew(eigenvectors @ solution @ _transpose(eigenvectors)) + _symmetric(value)

    result = jax.lax.custom_linear_solve(
        operator, _skew(right_hand_side), solve=solve, symmetric=True
    )
    return _skew(result)


@jax.custom_jvp
def _proper_procrustes(cross: Array) -> Array:
    """Orientation-preserving polar factor, smooth when the optimum is unique.

    The primal still selects an optimal rotation at a nonunique alignment;
    derivatives there are undefined. Public log/transport methods separately
    reject those cut-locus pairs using their uniqueness certificate.
    """
    if cross.shape[-1] == 1:
        return jnp.ones_like(cross)
    left, _, right_t = jnp.linalg.svd(cross, full_matrices=False)
    provisional = left @ right_t
    last_sign = jnp.where(jnp.linalg.det(provisional) < 0.0, -1.0, 1.0)
    signs = jnp.ones(left.shape[:-1], dtype=left.dtype)
    signs = signs.at[..., -1].set(last_sign)
    return (left * signs[..., None, :]) @ right_t


@_proper_procrustes.defjvp
def _proper_procrustes_jvp(primals, tangents):
    (cross,), (cross_dot,) = primals, tangents
    rotation = _proper_procrustes(cross)
    signed_polar = _symmetric(_transpose(rotation) @ cross)
    rhs = _transpose(rotation) @ cross_dot - _transpose(cross_dot) @ rotation
    omega = _skew_sylvester(signed_polar, rhs)
    return rotation, rotation @ omega


def _parse_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError("KendallShape size must be (landmarks, ambient_dim).")
    parsed = tuple(
        validate_integer(value, name="KendallShape size entry", minimum=1) for value in size
    )
    if len(parsed) != 2 or parsed[1] < 1 or parsed[0] <= parsed[1]:
        raise ValueError("KendallShape requires landmarks > ambient_dim >= 1.")
    return parsed


@dataclass(frozen=True, init=False)
class KendallShape(ExactGeometryMixin):
    """Regular Kendall shape space of centered, scale-normalized landmarks.

    Public points are pre-shape matrices of size ``landmarks x ambient_dim``.
    Frames related by a right action of ``SO(ambient_dim)`` represent the same
    shape. Tangents are represented by horizontal pre-shape vectors.
    """

    size: tuple[int, int]
    atol: float
    eps: float

    transport_is_isometric = False
    transport_is_parallel = False

    def __init__(self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10):
        object.__setattr__(self, "size", _parse_size(size))
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="KendallShape atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="KendallShape eps"))

    @property
    def landmarks(self) -> int:
        return self.size[0]

    @property
    def ambient_dim(self) -> int:
        return self.size[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    @property
    def dim(self) -> int:
        preshape_dim = (self.landmarks - 1) * self.ambient_dim - 1
        rotations_dim = self.ambient_dim * (self.ambient_dim - 1) // 2
        return preshape_dim - rotations_dim

    def center(self, X: Array) -> Array:
        X = self._check_shape(X, name="X")
        return X - jnp.mean(X, axis=-2, keepdims=True)

    def _template(self, dtype: jnp.dtype) -> Array:
        template = jnp.zeros(self.shape, dtype=dtype)
        template = template.at[: self.ambient_dim, :].set(jnp.eye(self.ambient_dim, dtype=dtype))
        template = self.center(template)
        return template / stable_norm(template, axis=(-2, -1))

    def _is_regular(self, X: Array) -> Array:
        """Return whether the right SO(m) action has trivial isotropy."""
        singular_values = jnp.linalg.svd(X, compute_uv=False)
        if self.ambient_dim == 1:
            return singular_values[..., 0] > 0.0
        # Nullity at most one is sufficient. Requiring full column rank would
        # incorrectly exclude collinear planar shapes from the regular space.
        return singular_values[..., -2] > 0.0

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        X = jnp.asarray(X)
        if not self._shape_matches(X):
            return self._shape_failure(X)
        centered = stable_norm(jnp.mean(X, axis=-2), axis=-1) <= tol
        normalized = jnp.abs(stable_norm(X, axis=(-2, -1)) - 1.0) <= tol
        regular = self._is_regular(X)
        return centered & normalized & regular

    def project(self, A: Array) -> Array:
        A = self._check_shape(A, name="A")
        centered = self.center(A)
        length = stable_norm(centered, axis=(-2, -1), keepdims=True)
        template = jnp.broadcast_to(self._template(centered.dtype), centered.shape)
        safe_length = jnp.where(length > 0.0, length, jnp.ones_like(length))
        normalized = centered / safe_length
        regular = self._is_regular(normalized)
        valid = (length[..., 0, 0] > 0.0) & regular
        return jnp.where(valid[..., None, None], normalized, template)

    def align(self, Y: Array, X: Array) -> tuple[Array, Array]:
        """Align ``Y`` to ``X`` by orientation-preserving Procrustes rotation."""
        aligned, rotation, _ = self._alignment(Y, X)
        return aligned, rotation

    def _alignment(self, Y: Array, X: Array) -> tuple[Array, Array, Array]:
        """Return an optimal alignment and whether that optimizer is unique."""
        Y, X = self._check_shapes(("Y", Y), ("X", X))
        cross = _transpose(Y) @ X
        rotation = _proper_procrustes(cross)
        # These values certify uniqueness only. The differentiable rotation
        # uses the implicit stationarity equation, not SVD-vector derivatives.
        singular_values = jnp.linalg.svd(jax.lax.stop_gradient(cross), compute_uv=False)
        dtype = jnp.result_type(cross, float)
        # A cross-covariance can be zero through cancellation, so its own norm
        # is not an adequate roundoff scale. Bound the matrix product error by
        # the two preshape norms and the landmark contraction dimension.
        product_scale = stable_norm(Y, axis=(-2, -1)) * stable_norm(X, axis=(-2, -1))
        threshold = 32.0 * self.landmarks * jnp.finfo(dtype).eps * product_scale
        if self.ambient_dim == 1:
            unique = jnp.ones(cross.shape[:-2], dtype=bool)
        else:
            rank_at_least_m_minus_one = singular_values[..., -2] > threshold
            full_rank = singular_values[..., -1] > threshold
            reflected = jnp.linalg.slogdet(cross)[0] < 0.0
            smallest_is_tied = (singular_values[..., -2] - singular_values[..., -1]) <= threshold
            unique = rank_at_least_m_minus_one & ~(reflected & full_rank & smallest_is_tied)
        return jnp.asarray(Y) @ rotation, rotation, unique

    def _vertical_generator(self, X: Array, U: Array) -> Array:
        gram = _transpose(X) @ X
        right_hand_side = _transpose(X) @ U - _transpose(U) @ X
        return _skew_sylvester(gram, right_hand_side)

    def tangent_project(self, X: Array, U: Array) -> Array:
        X = self.project(X)
        _, U = self._check_shapes(("X", X), ("U", U))
        U = self.center(U)
        U = U - X * _trace_inner(X, U)[..., None, None]
        return U - X @ self._vertical_generator(X, U)

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(X, U):
            return self._shape_failure(X)
        X, U = self._check_shapes(("X", X), ("U", U))
        centered = stable_norm(jnp.mean(U, axis=-2), axis=-1) <= tol
        spherical = jnp.abs(_trace_inner(X, U)) <= tol
        horizontal = _transpose(X) @ U
        horizontal = stable_norm(horizontal - _transpose(horizontal), axis=(-2, -1)) <= tol
        return centered & spherical & horizontal

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        _, U, V = self._check_shapes(("X", X), ("U", U), ("V", V))
        return _trace_inner(U, V)

    def norm(self, X: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(X, normalized, normalized),
            axis=(-2, -1),
        )

    def exp(self, X: Array, U: Array) -> Array:
        X = self.project(X)
        U = self.tangent_project(X, U)
        length_squared = nonnegative(self.inner(X, U, U))[..., None, None]
        result = (
            cos_from_squared_norm(length_squared) * X + sinc_from_squared_norm(length_squared) * U
        )
        valid = self.belongs(result)
        return jnp.where(valid[..., None, None], result, jnp.full_like(result, jnp.nan))

    def retr(self, X: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.project(jnp.asarray(X) + self._scale_tangent(self.tangent_project(X, U), t))

    def log(self, X: Array, Y: Array) -> Array:
        X = self.project(X)
        aligned, _, unique = self._alignment(self.project(Y), X)
        cosine = jnp.clip(_trace_inner(X, aligned), -1.0, 1.0)
        direction = aligned - cosine[..., None, None] * X
        sine_squared = jnp.maximum(_trace_inner(direction, direction), 0.0)
        tangent = acos_over_sin(cosine, sine_squared)[..., None, None] * direction
        tangent = self.tangent_project(X, tangent)
        dtype = jnp.result_type(X, aligned, float)
        at_cut = (1.0 + cosine) <= 32.0 * self.ambient_dim * jnp.finfo(dtype).eps
        at_cut = at_cut | ~unique
        return jnp.where(at_cut[..., None, None], jnp.full_like(tangent, jnp.nan), tangent)

    def dist(self, X: Array, Y: Array) -> Array:
        X = self.project(X)
        aligned, _ = self.align(self.project(Y), X)
        cosine = jnp.clip(_trace_inner(X, aligned), -1.0, 1.0)
        direction = aligned - cosine[..., None, None] * X
        tangent_norm = stable_norm(direction, axis=(-2, -1))
        return jnp.arctan2(tangent_norm, cosine)

    def squared_dist(self, X: Array, Y: Array) -> Array:
        X = self.project(X)
        aligned, _ = self.align(self.project(Y), X)
        return spherical_squared_dist(X, aligned, axis=(-2, -1))

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        X = self.project(X)
        Y = self.project(Y)
        aligned, rotation, unique = self._alignment(Y, X)
        U = self.tangent_project(X, U)
        denominator = 1.0 + _trace_inner(X, aligned)
        dtype = jnp.result_type(X, aligned, float)
        threshold = 32.0 * self.ambient_dim * jnp.finfo(dtype).eps
        valid = unique & (denominator > threshold)
        safe_denominator = jnp.where(valid, denominator, jnp.ones_like(denominator))
        coefficient = _trace_inner(U, aligned) / safe_denominator
        transported = U - coefficient[..., None, None] * (X + aligned)
        transported = self.tangent_project(aligned, transported)
        transported = self.tangent_project(Y, transported @ _transpose(rotation))
        return jnp.where(
            valid[..., None, None],
            transported,
            jnp.full_like(transported, jnp.nan),
        )

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        return self.tangent_project(X, egrad)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        normal = jax.random.normal(key, shape=as_sample_shape(sample_shape) + self.shape)
        return self.project(normal)

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


__all__ = ["KendallShape"]
