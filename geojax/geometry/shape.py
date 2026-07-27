"""Kendall landmark-shape geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from .base import ExactGeometryMixin, Shape, as_sample_shape
from ._numerics import cos_from_squared_norm, sinc_from_squared_norm

Array = Any


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(jnp.asarray(A) * jnp.asarray(B), axis=(-2, -1))


def _parse_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError("KendallShape size must be (landmarks, ambient_dim).")
    parsed = tuple(int(value) for value in size)
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
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

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
        return template / jnp.linalg.norm(template)

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        X = jnp.asarray(X)
        if not self._shape_matches(X):
            return self._shape_failure(X)
        centered = jnp.linalg.norm(jnp.mean(X, axis=-2), axis=-1) <= tol
        normalized = jnp.abs(jnp.linalg.norm(X, axis=(-2, -1)) - 1.0) <= tol
        singular_values = jnp.linalg.svd(X, compute_uv=False)
        regular = jnp.min(singular_values, axis=-1) > tol
        return centered & normalized & regular

    def project(self, A: Array) -> Array:
        A = self._check_shape(A, name="A")
        centered = self.center(A)
        length = jnp.linalg.norm(centered, axis=(-2, -1), keepdims=True)
        template = jnp.broadcast_to(self._template(centered.dtype), centered.shape)
        normalized = centered / jnp.maximum(length, self.eps)
        singular_values = jnp.linalg.svd(normalized, compute_uv=False)
        regular = jnp.min(singular_values, axis=-1) > self.atol
        valid = (length[..., 0, 0] > self.eps) & regular
        return jnp.where(valid[..., None, None], normalized, template)

    normalize = project

    def align(self, Y: Array, X: Array) -> tuple[Array, Array]:
        """Align ``Y`` to ``X`` by orientation-preserving Procrustes rotation."""
        Y, X = self._check_shapes(("Y", Y), ("X", X))
        cross = _transpose(Y) @ X
        left, _, right_t = jnp.linalg.svd(cross, full_matrices=False)
        provisional = left @ right_t
        last_sign = jnp.where(jnp.linalg.det(provisional) < 0.0, -1.0, 1.0)
        signs = jnp.ones(left.shape[:-1], dtype=left.dtype)
        signs = signs.at[..., -1].set(last_sign)
        rotation = (left * signs[..., None, :]) @ right_t
        return jnp.asarray(Y) @ rotation, rotation

    def _vertical_generator(self, X: Array, U: Array) -> Array:
        gram = _transpose(X) @ X
        right_hand_side = _transpose(X) @ U - _transpose(U) @ X
        eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
        rotated = _transpose(eigenvectors) @ right_hand_side @ eigenvectors
        denominator = eigenvalues[..., :, None] + eigenvalues[..., None, :]
        solution = rotated / jnp.maximum(denominator, self.eps)
        generator = eigenvectors @ solution @ _transpose(eigenvectors)
        return 0.5 * (generator - _transpose(generator))

    def tangent_project(self, X: Array, U: Array) -> Array:
        X = self.project(X)
        _, U = self._check_shapes(("X", X), ("U", U))
        U = self.center(U)
        U = U - X * _trace_inner(X, U)[..., None, None]
        return U - X @ self._vertical_generator(X, U)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(X, U):
            return self._shape_failure(X)
        X, U = self._check_shapes(("X", X), ("U", U))
        centered = jnp.linalg.norm(jnp.mean(U, axis=-2), axis=-1) <= tol
        spherical = jnp.abs(_trace_inner(X, U)) <= tol
        horizontal = _transpose(X) @ U
        horizontal = jnp.linalg.norm(horizontal - _transpose(horizontal), axis=(-2, -1)) <= tol
        return centered & spherical & horizontal

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        _, U, V = self._check_shapes(("X", X), ("U", U), ("V", V))
        return _trace_inner(U, V)

    def norm(self, X: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(X, U, U), 0.0))

    def exp(self, X: Array, U: Array) -> Array:
        X = self.project(X)
        U = self.tangent_project(X, U)
        length_squared = jnp.maximum(self.inner(X, U, U), 0.0)[..., None, None]
        return (
            cos_from_squared_norm(length_squared) * X + sinc_from_squared_norm(length_squared) * U
        )

    def retr(self, X: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.project(jnp.asarray(X) + t * self.tangent_project(X, U))

    def log(self, X: Array, Y: Array) -> Array:
        X = self.project(X)
        aligned, _ = self.align(self.project(Y), X)
        cosine = jnp.clip(_trace_inner(X, aligned), -1.0, 1.0)
        direction = aligned - cosine[..., None, None] * X
        sine = jnp.linalg.norm(direction, axis=(-2, -1))
        angle = jnp.arctan2(sine, cosine)
        tangent = jnp.where(
            sine[..., None, None] > self.eps,
            angle[..., None, None] * direction / jnp.maximum(sine[..., None, None], self.eps),
            jnp.zeros_like(direction),
        )
        tangent = self.tangent_project(X, tangent)
        at_cut = (1.0 + cosine) <= self.atol
        return jnp.where(at_cut[..., None, None], jnp.full_like(tangent, jnp.nan), tangent)

    def dist(self, X: Array, Y: Array) -> Array:
        X = self.project(X)
        aligned, _ = self.align(self.project(Y), X)
        cosine = jnp.clip(_trace_inner(X, aligned), -1.0, 1.0)
        tangent_norm = jnp.linalg.norm(aligned - cosine[..., None, None] * X, axis=(-2, -1))
        return jnp.arctan2(tangent_norm, cosine)

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        X = self.project(X)
        Y = self.project(Y)
        aligned, rotation = self.align(Y, X)
        U = self.tangent_project(X, U)
        denominator = 1.0 + _trace_inner(X, aligned)
        coefficient = _trace_inner(U, aligned) / denominator
        transported = U - coefficient[..., None, None] * (X + aligned)
        transported = self.tangent_project(aligned, transported)
        return self.tangent_project(Y, transported @ _transpose(rotation))

    transp = transport

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        return self.tangent_project(X, egrad)

    egrad2rgrad = egrad_to_rgrad

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
            tangent = jnp.where(length > self.eps, tangent / length, tangent)
        return scale * tangent


__all__ = ["KendallShape"]
