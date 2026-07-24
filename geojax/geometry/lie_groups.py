"""Matrix geometries for rotations and rigid transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import GeometryMixin, as_sample_shape
from ._numerics import matrix_expm

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _skew(A: Array) -> Array:
    return 0.5 * (A - _transpose(A))


def _sym(A: Array) -> Array:
    return 0.5 * (A + _transpose(A))


def _complex_dtype(dtype: jnp.dtype) -> jnp.dtype:
    return jnp.complex128 if jnp.dtype(dtype) == jnp.float64 else jnp.complex64


@jax.custom_jvp
def _principal_orthogonal_log(M: Array) -> Array:
    """Principal real skew logarithm away from eigenvalue -1.

    Orthogonal matrices are normal, so a unitary eigendecomposition evaluates
    the principal matrix logarithm.  The custom JVP below supplies its Frechet
    derivative without differentiating eigenvectors.
    """
    M = jnp.asarray(M)
    Mc = M.astype(_complex_dtype(M.dtype))
    eigvals, eigvecs = jnp.linalg.eig(Mc)
    log_eigvals = 1j * jnp.angle(eigvals)
    result = (eigvecs * log_eigvals[..., None, :]) @ _transpose(jnp.conj(eigvecs))
    return _skew(jnp.real(result)).astype(M.dtype)


@_principal_orthogonal_log.defjvp
def _principal_orthogonal_log_jvp(primals, tangents):
    (M,), (E,) = primals, tangents
    M = jnp.asarray(M)
    complex_dtype = _complex_dtype(M.dtype)
    Mc = M.astype(complex_dtype)
    Ec = jnp.asarray(E).astype(complex_dtype)
    eigvals, eigvecs = jnp.linalg.eig(Mc)
    log_eigvals = 1j * jnp.angle(eigvals)

    eig_i = eigvals[..., :, None]
    eig_j = eigvals[..., None, :]
    log_i = log_eigvals[..., :, None]
    log_j = log_eigvals[..., None, :]
    denominator = eig_i - eig_j
    eps = jnp.finfo(M.dtype).eps
    separated = jnp.abs(denominator) > 32.0 * eps
    safe_denominator = jnp.where(separated, denominator, jnp.ones_like(denominator))
    divided_difference = jnp.where(
        separated,
        (log_i - log_j) / safe_denominator,
        1.0 / eig_i,
    )

    eigvecs_h = _transpose(jnp.conj(eigvecs))
    rotated_E = eigvecs_h @ Ec @ eigvecs
    derivative = eigvecs @ (divided_difference * rotated_E) @ eigvecs_h
    tangent_out = _skew(jnp.real(derivative)).astype(M.dtype)
    return _principal_orthogonal_log(M), tangent_out


@dataclass(frozen=True, init=False)
class SpecialOrthogonal(GeometryMixin):
    """Rotation group SO(n) with the Frobenius bi-invariant metric.

    A point is an ``n x n`` matrix ``R`` satisfying ``R.T @ R = I`` and
    ``det(R) = 1``.  A tangent vector at ``R`` is represented in ambient form
    as ``R @ Omega`` for a skew-symmetric matrix ``Omega``.
    """

    hessian_conversion_is_exact = True
    riemannian_gradient_jvp_is_exact = True

    size: int
    atol: float
    eps: float

    def __init__(self, size: int, *, atol: float = 1e-6, eps: float = 1e-12) -> None:
        size = int(size)
        if size < 2:
            raise ValueError("SpecialOrthogonal size must be at least 2.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def n(self) -> int:
        return self.size

    @property
    def dim(self) -> int:
        return self.n * (self.n - 1) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def identity(self) -> Array:
        return jnp.eye(self.n)

    def belongs(self, R: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        R = jnp.asarray(R)
        identity = jnp.eye(self.n, dtype=R.dtype)
        orthogonal = jnp.linalg.norm(_transpose(R) @ R - identity, axis=(-2, -1)) <= tol
        orientation = jnp.abs(jnp.linalg.det(R) - 1.0) <= tol
        return orthogonal & orientation

    def project(self, A: Array) -> Array:
        A = jnp.asarray(A)
        U, _, Vh = jnp.linalg.svd(A, full_matrices=False)
        provisional = U @ Vh
        last_sign = jnp.where(jnp.linalg.det(provisional) < 0.0, -1.0, 1.0)
        signs = jnp.ones(U.shape[:-1], dtype=U.dtype)
        signs = signs.at[..., -1].set(last_sign)
        return (U * signs[..., None, :]) @ Vh

    def is_tangent(self, R: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        body = _transpose(jnp.asarray(R)) @ jnp.asarray(U)
        return jnp.linalg.norm(body + _transpose(body), axis=(-2, -1)) <= tol

    def tangent_project(self, R: Array, A: Array) -> Array:
        R = jnp.asarray(R)
        return R @ _skew(_transpose(R) @ jnp.asarray(A))

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, R: Array, U: Array, V: Array) -> Array:
        del R
        return jnp.sum(jnp.asarray(U) * jnp.asarray(V), axis=(-2, -1))

    def norm(self, R: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(R, U, U), 0.0))

    def _relative_log(self, relative: Array) -> Array:
        relative = jnp.asarray(relative)
        log_relative = _principal_orthogonal_log(relative)
        identity = jnp.eye(self.n, dtype=relative.dtype)
        distance_to_cut = jnp.min(jnp.linalg.svd(relative + identity, compute_uv=False), axis=-1)
        at_cut = distance_to_cut <= self.atol
        return jnp.where(at_cut[..., None, None], jnp.nan, log_relative)

    def exp(self, R: Array, U: Array) -> Array:
        R = jnp.asarray(R)
        U = self.tangent_project(R, U)
        omega = _skew(_transpose(R) @ U)
        return R @ matrix_expm(omega)

    def retr(self, R: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(R, t * U)

    def log(self, R: Array, Q: Array) -> Array:
        # Do not insert an SVD projection here: its repeated singular values on
        # SO(n) make an otherwise smooth distance nondifferentiable to JAX.
        R = jnp.asarray(R)
        Q = jnp.asarray(Q)
        return R @ self._relative_log(_transpose(R) @ Q)

    def dist(self, R: Array, Q: Array) -> Array:
        return jnp.sqrt(self.squared_dist(R, Q))

    def squared_dist(self, R: Array, Q: Array) -> Array:
        tangent = self.log(R, Q)
        return jnp.maximum(self.inner(R, tangent, tangent), 0.0)

    def transport(self, R: Array, Q: Array, U: Array) -> Array:
        """Parallel transport along the selected shortest geodesic."""
        R = jnp.asarray(R)
        Q = jnp.asarray(Q)
        U = self.tangent_project(R, U)
        omega = self._relative_log(_transpose(R) @ Q)
        half = matrix_expm(0.5 * omega)
        body = _skew(_transpose(R) @ U)
        transported = R @ half @ body @ half
        return self.tangent_project(Q, transported)

    transp = transport

    def egrad_to_rgrad(self, R: Array, egrad: Array) -> Array:
        return self.tangent_project(R, egrad)

    egrad2rgrad = egrad_to_rgrad

    def ehess_to_rhess(self, R: Array, egrad: Array, ehess_vec: Array, U: Array) -> Array:
        correction = jnp.asarray(U) @ _sym(_transpose(R) @ jnp.asarray(egrad))
        return self.tangent_project(R, jnp.asarray(ehess_vec) - correction)

    def compose(self, R: Array, Q: Array) -> Array:
        return jnp.asarray(R) @ jnp.asarray(Q)

    def inverse(self, R: Array) -> Array:
        return _transpose(jnp.asarray(R))

    def group_exp(self, omega: Array) -> Array:
        """Lie-group exponential from a skew matrix at the identity."""
        return matrix_expm(_skew(jnp.asarray(omega)))

    def group_log(self, R: Array) -> Array:
        """Principal Lie-group logarithm, undefined at rotations by pi."""
        return self._relative_log(jnp.asarray(R))

    def apply(self, R: Array, points: Array) -> Array:
        return jnp.einsum("...ij,...j->...i", jnp.asarray(R), jnp.asarray(points))

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        A = jax.random.normal(key, shape=sample_shape + self.shape)
        Q, upper = jnp.linalg.qr(A)
        diagonal = jnp.diagonal(upper, axis1=-2, axis2=-1)
        signs = jnp.where(diagonal < 0.0, -1.0, 1.0)
        Q = Q * signs[..., None, :]
        last_sign = jnp.where(jnp.linalg.det(Q) < 0.0, -1.0, 1.0)
        correction = jnp.ones(Q.shape[:-1], dtype=Q.dtype)
        correction = correction.at[..., -1].set(last_sign)
        return Q * correction[..., None, :]

    def random_tangent(
        self,
        key: Array,
        R: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        Z = jax.random.normal(key, shape=jnp.shape(R))
        U = self.tangent_project(R, Z)
        if normalize:
            norm = self.norm(R, U)[..., None, None]
            U = jnp.where(norm > self.eps, U / norm, U)
        return scale * U


@dataclass(frozen=True, init=False)
class SpecialEuclidean(GeometryMixin):
    """Rigid-motion group SE(n) with its canonical product metric.

    Points use homogeneous matrices ``[[R, t], [0, 1]]``.  The metric is the
    direct product of the Frobenius bi-invariant metric on ``SO(n)`` and the
    Euclidean metric on translations.  Its Riemannian exponential is distinct
    from the Lie-group exponential except for special tangent directions.
    """

    hessian_conversion_is_exact = True
    riemannian_gradient_jvp_is_exact = True

    size: int
    atol: float
    eps: float

    def __init__(self, size: int, *, atol: float = 1e-6, eps: float = 1e-12) -> None:
        size = int(size)
        if size < 2:
            raise ValueError("SpecialEuclidean size must be at least 2.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def n(self) -> int:
        return self.size

    @property
    def dim(self) -> int:
        return self.n * (self.n - 1) // 2 + self.n

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n + 1, self.n + 1)

    @property
    def _rotations(self) -> SpecialOrthogonal:
        return SpecialOrthogonal(size=self.n, atol=self.atol, eps=self.eps)

    @property
    def identity(self) -> Array:
        return jnp.eye(self.n + 1)

    def from_components(self, rotation: Array, translation: Array) -> Array:
        rotation = jnp.asarray(rotation)
        translation = jnp.asarray(translation)
        out = jnp.zeros(rotation.shape[:-2] + self.shape, dtype=rotation.dtype)
        out = out.at[..., : self.n, : self.n].set(rotation)
        out = out.at[..., : self.n, self.n].set(translation)
        return out.at[..., self.n, self.n].set(1.0)

    def tangent_from_components(self, rotation: Array, translation: Array) -> Array:
        rotation = jnp.asarray(rotation)
        translation = jnp.asarray(translation)
        out = jnp.zeros(rotation.shape[:-2] + self.shape, dtype=rotation.dtype)
        out = out.at[..., : self.n, : self.n].set(rotation)
        return out.at[..., : self.n, self.n].set(translation)

    def rotation(self, G: Array) -> Array:
        return jnp.asarray(G)[..., : self.n, : self.n]

    def translation(self, G: Array) -> Array:
        return jnp.asarray(G)[..., : self.n, self.n]

    def belongs(self, G: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        G = jnp.asarray(G)
        rotation_ok = self._rotations.belongs(self.rotation(G), atol=tol)
        expected_bottom = jnp.zeros(G.shape[:-2] + (self.n + 1,), dtype=G.dtype)
        expected_bottom = expected_bottom.at[..., -1].set(1.0)
        bottom_ok = jnp.linalg.norm(G[..., self.n, :] - expected_bottom, axis=-1) <= tol
        return rotation_ok & bottom_ok

    def project(self, A: Array) -> Array:
        A = jnp.asarray(A)
        return self.from_components(self._rotations.project(self.rotation(A)), self.translation(A))

    def is_tangent(self, G: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        U = jnp.asarray(U)
        rotation_ok = self._rotations.is_tangent(self.rotation(G), self.rotation(U), atol=tol)
        bottom_ok = jnp.linalg.norm(U[..., self.n, :], axis=-1) <= tol
        return rotation_ok & bottom_ok

    def tangent_project(self, G: Array, A: Array) -> Array:
        return self.tangent_from_components(
            self._rotations.tangent_project(self.rotation(G), self.rotation(A)),
            self.translation(A),
        )

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, G: Array, U: Array, V: Array) -> Array:
        rotation_inner = self._rotations.inner(self.rotation(G), self.rotation(U), self.rotation(V))
        translation_inner = jnp.sum(self.translation(U) * self.translation(V), axis=-1)
        return rotation_inner + translation_inner

    def norm(self, G: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(G, U, U), 0.0))

    def exp(self, G: Array, U: Array) -> Array:
        G = jnp.asarray(G)
        U = self.tangent_project(G, U)
        R = self.rotation(G)
        next_R = self._rotations.exp(R, self.rotation(U))
        next_t = self.translation(G) + self.translation(U)
        return self.from_components(next_R, next_t)

    def retr(self, G: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(G, t * U)

    def log(self, G: Array, H: Array) -> Array:
        G = jnp.asarray(G)
        H = jnp.asarray(H)
        return self.tangent_from_components(
            self._rotations.log(self.rotation(G), self.rotation(H)),
            self.translation(H) - self.translation(G),
        )

    def dist(self, G: Array, H: Array) -> Array:
        return jnp.sqrt(self.squared_dist(G, H))

    def squared_dist(self, G: Array, H: Array) -> Array:
        rotation_dist_sq = self._rotations.squared_dist(self.rotation(G), self.rotation(H))
        translation_dist_sq = jnp.sum((self.translation(H) - self.translation(G)) ** 2, axis=-1)
        return jnp.maximum(rotation_dist_sq + translation_dist_sq, 0.0)

    def transport(self, G: Array, H: Array, U: Array) -> Array:
        G = self.project(G)
        H = self.project(H)
        U = self.tangent_project(G, U)
        return self.tangent_from_components(
            self._rotations.transport(self.rotation(G), self.rotation(H), self.rotation(U)),
            self.translation(U),
        )

    transp = transport

    def egrad_to_rgrad(self, G: Array, egrad: Array) -> Array:
        return self.tangent_project(G, egrad)

    egrad2rgrad = egrad_to_rgrad

    def ehess_to_rhess(self, G: Array, egrad: Array, ehess_vec: Array, U: Array) -> Array:
        rotation_hess = self._rotations.ehess_to_rhess(
            self.rotation(G),
            self.rotation(egrad),
            self.rotation(ehess_vec),
            self.rotation(U),
        )
        return self.tangent_from_components(rotation_hess, self.translation(ehess_vec))

    def compose(self, G: Array, H: Array) -> Array:
        R = self.rotation(G)
        S = self.rotation(H)
        t = self.translation(G)
        s = self.translation(H)
        return self.from_components(R @ S, t + jnp.einsum("...ij,...j->...i", R, s))

    def inverse(self, G: Array) -> Array:
        R_inv = _transpose(self.rotation(G))
        t_inv = -jnp.einsum("...ij,...j->...i", R_inv, self.translation(G))
        return self.from_components(R_inv, t_inv)

    def _left_jacobian(self, omega: Array) -> Array:
        omega = _skew(jnp.asarray(omega))
        zeros = jnp.zeros_like(omega)
        identity = jnp.broadcast_to(jnp.eye(self.n, dtype=omega.dtype), omega.shape)
        top = jnp.concatenate([omega, identity], axis=-1)
        bottom = jnp.concatenate([zeros, zeros], axis=-1)
        augmented = jnp.concatenate([top, bottom], axis=-2)
        return matrix_expm(augmented)[..., : self.n, self.n :]

    def group_exp(self, tangent_at_identity: Array) -> Array:
        """Lie-group exponential of a homogeneous Lie-algebra matrix."""
        xi = jnp.asarray(tangent_at_identity)
        omega = _skew(self.rotation(xi))
        velocity = self.translation(xi)
        rotation = self._rotations.group_exp(omega)
        translation = jnp.einsum("...ij,...j->...i", self._left_jacobian(omega), velocity)
        return self.from_components(rotation, translation)

    def group_log(self, G: Array) -> Array:
        """Principal Lie-group logarithm, undefined at rotations by pi."""
        G = jnp.asarray(G)
        omega = self._rotations.group_log(self.rotation(G))
        jacobian = self._left_jacobian(omega)
        velocity = jnp.linalg.solve(jacobian, self.translation(G)[..., None])[..., 0]
        return self.tangent_from_components(omega, velocity)

    def apply(self, G: Array, points: Array) -> Array:
        return jnp.einsum(
            "...ij,...j->...i", self.rotation(G), jnp.asarray(points)
        ) + self.translation(G)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        key_rotation, key_translation = jax.random.split(key)
        rotation = self._rotations.random_point(key_rotation, sample_shape=sample_shape)
        translation = jax.random.normal(key_translation, shape=sample_shape + (self.n,))
        return self.from_components(rotation, translation)

    def random_tangent(
        self,
        key: Array,
        G: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        Z = jax.random.normal(key, shape=jnp.shape(G))
        U = self.tangent_project(G, Z)
        if normalize:
            norm = self.norm(G, U)[..., None, None]
            U = jnp.where(norm > self.eps, U / norm, U)
        return scale * U


__all__ = ["SpecialOrthogonal", "SpecialEuclidean"]
