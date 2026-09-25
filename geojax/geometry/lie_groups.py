"""Matrix geometries for rotations and rigid transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import (
    ExactGeometryMixin,
    as_sample_shape,
    check_event_shape,
    validate_integer,
    validate_nonnegative,
    validate_positive,
)
from ._numerics import matrix_expm, stable_metric_norm, stable_norm, sqrt_nonnegative
from .shape import _proper_procrustes

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
    the principal matrix logarithm. The custom JVP below supplies its group
    differential without differentiating eigenvectors.
    """
    M = jnp.asarray(M)
    Mc = M.astype(_complex_dtype(M.dtype))
    eigvals, eigvecs = jnp.linalg.eig(Mc)
    log_eigvals = 1j * jnp.angle(eigvals)
    eigvecs_inv = jnp.linalg.inv(eigvecs)
    result = (eigvecs * log_eigvals[..., None, :]) @ eigvecs_inv
    return _skew(jnp.real(result)).astype(M.dtype)


def _orthogonal_log_frechet(M: Array, E: Array) -> Array:
    """Spectral inverse of ``D exp(log(M))``, used only by an implicit solve.

    ``custom_linear_solve`` differentiates the defining linear equation, not
    this eigenbasis calculation.  Its repeated-eigenvalue limit is 1/lambda;
    no differentiation of nonsymmetric eigenvectors is needed at any order.
    """
    M = jnp.asarray(M)
    complex_dtype = _complex_dtype(M.dtype)
    eigvals, eigvecs = jnp.linalg.eig(M.astype(complex_dtype))
    log_eigvals = 1j * jnp.angle(eigvals)
    eig_i = eigvals[..., :, None]
    eig_j = eigvals[..., None, :]
    log_i = log_eigvals[..., :, None]
    log_j = log_eigvals[..., None, :]
    denominator = eig_i - eig_j
    separated = jnp.abs(denominator) > 32.0 * jnp.finfo(M.dtype).eps
    safe_denominator = jnp.where(separated, denominator, jnp.ones_like(denominator))
    divided_difference = jnp.where(separated, (log_i - log_j) / safe_denominator, 1.0 / eig_i)
    eigvecs_inv = jnp.linalg.inv(eigvecs)
    rotated_E = eigvecs_inv @ jnp.asarray(E).astype(complex_dtype) @ eigvecs
    result = eigvecs @ (divided_difference * rotated_E) @ eigvecs_inv
    return jnp.real(result).astype(M.dtype)


@_principal_orthogonal_log.defjvp
def _principal_orthogonal_log_jvp(primals, tangents):
    (M,), (E,) = primals, tangents
    logarithm = _principal_orthogonal_log(M)

    rotation = matrix_expm(logarithm)

    def operator(H):
        derivative = jax.jvp(matrix_expm, (logarithm,), (_skew(H),))[1]
        return _skew(_transpose(rotation) @ derivative) + _sym(H)

    def solve_with_rotation(operator, rhs, point):
        def spectral(B):
            return _skew(_orthogonal_log_frechet(point, point @ _skew(B))) + _sym(B)

        solution = spectral(rhs)
        # The ambient spectral inverse can amplify roundoff in symmetric
        # modes near a pi rotation even when its skew restriction is regular.
        # Refine against the actual tangent operator to remove those errors.
        for _ in range(2):
            solution = solution + spectral(rhs - operator(solution))
        return solution

    # Invert the left-trivialized differential on skew matrices. The identity
    # on the symmetric complement makes this a genuine full-space inverse for
    # implicit AD, without introducing the artificial ambient singularity at
    # a single pi rotation. The adjoint is the same operator at -logarithm.
    derivative = jax.lax.custom_linear_solve(
        operator,
        _skew(_transpose(rotation) @ E),
        solve=lambda op, rhs: solve_with_rotation(op, rhs, rotation),
        transpose_solve=lambda op, rhs: solve_with_rotation(op, rhs, _transpose(rotation)),
    )
    return logarithm, _skew(derivative)


def _orthogonal_cut_mask(M: Array) -> Array:
    """Detect the principal-log boundary without differentiating the test."""
    M = jax.lax.stop_gradient(M)
    identity = jnp.eye(M.shape[-1], dtype=M.dtype)
    distance = jnp.min(jnp.linalg.svd(M + identity, compute_uv=False), axis=-1)
    return distance <= 32.0 * M.shape[-1] * jnp.finfo(M.dtype).eps


@jax.custom_jvp
def _orthogonal_squared_distance(M: Array) -> Array:
    """Squared principal eigenangle norm, including its finite cut value."""
    logarithm = _principal_orthogonal_log(M)
    local_value = jnp.sum(logarithm * logarithm, axis=(-2, -1))
    eigenvalues = jnp.linalg.eigvals(M.astype(_complex_dtype(M.dtype)))
    cut_value = jnp.sum(jnp.angle(eigenvalues) ** 2, axis=-1).astype(M.dtype)
    return jnp.where(_orthogonal_cut_mask(M), cut_value, local_value)


@_orthogonal_squared_distance.defjvp
def _orthogonal_squared_distance_jvp(primals, tangents):
    (M,), (E,) = primals, tangents
    value = _orthogonal_squared_distance(M)
    logarithm = _principal_orthogonal_log(M)
    # At an orthogonal M, D ||log(M)||_F^2[E] = 2 <M log(M), E>.
    # This first variation also removes the inactive eigenangle fallback from
    # differentiated objectives. At a genuine cut, the distance remains finite
    # but there is no unique derivative, which is explicitly marked NaN.
    gradient = 2.0 * (M @ logarithm)
    gradient = jnp.where(_orthogonal_cut_mask(M)[..., None, None], jnp.nan, gradient)
    return value, jnp.sum(gradient * E, axis=(-2, -1))


@dataclass(frozen=True, init=False)
class SpecialOrthogonal(ExactGeometryMixin):
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
        size = validate_integer(size, name="SpecialOrthogonal size", minimum=2)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="SpecialOrthogonal atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="SpecialOrthogonal eps"))

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
        if not self._shape_matches(R):
            return self._shape_failure(R)
        identity = jnp.eye(self.n, dtype=R.dtype)
        roundoff = (
            10.0
            * self.n
            * jnp.finfo(jnp.result_type(R, float)).eps
            * jnp.maximum(jnp.linalg.norm(R, axis=(-2, -1)), 1.0)
        )
        orthogonal = jnp.linalg.norm(_transpose(R) @ R - identity, axis=(-2, -1)) <= tol + roundoff
        orientation = jnp.abs(jnp.linalg.det(R) - 1.0) <= tol + roundoff
        return orthogonal & orientation

    def project(self, A: Array) -> Array:
        """Nearest proper orthogonal factor, differentiable at unique optima.

        Repeated positive singular values and regular rank-(n-1) inputs are
        supported. On an orientation-reversing input, a tie in the smallest
        singular value makes the optimum nonunique and its derivative
        undefined; the primal still returns a selected closest rotation.
        """
        A = self._check_shape(A, name="A")
        # This is the same unique orientation-preserving polar problem as
        # shape alignment. Its implicit derivative survives repeated singular
        # values; raw singular-vector derivatives do not, even at identity.
        return _proper_procrustes(A)

    def is_tangent(self, R: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(R, U):
            return self._shape_failure(R)
        R, U = self._check_shapes(("R", R), ("U", U))
        body = _transpose(R) @ U
        return jnp.linalg.norm(body + _transpose(body), axis=(-2, -1)) <= tol

    def tangent_project(self, R: Array, A: Array) -> Array:
        R, A = self._check_shapes(("R", R), ("A", A))
        return R @ _skew(_transpose(R) @ A)

    def inner(self, R: Array, U: Array, V: Array) -> Array:
        _, U, V = self._check_shapes(("R", R), ("U", U), ("V", V))
        return jnp.sum(U * V, axis=(-2, -1))

    def norm(self, R: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(R, normalized, normalized),
            axis=(-2, -1),
        )

    def _relative_log(self, relative: Array) -> Array:
        relative = jnp.asarray(relative)
        log_relative = _principal_orthogonal_log(relative)
        at_cut = _orthogonal_cut_mask(relative)
        return jnp.where(at_cut[..., None, None], jnp.nan, log_relative)

    def exp(self, R: Array, U: Array) -> Array:
        R = self._check_shape(R, name="R")
        U = self.tangent_project(R, U)
        omega = _skew(_transpose(R) @ U)
        return R @ matrix_expm(omega)

    def retr(self, R: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(R, self._scale_tangent(U, t))

    def log(self, R: Array, Q: Array) -> Array:
        # Do not insert an SVD projection here: its repeated singular values on
        # SO(n) make an otherwise smooth distance nondifferentiable to JAX.
        R, Q = self._check_shapes(("R", R), ("Q", Q))
        return R @ self._relative_log(_transpose(R) @ Q)

    def dist(self, R: Array, Q: Array) -> Array:
        return sqrt_nonnegative(self.squared_dist(R, Q))

    def squared_dist(self, R: Array, Q: Array) -> Array:
        R, Q = self._check_shapes(("R", R), ("Q", Q))
        relative = _transpose(R) @ Q
        return _orthogonal_squared_distance(relative)

    def transport(self, R: Array, Q: Array, U: Array) -> Array:
        """Parallel transport along the selected shortest geodesic."""
        R, Q = self._check_shapes(("R", R), ("Q", Q))
        U = self.tangent_project(R, U)
        omega = self._relative_log(_transpose(R) @ Q)
        half = matrix_expm(0.5 * omega)
        body = _skew(_transpose(R) @ U)
        transported = R @ half @ body @ half
        return self.tangent_project(Q, transported)

    def egrad_to_rgrad(self, R: Array, egrad: Array) -> Array:
        return self.tangent_project(R, egrad)

    def ehess_to_rhess(self, R: Array, egrad: Array, ehess_vec: Array, U: Array) -> Array:
        correction = jnp.asarray(U) @ _sym(_transpose(R) @ jnp.asarray(egrad))
        return self.tangent_project(R, jnp.asarray(ehess_vec) - correction)

    def compose(self, R: Array, Q: Array) -> Array:
        R, Q = self._check_shapes(("R", R), ("Q", Q))
        return R @ Q

    def inverse(self, R: Array) -> Array:
        return _transpose(self._check_shape(R, name="R"))

    def group_exp(self, omega: Array) -> Array:
        """Lie-group exponential from a skew matrix at the identity."""
        return matrix_expm(_skew(self._check_shape(omega, name="omega")))

    def group_log(self, R: Array) -> Array:
        """Principal Lie-group logarithm, undefined at rotations by pi."""
        return self._relative_log(self._check_shape(R, name="R"))

    def apply(self, R: Array, points: Array) -> Array:
        R = self._check_shape(R, name="R")
        points = check_event_shape(points, (self.n,), name="points")
        return jnp.einsum("...ij,...j->...i", R, points)

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
            safe_norm = jnp.where(norm > 0.0, norm, jnp.ones_like(norm))
            U = jnp.where(norm > 0.0, U / safe_norm, U)
        return self._scale_tangent(U, scale)


@dataclass(frozen=True, init=False)
class SpecialEuclidean(ExactGeometryMixin):
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
        size = validate_integer(size, name="SpecialEuclidean size", minimum=2)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", validate_nonnegative(atol, name="SpecialEuclidean atol"))
        object.__setattr__(self, "eps", validate_positive(eps, name="SpecialEuclidean eps"))

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
        rotation = check_event_shape(rotation, (self.n, self.n), name="rotation")
        translation = check_event_shape(translation, (self.n,), name="translation")
        batch_shape = jnp.broadcast_shapes(rotation.shape[:-2], translation.shape[:-1])
        rotation = jnp.broadcast_to(rotation, batch_shape + (self.n, self.n))
        translation = jnp.broadcast_to(translation, batch_shape + (self.n,))
        dtype = jnp.result_type(rotation, translation, float)
        rotation = rotation.astype(dtype)
        translation = translation.astype(dtype)
        out = jnp.zeros(rotation.shape[:-2] + self.shape, dtype=dtype)
        out = out.at[..., : self.n, : self.n].set(rotation)
        out = out.at[..., : self.n, self.n].set(translation)
        return out.at[..., self.n, self.n].set(1.0)

    def tangent_from_components(self, rotation: Array, translation: Array) -> Array:
        rotation = check_event_shape(rotation, (self.n, self.n), name="rotation")
        translation = check_event_shape(translation, (self.n,), name="translation")
        batch_shape = jnp.broadcast_shapes(rotation.shape[:-2], translation.shape[:-1])
        rotation = jnp.broadcast_to(rotation, batch_shape + (self.n, self.n))
        translation = jnp.broadcast_to(translation, batch_shape + (self.n,))
        dtype = jnp.result_type(rotation, translation, float)
        rotation = rotation.astype(dtype)
        translation = translation.astype(dtype)
        out = jnp.zeros(rotation.shape[:-2] + self.shape, dtype=dtype)
        out = out.at[..., : self.n, : self.n].set(rotation)
        return out.at[..., : self.n, self.n].set(translation)

    def rotation(self, G: Array) -> Array:
        return self._check_shape(G, name="G")[..., : self.n, : self.n]

    def translation(self, G: Array) -> Array:
        return self._check_shape(G, name="G")[..., : self.n, self.n]

    def belongs(self, G: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        G = jnp.asarray(G)
        if not self._shape_matches(G):
            return self._shape_failure(G)
        rotation_ok = self._rotations.belongs(self.rotation(G), atol=tol)
        expected_bottom = jnp.zeros(G.shape[:-2] + (self.n + 1,), dtype=G.dtype)
        expected_bottom = expected_bottom.at[..., -1].set(1.0)
        bottom_ok = jnp.linalg.norm(G[..., self.n, :] - expected_bottom, axis=-1) <= tol
        return rotation_ok & bottom_ok

    def project(self, A: Array) -> Array:
        A = self._check_shape(A, name="A")
        return self.from_components(self._rotations.project(self.rotation(A)), self.translation(A))

    def is_tangent(self, G: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        if not self._shape_matches(G, U):
            return self._shape_failure(G)
        G, U = self._check_shapes(("G", G), ("U", U))
        rotation_ok = self._rotations.is_tangent(self.rotation(G), self.rotation(U), atol=tol)
        bottom_ok = jnp.linalg.norm(U[..., self.n, :], axis=-1) <= tol
        return rotation_ok & bottom_ok

    def tangent_project(self, G: Array, A: Array) -> Array:
        G, A = self._check_shapes(("G", G), ("A", A))
        return self.tangent_from_components(
            self._rotations.tangent_project(self.rotation(G), self.rotation(A)),
            self.translation(A),
        )

    def inner(self, G: Array, U: Array, V: Array) -> Array:
        G, U, V = self._check_shapes(("G", G), ("U", U), ("V", V))
        rotation_inner = self._rotations.inner(self.rotation(G), self.rotation(U), self.rotation(V))
        translation_inner = jnp.sum(self.translation(U) * self.translation(V), axis=-1)
        return rotation_inner + translation_inner

    def norm(self, G: Array, U: Array) -> Array:
        return stable_metric_norm(
            U,
            lambda normalized: self.inner(G, normalized, normalized),
            axis=(-2, -1),
        )

    def exp(self, G: Array, U: Array) -> Array:
        G = self._check_shape(G, name="G")
        U = self.tangent_project(G, U)
        R = self.rotation(G)
        next_R = self._rotations.exp(R, self.rotation(U))
        next_t = self.translation(G) + self.translation(U)
        return self.from_components(next_R, next_t)

    def retr(self, G: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(G, self._scale_tangent(U, t))

    def log(self, G: Array, H: Array) -> Array:
        G, H = self._check_shapes(("G", G), ("H", H))
        return self.tangent_from_components(
            self._rotations.log(self.rotation(G), self.rotation(H)),
            self.translation(H) - self.translation(G),
        )

    def dist(self, G: Array, H: Array) -> Array:
        rotation_distance = self._rotations.dist(self.rotation(G), self.rotation(H))
        translation_distance = stable_norm(
            self.translation(H) - self.translation(G),
            axis=-1,
        )
        return stable_norm(
            jnp.stack(jnp.broadcast_arrays(rotation_distance, translation_distance), axis=-1),
            axis=-1,
        )

    def squared_dist(self, G: Array, H: Array) -> Array:
        rotation_dist_sq = self._rotations.squared_dist(self.rotation(G), self.rotation(H))
        translation_dist_sq = jnp.sum((self.translation(H) - self.translation(G)) ** 2, axis=-1)
        return rotation_dist_sq + translation_dist_sq

    def transport(self, G: Array, H: Array, U: Array) -> Array:
        G = self.project(G)
        H = self.project(H)
        U = self.tangent_project(G, U)
        return self.tangent_from_components(
            self._rotations.transport(self.rotation(G), self.rotation(H), self.rotation(U)),
            self.translation(U),
        )

    def egrad_to_rgrad(self, G: Array, egrad: Array) -> Array:
        return self.tangent_project(G, egrad)

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
        points = check_event_shape(points, (self.n,), name="points")
        return jnp.einsum("...ij,...j->...i", self.rotation(G), points) + self.translation(G)

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
            safe_norm = jnp.where(norm > 0.0, norm, jnp.ones_like(norm))
            U = jnp.where(norm > 0.0, U / safe_norm, U)
        return self._scale_tangent(U, scale)


__all__ = ["SpecialOrthogonal", "SpecialEuclidean"]
