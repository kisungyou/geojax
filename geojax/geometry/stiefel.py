"""Canonical and Euclidean geometries on the real Stiefel manifold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from .base import GeometryMixin, Shape, as_sample_shape

Array = Any


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _sym(A: Array) -> Array:
    return 0.5 * (A + _transpose(A))


def _skew(A: Array) -> Array:
    return 0.5 * (A - _transpose(A))


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(jnp.asarray(A) * jnp.asarray(B), axis=(-2, -1))


def _matrix_expm(A: Array) -> Array:
    """Apply the matrix exponential over optional leading batch axes."""
    A = jnp.asarray(A)
    if A.ndim == 2:
        return jsp_linalg.expm(A)
    shape = A.shape
    flat = A.reshape((-1, shape[-2], shape[-1]))
    return jax.vmap(jsp_linalg.expm)(flat).reshape(shape)


def _parse_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError("Stiefel size must be a pair (ambient_dim, frame_size).")
    parsed = tuple(int(value) for value in size)
    if len(parsed) != 2:
        raise ValueError("Stiefel size must be a pair (ambient_dim, frame_size).")
    n, k = parsed
    if n < 1 or k < 1:
        raise ValueError("Stiefel dimensions must be positive.")
    if k > n:
        raise ValueError("Stiefel frame_size must satisfy frame_size <= ambient_dim.")
    return n, k


class StiefelLogInfo(NamedTuple):
    """Convergence data returned by ``Stiefel.log_with_info``.

    ``log`` returns nonfinite values when ``converged`` is false. Use
    ``log_with_info`` to inspect the best shooting iterate and these diagnostics.
    """

    converged: Array
    iterations: Array
    residual_norm: Array
    step_norm: Array


@dataclass(frozen=True, init=False)
class _StiefelBase(GeometryMixin):
    """Shared frame representation and numerical logarithm implementation."""

    transport_is_parallel = False

    size: tuple[int, int]
    atol: float
    eps: float
    log_maxiter: int
    log_tol: float
    log_damping: float

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        atol: float = 1e-6,
        eps: float = 1e-12,
        log_maxiter: int = 32,
        log_tol: float = 1e-9,
        log_damping: float = 1e-6,
    ) -> None:
        parsed = _parse_size(size)
        if log_maxiter < 1:
            raise ValueError("log_maxiter must be positive.")
        if log_tol <= 0.0 or log_damping <= 0.0:
            raise ValueError("log_tol and log_damping must be positive.")
        object.__setattr__(self, "size", parsed)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))
        object.__setattr__(self, "log_maxiter", int(log_maxiter))
        object.__setattr__(self, "log_tol", float(log_tol))
        object.__setattr__(self, "log_damping", float(log_damping))

    @property
    def n(self) -> int:
        """Ambient dimension."""
        return self.size[0]

    @property
    def k(self) -> int:
        """Number of orthonormal frame columns."""
        return self.size[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    @property
    def dim(self) -> int:
        return self.n * self.k - self.k * (self.k + 1) // 2

    @property
    def _vertical_basis_scale(self) -> float:
        raise NotImplementedError

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        X = jnp.asarray(X)
        identity = jnp.eye(self.k, dtype=X.dtype)
        return jnp.linalg.norm(_transpose(X) @ X - identity, axis=(-2, -1)) <= tol

    def project(self, A: Array) -> Array:
        """Return the nearest orthonormal frame in Frobenius norm."""
        A = jnp.asarray(A)
        U, _, Vh = jnp.linalg.svd(A, full_matrices=False)
        return U @ Vh

    normalize = project

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        constraint = _transpose(jnp.asarray(X)) @ jnp.asarray(U)
        return jnp.linalg.norm(constraint + _transpose(constraint), axis=(-2, -1)) <= tol

    def tangent_project(self, X: Array, A: Array) -> Array:
        """Orthogonally project an ambient matrix onto the frame tangent space."""
        X = jnp.asarray(X)
        A = jnp.asarray(A)
        return A - X @ _sym(_transpose(X) @ A)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def norm(self, X: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(X, U, U), 0.0))

    def _orthogonal_complement(self, X: Array) -> Array:
        if self.n == self.k:
            return jnp.zeros(X.shape[:-2] + (self.n, 0), dtype=X.dtype)
        full_Q, _ = jnp.linalg.qr(X, mode="complete")
        return full_Q[..., :, self.k :]

    def _tangent_basis(self, X: Array) -> Array:
        """Construct a metric-orthonormal basis of ``T_X St(n, k)``."""
        X = jnp.asarray(X)
        vertical = []
        for i in range(self.k):
            for j in range(i + 1, self.k):
                generator = jnp.zeros((self.k, self.k), dtype=X.dtype)
                generator = generator.at[i, j].set(self._vertical_basis_scale)
                generator = generator.at[j, i].set(-self._vertical_basis_scale)
                vertical.append(X @ generator)

        complement = self._orthogonal_complement(X)
        identity = jnp.eye(self.k, dtype=X.dtype)
        horizontal = jnp.einsum("ia,bj->abij", complement, identity).reshape(
            ((self.n - self.k) * self.k, self.n, self.k)
        )
        if vertical:
            return jnp.concatenate([jnp.stack(vertical), horizontal], axis=0)
        return horizontal

    def _coordinates(self, X: Array, basis: Array, U: Array) -> Array:
        return jax.vmap(lambda vector: self.inner(X, vector, U))(basis)

    def _shoot_log(self, X: Array, Y: Array) -> tuple[Array, StiefelLogInfo]:
        """Solve ``Exp_X(U) = Y`` by damped Gauss-Newton shooting."""
        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
        if self.dim == 0:
            tangent = jnp.zeros_like(X)
            residual = jnp.linalg.norm(X - Y)
            info = StiefelLogInfo(
                residual <= self.log_tol,
                jnp.asarray(0, dtype=jnp.int32),
                residual,
                jnp.asarray(0.0, dtype=X.dtype),
            )
            return tangent, info

        basis = self._tangent_basis(X)
        initial = self.tangent_project(X, Y - X)
        z0 = self._coordinates(X, basis, initial)
        dtype = X.dtype
        tolerance = jnp.maximum(
            jnp.asarray(self.log_tol, dtype=dtype),
            50.0 * jnp.finfo(dtype).eps,
        )

        def tangent_from_coordinates(z: Array) -> Array:
            return jnp.tensordot(z, basis, axes=1)

        def residual(z: Array) -> Array:
            return (self.exp(X, tangent_from_coordinates(z)) - Y).reshape(-1)

        initial_residual = residual(z0)
        initial_converged = jnp.linalg.norm(initial_residual) <= tolerance
        state = (
            z0,
            jnp.asarray(self.log_damping, dtype=dtype),
            initial_converged,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=dtype),
        )
        identity = jnp.eye(self.dim, dtype=dtype)
        alphas = jnp.asarray([1.0, 0.5, 0.25, 0.125, 0.0], dtype=dtype)

        def body(index: int, current: tuple[Array, Array, Array, Array, Array]):
            z, damping, converged, iterations, previous_step_norm = current
            value = residual(z)
            cost = jnp.vdot(value, value).real
            jacobian = jax.jacfwd(residual)(z)
            normal = _transpose(jacobian) @ jacobian + damping * identity
            step = jnp.linalg.solve(normal, -_transpose(jacobian) @ value)
            step_norm = jnp.linalg.norm(step)
            step = step * jnp.minimum(1.0, jnp.pi / jnp.maximum(step_norm, self.eps))

            candidates = z[None, :] + alphas[:, None] * step[None, :]
            candidate_values = jax.vmap(residual)(candidates)
            candidate_costs = jnp.sum(candidate_values * candidate_values, axis=-1)
            best_index = jnp.argmin(candidate_costs)
            best_z = candidates[best_index]
            best_cost = candidate_costs[best_index]
            improved = best_cost < cost
            active = ~converged
            accepted = active & improved
            next_z = jnp.where(accepted, best_z, z)
            next_damping = jnp.where(
                active,
                jnp.where(improved, jnp.maximum(0.5 * damping, self.eps), 10.0 * damping),
                damping,
            )
            next_residual_norm = jnp.linalg.norm(residual(next_z))
            next_converged = converged | (active & (next_residual_norm <= tolerance))
            next_iterations = jnp.where(active, index + 1, iterations)
            accepted_step_norm = jnp.where(
                accepted, alphas[best_index] * step_norm, previous_step_norm
            )
            return (
                next_z,
                next_damping,
                next_converged,
                next_iterations,
                accepted_step_norm,
            )

        z, _, converged, iterations, step_norm = jax.lax.fori_loop(0, self.log_maxiter, body, state)
        tangent = self.tangent_project(X, tangent_from_coordinates(z))
        residual_norm = jnp.linalg.norm(residual(z))
        return tangent, StiefelLogInfo(converged, iterations, residual_norm, step_norm)

    def log_with_info(self, X: Array, Y: Array) -> tuple[Array, StiefelLogInfo]:
        """Return a numerical logarithm and endpoint-shooting diagnostics.

        The tangent result is the best iterate even if the solver does not
        converge. Check ``info.converged`` before using it as a logarithm.
        """
        return self._shoot_log(jnp.asarray(X), jnp.asarray(Y))

    def log(self, X: Array, Y: Array) -> Array:
        """Return the selected local logarithm, or nonfinite values on failure."""
        tangent, info = self.log_with_info(X, Y)
        return jnp.where(info.converged, tangent, jnp.full_like(tangent, jnp.nan))

    def dist(self, X: Array, Y: Array) -> Array:
        return self.norm(X, self.log(X, Y))

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        """Apply an isometric group-action vector transport from ``X`` to ``Y``.

        This transport is tangent and exactly metric-preserving, but it is not
        the Levi-Civita parallel transport for either Stiefel metric.
        """
        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
        U = self.tangent_project(X, U)
        frame_X = jnp.concatenate([X, self._orthogonal_complement(X)], axis=-1)
        frame_Y = jnp.concatenate([Y, self._orthogonal_complement(Y)], axis=-1)
        action = frame_Y @ _transpose(frame_X)
        return self.tangent_project(Y, action @ U)

    transp = transport

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = as_sample_shape(sample_shape)
        normal = jax.random.normal(key, shape=sample_shape + self.shape)
        frame, upper = jnp.linalg.qr(normal, mode="reduced")
        diagonal = jnp.diagonal(upper, axis1=-2, axis2=-1)
        signs = jnp.where(diagonal < 0.0, -1.0, 1.0)
        return frame * signs[..., None, :]

    def random_tangent(
        self,
        key: Array,
        X: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        normal = jax.random.normal(key, shape=jnp.shape(X))
        tangent = self.tangent_project(X, normal)
        if normalize:
            tangent_norm = self.norm(X, tangent)[..., None, None]
            tangent = jnp.where(tangent_norm > self.eps, tangent / tangent_norm, tangent)
        return scale * tangent


class Stiefel(_StiefelBase):
    """Stiefel manifold with the canonical quotient metric.

    A point is an ``n x k`` orthonormal frame. The metric is the quotient metric
    induced by ``O(n) -> O(n) / O(n-k)``:

    ``g_X(U, V) = trace(U.T @ (I - 0.5 * X @ X.T) @ V)``.

    The exponential map is exact. The logarithm is computed by differentiable
    damped endpoint shooting; use :meth:`log_with_info` to inspect convergence.
    """

    @property
    def _vertical_basis_scale(self) -> float:
        return 1.0

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        X = jnp.asarray(X)
        U = jnp.asarray(U)
        V = jnp.asarray(V)
        return _trace_inner(U, V) - 0.5 * _trace_inner(_transpose(X) @ U, _transpose(X) @ V)

    def exp(self, X: Array, U: Array) -> Array:
        """Evaluate the exact canonical-metric exponential map."""
        X = jnp.asarray(X)
        U = self.tangent_project(X, U)
        vertical = _skew(_transpose(X) @ U)
        horizontal = U - X @ vertical
        generator = (
            horizontal @ _transpose(X) - X @ _transpose(horizontal) + X @ vertical @ _transpose(X)
        )
        return _matrix_expm(_skew(generator)) @ X

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        """Convert an ambient Euclidean gradient for the canonical metric."""
        X = jnp.asarray(X)
        egrad = jnp.asarray(egrad)
        return egrad - X @ _transpose(egrad) @ X

    egrad2rgrad = egrad_to_rgrad


class StiefelEuclidean(_StiefelBase):
    """Stiefel manifold with the metric induced by its Euclidean embedding.

    Points and tangents use the same ``n x k`` frame representation as
    :class:`Stiefel`, but ``g_X(U, V) = trace(U.T @ V)``. Its geodesics therefore
    differ whenever the tangent has a component ``X @ A`` with ``A`` skew.
    """

    @property
    def _vertical_basis_scale(self) -> float:
        return 1.0 / jnp.sqrt(2.0)

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        del X
        return _trace_inner(U, V)

    def exp(self, X: Array, U: Array) -> Array:
        """Evaluate the exact embedded-Euclidean exponential map."""
        X = jnp.asarray(X)
        U = self.tangent_project(X, U)
        A = _skew(_transpose(X) @ U)
        S = _sym(_transpose(U) @ U)
        identity = jnp.broadcast_to(jnp.eye(self.k, dtype=X.dtype), A.shape)
        top = jnp.concatenate([A, -S], axis=-1)
        bottom = jnp.concatenate([identity, A], axis=-1)
        block = jnp.concatenate([top, bottom], axis=-2)
        initial = _matrix_expm(block)[..., :, : self.k]
        return jnp.concatenate([X, U], axis=-1) @ initial @ _matrix_expm(-A)

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        return self.tangent_project(X, egrad)

    egrad2rgrad = egrad_to_rgrad

    def ehess_to_rhess(self, X: Array, egrad: Array, ehess_vec: Array, U: Array) -> Array:
        correction = jnp.asarray(U) @ _sym(_transpose(X) @ jnp.asarray(egrad))
        return self.tangent_project(X, jnp.asarray(ehess_vec) - correction)


__all__ = ["Stiefel", "StiefelEuclidean", "StiefelLogInfo"]
