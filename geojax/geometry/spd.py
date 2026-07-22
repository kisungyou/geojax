"""Symmetric positive-definite matrix geometries in JAX.

This module contains three Riemannian geometries on the same set

    SPD(n) = {P in Sym(n): P is positive definite}.

``SPDLogEuclidean`` uses the log-Euclidean metric, under which the matrix
logarithm is an isometry from SPD(n) to the vector space Sym(n).

``SPDAffineInvariant`` uses the affine-invariant metric

    g_P(U,V) = tr(P^{-1} U P^{-1} V).

``SPDBuresWasserstein`` uses the quotient metric induced by square-root
factors.  Its distance agrees with the 2-Wasserstein distance between
zero-mean Gaussian distributions.

Both classes expose the same small Manopt-style geometry interface used by the
optimization module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import GeometryMixin, as_sample_shape

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


def _as_sample_shape(sample_shape: Shape = ()) -> tuple[int, ...]:
    return as_sample_shape(sample_shape)


def _parse_spd_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError("SPD size must be square, e.g. size=(3, 3).")
    shape = tuple(int(v) for v in size)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("SPD size must be square, e.g. size=(3, 3).")
    if shape[0] < 1:
        raise ValueError("size must be positive for SPD geometry.")
    return shape


def _sym(A: Array) -> Array:
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(A * B, axis=(-2, -1))


def _matmul3(A: Array, B: Array, C: Array) -> Array:
    return A @ B @ C


def _eigh_sym(A: Array) -> tuple[Array, Array]:
    return jnp.linalg.eigh(_sym(A))


def _spd_from_eigh(Q: Array, vals: Array) -> Array:
    return (Q * vals[..., None, :]) @ jnp.swapaxes(Q, -1, -2)


def _spd_apply_eigfunc(A: Array, func, *, min_eig: float | None = None) -> Array:
    vals, Q = _eigh_sym(A)
    if min_eig is not None:
        vals = jnp.maximum(vals, min_eig)
    return _spd_from_eigh(Q, func(vals))


def _spd_expm(A: Array) -> Array:
    return _spd_apply_eigfunc(A, jnp.exp)


def _spd_logm(P: Array, eps: float) -> Array:
    return _spd_apply_eigfunc(P, jnp.log, min_eig=eps)


def _spd_sqrtm(P: Array, eps: float) -> Array:
    return _spd_apply_eigfunc(P, jnp.sqrt, min_eig=eps)


def _spd_invsqrtm(P: Array, eps: float) -> Array:
    return _spd_apply_eigfunc(P, lambda x: 1.0 / jnp.sqrt(x), min_eig=eps)


def _spd_invm(P: Array, eps: float) -> Array:
    return _spd_apply_eigfunc(P, lambda x: 1.0 / x, min_eig=eps)


@jax.custom_jvp
def _spd_sqrtm_differentiable(P: Array) -> Array:
    """Principal SPD square root with a degeneracy-safe custom derivative."""
    P = _sym(jnp.asarray(P))
    vals, eigvecs = _eigh_sym(P)
    roots = jnp.sqrt(jnp.maximum(vals, jnp.finfo(P.dtype).tiny))
    return _spd_from_eigh(eigvecs, roots)


@_spd_sqrtm_differentiable.defjvp
def _spd_sqrtm_differentiable_jvp(primals, tangents):
    (P,), (E,) = primals, tangents
    P = _sym(jnp.asarray(P))
    E = _sym(jnp.asarray(E))
    vals, eigvecs = _eigh_sym(P)
    roots = jnp.sqrt(jnp.maximum(vals, jnp.finfo(P.dtype).tiny))
    rotated = jnp.swapaxes(eigvecs, -1, -2) @ E @ eigvecs
    denominator = roots[..., :, None] + roots[..., None, :]
    derivative = eigvecs @ (rotated / denominator) @ jnp.swapaxes(eigvecs, -1, -2)
    return _spd_from_eigh(eigvecs, roots), _sym(derivative)


@jax.custom_jvp
def _spd_invsqrtm_differentiable(P: Array) -> Array:
    """Principal inverse SPD square root with a stable custom derivative."""
    P = _sym(jnp.asarray(P))
    vals, eigvecs = _eigh_sym(P)
    roots = jnp.sqrt(jnp.maximum(vals, jnp.finfo(P.dtype).tiny))
    return _spd_from_eigh(eigvecs, 1.0 / roots)


@_spd_invsqrtm_differentiable.defjvp
def _spd_invsqrtm_differentiable_jvp(primals, tangents):
    (P,), (E,) = primals, tangents
    P = _sym(jnp.asarray(P))
    E = _sym(jnp.asarray(E))
    vals, eigvecs = _eigh_sym(P)
    roots = jnp.sqrt(jnp.maximum(vals, jnp.finfo(P.dtype).tiny))
    rotated = jnp.swapaxes(eigvecs, -1, -2) @ E @ eigvecs
    denominator = (
        roots[..., :, None] * roots[..., None, :] * (roots[..., :, None] + roots[..., None, :])
    )
    derivative = eigvecs @ (-rotated / denominator) @ jnp.swapaxes(eigvecs, -1, -2)
    return _spd_from_eigh(eigvecs, 1.0 / roots), _sym(derivative)


def _frechet_spectral(A: Array, E: Array, func_name: str, eps: float) -> Array:
    """Frechet derivative of exp or log at a symmetric matrix A.

    For A = Q diag(lambda) Q^T and symmetric E,

        Df_A[E] = Q (L_f(lambda) * (Q^T E Q)) Q^T,

    where L_f is the divided-difference matrix.
    """
    A = _sym(A)
    E = _sym(E)
    vals, Q = _eigh_sym(A)
    Et = jnp.swapaxes(Q, -1, -2) @ E @ Q
    lam_i = vals[..., :, None]
    lam_j = vals[..., None, :]
    denom = lam_i - lam_j

    if func_name == "exp":
        f_i = jnp.exp(lam_i)
        f_j = jnp.exp(lam_j)
        diag_deriv = jnp.exp(lam_i)
    elif func_name == "log":
        safe_i = jnp.maximum(lam_i, eps)
        safe_j = jnp.maximum(lam_j, eps)
        f_i = jnp.log(safe_i)
        f_j = jnp.log(safe_j)
        diag_deriv = 1.0 / safe_i
    else:
        raise ValueError("func_name must be 'exp' or 'log'.")

    L = jnp.where(jnp.abs(denom) > eps, (f_i - f_j) / denom, diag_deriv)
    Ft = L * Et
    return _sym(Q @ Ft @ jnp.swapaxes(Q, -1, -2))


@dataclass(frozen=True, init=False)
class SPDLogEuclidean(GeometryMixin):
    """Log-Euclidean geometry on SPD(size).

    The logarithm map ``log: SPD(n) -> Sym(n)`` is an isometry.  Therefore
    distances and geodesics are Euclidean after applying the matrix logarithm.
    Points are represented as SPD matrices of shape ``(size, size)``.
    Tangent vectors are symmetric matrices of the same shape.
    """

    size: tuple[int, int]
    atol: float
    eps: float

    def __init__(
        self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10
    ) -> None:
        object.__setattr__(self, "size", _parse_spd_size(size))
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        P = jnp.asarray(P)
        sym_ok = jnp.linalg.norm(P - jnp.swapaxes(P, -1, -2), axis=(-2, -1)) <= tol
        vals = jnp.linalg.eigvalsh(_sym(P))
        pd_ok = jnp.min(vals, axis=-1) > tol
        return sym_ok & pd_ok

    def is_tangent(self, P: Array, U: Array, atol: float | None = None) -> Array:
        del P
        tol = self.atol if atol is None else atol
        U = jnp.asarray(U)
        return jnp.linalg.norm(U - jnp.swapaxes(U, -1, -2), axis=(-2, -1)) <= tol

    def project(self, P: Array) -> Array:
        P = _sym(jnp.asarray(P))
        vals, Q = _eigh_sym(P)
        vals = jnp.maximum(vals, self.eps)
        return _spd_from_eigh(Q, vals)

    def tangent_project(self, P: Array, U: Array) -> Array:
        del P
        return _sym(jnp.asarray(U))

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def logm(self, P: Array) -> Array:
        return _spd_logm(self.project(P), self.eps)

    def expm(self, A: Array) -> Array:
        return _spd_expm(_sym(A))

    def dlog(self, P: Array, U: Array) -> Array:
        return _frechet_spectral(self.project(P), self.tangent_project(P, U), "log", self.eps)

    def dexp(self, A: Array, W: Array) -> Array:
        return _frechet_spectral(_sym(A), _sym(W), "exp", self.eps)

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        dU = self.dlog(P, U)
        dV = self.dlog(P, V)
        return _trace_inner(dU, dV)

    def norm(self, P: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(P, U, U), 0.0))

    def exp(self, P: Array, U: Array) -> Array:
        P = self.project(P)
        U = self.tangent_project(P, U)
        A = self.logm(P)
        W = self.dlog(P, U)
        return self.expm(A + W)

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(P, t * U)

    def log(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        A = self.logm(P)
        B = self.logm(Q)
        return self.dexp(A, B - A)

    def dist(self, P: Array, Q: Array) -> Array:
        D = self.logm(self.project(Q)) - self.logm(self.project(P))
        return jnp.sqrt(jnp.maximum(_trace_inner(D, D), 0.0))

    def transport(self, P: Array, Q: Array, U: Array) -> Array:
        B = self.logm(self.project(Q))
        W = self.dlog(P, U)
        return self.dexp(B, W)

    transp = transport

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        P = self.project(P)
        E = self.tangent_project(P, egrad)
        A = self.logm(P)
        # Pull the covector to log coordinates and push the Euclidean gradient
        # back through the inverse chart.  Dexp_A is self-adjoint for symmetric A.
        grad_A = self.dexp(A, E)
        return self.dexp(A, grad_A)

    egrad2rgrad = egrad_to_rgrad

    def lincomb(self, P: Array, *terms: Any) -> Array:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = coeff * vec
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(P, out)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = _as_sample_shape(sample_shape)
        A = jax.random.normal(key, shape=sample_shape + self.shape)
        A = _sym(A) / jnp.sqrt(float(self.n))
        return self.expm(A)

    def random_tangent(
        self,
        key: Array,
        P: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        Z = jax.random.normal(key, shape=jnp.shape(P))
        U = self.tangent_project(P, Z)
        if normalize:
            n = self.norm(P, U)[..., None, None]
            U = jnp.where(n > self.eps, U / n, U)
        return scale * U


@dataclass(frozen=True, init=False)
class SPDAffineInvariant(GeometryMixin):
    """Affine-invariant geometry on SPD(size).

    The metric is

        g_P(U,V) = tr(P^{-1} U P^{-1} V).

    This is the canonical symmetric-space geometry of GL(n)/O(n).
    """

    size: tuple[int, int]
    atol: float
    eps: float

    def __init__(
        self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10
    ) -> None:
        object.__setattr__(self, "size", _parse_spd_size(size))
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        P = jnp.asarray(P)
        sym_ok = jnp.linalg.norm(P - jnp.swapaxes(P, -1, -2), axis=(-2, -1)) <= tol
        vals = jnp.linalg.eigvalsh(_sym(P))
        pd_ok = jnp.min(vals, axis=-1) > tol
        return sym_ok & pd_ok

    def is_tangent(self, P: Array, U: Array, atol: float | None = None) -> Array:
        del P
        tol = self.atol if atol is None else atol
        U = jnp.asarray(U)
        return jnp.linalg.norm(U - jnp.swapaxes(U, -1, -2), axis=(-2, -1)) <= tol

    def project(self, P: Array) -> Array:
        P = _sym(jnp.asarray(P))
        vals, Q = _eigh_sym(P)
        vals = jnp.maximum(vals, self.eps)
        return _spd_from_eigh(Q, vals)

    def tangent_project(self, P: Array, U: Array) -> Array:
        del P
        return _sym(jnp.asarray(U))

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        P = self.project(P)
        U = self.tangent_project(P, U)
        V = self.tangent_project(P, V)
        Pinv = _spd_invm(P, self.eps)
        return _trace_inner(Pinv @ U @ Pinv, V)

    def norm(self, P: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(P, U, U), 0.0))

    def exp(self, P: Array, U: Array) -> Array:
        P = self.project(P)
        U = self.tangent_project(P, U)
        Psqrt = _spd_sqrtm(P, self.eps)
        Pinvsqrt = _spd_invsqrtm(P, self.eps)
        A = Pinvsqrt @ U @ Pinvsqrt
        return self.project(Psqrt @ _spd_expm(A) @ Psqrt)

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(P, t * U)

    def log(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        Psqrt = _spd_sqrtm(P, self.eps)
        Pinvsqrt = _spd_invsqrtm(P, self.eps)
        A = Pinvsqrt @ Q @ Pinvsqrt
        return _sym(Psqrt @ _spd_logm(A, self.eps) @ Psqrt)

    def dist(self, P: Array, Q: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        Pinvsqrt = _spd_invsqrtm(P, self.eps)
        A = Pinvsqrt @ Q @ Pinvsqrt
        L = _spd_logm(A, self.eps)
        return jnp.sqrt(jnp.maximum(_trace_inner(L, L), 0.0))

    def transport(self, P: Array, Q: Array, U: Array) -> Array:
        P = self.project(P)
        Q = self.project(Q)
        U = self.tangent_project(P, U)
        Psqrt = _spd_sqrtm(P, self.eps)
        Pinvsqrt = _spd_invsqrtm(P, self.eps)
        A = Pinvsqrt @ Q @ Pinvsqrt
        Asqrt = _spd_sqrtm(A, self.eps)
        E = Psqrt @ Asqrt @ Pinvsqrt
        return _sym(E @ U @ jnp.swapaxes(E, -1, -2))

    transp = transport

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        P = self.project(P)
        E = self.tangent_project(P, egrad)
        return _sym(P @ E @ P)

    egrad2rgrad = egrad_to_rgrad

    def lincomb(self, P: Array, *terms: Any) -> Array:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = coeff * vec
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(P, out)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = _as_sample_shape(sample_shape)
        A = jax.random.normal(key, shape=sample_shape + self.shape)
        A = _sym(A) / jnp.sqrt(float(self.n))
        return _spd_expm(A)

    def random_tangent(
        self,
        key: Array,
        P: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        Z = jax.random.normal(key, shape=jnp.shape(P))
        U = self.tangent_project(P, Z)
        if normalize:
            n = self.norm(P, U)[..., None, None]
            U = jnp.where(n > self.eps, U / n, U)
        return scale * U


@dataclass(frozen=True, init=False)
class SPDBuresWasserstein(GeometryMixin):
    """Bures-Wasserstein geometry on SPD(size).

    If ``P = Q diag(d) Q.T`` and ``U_tilde = Q.T @ U @ Q``, the metric is

        g_P(U,V) = 1/2 sum_ij U_tilde_ij V_tilde_ij / (d_i + d_j).

    The exponential is defined only while its horizontal square-root lift is
    nonsingular.  ``transport`` is an exact isometric vector transport for
    optimization; it is not the Levi-Civita parallel transport, whose general
    evaluation requires integrating a differential equation.
    """

    transport_is_parallel = False

    size: tuple[int, int]
    atol: float
    eps: float

    def __init__(
        self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10
    ) -> None:
        object.__setattr__(self, "size", _parse_spd_size(size))
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    def belongs(self, P: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        P = jnp.asarray(P)
        sym_ok = jnp.linalg.norm(P - jnp.swapaxes(P, -1, -2), axis=(-2, -1)) <= tol
        vals = jnp.linalg.eigvalsh(_sym(P))
        pd_ok = jnp.min(vals, axis=-1) > tol
        return sym_ok & pd_ok

    def project(self, P: Array) -> Array:
        P = _sym(jnp.asarray(P))
        vals, eigvecs = _eigh_sym(P)
        return _spd_from_eigh(eigvecs, jnp.maximum(vals, self.eps))

    def is_tangent(self, P: Array, U: Array, atol: float | None = None) -> Array:
        del P
        tol = self.atol if atol is None else atol
        U = jnp.asarray(U)
        return jnp.linalg.norm(U - jnp.swapaxes(U, -1, -2), axis=(-2, -1)) <= tol

    def tangent_project(self, P: Array, U: Array) -> Array:
        del P
        return _sym(jnp.asarray(U))

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def sylvester(self, P: Array, U: Array) -> Array:
        """Solve ``P A + A P = U`` for symmetric ``A``."""
        P = _sym(jnp.asarray(P))
        U = self.tangent_project(P, U)
        identity = jnp.eye(self.n, dtype=P.dtype)
        operator = jnp.einsum("...ik,jl->...ijkl", P, identity)
        operator = operator + jnp.einsum("ik,...lj->...ijkl", identity, P)
        operator = operator.reshape(P.shape[:-2] + (self.n**2, self.n**2))
        right_hand_side = U.reshape(U.shape[:-2] + (self.n**2, 1))
        solution = jnp.linalg.solve(operator, right_hand_side)[..., 0]
        return _sym(solution.reshape(U.shape))

    def inner(self, P: Array, U: Array, V: Array) -> Array:
        P = _sym(jnp.asarray(P))
        V = self.tangent_project(P, V)
        return 0.5 * _trace_inner(self.sylvester(P, U), V)

    def norm(self, P: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(P, U, U), 0.0))

    def exp(self, P: Array, U: Array) -> Array:
        P = _sym(jnp.asarray(P))
        U = self.tangent_project(P, U)
        A = self.sylvester(P, U)
        # P + U + A P A = (I + A) P (I + A) on the valid branch.
        return _sym(P + U + A @ P @ A)

    def retr(self, P: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(P, t * U)

    def optimal_transport_map(self, P: Array, Q: Array) -> Array:
        """Return the optimal Gaussian transport map from covariance P to Q."""
        P = _sym(jnp.asarray(P))
        Q = _sym(jnp.asarray(Q))
        Psqrt = _spd_sqrtm_differentiable(P)
        Pinvsqrt = _spd_invsqrtm_differentiable(P)
        middle = _spd_sqrtm_differentiable(Psqrt @ Q @ Psqrt)
        return _sym(Pinvsqrt @ middle @ Pinvsqrt)

    def log(self, P: Array, Q: Array) -> Array:
        P = _sym(jnp.asarray(P))
        transport_map = self.optimal_transport_map(P, Q)
        displacement = transport_map - jnp.eye(self.n, dtype=P.dtype)
        return _sym(displacement @ P + P @ displacement)

    def squared_dist(self, P: Array, Q: Array) -> Array:
        P = _sym(jnp.asarray(P))
        Q = _sym(jnp.asarray(Q))
        Psqrt = _spd_sqrtm_differentiable(P)
        cross = _spd_sqrtm_differentiable(Psqrt @ Q @ Psqrt)
        value = jnp.trace(P, axis1=-2, axis2=-1)
        value = value + jnp.trace(Q, axis1=-2, axis2=-1)
        value = value - 2.0 * jnp.trace(cross, axis1=-2, axis2=-1)
        return jnp.maximum(value, 0.0)

    def dist(self, P: Array, Q: Array) -> Array:
        return jnp.sqrt(self.squared_dist(P, Q))

    def _metric_coordinates(self, P: Array, U: Array) -> Array:
        """Map a tangent isometrically to the fixed Euclidean Sym(n) space."""
        P = _sym(jnp.asarray(P))
        U = self.tangent_project(P, U)
        vals, eigvecs = _eigh_sym(P)
        rotated = jnp.swapaxes(eigvecs, -1, -2) @ U @ eigvecs
        weights = jnp.sqrt(2.0 * (vals[..., :, None] + vals[..., None, :]))
        coordinates = rotated / jnp.maximum(weights, self.eps)
        return _sym(eigvecs @ coordinates @ jnp.swapaxes(eigvecs, -1, -2))

    def _from_metric_coordinates(self, P: Array, coordinates: Array) -> Array:
        P = _sym(jnp.asarray(P))
        vals, eigvecs = _eigh_sym(P)
        rotated = jnp.swapaxes(eigvecs, -1, -2) @ _sym(coordinates) @ eigvecs
        weights = jnp.sqrt(2.0 * (vals[..., :, None] + vals[..., None, :]))
        return _sym(eigvecs @ (weights * rotated) @ jnp.swapaxes(eigvecs, -1, -2))

    def transport(self, P: Array, Q: Array, U: Array) -> Array:
        """Isometric vector transport between Bures-Wasserstein tangent spaces.

        This transport preserves the Bures-Wasserstein metric exactly but is
        not claimed to be Levi-Civita parallel transport.
        """
        return self._from_metric_coordinates(Q, self._metric_coordinates(P, U))

    transp = transport

    def egrad_to_rgrad(self, P: Array, egrad: Array) -> Array:
        P = _sym(jnp.asarray(P))
        E = self.tangent_project(P, egrad)
        return _sym(2.0 * (P @ E + E @ P))

    egrad2rgrad = egrad_to_rgrad

    def lincomb(self, P: Array, *terms: Any) -> Array:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = coeff * vec
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(P, out)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = _as_sample_shape(sample_shape)
        A = jax.random.normal(key, shape=sample_shape + self.shape)
        A = _sym(A) / jnp.sqrt(float(self.n))
        return _spd_expm(A)

    def random_tangent(
        self,
        key: Array,
        P: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        Z = jax.random.normal(key, shape=jnp.shape(P))
        U = self.tangent_project(P, Z)
        if normalize:
            norm = self.norm(P, U)[..., None, None]
            U = jnp.where(norm > self.eps, U / norm, U)
        return scale * U


__all__ = ["SPDLogEuclidean", "SPDAffineInvariant", "SPDBuresWasserstein"]
