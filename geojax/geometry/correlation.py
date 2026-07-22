"""Correlation-matrix geometries based on Cholesky coordinates.

The full-rank correlation manifold is

    Cor(n) = {C in SPD(n): diag(C) = 1}.

The two classes below use the Cholesky diffeomorphism

    Theta(C) = diag(chol(C))^{-1} chol(C),

which maps a correlation matrix to a unit lower-triangular matrix. Its inverse is

    Theta^{-1}(L) = D^{-1/2} L L^T D^{-1/2},
    D = diag(L L^T).

``CorrelationECM`` pulls back the Euclidean metric from the
strictly lower-triangular coordinates of ``Theta(C)``.

``CorrelationLEC`` pulls back the Euclidean metric from
``log(Theta(C))``, a strictly lower-triangular nilpotent matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import GeometryMixin, RetractionGeometryMixin, as_sample_shape

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


def _as_sample_shape(sample_shape: Shape = ()) -> tuple[int, ...]:
    return as_sample_shape(sample_shape)


def _parse_corr_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError("Correlation size must be square, e.g. size=(4, 4).")
    shape = tuple(map(int, tuple(size)))
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Correlation size must be square, e.g. size=(4, 4).")
    return shape


def _sym(A: Array) -> Array:
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def _transpose(A: Array) -> Array:
    return jnp.swapaxes(A, -1, -2)


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(A * B, axis=(-2, -1))


def _diag(A: Array) -> Array:
    return jnp.diagonal(A, axis1=-2, axis2=-1)


def _eye_like(A: Array, n: int) -> Array:
    eye = jnp.eye(n, dtype=A.dtype)
    return jnp.broadcast_to(eye, A.shape)


def _strict_lower(A: Array) -> Array:
    return jnp.tril(A, k=-1)


def _unit_lower_from_strict(A: Array, n: int) -> Array:
    A = _strict_lower(A)
    return A + _eye_like(A, n)


def _set_unit_diag_symmetric(C: Array, n: int) -> Array:
    eye = _eye_like(C, n)
    return _sym(C) * (1.0 - eye) + eye


def _canonical_correlation(C: Array) -> Array:
    """Cheap canonicalization for matrices already on/near Corr(n)."""
    C = jnp.asarray(C)
    return _set_unit_diag_symmetric(C, C.shape[-1])


def _corr_normalize(P: Array, eps: float) -> Array:
    P = _sym(P)
    d = jnp.sqrt(jnp.maximum(_diag(P), eps))
    C = P / (d[..., :, None] * d[..., None, :])
    return _set_unit_diag_symmetric(C, C.shape[-1])


def _theta(C: Array, eps: float) -> Array:
    """Cholesky diffeomorphism Cor(n) -> unit lower-triangular matrices."""
    C = _sym(C)
    del eps
    n = C.shape[-1]
    L = jnp.linalg.cholesky(C)
    d = jnp.diagonal(L, axis1=-2, axis2=-1)
    U = L / d[..., :, None]
    return _unit_lower_from_strict(U, n)


def _theta_inv(L: Array, eps: float) -> Array:
    L = _unit_lower_from_strict(L, L.shape[-1])
    P = L @ jnp.swapaxes(L, -1, -2)
    return _corr_normalize(P, eps)


def _lower_log_unit(L: Array, n: int) -> Array:
    """Finite matrix logarithm of a unit lower-triangular matrix."""
    L = _unit_lower_from_strict(L, n)
    identity = _eye_like(L, n)
    N = L - identity
    out = jnp.zeros_like(L)
    power = N
    for k in range(1, n):
        coeff = ((-1.0) ** (k + 1)) / float(k)
        out = out + coeff * power
        power = power @ N
    return _strict_lower(out)


def _lower_expm_strict(A: Array, n: int) -> Array:
    """Finite matrix exponential of a strictly lower-triangular matrix."""
    A = _strict_lower(A)
    identity = _eye_like(A, n)
    out = identity
    term = identity
    for k in range(1, n):
        term = (term @ A) / float(k)
        out = out + term
    return _unit_lower_from_strict(out, n)


@dataclass(frozen=True)
class _CorrelationCholeskyBase(GeometryMixin):
    size: tuple[int, int]
    atol: float = 1e-6
    eps: float = 1e-10

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.n * (self.n - 1) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    # ------------------------------------------------------------------
    # Chart methods implemented by subclasses
    # ------------------------------------------------------------------
    def chart(self, C: Array) -> Array:  # pragma: no cover - abstract-like
        raise NotImplementedError

    def chart_inverse(self, Z: Array) -> Array:  # pragma: no cover - abstract-like
        raise NotImplementedError

    def chart_jvp(self, C: Array, U: Array) -> Array:
        U = self.tangent_project(C, U)
        return jax.jvp(self.chart, (C,), (U,))[1]

    def inverse_chart_jvp(self, Z: Array, W: Array) -> Array:
        W = _strict_lower(W)
        return self.tangent_project(
            self.chart_inverse(Z), jax.jvp(self.chart_inverse, (Z,), (W,))[1]
        )

    # ------------------------------------------------------------------
    # Manifold interface
    # ------------------------------------------------------------------
    def belongs(self, C: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        C = jnp.asarray(C)
        sym_ok = jnp.linalg.norm(C - jnp.swapaxes(C, -1, -2), axis=(-2, -1)) <= tol
        diag_ok = jnp.linalg.norm(_diag(C) - 1.0, axis=-1) <= tol
        vals = jnp.linalg.eigvalsh(_sym(C))
        pd_ok = jnp.min(vals, axis=-1) > tol
        return sym_ok & diag_ok & pd_ok

    def project(self, C: Array) -> Array:
        vals, Q = jnp.linalg.eigh(_sym(C))
        P = (Q * jnp.maximum(vals, self.eps)[..., None, :]) @ jnp.swapaxes(Q, -1, -2)
        return _corr_normalize(P, self.eps)

    normalize = project

    def is_tangent(self, C: Array, U: Array, atol: float | None = None) -> Array:
        del C
        tol = self.atol if atol is None else atol
        U = jnp.asarray(U)
        sym_ok = jnp.linalg.norm(U - jnp.swapaxes(U, -1, -2), axis=(-2, -1)) <= tol
        diag_ok = jnp.linalg.norm(_diag(U), axis=-1) <= tol
        return sym_ok & diag_ok

    def tangent_project(self, C: Array, U: Array) -> Array:
        del C
        U = _sym(U)
        eye = _eye_like(U, self.n)
        return U * (1.0 - eye)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, C: Array, U: Array, V: Array) -> Array:
        dU = self.chart_jvp(C, U)
        dV = self.chart_jvp(C, V)
        return _trace_inner(dU, dV)

    def norm(self, C: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(C, U, U), 0.0))

    def lincomb(self, C: Array, *terms: Any) -> Array:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = coeff * vec
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(C, out)

    def exp(self, C: Array, U: Array) -> Array:
        C = _canonical_correlation(C)
        Z = self.chart(C)
        dZ = self.chart_jvp(C, U)
        return self.chart_inverse(Z + dZ)

    def retr(self, C: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(C, t * U)

    def log(self, C: Array, D: Array) -> Array:
        C = _canonical_correlation(C)
        D = _canonical_correlation(D)
        Z = self.chart(C)
        return self.inverse_chart_jvp(Z, self.chart(D) - Z)

    def dist(self, C: Array, D: Array) -> Array:
        Delta = self.chart(_canonical_correlation(D)) - self.chart(_canonical_correlation(C))
        return jnp.sqrt(jnp.maximum(_trace_inner(Delta, Delta), 0.0))

    def transport(self, C: Array, D: Array, U: Array) -> Array:
        D = _canonical_correlation(D)
        return self.inverse_chart_jvp(self.chart(D), self.chart_jvp(C, U))

    transp = transport

    def egrad_to_rgrad(self, C: Array, egrad: Array) -> Array:
        C = _canonical_correlation(C)
        Z = self.chart(C)
        # Pull ambient covector back to flat chart coordinates and push it to
        # the tangent space. This gives the gradient for the pullback metric.
        _, vjp_fun = jax.vjp(self.chart_inverse, Z)
        grad_Z = _strict_lower(vjp_fun(self.tangent_project(C, egrad))[0])
        return self.inverse_chart_jvp(Z, grad_Z)

    egrad2rgrad = egrad_to_rgrad

    def random_point(
        self, key: Array, sample_shape: Shape = (), *, scale: float | Array = 0.35
    ) -> Array:
        sample_shape = _as_sample_shape(sample_shape)
        Z = scale * jax.random.normal(key, shape=sample_shape + self.shape)
        return self.chart_inverse(_strict_lower(Z))

    def random_tangent(
        self, key: Array, C: Array, *, scale: float | Array = 1.0, normalize: bool = False
    ) -> Array:
        Z = scale * _strict_lower(jax.random.normal(key, shape=jnp.shape(C)))
        U = self.inverse_chart_jvp(self.chart(_canonical_correlation(C)), Z)
        if normalize:
            nrm = self.norm(C, U)
            U = jnp.where(nrm[..., None, None] > self.eps, U / nrm[..., None, None], U)
        return U

    def frechet_mean_closed_form(self, Cs: Array) -> Array:
        """Closed-form mean in the flat Cholesky chart."""
        Zs = jax.vmap(self.chart)(Cs)
        return self.chart_inverse(jnp.mean(Zs, axis=0))


@dataclass(frozen=True, init=False)
class CorrelationECM(_CorrelationCholeskyBase):
    """Correlation manifold with the Euclidean-Cholesky metric.

    ECM is the Euclidean-Cholesky metric. Its flat coordinates are the
    strictly lower-triangular entries of the unit-diagonal Cholesky factor.
    """

    def __init__(self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10):
        shape = _parse_corr_size(size)
        if shape[0] < 2:
            raise ValueError("Correlation matrix size must be at least 2.")
        object.__setattr__(self, "size", shape)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "eps", eps)

    def chart(self, C: Array) -> Array:
        return _strict_lower(_theta(C, self.eps))

    def chart_inverse(self, Z: Array) -> Array:
        return _theta_inv(_unit_lower_from_strict(Z, self.n), self.eps)


@dataclass(frozen=True, init=False)
class CorrelationLEC(_CorrelationCholeskyBase):
    """Correlation manifold with the log-Euclidean-Cholesky metric.

    LEC is the log-Euclidean-Cholesky metric (also abbreviated LECM in the
    literature). Its flat coordinates are the matrix logarithm of the
    unit-diagonal Cholesky factor.
    """

    def __init__(self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10):
        shape = _parse_corr_size(size)
        if shape[0] < 2:
            raise ValueError("Correlation matrix size must be at least 2.")
        object.__setattr__(self, "size", shape)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "eps", eps)

    def chart(self, C: Array) -> Array:
        return _lower_log_unit(_theta(C, self.eps), self.n)

    def chart_inverse(self, Z: Array) -> Array:
        return _theta_inv(_lower_expm_strict(Z, self.n), self.eps)


@dataclass(frozen=True, init=False)
class CorrelationAffineQuotient(RetractionGeometryMixin):
    """Full-rank correlations with the affine-invariant quotient metric.

    Correlation matrices are the quotient of SPD matrices by positive diagonal
    congruence. The metric is evaluated using affine-invariant horizontal lifts.
    Point updates use normalized-addition retractions; consequently ``exp``,
    ``log`` and ``dist`` are documented proxies.
    """

    size: tuple[int, int]
    atol: float
    eps: float

    def __init__(self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-10):
        shape = _parse_corr_size(size)
        if shape[0] < 2:
            raise ValueError("Correlation matrix size must be at least 2.")
        object.__setattr__(self, "size", shape)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.n * (self.n - 1) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    @property
    def _structural(self) -> CorrelationECM:
        return CorrelationECM(self.size, atol=self.atol, eps=self.eps)

    def belongs(self, C: Array, atol: float | None = None) -> Array:
        return self._structural.belongs(C, atol=atol)

    def project(self, C: Array) -> Array:
        return self._structural.project(C)

    normalize = project

    def is_tangent(self, C: Array, U: Array, atol: float | None = None) -> Array:
        return self._structural.is_tangent(C, U, atol=atol)

    def tangent_project(self, C: Array, U: Array) -> Array:
        return self._structural.tangent_project(C, U)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def horizontal_lift(self, C: Array, U: Array) -> Array:
        """Lift a correlation tangent horizontally to the SPD total space."""
        C = self.project(C)
        U = self.tangent_project(C, U)
        inverse = jnp.linalg.solve(C, jnp.eye(self.n, dtype=C.dtype))
        operator = jnp.eye(self.n, dtype=C.dtype) + inverse * _transpose(C)
        right_hand_side = -_diag(inverse @ U)
        diagonal = jnp.linalg.solve(operator, right_hand_side[..., None])[..., 0]
        D = jnp.eye(self.n, dtype=C.dtype) * diagonal[..., None, :]
        return _sym(U + D @ C + C @ D)

    def inner(self, C: Array, U: Array, V: Array) -> Array:
        C = self.project(C)
        inverse = jnp.linalg.solve(C, jnp.eye(self.n, dtype=C.dtype))
        lift_u = self.horizontal_lift(C, U)
        lift_v = self.horizontal_lift(C, V)
        return _trace_inner(inverse @ lift_u @ inverse, lift_v)

    def norm(self, C: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(C, U, U), 0.0))

    def retr(self, C: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.project(jnp.asarray(C) + t * self.tangent_project(C, U))

    def invretr(self, C: Array, D: Array) -> Array:
        return self.tangent_project(C, jnp.asarray(D) - jnp.asarray(C))

    def _tangent_basis(self, C: Array) -> Array:
        basis = []
        for row in range(self.n):
            for column in range(row + 1, self.n):
                element = jnp.zeros(self.shape, dtype=jnp.asarray(C).dtype)
                element = element.at[row, column].set(1.0 / jnp.sqrt(2.0))
                element = element.at[column, row].set(1.0 / jnp.sqrt(2.0))
                basis.append(element)
        return jnp.stack(basis)

    def egrad_to_rgrad(self, C: Array, egrad: Array) -> Array:
        basis = self._tangent_basis(C)
        gram = jax.vmap(lambda left: jax.vmap(lambda right: self.inner(C, left, right))(basis))(
            basis
        )
        covector = jax.vmap(lambda vector: _trace_inner(egrad, vector))(basis)
        coefficients = jnp.linalg.solve(gram, covector)
        return self.tangent_project(C, jnp.tensordot(coefficients, basis, axes=1))

    egrad2rgrad = egrad_to_rgrad

    def random_point(
        self,
        key: Array,
        sample_shape: Shape = (),
        *,
        scale: float | Array = 0.35,
    ) -> Array:
        return self._structural.random_point(key, sample_shape=sample_shape, scale=scale)

    def random_tangent(
        self,
        key: Array,
        C: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        tangent = self.tangent_project(C, jax.random.normal(key, shape=jnp.shape(C)))
        if normalize:
            length = self.norm(C, tangent)[..., None, None]
            tangent = jnp.where(length > self.eps, tangent / length, tangent)
        return scale * tangent


__all__ = [
    "CorrelationECM",
    "CorrelationLEC",
    "CorrelationAffineQuotient",
]
