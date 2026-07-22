"""Grassmann geometries in JAX.

``Grassmann`` uses an orthonormal-basis representative.  A point on
Gr(rank, size) is represented by a matrix X in R^{size x rank} with
orthonormal columns.  The actual Grassmann point is the subspace [X], so all
objective functions should be right-O(rank)-invariant: f(X R) = f(X).

Tangent vectors are represented by horizontal matrices U satisfying X^T U = 0.
The metric is the canonical quotient metric <U,V> = tr(U^T V).

``GrassmannProjection`` uses the equivariant projection embedding [X] -> XX^T.
Its points are symmetric rank-r orthogonal projectors and its tangent vectors
are symmetric matrices.  The normalized Frobenius metric 0.5 tr(HK) makes this
representation isometric to the canonical quotient geometry.  The ambient
Euclidean chord is available separately as ``chordal_dist``; it is not a
Riemannian geodesic distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from .base import GeometryMixin, as_sample_shape

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


def _as_sample_shape(sample_shape: Shape = ()) -> tuple[int, ...]:
    return as_sample_shape(sample_shape)


def _trace_inner(A: Array, B: Array) -> Array:
    return jnp.sum(A * B, axis=(-2, -1))


def _sym(A: Array) -> Array:
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def _orthonormalize(Y: Array, eps: float) -> Array:
    """QR orthonormalization with deterministic column signs."""
    Q, R = jnp.linalg.qr(Y, mode="reduced")
    signs = jnp.sign(jnp.diagonal(R, axis1=-2, axis2=-1))
    signs = jnp.where(jnp.abs(signs) > eps, signs, 1.0)
    return Q * signs[..., None, :]


def _parse_grassmann_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        raise ValueError("Grassmann size must be a pair (ambient_dim, rank).")
    shape = tuple(int(v) for v in size)
    if len(shape) != 2:
        raise ValueError("Grassmann size must be a pair (ambient_dim, rank).")
    n, k = shape
    if n < 1:
        raise ValueError("Grassmann ambient dimension must be positive.")
    if k < 1:
        raise ValueError("Grassmann rank must be positive.")
    if k > n:
        raise ValueError("Grassmann rank must satisfy rank <= ambient_dim.")
    return n, k


@dataclass(frozen=True, init=False)
class Grassmann(GeometryMixin):
    """Canonical Grassmann geometry Gr(rank, size).

    Parameters
    ----------
    size:
        Pair ``(ambient_dim, rank)``.
    """

    size: tuple[int, int]
    rank: int
    atol: float
    eps: float

    def __init__(
        self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-12
    ) -> None:
        parsed = _parse_grassmann_size(size)
        object.__setattr__(self, "size", parsed)
        object.__setattr__(self, "rank", parsed[1])
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def ambient_dim(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.rank * (self.ambient_dim - self.rank)

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        X = jnp.asarray(X)
        gram = jnp.swapaxes(X, -1, -2) @ X
        identity = jnp.eye(self.rank, dtype=X.dtype)
        return jnp.linalg.norm(gram - identity, axis=(-2, -1)) <= tol

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        tol = self.atol if atol is None else atol
        X = jnp.asarray(X)
        U = jnp.asarray(U)
        xtu = jnp.swapaxes(X, -1, -2) @ U
        return jnp.linalg.norm(xtu, axis=(-2, -1)) <= tol

    def project(self, X: Array) -> Array:
        X = jnp.asarray(X)
        return _orthonormalize(X, self.eps)

    normalize = project

    def tangent_project(self, X: Array, U: Array) -> Array:
        X = self.project(X)
        U = jnp.asarray(U)
        return U - X @ (jnp.swapaxes(X, -1, -2) @ U)

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        del X
        return _trace_inner(U, V)

    def norm(self, X: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(X, U, U), 0.0))

    def lincomb(self, X: Array, *terms: Any) -> Array:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        out = None
        for coeff, vec in zip(terms[0::2], terms[1::2]):
            term = coeff * vec
            out = term if out is None else out + term
        if out is None:
            raise ValueError("lincomb requires at least one coefficient/vector pair.")
        return self.tangent_project(X, out)

    def exp(self, X: Array, U: Array) -> Array:
        """Grassmann exponential map in ONB coordinates.

        If ``U = A diag(s) B^T`` is the compact SVD of a horizontal tangent
        vector, then a representative of ``Exp(X, U)`` is

            ``X B cos(s) B^T + A sin(s) B^T + X (I - B B^T)``.
        """
        X = self.project(X)
        U = self.tangent_project(X, U)
        A, s, Vt = jnp.linalg.svd(U, full_matrices=False)
        V = jnp.swapaxes(Vt, -1, -2)
        XV = X @ V
        term1 = (XV * jnp.cos(s)[..., None, :]) @ Vt
        term2 = (A * jnp.sin(s)[..., None, :]) @ Vt
        VVt = V @ Vt
        identity = jnp.eye(self.rank, dtype=X.dtype)
        term3 = X @ (identity - VVt)
        return self.project(term1 + term2 + term3)

    def retr(self, X: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(X, t * U)

    def log(self, X: Array, Y: Array) -> Array:
        """Grassmann logarithm map.

        This is well defined away from the cut locus; numerically this requires
        X^T Y to be nonsingular, i.e. no principal angle equal to pi/2.
        """
        X = self.project(X)
        Y = self.project(Y)
        XtY = jnp.swapaxes(X, -1, -2) @ Y
        Z = self.tangent_project(X, Y)
        # M = (I - X X^T) Y (X^T Y)^{-1} without forming an inverse.
        M = jnp.swapaxes(
            jnp.linalg.solve(jnp.swapaxes(XtY, -1, -2), jnp.swapaxes(Z, -1, -2)), -1, -2
        )
        U, s, Vt = jnp.linalg.svd(M, full_matrices=False)
        return self.tangent_project(X, (U * jnp.arctan(s)[..., None, :]) @ Vt)

    def dist(self, X: Array, Y: Array) -> Array:
        X = self.project(X)
        Y = self.project(Y)
        XtY = jnp.swapaxes(X, -1, -2) @ Y
        cos_theta = jnp.linalg.svd(XtY, compute_uv=False)
        normal_component = Y - X @ XtY
        # SVD sorts both spectra in descending order.  Principal angles are
        # ascending with cos(theta), so reverse the sin(theta) spectrum.
        sin_theta = jnp.flip(jnp.linalg.svd(normal_component, compute_uv=False), axis=-1)
        theta = jnp.arctan2(jnp.clip(sin_theta, 0.0, 1.0), jnp.clip(cos_theta, 0.0, 1.0))
        return jnp.sqrt(jnp.maximum(jnp.sum(theta * theta, axis=-1), 0.0))

    def transport(self, X: Array, Y: Array, Z: Array) -> Array:
        """Parallel transport along the shortest geodesic from X to Y.

        This uses the standard Grassmann formula based on the SVD of
        eta = Log_X(Y).  The returned vector is represented at the supplied
        endpoint representative Y and is horizontally projected there.
        """
        X = self.project(X)
        Y = self.project(Y)
        Z = self.tangent_project(X, Z)
        eta = self.log(X, Y)
        U, s, Vt = jnp.linalg.svd(eta, full_matrices=False)
        V = jnp.swapaxes(Vt, -1, -2)
        del V  # V is not needed in this transport expression.
        X_part = -((X @ jnp.swapaxes(Vt, -1, -2)) * jnp.sin(s)[..., None, :])
        U_part = U * jnp.cos(s)[..., None, :]
        basis = X_part + U_part
        transported = basis @ (jnp.swapaxes(U, -1, -2) @ Z) + (
            Z - U @ (jnp.swapaxes(U, -1, -2) @ Z)
        )
        return self.tangent_project(Y, transported)

    transp = transport

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        return self.tangent_project(X, egrad)

    egrad2rgrad = egrad_to_rgrad

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        sample_shape = _as_sample_shape(sample_shape)
        Z = jax.random.normal(key, shape=sample_shape + self.shape)
        return self.project(Z)

    def random_tangent(
        self,
        key: Array,
        X: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        Z = jax.random.normal(key, shape=jnp.shape(X))
        U = self.tangent_project(X, Z)
        if normalize:
            n = self.norm(X, U)[..., None, None]
            U = jnp.where(n > self.eps, U / n, U)
        return scale * U

    def projector(self, X: Array) -> Array:
        """Return the rank-``rank`` orthogonal projector XX^T."""
        X = self.project(X)
        return X @ jnp.swapaxes(X, -1, -2)


@dataclass(frozen=True, init=False)
class GrassmannProjection(GeometryMixin):
    """Projection-embedded Grassmann geometry.

    Public points are ``(n, k)`` orthonormal frames ``X``, exactly as for
    :class:`Grassmann`. Operations are evaluated after the equivariant
    embedding

    ``j([X]) = X @ X.T``.

    The embedded point is a symmetric rank-``k`` projector ``P``. A horizontal
    frame tangent ``U`` is embedded as

    ``dj_X(U) = U @ X.T + X @ U.T``.

    The Riemannian metric is the normalized Frobenius metric

    ``inner(P, H, K) = 0.5 * trace(H @ K)``,

    which makes ``j`` an isometric embedding. Public tangent vectors are
    returned in ``(n, k)`` horizontal form. Internally, exponential maps and
    transport use projector tangents and matrix exponentials.

    Returning from an embedded matrix uses the top ``k`` eigenvectors of its
    symmetric part. When a reference frame is supplied, an orthogonal
    Procrustes alignment fixes the otherwise arbitrary right-orthogonal gauge.
    Use :meth:`chordal_dist` for the extrinsic projection distance
    ``||j(X) - j(Y)||_F / sqrt(2)``.

    Parameters
    ----------
    size:
        Pair ``(ambient_dim, rank)``. Public points have the same shape.
    """

    size: tuple[int, int]
    rank: int
    atol: float
    eps: float

    def __init__(
        self, size: int | Sequence[int], *, atol: float = 1e-6, eps: float = 1e-12
    ) -> None:
        parsed = _parse_grassmann_size(size)
        object.__setattr__(self, "size", parsed)
        object.__setattr__(self, "rank", parsed[1])
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def ambient_dim(self) -> int:
        return self.size[0]

    @property
    def dim(self) -> int:
        return self.rank * (self.ambient_dim - self.rank)

    @property
    def shape(self) -> tuple[int, int]:
        return self.size

    def _canonical(self) -> Grassmann:
        return Grassmann(size=self.size, atol=self.atol, eps=self.eps)

    def belongs(self, X: Array, atol: float | None = None) -> Array:
        return self._canonical().belongs(X, atol=atol)

    def is_tangent(self, X: Array, U: Array, atol: float | None = None) -> Array:
        return self._canonical().is_tangent(X, U, atol=atol)

    def project(self, A: Array) -> Array:
        """Project an ambient ``(n, k)`` matrix to an orthonormal frame."""
        return self._canonical().project(A)

    normalize = project

    def _projector_tangent_project(self, P: Array, A: Array) -> Array:
        P = _sym(jnp.asarray(P))
        A = _sym(jnp.asarray(A))
        identity = jnp.eye(self.ambient_dim, dtype=P.dtype)
        complement = identity - P
        return P @ A @ complement + complement @ A @ P

    def tangent_project(self, X: Array, A: Array) -> Array:
        """Project an ambient frame vector through projector coordinates."""
        X = self.project(X)
        P = self.embed(X)
        ambient_projector_tangent = A @ jnp.swapaxes(X, -1, -2)
        ambient_projector_tangent += X @ jnp.swapaxes(A, -1, -2)
        H = self._projector_tangent_project(P, ambient_projector_tangent)
        return H @ X

    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, X: Array, U: Array, V: Array) -> Array:
        H = self.embed_tangent(X, U)
        K = self.embed_tangent(X, V)
        return 0.5 * _trace_inner(H, K)

    def norm(self, X: Array, U: Array) -> Array:
        return jnp.sqrt(jnp.maximum(self.inner(X, U, U), 0.0))

    def embed(self, X: Array) -> Array:
        """Map an orthonormal frame ``X`` to its projector ``X @ X.T``."""
        X = self._canonical().project(X)
        return _sym(X @ jnp.swapaxes(X, -1, -2))

    from_frame = embed

    def to_frame(self, P: Array, reference: Array | None = None) -> Array:
        """Recover a frame using a top-eigenspace decomposition.

        If ``reference`` is supplied, the recovered frame is right-aligned to
        it by the orthogonal Procrustes solution.
        """
        P = _sym(jnp.asarray(P))
        _, eigenvectors = jnp.linalg.eigh(P)
        frame = eigenvectors[..., :, -self.rank :]
        if reference is None:
            return frame
        reference = self.project(reference)
        left, _, right_t = jnp.linalg.svd(
            jnp.swapaxes(frame, -1, -2) @ reference,
            full_matrices=False,
        )
        return frame @ (left @ right_t)

    from_embedding = to_frame

    def project_embedding(self, A: Array, reference: Array | None = None) -> Array:
        """Project a symmetric matrix to a frame for its nearest rank-k projector."""
        return self.to_frame(A, reference=reference)

    def embed_tangent(self, X: Array, U: Array) -> Array:
        """Map a horizontal frame tangent to a symmetric projector tangent."""
        X = self.project(X)
        U = self._canonical().tangent_project(X, U)
        P = self.embed(X)
        H = U @ jnp.swapaxes(X, -1, -2) + X @ jnp.swapaxes(U, -1, -2)
        return self._projector_tangent_project(P, H)

    def from_projector_tangent(self, X: Array, H: Array) -> Array:
        """Map a projector tangent back to the horizontal frame gauge at ``X``."""
        X = self.project(X)
        P = self.embed(X)
        H = self._projector_tangent_project(P, H)
        return self._canonical().tangent_project(X, H @ X)

    def _matrix_exp(self, A: Array) -> Array:
        if A.ndim == 2:
            return jsp_linalg.expm(A)
        batch_shape = A.shape[:-2]
        flat = A.reshape((-1, self.ambient_dim, self.ambient_dim))
        result = jax.vmap(jsp_linalg.expm)(flat)
        return result.reshape(batch_shape + (self.ambient_dim, self.ambient_dim))

    def _projector_exp(self, P: Array, H: Array) -> Array:
        P = _sym(jnp.asarray(P))
        H = self._projector_tangent_project(P, H)
        generator = H @ P - P @ H
        rotation = self._matrix_exp(generator)
        result = rotation @ P @ jnp.swapaxes(rotation, -1, -2)
        return _sym(result)

    def exp(self, X: Array, U: Array) -> Array:
        """Evaluate the Riemannian exponential in projector coordinates."""
        X = self.project(X)
        P = self.embed(X)
        H = self.embed_tangent(X, U)
        return self.to_frame(self._projector_exp(P, H), reference=X)

    def retr(self, X: Array, U: Array, t: float | Array = 1.0) -> Array:
        return self.exp(X, t * U)

    def _projector_log(self, P: Array, Q: Array, reference: Array) -> Array:
        X = self.to_frame(P, reference=reference)
        Y = self.to_frame(Q)
        U = self._canonical().log(X, Y)
        return self.embed_tangent(X, U)

    def log(self, X: Array, Y: Array) -> Array:
        """Riemannian logarithm, defined away from the Grassmann cut locus."""
        X = self.project(X)
        P = self.embed(X)
        Q = self.embed(Y)
        return self.from_projector_tangent(X, self._projector_log(P, Q, X))

    def dist(self, X: Array, Y: Array) -> Array:
        """Intrinsic principal-angle geodesic distance."""
        P = self.embed(X)
        Q = self.embed(Y)
        return self._canonical().dist(
            self.to_frame(P, reference=X),
            self.to_frame(Q, reference=Y),
        )

    geodesic_dist = dist

    def squared_chordal_dist(self, X: Array, Y: Array) -> Array:
        """Squared extrinsic distance ``0.5 * ||j(X) - j(Y)||_F^2``."""
        difference = self.embed(X) - self.embed(Y)
        return jnp.maximum(0.5 * _trace_inner(difference, difference), 0.0)

    def chordal_dist(self, X: Array, Y: Array) -> Array:
        """Extrinsic projection distance ``||j(X) - j(Y)||_F / sqrt(2)``."""
        return jnp.sqrt(self.squared_chordal_dist(X, Y))

    projection_dist = chordal_dist

    def transport(self, X: Array, Y: Array, U: Array) -> Array:
        """Parallel transport evaluated in projector coordinates."""
        X = self.project(X)
        Y = self.project(Y)
        P = self.embed(X)
        Q = self.embed(Y)
        H = self.embed_tangent(X, U)
        eta = self._projector_log(P, Q, X)
        generator = eta @ P - P @ eta
        rotation = self._matrix_exp(generator)
        transported = rotation @ H @ jnp.swapaxes(rotation, -1, -2)
        return self.from_projector_tangent(Y, _sym(transported))

    transp = transport

    def egrad_to_rgrad(self, X: Array, egrad: Array) -> Array:
        X = self.project(X)
        egrad = jnp.asarray(egrad)
        projector_egrad = 0.5 * (egrad @ jnp.swapaxes(X, -1, -2) + X @ jnp.swapaxes(egrad, -1, -2))
        H = 2.0 * self._projector_tangent_project(self.embed(X), projector_egrad)
        return self.from_projector_tangent(X, H)

    egrad2rgrad = egrad_to_rgrad

    def ehess_to_rhess(self, X: Array, egrad: Array, ehess_vec: Array, U: Array) -> Array:
        """Convert an ambient frame Hessian-vector product to Grassmann form."""
        X = self.project(X)
        U = self.tangent_project(X, U)
        correction = U @ (jnp.swapaxes(X, -1, -2) @ egrad)
        return self.tangent_project(X, ehess_vec - correction)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        return self._canonical().random_point(key, sample_shape=sample_shape)

    def random_tangent(
        self,
        key: Array,
        X: Array,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        X = self.project(X)
        ambient = jax.random.normal(key, shape=jnp.shape(X))
        U = self.tangent_project(X, ambient)
        if normalize:
            n = self.norm(X, U)[..., None, None]
            U = jnp.where(n > self.eps, U / n, U)
        return scale * U

    def extrinsic_mean(self, points: Array, weights: Array | None = None) -> Array:
        """Project the weighted mean projector back to an orthonormal frame."""
        projectors = jax.vmap(self.embed)(points)
        if weights is None:
            ambient_mean = jnp.mean(projectors, axis=0)
        else:
            weights = jnp.asarray(weights)
            weights = weights / jnp.sum(weights)
            ambient_mean = jnp.sum(weights[..., None, None] * projectors, axis=0)
        return self.to_frame(ambient_mean)


__all__ = ["Grassmann", "GrassmannProjection"]
