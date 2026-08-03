"""Canonical unit-sphere geometry in JAX.

The representation is ambient: a point on S^d is stored as an array in
R^{d+1}. The geometry is intrinsic/canonical: tangent spaces, the metric,
exponential map, logarithm map, parallel transport, and geodesic flow are all
for the round unit sphere.

Core convention
---------------
The last array axis is the ambient coordinate axis. Leading axes are batch axes
and are broadcast by JAX in the usual way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import ExactGeometryMixin, as_sample_shape
from ._numerics import (
    acos_over_sin,
    acos_squared,
    cos_from_squared_norm,
    sinc_from_squared_norm,
    squared_norm,
)

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


@dataclass(frozen=True, init=False)
class Sphere(ExactGeometryMixin):
    """Canonical geometry of the unit sphere S^d in R^{d+1}.

    Parameters
    ----------
    size:
        Ambient Euclidean dimension, equal to d + 1 for S^d.
    atol:
        Absolute tolerance used by membership and tangency checks.
    eps:
        Small positive number used in numerically stable divisions.
    """

    hessian_conversion_is_exact = True
    riemannian_gradient_jvp_is_exact = True

    size: int
    atol: float
    eps: float

    def __init__(self, size: int, *, atol: float = 1e-6, eps: float = 1e-12) -> None:
        size = int(size)
        if size < 2:
            raise ValueError("Sphere size must be at least 2.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "eps", float(eps))

    @property
    def dim(self) -> int:
        """Intrinsic dimension d of S^d."""
        return self.size - 1

    @property
    def shape(self) -> tuple[int]:
        """Shape of one unbatched point."""
        return (self.size,)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _as_sample_shape(self, sample_shape: Shape = ()) -> tuple[int, ...]:
        return as_sample_shape(sample_shape)

    def _dot(self, u: Array, v: Array, keepdims: bool = False) -> Array:
        return jnp.sum(u * v, axis=-1, keepdims=keepdims)

    def _safe_unit_first_axis(self, x: Array) -> Array:
        """Return a broadcasted first basis vector with the same shape as x."""
        e0 = jnp.zeros_like(x)
        return e0.at[..., 0].set(1.0)

    # ------------------------------------------------------------------
    # Identity / validation
    # ------------------------------------------------------------------
    def belongs(self, x: Array, atol: float | None = None) -> Array:
        """Check whether x lies on the unit sphere."""
        tol = self.atol if atol is None else atol
        x = jnp.asarray(x)
        if not self._shape_matches(x):
            return self._shape_failure(x)
        return jnp.abs(jnp.linalg.norm(x, axis=-1) - 1.0) <= tol

    def is_tangent(self, x: Array, u: Array, atol: float | None = None) -> Array:
        """Check whether u is tangent at x, i.e. <x, u> = 0."""
        tol = self.atol if atol is None else atol
        if not self._shape_matches(x, u):
            return self._shape_failure(x)
        x, u = self._check_shapes(("x", x), ("u", u))
        return jnp.abs(self._dot(x, u)) <= tol

    def project(self, x: Array) -> Array:
        """Normalize a nonzero ambient vector to the unit sphere.

        This is for initialization or numerical repair. The core geometry
        methods use exact sphere formulas.
        """
        x = self._check_shape(x, name="x")
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        norm_safe = jnp.where(norm > self.eps, norm, 1.0)
        e0 = self._safe_unit_first_axis(x)
        return jnp.where(norm > self.eps, x / norm_safe, e0)

    normalize = project

    # ------------------------------------------------------------------
    # Tangent geometry
    # ------------------------------------------------------------------
    def tangent_project(self, x: Array, u: Array) -> Array:
        """Orthogonally project an ambient vector u to T_x S^d.

        Formula: Proj_x(u) = u - <x, u> x.
        """
        x, u = self._check_shapes(("x", x), ("u", u))
        return u - self._dot(x, u, keepdims=True) * x

    # Common aliases used by manifold libraries.
    projection = tangent_project
    proj = tangent_project
    to_tangent = tangent_project

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Canonical Riemannian inner product on T_x S^d."""
        _, u, v = self._check_shapes(("x", x), ("u", u), ("v", v))
        return self._dot(u, v)

    def norm(self, x: Array, u: Array) -> Array:
        """Canonical Riemannian norm of a tangent vector."""
        return jnp.sqrt(jnp.maximum(self.inner(x, u, u), 0.0))

    # ------------------------------------------------------------------
    # Geodesic geometry
    # ------------------------------------------------------------------
    def exp(self, x: Array, u: Array) -> Array:
        """Riemannian exponential map on the unit sphere.

        For u in T_x S^d and r = ||u||,
            Exp_x(u) = cos(r) x + sin(r) u / r.
        """
        x = jnp.asarray(x)
        u = self.tangent_project(x, jnp.asarray(u))
        r2 = squared_norm(u, axis=-1, keepdims=True)
        y = cos_from_squared_norm(r2) * x + sinc_from_squared_norm(r2) * u
        return self.project(y)

    def log(self, x: Array, y: Array) -> Array:
        """Riemannian logarithm map on the unit sphere.

        For ``y`` not antipodal to ``x``, the map is

        ``Log_x(y) = theta / sin(theta) * (y - cos(theta) x)``,

        where ``theta = arccos(<x, y>)``.

        At the antipode the logarithm is not unique; this implementation
        returns NaN there.
        """
        x = self.project(x)
        y = self.project(y)
        dot = jnp.clip(self._dot(x, y, keepdims=True), -1.0, 1.0)
        tangent = y - dot * x
        tangent_squared = squared_norm(tangent, axis=-1, keepdims=True)
        v = acos_over_sin(dot, tangent_squared) * tangent
        dtype = jnp.result_type(x, float)
        sine_cutoff = 64.0 * jnp.finfo(dtype).eps
        antipodal = (dot < 0.0) & (tangent_squared <= sine_cutoff**2)
        return jnp.where(antipodal, jnp.nan, self.tangent_project(x, v))

    def squared_dist(self, x: Array, y: Array) -> Array:
        """Squared geodesic distance with a finite coincident-point gradient."""
        x = self.project(x)
        y = self.project(y)
        dot = jnp.clip(self._dot(x, y), -1.0, 1.0)
        return acos_squared(dot)

    def dist(self, x: Array, y: Array) -> Array:
        """Geodesic distance on the unit sphere."""
        return jnp.sqrt(self.squared_dist(x, y))

    def transport(self, x: Array, y: Array, u: Array) -> Array:
        """Parallel transport along the unique shortest geodesic from x to y.

        For x != -y,
            PT_{x->y}(u) = u - (<u, y> / (1 + <x, y>)) (x + y).

        At antipodal endpoints the shortest geodesic is not unique; this
        implementation returns NaN there.
        """
        x = self.project(x)
        y = self.project(y)
        u = self.tangent_project(x, u)
        dot_xy = jnp.clip(self._dot(x, y, keepdims=True), -1.0, 1.0)
        direction = y - dot_xy * x
        sine_squared = squared_norm(direction, axis=-1, keepdims=True)
        sine = jnp.sqrt(sine_squared)
        dtype = jnp.result_type(x, float)
        sine_cutoff = 64.0 * jnp.finfo(dtype).eps
        safe_sine = jnp.where(sine > sine_cutoff, sine, 1.0)
        unit_direction = direction / safe_sine
        component = self._dot(u, unit_direction, keepdims=True)
        terminal_direction = -sine * x + dot_xy * unit_direction
        transported = u + component * (terminal_direction - unit_direction)
        transported = self.tangent_project(y, transported)
        antipodal = (dot_xy < 0.0) & (sine <= sine_cutoff)
        return jnp.where(antipodal, jnp.nan, transported)

    def geodesic_flow(self, x: Array, v: Array, t: float | Array = 1.0) -> tuple[Array, Array]:
        """Exact kinetic/geodesic flow on the sphere.

        Given x in S^d and v in T_x S^d, returns (x_t, v_t), where x_t is the
        geodesic position at time t and v_t is its velocity at time t.

        This is the key primitive for Geodesic Monte Carlo.
        """
        x = self.project(x)
        v = self.tangent_project(x, v)
        r2 = squared_norm(v, axis=-1, keepdims=True)
        t_array = jnp.asarray(t)
        t_array = jnp.reshape(t_array, t_array.shape + (1,))
        tr2 = t_array * t_array * r2
        cosine = cos_from_squared_norm(tr2)
        sinc = sinc_from_squared_norm(tr2)
        x_t = cosine * x + t_array * sinc * v
        v_t = -t_array * r2 * sinc * x + cosine * v
        x_t = self.project(x_t)
        v_t = self.tangent_project(x_t, v_t)
        return x_t, v_t

    # ------------------------------------------------------------------
    # Autodiff bridge
    # ------------------------------------------------------------------
    def egrad_to_rgrad(self, x: Array, egrad: Array) -> Array:
        """Convert an ambient Euclidean gradient to a Riemannian gradient."""
        return self.tangent_project(x, egrad)

    def ehess_to_rhess(
        self,
        x: Array,
        egrad: Array,
        ehess_vec: Array,
        u: Array,
    ) -> Array:
        """Convert an ambient Hessian product using the sphere shape operator."""
        x = self.project(x)
        u = self.tangent_project(x, u)
        correction = self._dot(x, jnp.asarray(egrad), keepdims=True) * u
        return self.tangent_project(x, jnp.asarray(ehess_vec)) - correction

    # ------------------------------------------------------------------
    # Random initialization / tangent Gaussian
    # ------------------------------------------------------------------
    def random_point(self, key: Array, sample_shape: Shape = ()) -> Array:
        """Sample uniformly from the unit sphere using normalized Gaussians."""
        sample_shape = self._as_sample_shape(sample_shape)
        z = jax.random.normal(key, shape=sample_shape + self.shape)
        return self.project(z)

    def random_tangent(
        self,
        key: Array,
        x: Array,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Array:
        """Sample a tangent Gaussian at x.

        By default this samples Proj_x(z), z ~ N(0, I), which is the canonical
        Gaussian on T_x S^d under the round metric. If normalize=True, the
        resulting tangent vector is normalized before scaling.
        """
        z = jax.random.normal(key, shape=jnp.shape(x))
        v = self.tangent_project(x, z)
        if normalize:
            n = jnp.linalg.norm(v, axis=-1, keepdims=True)
            n_safe = jnp.where(n > self.eps, n, 1.0)
            v = jnp.where(n > self.eps, v / n_safe, v)
        return scale * v


class SphereExtrinsic(Sphere):
    """Sphere geometry equipped with its identity equivariant embedding.

    The embedding ``j(x) = x`` maps the unit sphere into its ambient Euclidean
    space. Since its differential is also the identity, the pullback metric is
    the round metric already implemented by :class:`Sphere`; the Riemannian
    methods ``inner``, ``exp``, ``log``, ``dist``, and ``transport`` therefore
    remain unchanged.

    This class adds the genuinely extrinsic operations associated with the
    embedding: Euclidean chordal distance, nearest-point projection from the
    embedding space, and the projected ambient mean. The extrinsic mean is
    undefined when the weighted ambient mean is zero; ``extrinsic_mean``
    returns NaNs in that case.
    """

    def embed(self, x: Array) -> Array:
        """Apply the identity embedding ``j(x) = x``."""
        return self._check_shape(x, name="x")

    to_embedding = embed

    def from_embedding(self, z: Array) -> Array:
        """Project an ambient vector to its nearest point on the sphere."""
        return self.project(z)

    project_embedding = from_embedding

    def squared_chordal_dist(self, x: Array, y: Array) -> Array:
        """Squared Euclidean distance between the embedded points."""
        difference = self.project(x) - self.project(y)
        return jnp.sum(difference * difference, axis=-1)

    def chordal_dist(self, x: Array, y: Array) -> Array:
        """Euclidean chordal distance ``||j(x) - j(y)||``."""
        return jnp.sqrt(jnp.maximum(self.squared_chordal_dist(x, y), 0.0))

    embedding_dist = chordal_dist

    def extrinsic_mean(self, points: Array, weights: Array | None = None) -> Array:
        """Project the weighted ambient mean back to the sphere."""
        points = jnp.asarray(points)
        if weights is None:
            ambient_mean = jnp.mean(points, axis=0)
        else:
            weights = jnp.asarray(weights)
            weights = weights / jnp.sum(weights)
            ambient_mean = jnp.sum(weights[..., None] * points, axis=0)
        norm = jnp.linalg.norm(ambient_mean)
        normalized = ambient_mean / jnp.where(norm > self.eps, norm, 1.0)
        return jnp.where(norm > self.eps, normalized, jnp.full_like(normalized, jnp.nan))


__all__ = ["Sphere", "SphereExtrinsic"]
