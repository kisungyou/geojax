"""Product manifolds with arbitrary pytree state structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .base import GeometryMixin, GeometryProtocol, as_sample_shape, scale_tangent
from ._numerics import stable_norm

Array = Any
Shape = Union[int, Sequence[int], Tuple[int, ...]]


@dataclass(frozen=True, init=False, eq=False)
class Product(GeometryMixin):
    """Direct product of manifold geometries.

    ``factors`` may be any JAX pytree whose leaves are geometry objects. Product
    points and tangent vectors must have the same tree structure.
    """

    factors: Any
    _factor_leaves: tuple[Any, ...]
    _treedef: Any

    def __init__(self, factors: Any) -> None:
        leaves, treedef = jax.tree_util.tree_flatten(factors)
        if not leaves:
            raise ValueError("Product requires at least one factor geometry.")
        invalid = [
            type(factor).__name__ for factor in leaves if not isinstance(factor, GeometryProtocol)
        ]
        if invalid:
            names = ", ".join(invalid)
            raise TypeError(
                "Every Product factor must satisfy GeometryProtocol; "
                f"invalid factor types: {names}."
            )
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "_factor_leaves", tuple(leaves))
        object.__setattr__(self, "_treedef", treedef)

    def _flatten_like(self, tree: Any, name: str) -> tuple[Any, ...]:
        try:
            leaves = self._treedef.flatten_up_to(tree)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Product {name} must match the factor pytree structure.") from exc
        if len(leaves) != len(self._factor_leaves):
            raise ValueError(f"Product {name} must match the factor pytree structure.")
        return tuple(leaves)

    def _unflatten(self, leaves: Iterable[Any]) -> Any:
        return jax.tree_util.tree_unflatten(self._treedef, list(leaves))

    def _scale_tangent(self, vector: Any, coefficient: float | Array) -> Any:
        """Scale every factor tangent using the shared leading-batch contract."""
        leaves = self._flatten_like(vector, "tangent vector")
        scaled = []
        for M, leaf in zip(self._factor_leaves, leaves):
            if isinstance(M, Product):
                scaled.append(M._scale_tangent(leaf, coefficient))
            elif hasattr(M, "_scale_tangent"):
                scaled.append(M._scale_tangent(leaf, coefficient))
            else:
                scaled.append(
                    scale_tangent(
                        leaf,
                        coefficient,
                        event_ndim=len(tuple(M.shape)),
                    )
                )
        return self._unflatten(scaled)

    @staticmethod
    def _combine_checks(checks: list[Array]) -> Array:
        broadcast = jnp.broadcast_arrays(*(jnp.asarray(check, dtype=bool) for check in checks))
        result = jnp.ones_like(broadcast[0], dtype=bool)
        for check in broadcast:
            result = result & check
        return result

    @property
    def size(self) -> Any:
        return self._unflatten(
            getattr(M, "size", getattr(M, "shape", None)) for M in self._factor_leaves
        )

    @property
    def shape(self) -> Any:
        return self._unflatten(getattr(M, "shape", None) for M in self._factor_leaves)

    @property
    def dim(self) -> int:
        return int(sum(int(getattr(M, "dim")) for M in self._factor_leaves))

    @property
    def exp_is_exact(self) -> bool:
        return all(bool(getattr(M, "exp_is_exact", False)) for M in self._factor_leaves)

    @property
    def log_is_exact(self) -> bool:
        return all(bool(getattr(M, "log_is_exact", False)) for M in self._factor_leaves)

    @property
    def dist_is_exact(self) -> bool:
        return all(bool(getattr(M, "dist_is_exact", False)) for M in self._factor_leaves)

    @property
    def transport_is_isometric(self) -> bool:
        return all(bool(getattr(M, "transport_is_isometric", False)) for M in self._factor_leaves)

    @property
    def transport_is_parallel(self) -> bool:
        return all(bool(getattr(M, "transport_is_parallel", False)) for M in self._factor_leaves)

    @property
    def hessian_conversion_is_exact(self) -> bool:
        return all(
            bool(getattr(M, "hessian_conversion_is_exact", False)) for M in self._factor_leaves
        )

    @property
    def riemannian_gradient_jvp_is_exact(self) -> bool:
        return all(
            bool(getattr(M, "riemannian_gradient_jvp_is_exact", False)) for M in self._factor_leaves
        )

    def operation_kind(self, name: str) -> str:
        if name == "squared_dist":
            kinds = [M.operation_kind(name) for M in self._factor_leaves]
            if all(kind == "exact" for kind in kinds):
                return "exact"
            if any(kind == "proxy" for kind in kinds):
                return "proxy"
            return "numerical-local"
        if name in {"exp", "log", "dist"}:
            kinds = [
                M.operation_kind(name)
                if hasattr(M, "operation_kind")
                else ("exact" if bool(getattr(M, f"{name}_is_exact", False)) else "proxy")
                for M in self._factor_leaves
            ]
            if all(kind == "exact" for kind in kinds):
                return "exact"
            if any(kind == "proxy" for kind in kinds):
                return "proxy"
            return "numerical-local"
        return super().operation_kind(name)

    def belongs(self, x: Any, atol: float | None = None) -> Array:
        xs = self._flatten_like(x, "point")
        checks = []
        for M, xi in zip(self._factor_leaves, xs):
            check = M.belongs(xi, atol=atol) if atol is not None else M.belongs(xi)
            checks.append(check)
        return self._combine_checks(checks)

    def is_tangent(self, x: Any, u: Any, atol: float | None = None) -> Array:
        xs = self._flatten_like(x, "point")
        us = self._flatten_like(u, "tangent vector")
        checks = []
        for M, xi, ui in zip(self._factor_leaves, xs, us):
            check = M.is_tangent(xi, ui, atol=atol) if atol is not None else M.is_tangent(xi, ui)
            checks.append(check)
        return self._combine_checks(checks)

    def project(self, x: Any) -> Any:
        xs = self._flatten_like(x, "point")
        return self._unflatten(M.project(xi) for M, xi in zip(self._factor_leaves, xs))

    def tangent_project(self, x: Any, u: Any) -> Any:
        xs = self._flatten_like(x, "point")
        us = self._flatten_like(u, "tangent vector")
        return self._unflatten(
            M.tangent_project(xi, ui) for M, xi, ui in zip(self._factor_leaves, xs, us)
        )

    def inner(self, x: Any, u: Any, v: Any) -> Array:
        xs = self._flatten_like(x, "point")
        us = self._flatten_like(u, "tangent vector")
        vs = self._flatten_like(v, "tangent vector")
        vals = [M.inner(xi, ui, vi) for M, xi, ui, vi in zip(self._factor_leaves, xs, us, vs)]
        out = vals[0]
        for val in vals[1:]:
            out = out + val
        return out

    def norm(self, x: Any, u: Any) -> Array:
        xs = self._flatten_like(x, "point")
        us = self._flatten_like(u, "tangent vector")
        factor_norms = jnp.broadcast_arrays(
            *(M.norm(xi, ui) for M, xi, ui in zip(self._factor_leaves, xs, us))
        )
        return stable_norm(jnp.stack(factor_norms, axis=-1), axis=-1)

    def lincomb(self, x: Any, *terms: Any) -> Any:
        if len(terms) % 2 != 0:
            raise ValueError("lincomb expects coefficient/vector pairs.")
        xs = self._flatten_like(x, "point")
        flattened_vectors = [self._flatten_like(vec, "tangent vector") for vec in terms[1::2]]
        out = []
        for i, (M, xi) in enumerate(zip(self._factor_leaves, xs)):
            subterms = []
            for coeff, leaves in zip(terms[0::2], flattened_vectors):
                subterms.extend([coeff, leaves[i]])
            out.append(M.lincomb(xi, *subterms))
        return self._unflatten(out)

    def exp(self, x: Any, u: Any) -> Any:
        xs = self._flatten_like(x, "point")
        us = self._flatten_like(u, "tangent vector")
        return self._unflatten(M.exp(xi, ui) for M, xi, ui in zip(self._factor_leaves, xs, us))

    def retr(self, x: Any, u: Any, t: float | Array = 1.0) -> Any:
        xs = self._flatten_like(x, "point")
        us = self._flatten_like(u, "tangent vector")
        out = []
        for M, xi, ui in zip(self._factor_leaves, xs, us):
            out.append(M.retr(xi, ui, t))
        return self._unflatten(out)

    def log(self, x: Any, y: Any) -> Any:
        xs = self._flatten_like(x, "point")
        ys = self._flatten_like(y, "point")
        return self._unflatten(M.log(xi, yi) for M, xi, yi in zip(self._factor_leaves, xs, ys))

    def squared_dist(self, x: Any, y: Any) -> Array:
        xs = self._flatten_like(x, "point")
        ys = self._flatten_like(y, "point")
        vals = [M.squared_dist(xi, yi) for M, xi, yi in zip(self._factor_leaves, xs, ys)]
        out = vals[0]
        for val in vals[1:]:
            out = out + val
        return jnp.maximum(out, 0.0)

    def dist(self, x: Any, y: Any) -> Array:
        xs = self._flatten_like(x, "point")
        ys = self._flatten_like(y, "point")
        factor_distances = jnp.broadcast_arrays(
            *(M.dist(xi, yi) for M, xi, yi in zip(self._factor_leaves, xs, ys))
        )
        return stable_norm(jnp.stack(factor_distances, axis=-1), axis=-1)

    def transport(self, x: Any, y: Any, u: Any) -> Any:
        xs = self._flatten_like(x, "point")
        ys = self._flatten_like(y, "point")
        us = self._flatten_like(u, "tangent vector")
        out = []
        for M, xi, yi, ui in zip(self._factor_leaves, xs, ys, us):
            out.append(M.transport(xi, yi, ui))
        return self._unflatten(out)

    def pair_mean(self, x: Any, y: Any) -> Any:
        xs = self._flatten_like(x, "point")
        ys = self._flatten_like(y, "point")
        out = []
        for M, xi, yi in zip(self._factor_leaves, xs, ys):
            out.append(M.pair_mean(xi, yi))
        return self._unflatten(out)

    def egrad_to_rgrad(self, x: Any, egrad: Any) -> Any:
        xs = self._flatten_like(x, "point")
        gs = self._flatten_like(egrad, "Euclidean gradient")
        return self._unflatten(
            M.egrad_to_rgrad(xi, gi) for M, xi, gi in zip(self._factor_leaves, xs, gs)
        )

    def ehess_to_rhess(self, x: Any, egrad: Any, ehess_vec: Any, u: Any) -> Any:
        xs = self._flatten_like(x, "point")
        gs = self._flatten_like(egrad, "Euclidean gradient")
        hvs = self._flatten_like(ehess_vec, "Euclidean Hessian-vector product")
        us = self._flatten_like(u, "tangent vector")
        out = []
        for M, xi, gi, hvi, ui in zip(self._factor_leaves, xs, gs, hvs, us):
            out.append(M.ehess_to_rhess(xi, gi, hvi, ui))
        return self._unflatten(out)

    def random_point(self, key: Array, sample_shape: Shape = ()) -> Any:
        keys = jax.random.split(key, len(self._factor_leaves))
        sample_shape = as_sample_shape(sample_shape)
        return self._unflatten(
            M.random_point(k, sample_shape=sample_shape) for M, k in zip(self._factor_leaves, keys)
        )

    def random_tangent(
        self,
        key: Array,
        x: Any,
        *,
        scale: float | Array = 1.0,
        normalize: bool = False,
    ) -> Any:
        xs = self._flatten_like(x, "point")
        keys = jax.random.split(key, len(self._factor_leaves))
        leaf_scale = 1.0 if normalize else scale
        leaves = [
            M.random_tangent(k, xi, scale=leaf_scale, normalize=False)
            for M, k, xi in zip(self._factor_leaves, keys, xs)
        ]
        u = self._unflatten(leaves)
        if normalize:
            nrm = self.norm(x, u)
            safe_nrm = jnp.where(nrm > 0.0, nrm, jnp.ones_like(nrm))
            normalization = jnp.where(nrm > 0.0, 1.0 / safe_nrm, 1.0)
            u = self._scale_tangent(u, normalization)
            u = self._scale_tangent(u, scale)
        return u


__all__ = ["Product"]
