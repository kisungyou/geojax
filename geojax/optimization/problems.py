"""Specialized optimization problem contracts."""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .minimize import Array, Minimize, lincomb

ResidualFn = Callable[[Array], Any]
JacobianVecFn = Callable[[Array, Array], Any]
AdjointJacobianFn = Callable[[Array, Any], Array]
LossFn = Callable[[Array, Array], Array]


def tree_vdot(x: Any, y: Any) -> Array:
    """Euclidean inner product between residual pytrees."""
    products = [
        jnp.real(jnp.vdot(a, b))
        for a, b in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y))
    ]
    if not products:
        return jnp.asarray(0.0)
    out = products[0]
    for product in products[1:]:
        out = out + product
    return out


class LeastSquares(Minimize):
    """Nonlinear least-squares problem on a manifold.

    The objective is ``0.5 * ||residual(x)||_2**2``. Residuals may be arrays or
    arbitrary pytrees of arrays. Jacobian-vector and adjoint-Jacobian products
    are obtained with JAX autodiff unless callbacks are supplied.
    """

    def __init__(
        self,
        *,
        M: Any,
        residual: ResidualFn,
        x0: Optional[Array] = None,
        solver: Optional[Any] = None,
        key: Optional[Array | int] = None,
        jacobian_vec: Optional[JacobianVecFn] = None,
        adjoint_jacobian: Optional[AdjointJacobianFn] = None,
        precon: Optional[Callable[[Array, Array], Array]] = None,
    ) -> None:
        self.residual = residual
        self._jacobian_vec_callback = jacobian_vec
        self._adjoint_jacobian_callback = adjoint_jacobian
        super().__init__(
            M=M,
            cost=self._least_squares_cost,
            x0=x0,
            solver=solver,
            key=key,
            grad=self._least_squares_gradient,
            precon=precon,
        )

    def residual_value(self, x: Array) -> Any:
        """Evaluate the residual pytree."""
        return self.residual(x)

    def residual_norm(self, x: Array) -> Array:
        """Return the Euclidean norm of the residual."""
        residual = self.residual_value(x)
        return jnp.sqrt(jnp.maximum(tree_vdot(residual, residual), 0.0))

    def jacobian_vec(self, x: Array, u: Array) -> Any:
        """Apply the residual Jacobian to tangent vector ``u``."""
        if self._jacobian_vec_callback is not None:
            return self._jacobian_vec_callback(x, u)
        return jax.jvp(self.residual, (x,), (u,))[1]

    def adjoint_jacobian(self, x: Array, z: Any) -> Array:
        """Apply the adjoint residual Jacobian and return a tangent vector."""
        if self._adjoint_jacobian_callback is not None:
            tangent = self._adjoint_jacobian_callback(x, z)
            return self.M.tangent_project(x, tangent)
        _, pullback = jax.vjp(self.residual, x)
        ambient = pullback(z)[0]
        return self.M.egrad_to_rgrad(x, ambient)

    def normal_operator(self, x: Array, u: Array, damping: float = 0.0) -> Array:
        """Apply ``J(x)^* J(x) + damping * I`` to ``u``."""
        normal = self.adjoint_jacobian(x, self.jacobian_vec(x, u))
        if damping == 0.0:
            return normal
        return lincomb(self.M, x, 1.0, normal, float(damping), u)

    def _least_squares_cost(self, x: Array) -> Array:
        residual = self.residual_value(x)
        return 0.5 * tree_vdot(residual, residual)

    def _least_squares_gradient(self, x: Array) -> Array:
        residual = self.residual_value(x)
        return self.adjoint_jacobian(x, residual)


class FiniteSum(Minimize):
    """Finite-sum problem ``mean_i loss(x, i)`` for stochastic solvers."""

    def __init__(
        self,
        *,
        M: Any,
        loss: LossFn,
        num_terms: int,
        x0: Optional[Array] = None,
        solver: Optional[Any] = None,
        key: Optional[Array | int] = None,
        precon: Optional[Callable[[Array, Array], Array]] = None,
    ) -> None:
        if int(num_terms) <= 0:
            raise ValueError("num_terms must be positive.")
        self.loss = loss
        self.num_terms = int(num_terms)
        super().__init__(
            M=M,
            cost=self._full_cost,
            x0=x0,
            solver=solver,
            key=key,
            precon=precon,
        )

    def _batch_cost(self, x: Array, indices: Array) -> Array:
        values = jax.vmap(lambda index: self.loss(x, index))(indices)
        return jnp.mean(values)

    def _full_cost(self, x: Array) -> Array:
        return self._batch_cost(x, jnp.arange(self.num_terms))

    def batch_cost_and_grad(self, x: Array, indices: Array) -> tuple[Array, Array]:
        """Return mini-batch cost and Riemannian gradient."""
        def objective(point: Array) -> Array:
            return self._batch_cost(point, indices)

        cost, ambient_gradient = jax.value_and_grad(objective)(x)
        return cost, self.M.egrad_to_rgrad(x, ambient_gradient)

    def sample_batch(self, key: Array, batch_size: int, *, replace: bool = True) -> Array:
        """Draw uniformly distributed term indices."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not replace and batch_size > self.num_terms:
            raise ValueError("batch_size cannot exceed num_terms without replacement.")
        return jax.random.choice(
            key,
            self.num_terms,
            shape=(batch_size,),
            replace=bool(replace),
        )


__all__ = [
    "LeastSquares",
    "FiniteSum",
    "ResidualFn",
    "JacobianVecFn",
    "AdjointJacobianFn",
    "LossFn",
    "tree_vdot",
]
