"""Specialized optimization problem contracts."""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from geojax.geometry.base import validate_integer

from .minimize import Array, Minimize, lincomb, validate_tangent

ResidualFn = Callable[[Array], Any]
JacobianVecFn = Callable[[Array, Array], Any]
AdjointJacobianFn = Callable[[Array, Any], Array]
LossFn = Callable[[Array, Array], Array]


def tree_vdot(x: Any, y: Any) -> Array:
    """Euclidean inner product between residual pytrees."""
    x_leaves, x_tree = jax.tree_util.tree_flatten(x)
    y_leaves, y_tree = jax.tree_util.tree_flatten(y)
    if x_tree != y_tree:
        raise ValueError("Residual pytrees must have identical structures.")
    if len(x_leaves) != len(y_leaves):
        raise ValueError("Residual pytrees must have the same number of leaves.")
    for left, right in zip(x_leaves, y_leaves):
        if jnp.shape(left) != jnp.shape(right):
            raise ValueError(
                "Corresponding residual leaves must have identical shapes; "
                f"received {jnp.shape(left)} and {jnp.shape(right)}."
            )
    products = [jnp.real(jnp.vdot(a, b)) for a, b in zip(x_leaves, y_leaves)]
    if not products:
        return jnp.asarray(0.0)
    out = products[0]
    for product in products[1:]:
        out = out + product
    return out


def _contains_tracer(value: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value))


def _validate_real_finite_leaves(value: Any, *, name: str) -> Any:
    """Validate static residual properties and eager numerical values."""
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves:
        raise ValueError(f"{name} must be a nonempty pytree of arrays.")
    for index, leaf in enumerate(leaves):
        array = jnp.asarray(leaf)
        if jnp.iscomplexobj(array):
            raise ValueError(
                f"{name} leaf {index} must be real-valued; "
                "LeastSquares models residual maps into a real Euclidean space."
            )
        if not isinstance(array, jax.core.Tracer) and not bool(jnp.all(jnp.isfinite(array))):
            raise FloatingPointError(f"{name} must contain only finite values.")
    return value


def _validate_residual_like(reference: Any, value: Any, *, name: str) -> Any:
    """Validate a residual-space vector against a reference residual pytree."""
    reference_leaves, reference_tree = jax.tree_util.tree_flatten(reference)
    value_leaves, value_tree = jax.tree_util.tree_flatten(value)
    if reference_tree != value_tree or len(reference_leaves) != len(value_leaves):
        raise ValueError(f"{name} must match the residual pytree structure.")
    for index, (reference_leaf, value_leaf) in enumerate(zip(reference_leaves, value_leaves)):
        if jnp.shape(reference_leaf) != jnp.shape(value_leaf):
            raise ValueError(
                f"{name} leaf {index} must have shape {jnp.shape(reference_leaf)}; "
                f"received {jnp.shape(value_leaf)}."
            )
    return _validate_real_finite_leaves(value, name=name)


def _validate_nonnegative_scalar(value: Any, *, name: str) -> Array:
    """Return a real scalar, checking its value whenever it is concrete."""
    array = jnp.asarray(value)
    if array.shape != () or jnp.iscomplexobj(array) or jnp.issubdtype(array.dtype, jnp.bool_):
        raise TypeError(f"{name} must be a real scalar.")
    if not _contains_tracer(array):
        scalar = float(array)
        if not jnp.isfinite(array) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    return array


class LeastSquares(Minimize):
    """Nonlinear least-squares problem on a manifold.

    The objective is ``0.5 * ||residual(x)||_2**2``. Residuals may be real
    arrays or arbitrary pytrees of real arrays. Jacobian-vector and
    adjoint-Jacobian products are obtained with JAX autodiff unless callbacks
    are supplied.
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
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if jacobian_vec is not None and not callable(jacobian_vec):
            raise TypeError("jacobian_vec must be callable or None.")
        if adjoint_jacobian is not None and not callable(adjoint_jacobian):
            raise TypeError("adjoint_jacobian must be callable or None.")
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
        return _validate_real_finite_leaves(self.residual(x), name="residual")

    def residual_norm(self, x: Array) -> Array:
        """Return the Euclidean norm of the residual."""
        residual = self.residual_value(x)
        return jnp.sqrt(jnp.maximum(tree_vdot(residual, residual), 0.0))

    def jacobian_vec(self, x: Array, u: Array) -> Any:
        """Apply the residual Jacobian to tangent vector ``u``."""
        validate_tangent(self.M, x, u, name="Jacobian direction")
        if self._jacobian_vec_callback is not None:
            residual = self.residual_value(x)
            result = self._jacobian_vec_callback(x, u)
        else:
            residual, result = jax.jvp(self.residual_value, (x,), (u,))
            residual = _validate_real_finite_leaves(residual, name="residual")
        return _validate_residual_like(residual, result, name="Jacobian-vector product")

    def adjoint_jacobian(self, x: Array, z: Any) -> Array:
        """Apply the metric adjoint residual Jacobian and return a tangent vector.

        A user callback must return the Riemannian adjoint in ``T_x M``. The
        autodiff path first forms the ambient pullback and then converts it with
        ``M.egrad_to_rgrad``.
        """
        if self._adjoint_jacobian_callback is not None:
            residual = self.residual_value(x)
            z = _validate_residual_like(residual, z, name="adjoint cotangent")
            tangent = self._adjoint_jacobian_callback(x, z)
            return validate_tangent(self.M, x, tangent, name="adjoint Jacobian output")
        residual, pullback = jax.vjp(self.residual_value, x)
        residual = _validate_real_finite_leaves(residual, name="residual")
        z = _validate_residual_like(residual, z, name="adjoint cotangent")
        ambient = pullback(z)[0]
        return validate_tangent(
            self.M,
            x,
            self.M.egrad_to_rgrad(x, ambient),
            name="adjoint Jacobian output",
        )

    def normal_operator(self, x: Array, u: Array, damping: float = 0.0) -> Array:
        """Apply ``J(x)^* J(x) + damping * I`` to ``u``."""
        validate_tangent(self.M, x, u, name="normal-operator direction")
        damping = _validate_nonnegative_scalar(damping, name="damping")
        normal = self.adjoint_jacobian(x, self.jacobian_vec(x, u))
        return lincomb(self.M, x, 1.0, normal, damping, u)

    def cost_and_grad(self, x: Array) -> tuple[Array, Array]:
        """Evaluate the least-squares objective and gradient with one residual trace."""
        if self._adjoint_jacobian_callback is not None:
            residual = self.residual_value(x)
            cost = 0.5 * tree_vdot(residual, residual)
            gradient = self._adjoint_jacobian_callback(x, residual)
            return cost, validate_tangent(self.M, x, gradient, name="adjoint Jacobian output")
        residual, pullback = jax.vjp(self.residual_value, x)
        residual = _validate_real_finite_leaves(residual, name="residual")
        cost = 0.5 * tree_vdot(residual, residual)
        ambient = pullback(residual)[0]
        gradient = self.M.egrad_to_rgrad(x, ambient)
        return cost, validate_tangent(
            self.M,
            x,
            gradient,
            name="adjoint Jacobian output",
        )

    def _least_squares_cost(self, x: Array) -> Array:
        residual = self.residual_value(x)
        return 0.5 * tree_vdot(residual, residual)

    def _least_squares_gradient(self, x: Array) -> Array:
        if self._adjoint_jacobian_callback is not None:
            residual = self.residual_value(x)
            tangent = self._adjoint_jacobian_callback(x, residual)
            return validate_tangent(self.M, x, tangent, name="adjoint Jacobian output")
        residual, pullback = jax.vjp(self.residual_value, x)
        residual = _validate_real_finite_leaves(residual, name="residual")
        ambient = pullback(residual)[0]
        return validate_tangent(
            self.M,
            x,
            self.M.egrad_to_rgrad(x, ambient),
            name="adjoint Jacobian output",
        )


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
        if not callable(loss):
            raise TypeError("loss must be callable.")
        num_terms = validate_integer(num_terms, name="num_terms")
        if num_terms <= 0:
            raise ValueError("num_terms must be positive.")
        self.loss = loss
        self.num_terms = num_terms
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
        if values.shape != indices.shape:
            raise ValueError("loss(x, index) must return a real scalar for each term.")
        if jnp.iscomplexobj(values):
            raise ValueError("loss(x, index) must return a real scalar.")
        return jnp.mean(values)

    def _full_cost(self, x: Array) -> Array:
        return self._batch_cost(x, jnp.arange(self.num_terms))

    def batch_cost_and_grad(self, x: Array, indices: Array) -> tuple[Array, Array]:
        """Return mini-batch cost and Riemannian gradient."""

        indices = jnp.asarray(indices)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("indices must be a nonempty one-dimensional array.")
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("indices must contain integers.")
        if not _contains_tracer(indices) and not bool(
            jnp.all((indices >= 0) & (indices < self.num_terms))
        ):
            raise ValueError("indices must lie in [0, num_terms).")

        def objective(point: Array) -> Array:
            return self._batch_cost(point, indices)

        cost, ambient_gradient = jax.value_and_grad(objective)(x)
        gradient = self.M.egrad_to_rgrad(x, ambient_gradient)
        return cost, validate_tangent(self.M, x, gradient, name="mini-batch Riemannian gradient")

    def sample_batch(self, key: Array, batch_size: int, *, replace: bool = True) -> Array:
        """Draw uniformly distributed term indices."""
        batch_size = validate_integer(batch_size, name="batch_size")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not replace and batch_size > self.num_terms:
            raise ValueError("batch_size cannot exceed num_terms without replacement.")
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean.")
        try:
            jax.random.key_data(key)
        except (TypeError, ValueError) as exc:
            raise TypeError("key must be a JAX random key.") from exc
        return jax.random.choice(
            key,
            self.num_terms,
            shape=(batch_size,),
            replace=replace,
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
