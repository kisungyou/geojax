"""Core minimization problem container and shared solver utilities.

This module intentionally separates the optimization problem description from
individual class-style solvers. A solver consumes a :class:`Minimize` object
and returns the common GeoJAX optimization triple

    sol, final_cost, info

where ``sol`` is the final point, ``final_cost`` is a Python float and ``info``
is a list of per-iteration records.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from numbers import Integral
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import math
import time

import jax
import jax.numpy as jnp

from geojax.geometry.base import validate_boolean, validate_integer, validate_nonnegative

Array = Any
CostFn = Callable[[Array], Array]
GradFn = Callable[[Array], Array]
PreconFn = Callable[[Array, Array], Array]
HessVecFn = Callable[[Array, Array], Array]
StatsFn = Callable[[Any, Array, "InfoEntry"], Dict[str, Any]]
StopFn = Callable[[Any, Array, "InfoEntry"], Tuple[bool, str]]


class Minimize:
    """Riemannian minimization problem.

    Parameters
    ----------
    M:
        Manifold object.
    cost:
        Scalar objective ``cost(x)`` to minimize.
    grad:
        Optional Riemannian gradient ``grad(x)``. If supplied, it takes priority
        over ``egrad`` and autodiff.
    egrad:
        Optional ambient Euclidean gradient ``egrad(x)``. It is converted to a
        Riemannian gradient using ``M.egrad_to_rgrad``.
    precon:
        Optional preconditioner ``precon(x, grad)``. If omitted, the identity
        preconditioner is used.
    ehess_vec:
        Optional ambient Euclidean Hessian-vector product. Automatic conversion
        requires the geometry to advertise exact ``ehess_to_rhess`` support.
    rhess_vec:
        Optional Riemannian Hessian-vector product. Supply this for second-order
        methods whenever the geometry does not advertise an exact automatic
        conversion.
    """

    def __init__(
        self,
        *,
        M: Any,
        cost: CostFn,
        x0: Optional[Array] = None,
        solver: Optional[Any] = None,
        key: Optional[Array | int] = None,
        grad: Optional[GradFn] = None,
        egrad: Optional[GradFn] = None,
        precon: Optional[PreconFn] = None,
        ehess_vec: Optional[HessVecFn] = None,
        rhess_vec: Optional[HessVecFn] = None,
    ) -> None:
        if M is None:
            raise ValueError("M must be a manifold object.")
        if not callable(cost):
            raise TypeError("cost must be callable.")
        for name, callback in {
            "grad": grad,
            "egrad": egrad,
            "precon": precon,
            "ehess_vec": ehess_vec,
            "rhess_vec": rhess_vec,
        }.items():
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None.")
        self.M = M
        self.cost = cost
        self.x0 = x0
        self.solver = solver
        self.key = key
        self.grad = grad
        self.egrad = egrad
        self.precon = precon
        self._ehess_vec = ehess_vec
        self._rhess_vec = rhess_vec
        self._key = self._coerce_key(key)

    @staticmethod
    def _coerce_key(key: Optional[Array | int]) -> Optional[Array]:
        return coerce_key(key)

    def split_key(self) -> Array:
        """Return a fresh JAX PRNG key and advance the problem key."""
        if self._key is None:
            self._key = jax.random.key(0)
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def solve(self) -> tuple[Array, float, List["InfoEntry"]]:
        """Solve the problem using the configured class-style solver."""
        self.x0 = initial_point(
            self,
            self.x0,
            self.split_key() if self.x0 is None else None,
        )

        if self.solver is None:
            raise ValueError("Minimize.solve() requires a solver.")
        if not hasattr(self.solver, "solve"):
            raise TypeError("solver must be a class-style solver with solve(problem).")
        return self.solver.solve(self)

    def ehess_vec(self, x: Array, u: Array) -> Array:
        """Ambient Euclidean Hessian-vector product.

        When only a Riemannian ``grad`` callback is available, this method
        returns its directional derivative instead. ``rhess_vec`` handles that
        path separately through the geometry's advertised connection support.
        """
        validate_tangent(self.M, x, u, name="Hessian direction")
        if self._ehess_vec is not None:
            result = self._ehess_vec(x, u)
        elif self.egrad is not None:
            result = jax.jvp(self.egrad, (x,), (u,))[1]
        elif self.grad is not None:
            result = jax.jvp(self.grad, (x,), (u,))[1]
        else:
            result = jax.jvp(jax.grad(self.cost), (x,), (u,))[1]
        return _validate_vector_like(x, result, name="Euclidean Hessian-vector product")

    def rhess_vec(self, x: Array, u: Array) -> Array:
        """Riemannian Hessian-vector product."""
        validate_tangent(self.M, x, u, name="Hessian direction")
        if self._rhess_vec is not None:
            return validate_tangent(
                self.M,
                x,
                self._rhess_vec(x, u),
                name="Riemannian Hessian-vector product",
            )

        def require_exact(operation: str, description: str) -> None:
            if hasattr(self.M, "operation_kind"):
                try:
                    kind = self.M.operation_kind(operation)
                except ValueError:
                    kind = "unknown"
            else:
                attribute = (
                    "hessian_conversion_is_exact"
                    if operation == "ehess_to_rhess"
                    else "riemannian_gradient_jvp_is_exact"
                )
                kind = "exact" if bool(getattr(self.M, attribute, False)) else "unknown"
            if kind != "exact":
                geometry_name = type(self.M).__name__
                raise ValueError(
                    f"{geometry_name} does not advertise an exact {description}. "
                    "Supply rhess_vec explicitly for second-order optimization."
                )

        if self._ehess_vec is not None:
            require_exact("ehess_to_rhess", "ambient-to-Riemannian Hessian conversion")
            egrad = _validate_vector_like(
                x,
                self.egrad(x) if self.egrad is not None else jax.grad(self.cost)(x),
                name="Euclidean gradient",
            )
            result = self.M.ehess_to_rhess(x, egrad, self.ehess_vec(x, u), u)
            return validate_tangent(self.M, x, result, name="Riemannian Hessian-vector product")

        if self.egrad is not None:
            require_exact("ehess_to_rhess", "ambient-to-Riemannian Hessian conversion")
            egrad = _validate_vector_like(x, self.egrad(x), name="Euclidean gradient")
            ehess_u = self.ehess_vec(x, u)
            result = self.M.ehess_to_rhess(x, egrad, ehess_u, u)
            return validate_tangent(self.M, x, result, name="Riemannian Hessian-vector product")

        if self.grad is not None:
            require_exact("rgrad_jvp", "Riemannian-gradient JVP conversion")
            gradient, grad_jvp = jax.jvp(self.grad, (x,), (u,))
            validate_tangent(self.M, x, gradient, name="Riemannian gradient")
            _validate_vector_like(x, grad_jvp, name="Riemannian-gradient JVP")
            result = self.M.tangent_project(x, grad_jvp)
            return validate_tangent(self.M, x, result, name="Riemannian Hessian-vector product")

        require_exact("ehess_to_rhess", "ambient-to-Riemannian Hessian conversion")
        egrad = _validate_vector_like(x, jax.grad(self.cost)(x), name="Euclidean gradient")
        ehess_u = self.ehess_vec(x, u)
        result = self.M.ehess_to_rhess(x, egrad, ehess_u, u)
        return validate_tangent(self.M, x, result, name="Riemannian Hessian-vector product")

    def hessian_operator(self, x: Array) -> Callable[[Array], Array]:
        """Return ``u -> rhess_vec(x, u)``."""
        return lambda u: self.rhess_vec(x, u)


@dataclass(frozen=True)
class LineSearchStats:
    """Common diagnostics returned by a line-search strategy."""

    costevals: int
    stepsize: float
    alpha: float
    accepted: bool
    gradevals: int = 0
    method: str = ""
    reason: str = ""


@dataclass(frozen=True)
class InfoEntry:
    """Per-iteration optimization statistics.

    This mirrors the useful parts of Manopt's ``info`` struct-array in a Python
    dataclass. ``beta`` is used by conjugate-gradient methods and left as
    ``None`` by algorithms that do not define a beta parameter.
    """

    iter: int
    cost: float
    gradnorm: float
    stepsize: float
    time: float
    linesearch: Optional[LineSearchStats] = None
    beta: Optional[float] = None
    reason: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def as_options(cls: type, options: Optional[Any]) -> Any:
    """Build an options dataclass from ``None``, an existing instance or mapping."""
    if options is None:
        return cls()
    if isinstance(options, cls):
        return options
    if isinstance(options, Mapping):
        valid = {f.name for f in fields(cls)}
        unknown = set(options) - valid
        if unknown:
            raise ValueError(f"Unknown option(s) for {cls.__name__}: {sorted(unknown)}")
        return cls(**dict(options))
    raise TypeError(f"options must be None, a mapping, or {cls.__name__}")


def as_float(x: Array) -> float:
    """Convert a scalar JAX value to a Python float."""
    return float(jnp.asarray(x))


def coerce_key(key: Optional[Array | int], *, name: str = "key") -> Optional[Array]:
    """Normalize an integer seed or validate a scalar JAX random key."""
    if key is None:
        return None
    if isinstance(key, bool):
        raise TypeError(f"{name} must be an integer seed or JAX random key.")
    if isinstance(key, Integral):
        return jax.random.key(int(key))
    try:
        key_data = jax.random.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer seed or JAX random key.") from exc
    if jnp.shape(key_data) != (2,):
        raise TypeError(f"{name} must be a scalar JAX random key, not a key batch.")
    return key


def tree_scale(a: Any, x: Any) -> Any:
    return jax.tree_util.tree_map(lambda z: a * z, x)


def tree_add(x: Any, y: Any) -> Any:
    return jax.tree_util.tree_map(lambda a, b: a + b, x, y)


def tree_sub(x: Any, y: Any) -> Any:
    return jax.tree_util.tree_map(lambda a, b: a - b, x, y)


def tree_neg(x: Any) -> Any:
    return jax.tree_util.tree_map(lambda z: -z, x)


def tree_zeros_like(x: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, x)


def tree_lincomb(*terms: Any) -> Any:
    """Linear combination of pytrees from coefficient/vector pairs."""
    if len(terms) % 2 != 0:
        raise ValueError("tree_lincomb expects coefficient/vector pairs.")
    out = None
    for coeff, vec in zip(terms[0::2], terms[1::2]):
        term = tree_scale(coeff, vec)
        out = term if out is None else tree_add(out, term)
    if out is None:
        raise ValueError("tree_lincomb requires at least one coefficient/vector pair.")
    return out


def _validate_vector_like(reference: Any, value: Any, *, name: str) -> Any:
    """Validate the structure, shapes and numerical values of a vector pytree."""
    reference_leaves, reference_tree = jax.tree_util.tree_flatten(reference)
    value_leaves, value_tree = jax.tree_util.tree_flatten(value)
    if reference_tree != value_tree:
        raise ValueError(f"{name} must have the same pytree structure as the point.")
    if len(reference_leaves) != len(value_leaves):
        raise ValueError(f"{name} must have the same number of leaves as the point.")
    if not value_leaves:
        raise ValueError(f"{name} must be a nonempty pytree of arrays.")
    for index, (point_leaf, vector_leaf) in enumerate(zip(reference_leaves, value_leaves)):
        point_array = jnp.asarray(point_leaf)
        vector_array = jnp.asarray(vector_leaf)
        if point_array.shape != vector_array.shape:
            raise ValueError(
                f"{name} leaf {index} must have shape {point_array.shape}; "
                f"received {vector_array.shape}."
            )
        if jnp.iscomplexobj(vector_array):
            raise ValueError(f"{name} must be real-valued.")
        if not isinstance(vector_array, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(vector_array))
        ):
            raise FloatingPointError(f"{name} must contain only finite values.")
    return value


def validate_tangent(M: Any, x: Any, value: Any, *, name: str) -> Any:
    """Validate a tangent-valued callback result at ``x``."""
    _validate_vector_like(x, value, name=name)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves((x, value))):
        return value
    is_tangent = getattr(M, "is_tangent", None)
    if callable(is_tangent) and not bool(jnp.all(jnp.asarray(is_tangent(x, value)))):
        raise ValueError(f"{name} is not tangent at the supplied manifold point.")
    return value


def get(problem: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dataclass/object or mapping."""
    if isinstance(problem, Mapping):
        return problem.get(name, default)
    return getattr(problem, name, default)


def require(problem: Any, name: str) -> Any:
    """Read a required problem field."""
    value = get(problem, name, None)
    if value is None:
        raise ValueError(f"minimization problem must define field {name!r}")
    return value


require_field = require


def initial_point(problem: Any, x: Optional[Array], key: Optional[Array]) -> Array:
    """Return a projected user initial point or draw a random manifold point."""
    M = require(problem, "M")
    if x is None:
        if key is None:
            key = jax.random.key(0)
        if not hasattr(M, "random_point"):
            raise ValueError(
                "No initial point was supplied and problem.M has no random_point(key) method."
            )
        point = M.random_point(key)
    else:
        point = M.project(x) if hasattr(M, "project") else x
    return validate_point(M, point, name="initial manifold point")


def validate_point(M: Any, point: Any, *, name: str = "manifold point") -> Any:
    """Reject empty, nonfinite, or off-manifold points without repairing them."""
    leaves = jax.tree_util.tree_leaves(point)
    if not leaves:
        raise ValueError(f"The {name} must contain at least one array leaf.")
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return point
    if not all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves):
        raise ValueError(f"The {name} must contain only finite values.")
    if hasattr(M, "belongs") and not bool(jnp.all(jnp.asarray(M.belongs(point)))):
        raise ValueError(f"The {name} does not belong to M.")
    return point


def require_geometry_methods(M: Any, *names: str, context: str) -> None:
    """Require callable geometric primitives for an optimization algorithm."""
    missing = [name for name in names if not callable(getattr(M, name, None))]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{context} requires geometry method(s): {joined}.")


def retract(M: Any, x: Array, direction: Array, alpha: float | Array) -> Array:
    """Take a trial step using ``M.retr`` if available, otherwise ``M.exp``."""
    validate_tangent(M, x, direction, name="retraction direction")
    alpha_array = jnp.asarray(alpha)
    if alpha_array.shape != () or jnp.iscomplexobj(alpha_array):
        raise ValueError("The retraction multiplier must be a real scalar.")
    if not isinstance(alpha_array, jax.core.Tracer) and not math.isfinite(as_float(alpha_array)):
        raise ValueError("The retraction multiplier must be finite.")
    if hasattr(M, "retr"):
        point = M.retr(x, direction, alpha)
    else:
        point = M.exp(x, tree_lincomb(alpha, direction))
    return validate_point(M, point, name="retraction trial point")


def inner(M: Any, x: Array, u: Array, v: Array) -> Array:
    """Return the Riemannian inner product from the manifold object."""
    return M.inner(x, u, v)


def lincomb(M: Any, x: Array, *terms: Any) -> Array:
    """Linear combination of tangent vectors, projected if needed."""
    if hasattr(M, "lincomb"):
        out = M.lincomb(x, *terms)
    else:
        out = tree_lincomb(*terms)
        if hasattr(M, "tangent_project"):
            out = M.tangent_project(x, out)
        elif hasattr(M, "proj"):
            out = M.proj(x, out)
    return validate_tangent(M, x, out, name="tangent linear combination")


def transport(M: Any, x: Array, y: Array, u: Array) -> Array:
    """Transport a tangent vector from ``x`` to ``y``."""
    validate_tangent(M, x, u, name="vector-transport input")
    validate_point(M, y, name="vector-transport endpoint")
    if hasattr(M, "transport"):
        out = M.transport(x, y, u)
    elif hasattr(M, "transp"):
        out = M.transp(x, y, u)
    elif hasattr(M, "tangent_project"):
        out = M.tangent_project(y, u)
    elif hasattr(M, "proj"):
        out = M.proj(y, u)
    else:
        raise ValueError("Manifold must provide transport, tangent_project or proj.")
    return validate_tangent(M, y, out, name="vector-transport output")


def pair_mean(M: Any, x: Array, y: Array) -> Array:
    """Return a midpoint-like mean between two manifold points."""
    if hasattr(M, "pair_mean"):
        point = M.pair_mean(x, y)
    elif hasattr(M, "exp") and hasattr(M, "log"):
        displacement = validate_tangent(M, x, M.log(x, y), name="pair-mean logarithm")
        point = M.exp(x, tree_lincomb(0.5, displacement))
    else:
        raise ValueError("Manifold must provide pair_mean or both exp and log.")
    return validate_point(M, point, name="pair mean")


def _validated_cost_output(value: Any) -> Array:
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"cost must return a scalar; received shape {array.shape}.")
    if jnp.iscomplexobj(array):
        raise ValueError("cost must return a real scalar.")
    return array


def cost_and_grad(problem: Any, x: Array) -> tuple[Array, Array]:
    """Return ``cost(x)`` and the Riemannian gradient at ``x``."""
    M = require(problem, "M")
    validate_point(M, x, name="objective evaluation point")
    cost_fn = require(problem, "cost")
    combined_fn = get(problem, "cost_and_grad", None)
    grad_fn = get(problem, "grad", None)
    egrad_fn = get(problem, "egrad", None)

    if combined_fn is not None:
        c, g = combined_fn(x)
        c = _validated_cost_output(c)
        return c, validate_tangent(M, x, g, name="Riemannian gradient")

    if grad_fn is not None:
        c = _validated_cost_output(cost_fn(x))
        g = validate_tangent(M, x, grad_fn(x), name="Riemannian gradient")
        return c, g

    if egrad_fn is not None:
        c = _validated_cost_output(cost_fn(x))
        eg = _validate_vector_like(x, egrad_fn(x), name="Euclidean gradient")
        g = M.egrad_to_rgrad(x, eg)
        return c, validate_tangent(M, x, g, name="Riemannian gradient")

    def scalar_cost(point: Any) -> Array:
        return _validated_cost_output(cost_fn(point))

    c, eg = jax.value_and_grad(scalar_cost)(x)
    g = M.egrad_to_rgrad(x, eg)
    return c, validate_tangent(M, x, g, name="Riemannian gradient")


def gradient_value(problem: Any, x: Array) -> Array:
    """Return only the Riemannian gradient at ``x``."""
    M = require(problem, "M")
    validate_point(M, x, name="gradient evaluation point")
    cost_fn = require(problem, "cost")
    grad_fn = get(problem, "grad", None)
    egrad_fn = get(problem, "egrad", None)
    if grad_fn is not None:
        return validate_tangent(M, x, grad_fn(x), name="Riemannian gradient")
    if egrad_fn is not None:
        eg = _validate_vector_like(x, egrad_fn(x), name="Euclidean gradient")
        return validate_tangent(M, x, M.egrad_to_rgrad(x, eg), name="Riemannian gradient")

    def scalar_cost(point: Any) -> Array:
        return _validated_cost_output(cost_fn(point))

    return validate_tangent(
        M,
        x,
        M.egrad_to_rgrad(x, jax.grad(scalar_cost)(x)),
        name="Riemannian gradient",
    )


def cost_value(problem: Any, x: Array) -> Array:
    """Return ``cost(x)``."""
    validate_point(require(problem, "M"), x, name="objective evaluation point")
    return _validated_cost_output(require(problem, "cost")(x))


def validate_line_search_result(problem: Any, result: Any) -> Any:
    """Validate a custom line search result before a solver consumes it."""
    required = ("point", "cost", "gradient", "stepsize", "alpha", "stats", "state")
    missing = [name for name in required if not hasattr(result, name)]
    if missing:
        raise TypeError(
            "line_search.search(...) returned an incomplete result; missing "
            + ", ".join(missing)
            + "."
        )
    M = require(problem, "M")
    validate_point(M, result.point, name="line-search result point")
    value = _validated_cost_output(result.cost)
    for name in ("stepsize", "alpha"):
        raw = getattr(result, name)
        if isinstance(raw, bool):
            raise TypeError(f"line-search {name} must be a real scalar.")
        scalar = float(raw)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"line-search {name} must be finite and nonnegative.")
    stats = result.stats
    if not isinstance(stats, LineSearchStats):
        raise TypeError("line-search stats must be a LineSearchStats instance.")
    if not isinstance(stats.accepted, bool):
        raise TypeError("line-search accepted status must be a boolean.")
    if stats.accepted and not math.isfinite(as_float(value)):
        raise FloatingPointError("An accepted line-search cost must be finite.")
    if result.gradient is not None:
        validate_tangent(
            M,
            result.point,
            result.gradient,
            name="line-search result gradient",
        )
    return result


def precondition_gradient(problem: Any, x: Array, grad: Array) -> Array:
    """Apply problem preconditioner, or return ``grad`` if none is supplied."""
    precon = get(problem, "precon", None)
    if precon is None:
        return grad
    return validate_tangent(
        require(problem, "M"),
        x,
        precon(x, grad),
        name="preconditioned gradient",
    )


precondition = precondition_gradient


def make_info(
    *,
    iter: int,
    cost: Array,
    gradnorm: Array,
    stepsize: float,
    start_time: float,
    linesearch: Optional[LineSearchStats],
    problem: Any,
    x: Array,
    options: Any | None = None,
    solver: Any | None = None,
    beta: Optional[float] = None,
    reason: str = "",
    **extra_fields: Any,
) -> InfoEntry:
    """Build an :class:`InfoEntry`, applying ``options.statsfun`` if present."""
    stats_source = options if options is not None else solver
    if stats_source is not None:
        validate_stopping_controls(stats_source)
    extra = {name: value for name, value in extra_fields.items() if value is not None}
    entry = InfoEntry(
        iter=iter,
        cost=as_float(cost),
        gradnorm=as_float(gradnorm),
        stepsize=float(stepsize),
        time=time.perf_counter() - start_time,
        linesearch=linesearch,
        beta=beta,
        reason=reason,
        extra=extra,
    )
    statsfun = getattr(stats_source, "statsfun", None)
    if statsfun is not None:
        user_extra = statsfun(problem, x, entry)
        if user_extra is not None:
            if not isinstance(user_extra, Mapping):
                raise TypeError("statsfun must return a mapping or None.")
            merged = {**entry.extra, **dict(user_extra)}
            entry = replace(entry, extra=merged)
    return entry


def validate_stopping_controls(options: Any) -> tuple[bool, float, int, float, float]:
    """Validate controls shared by all optimization solvers."""
    requires_gradient = validate_boolean(
        getattr(options, "requires_gradient", True), name="requires_gradient"
    )
    tolgradnorm = float(getattr(options, "tolgradnorm", 1e-6))
    maxiter = validate_integer(getattr(options, "maxiter", 1000), name="maxiter")
    maxtime = float(getattr(options, "maxtime", math.inf))
    minstepsize = validate_nonnegative(getattr(options, "minstepsize", 1e-10), name="minstepsize")
    validate_integer(getattr(options, "verbosity", 2), name="verbosity", minimum=0)
    if maxiter < 0:
        raise ValueError("maxiter must be nonnegative.")
    if isinstance(getattr(options, "maxtime", math.inf), bool):
        raise TypeError("maxtime must be a real scalar, not a boolean.")
    if math.isnan(maxtime) or maxtime < 0.0:
        raise ValueError("maxtime must be nonnegative and not NaN.")
    if requires_gradient and (
        isinstance(getattr(options, "tolgradnorm", 1e-6), bool)
        or not math.isfinite(tolgradnorm)
        or tolgradnorm < 0.0
    ):
        raise ValueError("tolgradnorm must be finite and nonnegative.")
    for name in ("statsfun", "stopfun"):
        callback = getattr(options, name, None)
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None.")
    return requires_gradient, tolgradnorm, maxiter, maxtime, minstepsize


def stopping_reason(problem: Any, x: Array, info: List[InfoEntry], options: Any) -> str:
    """Evaluate standard Manopt-style stopping criteria."""
    current = info[-1]
    requires_gradient, tolgradnorm, maxiter, maxtime, minstepsize = validate_stopping_controls(
        options
    )

    if not math.isfinite(current.cost):
        return "Non-finite objective value encountered."
    if requires_gradient and not math.isfinite(current.gradnorm):
        return "Non-finite gradient norm encountered."
    if requires_gradient and current.gradnorm <= tolgradnorm:
        return f"Gradient norm tolerance reached: {current.gradnorm:g} <= {tolgradnorm:g}."
    if current.iter >= maxiter:
        return f"Maximum iteration count reached: options.maxiter = {maxiter}."
    if current.time >= maxtime:
        return f"Maximum time reached: options.maxtime = {maxtime:g}."
    if current.iter > 0 and current.stepsize < minstepsize:
        return f"Last stepsize smaller than options.minstepsize = {minstepsize:g}."

    stopfun = getattr(options, "stopfun", None)
    if stopfun is not None:
        result = stopfun(problem, x, current)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("stopfun must return a pair (stop, reason).")
        stop, reason = result
        if not isinstance(stop, bool):
            raise TypeError("stopfun's stop value must be a boolean.")
        if not isinstance(reason, str):
            raise TypeError("stopfun's reason must be a string.")
        if stop:
            return reason or "User stopfun triggered."
    return ""


def print_iteration_header(
    verbosity: int, include_beta: bool = False, include_rho: bool = False
) -> None:
    """Print a compact iteration table header."""
    if verbosity >= 2:
        if include_rho:
            print(" iter\t        cost val\t    grad. norm\t         rho")
        elif include_beta:
            print(" iter\t        cost val\t    grad. norm\t        beta")
        else:
            print(" iter\t        cost val\t    grad. norm")


def print_iteration(
    entry: InfoEntry, verbosity: int, include_beta: bool = False, include_rho: bool = False
) -> None:
    """Print one iteration row."""
    if verbosity >= 2:
        if include_rho:
            rho = entry.extra.get("rho", float("nan"))
            print(f"{entry.iter:5d}\t{entry.cost:+.16e}\t{entry.gradnorm:.8e}\t{rho:+.3e}")
        elif include_beta:
            beta = float("nan") if entry.beta is None else entry.beta
            print(f"{entry.iter:5d}\t{entry.cost:+.16e}\t{entry.gradnorm:.8e}\t{beta:+.3e}")
        else:
            print(f"{entry.iter:5d}\t{entry.cost:+.16e}\t{entry.gradnorm:.8e}")


__all__ = [
    "Array",
    "CostFn",
    "GradFn",
    "PreconFn",
    "HessVecFn",
    "StatsFn",
    "StopFn",
    "Minimize",
    "LineSearchStats",
    "InfoEntry",
    "as_options",
    "as_float",
    "tree_scale",
    "tree_add",
    "tree_sub",
    "tree_neg",
    "tree_zeros_like",
    "tree_lincomb",
    "get",
    "require",
    "require_field",
    "initial_point",
    "validate_point",
    "validate_tangent",
    "require_geometry_methods",
    "retract",
    "inner",
    "lincomb",
    "transport",
    "pair_mean",
    "cost_and_grad",
    "gradient_value",
    "cost_value",
    "validate_line_search_result",
    "precondition_gradient",
    "precondition",
    "make_info",
    "validate_stopping_controls",
    "stopping_reason",
    "print_iteration_header",
    "print_iteration",
]
