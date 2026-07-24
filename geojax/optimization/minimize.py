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
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import math
import time

import jax
import jax.numpy as jnp

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
        if key is None:
            return None
        return jax.random.key(key) if isinstance(key, int) else key

    def split_key(self) -> Array:
        """Return a fresh JAX PRNG key and advance the problem key."""
        if self._key is None:
            self._key = jax.random.key(0)
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def solve(self) -> tuple[Array, float, List["InfoEntry"]]:
        """Solve the problem using the configured class-style solver."""
        if self.x0 is None:
            self.x0 = initial_point(self, None, self.split_key())
        elif hasattr(self.M, "project"):
            self.x0 = self.M.project(self.x0)

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
        if self._ehess_vec is not None:
            return self._ehess_vec(x, u)
        if self.egrad is not None:
            return jax.jvp(self.egrad, (x,), (u,))[1]
        if self.grad is not None:
            return jax.jvp(self.grad, (x,), (u,))[1]
        return jax.jvp(jax.grad(self.cost), (x,), (u,))[1]

    def rhess_vec(self, x: Array, u: Array) -> Array:
        """Riemannian Hessian-vector product."""
        if self._rhess_vec is not None:
            return self._rhess_vec(x, u)

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
            egrad = self.egrad(x) if self.egrad is not None else jax.grad(self.cost)(x)
            return self.M.ehess_to_rhess(x, egrad, self._ehess_vec(x, u), u)

        if self.egrad is not None:
            require_exact("ehess_to_rhess", "ambient-to-Riemannian Hessian conversion")
            egrad = self.egrad(x)
            ehess_u = self.ehess_vec(x, u)
            return self.M.ehess_to_rhess(x, egrad, ehess_u, u)

        if self.grad is not None:
            require_exact("rgrad_jvp", "Riemannian-gradient JVP conversion")
            grad_jvp = jax.jvp(self.grad, (x,), (u,))[1]
            return self.M.tangent_project(x, grad_jvp)

        require_exact("ehess_to_rhess", "ambient-to-Riemannian Hessian conversion")
        egrad = jax.grad(self.cost)(x)
        ehess_u = self.ehess_vec(x, u)
        return self.M.ehess_to_rhess(x, egrad, ehess_u, u)

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
        return M.random_point(key)
    return M.project(x) if hasattr(M, "project") else x


def retract(M: Any, x: Array, direction: Array, alpha: float | Array) -> Array:
    """Take a trial step using ``M.retr`` if available, otherwise ``M.exp``."""
    if hasattr(M, "retr"):
        return M.retr(x, direction, alpha)
    return M.exp(x, tree_lincomb(alpha, direction))


def inner(M: Any, x: Array, u: Array, v: Array) -> Array:
    """Return the Riemannian inner product from the manifold object."""
    return M.inner(x, u, v)


def lincomb(M: Any, x: Array, *terms: Any) -> Array:
    """Linear combination of tangent vectors, projected if needed."""
    if hasattr(M, "lincomb"):
        return M.lincomb(x, *terms)
    out = tree_lincomb(*terms)
    if hasattr(M, "tangent_project"):
        return M.tangent_project(x, out)
    if hasattr(M, "proj"):
        return M.proj(x, out)
    return out


def transport(M: Any, x: Array, y: Array, u: Array) -> Array:
    """Transport a tangent vector from ``x`` to ``y``."""
    if hasattr(M, "transp"):
        return M.transp(x, y, u)
    if hasattr(M, "transport"):
        return M.transport(x, y, u)
    if hasattr(M, "tangent_project"):
        return M.tangent_project(y, u)
    if hasattr(M, "proj"):
        return M.proj(y, u)
    raise ValueError("Manifold must provide transport, tangent_project or proj.")


def pair_mean(M: Any, x: Array, y: Array) -> Array:
    """Return a midpoint-like mean between two manifold points."""
    if hasattr(M, "pair_mean"):
        return M.pair_mean(x, y)
    if hasattr(M, "exp") and hasattr(M, "log"):
        return M.exp(x, tree_lincomb(0.5, M.log(x, y)))
    return tree_lincomb(0.5, x, 0.5, y)


def cost_and_grad(problem: Any, x: Array) -> tuple[Array, Array]:
    """Return ``cost(x)`` and the Riemannian gradient at ``x``."""
    M = require(problem, "M")
    cost_fn = require(problem, "cost")
    grad_fn = get(problem, "grad", None)
    egrad_fn = get(problem, "egrad", None)

    if grad_fn is not None:
        c = cost_fn(x)
        g = grad_fn(x)
        return c, g

    if egrad_fn is not None:
        c = cost_fn(x)
        eg = egrad_fn(x)
        return c, M.egrad_to_rgrad(x, eg)

    c, eg = jax.value_and_grad(cost_fn)(x)
    return c, M.egrad_to_rgrad(x, eg)


def gradient_value(problem: Any, x: Array) -> Array:
    """Return only the Riemannian gradient at ``x``."""
    M = require(problem, "M")
    cost_fn = require(problem, "cost")
    grad_fn = get(problem, "grad", None)
    egrad_fn = get(problem, "egrad", None)
    if grad_fn is not None:
        return grad_fn(x)
    if egrad_fn is not None:
        return M.egrad_to_rgrad(x, egrad_fn(x))
    return M.egrad_to_rgrad(x, jax.grad(cost_fn)(x))


def cost_value(problem: Any, x: Array) -> Array:
    """Return ``cost(x)``."""
    return require(problem, "cost")(x)


def precondition_gradient(problem: Any, x: Array, grad: Array) -> Array:
    """Apply problem preconditioner, or return ``grad`` if none is supplied."""
    precon = get(problem, "precon", None)
    if precon is None:
        return grad
    return precon(x, grad)


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
    stats_source = options if options is not None else solver
    statsfun = getattr(stats_source, "statsfun", None)
    if statsfun is not None:
        user_extra = statsfun(problem, x, entry)
        if user_extra is not None:
            merged = {**entry.extra, **dict(user_extra)}
            entry = replace(entry, extra=merged)
    return entry


def stopping_reason(problem: Any, x: Array, info: List[InfoEntry], options: Any) -> str:
    """Evaluate standard Manopt-style stopping criteria."""
    current = info[-1]
    tolgradnorm = float(getattr(options, "tolgradnorm", 1e-6))
    maxiter = int(getattr(options, "maxiter", 1000))
    maxtime = float(getattr(options, "maxtime", math.inf))
    minstepsize = float(getattr(options, "minstepsize", 1e-10))

    if current.gradnorm <= tolgradnorm:
        return f"Gradient norm tolerance reached: {current.gradnorm:g} <= {tolgradnorm:g}."
    if current.iter >= maxiter:
        return f"Maximum iteration count reached: options.maxiter = {maxiter}."
    if current.time >= maxtime:
        return f"Maximum time reached: options.maxtime = {maxtime:g}."
    if current.iter > 0 and current.stepsize < minstepsize:
        return f"Last stepsize smaller than options.minstepsize = {minstepsize:g}."

    stopfun = getattr(options, "stopfun", None)
    if stopfun is not None:
        stop, reason = stopfun(problem, x, current)
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
    "retract",
    "inner",
    "lincomb",
    "transport",
    "pair_mean",
    "cost_and_grad",
    "gradient_value",
    "cost_value",
    "precondition_gradient",
    "precondition",
    "make_info",
    "stopping_reason",
    "print_iteration_header",
    "print_iteration",
]
