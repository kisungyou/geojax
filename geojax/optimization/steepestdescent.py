"""Manopt-style steepest descent solver for Riemannian optimization.

The public entry point is :func:`steepestdescent`, following the naming and
high-level calling convention of Manopt's MATLAB solver::

    x, cost, info, options = steepestdescent(problem, x0, options)

A problem can be either a :class:`Problem` instance or a dictionary/object with
at least two fields:

    M       manifold object
    cost    callable cost(x) -> scalar

Optionally, the problem may provide either

    grad    callable returning the Riemannian gradient, or
    egrad   callable returning the ambient Euclidean gradient.

If neither gradient is provided, JAX autodiff is used on ``cost`` and the
manifold method ``egrad_to_rgrad`` is used to convert the Euclidean gradient to
a Riemannian gradient.

The solver uses the negative Riemannian gradient as descent direction and a
Manopt-inspired Armijo backtracking line search.  It expects the manifold to
provide at least:

    random_point(key), exp(x, u), norm(x, u), egrad_to_rgrad(x, egrad)

If the manifold provides ``retr(x, u, t)``, that is used by the line search;
otherwise ``exp(x, t * u)`` is used.  For the sphere geometry in this package,
``retr`` is the exact exponential map, so the algorithm follows intrinsic
geodesic updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple
import math
import time

import jax
import jax.numpy as jnp

from .minimize import tree_lincomb, tree_neg

Array = Any
CostFn = Callable[[Array], Array]
GradFn = Callable[[Array], Array]
StatsFn = Callable[[Any, Array, "InfoEntry"], Dict[str, Any]]
StopFn = Callable[[Any, Array, "InfoEntry"], Tuple[bool, str]]


@dataclass(frozen=True)
class Problem:
    """Riemannian optimization problem.

    Parameters
    ----------
    M:
        Manifold object.
    cost:
        Scalar objective function to minimize.
    grad:
        Optional Riemannian gradient function.  If provided, it takes priority
        over ``egrad`` and autodiff.
    egrad:
        Optional ambient Euclidean gradient function.  It is converted with
        ``M.egrad_to_rgrad``.
    """

    M: Any
    cost: CostFn
    grad: Optional[GradFn] = None
    egrad: Optional[GradFn] = None


@dataclass(frozen=True)
class LineSearchStats:
    """Statistics returned by the backtracking line search."""

    costevals: int
    stepsize: float
    alpha: float
    accepted: bool


@dataclass(frozen=True)
class InfoEntry:
    """Per-iteration statistics, mirroring Manopt's ``info`` struct-array."""

    iter: int
    cost: float
    gradnorm: float
    stepsize: float
    time: float
    linesearch: Optional[LineSearchStats] = None
    reason: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SteepestDescentOptions:
    """Options for :func:`steepestdescent`.

    The option names intentionally follow Manopt's documentation where
    practical: ``tolgradnorm``, ``maxiter``, ``maxtime`` and
    ``minstepsize``.  The line-search option names similarly mirror Manopt's
    standard backtracking line search.
    """

    tolgradnorm: float = 1e-6
    maxiter: int = 1000
    maxtime: float = math.inf
    minstepsize: float = 1e-10
    verbosity: int = 2

    # Backtracking line search parameters, matching Manopt-style names.
    ls_contraction_factor: float = 0.5
    ls_optimism: float = 2.0
    ls_suff_decr: float = 2.0**-13
    ls_max_steps: int = 25
    ls_initial_stepsize: float = 1.0

    # Optional extension hooks.
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    # JAX requires explicit randomness.  If x0 is not supplied, this key is
    # used to draw an initial point.  The deterministic default is convenient
    # for examples; production code should pass its own key.
    key: Optional[Array] = None


@dataclass(frozen=True)
class SteepestDescent(SteepestDescentOptions):
    """Class-style steepest descent solver for ``Minimize(...).solve()``."""

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        x, cost, info, _ = steepestdescent(problem, getattr(problem, "x0", None), self)
        return x, cost, info


def _as_options(
    options: Optional[SteepestDescentOptions | Mapping[str, Any]],
) -> SteepestDescentOptions:
    if options is None:
        return SteepestDescentOptions()
    if isinstance(options, SteepestDescentOptions):
        return options
    if isinstance(options, Mapping):
        valid = {f.name for f in fields(SteepestDescentOptions)}
        unknown = set(options) - valid
        if unknown:
            raise ValueError(f"Unknown steepestdescent option(s): {sorted(unknown)}")
        return SteepestDescentOptions(**dict(options))
    raise TypeError("options must be None, a dict-like mapping, or SteepestDescentOptions")


def _get(problem: Any, name: str, default: Any = None) -> Any:
    if isinstance(problem, Mapping):
        return problem.get(name, default)
    return getattr(problem, name, default)


def _require(problem: Any, name: str) -> Any:
    value = _get(problem, name, None)
    if value is None:
        raise ValueError(f"problem must define field {name!r}")
    return value


def _retract(M: Any, x: Array, direction: Array, alpha: float | Array) -> Array:
    """Take a trial step using M.retr if available, otherwise M.exp."""
    if hasattr(M, "retr"):
        return M.retr(x, direction, alpha)
    return M.exp(x, tree_lincomb(alpha, direction))


def _cost_and_grad(problem: Any, x: Array) -> tuple[Array, Array]:
    M = _require(problem, "M")
    cost_fn = _require(problem, "cost")
    grad_fn = _get(problem, "grad", None)
    egrad_fn = _get(problem, "egrad", None)

    if grad_fn is not None:
        cost = cost_fn(x)
        grad = grad_fn(x)
        return cost, grad

    if egrad_fn is not None:
        cost = cost_fn(x)
        egrad = egrad_fn(x)
        return cost, M.egrad_to_rgrad(x, egrad)

    cost, egrad = jax.value_and_grad(cost_fn)(x)
    return cost, M.egrad_to_rgrad(x, egrad)


def _cost(problem: Any, x: Array) -> Array:
    cost_fn = _require(problem, "cost")
    return cost_fn(x)


def _line_search_backtracking(
    problem: Any,
    x: Array,
    direction: Array,
    f0: Array,
    df0: Array,
    options: SteepestDescentOptions,
    lsmem: MutableMapping[str, float],
) -> tuple[float, Array, LineSearchStats]:
    """Manopt-inspired Armijo backtracking line search.

    The returned ``stepsize`` is the Riemannian norm of the displacement
    ``alpha * direction``.  As in Manopt, the initial trial step is invariant
    under positive rescaling of the search direction.
    """
    M = _require(problem, "M")

    norm_d = float(jnp.asarray(M.norm(x, direction)))
    if not math.isfinite(norm_d) or norm_d <= 0.0:
        stats = LineSearchStats(costevals=0, stepsize=0.0, alpha=0.0, accepted=False)
        return 0.0, x, stats

    f0_float = float(jnp.asarray(f0))
    df0_float = float(jnp.asarray(df0))

    alpha = math.nan
    if "f0" in lsmem and df0_float != 0.0:
        alpha = 2.0 * (f0_float - float(lsmem["f0"])) / df0_float
        alpha = options.ls_optimism * alpha

    if not math.isfinite(alpha) or alpha * norm_d <= jnp.finfo(jnp.asarray(f0).dtype).eps:
        alpha = options.ls_initial_stepsize / norm_d

    newx = _retract(M, x, direction, alpha)
    newf = _cost(problem, newx)
    costevals = 1

    while float(jnp.asarray(newf)) > f0_float + options.ls_suff_decr * alpha * df0_float:
        alpha *= options.ls_contraction_factor
        newx = _retract(M, x, direction, alpha)
        newf = _cost(problem, newx)
        costevals += 1
        if costevals >= options.ls_max_steps:
            break

    accepted = float(jnp.asarray(newf)) <= f0_float
    if not accepted:
        alpha = 0.0
        newx = x
        newf = f0

    stepsize = float(alpha * norm_d)
    lsmem["f0"] = f0_float
    lsmem["df0"] = df0_float
    lsmem["stepsize"] = stepsize

    stats = LineSearchStats(
        costevals=costevals,
        stepsize=stepsize,
        alpha=float(alpha),
        accepted=accepted,
    )
    return stepsize, newx, stats


def _make_info(
    *,
    iter: int,
    cost: Array,
    gradnorm: Array,
    stepsize: float,
    start_time: float,
    linesearch: Optional[LineSearchStats],
    reason: str = "",
    problem: Any,
    x: Array,
    options: SteepestDescentOptions,
) -> InfoEntry:
    entry = InfoEntry(
        iter=iter,
        cost=float(jnp.asarray(cost)),
        gradnorm=float(jnp.asarray(gradnorm)),
        stepsize=float(stepsize),
        time=time.perf_counter() - start_time,
        linesearch=linesearch,
        reason=reason,
    )
    if options.statsfun is not None:
        extra = options.statsfun(problem, x, entry)
        if extra is not None:
            entry = replace(entry, extra=dict(extra))
    return entry


def _stopping_reason(
    problem: Any,
    x: Array,
    info: List[InfoEntry],
    options: SteepestDescentOptions,
) -> str:
    current = info[-1]
    if current.gradnorm <= options.tolgradnorm:
        return f"Gradient norm tolerance reached: {current.gradnorm:g} <= {options.tolgradnorm:g}."
    if current.iter >= options.maxiter:
        return f"Maximum iteration count reached: options.maxiter = {options.maxiter}."
    if current.time >= options.maxtime:
        return f"Maximum time reached: options.maxtime = {options.maxtime:g}."
    if current.iter > 0 and current.stepsize < options.minstepsize:
        return f"Last stepsize smaller than options.minstepsize = {options.minstepsize:g}."
    if options.stopfun is not None:
        stop, reason = options.stopfun(problem, x, current)
        if stop:
            return reason or "User stopfun triggered."
    return ""


def steepestdescent(
    problem: Problem | Mapping[str, Any] | Any,
    x: Optional[Array] = None,
    options: Optional[SteepestDescentOptions | Mapping[str, Any]] = None,
) -> tuple[Array, float, List[InfoEntry], SteepestDescentOptions]:
    """Minimize a smooth function on a manifold by steepest descent.

    Parameters
    ----------
    problem:
        A :class:`Problem` instance, dictionary, or object with fields ``M``
        and ``cost``.  Optional fields ``grad`` and ``egrad`` are supported.
    x:
        Optional initial point.  If omitted, ``problem.M.random_point`` is used
        with ``options.key`` or a deterministic default key.
    options:
        Optional :class:`SteepestDescentOptions` or dict with matching fields.

    Returns
    -------
    x:
        Final point, equal to the best point reached for the monotone Armijo
        line search used here.
    cost:
        Final cost value.
    info:
        List of :class:`InfoEntry`, with entry 0 corresponding to the initial
        point and one further entry per completed iteration.
    options:
        The resolved options object.
    """
    options = _as_options(options)
    M = _require(problem, "M")
    _require(problem, "cost")

    if x is None:
        key = options.key if options.key is not None else jax.random.key(0)
        if not hasattr(M, "random_point"):
            raise ValueError(
                "No initial point was supplied and problem.M has no random_point(key) method."
            )
        x = M.random_point(key)
    else:
        x = M.project(x) if hasattr(M, "project") else x

    start_time = time.perf_counter()
    lsmem: Dict[str, float] = {}
    info: List[InfoEntry] = []

    cost, grad = _cost_and_grad(problem, x)
    gradnorm = M.norm(x, grad)
    info.append(
        _make_info(
            iter=0,
            cost=cost,
            gradnorm=gradnorm,
            stepsize=math.nan,
            start_time=start_time,
            linesearch=None,
            problem=problem,
            x=x,
            options=options,
        )
    )

    if options.verbosity >= 2:
        print(" iter\t        cost val\t    grad. norm")

    while True:
        current = info[-1]
        if options.verbosity >= 2:
            print(f"{current.iter:5d}\t{current.cost:+.16e}\t{current.gradnorm:.8e}")

        reason = _stopping_reason(problem, x, info, options)
        if reason:
            info[-1] = replace(info[-1], reason=reason)
            if options.verbosity >= 1:
                print(reason)
            break

        # Steepest descent direction: negative Riemannian gradient.
        direction = tree_neg(grad)
        df0 = -gradnorm * gradnorm

        stepsize, newx, lsstats = _line_search_backtracking(
            problem, x, direction, cost, df0, options, lsmem
        )

        x = newx
        cost, grad = _cost_and_grad(problem, x)
        gradnorm = M.norm(x, grad)

        iter_next = current.iter + 1
        info.append(
            _make_info(
                iter=iter_next,
                cost=cost,
                gradnorm=gradnorm,
                stepsize=stepsize,
                start_time=start_time,
                linesearch=lsstats,
                problem=problem,
                x=x,
                options=options,
            )
        )

    if options.verbosity >= 1:
        print(f"Total time is {info[-1].time:.6f} [s]")

    return x, info[-1].cost, info, options


__all__ = [
    "Problem",
    "SteepestDescent",
    "SteepestDescentOptions",
    "InfoEntry",
    "LineSearchStats",
    "steepestdescent",
]
