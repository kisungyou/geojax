"""Riemannian steepest descent with pluggable line searches."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Mapping, Optional
import math
import time

from .linesearch import AdaptiveArmijo, LineSearchProtocol, LineSearchState
from .minimize import (
    Array,
    InfoEntry,
    LineSearchStats,
    StatsFn,
    StopFn,
    as_options,
    cost_and_grad,
    gradient_value,
    initial_point,
    make_info,
    print_iteration,
    print_iteration_header,
    require,
    stopping_reason,
    tree_neg,
)


@dataclass(frozen=True)
class SteepestDescentOptions:
    """Options shared by the functional and class-style interfaces."""

    requires_gradient: bool = True
    tolgradnorm: float = 1e-6
    maxiter: int = 1000
    maxtime: float = math.inf
    minstepsize: float = 1e-10
    verbosity: int = 2
    line_search: LineSearchProtocol = field(default_factory=AdaptiveArmijo)
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None
    key: Optional[Array] = None


@dataclass(frozen=True)
class SteepestDescent(SteepestDescentOptions):
    """Minimize a smooth objective along its negative Riemannian gradient."""

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        x, cost, info, _ = steepestdescent(problem, getattr(problem, "x0", None), self)
        return x, cost, info


def steepestdescent(
    problem: Mapping[str, Any] | Any,
    x: Optional[Array] = None,
    options: Optional[SteepestDescentOptions | Mapping[str, Any]] = None,
) -> tuple[Array, float, List[InfoEntry], SteepestDescentOptions]:
    """Minimize a smooth function by Riemannian steepest descent.

    The functional interface returns ``(x, cost, info, options)``. The public
    class interface used by :class:`~geojax.optimization.Minimize` returns the
    common ``(x, cost, info)`` triple.
    """

    options = as_options(SteepestDescentOptions, options)
    M = require(problem, "M")
    require(problem, "cost")
    x = initial_point(problem, x, options.key)

    start_time = time.perf_counter()
    search_state: LineSearchState | None = None
    info: List[InfoEntry] = []
    cost, grad = cost_and_grad(problem, x)
    gradnorm = M.norm(x, grad)
    info.append(
        make_info(
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

    print_iteration_header(options.verbosity)
    while True:
        current = info[-1]
        print_iteration(current, options.verbosity)
        reason = stopping_reason(problem, x, info, options)
        if reason:
            info[-1] = replace(current, reason=reason)
            if options.verbosity >= 1:
                print(reason)
            break

        direction = tree_neg(grad)
        directional_derivative = -(gradnorm * gradnorm)
        result = options.line_search.search(
            problem,
            x,
            direction,
            cost,
            directional_derivative,
            state=search_state,
        )
        search_state = result.state
        x = result.point
        cost = result.cost
        grad = result.gradient if result.gradient is not None else gradient_value(problem, x)
        gradnorm = M.norm(x, grad)
        info.append(
            make_info(
                iter=current.iter + 1,
                cost=cost,
                gradnorm=gradnorm,
                stepsize=result.stepsize,
                start_time=start_time,
                linesearch=result.stats,
                problem=problem,
                x=x,
                options=options,
            )
        )

    if options.verbosity >= 1:
        print(f"Total time is {info[-1].time:.6f} [s]")
    return x, info[-1].cost, info, options


__all__ = [
    "SteepestDescent",
    "SteepestDescentOptions",
    "InfoEntry",
    "LineSearchStats",
    "steepestdescent",
]
