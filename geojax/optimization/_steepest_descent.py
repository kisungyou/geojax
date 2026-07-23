"""Internal implementation for class-style Riemannian steepest descent."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional
import math
import time

from .linesearch import AdaptiveArmijo, LineSearchProtocol, LineSearchState
from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
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
class SteepestDescent:
    """Minimize a smooth objective along its negative Riemannian gradient."""

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

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        """Solve ``problem`` and return ``(solution, final_cost, history)``."""
        return _solve_steepest_descent(problem, getattr(problem, "x0", None), self)


def _solve_steepest_descent(
    problem: Any,
    x: Optional[Array] = None,
    options: SteepestDescent | None = None,
) -> tuple[Array, float, List[InfoEntry]]:
    """Internal Riemannian steepest-descent iteration engine."""
    options = SteepestDescent() if options is None else options
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
    return x, info[-1].cost, info


__all__ = ["SteepestDescent"]
