"""Riemannian Barzilai-Borwein gradient method."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional
import math
import time

from geojax.geometry.base import validate_positive

from .linesearch import BacktrackingArmijo, LineSearchProtocol, LineSearchState
from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
    cost_and_grad,
    gradient_value,
    inner,
    make_info,
    print_iteration,
    print_iteration_header,
    require,
    stopping_reason,
    transport,
    tree_lincomb,
    tree_neg,
    tree_sub,
    validate_line_search_result,
)


@dataclass(frozen=True)
class BarzilaiBorwein:
    """Riemannian gradient descent with BB1, BB2, or alternating step estimates."""

    requires_gradient: bool = True
    bb_type: str = "alternate"  # "BB1", "BB2", or "alternate"
    initial_stepsize: float = 1.0
    min_stepsize: float = 1e-12
    max_stepsize: float = 1e12
    line_search: LineSearchProtocol = field(
        default_factory=lambda: BacktrackingArmijo(normalize_step=False)
    )
    tolgradnorm: float = 1e-6
    maxiter: int = 1000
    maxtime: float = math.inf
    minstepsize: float = 1e-14
    verbosity: int = 2
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        if not isinstance(self.bb_type, str):
            raise TypeError("bb_type must be a string.")
        if self.bb_type.upper() not in {"BB1", "BB2", "ALTERNATE"}:
            raise ValueError("bb_type must be 'BB1', 'BB2', or 'alternate'.")
        min_stepsize = validate_positive(self.min_stepsize, name="min_stepsize")
        initial_stepsize = validate_positive(self.initial_stepsize, name="initial_stepsize")
        max_stepsize = validate_positive(self.max_stepsize, name="max_stepsize")
        if not min_stepsize <= initial_stepsize <= max_stepsize:
            raise ValueError(
                "Step bounds must satisfy 0 < min_stepsize <= initial_stepsize <= max_stepsize."
            )
        M = require(problem, "M")
        x = require(problem, "x0")
        start_time = time.perf_counter()
        info: List[InfoEntry] = []
        alpha = initial_stepsize
        search_state: LineSearchState | None = None

        f, g = cost_and_grad(problem, x)
        gnorm = M.norm(x, g)
        info.append(
            make_info(
                iter=0,
                cost=f,
                gradnorm=gnorm,
                stepsize=math.nan,
                start_time=start_time,
                linesearch=None,
                problem=problem,
                x=x,
                solver=self,
            )
        )
        print_iteration_header(self.verbosity)
        while True:
            print_iteration(info[-1], self.verbosity)
            reason = stopping_reason(problem, x, info, self)
            if reason:
                info[-1] = replace(info[-1], reason=reason)
                if self.verbosity >= 1:
                    print(reason)
                break

            d = tree_neg(g)
            df0 = inner(M, x, g, d)
            result = validate_line_search_result(
                problem,
                self.line_search.search(
                    problem,
                    x,
                    d,
                    f,
                    df0,
                    state=search_state,
                    initial_alpha=alpha,
                ),
            )
            search_state = result.state
            newx = result.point
            newf = result.cost
            newg = result.gradient if result.gradient is not None else gradient_value(problem, newx)
            step = transport(M, x, newx, tree_lincomb(result.alpha, d))
            oldg = transport(M, x, newx, g)
            y = tree_sub(newg, oldg)
            sy = as_float(inner(M, newx, step, y))
            ss = as_float(inner(M, newx, step, step))
            yy = as_float(inner(M, newx, y, y))
            next_alpha = alpha
            mode = self.bb_type.upper()
            if mode == "ALTERNATE":
                mode = "BB1" if (info[-1].iter % 2 == 0) else "BB2"
            if sy > 1e-300 and ss > 0.0 and yy > 0.0:
                if mode == "BB1":
                    next_alpha = ss / sy
                elif mode == "BB2":
                    next_alpha = sy / yy
            alpha = min(max(float(next_alpha), min_stepsize), max_stepsize)

            x, f, g = newx, newf, newg
            gnorm = M.norm(x, g)
            info.append(
                make_info(
                    iter=info[-1].iter + 1,
                    cost=f,
                    gradnorm=gnorm,
                    stepsize=result.stepsize,
                    start_time=start_time,
                    linesearch=result.stats,
                    problem=problem,
                    x=x,
                    solver=self,
                )
            )
        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return x, info[-1].cost, info


__all__ = ["BarzilaiBorwein"]
