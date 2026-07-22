"""Riemannian Barzilai-Borwein gradient method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import math
import time

from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
    cost_and_grad,
    cost_value,
    inner,
    make_info,
    print_iteration,
    print_iteration_header,
    require,
    retract,
    stopping_reason,
    transport,
    tree_lincomb,
    tree_neg,
    tree_sub,
)


@dataclass(frozen=True)
class BarzilaiBorwein:
    requires_gradient: bool = True
    bb_type: str = "alternate"  # "BB1", "BB2", or "alternate"
    initial_stepsize: float = 1.0
    min_stepsize: float = 1e-12
    max_stepsize: float = 1e12
    backtrack: bool = True
    contraction_factor: float = 0.5
    suff_decr: float = 1e-4
    max_backtracks: int = 20
    tolgradnorm: float = 1e-6
    maxiter: int = 1000
    maxtime: float = math.inf
    minstepsize: float = 1e-14
    verbosity: int = 2
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        M = require(problem, "M")
        x = require(problem, "x0")
        start_time = time.perf_counter()
        info: List[InfoEntry] = []
        alpha = float(self.initial_stepsize)

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
                info[-1] = InfoEntry(**{**info[-1].__dict__, "reason": reason})
                if self.verbosity >= 1:
                    print(reason)
                break

            d = tree_neg(g)
            df0 = as_float(inner(M, x, g, d))
            trial_alpha = alpha
            newx = retract(M, x, d, trial_alpha)
            newf = cost_value(problem, newx)
            bt = 0
            if self.backtrack:
                f0 = as_float(f)
                while (
                    as_float(newf) > f0 + self.suff_decr * trial_alpha * df0
                    and bt < self.max_backtracks
                ):
                    trial_alpha *= self.contraction_factor
                    newx = retract(M, x, d, trial_alpha)
                    newf = cost_value(problem, newx)
                    bt += 1
            if as_float(newf) > as_float(f) and self.backtrack:
                trial_alpha = 0.0
                newx = x
                newf = f

            newf, newg = cost_and_grad(problem, newx)
            step = transport(M, x, newx, tree_lincomb(trial_alpha, d))
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
                else:
                    raise ValueError("bb_type must be 'BB1', 'BB2', or 'alternate'.")
            alpha = min(max(float(next_alpha), float(self.min_stepsize)), float(self.max_stepsize))

            x, f, g = newx, newf, newg
            gnorm = M.norm(x, g)
            stepsize = as_float(M.norm(x, step)) if trial_alpha != 0.0 else 0.0
            info.append(
                make_info(
                    iter=info[-1].iter + 1,
                    cost=f,
                    gradnorm=gnorm,
                    stepsize=stepsize,
                    start_time=start_time,
                    linesearch=None,
                    problem=problem,
                    x=x,
                    solver=self,
                )
            )
        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return x, info[-1].cost, info


__all__ = ["BarzilaiBorwein"]
