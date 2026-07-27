"""Limited-memory Riemannian BFGS solver."""

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
)


@dataclass(frozen=True)
class LBFGS:
    requires_gradient: bool = True
    memory: int = 10
    tolgradnorm: float = 1e-6
    maxiter: int = 1000
    maxtime: float = math.inf
    minstepsize: float = 1e-10
    verbosity: int = 2
    line_search: LineSearchProtocol = field(default_factory=AdaptiveArmijo)
    cautious_update: bool = True
    cautious_threshold: float = 1e-10
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        M = require(problem, "M")
        x = require(problem, "x0")
        start_time = time.perf_counter()
        search_state: LineSearchState | None = None
        memory: list[tuple[Array, Array, float]] = []
        info: List[InfoEntry] = []

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

            d = tree_neg(_two_loop(M, x, g, memory))
            if as_float(inner(M, x, g, d)) >= 0.0:
                d = tree_neg(g)
            df0 = inner(M, x, g, d)
            result = self.line_search.search(
                problem,
                x,
                d,
                f,
                df0,
                state=search_state,
            )
            search_state = result.state
            newx = result.point
            newf = result.cost
            newg = result.gradient if result.gradient is not None else gradient_value(problem, newx)

            step = transport(M, x, newx, tree_lincomb(result.alpha, d))
            oldg = transport(M, x, newx, g)
            y = tree_sub(newg, oldg)
            sy = as_float(inner(M, newx, step, y))
            yy = as_float(inner(M, newx, y, y))
            ss = as_float(inner(M, newx, step, step))

            memory = _transport_memory(
                M,
                x,
                newx,
                memory,
                cautious_update=self.cautious_update,
                cautious_threshold=self.cautious_threshold,
            )
            if sy > 1e-300 and yy > 0.0 and ss > 0.0:
                if (not self.cautious_update) or sy >= self.cautious_threshold * ss:
                    memory.append((step, y, 1.0 / sy))
                    if len(memory) > int(self.memory):
                        memory = memory[-int(self.memory) :]

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


def _two_loop(M: Any, x: Array, grad: Array, memory: list[tuple[Array, Array, float]]) -> Array:
    if not memory:
        return grad
    q = grad
    alphas: list[float] = []
    for s, y, rho in reversed(memory):
        a = rho * as_float(inner(M, x, s, q))
        alphas.append(a)
        q = tree_lincomb(1.0, q, -a, y)
    s_last, y_last, _ = memory[-1]
    sy = as_float(inner(M, x, s_last, y_last))
    yy = as_float(inner(M, x, y_last, y_last))
    gamma = sy / yy if yy > 1e-300 else 1.0
    r = tree_lincomb(gamma, q)
    for (s, y, rho), a in zip(memory, reversed(alphas)):
        b = rho * as_float(inner(M, x, y, r))
        r = tree_lincomb(1.0, r, a - b, s)
    return r


def _transport_memory(
    M: Any,
    x: Array,
    newx: Array,
    memory: list[tuple[Array, Array, float]],
    *,
    cautious_update: bool,
    cautious_threshold: float,
) -> list[tuple[Array, Array, float]]:
    """Transport L-BFGS pairs and rebuild their curvature reciprocals.

    A general manifold vector transport need not preserve the metric. The
    reciprocal ``rho = 1 / <s, y>`` must therefore be evaluated after both
    vectors reach the new tangent space. Pairs that lose positive curvature
    under a non-isometric transport are discarded.
    """
    transported: list[tuple[Array, Array, float]] = []
    for s_i, y_i, _ in memory:
        new_s = transport(M, x, newx, s_i)
        new_y = transport(M, x, newx, y_i)
        sy = as_float(inner(M, newx, new_s, new_y))
        ss = as_float(inner(M, newx, new_s, new_s))
        yy = as_float(inner(M, newx, new_y, new_y))
        cautious_ok = (not cautious_update) or sy >= cautious_threshold * ss
        if math.isfinite(sy) and math.isfinite(ss) and math.isfinite(yy):
            if sy > 1e-300 and ss > 0.0 and yy > 0.0 and cautious_ok:
                transported.append((new_s, new_y, 1.0 / sy))
    return transported


__all__ = ["LBFGS"]
