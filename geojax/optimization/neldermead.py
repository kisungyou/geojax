"""Derivative-free Riemannian Nelder-Mead simplex method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import math
import time

import jax.numpy as jnp

from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
    cost_value,
    make_info,
    pair_mean,
    require,
    stopping_reason,
    tree_lincomb,
)


@dataclass(frozen=True)
class NelderMead:
    requires_gradient: bool = False
    initial_scale: float = 0.1
    reflection: float = 1.0
    expansion: float = 2.0
    contraction: float = 0.5
    shrink: float = 0.5
    tolcostspread: float = 1e-10
    maxiter: int = 1000
    maxtime: float = math.inf
    minstepsize: float = 0.0
    tolgradnorm: float = -math.inf
    verbosity: int = 2
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        M = require(problem, "M")
        x0 = require(problem, "x0")
        dim = int(getattr(M, "dim", 1))
        start_time = time.perf_counter()
        simplex = [x0]
        for _ in range(dim):
            if hasattr(M, "random_tangent"):
                u = M.random_tangent(
                    problem.split_key(), x0, scale=self.initial_scale, normalize=True
                )
                simplex.append(
                    M.retr(x0, u, self.initial_scale)
                    if hasattr(M, "retr")
                    else M.exp(x0, tree_lincomb(self.initial_scale, u))
                )
            else:
                simplex.append(M.random_point(problem.split_key()))
        values = [as_float(cost_value(problem, x)) for x in simplex]
        simplex, values = _sort(simplex, values)
        spread = float(jnp.std(jnp.asarray(values)))
        info: List[InfoEntry] = [
            make_info(
                iter=0,
                cost=values[0],
                gradnorm=spread,
                stepsize=math.nan,
                start_time=start_time,
                linesearch=None,
                problem=problem,
                x=simplex[0],
                solver=self,
            )
        ]
        if self.verbosity >= 2:
            print(" iter\t        best cost\t  cost spread")
        while True:
            if self.verbosity >= 2:
                print(f"{info[-1].iter:5d}\t{info[-1].cost:+.16e}\t{info[-1].gradnorm:.8e}")
            if spread <= self.tolcostspread:
                reason = f"Simplex cost spread tolerance reached: {spread:.3e}."
                info[-1] = InfoEntry(**{**info[-1].__dict__, "reason": reason})
                if self.verbosity >= 1:
                    print(reason)
                break
            reason = stopping_reason(problem, simplex[0], info, self)
            if reason:
                info[-1] = InfoEntry(**{**info[-1].__dict__, "reason": reason})
                if self.verbosity >= 1:
                    print(reason)
                break

            best = simplex[0]
            worst = simplex[-1]
            centroid = _centroid(M, simplex[:-1])
            v = M.log(centroid, worst)
            xr = M.exp(centroid, tree_lincomb(-self.reflection, v))
            fr = as_float(cost_value(problem, xr))

            if fr < values[0]:
                xe = M.exp(centroid, tree_lincomb(-self.expansion, v))
                fe = as_float(cost_value(problem, xe))
                if fe < fr:
                    simplex[-1], values[-1] = xe, fe
                else:
                    simplex[-1], values[-1] = xr, fr
            elif fr < values[-2]:
                simplex[-1], values[-1] = xr, fr
            else:
                if fr < values[-1]:
                    xc = M.exp(centroid, tree_lincomb(-self.contraction, v))
                else:
                    xc = M.exp(centroid, tree_lincomb(self.contraction, v))
                fc = as_float(cost_value(problem, xc))
                if fc < min(fr, values[-1]):
                    simplex[-1], values[-1] = xc, fc
                else:
                    # Shrink toward the best point.
                    new_simplex = [best]
                    new_values = [values[0]]
                    for x in simplex[1:]:
                        xs = M.exp(best, tree_lincomb(self.shrink, M.log(best, x)))
                        new_simplex.append(xs)
                        new_values.append(as_float(cost_value(problem, xs)))
                    simplex, values = new_simplex, new_values
            simplex, values = _sort(simplex, values)
            spread = float(jnp.std(jnp.asarray(values)))
            diameter = _diameter(M, simplex)
            info.append(
                make_info(
                    iter=info[-1].iter + 1,
                    cost=values[0],
                    gradnorm=spread,
                    stepsize=diameter,
                    start_time=start_time,
                    linesearch=None,
                    problem=problem,
                    x=simplex[0],
                    solver=self,
                )
            )
        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return simplex[0], info[-1].cost, info


def _sort(simplex: list[Array], values: list[float]) -> tuple[list[Array], list[float]]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    return [simplex[i] for i in order], [values[i] for i in order]


def _centroid(M: Any, points: list[Array]) -> Array:
    c = points[0]
    for k, p in enumerate(points[1:], start=1):
        c = (
            M.exp(c, tree_lincomb(1.0 / (k + 1.0), M.log(c, p)))
            if hasattr(M, "exp") and hasattr(M, "log")
            else pair_mean(M, c, p)
        )
    return c


def _diameter(M: Any, points: list[Array]) -> float:
    if not hasattr(M, "dist"):
        return math.nan
    best = points[0]
    vals = [as_float(M.dist(best, p)) for p in points[1:]]
    return max(vals) if vals else 0.0


__all__ = ["NelderMead"]
