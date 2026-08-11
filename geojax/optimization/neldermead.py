"""Derivative-free Riemannian Nelder-Mead simplex method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import math
import time

import jax.numpy as jnp

from geojax.geometry.base import (
    validate_integer,
    validate_nonnegative,
    validate_positive,
)

from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
    cost_value,
    make_info,
    require,
    require_geometry_methods,
    stopping_reason,
    tree_lincomb,
    validate_point,
    validate_tangent,
)


@dataclass(frozen=True)
class NelderMead:
    """Derivative-free manifold simplex search using intrinsic trial points."""

    requires_gradient: bool = False
    initial_scale: float = 0.1
    reflection: float = 1.0
    expansion: float = 2.0
    contraction: float = 0.5
    shrink: float = 0.5
    tolcostspread: float = 1e-10
    tolsimplexdiameter: float = 1e-8
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
        initial_scale = validate_positive(self.initial_scale, name="initial_scale")
        reflection = validate_positive(self.reflection, name="reflection")
        expansion = validate_positive(self.expansion, name="expansion")
        contraction = validate_positive(self.contraction, name="contraction")
        shrink = validate_positive(self.shrink, name="shrink")
        tolcostspread = validate_nonnegative(self.tolcostspread, name="tolcostspread")
        tolsimplexdiameter = validate_nonnegative(
            self.tolsimplexdiameter,
            name="tolsimplexdiameter",
        )
        if expansion <= 1.0:
            raise ValueError("reflection must be positive and expansion must exceed one.")
        if contraction >= 1.0 or shrink >= 1.0:
            raise ValueError("contraction and shrink must lie in (0, 1).")
        dim = validate_integer(getattr(M, "dim", None), name="M.dim", minimum=1)
        require_geometry_methods(
            M,
            "random_tangent",
            "retr",
            "exp",
            "log",
            "dist",
            context="NelderMead",
        )
        start_time = time.perf_counter()
        simplex = [x0]
        for _ in range(dim):
            u = validate_tangent(
                M,
                x0,
                M.random_tangent(problem.split_key(), x0, scale=initial_scale, normalize=True),
                name="Nelder-Mead simplex direction",
            )
            simplex.append(validate_point(M, M.retr(x0, u), name="Nelder-Mead simplex vertex"))
        values = [as_float(cost_value(problem, x)) for x in simplex]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(
                "NelderMead requires finite objective values at every initial vertex."
            )
        simplex, values = _sort(simplex, values)
        spread = _cost_spread(values)
        diameter = _diameter(M, simplex)
        info: List[InfoEntry] = [
            make_info(
                iter=0,
                cost=values[0],
                gradnorm=spread,
                stepsize=diameter,
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
            if spread <= tolcostspread and diameter <= tolsimplexdiameter:
                reason = (
                    "Simplex tolerances reached: "
                    f"cost spread {spread:.3e} <= {tolcostspread:.3e} and "
                    f"diameter {diameter:.3e} <= {tolsimplexdiameter:.3e}."
                )
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
            v = validate_tangent(
                M,
                centroid,
                M.log(centroid, worst),
                name="Nelder-Mead reflection displacement",
            )
            xr = M.exp(centroid, tree_lincomb(-reflection, v))
            fr = _trial_cost(M, problem, xr)

            if fr < values[0]:
                xe = M.exp(
                    centroid,
                    tree_lincomb(-expansion * reflection, v),
                )
                fe = _trial_cost(M, problem, xe)
                if fe < fr:
                    simplex[-1], values[-1] = xe, fe
                else:
                    simplex[-1], values[-1] = xr, fr
            elif fr < values[-2]:
                simplex[-1], values[-1] = xr, fr
            else:
                if fr < values[-1]:
                    xc = M.exp(
                        centroid,
                        tree_lincomb(-contraction * reflection, v),
                    )
                else:
                    xc = M.exp(centroid, tree_lincomb(contraction, v))
                fc = _trial_cost(M, problem, xc)
                if fc < min(fr, values[-1]):
                    simplex[-1], values[-1] = xc, fc
                else:
                    # Shrink toward the best point.
                    new_simplex = [best]
                    new_values = [values[0]]
                    for x in simplex[1:]:
                        xs = M.exp(best, tree_lincomb(shrink, M.log(best, x)))
                        new_simplex.append(xs)
                        new_values.append(_trial_cost(M, problem, xs))
                    simplex, values = new_simplex, new_values
            simplex, values = _sort(simplex, values)
            spread = _cost_spread(values)
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


def _cost_spread(values: list[float]) -> float:
    if not all(math.isfinite(value) for value in values):
        return math.inf
    return float(jnp.std(jnp.asarray(values)))


def _trial_cost(M: Any, problem: Any, point: Array) -> float:
    validate_point(M, point, name="Nelder-Mead trial point")
    value = as_float(cost_value(problem, point))
    return value if math.isfinite(value) else math.inf


def _centroid(M: Any, points: list[Array]) -> Array:
    c = points[0]
    for k, p in enumerate(points[1:], start=1):
        c = M.exp(
            c,
            tree_lincomb(1.0 / (k + 1.0), M.log(c, p)),
        )
    return c


def _diameter(M: Any, points: list[Array]) -> float:
    if not hasattr(M, "dist"):
        return math.nan
    vals = [
        as_float(M.dist(points[i], points[j]))
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    if not all(math.isfinite(value) for value in vals):
        return math.inf
    return max(vals) if vals else 0.0


__all__ = ["NelderMead"]
