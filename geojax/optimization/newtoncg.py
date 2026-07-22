"""Truncated Riemannian Newton method with tangent conjugate gradients."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional
import math
import time

from ._tangent_cg import tangent_conjugate_gradient
from .linesearch import AdaptiveArmijo, LineSearchProtocol, LineSearchState
from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
    cost_and_grad,
    get,
    gradient_value,
    inner,
    make_info,
    precondition,
    print_iteration,
    print_iteration_header,
    require,
    stopping_reason,
    tree_neg,
)


@dataclass(frozen=True)
class NewtonCG:
    """Riemannian Newton method with an inexact Hessian solve.

    Hessian-vector products come from ``problem.rhess_vec``. The inner
    conjugate-gradient solve is truncated on non-positive curvature; a
    non-descent Newton direction falls back to preconditioned steepest descent.
    """

    requires_gradient: bool = True
    tolgradnorm: float = 1e-6
    maxiter: int = 200
    maxtime: float = math.inf
    minstepsize: float = 1e-10
    verbosity: int = 2
    maxinner: int = 100
    cg_relative_tolerance: float | None = None
    cg_absolute_tolerance: float = 1e-10
    curvature_tolerance: float = 0.0
    line_search: LineSearchProtocol = field(
        default_factory=lambda: AdaptiveArmijo(normalize_step=False)
    )
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        M = require(problem, "M")
        x = require(problem, "x0")
        hessian_vector = get(problem, "rhess_vec", None)
        if hessian_vector is None:
            raise ValueError("NewtonCG requires problem.rhess_vec(x, u).")

        start_time = time.perf_counter()
        search_state: LineSearchState | None = None
        info: List[InfoEntry] = []
        f, g = cost_and_grad(problem, x)
        gradnorm = M.norm(x, g)
        info.append(
            make_info(
                iter=0,
                cost=f,
                gradnorm=gradnorm,
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
            current = info[-1]
            print_iteration(current, self.verbosity)
            reason = stopping_reason(problem, x, info, self)
            if reason:
                info[-1] = replace(current, reason=reason)
                if self.verbosity >= 1:
                    print(reason)
                break

            relative_tolerance = self.cg_relative_tolerance
            if relative_tolerance is None:
                relative_tolerance = min(0.5, math.sqrt(max(as_float(gradnorm), 0.0)))
            cg = tangent_conjugate_gradient(
                M,
                x,
                lambda direction: hessian_vector(x, direction),
                tree_neg(g),
                preconditioner=lambda residual: precondition(problem, x, residual),
                relative_tolerance=float(relative_tolerance),
                absolute_tolerance=self.cg_absolute_tolerance,
                max_iterations=self.maxinner,
                curvature_tolerance=self.curvature_tolerance,
            )
            direction = cg.solution
            directional_derivative = inner(M, x, g, direction)
            if not math.isfinite(as_float(directional_derivative)) or as_float(
                directional_derivative
            ) >= 0.0:
                direction = tree_neg(precondition(problem, x, g))
                directional_derivative = inner(M, x, g, direction)
            if as_float(directional_derivative) >= 0.0:
                direction = tree_neg(g)
                directional_derivative = -(gradnorm * gradnorm)

            line_result = self.line_search.search(
                problem,
                x,
                direction,
                f,
                directional_derivative,
                state=search_state,
            )
            search_state = line_result.state
            x = line_result.point
            f = line_result.cost
            g = (
                line_result.gradient
                if line_result.gradient is not None
                else gradient_value(problem, x)
            )
            gradnorm = M.norm(x, g)
            info.append(
                make_info(
                    iter=current.iter + 1,
                    cost=f,
                    gradnorm=gradnorm,
                    stepsize=line_result.stepsize,
                    start_time=start_time,
                    linesearch=line_result.stats,
                    problem=problem,
                    x=x,
                    solver=self,
                    cg_iterations=cg.iterations,
                    cg_residual_norm=cg.residual_norm,
                    negative_curvature=cg.negative_curvature,
                    cg_reason=cg.reason,
                )
            )

        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return x, info[-1].cost, info


__all__ = ["NewtonCG"]
