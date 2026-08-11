"""Riemannian Levenberg-Marquardt solver for nonlinear least squares."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, List, Optional
import math
import time

from geojax.geometry.base import validate_integer, validate_nonnegative, validate_positive

from ._tangent_cg import tangent_conjugate_gradient
from .gaussnewton import _least_squares_method, _residual_norm_from_cost
from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
    cost_and_grad,
    cost_value,
    gradient_value,
    inner,
    make_info,
    precondition,
    print_iteration,
    print_iteration_header,
    require,
    retract,
    stopping_reason,
    tree_neg,
)
from .problems import tree_vdot


@dataclass(frozen=True)
class LevenbergMarquardt:
    """Damped Gauss-Newton method with gain-ratio damping updates."""

    requires_gradient: bool = True
    tolgradnorm: float = 1e-6
    maxiter: int = 200
    maxtime: float = math.inf
    minstepsize: float = 0.0
    verbosity: int = 2
    initial_damping: float = 1e-3
    min_damping: float = 1e-12
    max_damping: float = 1e12
    damping_increase: float = 2.0
    damping_decrease: float = 3.0
    acceptance_threshold: float = 1e-4
    maxinner: int = 100
    cg_relative_tolerance: float = 1e-3
    cg_absolute_tolerance: float = 1e-10
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        M = require(problem, "M")
        x = require(problem, "x0")
        normal_operator = _least_squares_method(problem, "normal_operator")
        jacobian_vec = _least_squares_method(problem, "jacobian_vec")
        initial_damping = validate_positive(self.initial_damping, name="initial_damping")
        min_damping = validate_positive(self.min_damping, name="min_damping")
        max_damping = validate_positive(self.max_damping, name="max_damping")
        if not min_damping <= initial_damping <= max_damping:
            raise ValueError(
                "Damping bounds must satisfy 0 < min_damping <= initial_damping <= max_damping."
            )
        damping_increase = validate_positive(
            self.damping_increase,
            name="damping_increase",
        )
        damping_decrease = validate_positive(
            self.damping_decrease,
            name="damping_decrease",
        )
        if damping_increase <= 1.0 or damping_decrease <= 1.0:
            raise ValueError("damping_increase and damping_decrease must exceed one.")
        acceptance_threshold = validate_nonnegative(
            self.acceptance_threshold, name="acceptance_threshold"
        )
        if acceptance_threshold >= 1.0:
            raise ValueError("acceptance_threshold must lie in [0, 1).")
        maxinner = validate_integer(self.maxinner, name="maxinner")
        if maxinner <= 0:
            raise ValueError("maxinner must be positive.")
        cg_relative_tolerance = validate_nonnegative(
            self.cg_relative_tolerance,
            name="cg_relative_tolerance",
        )
        cg_absolute_tolerance = validate_nonnegative(
            self.cg_absolute_tolerance,
            name="cg_absolute_tolerance",
        )

        damping = initial_damping
        start_time = time.perf_counter()
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
                damping=damping,
                residual_norm=_residual_norm_from_cost(f),
            )
        )
        print_iteration_header(self.verbosity, include_rho=True)

        while True:
            current = info[-1]
            print_iteration(current, self.verbosity, include_rho=True)
            reason = stopping_reason(problem, x, info, self)
            if reason:
                info[-1] = replace(current, reason=reason)
                if self.verbosity >= 1:
                    print(reason)
                break

            damping_used = damping
            cg = tangent_conjugate_gradient(
                M,
                x,
                lambda direction: normal_operator(x, direction, damping_used),
                tree_neg(g),
                preconditioner=lambda value: precondition(problem, x, value),
                relative_tolerance=cg_relative_tolerance,
                absolute_tolerance=cg_absolute_tolerance,
                max_iterations=maxinner,
            )
            direction = cg.solution
            directional_derivative = as_float(inner(M, x, g, direction))
            direction_fallback = False
            if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
                direction = tree_neg(g)
                directional_derivative = -as_float(inner(M, x, g, g))
                direction_fallback = True
            stepnorm = as_float(M.norm(x, direction))
            jacobian_direction = jacobian_vec(x, direction)
            predicted = -directional_derivative - 0.5 * as_float(
                tree_vdot(jacobian_direction, jacobian_direction)
            )
            damping_penalty = 0.5 * damping_used * stepnorm * stepnorm
            trial = retract(M, x, direction, 1.0)
            trial_cost = cost_value(problem, trial)
            actual = as_float(f) - as_float(trial_cost)
            rho = actual / predicted if predicted > 0.0 else -math.inf
            accepted = bool(math.isfinite(rho) and rho > acceptance_threshold)

            if rho > 0.75:
                damping = max(damping / damping_decrease, min_damping)
            elif rho < 0.25 or not math.isfinite(rho):
                damping = min(damping * damping_increase, max_damping)

            if accepted:
                x = trial
                f = trial_cost
                g = gradient_value(problem, x)
                gradnorm = M.norm(x, g)

            info.append(
                make_info(
                    iter=current.iter + 1,
                    cost=f,
                    gradnorm=gradnorm,
                    stepsize=stepnorm,
                    start_time=start_time,
                    linesearch=None,
                    problem=problem,
                    x=x,
                    solver=self,
                    rho=rho,
                    accepted=accepted,
                    damping=damping_used,
                    next_damping=damping,
                    residual_norm=_residual_norm_from_cost(f),
                    cg_iterations=cg.iterations,
                    cg_residual_norm=cg.residual_norm,
                    negative_curvature=cg.negative_curvature,
                    preconditioner_fallback=cg.preconditioner_fallback,
                    cg_reason=cg.reason,
                    direction_fallback=direction_fallback,
                    predicted_decrease=predicted,
                    actual_decrease=actual,
                    damping_penalty=damping_penalty,
                )
            )

        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return x, info[-1].cost, info


__all__ = ["LevenbergMarquardt"]
