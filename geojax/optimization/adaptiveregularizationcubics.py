"""Adaptive regularization with cubics for Riemannian optimization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, List, Optional
import math
import time

from geojax.geometry.base import validate_integer, validate_nonnegative, validate_positive

from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
    cost_and_grad,
    cost_value,
    get,
    inner,
    make_info,
    print_iteration,
    print_iteration_header,
    require,
    retract,
    stopping_reason,
    tree_lincomb,
    tree_neg,
    validate_tangent,
)


@dataclass(frozen=True)
class AdaptiveRegularizationCubics:
    r"""Adaptive cubic regularization on a Riemannian manifold.

    At ``x`` the solver approximately minimizes

    ``<g, eta> + 0.5 <eta, Hess f(x)[eta]> + sigma/3 ||eta||^3``

    in the tangent space. The subproblem begins at its exact Cauchy point and
    may be refined by model-gradient steps. The ratio of actual to predicted
    decrease controls acceptance and the next regularization parameter.
    """

    requires_gradient: bool = True
    tolgradnorm: float = 1e-6
    maxiter: int = 200
    maxtime: float = math.inf
    minstepsize: float = 0.0
    verbosity: int = 2
    initial_sigma: float = 1.0
    min_sigma: float = 1e-12
    max_sigma: float = 1e12
    acceptance_threshold: float = 0.1
    very_successful_threshold: float = 0.9
    decrease_factor: float = 0.5
    increase_factor: float = 2.0
    subproblem_iterations: int = 10
    subproblem_tolerance: float = 0.1
    subproblem_backtracks: int = 20
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        M = require(problem, "M")
        x = require(problem, "x0")
        hessian_vector = get(problem, "rhess_vec", None)
        if hessian_vector is None:
            raise ValueError("AdaptiveRegularizationCubics requires problem.rhess_vec(x, u).")
        initial_sigma = validate_positive(self.initial_sigma, name="initial_sigma")
        min_sigma = validate_positive(self.min_sigma, name="min_sigma")
        max_sigma = validate_positive(self.max_sigma, name="max_sigma")
        if not min_sigma <= initial_sigma <= max_sigma:
            raise ValueError(
                "Sigma bounds must satisfy 0 < min_sigma <= initial_sigma <= max_sigma."
            )
        decrease_factor = validate_positive(self.decrease_factor, name="decrease_factor")
        increase_factor = validate_positive(self.increase_factor, name="increase_factor")
        acceptance_threshold = validate_nonnegative(
            self.acceptance_threshold, name="acceptance_threshold"
        )
        very_successful_threshold = validate_positive(
            self.very_successful_threshold, name="very_successful_threshold"
        )
        if decrease_factor >= 1.0 or increase_factor <= 1.0:
            raise ValueError("decrease_factor must be in (0, 1) and increase_factor must exceed 1.")
        if not acceptance_threshold < very_successful_threshold < 1.0:
            raise ValueError(
                "Thresholds must satisfy 0 <= acceptance_threshold < very_successful_threshold < 1."
            )
        subproblem_iterations = validate_integer(
            self.subproblem_iterations,
            name="subproblem_iterations",
        )
        subproblem_backtracks = validate_integer(
            self.subproblem_backtracks,
            name="subproblem_backtracks",
        )
        if subproblem_iterations < 0 or subproblem_backtracks <= 0:
            raise ValueError(
                "subproblem_iterations must be nonnegative and subproblem_backtracks positive."
            )
        subproblem_tolerance = validate_nonnegative(
            self.subproblem_tolerance,
            name="subproblem_tolerance",
        )

        sigma = initial_sigma
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
                sigma=sigma,
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

            sigma_used = sigma
            step, model_value, subproblem_info = _solve_cubic_subproblem(
                M,
                x,
                g,
                lambda direction: hessian_vector(x, direction),
                sigma_used,
                max_iterations=subproblem_iterations,
                tolerance=subproblem_tolerance,
                max_backtracks=subproblem_backtracks,
            )
            predicted = max(-model_value, 0.0)
            stepnorm = as_float(M.norm(x, step))
            trial = retract(M, x, step, 1.0)
            trial_cost = cost_value(problem, trial)
            actual = as_float(f) - as_float(trial_cost)
            rho = actual / predicted if predicted > 0.0 else -math.inf
            accepted = bool(math.isfinite(rho) and rho >= acceptance_threshold)

            if rho >= very_successful_threshold:
                sigma = max(min_sigma, decrease_factor * sigma)
            elif rho < acceptance_threshold or not math.isfinite(rho):
                sigma = min(max_sigma, increase_factor * sigma)

            if accepted:
                x = trial
                f, g = cost_and_grad(problem, x)
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
                    sigma=sigma_used,
                    next_sigma=sigma,
                    predicted_decrease=predicted,
                    actual_decrease=actual,
                    **subproblem_info,
                )
            )

        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return x, info[-1].cost, info


def _solve_cubic_subproblem(
    M: Any,
    x: Array,
    gradient: Array,
    hessian_vector: Any,
    sigma: float,
    *,
    max_iterations: int,
    tolerance: float,
    max_backtracks: int,
) -> tuple[Array, float, dict[str, Any]]:
    """Return a Cauchy-initialized approximate minimizer of the cubic model."""

    gradient_norm = as_float(M.norm(x, gradient))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("The cubic subproblem requires a finite nonzero gradient.")
    unit_descent = tree_lincomb(-1.0 / gradient_norm, gradient)
    hessian_unit = validate_tangent(
        M,
        x,
        hessian_vector(unit_descent),
        name="cubic Hessian output",
    )
    curvature = as_float(inner(M, x, unit_descent, hessian_unit))
    if not math.isfinite(curvature):
        raise FloatingPointError("The cubic Hessian produced nonfinite Cauchy curvature.")
    discriminant = curvature * curvature + 4.0 * sigma * gradient_norm
    if not math.isfinite(discriminant) or discriminant < 0.0:
        raise FloatingPointError("The cubic Cauchy-point discriminant is nonfinite.")
    root = math.sqrt(discriminant)
    if curvature > 0.0:
        cauchy_length = 2.0 * gradient_norm / (curvature + root)
    else:
        cauchy_length = (-curvature + root) / (2.0 * sigma)
    step = tree_lincomb(cauchy_length, unit_descent)

    def model(point: Array) -> tuple[float, Array]:
        hessian_point = validate_tangent(
            M,
            x,
            hessian_vector(point),
            name="cubic Hessian output",
        )
        norm_point = as_float(M.norm(x, point))
        value = (
            as_float(inner(M, x, gradient, point))
            + 0.5 * as_float(inner(M, x, point, hessian_point))
            + sigma * norm_point**3 / 3.0
        )
        model_gradient = tree_lincomb(
            1.0,
            gradient,
            1.0,
            hessian_point,
            sigma * norm_point,
            point,
        )
        return value, model_gradient

    model_value, model_gradient = model(step)
    model_gradnorm = as_float(M.norm(x, model_gradient))
    if not math.isfinite(model_value) or not math.isfinite(model_gradnorm):
        raise FloatingPointError("The cubic subproblem model is nonfinite at its Cauchy point.")
    inner_iterations = 0
    converged = model_gradnorm <= tolerance * gradient_norm
    reason = "model-gradient tolerance" if converged else "maximum inner iterations"
    for candidate_iteration in range(1, max_iterations + 1):
        if converged:
            break
        inner_iterations = candidate_iteration
        direction = tree_neg(model_gradient)
        slope = -(model_gradnorm * model_gradnorm)
        alpha = 1.0
        accepted = False
        for _ in range(max(1, int(max_backtracks))):
            candidate = tree_lincomb(1.0, step, alpha, direction)
            candidate_value, candidate_gradient = model(candidate)
            if (
                math.isfinite(candidate_value)
                and candidate_value <= model_value + 1e-4 * alpha * slope
            ):
                step = candidate
                model_value = candidate_value
                model_gradient = candidate_gradient
                model_gradnorm = as_float(M.norm(x, model_gradient))
                if not math.isfinite(model_gradnorm):
                    raise FloatingPointError("The cubic model gradient became nonfinite.")
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            reason = "subproblem line search failed"
            break
        if model_gradnorm <= tolerance * gradient_norm:
            converged = True
            reason = "model-gradient tolerance"
            break

    return (
        step,
        model_value,
        {
            "subproblem_iterations": inner_iterations,
            "subproblem_gradient_norm": model_gradnorm,
            "cauchy_length": cauchy_length,
            "subproblem_converged": converged,
            "subproblem_reason": reason,
        },
    )


__all__ = ["AdaptiveRegularizationCubics"]
