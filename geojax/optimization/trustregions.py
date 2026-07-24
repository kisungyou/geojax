"""Riemannian trust-regions solver."""

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
    precondition,
    print_iteration,
    print_iteration_header,
    retract,
    require,
    stopping_reason,
    tree_lincomb,
    tree_neg,
    tree_zeros_like,
)


@dataclass(frozen=True)
class TrustRegions:
    """Approximate Riemannian trust-regions method.

    Uses truncated conjugate gradient to approximately minimize the quadratic
    model. A geometry must advertise an exact automatic Hessian conversion or
    the problem must supply ``rhess_vec`` explicitly.
    """

    requires_gradient: bool = True
    tolgradnorm: float = 1e-6
    maxiter: int = 200
    maxtime: float = math.inf
    minstepsize: float = 0.0
    verbosity: int = 2
    initial_radius: float = 1.0
    max_radius: float = 100.0
    rho_prime: float = 0.1
    kappa: float = 0.1
    theta: float = 1.0
    maxinner: int = 250
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        M = require(problem, "M")
        x = require(problem, "x0")
        Delta = float(self.initial_radius)
        start_time = time.perf_counter()
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
                rho=None,
                accepted=None,
            )
        )
        print_iteration_header(self.verbosity, include_rho=True)
        while True:
            print_iteration(info[-1], self.verbosity, include_rho=True)
            reason = stopping_reason(problem, x, info, self)
            if reason:
                info[-1] = InfoEntry(**{**info[-1].__dict__, "reason": reason})
                if self.verbosity >= 1:
                    print(reason)
                break

            eta, hit_boundary, tcg_info = _truncated_cg(problem, x, g, Delta, self)
            Heta = problem.rhess_vec(x, eta)
            pred = -inner(M, x, g, eta) - 0.5 * inner(M, x, eta, Heta)
            pred_f = max(as_float(pred), 0.0)
            stepnorm = as_float(M.norm(x, eta))
            x_trial = retract(M, x, eta, 1.0)
            f_trial = cost_value(problem, x_trial)
            actual = as_float(f) - as_float(f_trial)
            rho = actual / pred_f if pred_f > 1e-300 else -math.inf

            accepted = bool(rho > self.rho_prime)
            if accepted:
                x = x_trial
                f, g = cost_and_grad(problem, x)
                gnorm = M.norm(x, g)
            else:
                f, g = cost_and_grad(problem, x)
                gnorm = M.norm(x, g)

            if rho < 0.25:
                Delta = max(0.25 * stepnorm, 1e-16)
            elif rho > 0.75 and hit_boundary:
                Delta = min(2.0 * Delta, float(self.max_radius))

            info.append(
                make_info(
                    iter=info[-1].iter + 1,
                    cost=f,
                    gradnorm=gnorm,
                    stepsize=stepnorm,
                    start_time=start_time,
                    linesearch=None,
                    problem=problem,
                    x=x,
                    solver=self,
                    rho=rho,
                    accepted=accepted,
                    hit_boundary=hit_boundary,
                    **tcg_info,
                )
            )
        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return x, info[-1].cost, info


def _tau_to_boundary(M: Any, x: Array, eta: Array, d: Array, Delta: float) -> float:
    a = as_float(inner(M, x, d, d))
    b = 2.0 * as_float(inner(M, x, eta, d))
    c = as_float(inner(M, x, eta, eta)) - Delta * Delta
    disc = max(b * b - 4.0 * a * c, 0.0)
    if a <= 0.0:
        return 0.0
    return float((-b + math.sqrt(disc)) / (2.0 * a))


def _truncated_cg(
    problem: Any, x: Array, g: Array, Delta: float, solver: TrustRegions
) -> tuple[Array, bool, dict[str, Any]]:
    M = problem.M
    eta = tree_zeros_like(g)
    r = g
    z = precondition(problem, x, r)
    d = tree_neg(z)
    rnorm0 = as_float(M.norm(x, r))
    rnorm = rnorm0
    tol = min(float(solver.kappa) * rnorm0, rnorm0 ** (1.0 + float(solver.theta)))
    tol = max(tol, 1e-14)
    hit_boundary = False
    negative_curvature = False
    inner_iterations = 0

    for inner_iterations in range(1, int(solver.maxinner) + 1):
        Hd = problem.rhess_vec(x, d)
        dHd = as_float(inner(M, x, d, Hd))
        if dHd <= 0.0 or not math.isfinite(dHd):
            tau = _tau_to_boundary(M, x, eta, d, Delta)
            eta = tree_lincomb(1.0, eta, tau, d)
            hit_boundary = True
            negative_curvature = True
            break
        rz = as_float(inner(M, x, r, z))
        alpha = rz / dHd
        eta_next = tree_lincomb(1.0, eta, alpha, d)
        if as_float(M.norm(x, eta_next)) >= Delta:
            tau = _tau_to_boundary(M, x, eta, d, Delta)
            eta = tree_lincomb(1.0, eta, tau, d)
            hit_boundary = True
            break
        eta = eta_next
        r_next = tree_lincomb(1.0, r, alpha, Hd)
        rnorm_next = as_float(M.norm(x, r_next))
        if rnorm_next <= tol:
            r = r_next
            break
        z_next = precondition(problem, x, r_next)
        rz_next = as_float(inner(M, x, r_next, z_next))
        beta = rz_next / max(rz, 1e-300)
        d = tree_lincomb(-1.0, z_next, beta, d)
        r = r_next
        z = z_next
        rnorm = rnorm_next
    return (
        eta,
        hit_boundary,
        {
            "negative_curvature": negative_curvature,
            "tcg_inner_iterations": inner_iterations,
            "tcg_residual_norm": rnorm,
        },
    )


__all__ = ["TrustRegions"]
