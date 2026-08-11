"""Riemannian trust-regions solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import math
import time

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
    cost_and_grad,
    cost_value,
    gradient_value,
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
    validate_tangent,
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
        initial_radius = validate_positive(self.initial_radius, name="initial_radius")
        max_radius = validate_positive(self.max_radius, name="max_radius")
        if initial_radius > max_radius:
            raise ValueError("Radii must satisfy 0 < initial_radius <= max_radius.")
        rho_prime = validate_nonnegative(self.rho_prime, name="rho_prime")
        kappa = validate_positive(self.kappa, name="kappa")
        theta = validate_positive(self.theta, name="theta")
        if rho_prime >= 0.25:
            raise ValueError("rho_prime must lie in [0, 0.25).")
        if kappa >= 1.0 or theta > 1.0:
            raise ValueError("kappa must lie in (0, 1) and theta in (0, 1].")
        validate_integer(self.maxinner, name="maxinner", minimum=1)
        Delta = initial_radius
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

            radius_used = Delta
            eta, hit_boundary, tcg_info = _truncated_cg(problem, x, g, radius_used, self)
            Heta = validate_tangent(
                M,
                x,
                problem.rhess_vec(x, eta),
                name="trust-region Hessian output",
            )
            pred = -inner(M, x, g, eta) - 0.5 * inner(M, x, eta, Heta)
            pred_f = max(as_float(pred), 0.0)
            stepnorm = as_float(M.norm(x, eta))
            x_trial = retract(M, x, eta, 1.0)
            f_trial = cost_value(problem, x_trial)
            actual = as_float(f) - as_float(f_trial)
            rho = actual / pred_f if pred_f > 0.0 else -math.inf
            if not math.isfinite(rho):
                rho = -math.inf

            accepted = bool(rho > rho_prime)
            if accepted:
                x = x_trial
                f = f_trial
                g = gradient_value(problem, x)
                gnorm = M.norm(x, g)

            if rho < 0.25:
                Delta = 0.25 * Delta
            elif rho > 0.75 and hit_boundary:
                Delta = min(2.0 * Delta, max_radius)

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
                    radius=Delta,
                    radius_used=radius_used,
                    predicted_decrease=pred_f,
                    actual_decrease=actual,
                    **tcg_info,
                )
            )
        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return x, info[-1].cost, info


def _tau_to_boundary(M: Any, x: Array, eta: Array, d: Array, Delta: float) -> float:
    dnorm = as_float(M.norm(x, d))
    etanorm = as_float(M.norm(x, eta))
    if not all(math.isfinite(value) for value in (dnorm, etanorm, Delta)):
        raise FloatingPointError("Trust-region boundary equation is nonfinite.")
    if dnorm <= 0.0 or Delta <= 0.0:
        raise ValueError("Trust-region boundary direction must have positive norm.")

    # Solve in units of max(||eta||, Delta). This avoids underflow in Delta**2
    # and the quadratic coefficients when a very small trust region is used.
    scale = max(etanorm, Delta)
    d_unit = tree_lincomb(1.0 / dnorm, d)
    eta_scaled = tree_lincomb(1.0 / scale, eta)
    a = as_float(inner(M, x, d_unit, d_unit))
    b = 2.0 * as_float(inner(M, x, eta_scaled, d_unit))
    c = as_float(inner(M, x, eta_scaled, eta_scaled)) - (Delta / scale) ** 2
    if not all(math.isfinite(value) for value in (a, b, c)) or a <= 0.0:
        raise FloatingPointError("Trust-region boundary equation is ill-conditioned.")
    disc = max(b * b - 4.0 * a * c, 0.0)
    root = math.sqrt(disc)
    if b >= 0.0 and b + root > 0.0:
        tau = (-2.0 * c) / (b + root)
    else:
        tau = (-b + root) / (2.0 * a)
    return float(max(tau, 0.0) * scale / dnorm)


def _truncated_cg(
    problem: Any, x: Array, g: Array, Delta: float, solver: TrustRegions
) -> tuple[Array, bool, dict[str, Any]]:
    M = problem.M
    eta = tree_zeros_like(g)
    r = g
    z = precondition(problem, x, r)
    rz = as_float(inner(M, x, r, z))
    preconditioner_fallback = False
    if not math.isfinite(rz) or rz <= 0.0:
        z = r
        rz = as_float(inner(M, x, r, r))
        preconditioner_fallback = True
    if not math.isfinite(rz) or (rz <= 0.0 and as_float(M.norm(x, r)) > 0.0):
        raise ValueError("The tangent metric must be positive on the trust-region residual.")
    d = tree_neg(z)
    rnorm0 = as_float(M.norm(x, r))
    if not math.isfinite(rnorm0):
        raise ValueError("The initial trust-region residual norm must be finite.")
    rnorm = rnorm0
    tol = min(float(solver.kappa) * rnorm0, rnorm0 ** (1.0 + float(solver.theta)))
    hit_boundary = False
    negative_curvature = False
    inner_iterations = 0

    converged = rnorm0 <= tol
    reason = "initial residual tolerance" if converged else "maximum inner iterations"
    maxinner = validate_integer(solver.maxinner, name="maxinner", minimum=1)
    for inner_iterations in range(1, maxinner + 1):
        if converged:
            inner_iterations = 0
            break
        Hd = validate_tangent(
            M,
            x,
            problem.rhess_vec(x, d),
            name="trust-region Hessian output",
        )
        dHd = as_float(inner(M, x, d, Hd))
        if not math.isfinite(dHd):
            raise FloatingPointError(
                "Trust-region Hessian produced nonfinite directional curvature."
            )
        if dHd <= 0.0:
            tau = _tau_to_boundary(M, x, eta, d, Delta)
            eta = tree_lincomb(1.0, eta, tau, d)
            hit_boundary = True
            negative_curvature = True
            reason = "negative curvature boundary"
            break
        alpha = rz / dHd
        eta_next = tree_lincomb(1.0, eta, alpha, d)
        if as_float(M.norm(x, eta_next)) >= Delta:
            tau = _tau_to_boundary(M, x, eta, d, Delta)
            eta = tree_lincomb(1.0, eta, tau, d)
            hit_boundary = True
            reason = "trust-region boundary"
            break
        eta = eta_next
        r_next = tree_lincomb(1.0, r, alpha, Hd)
        rnorm_next = as_float(M.norm(x, r_next))
        if not math.isfinite(rnorm_next):
            raise FloatingPointError(
                "Trust-region truncated CG produced a nonfinite residual norm."
            )
        if rnorm_next <= tol:
            r = r_next
            rnorm = rnorm_next
            converged = True
            reason = "residual tolerance"
            break
        z_next = precondition(problem, x, r_next)
        rz_next = as_float(inner(M, x, r_next, z_next))
        if not math.isfinite(rz_next) or rz_next <= 0.0:
            z_next = r_next
            rz_next = as_float(inner(M, x, r_next, r_next))
            if not math.isfinite(rz_next) or rz_next <= 0.0:
                raise ValueError(
                    "The tangent metric must be positive on a nonzero trust-region residual."
                )
            d = tree_neg(z_next)
            preconditioner_fallback = True
        else:
            beta = rz_next / max(rz, 1e-300)
            d = tree_lincomb(-1.0, z_next, beta, d)
        r = r_next
        z = z_next
        rz = rz_next
        rnorm = rnorm_next
    return (
        eta,
        hit_boundary,
        {
            "negative_curvature": negative_curvature,
            "tcg_inner_iterations": inner_iterations,
            "tcg_residual_norm": rnorm,
            "tcg_converged": converged,
            "tcg_reason": reason,
            "preconditioner_fallback": preconditioner_fallback,
        },
    )


__all__ = ["TrustRegions"]
