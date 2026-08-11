"""Internal implementation for nonlinear Riemannian conjugate gradient.

This implements the main structure of Manopt's MATLAB
``conjugategradient.m`` in a compact JAX-oriented form: optional
preconditioning, transported search directions, several common beta rules,
Powell restart, a pluggable line search and Manopt-like iteration
statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional
import math
import time

import jax.numpy as jnp

from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    cost_and_grad,
    gradient_value,
    initial_point,
    inner,
    make_info,
    precondition,
    require,
    stopping_reason,
    tree_lincomb,
    tree_neg,
    tree_sub,
    transport,
    validate_line_search_result,
)
from .linesearch import AdaptiveArmijo, LineSearchProtocol, LineSearchState
from geojax.geometry.base import validate_positive


@dataclass(frozen=True)
class ConjugateGradient:
    """Nonlinear Riemannian conjugate-gradient solver.

    ``beta_type`` may be ``'S-D'``/``'steep'``, ``'F-R'``, ``'P-R'``,
    ``'H-S'``, ``'H-Z'``, ``'L-S'``, ``'P-R-SATO'`` or ``'H-S-SATO'``.
    The default is ``'H-S'``, following Manopt.
    """

    tolgradnorm: float = 1e-6
    maxiter: int = 1000
    maxtime: float = math.inf
    minstepsize: float = 1e-10
    verbosity: int = 2

    beta_type: str = "H-S"
    orth_value: float = math.inf

    line_search: LineSearchProtocol = field(default_factory=AdaptiveArmijo)

    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None
    key: Optional[Array] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        """Solve ``problem`` and return ``(solution, final_cost, history)``."""
        return _solve_conjugate_gradient(problem, getattr(problem, "x0", None), self)


def _finite_scalar(value: Any, default: float = 0.0) -> float:
    out = float(jnp.asarray(value))
    return out if math.isfinite(out) else default


def _safe_divide(num: Any, den: Any, default: float = 0.0) -> float:
    num_f = _finite_scalar(num, default=default)
    den_f = _finite_scalar(den, default=math.nan)
    if not math.isfinite(den_f) or abs(den_f) <= 1e-300:
        return default
    out = num_f / den_f
    return out if math.isfinite(out) else default


def _safe_preconditioned_gradient(
    M: Any, x: Array, grad: Array, candidate: Array
) -> tuple[Array, Array, bool]:
    """Return a positive preconditioned gradient or the raw gradient."""
    pairing = inner(M, x, grad, candidate)
    pairing_f = _finite_scalar(pairing, default=math.nan)
    if math.isfinite(pairing_f) and pairing_f > 0.0:
        return candidate, pairing, False
    raw_pairing = inner(M, x, grad, grad)
    return grad, raw_pairing, True


def _compute_beta_and_direction(
    *,
    M: Any,
    options: ConjugateGradient,
    x: Array,
    newx: Array,
    grad: Array,
    newgrad: Array,
    Pgrad: Array,
    Pnewgrad: Array,
    desc_dir: Array,
    gradPgrad: Array,
    newgradPnewgrad: Array,
    gradnorm: Array,
) -> tuple[float, Array]:
    beta_type = options.beta_type.upper()
    if beta_type in {"STEEP", "S-D"}:
        return 0.0, tree_neg(Pnewgrad)

    oldgrad = transport(M, x, newx, grad)
    oldPgrad = transport(M, x, newx, Pgrad)

    newgradPnewgrad_f = _finite_scalar(newgradPnewgrad, default=0.0)
    if newgradPnewgrad_f <= 0.0:
        return 0.0, tree_neg(Pnewgrad)

    orth_grads = _safe_divide(inner(M, newx, oldgrad, Pnewgrad), newgradPnewgrad, default=0.0)
    if abs(orth_grads) >= options.orth_value:
        return 0.0, tree_neg(Pnewgrad)

    old_desc_dir = desc_dir
    desc_transp = transport(M, x, newx, desc_dir)
    beta = 0.0

    if beta_type == "F-R":
        beta = _safe_divide(newgradPnewgrad, gradPgrad)

    elif beta_type == "P-R":
        diff = tree_sub(newgrad, oldgrad)
        ip_diff = inner(M, newx, Pnewgrad, diff)
        beta = max(0.0, _safe_divide(ip_diff, gradPgrad))

    elif beta_type == "P-R-SATO":
        numo = newgradPnewgrad - inner(M, newx, newgrad, oldPgrad)
        beta_prp = _safe_divide(numo, gradPgrad)
        beta_fr = _safe_divide(newgradPnewgrad, gradPgrad)
        beta = max(0.0, min(beta_prp, beta_fr))

    elif beta_type == "H-S":
        diff = tree_sub(newgrad, oldgrad)
        ip_diff = inner(M, newx, Pnewgrad, diff)
        deno = inner(M, newx, diff, desc_transp)
        beta = max(0.0, _safe_divide(ip_diff, deno))

    elif beta_type == "H-S-SATO":
        numo = newgradPnewgrad - inner(M, newx, newgrad, oldPgrad)
        deno = inner(M, newx, newgrad, desc_transp) - inner(M, x, grad, old_desc_dir)
        beta_hs = _safe_divide(numo, deno)
        beta_dy = _safe_divide(newgradPnewgrad, deno)
        beta = max(min(beta_hs, beta_dy), 0.0)

    elif beta_type == "H-Z":
        diff = tree_sub(newgrad, oldgrad)
        Pdiff = tree_sub(Pnewgrad, oldPgrad)
        deno = inner(M, newx, diff, desc_transp)
        deno_f = _finite_scalar(deno, default=math.nan)
        if not math.isfinite(deno_f) or abs(deno_f) <= 1e-300:
            beta = 0.0
        else:
            numo = inner(M, newx, diff, Pnewgrad)
            correction = (
                2.0 * inner(M, newx, diff, Pdiff) * inner(M, newx, desc_transp, newgrad) / deno
            )
            beta = _safe_divide(numo - correction, deno)
            desc_norm = _finite_scalar(M.norm(newx, desc_transp), default=0.0)
            gradnorm_f = max(_finite_scalar(gradnorm, default=0.0), 1e-300)
            if desc_norm > 0.0:
                eta_hz = -1.0 / (desc_norm * min(0.01, gradnorm_f))
                beta = max(beta, eta_hz)

    elif beta_type == "L-S":
        numo = newgradPnewgrad - inner(M, newx, newgrad, oldPgrad)
        deno = -inner(M, x, grad, old_desc_dir)
        beta_ls = _safe_divide(numo, deno)
        beta_cd = _safe_divide(newgradPnewgrad, deno)
        beta = max(0.0, min(beta_ls, beta_cd))

    else:
        raise ValueError(
            "Unknown beta_type. Expected one of: 'steep', 'S-D', 'F-R', "
            "'P-R', 'H-S', 'H-Z', 'L-S', 'P-R-SATO', 'H-S-SATO'."
        )

    if not math.isfinite(beta):
        beta = 0.0

    new_desc_dir = tree_lincomb(-1.0, Pnewgrad, beta, desc_transp)
    return float(beta), new_desc_dir


def _solve_conjugate_gradient(
    problem: Any,
    x: Optional[Array] = None,
    options: ConjugateGradient | None = None,
) -> tuple[Array, float, List[InfoEntry]]:
    """Internal nonlinear Riemannian conjugate-gradient iteration engine."""
    options = ConjugateGradient() if options is None else options
    valid_beta_types = {
        "STEEP",
        "S-D",
        "F-R",
        "P-R",
        "H-S",
        "H-Z",
        "L-S",
        "P-R-SATO",
        "H-S-SATO",
    }
    if not isinstance(options.beta_type, str):
        raise TypeError("beta_type must be a string.")
    if options.beta_type.upper() not in valid_beta_types:
        raise ValueError(
            "Unknown beta_type. Expected one of: 'steep', 'S-D', 'F-R', "
            "'P-R', 'H-S', 'H-Z', 'L-S', 'P-R-SATO', 'H-S-SATO'."
        )
    if isinstance(options.orth_value, bool):
        raise TypeError("orth_value must be a real scalar, not a boolean.")
    if math.isinf(float(options.orth_value)):
        if float(options.orth_value) < 0.0:
            raise ValueError("orth_value must be positive and not NaN.")
    else:
        validate_positive(options.orth_value, name="orth_value")
    if math.isnan(float(options.orth_value)):
        raise ValueError("orth_value must be positive and not NaN.")
    M = require(problem, "M")
    require(problem, "cost")

    sol = initial_point(problem, x, options.key)
    start_time = time.perf_counter()
    search_state: LineSearchState | None = None
    info: List[InfoEntry] = []

    cost_value, grad = cost_and_grad(problem, sol)
    gradnorm = M.norm(sol, grad)
    Pgrad, gradPgrad, preconditioner_fallback = _safe_preconditioned_gradient(
        M, sol, grad, precondition(problem, sol, grad)
    )
    desc_dir = tree_neg(Pgrad)
    beta = 0.0

    info.append(
        make_info(
            iter=0,
            cost=cost_value,
            gradnorm=gradnorm,
            stepsize=math.nan,
            start_time=start_time,
            linesearch=None,
            problem=problem,
            x=sol,
            options=options,
            beta=beta,
            preconditioner_fallback=preconditioner_fallback,
        )
    )

    if options.verbosity >= 2:
        print(" iter\t        cost val\t    grad. norm\t       beta")

    while True:
        current = info[-1]
        if options.verbosity >= 2:
            beta_display = 0.0 if current.beta is None else current.beta
            print(
                f"{current.iter:5d}\t{current.cost:+.16e}\t{current.gradnorm:.8e}\t{beta_display:+.3e}"
            )

        reason = stopping_reason(problem, sol, info, options)
        if reason:
            info[-1] = replace(info[-1], reason=reason)
            if options.verbosity >= 1:
                print(reason)
            break

        df0 = inner(M, sol, grad, desc_dir)
        df0_float = _finite_scalar(df0, default=math.inf)

        if df0_float >= 0.0 or not math.isfinite(df0_float):
            desc_dir = tree_neg(grad)
            df0 = -inner(M, sol, grad, grad)
            df0_float = _finite_scalar(df0, default=-math.inf)
            beta = 0.0
            preconditioner_fallback = True

        result = validate_line_search_result(
            problem,
            options.line_search.search(
                problem,
                sol,
                desc_dir,
                cost_value,
                df0,
                state=search_state,
            ),
        )
        search_state = result.state
        newsol = result.point

        newcost = result.cost
        newgrad = (
            result.gradient if result.gradient is not None else gradient_value(problem, newsol)
        )
        newgradnorm = M.norm(newsol, newgrad)
        Pnewgrad, newgradPnewgrad, new_preconditioner_fallback = _safe_preconditioned_gradient(
            M, newsol, newgrad, precondition(problem, newsol, newgrad)
        )

        beta, new_desc_dir = _compute_beta_and_direction(
            M=M,
            options=options,
            x=sol,
            newx=newsol,
            grad=grad,
            newgrad=newgrad,
            Pgrad=Pgrad,
            Pnewgrad=Pnewgrad,
            desc_dir=desc_dir,
            gradPgrad=gradPgrad,
            newgradPnewgrad=newgradPnewgrad,
            gradnorm=gradnorm,
        )

        sol = newsol
        cost_value = newcost
        grad = newgrad
        gradnorm = newgradnorm
        Pgrad = Pnewgrad
        gradPgrad = newgradPnewgrad
        desc_dir = new_desc_dir
        preconditioner_fallback = new_preconditioner_fallback

        info.append(
            make_info(
                iter=current.iter + 1,
                cost=cost_value,
                gradnorm=gradnorm,
                stepsize=result.stepsize,
                start_time=start_time,
                linesearch=result.stats,
                problem=problem,
                x=sol,
                options=options,
                beta=beta,
                preconditioner_fallback=preconditioner_fallback,
            )
        )

    if options.verbosity >= 1:
        print(f"Total time is {info[-1].time:.6f} [s]")

    final_cost = info[-1].cost
    return sol, final_cost, info


__all__ = ["ConjugateGradient"]
