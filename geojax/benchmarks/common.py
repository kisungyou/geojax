"""Shared benchmark helpers."""

from __future__ import annotations

from typing import Any, Callable
import time

import jax

from geojax.optimization import (
    BarzilaiBorwein,
    ConjugateGradient,
    LBFGS,
    Minimize,
    SteepestDescent,
    TrustRegions,
)


def default_solvers(maxiter: int = 50, tolgradnorm: float = 1e-6) -> dict[str, Any]:
    options = dict(maxiter=maxiter, tolgradnorm=tolgradnorm, verbosity=0)
    return {
        "SteepestDescent": SteepestDescent(**options),
        "ConjugateGradient": ConjugateGradient(**options),
        "TrustRegions": TrustRegions(**options),
        "BarzilaiBorwein": BarzilaiBorwein(**options),
        "LBFGS": LBFGS(**options),
    }


def run_solver(
    M: Any, cost: Callable[[Any], Any], x0: Any, name: str, solver: Any
) -> dict[str, Any]:
    problem = Minimize(M=M, cost=cost, x0=x0, solver=solver, key=0)
    tic = time.perf_counter()
    sol, final_cost, info = problem.solve()
    jax.tree_util.tree_map(
        lambda z: z.block_until_ready() if hasattr(z, "block_until_ready") else z, sol
    )
    elapsed = time.perf_counter() - tic
    last = info[-1]
    return {
        "solver": name,
        "final_cost": float(final_cost),
        "gradnorm": float(last.gradnorm),
        "iterations": int(last.iter),
        "time_sec": float(elapsed),
        "success": "tolerance" in last.reason.lower(),
        "reason": last.reason,
    }


def run_suite(
    M: Any, cost: Callable[[Any], Any], x0: Any, *, maxiter: int = 50
) -> list[dict[str, Any]]:
    return [
        run_solver(M, cost, x0, name, solver)
        for name, solver in default_solvers(maxiter=maxiter).items()
    ]


def print_rows(rows: list[dict[str, Any]]) -> None:
    columns = ["solver", "final_cost", "gradnorm", "iterations", "time_sec", "success"]
    widths = {c: max(len(c), *(len(_fmt(row[c])) for row in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    print("-" * (sum(widths.values()) + 2 * (len(columns) - 1)))
    for row in rows:
        print("  ".join(_fmt(row[c]).ljust(widths[c]) for c in columns))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = ["default_solvers", "run_solver", "run_suite", "print_rows"]
