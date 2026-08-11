"""Shared benchmark helpers."""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

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
    M: Any,
    cost: Callable[[Any], Any],
    x0: Any,
    name: str,
    solver: Any,
    *,
    rhess_vec: Callable[[Any, Any], Any] | None = None,
    warmup: int = 1,
    repeats: int = 5,
) -> dict[str, Any]:
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer.")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")

    def solve_once() -> tuple[Any, Any, Any, float]:
        problem = Minimize(
            M=M,
            cost=cost,
            x0=x0,
            solver=solver,
            key=0,
            rhess_vec=rhess_vec,
        )
        tic = time.perf_counter()
        solution, value, history = problem.solve()
        jax.tree_util.tree_map(
            lambda z: z.block_until_ready() if hasattr(z, "block_until_ready") else z,
            (solution, value),
        )
        return solution, value, history, time.perf_counter() - tic

    sol, final_cost, info, cold_time = solve_once()
    for _ in range(warmup):
        sol, final_cost, info, _ = solve_once()
    timings = []
    for _ in range(repeats):
        sol, final_cost, info, elapsed = solve_once()
        timings.append(elapsed)

    last = info[-1]
    dtypes = sorted(
        {
            str(getattr(leaf, "dtype"))
            for leaf in jax.tree_util.tree_leaves(x0)
            if hasattr(leaf, "dtype")
        }
    )
    dtype = ",".join(dtypes) if dtypes else "unknown"
    line_search_cost_evaluations = sum(
        entry.linesearch.costevals for entry in info if entry.linesearch is not None
    )
    line_search_gradient_evaluations = sum(
        entry.linesearch.gradevals for entry in info if entry.linesearch is not None
    )
    return {
        "solver": name,
        "final_cost": float(final_cost),
        "gradnorm": float(last.gradnorm),
        "iterations": int(last.iter),
        "cold_time_sec": float(cold_time),
        "time_sec": float(statistics.median(timings)),
        "time_std_sec": float(statistics.pstdev(timings)),
        "repeats": repeats,
        "backend": jax.default_backend(),
        "dtype": dtype,
        "line_search_cost_evaluations": line_search_cost_evaluations,
        "line_search_gradient_evaluations": line_search_gradient_evaluations,
        "success": "tolerance" in last.reason.lower(),
        "reason": last.reason,
    }


def run_suite(
    M: Any,
    cost: Callable[[Any], Any],
    x0: Any,
    *,
    maxiter: int = 50,
    rhess_vec: Callable[[Any, Any], Any] | None = None,
    warmup: int = 1,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    rows = []
    for name, solver in default_solvers(maxiter=maxiter).items():
        # Avoid crediting a later solver with executable caches populated by an
        # earlier solver. ``cold_time_sec`` still includes the first full solve,
        # while the repeated timing below represents warmed steady-state use.
        jax.clear_caches()
        if (
            isinstance(solver, TrustRegions)
            and rhess_vec is None
            and M.operation_kind("ehess_to_rhess") != "exact"
        ):
            rows.append(
                {
                    "solver": name,
                    "final_cost": float("nan"),
                    "gradnorm": float("nan"),
                    "iterations": 0,
                    "cold_time_sec": 0.0,
                    "time_sec": 0.0,
                    "time_std_sec": 0.0,
                    "repeats": repeats,
                    "backend": jax.default_backend(),
                    "dtype": ",".join(
                        sorted(
                            {
                                str(getattr(leaf, "dtype"))
                                for leaf in jax.tree_util.tree_leaves(x0)
                                if hasattr(leaf, "dtype")
                            }
                        )
                    )
                    or "unknown",
                    "line_search_cost_evaluations": 0,
                    "line_search_gradient_evaluations": 0,
                    "success": False,
                    "reason": "unsupported without an explicit rhess_vec",
                }
            )
            continue
        rows.append(
            run_solver(
                M,
                cost,
                x0,
                name,
                solver,
                rhess_vec=rhess_vec,
                warmup=warmup,
                repeats=repeats,
            )
        )
    return rows


def print_rows(rows: list[dict[str, Any]]) -> None:
    columns = [
        "solver",
        "final_cost",
        "gradnorm",
        "iterations",
        "cold_time_sec",
        "time_sec",
        "time_std_sec",
        "repeats",
        "backend",
        "dtype",
        "line_search_cost_evaluations",
        "line_search_gradient_evaluations",
        "success",
    ]
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
