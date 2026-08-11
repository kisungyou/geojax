from __future__ import annotations

import math

from geojax.benchmarks import (
    correlation_frechet,
    grassmann_subspace,
    product_model,
    spd_frechet,
    sphere_eigen,
)


def test_benchmark_modules_return_solver_rows(monkeypatch):
    # The benchmark CLI deliberately clears caches between solvers. The schema
    # test only verifies result contracts and must not invalidate caches shared
    # with the rest of the test process.
    monkeypatch.setattr("geojax.benchmarks.common.jax.clear_caches", lambda: None)
    scenarios = [
        sphere_eigen.run(n=3, maxiter=1, warmup=0, repeats=1),
        grassmann_subspace.run(n=4, rank=1, maxiter=1, warmup=0, repeats=1),
        spd_frechet.run(n=2, samples=3, maxiter=1, warmup=0, repeats=1),
        correlation_frechet.run(n=3, samples=3, maxiter=1, warmup=0, repeats=1),
        product_model.run(maxiter=1, warmup=0, repeats=1),
    ]
    for rows in scenarios:
        assert len(rows) == 5
        assert {"solver", "final_cost", "gradnorm", "iterations", "time_sec", "success"} <= set(
            rows[0]
        )
        assert {
            "cold_time_sec",
            "time_std_sec",
            "repeats",
            "backend",
            "dtype",
            "line_search_cost_evaluations",
            "line_search_gradient_evaluations",
        } <= set(rows[0])

    # Grassmann now advertises an exact ambient-to-Riemannian Hessian
    # conversion, so TrustRegions executes. The SPD and correlation objectives
    # still need an explicit Riemannian Hessian callback and report that
    # limitation instead of silently using a projected proxy.
    trust_region = next(row for row in scenarios[1] if row["solver"] == "TrustRegions")
    assert trust_region["success"] is False
    assert trust_region["iterations"] == 1
    assert math.isfinite(trust_region["final_cost"])
    assert math.isfinite(trust_region["gradnorm"])
    assert "maximum iteration" in trust_region["reason"].lower()

    for rows in scenarios[2:4]:
        trust_region = next(row for row in rows if row["solver"] == "TrustRegions")
        assert trust_region["success"] is False
        assert trust_region["iterations"] == 0
        assert "rhess_vec" in trust_region["reason"]
