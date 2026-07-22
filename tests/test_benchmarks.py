from __future__ import annotations

from geojax.benchmarks import (
    correlation_frechet,
    grassmann_subspace,
    product_model,
    spd_frechet,
    sphere_eigen,
)


def test_benchmark_modules_return_solver_rows():
    scenarios = [
        sphere_eigen.run(n=3, maxiter=1),
        grassmann_subspace.run(n=4, rank=1, maxiter=1),
        spd_frechet.run(n=2, samples=3, maxiter=1),
        correlation_frechet.run(n=3, samples=3, maxiter=1),
        product_model.run(maxiter=1),
    ]
    for rows in scenarios:
        assert len(rows) == 5
        assert {"solver", "final_cost", "gradnorm", "iterations", "time_sec", "success"} <= set(
            rows[0]
        )
