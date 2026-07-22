"""Benchmark dominant eigenvector optimization on the sphere."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.benchmarks.common import print_rows, run_suite
from geojax.geometry import Sphere


def run(n: int = 6, maxiter: int = 50, seed: int = 0) -> list[dict[str, object]]:
    key = jax.random.key(seed)
    key_A, key_x = jax.random.split(key)
    B = jax.random.normal(key_A, shape=(n, n))
    A = 0.5 * (B + B.T)
    M = Sphere(size=n)
    x0 = M.random_point(key_x)

    def cost(x):
        return -jnp.dot(x, A @ x)

    return run_suite(M, cost, x0, maxiter=maxiter)


def main() -> None:
    print_rows(run())


if __name__ == "__main__":
    main()
