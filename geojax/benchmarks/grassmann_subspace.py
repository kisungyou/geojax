"""Benchmark subspace optimization on the Grassmann manifold."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.benchmarks.common import print_rows, run_suite
from geojax.geometry import Grassmann


def run(n: int = 6, rank: int = 2, maxiter: int = 50, seed: int = 1) -> list[dict[str, object]]:
    key = jax.random.key(seed)
    key_A, key_x = jax.random.split(key)
    B = jax.random.normal(key_A, shape=(n, n))
    A = 0.5 * (B + B.T)
    M = Grassmann(size=(n, rank))
    x0 = M.random_point(key_x)

    def cost(X):
        return -jnp.trace(X.T @ A @ X)

    return run_suite(M, cost, x0, maxiter=maxiter)


def main() -> None:
    print_rows(run())


if __name__ == "__main__":
    main()
