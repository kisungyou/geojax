"""Benchmark correlation-matrix Fréchet mean optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.benchmarks.common import print_rows, run_suite
from geojax.geometry import CorrelationECM


def run(n: int = 3, samples: int = 8, maxiter: int = 50, seed: int = 3) -> list[dict[str, object]]:
    key = jax.random.key(seed)
    key_data, key_x = jax.random.split(key)
    M = CorrelationECM(size=(n, n))
    data = M.random_point(key_data, sample_shape=(samples,))
    x0 = M.random_point(key_x)

    def cost(C):
        d = jax.vmap(lambda D: M.dist(C, D))(data)
        return 0.5 * jnp.mean(d * d)

    return run_suite(M, cost, x0, maxiter=maxiter)


def main() -> None:
    print_rows(run())


if __name__ == "__main__":
    main()
