"""Benchmark a small product-manifold objective."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.benchmarks.common import print_rows, run_suite
from geojax.geometry import Product, Sphere, Torus


def run(maxiter: int = 50, seed: int = 4) -> list[dict[str, object]]:
    key = jax.random.key(seed)
    key_x, key_t1, key_t2 = jax.random.split(key, 3)
    M = Product({"direction": Sphere(size=3), "phase": Torus(size=2)})
    x0 = M.random_point(key_x)
    target = {
        "direction": M.factors["direction"].random_point(key_t1),
        "phase": M.factors["phase"].random_point(key_t2),
    }

    def cost(x):
        return 0.5 * M.dist(x, target) ** 2 + 0.1 * jnp.sum(jnp.sin(x["phase"]))

    return run_suite(M, cost, x0, maxiter=maxiter)


def main() -> None:
    print_rows(run())


if __name__ == "__main__":
    main()
