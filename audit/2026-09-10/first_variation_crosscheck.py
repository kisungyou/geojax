"""Check first variation of squared distance at regular, noncoincident pairs."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from geojax import geometry as g

jax.config.update("jax_enable_x64", True)
geometries = [
    g.Euclidean(3),
    g.Sphere(4),
    g.Torus(3),
    g.Hyperboloid(4),
    g.PoincareBall(3),
    g.ProbabilitySimplex(4),
    g.Oblique((4, 2)),
    g.Grassmann((5, 2)),
    g.GrassmannProjection((5, 2)),
    g.GeneralizedGrassmann((5, 2), metric=jnp.diag(jnp.arange(1.0, 6.0))),
    g.SPDLogEuclidean((3, 3)),
    g.SPDAffineInvariant((3, 3)),
    g.SPDBuresWasserstein((3, 3)),
    g.CorrelationECM((3, 3)),
    g.CorrelationLEC((3, 3)),
    g.SpecialOrthogonal(3),
    g.SpecialEuclidean(3),
    g.KendallShape((5, 2)),
]
results = []
for index, manifold in enumerate(geometries):
    keys = jax.random.split(jax.random.key(260910 + index), 3)
    try:
        point = manifold.random_point(keys[0])
        u = manifold.random_tangent(keys[1], point, normalize=True)
        v = manifold.random_tangent(keys[2], point, normalize=True)
        target = manifold.exp(point, 0.1 * v)
        expected = -2 * manifold.inner(point, manifold.log(point, target), u)
        actual = jax.grad(lambda t: manifold.squared_dist(manifold.exp(point, t * u), target))(0.0)
        ok = bool(jnp.isfinite(actual) and np.allclose(actual, expected, rtol=2e-5, atol=2e-7))
        results.append(
            {
                "geometry": type(manifold).__name__,
                "actual": float(actual),
                "expected": float(expected),
                "passed": ok,
            }
        )
    except Exception as exc:
        results.append(
            {
                "geometry": type(manifold).__name__,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
print(json.dumps(results, indent=2))
raise SystemExit(0 if all(item["passed"] for item in results) else 1)
