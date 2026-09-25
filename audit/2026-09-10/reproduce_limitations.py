"""Recheck the initial audit findings; run with PYTHONPATH=.

The analytic inputs and expected results are preserved from the initial audit.
It prints JSON; a nonzero exit indicates a reproduced correctness failure.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from geojax import geometry as geo
from geojax.learning import frechet_mean

jax.config.update("jax_enable_x64", True)
results = []


def json_array(value):
    value = np.asarray(value)
    if value.ndim:
        return [json_array(part) for part in value]
    scalar = value.item()
    return scalar if np.isfinite(scalar) else str(scalar)


def check(name, actual, expected):
    try:
        value = np.asarray(actual())
        reference = np.asarray(expected)
        passed = bool(
            np.all(np.isfinite(value)) and np.allclose(value, reference, rtol=1e-6, atol=1e-8)
        )
        results.append(
            {
                "name": name,
                "passed": passed,
                "actual": json_array(value),
                "expected": reference.tolist(),
            }
        )
    except Exception as exc:
        results.append(
            {
                "name": name,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "expected": np.asarray(expected).tolist(),
            }
        )


identity = jnp.eye(2)
direction = jnp.array([[0.3, 0.2], [0.2, -0.1]])
for geometry in (geo.SPDLogEuclidean, geo.SPDAffineInvariant, geo.SPDBuresWasserstein):
    manifold = geometry((2, 2))
    coefficient = 1 / np.sqrt(2) if geometry is geo.SPDBuresWasserstein else 2 * (1 + np.log(2))
    check(
        f"{geometry.__name__}: squared-distance Hessian at identity",
        lambda m=manifold: jax.jvp(
            jax.grad(lambda p: m.squared_dist(p, 2 * identity)), (identity,), (direction,)
        )[1],
        coefficient * direction,
    )

manifold = geo.RankKPSDBuresWasserstein((3, 3), rank=1)
point = jnp.diag(jnp.array([1.0, 0.0, 0.0]))
target = 2 * point
check(
    "RankKPSDBuresWasserstein: scalar distance gradient on fixed support",
    lambda: jax.grad(lambda s: manifold.squared_dist(s * point, target))(1.0),
    1 - np.sqrt(2),
)
check(
    "RankKPSDBuresWasserstein: Frechet mean of two commuting rank-one matrices",
    lambda: frechet_mean(manifold, jnp.stack([point, target])).point,
    ((1 + np.sqrt(2)) / 2) ** 2 * point,
)

rotation_manifold = geo.SpecialOrthogonal(2)
rotation_tangent = jnp.array([[0.0, -1.0], [1.0, 0.0]])
check(
    "SpecialOrthogonal: squared-distance second derivative away from cut locus",
    lambda: jax.grad(
        jax.grad(
            lambda t: rotation_manifold.squared_dist(
                identity, rotation_manifold.exp(identity, t * rotation_tangent)
            )
        )
    )(0.2),
    4.0,
)

x = jnp.eye(4)[:, :2]
z = jnp.eye(4)[:, 2:]
grassmann = geo.Grassmann((4, 2))
check(
    "Grassmann: derivative of log at equal nonzero principal angles",
    lambda: jax.jacfwd(lambda t: grassmann.log(x, jnp.cos(t) * x + jnp.sin(t) * z))(0.2),
    z,
)

shape = geo.KendallShape((4, 2))
x_shape = 0.5 * jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
u_shape = 0.5 * jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0]])
# The two matrices are centered, horizontal, orthogonal, and unit norm.
check(
    "KendallShape: squared-distance derivative from balanced regular shape",
    lambda: jax.grad(
        lambda t: shape.squared_dist(x_shape, jnp.cos(t) * x_shape + jnp.sin(t) * u_shape)
    )(0.1),
    0.2,
)
check(
    "KendallShape: Frechet mean of two nearby regular shapes",
    lambda: (
        frechet_mean(
            shape, jnp.stack([x_shape, jnp.cos(0.1) * x_shape + jnp.sin(0.1) * u_shape])
        ).point
    ),
    jnp.cos(0.05) * x_shape + jnp.sin(0.05) * u_shape,
)

rank_one = geo.RankKPSDBuresWasserstein((2, 2), rank=1)
p_rank_one = jnp.diag(jnp.array([1.0, 0.0]))
u_rank_one = jnp.array([[-4.0, 1.0], [1.0, 0.0]])
check(
    "RankKPSDBuresWasserstein: regular geodesic beyond positive-overlap certificate",
    lambda: rank_one.exp(p_rank_one, u_rank_one),
    jnp.array([[1.0, -1.0], [-1.0, 1.0]]),
)

largest = jnp.finfo(jnp.float64).max
check(
    "Euclidean: distance at largest representable finite coordinate",
    lambda: geo.Euclidean(1).dist(jnp.array([0.0]), jnp.array([largest])),
    largest,
)

print(
    json.dumps(
        {"jax": jax.__version__, "x64": jax.config.jax_enable_x64, "results": results}, indent=2
    )
)
raise SystemExit(0 if all(item["passed"] for item in results) else 1)
