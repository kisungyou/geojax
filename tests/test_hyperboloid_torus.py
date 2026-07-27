from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.geometry import Hyperboloid, PoincareBall, Torus


def test_hyperboloid_lorentz_model():
    M = Hyperboloid(size=3)
    x = M.random_point(jax.random.key(0))
    u = M.random_tangent(jax.random.key(1), x, scale=0.1)
    y = M.exp(x, u)
    assert bool(M.belongs(y))
    assert bool(M.is_tangent(x, u))
    assert jnp.allclose(M.lorentz_inner(y, y), -1.0, atol=1e-8)
    assert jnp.allclose(M.dist(x, y), M.norm(x, u), atol=1e-5)


def test_torus_wrap_and_short_log():
    M = Torus(size=2)
    x = jnp.array([jnp.pi - 0.1, 0.0])
    y = jnp.array([-jnp.pi + 0.1, 0.2])
    v = M.log(x, y)
    assert jnp.allclose(v, jnp.array([0.2, 0.2]), atol=1e-8)
    assert bool(M.belongs(M.exp(x, v)))


def test_open_ball_repairs_keep_a_float32_interior_margin():
    ball = PoincareBall(size=2)
    outside = jnp.array([2.0, 0.0], dtype=jnp.float32)
    projected = ball.project(outside)

    assert bool(ball.belongs(projected))
    assert jnp.linalg.norm(projected) < 1.0
    assert jnp.isfinite(ball.conformal_factor(projected))
    assert ball.conformal_factor(projected) < 1e7

    hyperboloid = Hyperboloid(size=3)
    lifted = hyperboloid.from_poincare(outside)
    recovered = hyperboloid.to_poincare(lifted)
    assert bool(hyperboloid.belongs(lifted))
    assert jnp.all(jnp.isfinite(lifted))
    assert jnp.linalg.norm(recovered) < 1.0
