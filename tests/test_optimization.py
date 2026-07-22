from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import Euclidean, Product, Sphere, Torus
from geojax.optimization import (
    BarzilaiBorwein,
    ConjugateGradient,
    LBFGS,
    Minimize,
    NelderMead,
    ParticleSwarm,
    SteepestDescent,
    TrustRegions,
)


def sphere_problem(solver):
    M = Sphere(size=3)
    target = jnp.array([1.0, 0.0, 0.0])
    x0 = M.project(jnp.array([0.2, 1.0, 0.1]))

    def cost(x):
        return 0.5 * M.dist(x, target) ** 2

    return Minimize(M=M, cost=cost, x0=x0, solver=solver)


@pytest.mark.parametrize(
    "solver",
    [
        SteepestDescent(maxiter=5, verbosity=0),
        ConjugateGradient(maxiter=5, verbosity=0),
        TrustRegions(maxiter=5, verbosity=0),
        BarzilaiBorwein(maxiter=5, verbosity=0),
        LBFGS(maxiter=5, verbosity=0),
        ParticleSwarm(maxiter=3, swarm_size=4, verbosity=0),
        NelderMead(maxiter=3, verbosity=0),
    ],
)
def test_public_solvers_smoke(solver):
    sol, final_cost, info = sphere_problem(solver).solve()
    assert bool(jnp.all(jnp.isfinite(sol)))
    assert jnp.isfinite(final_cost)
    assert len(info) >= 1
    assert info[-1].reason


def test_minimize_hessian_vector_fallback_and_user_rhess():
    M = Euclidean(size=2)
    problem = Minimize(
        M=M, cost=lambda x: jnp.sum(x * x), x0=jnp.array([1.0, 2.0]), solver=SteepestDescent()
    )
    u = jnp.array([0.5, -0.25])
    assert jnp.allclose(problem.rhess_vec(problem.x0, u), 2.0 * u)

    custom = Minimize(
        M=M,
        cost=lambda x: jnp.sum(x * x),
        x0=jnp.array([1.0, 2.0]),
        solver=SteepestDescent(),
        rhess_vec=lambda x, v: 3.0 * v,
    )
    assert jnp.allclose(custom.rhess_vec(custom.x0, u), 3.0 * u)


def test_conjugate_gradient_product_pytree_smoke():
    M = Product({"direction": Sphere(size=3), "phase": Torus(size=2)})
    target = M.random_point(jax.random.key(0))
    x0 = M.random_point(jax.random.key(1))

    def cost(x):
        return 0.5 * M.dist(x, target) ** 2

    sol, final_cost, info = Minimize(
        M=M,
        cost=cost,
        x0=x0,
        solver=ConjugateGradient(maxiter=3, verbosity=0),
    ).solve()
    assert bool(M.belongs(sol))
    assert jnp.isfinite(final_cost)
    assert len(info) >= 1
