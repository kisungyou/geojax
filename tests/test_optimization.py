from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import geojax
import geojax.optimization as optimization
from geojax.geometry import Euclidean, Product, Sphere, Torus
from geojax.optimization._conjugate_gradient import (
    _compute_beta_and_direction,
    _finite_scalar,
    _safe_divide,
)
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


class MinimalEuclidean:
    """Flat test geometry without a specialized Hessian conversion."""

    def tangent_project(self, x, u):
        del x
        return u

    def egrad_to_rgrad(self, x, u):
        del x
        return u


def test_only_class_style_solvers_are_public():
    legacy_names = {
        "steepestdescent",
        "conjugategradient",
        "SteepestDescentOptions",
        "ConjugateGradientOptions",
    }
    assert legacy_names.isdisjoint(geojax.__all__)
    assert legacy_names.isdisjoint(optimization.__all__)
    for name in legacy_names:
        assert not hasattr(geojax, name)
        assert not hasattr(optimization, name)


def test_minimize_rejects_functional_solver_callbacks():
    problem = Minimize(
        M=Euclidean(size=1),
        cost=lambda x: jnp.sum(x**2),
        x0=jnp.ones(1),
        solver=lambda problem, x: (x, 0.0, []),
    )
    with pytest.raises(TypeError, match="class-style solver"):
        problem.solve()


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


def test_minimize_hessian_vector_construction_paths():
    x = jnp.array([1.0, -2.0])
    u = jnp.array([0.25, 0.5])
    manifold = MinimalEuclidean()

    explicit = Minimize(
        M=manifold,
        cost=lambda z: jnp.sum(z**2),
        ehess_vec=lambda z, v: 7.0 * v,
    )
    assert jnp.allclose(explicit.ehess_vec(x, u), 7.0 * u)

    from_egrad = Minimize(
        M=manifold,
        cost=lambda z: jnp.sum(z**2),
        egrad=lambda z: 3.0 * z,
    )
    assert jnp.allclose(from_egrad.ehess_vec(x, u), 3.0 * u)
    assert jnp.allclose(from_egrad.rhess_vec(x, u), 3.0 * u)

    from_rgrad = Minimize(
        M=manifold,
        cost=lambda z: jnp.sum(z**2),
        grad=lambda z: 4.0 * z,
    )
    assert jnp.allclose(from_rgrad.ehess_vec(x, u), 4.0 * u)
    assert jnp.allclose(from_rgrad.rhess_vec(x, u), 4.0 * u)

    from_cost = Minimize(M=manifold, cost=lambda z: 0.5 * jnp.sum(5.0 * z**2))
    assert jnp.allclose(from_cost.ehess_vec(x, u), 5.0 * u)
    assert jnp.allclose(from_cost.rhess_vec(x, u), 5.0 * u)
    assert jnp.allclose(from_cost.hessian_operator(x)(u), 5.0 * u)


@pytest.mark.parametrize(
    "beta_type",
    ["steep", "S-D", "F-R", "P-R", "P-R-SATO", "H-S", "H-S-SATO", "H-Z", "L-S"],
)
def test_conjugate_gradient_beta_rules_are_finite_descent_updates(beta_type):
    manifold = Euclidean(size=2)
    grad = jnp.array([1.0, 2.0])
    newgrad = jnp.array([2.0, 1.0])
    beta, direction = _compute_beta_and_direction(
        M=manifold,
        options=ConjugateGradient(beta_type=beta_type, verbosity=0),
        x=jnp.zeros(2),
        newx=jnp.ones(2),
        grad=grad,
        newgrad=newgrad,
        Pgrad=grad,
        Pnewgrad=newgrad,
        desc_dir=-grad,
        gradPgrad=jnp.dot(grad, grad),
        newgradPnewgrad=jnp.dot(newgrad, newgrad),
        gradnorm=jnp.linalg.norm(grad),
    )
    assert jnp.isfinite(beta)
    assert beta >= 0.0
    assert bool(jnp.all(jnp.isfinite(direction)))


def test_conjugate_gradient_beta_safeguards_and_unknown_rule():
    manifold = Euclidean(size=2)
    grad = jnp.array([1.0, 2.0])
    newgrad = jnp.array([2.0, 1.0])

    def compute(options, new_inner=5.0):
        return _compute_beta_and_direction(
            M=manifold,
            options=options,
            x=jnp.zeros(2),
            newx=jnp.ones(2),
            grad=grad,
            newgrad=newgrad,
            Pgrad=grad,
            Pnewgrad=newgrad,
            desc_dir=-grad,
            gradPgrad=5.0,
            newgradPnewgrad=new_inner,
            gradnorm=jnp.linalg.norm(grad),
        )

    beta, direction = compute(ConjugateGradient(beta_type="F-R", verbosity=0), 0.0)
    assert beta == 0.0
    assert jnp.allclose(direction, -newgrad)

    beta, direction = compute(
        ConjugateGradient(beta_type="F-R", orth_value=0.0, verbosity=0)
    )
    assert beta == 0.0
    assert jnp.allclose(direction, -newgrad)

    with pytest.raises(ValueError, match="Unknown beta_type"):
        compute(ConjugateGradient(beta_type="not-a-rule", verbosity=0))

    assert _finite_scalar(jnp.nan, default=3.0) == 3.0
    assert _safe_divide(1.0, 0.0, default=4.0) == 4.0
    assert _safe_divide(jnp.inf, 2.0, default=5.0) == 2.5


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
