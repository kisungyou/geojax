from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import geojax
import geojax.optimization as optimization
from geojax.geometry import Euclidean, Grassmann, Product, RankKPSD, Sphere, Torus
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
from geojax.optimization.lbfgs import _transport_memory
from geojax.optimization.neldermead import _diameter
from geojax.optimization.trustregions import _truncated_cg


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


def test_sphere_hessian_conversion_includes_embedding_curvature():
    M = Sphere(size=3)
    x = M.project(jnp.array([1.0, 2.0, -1.0]))
    u = M.tangent_project(x, jnp.array([0.3, -0.2, 0.4]))
    coefficient = jnp.array([0.5, -1.0, 0.25])
    problem = Minimize(M=M, cost=lambda point: jnp.dot(coefficient, point))

    expected = -jnp.dot(coefficient, x) * u
    assert jnp.allclose(problem.rhess_vec(x, u), expected, atol=2e-6, rtol=2e-6)


def test_automatic_hessian_rejects_geometry_without_exact_conversion():
    M = Grassmann(size=(4, 2))
    x = M.random_point(jax.random.key(90))
    u = M.random_tangent(jax.random.key(91), x, scale=0.1)
    problem = Minimize(M=M, cost=lambda point: jnp.sum(point * point))

    with pytest.raises(ValueError, match="Supply rhess_vec explicitly"):
        problem.rhess_vec(x, u)


def test_minimize_hessian_vector_construction_paths():
    x = jnp.array([1.0, -2.0])
    u = jnp.array([0.25, 0.5])
    manifold = Euclidean(size=2)

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

    beta, direction = compute(ConjugateGradient(beta_type="F-R", orth_value=0.0, verbosity=0))
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


def test_lbfgs_recomputes_curvature_after_nonisometric_transport():
    class ScaledTransportEuclidean(Euclidean):
        transport_is_isometric = False
        transport_is_parallel = False

        def transport(self, x, y, u):
            del x, y
            return 2.0 * u

    M = ScaledTransportEuclidean(1)
    s = jnp.array([2.0])
    y = jnp.array([3.0])
    memory = [(s, y, 1.0 / 6.0)]
    transported = _transport_memory(
        M,
        jnp.array([0.0]),
        jnp.array([1.0]),
        memory,
        cautious_update=False,
        cautious_threshold=0.0,
    )

    new_s, new_y, new_rho = transported[0]
    assert jnp.allclose(new_s, 2.0 * s)
    assert jnp.allclose(new_y, 2.0 * y)
    assert jnp.allclose(new_rho, 1.0 / jnp.vdot(new_s, new_y))
    assert not jnp.allclose(new_rho, memory[0][2])


def test_lbfgs_rejects_nonpositive_memory():
    problem = Minimize(
        M=Euclidean(1),
        cost=lambda point: jnp.sum(point**2),
        x0=jnp.ones(1),
        solver=LBFGS(memory=0, verbosity=0),
    )
    with pytest.raises(ValueError, match="memory must be positive"):
        problem.solve()


def test_truncated_cg_reports_the_final_residual():
    manifold = Euclidean(2)
    problem = Minimize(M=manifold, cost=lambda point: 0.5 * jnp.sum(point**2))
    eta, hit_boundary, diagnostics = _truncated_cg(
        problem,
        jnp.zeros(2),
        jnp.array([1.0, 0.0]),
        10.0,
        TrustRegions(maxinner=5, verbosity=0),
    )
    assert not hit_boundary
    assert jnp.allclose(eta, jnp.array([-1.0, 0.0]))
    assert diagnostics["tcg_residual_norm"] == pytest.approx(0.0)


def test_lbfgs_smoke_with_nonisometric_rank_stratum_transport():
    M = RankKPSD(size=(3, 3), rank=2)
    target = M.random_point(jax.random.key(120))
    x0 = M.random_point(jax.random.key(121))

    solution, final_cost, info = Minimize(
        M=M,
        cost=lambda point: 0.5 * jnp.sum((point - target) ** 2),
        x0=x0,
        solver=LBFGS(maxiter=4, verbosity=0),
    ).solve()

    assert bool(M.belongs(solution))
    assert jnp.isfinite(final_cost)
    assert all(jnp.isfinite(entry.cost) for entry in info)


def test_nelder_mead_initial_scale_is_applied_once():
    scale = 0.2
    solution, _, _ = Minimize(
        M=Euclidean(1),
        cost=lambda x: -jnp.sum(x * x),
        x0=jnp.zeros(1),
        key=123,
        solver=NelderMead(
            initial_scale=scale,
            maxiter=0,
            tolcostspread=-1.0,
            verbosity=0,
        ),
    ).solve()
    assert jnp.allclose(jnp.linalg.norm(solution), scale, atol=1e-6)


def test_nelder_mead_diameter_uses_every_vertex_pair():
    points = [jnp.array([0.0]), jnp.array([1.0]), jnp.array([-1.0])]
    assert _diameter(Euclidean(1), points) == pytest.approx(2.0)
