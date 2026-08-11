from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import Euclidean, Product
from geojax.optimization import (
    AdaptiveRegularizationCubics,
    AlternatingGradient,
    ConstantSchedule,
    FiniteSum,
    GaussNewton,
    LeastSquares,
    LevenbergMarquardt,
    Minimize,
    NelderMead,
    NewtonCG,
    StochasticGradient,
    TrustRegions,
)
from geojax.optimization._tangent_cg import tangent_conjugate_gradient
from geojax.optimization.adaptiveregularizationcubics import _solve_cubic_subproblem
from geojax.optimization.minimize import cost_and_grad


@pytest.mark.parametrize(
    "solver",
    [
        NewtonCG(maxiter=5, verbosity=0),
        AdaptiveRegularizationCubics(maxiter=20, verbosity=0),
    ],
)
def test_new_second_order_solvers_converge_on_quadratic(solver):
    manifold = Euclidean(size=2)
    x0 = jnp.array([3.0, -4.0])
    problem = Minimize(
        M=manifold,
        cost=lambda x: 0.5 * jnp.dot(x, x),
        x0=x0,
        solver=solver,
    )

    solution, final_cost, info = problem.solve()
    assert final_cost < 1e-8
    assert jnp.linalg.norm(solution) < 1e-3
    assert info[-1].reason
    if isinstance(solver, NewtonCG):
        assert info[1].extra["cg_iterations"] >= 1
    else:
        assert "sigma" in info[1].extra
        assert info[1].extra["predicted_decrease"] > 0.0


def test_newton_cg_uses_user_hessian_vector_product():
    manifold = Euclidean(size=2)
    calls = []

    def rhess_vec(x, u):
        del x
        calls.append(1)
        return 2.0 * u

    problem = Minimize(
        M=manifold,
        cost=lambda x: jnp.dot(x, x),
        rhess_vec=rhess_vec,
        x0=jnp.array([2.0, -1.0]),
        solver=NewtonCG(maxiter=3, verbosity=0),
    )
    solution, final_cost, _ = problem.solve()
    assert calls
    assert final_cost < 1e-8
    assert jnp.linalg.norm(solution) < 1e-3


def test_least_squares_autodiff_products_are_adjoint():
    manifold = Euclidean(size=2)
    matrix = jnp.array([[2.0, -1.0], [0.5, 3.0], [1.0, 1.0]])
    problem = LeastSquares(M=manifold, residual=lambda x: matrix @ x)
    x = jnp.array([0.2, -0.3])
    u = jnp.array([0.7, 0.4])
    z = jnp.array([-0.5, 0.1, 1.2])
    lhs = jnp.vdot(problem.jacobian_vec(x, u), z)
    rhs = manifold.inner(x, u, problem.adjoint_jacobian(x, z))
    assert jnp.allclose(lhs, rhs, atol=1e-6)


def test_least_squares_products_are_independently_jittable():
    manifold = Euclidean(size=2)
    matrix = jnp.array([[2.0, -1.0], [0.5, 3.0], [1.0, 1.0]])
    problem = LeastSquares(M=manifold, residual=lambda x: matrix @ x)
    x = jnp.array([0.2, -0.3])
    u = jnp.array([0.7, 0.4])
    z = jnp.array([-0.5, 0.1, 1.2])

    jacobian_u = jax.jit(problem.jacobian_vec)(x, u)
    adjoint_z = jax.jit(problem.adjoint_jacobian)(x, z)
    normal_u = jax.jit(problem.normal_operator)(x, u, jnp.asarray(0.25))
    combined_cost, combined_gradient = jax.jit(cost_and_grad, static_argnums=0)(problem, x)

    assert jnp.allclose(jacobian_u, matrix @ u)
    assert jnp.allclose(adjoint_z, matrix.T @ z)
    assert jnp.allclose(normal_u, matrix.T @ matrix @ u + 0.25 * u)
    assert jnp.allclose(combined_cost, 0.5 * jnp.sum((matrix @ x) ** 2))
    assert jnp.allclose(combined_gradient, matrix.T @ matrix @ x)


def test_least_squares_rejects_complex_residuals_before_forming_a_wrong_gradient():
    problem = LeastSquares(
        M=Euclidean(1),
        residual=lambda x: (1.0 + 1.0j) * x,
    )
    point = jnp.array([2.0])

    with pytest.raises(ValueError, match="real-valued"):
        problem.grad(point)
    with pytest.raises(ValueError, match="real-valued"):
        jax.jit(problem.cost)(point)


def test_least_squares_rejects_an_off_tangent_user_adjoint():
    from geojax.geometry import Sphere

    manifold = Sphere(3)
    point = jnp.array([1.0, 0.0, 0.0])
    problem = LeastSquares(
        M=manifold,
        residual=lambda x: x[1:],
        adjoint_jacobian=lambda x, z: jnp.array([1.0, z[0], z[1]]),
    )
    with pytest.raises(ValueError, match="not tangent"):
        problem.adjoint_jacobian(point, jnp.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "solver",
    [
        GaussNewton(maxiter=10, verbosity=0),
        LevenbergMarquardt(maxiter=20, verbosity=0),
    ],
)
def test_least_squares_solvers_fit_nonlinear_residual(solver):
    manifold = Euclidean(size=2)
    target = jnp.array([1.5, -2.0])

    def residual(x):
        return jnp.array([x[0] ** 2 - target[0] ** 2, x[1] - target[1]])

    solution, final_cost, info = LeastSquares(
        M=manifold,
        residual=residual,
        x0=jnp.array([2.0, 0.0]),
        solver=solver,
    ).solve()
    assert final_cost < 1e-8
    assert jnp.allclose(solution, target, atol=1e-3)
    assert info[-1].reason
    assert "residual_norm" in info[-1].extra


def test_levenberg_marquardt_gain_ratio_uses_linearized_residual_model():
    _, _, info = LeastSquares(
        M=Euclidean(1),
        residual=lambda point: point,
        x0=jnp.array([2.0]),
        solver=LevenbergMarquardt(
            initial_damping=1.0,
            min_damping=1e-6,
            maxiter=1,
            verbosity=0,
        ),
    ).solve()
    diagnostics = info[1].extra
    assert diagnostics["predicted_decrease"] == pytest.approx(1.5)
    assert diagnostics["actual_decrease"] == pytest.approx(1.5)
    assert diagnostics["damping_penalty"] == pytest.approx(0.5)
    assert diagnostics["rho"] == pytest.approx(1.0)


def test_tangent_cg_restarts_without_nonpositive_preconditioner():
    manifold = Euclidean(2)
    result = tangent_conjugate_gradient(
        manifold,
        jnp.zeros(2),
        lambda tangent: tangent,
        jnp.array([1.0, -2.0]),
        preconditioner=lambda residual: -residual,
    )
    assert result.converged
    assert result.preconditioner_fallback
    assert not result.negative_curvature
    assert jnp.allclose(result.solution, jnp.array([1.0, -2.0]))


def test_cubic_cauchy_point_avoids_positive_curvature_cancellation():
    manifold = Euclidean(1)
    step, model_value, diagnostics = _solve_cubic_subproblem(
        manifold,
        jnp.zeros(1),
        jnp.array([1e-8]),
        lambda tangent: 1e8 * tangent,
        1.0,
        max_iterations=0,
        tolerance=0.0,
        max_backtracks=1,
    )
    assert float(jnp.linalg.norm(step)) > 0.0
    assert diagnostics["cauchy_length"] == pytest.approx(1e-16, rel=1e-5)
    assert model_value < 0.0


def test_stochastic_gradient_lowers_finite_sum_objective():
    manifold = Euclidean(size=2)
    data = jnp.array([[1.0, 2.0], [1.2, 1.8], [0.8, 2.2], [1.1, 2.1]])
    x0 = jnp.zeros(2)
    problem = FiniteSum(
        M=manifold,
        loss=lambda x, i: 0.5 * jnp.sum((x - data[i]) ** 2),
        num_terms=data.shape[0],
        x0=x0,
        key=0,
        solver=StochasticGradient(
            batch_size=2,
            step_schedule=ConstantSchedule(0.2),
            evaluation_period=5,
            maxiter=60,
            verbosity=0,
        ),
    )
    initial_cost = float(problem.cost(x0))
    solution, final_cost, info = problem.solve()
    assert final_cost < 0.05 * initial_cost
    assert jnp.allclose(solution, jnp.mean(data, axis=0), atol=0.1)
    assert info[-1].extra["full_evaluation"]


def test_finite_sum_validates_batch_indices():
    problem = FiniteSum(
        M=Euclidean(1),
        loss=lambda x, i: 0.5 * (x[0] - i) ** 2,
        num_terms=3,
    )
    with pytest.raises(TypeError, match="integers"):
        problem.batch_cost_and_grad(jnp.zeros(1), jnp.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="num_terms"):
        problem.batch_cost_and_grad(jnp.zeros(1), jnp.array([0, 3]))


def test_finite_sum_batch_kernel_is_independently_jittable():
    data = jnp.array([0.0, 2.0, 4.0])
    problem = FiniteSum(
        M=Euclidean(1),
        loss=lambda x, i: 0.5 * (x[0] - data[i]) ** 2,
        num_terms=data.size,
    )

    cost, gradient = jax.jit(problem.batch_cost_and_grad)(
        jnp.array([1.0]),
        jnp.array([0, 2]),
    )

    assert cost == pytest.approx(2.5)
    assert jnp.allclose(gradient, jnp.array([-1.0]))


def test_stochastic_zero_schedule_returns_full_objective():
    problem = FiniteSum(
        M=Euclidean(1),
        loss=lambda x, i: 0.5 * (x[0] - jnp.asarray(i, dtype=x.dtype)) ** 2,
        num_terms=4,
        x0=jnp.zeros(1),
        key=5,
        solver=StochasticGradient(
            batch_size=1,
            step_schedule=lambda iteration: 0.1 if iteration == 0 else 0.0,
            evaluation_period=100,
            maxiter=10,
            verbosity=0,
        ),
    )
    solution, final_cost, info = problem.solve()
    assert info[-1].extra["full_evaluation"]
    assert info[-1].extra["evaluation_scope"] == "full"
    assert {row.extra["evaluation_scope"] for row in info} <= {"full", "mini_batch"}
    assert final_cost == pytest.approx(float(problem.cost(solution)))


def test_nelder_mead_does_not_converge_on_cost_spread_alone():
    _, _, info = Minimize(
        M=Euclidean(1),
        cost=lambda point: jnp.asarray(1.0),
        x0=jnp.zeros(1),
        key=7,
        solver=NelderMead(
            initial_scale=1.0,
            tolcostspread=0.0,
            tolsimplexdiameter=1e-12,
            maxiter=1,
            verbosity=0,
        ),
    ).solve()

    assert info[0].gradnorm == pytest.approx(0.0)
    assert info[0].stepsize > 1e-12
    assert info[-1].iter == 1
    assert "Maximum iteration count" in info[-1].reason


def test_trust_region_reuses_trial_cost_and_state_after_acceptance():
    evaluations = 0

    def cost(point):
        nonlocal evaluations
        evaluations += 1
        return 0.5 * jnp.sum(point * point)

    Minimize(
        M=Euclidean(1),
        cost=cost,
        grad=lambda point: point,
        rhess_vec=lambda point, tangent: tangent,
        x0=jnp.ones(1),
        solver=TrustRegions(initial_radius=1.0, maxiter=1, verbosity=0),
    ).solve()

    assert evaluations == 2


def test_alternating_gradient_preserves_product_pytree():
    manifold = Product({"left": Euclidean(1), "nested": (Euclidean(1), Euclidean(1))})
    target = {
        "left": jnp.array([1.0]),
        "nested": (jnp.array([-2.0]), jnp.array([0.5])),
    }
    x0 = {
        "left": jnp.array([3.0]),
        "nested": (jnp.array([-4.0]), jnp.array([2.0])),
    }

    def cost(x):
        leaves = manifold._flatten_like(x, "point")
        targets = manifold._flatten_like(target, "target")
        return 0.5 * sum(jnp.sum((a - b) ** 2) for a, b in zip(leaves, targets))

    solution, final_cost, info = Minimize(
        M=manifold,
        cost=cost,
        x0=x0,
        solver=AlternatingGradient(maxiter=20, verbosity=0),
    ).solve()
    assert manifold._treedef == jax.tree_util.tree_structure(solution)
    assert final_cost < 1e-8
    assert info[1].extra["accepted_blocks"] == 3


def test_alternating_gradient_rejects_nonproduct_geometry():
    problem = Minimize(
        M=Euclidean(2),
        cost=lambda x: jnp.dot(x, x),
        x0=jnp.ones(2),
        solver=AlternatingGradient(maxiter=1, verbosity=0),
    )
    with pytest.raises(ValueError, match="Product"):
        problem.solve()


def test_alternating_gradient_does_not_require_product_distance():
    class ProductWithoutDistance(Product):
        def dist(self, x, y):
            raise AssertionError("AlternatingGradient must not call Product.dist")

    manifold = ProductWithoutDistance((Euclidean(1), Euclidean(1)))
    solution, final_cost, info = Minimize(
        M=manifold,
        cost=lambda point: 0.5 * (jnp.sum((point[0] - 1.0) ** 2) + jnp.sum((point[1] + 1.0) ** 2)),
        x0=(jnp.array([2.0]), jnp.array([-3.0])),
        solver=AlternatingGradient(maxiter=10, verbosity=0),
    ).solve()
    # The short run only needs to establish useful progress without a distance call.
    assert final_cost < 1e-5
    assert manifold.belongs(solution)
    assert info[1].stepsize == pytest.approx(
        math.sqrt(sum(step**2 for step in info[1].extra["block_steps"]))
    )
