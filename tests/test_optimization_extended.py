from __future__ import annotations

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
    NewtonCG,
    StochasticGradient,
)


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
