from __future__ import annotations

import jax.numpy as jnp

from geojax.geometry import Euclidean
from geojax.optimization import (
    AdaptiveArmijo,
    BacktrackingArmijo,
    ConstantStep,
    LineSearchProtocol,
    Minimize,
    StrongWolfe,
)


def quadratic_problem():
    manifold = Euclidean(size=2)
    problem = Minimize(
        M=manifold,
        cost=lambda x: 0.5 * jnp.dot(x, x),
        x0=jnp.array([3.0, -4.0]),
    )
    return manifold, problem


def test_line_search_protocol_and_fixed_step():
    manifold, problem = quadratic_problem()
    x = problem.x0
    gradient = x
    direction = -gradient
    cost = problem.cost(x)
    derivative = manifold.inner(x, gradient, direction)

    strategy = ConstantStep(stepsize=0.1, normalize_step=False)
    assert isinstance(strategy, LineSearchProtocol)
    result = strategy.search(problem, x, direction, cost, derivative)
    assert result.stats.accepted
    assert result.stats.method == "constant"
    assert jnp.allclose(result.point, 0.9 * x)


def test_backtracking_armijo_satisfies_sufficient_decrease():
    manifold, problem = quadratic_problem()
    x = problem.x0
    gradient = x
    direction = -gradient
    cost = problem.cost(x)
    derivative = manifold.inner(x, gradient, direction)
    strategy = BacktrackingArmijo(initial_stepsize=10.0, normalize_step=False)

    result = strategy.search(problem, x, direction, cost, derivative)
    rhs = cost + strategy.sufficient_decrease * result.alpha * derivative
    assert result.stats.accepted
    assert result.stats.costevals > 1
    assert result.cost <= rhs


def test_adaptive_armijo_carries_state_between_searches():
    manifold, problem = quadratic_problem()
    strategy = AdaptiveArmijo()
    x = problem.x0
    gradient = x
    direction = -gradient
    first = strategy.search(
        problem,
        x,
        direction,
        problem.cost(x),
        manifold.inner(x, gradient, direction),
    )
    new_gradient = first.point
    second = strategy.search(
        problem,
        first.point,
        -new_gradient,
        first.cost,
        manifold.inner(first.point, new_gradient, -new_gradient),
        state=first.state,
    )
    assert first.stats.accepted
    assert second.stats.accepted
    assert second.cost < first.cost


def test_strong_wolfe_conditions_hold_on_quadratic():
    manifold, problem = quadratic_problem()
    x = problem.x0
    gradient = x
    direction = -gradient
    cost = problem.cost(x)
    derivative0 = manifold.inner(x, gradient, direction)
    strategy = StrongWolfe()

    result = strategy.search(problem, x, direction, cost, derivative0)
    derivative = manifold.inner(result.point, result.gradient, direction)
    assert result.stats.accepted
    assert result.cost <= cost + strategy.sufficient_decrease * result.alpha * derivative0
    assert jnp.abs(derivative) <= strategy.curvature * jnp.abs(derivative0)
