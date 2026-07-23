from __future__ import annotations

import jax.numpy as jnp
import pytest

from geojax.geometry import Euclidean
from geojax.optimization import (
    AdaptiveArmijo,
    BacktrackingArmijo,
    ConstantStep,
    LineSearchProtocol,
    Minimize,
    StrongWolfe,
)
from geojax.optimization.linesearch import LineSearchState


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


@pytest.mark.parametrize(
    ("strategy", "direction", "derivative", "reason"),
    [
        (ConstantStep(), jnp.zeros(2), 0.0, "zero or non-finite direction"),
        (
            ConstantStep(stepsize=-1.0),
            jnp.array([-1.0, 0.0]),
            -1.0,
            "non-positive trial multiplier",
        ),
        (BacktrackingArmijo(), jnp.zeros(2), 0.0, "zero or non-finite direction"),
        (
            BacktrackingArmijo(),
            jnp.array([1.0, 0.0]),
            1.0,
            "direction is not descending",
        ),
        (StrongWolfe(), jnp.zeros(2), 0.0, "zero or non-finite direction"),
        (
            StrongWolfe(),
            jnp.array([1.0, 0.0]),
            1.0,
            "direction is not descending",
        ),
    ],
)
def test_line_search_rejects_invalid_directions(strategy, direction, derivative, reason):
    _, problem = quadratic_problem()
    result = strategy.search(
        problem,
        problem.x0,
        direction,
        problem.cost(problem.x0),
        derivative,
    )
    assert not result.stats.accepted
    assert result.stats.reason == reason
    assert result.stepsize == 0.0
    assert result.point is problem.x0


@pytest.mark.parametrize(
    "strategy",
    [
        BacktrackingArmijo(contraction_factor=1.0),
        BacktrackingArmijo(sufficient_decrease=1.0),
        BacktrackingArmijo(max_steps=0),
    ],
)
def test_armijo_validates_parameters(strategy):
    manifold, problem = quadratic_problem()
    direction = -problem.x0
    with pytest.raises(ValueError):
        strategy.search(
            problem,
            problem.x0,
            direction,
            problem.cost(problem.x0),
            manifold.inner(problem.x0, problem.x0, direction),
        )


@pytest.mark.parametrize(
    "strategy",
    [
        StrongWolfe(sufficient_decrease=0.9, curvature=0.1),
        StrongWolfe(expansion=1.0),
        StrongWolfe(max_steps=0),
        StrongWolfe(max_zoom_steps=0),
    ],
)
def test_strong_wolfe_validates_parameters(strategy):
    manifold, problem = quadratic_problem()
    direction = -problem.x0
    with pytest.raises(ValueError):
        strategy.search(
            problem,
            problem.x0,
            direction,
            problem.cost(problem.x0),
            manifold.inner(problem.x0, problem.x0, direction),
        )


def test_armijo_failure_and_adaptive_state_fallbacks():
    manifold, problem = quadratic_problem()
    direction = -problem.x0
    derivative = manifold.inner(problem.x0, problem.x0, direction)

    failed = BacktrackingArmijo(
        sufficient_decrease=0.9,
        initial_stepsize=10.0,
        contraction_factor=0.9,
        max_steps=1,
        normalize_step=False,
    ).search(problem, problem.x0, direction, problem.cost(problem.x0), derivative)
    assert not failed.stats.accepted
    assert failed.stats.costevals == 1
    assert failed.stats.reason == "Armijo condition was not satisfied"

    adaptive = AdaptiveArmijo(initial_stepsize=0.25, normalize_step=False)
    invalid_state = LineSearchState(previous_cost=0.0, previous_alpha=0.1)
    result = adaptive.search(
        problem,
        problem.x0,
        direction,
        problem.cost(problem.x0),
        derivative,
        state=invalid_state,
    )
    assert result.stats.accepted
    assert result.alpha == pytest.approx(0.25)


def test_strong_wolfe_reports_unsatisfied_conditions():
    manifold, problem = quadratic_problem()
    direction = -problem.x0
    derivative = manifold.inner(problem.x0, problem.x0, direction)
    result = StrongWolfe(
        initial_stepsize=1e-4,
        max_stepsize=1e-4,
        max_steps=1,
        max_zoom_steps=1,
        normalize_step=False,
    ).search(problem, problem.x0, direction, problem.cost(problem.x0), derivative)
    assert not result.stats.accepted
    assert result.stats.costevals == 1
    assert result.stats.gradevals == 1
    assert result.stats.reason == "strong-Wolfe conditions were not satisfied"
