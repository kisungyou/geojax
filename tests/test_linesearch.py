from __future__ import annotations

import jax
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


def test_strong_wolfe_uses_the_nonlinear_retraction_curve_derivative():
    class NonlinearRetraction(Euclidean):
        def retr(self, x, u, alpha=1.0):
            return x + alpha * u + 0.5 * alpha**2 * u

    manifold = NonlinearRetraction(1)
    point = jnp.ones(1)
    direction = -point
    problem = Minimize(
        M=manifold,
        cost=lambda value: 0.5 * jnp.sum(value * value),
        x0=point,
    )
    strategy = StrongWolfe(normalize_step=False)

    result = strategy.search(problem, point, direction, problem.cost(point), -1.0)
    _, curve_velocity = jax.jvp(
        lambda alpha: manifold.retr(point, direction, alpha),
        (jnp.asarray(result.alpha),),
        (jnp.asarray(1.0),),
    )
    derivative = manifold.inner(result.point, result.gradient, curve_velocity)

    assert result.stats.accepted
    assert result.alpha < 1.0
    assert jnp.abs(derivative) <= strategy.curvature


@pytest.mark.parametrize(
    ("strategy", "direction", "derivative", "reason"),
    [
        (ConstantStep(), jnp.zeros(2), 0.0, "zero or non-finite direction"),
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


def test_constant_step_rejects_invalid_configuration():
    _, problem = quadratic_problem()
    with pytest.raises(ValueError, match="stepsize must be finite and positive"):
        ConstantStep(stepsize=-1.0).search(
            problem,
            problem.x0,
            jnp.array([-1.0, 0.0]),
            problem.cost(problem.x0),
            -1.0,
        )


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


def test_armijo_backtracks_when_a_trial_leaves_the_manifold_domain():
    class LocallyDefinedRetraction(Euclidean):
        def retr(self, x, u, alpha=1.0):
            candidate = x + alpha * u
            return jnp.where(alpha > 0.5, jnp.nan, candidate)

    manifold = LocallyDefinedRetraction(1)
    problem = Minimize(
        M=manifold,
        cost=lambda x: 0.5 * jnp.sum(x * x),
        x0=jnp.array([1.0]),
    )
    result = BacktrackingArmijo(
        initial_stepsize=1.0,
        normalize_step=False,
    ).search(problem, problem.x0, -problem.x0, problem.cost(problem.x0), -1.0)
    assert result.stats.accepted
    assert result.alpha == pytest.approx(0.5)
    assert jnp.allclose(result.point, jnp.array([0.5]))


def test_strong_wolfe_does_not_differentiate_nonfinite_trial_costs():
    manifold = Euclidean(1)

    def cost(point):
        return jnp.where(jnp.abs(point[0]) <= 2.0, 0.5 * point[0] ** 2, jnp.nan)

    def gradient(point):
        if abs(float(point[0])) > 2.0:
            raise AssertionError("gradient evaluated at a nonfinite-cost trial")
        return point

    problem = Minimize(
        M=manifold,
        cost=cost,
        grad=gradient,
        x0=jnp.array([1.0]),
    )
    result = StrongWolfe(
        initial_stepsize=10.0,
        max_stepsize=10.0,
        normalize_step=False,
    ).search(problem, problem.x0, -problem.x0, cost(problem.x0), -1.0)
    assert result.stats.accepted
    assert result.stats.costevals > result.stats.gradevals


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_approximate_wolfe_resolves_rounded_quadratic_without_relaxing_gradient(scale):
    # The quadratic is well resolved in the gradient, but is smaller than one
    # ULP of the additive constant. Strict Armijo cannot certify its decrease.
    dtype = jnp.float32
    baseline = jnp.asarray(scale, dtype=dtype)
    target = jnp.sqrt(jnp.finfo(dtype).eps * baseline) / 8
    point = jnp.zeros(1, dtype=dtype)
    manifold = Euclidean(1)
    problem = Minimize(
        M=manifold,
        cost=lambda value: baseline + jnp.sum((value - target) ** 2),
        x0=point,
    )
    gradient = jax.grad(problem.cost)(point)
    direction = -gradient
    derivative0 = manifold.inner(point, gradient, direction)
    strict = StrongWolfe(initial_stepsize=0.5, normalize_step=False)
    rejected = strict.search(problem, point, direction, problem.cost(point), derivative0)
    assert not rejected.stats.accepted

    strategy = StrongWolfe(initial_stepsize=0.5, normalize_step=False, approximate_wolfe=True)
    result = strategy.search(problem, point, direction, problem.cost(point), derivative0)
    assert result.stats.accepted
    assert "approximate-Wolfe" in result.stats.reason
    assert result.cost == problem.cost(point)
    assert result.alpha == 0.5
    assert jnp.all(result.point == target)
    assert jnp.all(result.gradient == 0.0)
    assert result.gradient is not None


@pytest.mark.parametrize("large_point", [False, True])
def test_approximate_wolfe_rejects_constant_slope_and_unrepresentable_progress(large_point):
    point = jnp.array([1e10 if large_point else 0.0], dtype=jnp.float32)
    direction = jnp.array([-1.0 if large_point else -1e-9], dtype=jnp.float32)
    manifold = Euclidean(1)
    problem = Minimize(M=manifold, cost=lambda value: 1.0 + value[0], x0=point)
    derivative0 = direction[0]
    result = StrongWolfe(
        normalize_step=False,
        approximate_wolfe=True,
        max_steps=4,
        max_zoom_steps=4,
    ).search(problem, point, direction, problem.cost(point), derivative0)
    assert not result.stats.accepted
    assert result.point is point
    assert result.stepsize == 0.0


@pytest.mark.parametrize("baseline", [0.0, 1e-20, 1.0])
def test_approximate_wolfe_rejects_cost_changes_outside_relative_roundoff_budget(baseline):
    dtype = jnp.float32
    base = jnp.asarray(baseline, dtype=dtype)
    target = jnp.asarray(1e-4 if baseline == 0.0 else (baseline * 1e-8) ** 0.5, dtype=dtype)
    point = jnp.zeros(1, dtype=dtype)
    # Model a deliberately inaccurate trial value without changing its local
    # derivative. Even perfect trial stationarity cannot justify an increase
    # outside the stated floating-point budget, including at tiny/zero scale.
    original = base + target**2
    jump = 2 * original if baseline == 0.0 else 32 * jnp.finfo(dtype).eps * original

    def cost(value):
        trial_bias = jax.lax.stop_gradient(jnp.where(value[0] != 0.0, jump, 0.0))
        return base + jnp.sum((value - target) ** 2) + trial_bias

    if baseline == 0.0:
        # Subtract the initial value so the budget is exactly zero.
        uncentered_cost = cost

        def cost(value):
            return uncentered_cost(value) - original

    problem = Minimize(M=Euclidean(1), cost=cost, x0=point)
    gradient = jax.grad(cost)(point)
    result = StrongWolfe(
        initial_stepsize=0.5,
        normalize_step=False,
        approximate_wolfe=True,
        max_zoom_steps=5,
    ).search(problem, point, -gradient, cost(point), -jnp.sum(gradient * gradient))
    assert not result.stats.accepted
    assert result.point is point


def test_approximate_wolfe_rejects_uphill_and_invalid_domain_trials():
    class BoundedLine(Euclidean):
        def belongs(self, point, atol=None):
            return super().belongs(point, atol=atol) & jnp.all(point > -2.0)

    manifold = BoundedLine(1)
    point = jnp.ones(1)
    evaluated = []

    def cost(value):
        assert bool(jnp.all(value > -2.0)), "invalid-domain cost evaluation"
        evaluated.append(float(value[0]))
        return jnp.sum((value - 0.5) ** 2)

    def gradient(value):
        return 2 * (value - 0.5)

    problem = Minimize(M=manifold, cost=cost, grad=gradient, x0=point)
    result = StrongWolfe(initial_stepsize=4.0, normalize_step=False, approximate_wolfe=True).search(
        problem, point, -jnp.ones(1), cost(point), -1.0
    )
    assert result.stats.accepted
    assert result.alpha == 0.5
    assert jnp.allclose(result.point, jnp.array([0.5]))
    assert result.cost < cost(point)
    assert result.stats.reason == ""
    assert -1.0 in evaluated  # A valid, genuinely uphill trial was rejected.
    assert -3.0 not in evaluated  # The out-of-domain trial never reached cost().


@pytest.mark.parametrize(
    "strategy,error",
    [
        (StrongWolfe(approximate_wolfe=1), TypeError),
        (StrongWolfe(approximate_wolfe=True, sufficient_decrease=0.5), ValueError),
        (StrongWolfe(approximate_wolfe=True, roundoff_factor=0.0), ValueError),
        (StrongWolfe(approximate_wolfe=True, roundoff_factor=float("nan")), ValueError),
        (StrongWolfe(approximate_wolfe=True, roundoff_factor=float("inf")), ValueError),
    ],
)
def test_approximate_wolfe_validates_its_opt_in_controls(strategy, error):
    manifold, problem = quadratic_problem()
    point = problem.x0
    with pytest.raises(error):
        strategy.search(
            problem, point, -point, problem.cost(point), -manifold.inner(point, point, point)
        )


def test_strict_wolfe_still_allows_sufficient_decrease_above_one_half():
    manifold, problem = quadratic_problem()
    point = problem.x0
    result = StrongWolfe(sufficient_decrease=0.6, curvature=0.9).search(
        problem, point, -point, problem.cost(point), -manifold.inner(point, point, point)
    )
    assert result.stats.accepted
    assert result.stats.reason == ""


@pytest.mark.parametrize("conversion", ["python", "numpy"])
def test_default_frechet_mean_preserves_python_guarded_retractions(conversion):
    import numpy as np

    from geojax.learning import frechet_mean

    class PythonGuardedRetraction(Euclidean):
        def retr(self, point, tangent, alpha=1.0):
            multiplier = float(np.asarray(alpha)) if conversion == "numpy" else float(alpha)
            if multiplier > 1.0:
                raise ValueError("step exceeds the custom retraction domain")
            return point + multiplier * tangent

    manifold = PythonGuardedRetraction(1)
    data = jnp.array([[0.0], [1.0], [2.0]])
    result = frechet_mean(manifold, data, initial_point=jnp.zeros(1))
    assert result.converged
    assert jnp.allclose(result.point, jnp.ones(1))
    assert result.gradient_norm <= result.diagnostics["effective_tolerance"]
    assert result.diagnostics["history"][1].linesearch.method == "adaptive_armijo"


def test_default_frechet_mean_does_not_hide_unrelated_retraction_errors():
    from geojax.learning import frechet_mean

    class BrokenRetraction(Euclidean):
        def retr(self, point, tangent, alpha=1.0):
            raise TypeError("unrelated user geometry defect")

    with pytest.raises(TypeError, match="unrelated user geometry defect"):
        frechet_mean(BrokenRetraction(1), jnp.array([[0.0], [1.0]]), initial_point=jnp.zeros(1))
