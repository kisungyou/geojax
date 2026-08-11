"""Reusable line-search strategies for Riemannian optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
import math

import jax
import jax.numpy as jnp

from geojax.geometry.base import validate_integer, validate_positive

from .minimize import (
    Array,
    LineSearchStats,
    as_float,
    cost_value,
    gradient_value,
    inner,
    require,
    retract,
    validate_point,
    validate_tangent,
)


def _trial_point(M: Any, x: Any, direction: Any, alpha: float) -> Any | None:
    """Return a valid trial point, or ``None`` when the step leaves the domain."""
    try:
        return retract(M, x, direction, alpha)
    except (FloatingPointError, ValueError):
        return None


def _trial_curve(
    M: Any,
    x: Any,
    direction: Any,
    alpha: float,
) -> tuple[Any, Any] | None:
    """Return a trial point and the exact retraction-curve velocity."""
    alpha_array = jnp.asarray(alpha)
    try:
        point, velocity = jax.jvp(
            lambda multiplier: retract(M, x, direction, multiplier),
            (alpha_array,),
            (jnp.ones_like(alpha_array),),
        )
    except jax.errors.ConcretizationTypeError as exc:
        raise TypeError(
            "StrongWolfe requires a JAX-differentiable retraction in its scalar multiplier."
        ) from exc
    try:
        validate_point(M, point, name="Strong-Wolfe trial point")
        validate_tangent(M, point, velocity, name="Strong-Wolfe retraction-curve velocity")
    except (FloatingPointError, ValueError):
        return None
    return point, velocity


@dataclass(frozen=True)
class LineSearchState:
    """Information carried between consecutive line searches."""

    previous_cost: float | None = None
    previous_directional_derivative: float | None = None
    previous_alpha: float | None = None
    previous_stepsize: float | None = None


@dataclass(frozen=True)
class LineSearchResult:
    """Point, values, diagnostics, and reusable state from a line search."""

    point: Any
    cost: Array
    gradient: Any | None
    stepsize: float
    alpha: float
    stats: LineSearchStats
    state: LineSearchState


@runtime_checkable
class LineSearchProtocol(Protocol):
    """Protocol implemented by all public line-search strategies."""

    def search(
        self,
        problem: Any,
        x: Any,
        direction: Any,
        cost: Array,
        directional_derivative: Array,
        *,
        state: LineSearchState | None = None,
        initial_alpha: float | None = None,
    ) -> LineSearchResult: ...


def _state(
    cost: float,
    directional_derivative: float,
    alpha: float,
    stepsize: float,
) -> LineSearchState:
    return LineSearchState(
        previous_cost=cost,
        previous_directional_derivative=directional_derivative,
        previous_alpha=alpha,
        previous_stepsize=stepsize,
    )


def _failure(
    method: str,
    x: Any,
    cost: Array,
    cost0: float,
    derivative0: float,
    reason: str,
    *,
    costevals: int = 0,
    gradevals: int = 0,
) -> LineSearchResult:
    stats = LineSearchStats(
        costevals=costevals,
        gradevals=gradevals,
        stepsize=0.0,
        alpha=0.0,
        accepted=False,
        method=method,
        reason=reason,
    )
    return LineSearchResult(
        point=x,
        cost=cost,
        gradient=None,
        stepsize=0.0,
        alpha=0.0,
        stats=stats,
        state=_state(cost0, derivative0, 0.0, 0.0),
    )


@dataclass(frozen=True)
class ConstantStep:
    """Take a fixed multiplier or fixed Riemannian-length step."""

    stepsize: float = 1.0
    normalize_step: bool = False

    def search(
        self,
        problem: Any,
        x: Any,
        direction: Any,
        cost: Array,
        directional_derivative: Array,
        *,
        state: LineSearchState | None = None,
        initial_alpha: float | None = None,
    ) -> LineSearchResult:
        del state
        M = require(problem, "M")
        if not isinstance(self.normalize_step, bool):
            raise TypeError("normalize_step must be a boolean.")
        configured_stepsize = validate_positive(self.stepsize, name="stepsize")
        norm_d = as_float(M.norm(x, direction))
        f0 = as_float(cost)
        df0 = as_float(directional_derivative)
        if not math.isfinite(norm_d) or norm_d <= 0.0:
            return _failure("constant", x, cost, f0, df0, "zero or non-finite direction")
        alpha = float(configured_stepsize if initial_alpha is None else initial_alpha)
        if self.normalize_step:
            alpha /= norm_d
        if not math.isfinite(alpha) or alpha <= 0.0:
            return _failure("constant", x, cost, f0, df0, "non-positive trial multiplier")
        newx = _trial_point(M, x, direction, alpha)
        if newx is None:
            return _failure(
                "constant",
                x,
                cost,
                f0,
                df0,
                "trial point left the manifold domain",
            )
        newcost = cost_value(problem, newx)
        accepted = math.isfinite(as_float(newcost))
        if not accepted:
            return _failure("constant", x, cost, f0, df0, "non-finite trial cost", costevals=1)
        step = alpha * norm_d
        stats = LineSearchStats(
            costevals=1,
            stepsize=step,
            alpha=alpha,
            accepted=True,
            method="constant",
        )
        return LineSearchResult(
            point=newx,
            cost=newcost,
            gradient=None,
            stepsize=step,
            alpha=alpha,
            stats=stats,
            state=_state(f0, df0, alpha, step),
        )


@dataclass(frozen=True)
class BacktrackingArmijo:
    """Monotone Armijo search along a retraction curve."""

    contraction_factor: float = 0.5
    sufficient_decrease: float = 1e-4
    max_steps: int = 25
    initial_stepsize: float = 1.0
    normalize_step: bool = True

    @property
    def method_name(self) -> str:
        return "armijo"

    def _initial_alpha(
        self,
        norm_d: float,
        cost0: float,
        derivative0: float,
        state: LineSearchState | None,
        initial_alpha: float | None,
    ) -> float:
        del cost0, derivative0, state
        if initial_alpha is not None:
            return float(initial_alpha)
        alpha = float(self.initial_stepsize)
        return alpha / norm_d if self.normalize_step else alpha

    def search(
        self,
        problem: Any,
        x: Any,
        direction: Any,
        cost: Array,
        directional_derivative: Array,
        *,
        state: LineSearchState | None = None,
        initial_alpha: float | None = None,
    ) -> LineSearchResult:
        M = require(problem, "M")
        norm_d = as_float(M.norm(x, direction))
        f0 = as_float(cost)
        df0 = as_float(directional_derivative)
        if not math.isfinite(norm_d) or norm_d <= 0.0:
            return _failure(self.method_name, x, cost, f0, df0, "zero or non-finite direction")
        if not math.isfinite(df0) or df0 >= 0.0:
            return _failure(self.method_name, x, cost, f0, df0, "direction is not descending")
        if not 0.0 < self.contraction_factor < 1.0:
            raise ValueError("contraction_factor must lie in (0, 1).")
        if not 0.0 < self.sufficient_decrease < 1.0:
            raise ValueError("sufficient_decrease must lie in (0, 1).")
        validate_positive(self.initial_stepsize, name="initial_stepsize")
        max_steps = validate_integer(self.max_steps, name="max_steps")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if not isinstance(self.normalize_step, bool):
            raise TypeError("normalize_step must be a boolean.")

        alpha = self._initial_alpha(norm_d, f0, df0, state, initial_alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            return _failure(self.method_name, x, cost, f0, df0, "non-positive trial multiplier")

        costevals = 0
        newx = x
        newcost = cost
        accepted = False
        for _ in range(max_steps):
            candidate = _trial_point(M, x, direction, alpha)
            if candidate is None:
                alpha *= float(self.contraction_factor)
                continue
            newx = candidate
            newcost = cost_value(problem, newx)
            costevals += 1
            trial = as_float(newcost)
            if math.isfinite(trial) and trial <= f0 + self.sufficient_decrease * alpha * df0:
                accepted = True
                break
            alpha *= float(self.contraction_factor)

        if not accepted:
            return _failure(
                self.method_name,
                x,
                cost,
                f0,
                df0,
                "Armijo condition was not satisfied",
                costevals=costevals,
            )

        step = alpha * norm_d
        stats = LineSearchStats(
            costevals=costevals,
            stepsize=step,
            alpha=alpha,
            accepted=True,
            method=self.method_name,
        )
        return LineSearchResult(
            point=newx,
            cost=newcost,
            gradient=None,
            stepsize=step,
            alpha=alpha,
            stats=stats,
            state=_state(f0, df0, alpha, step),
        )


@dataclass(frozen=True)
class AdaptiveArmijo(BacktrackingArmijo):
    """Armijo search initialized from progress in the preceding iteration."""

    optimism: float = 2.0

    @property
    def method_name(self) -> str:
        return "adaptive_armijo"

    def _initial_alpha(
        self,
        norm_d: float,
        cost0: float,
        derivative0: float,
        state: LineSearchState | None,
        initial_alpha: float | None,
    ) -> float:
        optimism = validate_positive(self.optimism, name="optimism")
        if initial_alpha is not None:
            return float(initial_alpha)
        if state is not None and state.previous_cost is not None and derivative0 != 0.0:
            alpha = optimism * 2.0 * (cost0 - state.previous_cost) / derivative0
            if math.isfinite(alpha) and alpha > 0.0:
                return float(alpha)
        return super()._initial_alpha(norm_d, cost0, derivative0, state, None)


@dataclass(frozen=True)
class StrongWolfe:
    """Strong-Wolfe search using exact retraction-curve derivatives.

    JAX forward-mode differentiation supplies the velocity of
    ``alpha -> retr(x, direction, alpha)``. Pairing that velocity with the
    trial gradient gives the actual derivative used by the curvature test,
    including for nonlinear retractions.
    """

    sufficient_decrease: float = 1e-4
    curvature: float = 0.9
    initial_stepsize: float = 1.0
    expansion: float = 2.0
    max_stepsize: float = 50.0
    max_steps: int = 20
    max_zoom_steps: int = 25
    normalize_step: bool = True

    def search(
        self,
        problem: Any,
        x: Any,
        direction: Any,
        cost: Array,
        directional_derivative: Array,
        *,
        state: LineSearchState | None = None,
        initial_alpha: float | None = None,
    ) -> LineSearchResult:
        M = require(problem, "M")
        norm_d = as_float(M.norm(x, direction))
        f0 = as_float(cost)
        df0 = as_float(directional_derivative)
        if not math.isfinite(norm_d) or norm_d <= 0.0:
            return _failure("strong_wolfe", x, cost, f0, df0, "zero or non-finite direction")
        if not math.isfinite(df0) or df0 >= 0.0:
            return _failure("strong_wolfe", x, cost, f0, df0, "direction is not descending")
        controls = (
            self.sufficient_decrease,
            self.curvature,
            self.initial_stepsize,
            self.expansion,
            self.max_stepsize,
        )
        if not all(math.isfinite(float(value)) for value in controls):
            raise ValueError("Strong-Wolfe numeric controls must be finite.")
        if not 0.0 < self.sufficient_decrease < self.curvature < 1.0:
            raise ValueError("Strong-Wolfe constants must satisfy 0 < c1 < c2 < 1.")
        if self.initial_stepsize <= 0.0 or self.expansion <= 1.0 or self.max_stepsize <= 0.0:
            raise ValueError(
                "initial_stepsize and max_stepsize must be positive, and expansion must exceed one."
            )
        max_steps = validate_integer(self.max_steps, name="max_steps")
        max_zoom_steps = validate_integer(self.max_zoom_steps, name="max_zoom_steps")
        if max_steps <= 0 or max_zoom_steps <= 0:
            raise ValueError("max_steps and max_zoom_steps must be positive.")
        if not isinstance(self.normalize_step, bool):
            raise TypeError("normalize_step must be a boolean.")

        base_alpha = (
            float(self.initial_stepsize) / norm_d
            if self.normalize_step
            else float(self.initial_stepsize)
        )
        if initial_alpha is not None:
            base_alpha = float(initial_alpha)
        elif state is not None and state.previous_alpha is not None:
            base_alpha = max(base_alpha, float(state.previous_alpha))
        max_alpha = (
            float(self.max_stepsize) / norm_d if self.normalize_step else float(self.max_stepsize)
        )
        alpha = min(base_alpha, max_alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            return _failure("strong_wolfe", x, cost, f0, df0, "non-positive trial multiplier")

        costevals = 0
        gradevals = 0

        def evaluate(a: float) -> tuple[Any, Array, Any | None, float]:
            nonlocal costevals, gradevals
            trial = _trial_curve(M, x, direction, a)
            if trial is None:
                return x, jnp.asarray(math.inf), None, math.nan
            point, curve_velocity = trial
            value = cost_value(problem, point)
            costevals += 1
            if not math.isfinite(as_float(value)):
                return point, value, None, math.nan
            grad = gradient_value(problem, point)
            derivative = as_float(inner(M, point, grad, curve_velocity))
            gradevals += 1
            return point, value, grad, derivative

        def success(
            point: Any, value: Array, grad: Any | None, a: float, reason: str = ""
        ) -> LineSearchResult:
            if grad is None:
                raise RuntimeError(
                    "A Strong-Wolfe step cannot succeed without a finite trial gradient."
                )
            step = a * norm_d
            stats = LineSearchStats(
                costevals=costevals,
                gradevals=gradevals,
                stepsize=step,
                alpha=a,
                accepted=True,
                method="strong_wolfe",
                reason=reason,
            )
            return LineSearchResult(
                point=point,
                cost=value,
                gradient=grad,
                stepsize=step,
                alpha=a,
                stats=stats,
                state=_state(f0, df0, a, step),
            )

        def zoom(
            lo: float,
            hi: float,
            phi_lo: float,
        ) -> LineSearchResult | None:
            for _ in range(max_zoom_steps):
                a = 0.5 * (lo + hi)
                point, value, grad, derivative = evaluate(a)
                phi = as_float(value)
                if (
                    (not math.isfinite(phi))
                    or phi > f0 + self.sufficient_decrease * a * df0
                    or phi >= phi_lo
                ):
                    hi = a
                    continue
                if not math.isfinite(derivative):
                    hi = a
                    continue
                if abs(derivative) <= -self.curvature * df0:
                    return success(point, value, grad, a)
                if derivative * (hi - lo) >= 0.0:
                    hi = lo
                lo = a
                phi_lo = phi
            return None

        previous_alpha = 0.0
        previous_phi = f0
        for iteration in range(max_steps):
            point, value, grad, derivative = evaluate(alpha)
            phi = as_float(value)
            if (
                (not math.isfinite(phi))
                or phi > f0 + self.sufficient_decrease * alpha * df0
                or (iteration > 0 and phi >= previous_phi)
            ):
                result = zoom(previous_alpha, alpha, previous_phi)
                if result is not None:
                    return result
                break
            if not math.isfinite(derivative):
                result = zoom(previous_alpha, alpha, previous_phi)
                if result is not None:
                    return result
                break
            if abs(derivative) <= -self.curvature * df0:
                return success(point, value, grad, alpha)
            if derivative >= 0.0:
                result = zoom(alpha, previous_alpha, phi)
                if result is not None:
                    return result
                break
            previous_alpha = alpha
            previous_phi = phi
            alpha = min(alpha * float(self.expansion), max_alpha)
            if alpha <= previous_alpha:
                break

        return _failure(
            "strong_wolfe",
            x,
            cost,
            f0,
            df0,
            "strong-Wolfe conditions were not satisfied",
            costevals=costevals,
            gradevals=gradevals,
        )


__all__ = [
    "LineSearchProtocol",
    "LineSearchState",
    "LineSearchResult",
    "ConstantStep",
    "BacktrackingArmijo",
    "AdaptiveArmijo",
    "StrongWolfe",
]
