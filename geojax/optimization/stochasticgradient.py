"""Riemannian stochastic-gradient optimization for finite sums."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Protocol, runtime_checkable
import math
import time

import jax

from geojax.geometry.base import (
    validate_boolean,
    validate_integer,
    validate_nonnegative,
    validate_positive,
)

from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
    coerce_key,
    cost_and_grad,
    get,
    make_info,
    print_iteration,
    print_iteration_header,
    require,
    retract,
    transport,
    tree_lincomb,
    tree_neg,
    tree_zeros_like,
    validate_stopping_controls,
    validate_tangent,
)


@runtime_checkable
class StepScheduleProtocol(Protocol):
    """Callable schedule returning the multiplier for iteration ``k``."""

    def __call__(self, iteration: int) -> float: ...


@dataclass(frozen=True)
class ConstantSchedule:
    """Constant stochastic-gradient step multiplier."""

    stepsize: float = 1e-2

    def __call__(self, iteration: int) -> float:
        validate_integer(iteration, name="iteration", minimum=0)
        return validate_positive(self.stepsize, name="stepsize")


@dataclass(frozen=True)
class PolynomialDecay:
    """Schedule ``initial_stepsize / (1 + decay_rate * k)**power``."""

    initial_stepsize: float = 1e-1
    decay_rate: float = 1e-2
    power: float = 0.5
    minimum_stepsize: float = 0.0

    def __call__(self, iteration: int) -> float:
        iteration = validate_integer(iteration, name="iteration", minimum=0)
        initial_stepsize = validate_positive(self.initial_stepsize, name="initial_stepsize")
        decay_rate = validate_nonnegative(self.decay_rate, name="decay_rate")
        power = validate_nonnegative(self.power, name="power")
        minimum_stepsize = validate_nonnegative(self.minimum_stepsize, name="minimum_stepsize")
        if minimum_stepsize > initial_stepsize:
            raise ValueError("minimum_stepsize cannot exceed initial_stepsize.")
        value = initial_stepsize / (1.0 + decay_rate * iteration) ** power
        return max(float(value), minimum_stepsize)


@dataclass(frozen=True)
class CosineDecay:
    """Cosine interpolation from ``initial_stepsize`` to ``final_stepsize``."""

    initial_stepsize: float = 1e-1
    final_stepsize: float = 0.0
    decay_steps: int = 1000

    def __call__(self, iteration: int) -> float:
        iteration = validate_integer(iteration, name="iteration", minimum=0)
        decay_steps = validate_integer(self.decay_steps, name="decay_steps", minimum=1)
        initial_stepsize = validate_positive(self.initial_stepsize, name="initial_stepsize")
        final_stepsize = validate_nonnegative(self.final_stepsize, name="final_stepsize")
        if final_stepsize > initial_stepsize:
            raise ValueError("final_stepsize cannot exceed initial_stepsize.")
        progress = min(max(float(iteration) / decay_steps, 0.0), 1.0)
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(final_stepsize + weight * (initial_stepsize - final_stepsize))


@dataclass(frozen=True)
class StochasticGradient:
    """Mini-batch Riemannian stochastic gradient with optional momentum."""

    requires_gradient: bool = True
    batch_size: int = 1
    step_schedule: StepScheduleProtocol = field(default_factory=PolynomialDecay)
    momentum: float = 0.0
    clip_norm: float | None = None
    replace: bool = True
    evaluation_period: int = 10
    tolgradnorm: float = 0.0
    maxiter: int = 1000
    maxtime: float = math.inf
    minstepsize: float = 0.0
    verbosity: int = 2
    key: Optional[Array | int] = None
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        validate_stopping_controls(self)
        M = require(problem, "M")
        x = require(problem, "x0")
        sample_batch = get(problem, "sample_batch", None)
        batch_cost_and_grad = get(problem, "batch_cost_and_grad", None)
        if not callable(sample_batch) or not callable(batch_cost_and_grad):
            raise ValueError(
                "StochasticGradient requires a FiniteSum-like problem with "
                "sample_batch and batch_cost_and_grad methods."
            )
        batch_size = validate_integer(self.batch_size, name="batch_size", minimum=1)
        evaluation_period = validate_integer(
            self.evaluation_period, name="evaluation_period", minimum=1
        )
        momentum = validate_nonnegative(self.momentum, name="momentum")
        if momentum >= 1.0:
            raise ValueError("momentum must lie in [0, 1).")
        clip_norm = (
            None if self.clip_norm is None else validate_positive(self.clip_norm, name="clip_norm")
        )
        replace_batches = validate_boolean(self.replace, name="replace")
        if not callable(self.step_schedule):
            raise TypeError("step_schedule must be callable.")

        local_key = None
        split_key = get(problem, "split_key", None) if self.key is None else None
        if not callable(split_key):
            local_key = coerce_key(self.key)
            if local_key is None:
                local_key = jax.random.key(0)

        def next_key() -> Array:
            nonlocal local_key
            if callable(split_key):
                return split_key()
            local_key, subkey = jax.random.split(local_key)
            return subkey

        start_time = time.perf_counter()
        info: List[InfoEntry] = []
        f, full_gradient = cost_and_grad(problem, x)
        gradnorm = M.norm(x, full_gradient)
        velocity = tree_zeros_like(full_gradient)
        info.append(
            make_info(
                iter=0,
                cost=f,
                gradnorm=gradnorm,
                stepsize=math.nan,
                start_time=start_time,
                linesearch=None,
                problem=problem,
                x=x,
                solver=self,
                full_evaluation=True,
                evaluation_scope="full",
            )
        )
        print_iteration_header(self.verbosity)

        while True:
            current = info[-1]
            print_iteration(current, self.verbosity)
            reason = _stochastic_stopping_reason(problem, x, info, self)
            if reason:
                if not bool(current.extra.get("full_evaluation", False)):
                    f, full_gradient = cost_and_grad(problem, x)
                    current = replace(
                        current,
                        cost=as_float(f),
                        gradnorm=as_float(M.norm(x, full_gradient)),
                        extra={
                            **current.extra,
                            "full_evaluation": True,
                            "evaluation_scope": "full",
                        },
                    )
                info[-1] = replace(current, reason=reason)
                if self.verbosity >= 1:
                    print(reason)
                break

            indices = sample_batch(next_key(), batch_size, replace=replace_batches)
            _, stochastic_gradient = batch_cost_and_grad(x, indices)
            stochastic_gradient = validate_tangent(
                M,
                x,
                stochastic_gradient,
                name="mini-batch Riemannian gradient",
            )
            stochastic_norm = as_float(M.norm(x, stochastic_gradient))
            if clip_norm is not None and stochastic_norm > clip_norm:
                stochastic_gradient = tree_lincomb(
                    clip_norm / max(stochastic_norm, 1e-300),
                    stochastic_gradient,
                )
                stochastic_norm = clip_norm

            velocity = tree_lincomb(
                momentum,
                velocity,
                1.0,
                stochastic_gradient,
            )
            direction = tree_neg(velocity)
            scheduled_value = self.step_schedule(current.iter)
            if isinstance(scheduled_value, bool):
                raise TypeError("step_schedule must return a real scalar, not a boolean.")
            learning_rate = float(scheduled_value)
            if not math.isfinite(learning_rate) or learning_rate < 0.0:
                raise ValueError("step_schedule must return a finite nonnegative value.")
            if learning_rate == 0.0:
                reason = "Step schedule reached a zero learning rate."
                if not bool(current.extra.get("full_evaluation", False)):
                    f, full_gradient = cost_and_grad(problem, x)
                    current = replace(
                        current,
                        cost=as_float(f),
                        gradnorm=as_float(M.norm(x, full_gradient)),
                        extra={
                            **current.extra,
                            "full_evaluation": True,
                            "evaluation_scope": "full",
                        },
                    )
                info[-1] = replace(current, reason=reason)
                if self.verbosity >= 1:
                    print(reason)
                break
            stepnorm = learning_rate * as_float(M.norm(x, direction))
            newx = retract(M, x, direction, learning_rate)
            velocity = transport(M, x, newx, velocity)
            x = newx

            next_iteration = current.iter + 1
            full_evaluation = next_iteration % evaluation_period == 0
            if full_evaluation:
                f, diagnostic_gradient = cost_and_grad(problem, x)
            else:
                f, diagnostic_gradient = batch_cost_and_grad(x, indices)
                diagnostic_gradient = validate_tangent(
                    M,
                    x,
                    diagnostic_gradient,
                    name="diagnostic mini-batch Riemannian gradient",
                )
            gradnorm = M.norm(x, diagnostic_gradient)
            info.append(
                make_info(
                    iter=next_iteration,
                    cost=f,
                    gradnorm=gradnorm,
                    stepsize=stepnorm,
                    start_time=start_time,
                    linesearch=None,
                    problem=problem,
                    x=x,
                    solver=self,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    stochastic_gradnorm=stochastic_norm,
                    full_evaluation=full_evaluation,
                    evaluation_scope="full" if full_evaluation else "mini_batch",
                )
            )

        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return x, info[-1].cost, info


def _stochastic_stopping_reason(
    problem: Any,
    x: Array,
    info: List[InfoEntry],
    solver: StochasticGradient,
) -> str:
    current = info[-1]
    if not math.isfinite(float(current.cost)):
        return "Nonfinite cost encountered."
    if bool(current.extra.get("full_evaluation", False)) and not math.isfinite(
        float(current.gradnorm)
    ):
        return "Nonfinite gradient norm encountered."
    if bool(current.extra.get("full_evaluation", False)) and current.gradnorm <= solver.tolgradnorm:
        return f"Gradient norm tolerance reached: {current.gradnorm:g} <= {solver.tolgradnorm:g}."
    if current.iter >= solver.maxiter:
        return f"Maximum iteration count reached: options.maxiter = {solver.maxiter}."
    if current.time >= solver.maxtime:
        return f"Maximum time reached: options.maxtime = {solver.maxtime:g}."
    if current.iter > 0 and current.stepsize < solver.minstepsize:
        return f"Last stepsize smaller than options.minstepsize = {solver.minstepsize:g}."
    if solver.stopfun is not None:
        result = solver.stopfun(problem, x, current)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("stopfun must return a pair (stop, reason).")
        stop, reason = result
        if not isinstance(stop, bool):
            raise TypeError("stopfun's stop value must be a boolean.")
        if not isinstance(reason, str):
            raise TypeError("stopfun's reason must be a string.")
        if stop:
            return reason or "User stopfun triggered."
    return ""


__all__ = [
    "StepScheduleProtocol",
    "ConstantSchedule",
    "PolynomialDecay",
    "CosineDecay",
    "StochasticGradient",
]
