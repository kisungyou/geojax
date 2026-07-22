"""Riemannian stochastic-gradient optimization for finite sums."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Protocol, runtime_checkable
import math
import time

import jax

from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
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
        del iteration
        return float(self.stepsize)


@dataclass(frozen=True)
class PolynomialDecay:
    """Schedule ``initial_stepsize / (1 + decay_rate * k)**power``."""

    initial_stepsize: float = 1e-1
    decay_rate: float = 1e-2
    power: float = 0.5
    minimum_stepsize: float = 0.0

    def __call__(self, iteration: int) -> float:
        value = self.initial_stepsize / (1.0 + self.decay_rate * iteration) ** self.power
        return max(float(value), float(self.minimum_stepsize))


@dataclass(frozen=True)
class CosineDecay:
    """Cosine interpolation from ``initial_stepsize`` to ``final_stepsize``."""

    initial_stepsize: float = 1e-1
    final_stepsize: float = 0.0
    decay_steps: int = 1000

    def __call__(self, iteration: int) -> float:
        progress = min(max(float(iteration) / max(int(self.decay_steps), 1), 0.0), 1.0)
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.final_stepsize + weight * (self.initial_stepsize - self.final_stepsize))


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
        M = require(problem, "M")
        x = require(problem, "x0")
        sample_batch = get(problem, "sample_batch", None)
        batch_cost_and_grad = get(problem, "batch_cost_and_grad", None)
        if not callable(sample_batch) or not callable(batch_cost_and_grad):
            raise ValueError(
                "StochasticGradient requires a FiniteSum-like problem with "
                "sample_batch and batch_cost_and_grad methods."
            )
        if self.batch_size <= 0 or self.evaluation_period <= 0:
            raise ValueError("batch_size and evaluation_period must be positive.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1).")

        local_key = None
        split_key = get(problem, "split_key", None) if self.key is None else None
        if not callable(split_key):
            local_key = jax.random.key(0) if self.key is None else (
                jax.random.key(self.key) if isinstance(self.key, int) else self.key
            )

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
                        extra={**current.extra, "full_evaluation": True},
                    )
                info[-1] = replace(current, reason=reason)
                if self.verbosity >= 1:
                    print(reason)
                break

            indices = sample_batch(next_key(), self.batch_size, replace=self.replace)
            _, stochastic_gradient = batch_cost_and_grad(x, indices)
            stochastic_norm = as_float(M.norm(x, stochastic_gradient))
            if self.clip_norm is not None and stochastic_norm > float(self.clip_norm):
                stochastic_gradient = tree_lincomb(
                    float(self.clip_norm) / max(stochastic_norm, 1e-300),
                    stochastic_gradient,
                )
                stochastic_norm = float(self.clip_norm)

            velocity = tree_lincomb(
                float(self.momentum),
                velocity,
                1.0,
                stochastic_gradient,
            )
            direction = tree_neg(velocity)
            learning_rate = float(self.step_schedule(current.iter))
            if not math.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError("step_schedule must return a positive finite value.")
            stepnorm = learning_rate * as_float(M.norm(x, direction))
            newx = retract(M, x, direction, learning_rate)
            velocity = transport(M, x, newx, velocity)
            x = newx

            next_iteration = current.iter + 1
            full_evaluation = next_iteration % int(self.evaluation_period) == 0
            if full_evaluation:
                f, diagnostic_gradient = cost_and_grad(problem, x)
            else:
                f, diagnostic_gradient = batch_cost_and_grad(x, indices)
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
                    batch_size=int(self.batch_size),
                    stochastic_gradnorm=stochastic_norm,
                    full_evaluation=full_evaluation,
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
    if bool(current.extra.get("full_evaluation", False)) and current.gradnorm <= solver.tolgradnorm:
        return f"Gradient norm tolerance reached: {current.gradnorm:g} <= {solver.tolgradnorm:g}."
    if current.iter >= solver.maxiter:
        return f"Maximum iteration count reached: options.maxiter = {solver.maxiter}."
    if current.time >= solver.maxtime:
        return f"Maximum time reached: options.maxtime = {solver.maxtime:g}."
    if current.iter > 0 and current.stepsize < solver.minstepsize:
        return f"Last stepsize smaller than options.minstepsize = {solver.minstepsize:g}."
    if solver.stopfun is not None:
        stop, reason = solver.stopfun(problem, x, current)
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
