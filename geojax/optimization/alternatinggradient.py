"""Block-alternating Riemannian gradient method for Product geometries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Sequence
import math
import time


from geojax.geometry.base import validate_integer

from .linesearch import AdaptiveArmijo, LineSearchProtocol, LineSearchState
from .minimize import (
    Array,
    InfoEntry,
    LineSearchStats,
    StatsFn,
    StopFn,
    cost_and_grad,
    gradient_value,
    inner,
    make_info,
    print_iteration,
    print_iteration_header,
    require,
    stopping_reason,
    tree_neg,
    tree_zeros_like,
    validate_line_search_result,
)


@dataclass(frozen=True)
class AlternatingGradient:
    """Cycle through Product factors using one gradient block at a time.

    ``block_order`` refers to the leaves of the Product factor pytree in JAX's
    deterministic flattening order. When omitted, every leaf is visited once
    per outer iteration.
    """

    requires_gradient: bool = True
    block_order: Sequence[int] | None = None
    tolgradnorm: float = 1e-6
    maxiter: int = 1000
    maxtime: float = math.inf
    minstepsize: float = 1e-10
    verbosity: int = 2
    line_search: LineSearchProtocol = field(default_factory=AdaptiveArmijo)
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        M = require(problem, "M")
        x = require(problem, "x0")
        factor_leaves = getattr(M, "_factor_leaves", None)
        flatten_like = getattr(M, "_flatten_like", None)
        unflatten = getattr(M, "_unflatten", None)
        if factor_leaves is None or not callable(flatten_like) or not callable(unflatten):
            raise ValueError("AlternatingGradient requires a Product geometry.")

        num_blocks = len(factor_leaves)
        if self.block_order is None:
            order = tuple(range(num_blocks))
        else:
            if isinstance(self.block_order, (str, bytes)):
                raise TypeError("block_order must be a sequence of integer block indices.")
            try:
                supplied_order = tuple(self.block_order)
            except TypeError as exc:
                raise TypeError("block_order must be a sequence of integer block indices.") from exc
            order = tuple(
                validate_integer(value, name="block_order entry") for value in supplied_order
            )
        if sorted(order) != list(range(num_blocks)):
            raise ValueError("block_order must contain each flattened Product block exactly once.")

        search_states: list[LineSearchState | None] = [None] * num_blocks
        start_time = time.perf_counter()
        info: List[InfoEntry] = []
        f, g = cost_and_grad(problem, x)
        gradnorm = M.norm(x, g)
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
            )
        )
        print_iteration_header(self.verbosity)

        while True:
            current = info[-1]
            print_iteration(current, self.verbosity)
            reason = stopping_reason(problem, x, info, self)
            if reason:
                info[-1] = replace(current, reason=reason)
                if self.verbosity >= 1:
                    print(reason)
                break

            block_costs: list[float] = []
            block_steps: list[float] = []
            costevals = 0
            gradevals = 0
            accepted_blocks = 0
            last_alpha = 0.0

            for block in order:
                zero_leaves = list(flatten_like(tree_zeros_like(g), "zero tangent vector"))
                gradient_leaves = flatten_like(g, "gradient")
                zero_leaves[block] = gradient_leaves[block]
                block_gradient = unflatten(zero_leaves)
                direction = tree_neg(block_gradient)
                directional_derivative = inner(M, x, g, direction)
                result = validate_line_search_result(
                    problem,
                    self.line_search.search(
                        problem,
                        x,
                        direction,
                        f,
                        directional_derivative,
                        state=search_states[block],
                    ),
                )
                search_states[block] = result.state
                x = result.point
                f = result.cost
                g = result.gradient if result.gradient is not None else gradient_value(problem, x)
                costevals += result.stats.costevals
                gradevals += result.stats.gradevals
                accepted_blocks += int(result.stats.accepted)
                last_alpha = result.alpha
                block_costs.append(float(result.cost))
                block_steps.append(float(result.stepsize))

            gradnorm = M.norm(x, g)
            cycle_stepsize = math.sqrt(sum(step * step for step in block_steps))
            line_stats = LineSearchStats(
                costevals=costevals,
                gradevals=gradevals,
                stepsize=cycle_stepsize,
                alpha=last_alpha,
                accepted=accepted_blocks == num_blocks,
                method="alternating",
                reason=f"{accepted_blocks}/{num_blocks} blocks accepted",
            )
            info.append(
                make_info(
                    iter=current.iter + 1,
                    cost=f,
                    gradnorm=gradnorm,
                    stepsize=cycle_stepsize,
                    start_time=start_time,
                    linesearch=line_stats,
                    problem=problem,
                    x=x,
                    solver=self,
                    block_order=order,
                    block_costs=tuple(block_costs),
                    block_steps=tuple(block_steps),
                    accepted_blocks=accepted_blocks,
                )
            )

        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return x, info[-1].cost, info


__all__ = ["AlternatingGradient"]
