"""Shared tangent-space conjugate-gradient linear solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

from .minimize import Array, as_float, inner, tree_lincomb, tree_zeros_like


@dataclass(frozen=True)
class TangentCGResult:
    """Result of a tangent-space conjugate-gradient solve."""

    solution: Array
    iterations: int
    residual_norm: float
    converged: bool
    negative_curvature: bool
    reason: str


def tangent_conjugate_gradient(
    M: Any,
    x: Array,
    operator: Callable[[Array], Array],
    rhs: Array,
    *,
    preconditioner: Callable[[Array], Array] | None = None,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-10,
    max_iterations: int = 100,
    curvature_tolerance: float = 0.0,
) -> TangentCGResult:
    """Approximately solve ``operator(solution) = rhs`` in ``T_x M``.

    The routine assumes a self-adjoint positive operator. It terminates early
    when non-positive curvature is encountered, which lets Newton-type methods
    fall back to a usable truncated direction.
    """

    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive.")
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("CG tolerances must be nonnegative.")

    solution = tree_zeros_like(rhs)
    residual = rhs
    residual_norm0 = as_float(M.norm(x, residual))
    target = max(float(absolute_tolerance), float(relative_tolerance) * residual_norm0)
    if residual_norm0 <= target:
        return TangentCGResult(solution, 0, residual_norm0, True, False, "initial residual")

    precondition = preconditioner if preconditioner is not None else lambda value: value
    z = precondition(residual)
    direction = z
    rz = as_float(inner(M, x, residual, z))
    if not math.isfinite(rz) or rz <= 0.0:
        return TangentCGResult(
            solution,
            0,
            residual_norm0,
            False,
            True,
            "preconditioner is not positive",
        )

    residual_norm = residual_norm0
    for iteration in range(1, int(max_iterations) + 1):
        operator_direction = operator(direction)
        curvature = as_float(inner(M, x, direction, operator_direction))
        direction_norm2 = max(as_float(inner(M, x, direction, direction)), 0.0)
        threshold = float(curvature_tolerance) * direction_norm2
        if not math.isfinite(curvature) or curvature <= threshold:
            if iteration == 1:
                solution = direction
            return TangentCGResult(
                solution,
                iteration,
                residual_norm,
                False,
                True,
                "non-positive curvature",
            )

        alpha = rz / curvature
        solution = tree_lincomb(1.0, solution, alpha, direction)
        residual = tree_lincomb(1.0, residual, -alpha, operator_direction)
        residual_norm = as_float(M.norm(x, residual))
        if residual_norm <= target:
            return TangentCGResult(
                solution,
                iteration,
                residual_norm,
                True,
                False,
                "residual tolerance",
            )

        z_next = precondition(residual)
        rz_next = as_float(inner(M, x, residual, z_next))
        if not math.isfinite(rz_next) or rz_next <= 0.0:
            return TangentCGResult(
                solution,
                iteration,
                residual_norm,
                False,
                True,
                "preconditioner lost positivity",
            )
        beta = rz_next / max(rz, 1e-300)
        direction = tree_lincomb(1.0, z_next, beta, direction)
        z = z_next
        rz = rz_next

    return TangentCGResult(
        solution,
        int(max_iterations),
        residual_norm,
        False,
        False,
        "maximum iterations",
    )


__all__ = ["TangentCGResult", "tangent_conjugate_gradient"]
