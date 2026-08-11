"""Shared tangent-space conjugate-gradient linear solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

from geojax.geometry.base import validate_integer, validate_nonnegative

from .minimize import (
    Array,
    as_float,
    inner,
    tree_lincomb,
    tree_zeros_like,
    validate_tangent,
)


@dataclass(frozen=True)
class TangentCGResult:
    """Result of a tangent-space conjugate-gradient solve."""

    solution: Array
    iterations: int
    residual_norm: float
    converged: bool
    negative_curvature: bool
    preconditioner_fallback: bool
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

    if not callable(operator):
        raise TypeError("operator must be callable.")
    if preconditioner is not None and not callable(preconditioner):
        raise TypeError("preconditioner must be callable or None.")
    max_iterations = validate_integer(max_iterations, name="max_iterations", minimum=1)
    relative_tolerance = validate_nonnegative(
        relative_tolerance,
        name="relative_tolerance",
    )
    absolute_tolerance = validate_nonnegative(
        absolute_tolerance,
        name="absolute_tolerance",
    )
    curvature_tolerance = validate_nonnegative(
        curvature_tolerance,
        name="curvature_tolerance",
    )

    rhs = validate_tangent(M, x, rhs, name="CG right-hand side")
    solution = tree_zeros_like(rhs)
    residual = rhs
    residual_norm0 = as_float(M.norm(x, residual))
    if not math.isfinite(residual_norm0):
        raise ValueError("The initial CG residual norm must be finite.")
    target = max(absolute_tolerance, relative_tolerance * residual_norm0)
    if residual_norm0 <= target:
        return TangentCGResult(solution, 0, residual_norm0, True, False, False, "initial residual")

    precondition = preconditioner if preconditioner is not None else lambda value: value
    z = validate_tangent(M, x, precondition(residual), name="preconditioned CG residual")
    direction = z
    rz = as_float(inner(M, x, residual, z))
    preconditioner_fallback = False
    if not math.isfinite(rz) or rz <= 0.0:
        z = residual
        direction = residual
        rz = as_float(inner(M, x, residual, residual))
        preconditioner_fallback = True
    if not math.isfinite(rz) or rz <= 0.0:
        raise ValueError("The tangent metric must be positive on a nonzero CG residual.")

    residual_norm = residual_norm0
    for iteration in range(1, max_iterations + 1):
        operator_direction = validate_tangent(M, x, operator(direction), name="CG operator output")
        curvature = as_float(inner(M, x, direction, operator_direction))
        direction_norm2_raw = as_float(inner(M, x, direction, direction))
        if not math.isfinite(curvature) or not math.isfinite(direction_norm2_raw):
            return TangentCGResult(
                solution,
                iteration,
                residual_norm,
                False,
                False,
                preconditioner_fallback,
                "non-finite operator curvature",
            )
        direction_norm2 = max(direction_norm2_raw, 0.0)
        threshold = curvature_tolerance * direction_norm2
        if curvature <= threshold:
            if iteration == 1:
                solution = direction
            return TangentCGResult(
                solution,
                iteration,
                residual_norm,
                False,
                True,
                preconditioner_fallback,
                "non-positive curvature",
            )

        alpha = rz / curvature
        solution = tree_lincomb(1.0, solution, alpha, direction)
        residual = tree_lincomb(1.0, residual, -alpha, operator_direction)
        residual_norm = as_float(M.norm(x, residual))
        if not math.isfinite(residual_norm):
            return TangentCGResult(
                solution,
                iteration,
                residual_norm,
                False,
                False,
                preconditioner_fallback,
                "non-finite residual",
            )
        if residual_norm <= target:
            return TangentCGResult(
                solution,
                iteration,
                residual_norm,
                True,
                False,
                preconditioner_fallback,
                "residual tolerance",
            )

        z_next = validate_tangent(M, x, precondition(residual), name="preconditioned CG residual")
        rz_next = as_float(inner(M, x, residual, z_next))
        if not math.isfinite(rz_next) or rz_next <= 0.0:
            z_next = residual
            rz_next = as_float(inner(M, x, residual, residual))
            if not math.isfinite(rz_next) or rz_next <= 0.0:
                raise ValueError("The tangent metric must be positive on a nonzero CG residual.")
            direction = z_next
            preconditioner_fallback = True
        else:
            beta = rz_next / max(rz, 1e-300)
            direction = tree_lincomb(1.0, z_next, beta, direction)
        z = z_next
        rz = rz_next

    return TangentCGResult(
        solution,
        max_iterations,
        residual_norm,
        False,
        False,
        preconditioner_fallback,
        "maximum iterations",
    )


__all__ = ["TangentCGResult", "tangent_conjugate_gradient"]
