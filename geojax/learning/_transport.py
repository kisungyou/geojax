"""Exact and regularized transport between empirical manifold measures."""

from __future__ import annotations

from collections import deque
from typing import Any

import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import TransportResult
from ._utils import (
    as_real_array,
    integer_control,
    normalize_weights,
    positive_control,
    require_unbatched,
    tree_contains_tracer,
)


def _complete_tree_basis(plan: Any, costs: Any, tolerance: float) -> Any:
    """Complete positive transport edges to a deterministic spanning-tree basis."""
    rows, columns = plan.shape
    basis = jnp.asarray(plan > 0.0)
    parent = list(range(rows + columns))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return False
        parent[root_right] = root_left
        return True

    positive = [
        (row, column)
        for row in range(rows)
        for column in range(columns)
        if bool(basis[row, column])
    ]
    rebuilt = jnp.zeros_like(basis)
    for row, column in positive:
        if union(row, rows + column):
            rebuilt = rebuilt.at[row, column].set(True)
    candidates = sorted(
        (
            (float(costs[row, column]), row, column)
            for row in range(rows)
            for column in range(columns)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    count = int(jnp.sum(rebuilt))
    for _, row, column in candidates:
        if count >= rows + columns - 1:
            break
        if bool(rebuilt[row, column]):
            continue
        if union(row, rows + column):
            rebuilt = rebuilt.at[row, column].set(True)
            count += 1
    if count != rows + columns - 1:
        raise RuntimeError("Could not construct a transportation-tree basis.")
    return rebuilt


def _northwest_corner(a: Any, b: Any, costs: Any, tolerance: float) -> tuple[Any, Any]:
    rows, columns = costs.shape
    supply = [float(value) for value in a]
    demand = [float(value) for value in b]
    plan = jnp.zeros((rows, columns), dtype=costs.dtype)
    row = column = 0
    while row < rows and column < columns:
        mass = min(supply[row], demand[column])
        plan = plan.at[row, column].set(mass)
        supply[row] -= mass
        demand[column] -= mass
        row_done = supply[row] <= tolerance
        column_done = demand[column] <= tolerance
        if row_done:
            row += 1
        if column_done:
            column += 1
    return plan, _complete_tree_basis(plan, costs, tolerance)


def _dual_potentials(costs: Any, basis: Any) -> tuple[Any, Any]:
    rows, columns = costs.shape
    row_potentials: list[float | None] = [None] * rows
    column_potentials: list[float | None] = [None] * columns
    row_potentials[0] = 0.0
    queue: deque[tuple[str, int]] = deque([("row", 0)])
    while queue:
        kind, index = queue.popleft()
        if kind == "row":
            for column in range(columns):
                if bool(basis[index, column]) and column_potentials[column] is None:
                    column_potentials[column] = float(costs[index, column]) - float(
                        row_potentials[index]
                    )
                    queue.append(("column", column))
        else:
            for row in range(rows):
                if bool(basis[row, index]) and row_potentials[row] is None:
                    row_potentials[row] = float(costs[row, index]) - float(column_potentials[index])
                    queue.append(("row", row))
    if any(value is None for value in row_potentials + column_potentials):
        raise RuntimeError("Transportation basis is disconnected.")
    return (
        jnp.asarray(row_potentials, dtype=costs.dtype),
        jnp.asarray(column_potentials, dtype=costs.dtype),
    )


def _basis_path(basis: Any, entering: tuple[int, int]) -> list[tuple[int, int]]:
    """Return tree edges on the path from entering column to entering row."""
    rows, columns = basis.shape
    start = rows + entering[1]
    target = entering[0]
    parent: dict[int, int | None] = {start: None}
    queue = deque([start])
    while queue and target not in parent:
        node = queue.popleft()
        if node < rows:
            neighbors = [rows + column for column in range(columns) if bool(basis[node, column])]
        else:
            column = node - rows
            neighbors = [row for row in range(rows) if bool(basis[row, column])]
        for neighbor in neighbors:
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if target not in parent:
        raise RuntimeError("Entering edge did not close a basis cycle.")
    nodes = [target]
    while nodes[-1] != start:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    edges = []
    for left, right in zip(nodes[:-1], nodes[1:]):
        row, column_node = (left, right) if left < rows else (right, left)
        edges.append((row, column_node - rows))
    return edges


def _transportation_simplex(
    costs: Any,
    a: Any,
    b: Any,
    *,
    tolerance: float,
    max_pivots: int,
) -> TransportResult:
    """Solve a balanced transportation problem with deterministic Bland pivots."""
    costs = as_real_array(costs, name="transport costs")
    a = jnp.asarray(a, dtype=costs.dtype)
    b = jnp.asarray(b, dtype=costs.dtype)
    requested_tolerance = positive_control(tolerance, name="tolerance")
    max_pivots = integer_control(max_pivots, name="max_pivots")
    if max_pivots < 0:
        raise ValueError("max_pivots must be nonnegative.")
    if a.ndim != 1 or b.ndim != 1 or a.size < 1 or b.size < 1:
        raise ValueError("a and b must be nonempty one-dimensional marginals.")
    if costs.ndim != 2 or costs.shape != (a.size, b.size):
        raise ValueError("costs must have shape (len(a), len(b)).")
    if (
        not bool(jnp.all(jnp.isfinite(costs)))
        or not bool(jnp.all(jnp.isfinite(a)))
        or not bool(jnp.all(jnp.isfinite(b)))
    ):
        raise ValueError("transport costs and marginals must be finite.")
    if bool(jnp.any(a < 0.0)) or bool(jnp.any(b < 0.0)):
        raise ValueError("transport marginals must be nonnegative.")
    mass_a = float(jnp.sum(a))
    mass_b = float(jnp.sum(b))
    if mass_a <= 0.0 or mass_b <= 0.0:
        raise ValueError("transport marginals must have positive total mass.")
    dtype_epsilon = float(jnp.finfo(costs.dtype).eps)
    mass_scale = max(mass_a, mass_b)
    mass_tolerance = max(
        requested_tolerance * mass_scale,
        100.0 * dtype_epsilon * max(costs.shape) * mass_scale,
    )
    if abs(mass_a - mass_b) > mass_tolerance:
        raise ValueError("transport marginals must have equal total mass.")
    normalized_a = a / mass_a
    normalized_b = b / mass_b
    cost_scale = float(jnp.max(jnp.abs(costs)))
    safe_cost_scale = cost_scale if cost_scale > 0.0 else 1.0
    normalized_costs = costs / safe_cost_scale
    tolerance = max(
        requested_tolerance,
        100.0 * dtype_epsilon * max(costs.shape),
    )
    positive_rows = jnp.nonzero(normalized_a > 0.0, size=a.size, fill_value=-1)[0]
    positive_rows = positive_rows[positive_rows >= 0]
    positive_columns = jnp.nonzero(
        normalized_b > 0.0,
        size=b.size,
        fill_value=-1,
    )[0]
    positive_columns = positive_columns[positive_columns >= 0]
    reduced_costs = normalized_costs[
        positive_rows[:, None],
        positive_columns[None, :],
    ]
    reduced_a = normalized_a[positive_rows]
    reduced_b = normalized_b[positive_columns]
    plan, basis = _northwest_corner(reduced_a, reduced_b, reduced_costs, tolerance)
    converged = False
    minimum_reduced_cost = -jnp.inf
    row_potentials = column_potentials = None
    pivots = 0
    while True:
        row_potentials, column_potentials = _dual_potentials(reduced_costs, basis)
        reduced = reduced_costs - row_potentials[:, None] - column_potentials[None, :]
        reduced = jnp.where(basis, jnp.inf, reduced)
        minimum_reduced_cost = jnp.min(reduced)
        if float(minimum_reduced_cost) >= -tolerance:
            converged = True
            break
        if pivots >= max_pivots:
            break
        entering_candidates = [
            (row, column)
            for row in range(plan.shape[0])
            for column in range(plan.shape[1])
            if not bool(basis[row, column]) and float(reduced[row, column]) < -tolerance
        ]
        if not entering_candidates:
            raise RuntimeError("Negative reduced cost was reported but no entering edge was found.")
        entering = min(entering_candidates)
        path = _basis_path(basis, entering)
        minus_edges = path[0::2]
        plus_edges = path[1::2]
        theta = min(float(plan[row, column]) for row, column in minus_edges)
        leaving = min(
            (edge for edge in minus_edges if float(plan[edge]) <= theta + tolerance),
            key=lambda edge: (edge[0], edge[1]),
        )
        plan = plan.at[entering].add(theta)
        for edge in plus_edges:
            plan = plan.at[edge].add(theta)
        for edge in minus_edges:
            plan = plan.at[edge].add(-theta)
        plan = jnp.where((plan < 0.0) & (plan >= -tolerance), 0.0, plan)
        plan = plan.at[leaving].set(0.0)
        if bool(jnp.any(plan < -tolerance)):
            raise RuntimeError("Transportation pivot produced a negative basic mass.")
        basis = basis.at[entering].set(True)
        basis = basis.at[leaving].set(False)
        pivots += 1
    normalized_plan = jnp.zeros_like(costs)
    normalized_plan = normalized_plan.at[
        positive_rows[:, None],
        positive_columns[None, :],
    ].set(plan)
    full_plan = mass_a * normalized_plan
    normalized_primal = jnp.sum(normalized_plan * normalized_costs)
    normalized_row_residual = jnp.max(jnp.abs(jnp.sum(normalized_plan, axis=1) - normalized_a))
    normalized_column_residual = jnp.max(jnp.abs(jnp.sum(normalized_plan, axis=0) - normalized_b))
    normalized_dual = jnp.sum(reduced_a * row_potentials) + jnp.sum(reduced_b * column_potentials)
    normalized_duality_gap = normalized_primal - normalized_dual
    certificate_scale = max(
        abs(float(normalized_primal)),
        abs(float(normalized_dual)),
        1.0,
    )
    feasible = (
        float(normalized_row_residual) <= tolerance
        and float(normalized_column_residual) <= tolerance
        and abs(float(normalized_duality_gap)) <= tolerance * certificate_scale
    )
    reduced_optimal = float(minimum_reduced_cost) >= -tolerance
    converged = bool(converged and feasible and reduced_optimal)
    if converged:
        reason = "optimality, feasibility, and duality certificates satisfied"
    elif not feasible:
        reason = "transport certificate failed feasibility or duality checks"
    else:
        reason = "maximum pivots reached"
    return TransportResult(
        distance=jnp.nan,
        cost=mass_a * safe_cost_scale * normalized_primal,
        plan=full_plan,
        iterations=pivots,
        converged=converged,
        reason=reason,
        diagnostics={
            "row_residual": mass_a * normalized_row_residual,
            "column_residual": mass_b * normalized_column_residual,
            "duality_gap": mass_a * safe_cost_scale * normalized_duality_gap,
            "minimum_reduced_cost": safe_cost_scale * minimum_reduced_cost,
            "normalized_row_residual": normalized_row_residual,
            "normalized_column_residual": normalized_column_residual,
            "normalized_duality_gap": normalized_duality_gap,
            "normalized_minimum_reduced_cost": minimum_reduced_cost,
            "normalized_cost": normalized_primal,
            "mass_scale": mass_a,
            "cost_scale": safe_cost_scale,
            "basis": basis,
            "requested_tolerance": requested_tolerance,
            "effective_tolerance": tolerance,
            "mass_tolerance": mass_tolerance,
        },
    )


def empirical_wasserstein_distance(
    manifold: Any,
    x: Any,
    y: Any,
    *,
    p: float = 2.0,
    weights_x: Any | None = None,
    weights_y: Any | None = None,
    tolerance: float = 1e-10,
    max_pivots: int = 10_000,
) -> TransportResult:
    """Compute exact weighted empirical Wasserstein distance by transportation simplex."""
    require_exact_operations(manifold, "empirical_wasserstein_distance", "dist")
    left = as_manifold_data(manifold, x)
    right = as_manifold_data(manifold, y)
    require_unbatched(left, "empirical_wasserstein_distance")
    require_unbatched(right, "empirical_wasserstein_distance")
    p = positive_control(p, name="p")
    if p < 1.0:
        raise ValueError("p must be finite and at least 1.")
    tolerance = positive_control(tolerance, name="tolerance")
    max_pivots = integer_control(max_pivots, name="max_pivots")
    if max_pivots < 0:
        raise ValueError("max_pivots must be nonnegative.")
    a = normalize_weights(left.n_samples, weights_x)
    b = normalize_weights(right.n_samples, weights_y)
    distances = pairwise_distances(manifold, left, right)
    distance_scale = jnp.max(distances)
    safe_distance_scale = jnp.where(
        distance_scale > 0.0,
        distance_scale,
        jnp.ones_like(distance_scale),
    )
    normalized_costs = (distances / safe_distance_scale) ** p
    result = _transportation_simplex(
        normalized_costs,
        a,
        b,
        tolerance=tolerance,
        max_pivots=max_pivots,
    )
    effective_tolerance = result.diagnostics["effective_tolerance"]
    if float(result.cost) < -effective_tolerance:
        raise FloatingPointError("Exact transport returned a negative nontrivial cost.")
    normalized_cost = jnp.maximum(result.cost, 0.0)
    distance = safe_distance_scale * normalized_cost ** (1.0 / p)
    distance = jnp.where(distance_scale > 0.0, distance, 0.0)
    cost_scale = safe_distance_scale**p
    cost = cost_scale * normalized_cost
    return TransportResult(
        distance=distance,
        cost=cost,
        plan=result.plan,
        iterations=result.iterations,
        converged=result.converged,
        reason=result.reason,
        diagnostics={
            **result.diagnostics,
            "p": p,
            "distance_scale": distance_scale,
            "normalized_cost": normalized_cost,
            "normalized_cost_matrix": normalized_costs,
            "cost_matrix": cost_scale * normalized_costs,
        },
    )


def _ott_cost(costs: Any, a: Any, b: Any, epsilon: float) -> Any:
    try:
        from ott.geometry import geometry
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "sinkhorn_divergence requires OTT-JAX; install GeoJAX with the 'ot' extra."
        ) from exc
    geom = geometry.Geometry(cost_matrix=costs, epsilon=epsilon)
    problem = linear_problem.LinearProblem(geom, a=a, b=b)
    output = sinkhorn.Sinkhorn()(problem)
    return output.reg_ot_cost


def sinkhorn_divergence(
    manifold: Any,
    x: Any,
    y: Any,
    *,
    epsilon: float = 0.05,
    p: float = 2.0,
    weights_x: Any | None = None,
    weights_y: Any | None = None,
) -> Any:
    """Return debiased entropic transport divergence through optional OTT-JAX."""
    require_exact_operations(manifold, "sinkhorn_divergence", "dist")
    raw_left = x.values if isinstance(x, ManifoldData) else x
    raw_right = y.values if isinstance(y, ManifoldData) else y
    check = "shape" if tree_contains_tracer((raw_left, raw_right)) else "belongs"
    left = as_manifold_data(manifold, x, check=check)
    right = as_manifold_data(manifold, y, check=check)
    require_unbatched(left, "sinkhorn_divergence")
    require_unbatched(right, "sinkhorn_divergence")
    epsilon = positive_control(epsilon, name="epsilon")
    p = positive_control(p, name="p")
    if p < 1.0:
        raise ValueError("p must be at least 1.")
    a = normalize_weights(left.n_samples, weights_x)
    b = normalize_weights(right.n_samples, weights_y)
    cross = pairwise_distances(manifold, left, right) ** p
    left_cost = pairwise_distances(manifold, left, left) ** p
    right_cost = pairwise_distances(manifold, right, right) ** p
    return (
        _ott_cost(cross, a, b, epsilon)
        - 0.5 * _ott_cost(left_cost, a, a, epsilon)
        - 0.5 * _ott_cost(right_cost, b, b, epsilon)
    )


__all__ = ["empirical_wasserstein_distance", "sinkhorn_divergence"]
