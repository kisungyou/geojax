"""Exact and regularized transport between empirical manifold measures."""

from __future__ import annotations

from collections import deque
from typing import Any

import jax.numpy as jnp

from ._capabilities import require_exact_operations
from ._data import as_manifold_data
from ._geometry import pairwise_distances
from ._results import TransportResult
from ._utils import normalize_weights, require_unbatched


def _complete_tree_basis(plan: Any, costs: Any, tolerance: float) -> Any:
    """Complete positive transport edges to a deterministic spanning-tree basis."""
    rows, columns = plan.shape
    basis = jnp.asarray(plan > tolerance)
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

    positive = [(row, column) for row in range(rows) for column in range(columns) if bool(basis[row, column])]
    rebuilt = jnp.zeros_like(basis)
    for row, column in positive:
        if union(row, rows + column):
            rebuilt = rebuilt.at[row, column].set(True)
    candidates = sorted(
        ((float(costs[row, column]), row, column) for row in range(rows) for column in range(columns)),
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
                    column_potentials[column] = float(costs[index, column]) - float(row_potentials[index])
                    queue.append(("column", column))
        else:
            for row in range(rows):
                if bool(basis[row, index]) and row_potentials[row] is None:
                    row_potentials[row] = float(costs[row, index]) - float(column_potentials[index])
                    queue.append(("row", row))
    if any(value is None for value in row_potentials + column_potentials):
        raise RuntimeError("Transportation basis is disconnected.")
    return jnp.asarray(row_potentials), jnp.asarray(column_potentials)


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
    costs = jnp.asarray(costs, dtype=float)
    positive_rows = jnp.nonzero(a > tolerance, size=a.size, fill_value=-1)[0]
    positive_rows = positive_rows[positive_rows >= 0]
    positive_columns = jnp.nonzero(b > tolerance, size=b.size, fill_value=-1)[0]
    positive_columns = positive_columns[positive_columns >= 0]
    reduced_costs = costs[positive_rows[:, None], positive_columns[None, :]]
    reduced_a = a[positive_rows]
    reduced_b = b[positive_columns]
    plan, basis = _northwest_corner(reduced_a, reduced_b, reduced_costs, tolerance)
    converged = False
    minimum_reduced_cost = -jnp.inf
    row_potentials = column_potentials = None
    for iteration in range(int(max_pivots) + 1):
        row_potentials, column_potentials = _dual_potentials(reduced_costs, basis)
        reduced = reduced_costs - row_potentials[:, None] - column_potentials[None, :]
        reduced = jnp.where(basis, jnp.inf, reduced)
        minimum_reduced_cost = jnp.min(reduced)
        if float(minimum_reduced_cost) >= -tolerance:
            converged = True
            break
        entering_candidates = [
            (row, column)
            for row in range(plan.shape[0])
            for column in range(plan.shape[1])
            if not bool(basis[row, column]) and float(reduced[row, column]) < -tolerance
        ]
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
        plan = jnp.where(jnp.abs(plan) <= tolerance, 0.0, plan)
        basis = basis.at[entering].set(True)
        basis = basis.at[leaving].set(False)
    full_plan = jnp.zeros_like(costs)
    full_plan = full_plan.at[positive_rows[:, None], positive_columns[None, :]].set(plan)
    primal = jnp.sum(full_plan * costs)
    row_residual = jnp.max(jnp.abs(jnp.sum(full_plan, axis=1) - a))
    column_residual = jnp.max(jnp.abs(jnp.sum(full_plan, axis=0) - b))
    dual = jnp.sum(reduced_a * row_potentials) + jnp.sum(reduced_b * column_potentials)
    return TransportResult(
        distance=jnp.nan,
        cost=primal,
        plan=full_plan,
        iterations=iteration,
        converged=converged,
        reason="optimal reduced costs" if converged else "maximum pivots reached",
        diagnostics={
            "row_residual": row_residual,
            "column_residual": column_residual,
            "duality_gap": primal - dual,
            "minimum_reduced_cost": minimum_reduced_cost,
            "basis": basis,
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
    if float(p) < 1.0:
        raise ValueError("p must be at least 1.")
    a = normalize_weights(left.n_samples, weights_x)
    b = normalize_weights(right.n_samples, weights_y)
    distances = pairwise_distances(manifold, left, right)
    costs = distances ** float(p)
    result = _transportation_simplex(
        costs,
        a,
        b,
        tolerance=float(tolerance),
        max_pivots=int(max_pivots),
    )
    distance = jnp.maximum(result.cost, 0.0) ** (1.0 / float(p))
    return TransportResult(
        distance=distance,
        cost=result.cost,
        plan=result.plan,
        iterations=result.iterations,
        converged=result.converged,
        reason=result.reason,
        diagnostics={**result.diagnostics, "p": float(p), "cost_matrix": costs},
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
    left = as_manifold_data(manifold, x, check="shape")
    right = as_manifold_data(manifold, y, check="shape")
    require_unbatched(left, "sinkhorn_divergence")
    require_unbatched(right, "sinkhorn_divergence")
    if epsilon <= 0.0 or p < 1.0:
        raise ValueError("epsilon must be positive and p must be at least 1.")
    a = normalize_weights(left.n_samples, weights_x)
    b = normalize_weights(right.n_samples, weights_y)
    cross = pairwise_distances(manifold, left, right) ** p
    left_cost = pairwise_distances(manifold, left, left) ** p
    right_cost = pairwise_distances(manifold, right, right) ** p
    return _ott_cost(cross, a, b, epsilon) - 0.5 * _ott_cost(
        left_cost, a, a, epsilon
    ) - 0.5 * _ott_cost(right_cost, b, b, epsilon)


__all__ = ["empirical_wasserstein_distance", "sinkhorn_divergence"]
