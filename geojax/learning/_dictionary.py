"""Intrinsic barycentric coding and manifold dictionary learning."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from geojax.geometry import Product
from geojax.optimization import ConjugateGradient, Minimize

from ._capabilities import require_exact_operations
from ._clustering import _initial_indices
from ._data import ManifoldData, as_manifold_data
from ._results import BarycentricCodingResult, DictionaryLearningResult
from ._statistics import frechet_mean
from ._utils import (
    as_key,
    normalize_weights,
    require_unbatched,
    stack_points,
    take_point,
    take_samples,
)


def _project_simplex(vector: Any) -> Any:
    values = jnp.asarray(vector, dtype=float)
    ordered = jnp.sort(values)[::-1]
    cumulative = jnp.cumsum(ordered) - 1.0
    indices = jnp.arange(1, values.size + 1)
    valid = ordered - cumulative / indices > 0.0
    rho = jnp.max(jnp.where(valid, indices, 1))
    threshold = cumulative[rho - 1] / rho
    return jnp.maximum(values - threshold, 0.0)


def _code_one(
    manifold: Any,
    point: Any,
    atoms: ManifoldData,
    *,
    ridge: float,
    maxiter: int,
    tol: float,
) -> tuple[Any, int, bool, Any]:
    logs = manifold.log(point, atoms.values)
    tangent_atoms = [take_point(manifold, logs, index) for index in range(atoms.n_samples)]
    gram = jnp.stack(
        [
            jnp.stack([manifold.inner(point, left, right) for right in tangent_atoms])
            for left in tangent_atoms
        ]
    )
    gram = 0.5 * (gram + gram.T) + float(ridge) * jnp.eye(atoms.n_samples)
    largest = float(jnp.max(jnp.linalg.eigvalsh(gram)))
    step = 1.0 / max(largest, 1e-12)
    code = jnp.full((atoms.n_samples,), 1.0 / atoms.n_samples)
    history = []
    converged = False
    for iteration in range(1, int(maxiter) + 1):
        objective = 0.5 * code @ gram @ code
        history.append(objective)
        candidate = _project_simplex(code - step * (gram @ code))
        if float(jnp.linalg.norm(candidate - code)) <= float(tol):
            code = candidate
            converged = True
            break
        code = candidate
    return code, iteration, converged, jnp.asarray(history)


def geodesic_barycentric_coding(
    manifold: Any,
    data: Any,
    atoms: Any,
    *,
    ridge: float = 1e-6,
    maxiter: int = 200,
    tol: float = 1e-7,
    reconstruction_maxiter: int = 100,
) -> BarycentricCodingResult:
    r"""Code points by simplex weights minimizing a log-map barycentric residual."""
    require_exact_operations(
        manifold,
        "geodesic_barycentric_coding",
        "dist",
        "log",
        "exp",
    )
    adapted = data if isinstance(data, ManifoldData) else as_manifold_data(manifold, data)
    atom_data = atoms if isinstance(atoms, ManifoldData) else as_manifold_data(manifold, atoms)
    require_unbatched(adapted, "geodesic_barycentric_coding")
    require_unbatched(atom_data, "geodesic_barycentric_coding")
    if atom_data.n_samples < 1:
        raise ValueError("atoms must contain at least one point.")
    if ridge < 0.0 or int(maxiter) < 1 or int(reconstruction_maxiter) < 1 or tol < 0.0:
        raise ValueError(
            "ridge and tol must be nonnegative; iteration limits must be positive."
        )
    codes = []
    reconstructions = []
    solver_iterations = []
    histories = []
    all_converged = True
    for index in range(adapted.n_samples):
        point = take_point(manifold, adapted.values, index)
        code, iterations, converged, history = _code_one(
            manifold,
            point,
            atom_data,
            ridge=float(ridge),
            maxiter=int(maxiter),
            tol=float(tol),
        )
        reconstruction = frechet_mean(
            manifold,
            atom_data,
            sample_weight=code,
            initial_point=point,
            maxiter=int(reconstruction_maxiter),
            tol=max(float(tol), 1e-7),
        ).point
        codes.append(code)
        reconstructions.append(reconstruction)
        solver_iterations.append(iterations)
        histories.append(history)
        all_converged = all_converged and converged
    code_matrix = jnp.stack(codes)
    reconstruction_tree = stack_points(manifold, reconstructions)
    errors = manifold.squared_dist(reconstruction_tree, adapted.values)
    objective = jnp.mean(errors + float(ridge) * jnp.sum(code_matrix**2, axis=1))
    return BarycentricCodingResult(
        codes=code_matrix,
        reconstructions=reconstruction_tree,
        objective=objective,
        iterations=max(solver_iterations),
        converged=all_converged,
        reason=(
            "all projected-gradient solves converged"
            if all_converged
            else "at least one code reached the iteration limit"
        ),
        diagnostics={
            "reconstruction_errors": errors,
            "solver_iterations": jnp.asarray(solver_iterations),
            "objective_histories": tuple(histories),
            "ridge": float(ridge),
        },
    )


def _interpolate_atoms(manifold: Any, old: Any, candidate: Any, fraction: float) -> Any:
    old_data = as_manifold_data(manifold, old)
    candidate_data = as_manifold_data(manifold, candidate)
    points = []
    for index in range(old_data.n_samples):
        source = take_point(manifold, old_data.values, index)
        target = take_point(manifold, candidate_data.values, index)
        direction = manifold.log(source, target)
        points.append(manifold.exp(source, manifold.lincomb(source, fraction, direction)))
    return stack_points(manifold, points)


def _weighted_coding_objective(coding: BarycentricCodingResult, weights: Any, ridge: float) -> Any:
    penalties = float(ridge) * jnp.sum(coding.codes**2, axis=1)
    errors = coding.diagnostics["reconstruction_errors"]
    return jnp.sum(weights * (errors + penalties))


def _optimize_atoms(
    manifold: Any,
    data: ManifoldData,
    atoms: Any,
    codes: Any,
    weights: Any,
    *,
    maxiter: int,
    tol: float,
) -> tuple[Any, Any]:
    atom_data = as_manifold_data(manifold, atoms)
    initial_state = tuple(
        take_point(manifold, atom_data.values, index) for index in range(atom_data.n_samples)
    )
    dictionary_manifold = Product(tuple(manifold for _ in range(atom_data.n_samples)))

    def objective(atom_state: Any) -> Any:
        total = jnp.asarray(0.0)
        for sample in range(data.n_samples):
            point = take_point(manifold, data.values, sample)
            terms = []
            for atom, atom_point in enumerate(atom_state):
                terms.extend([codes[sample, atom], manifold.log(point, atom_point)])
            residual = manifold.lincomb(point, *terms)
            total = total + weights[sample] * manifold.inner(point, residual, residual)
        return total

    state, _, history = Minimize(
        M=dictionary_manifold,
        cost=objective,
        x0=initial_state,
        solver=ConjugateGradient(
            maxiter=int(maxiter),
            tolgradnorm=float(tol),
            verbosity=0,
        ),
    ).solve()
    return stack_points(manifold, list(state)), tuple(history)


def manifold_dictionary_learning(
    manifold: Any,
    data: Any,
    *,
    n_atoms: int,
    key: Any | int | None = None,
    initial_atoms: Any | None = None,
    sample_weight: Any | None = None,
    ridge: float = 1e-6,
    maxiter: int = 20,
    coding_maxiter: int = 100,
    center_maxiter: int = 100,
    tol: float = 1e-5,
) -> DictionaryLearningResult:
    """Alternate intrinsic barycentric codes and weighted atom updates."""
    require_exact_operations(
        manifold,
        "manifold_dictionary_learning",
        "dist",
        "log",
        "exp",
    )
    adapted = data if isinstance(data, ManifoldData) else as_manifold_data(manifold, data)
    require_unbatched(adapted, "manifold_dictionary_learning")
    n_atoms = int(n_atoms)
    if not 1 <= n_atoms <= adapted.n_samples:
        raise ValueError("n_atoms must be between 1 and n_samples.")
    if int(maxiter) < 1 or int(coding_maxiter) < 1 or int(center_maxiter) < 1:
        raise ValueError("all iteration limits must be positive.")
    if ridge < 0.0 or tol < 0.0:
        raise ValueError("ridge and tol must be nonnegative.")
    weights = normalize_weights(adapted.n_samples, sample_weight)
    if initial_atoms is None:
        initialization_key = as_key(key, "manifold_dictionary_learning")
        indices = _initial_indices(
            manifold,
            adapted,
            n_atoms,
            initialization_key,
            "kmeans++",
            weights,
        )
        atoms = take_samples(manifold, adapted.values, indices)
    else:
        atom_data = as_manifold_data(manifold, initial_atoms)
        require_unbatched(atom_data, "manifold_dictionary_learning")
        if atom_data.n_samples != n_atoms:
            raise ValueError("initial_atoms must contain exactly n_atoms points.")
        atoms = atom_data.values
        indices = None

    coding = geodesic_barycentric_coding(
        manifold,
        adapted,
        atoms,
        ridge=float(ridge),
        maxiter=int(coding_maxiter),
        tol=max(float(tol), 1e-7),
        reconstruction_maxiter=int(center_maxiter),
    )
    objective = _weighted_coding_objective(coding, weights, float(ridge))
    objective_history = [objective]
    accepted_steps = []
    atom_histories = []
    converged = False
    for iteration in range(1, int(maxiter) + 1):
        candidate_atoms, optimizer_history = _optimize_atoms(
            manifold,
            adapted,
            atoms,
            coding.codes,
            weights,
            maxiter=int(center_maxiter),
            tol=max(float(tol), 1e-7),
        )
        accepted = None
        accepted_fraction = 0.0
        for fraction in (1.0, 0.5, 0.25, 0.125):
            trial_atoms = _interpolate_atoms(manifold, atoms, candidate_atoms, fraction)
            trial = geodesic_barycentric_coding(
                manifold,
                adapted,
                trial_atoms,
                ridge=float(ridge),
                maxiter=int(coding_maxiter),
                tol=max(float(tol), 1e-7),
                reconstruction_maxiter=int(center_maxiter),
            )
            trial_objective = _weighted_coding_objective(trial, weights, float(ridge))
            if float(trial_objective) <= float(objective) + 1e-10:
                accepted = (trial_atoms, trial, trial_objective)
                accepted_fraction = fraction
                break
        accepted_steps.append(accepted_fraction)
        if accepted is None:
            converged = True
            reason = "no decreasing atom update found"
            break
        previous = float(objective)
        atoms, coding, objective = accepted
        atom_histories.append(optimizer_history)
        objective_history.append(objective)
        if abs(previous - float(objective)) <= float(tol) * max(1.0, previous):
            converged = True
            reason = "objective tolerance reached"
            break
    else:
        reason = "maximum iterations reached"
    return DictionaryLearningResult(
        atoms=atoms,
        codes=coding.codes,
        reconstructions=coding.reconstructions,
        objective=objective,
        iterations=iteration,
        converged=converged,
        reason=reason,
        diagnostics={
            "objective_history": jnp.asarray(objective_history),
            "accepted_step_fractions": jnp.asarray(accepted_steps),
            "initial_indices": indices,
            "weights": weights,
            "coding_result": coding,
            "atom_optimizer_histories": tuple(atom_histories),
        },
    )


__all__ = ["geodesic_barycentric_coding", "manifold_dictionary_learning"]
