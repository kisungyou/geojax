"""Intrinsic barycentric coding and manifold dictionary learning."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from geojax.geometry import Product
from geojax.optimization import ConjugateGradient, Minimize

from ._capabilities import require_exact_operations
from ._clustering import _initial_indices
from ._data import ManifoldData, as_manifold_data
from ._results import BarycentricCodingResult, DictionaryLearningResult
from ._statistics import _gradient_tolerances, frechet_mean
from ._utils import (
    as_key,
    as_real_array,
    integer_control,
    nonnegative_control,
    normalize_weights,
    require_unbatched,
    stack_points,
    take_point,
    take_samples,
    tree_all_finite,
)


def _project_simplex(vector: Any) -> Any:
    values = as_real_array(vector, name="simplex vector")
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
) -> tuple[Any, int, bool, Any, Any, float]:
    logs = manifold.log(point, atoms.values)
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(logs)):
        raise ValueError("geodesic_barycentric_coding encountered an undefined atom logarithm.")
    tangent_atoms = [take_point(manifold, logs, index) for index in range(atoms.n_samples)]
    gram = jnp.stack(
        [
            jnp.stack([manifold.inner(point, left, right) for right in tangent_atoms])
            for left in tangent_atoms
        ]
    )
    gram = 0.5 * (gram + gram.T) + ridge * jnp.eye(atoms.n_samples)
    if not bool(jnp.all(jnp.isfinite(gram))):
        raise FloatingPointError("Barycentric tangent Gram matrix is nonfinite.")
    eigenvalues = jnp.linalg.eigvalsh(gram)
    spectral_scale = max(
        float(jnp.max(jnp.abs(eigenvalues))),
        float(jnp.finfo(gram.dtype).tiny),
    )
    backward_tolerance = 100.0 * float(jnp.finfo(gram.dtype).eps) * atoms.n_samples * spectral_scale
    if float(jnp.min(eigenvalues)) < -backward_tolerance:
        raise FloatingPointError("Barycentric tangent Gram matrix is not positive semidefinite.")
    largest = max(float(jnp.max(eigenvalues)), 0.0)
    normalized_gram = gram / largest if largest > 0.0 else jnp.zeros_like(gram)
    step = 1.0
    effective_tol = max(
        tol,
        100.0 * float(jnp.finfo(gram.dtype).eps) * atoms.n_samples,
    )
    code = jnp.full((atoms.n_samples,), 1.0 / atoms.n_samples)
    history = []
    converged = False
    for iteration in range(1, maxiter + 1):
        objective = 0.5 * code @ gram @ code
        history.append(objective)
        candidate = _project_simplex(code - step * (normalized_gram @ code))
        stationarity = jnp.linalg.norm(candidate - code) / step
        if float(stationarity) <= effective_tol:
            code = candidate
            converged = True
            break
        code = candidate
    projected = _project_simplex(code - step * (normalized_gram @ code))
    stationarity = jnp.linalg.norm(projected - code) / step
    final_objective = 0.5 * code @ gram @ code
    if not history or float(final_objective) != float(history[-1]):
        history.append(final_objective)
    return (
        code,
        iteration,
        converged,
        jnp.asarray(history),
        stationarity,
        effective_tol,
    )


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
    adapted = as_manifold_data(manifold, data)
    atom_data = as_manifold_data(manifold, atoms)
    require_unbatched(adapted, "geodesic_barycentric_coding")
    require_unbatched(atom_data, "geodesic_barycentric_coding")
    if atom_data.n_samples < 1:
        raise ValueError("atoms must contain at least one point.")
    ridge = nonnegative_control(ridge, name="ridge")
    tol = nonnegative_control(tol, name="tol")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    reconstruction_maxiter = integer_control(
        reconstruction_maxiter,
        name="reconstruction_maxiter",
        minimum=1,
    )
    codes = []
    reconstructions = []
    solver_iterations = []
    histories = []
    coding_objectives = []
    stationarity_residuals = []
    effective_stationarity_tolerances = []
    reconstruction_fits = []
    all_converged = True
    for index in range(adapted.n_samples):
        point = take_point(manifold, adapted.values, index)
        code, iterations, converged, history, stationarity, effective_tol = _code_one(
            manifold,
            point,
            atom_data,
            ridge=ridge,
            maxiter=maxiter,
            tol=tol,
        )
        reconstruction_fit = frechet_mean(
            manifold,
            atom_data,
            sample_weight=code,
            initial_point=point,
            maxiter=reconstruction_maxiter,
            tol=tol,
        )
        reconstruction = reconstruction_fit.point
        codes.append(code)
        reconstructions.append(reconstruction)
        solver_iterations.append(iterations)
        histories.append(history)
        coding_objectives.append(2.0 * history[-1])
        stationarity_residuals.append(stationarity)
        effective_stationarity_tolerances.append(effective_tol)
        reconstruction_fits.append(reconstruction_fit)
        all_converged = all_converged and converged and reconstruction_fit.converged
    code_matrix = jnp.stack(codes)
    reconstruction_tree = stack_points(manifold, reconstructions)
    errors = manifold.squared_dist(reconstruction_tree, adapted.values)
    coding_objective_array = jnp.asarray(coding_objectives)
    objective = jnp.mean(coding_objective_array)
    if (
        not bool(jnp.all(jnp.isfinite(code_matrix)))
        or not tree_all_finite(reconstruction_tree)
        or not bool(jnp.isfinite(objective))
    ):
        raise FloatingPointError("geodesic_barycentric_coding produced a nonfinite result.")
    return BarycentricCodingResult(
        codes=code_matrix,
        reconstructions=reconstruction_tree,
        objective=objective,
        iterations=max(solver_iterations),
        converged=all_converged,
        reason=(
            "all code and reconstruction solves converged"
            if all_converged
            else "at least one code or reconstruction solve was incomplete"
        ),
        diagnostics={
            "reconstruction_errors": errors,
            "solver_iterations": jnp.asarray(solver_iterations),
            "objective_histories": tuple(histories),
            "coding_objectives": coding_objective_array,
            "ridge": ridge,
            "code_stationarity_residuals": jnp.asarray(stationarity_residuals),
            "effective_code_stationarity_tolerances": jnp.asarray(
                effective_stationarity_tolerances
            ),
            "reconstruction_fits": tuple(reconstruction_fits),
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
        if not tree_all_finite(direction):
            raise FloatingPointError(
                "Dictionary atom interpolation encountered an undefined logarithm."
            )
        points.append(manifold.exp(source, manifold.lincomb(source, fraction, direction)))
    result = stack_points(manifold, points)
    if not tree_all_finite(result):
        raise FloatingPointError("Dictionary atom interpolation left the exponential-map domain.")
    return result


def _weighted_coding_objective(coding: BarycentricCodingResult, weights: Any, ridge: float) -> Any:
    del ridge
    return jnp.sum(weights * coding.diagnostics["coding_objectives"])


def _optimize_atoms(
    manifold: Any,
    data: ManifoldData,
    atoms: Any,
    codes: Any,
    weights: Any,
    *,
    maxiter: int,
    tol: float,
) -> tuple[Any, Any, bool, float]:
    atom_data = as_manifold_data(manifold, atoms)
    initial_state = tuple(
        take_point(manifold, atom_data.values, index) for index in range(atom_data.n_samples)
    )
    # A Product point has the pytree of ``manifold.factors`` rather than a
    # Product object as a leaf.  Replicate that factor pytree so the optimizer
    # geometry exactly matches the tuple of atom-point pytrees in
    # ``initial_state``.  Non-Product manifolds remain one leaf per atom.
    atom_factor = manifold.factors if isinstance(manifold, Product) else manifold
    dictionary_manifold = Product(tuple(atom_factor for _ in range(atom_data.n_samples)))

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

    initial_objective = objective(initial_state)
    _, effective_tol = _gradient_tolerances(
        initial_state,
        jnp.sqrt(jnp.maximum(initial_objective, 0.0)),
        tol,
    )
    state, _, history = Minimize(
        M=dictionary_manifold,
        cost=objective,
        x0=initial_state,
        solver=ConjugateGradient(
            maxiter=maxiter,
            tolgradnorm=effective_tol,
            verbosity=0,
        ),
    ).solve()
    result = stack_points(manifold, list(state))
    if not tree_all_finite(result):
        raise FloatingPointError("Dictionary atom optimization produced nonfinite atoms.")
    return (
        result,
        tuple(history),
        bool(history[-1].gradnorm <= effective_tol),
        effective_tol,
    )


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
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, "manifold_dictionary_learning")
    n_atoms = integer_control(n_atoms, name="n_atoms")
    if not 1 <= n_atoms <= adapted.n_samples:
        raise ValueError("n_atoms must be between 1 and n_samples.")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    coding_maxiter = integer_control(coding_maxiter, name="coding_maxiter", minimum=1)
    center_maxiter = integer_control(center_maxiter, name="center_maxiter", minimum=1)
    ridge = nonnegative_control(ridge, name="ridge")
    tol = nonnegative_control(tol, name="tol")
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
        ridge=ridge,
        maxiter=coding_maxiter,
        tol=tol,
        reconstruction_maxiter=center_maxiter,
    )
    objective = _weighted_coding_objective(coding, weights, ridge)
    objective_history = [objective]
    accepted_steps = []
    atom_histories = []
    atom_optimizer_converged = []
    atom_optimizer_tolerances = []
    converged = False
    for iteration in range(1, maxiter + 1):
        (
            candidate_atoms,
            optimizer_history,
            optimizer_converged,
            optimizer_tolerance,
        ) = _optimize_atoms(
            manifold,
            adapted,
            atoms,
            coding.codes,
            weights,
            maxiter=center_maxiter,
            tol=tol,
        )
        atom_histories.append(optimizer_history)
        atom_optimizer_converged.append(optimizer_converged)
        atom_optimizer_tolerances.append(optimizer_tolerance)
        accepted = None
        accepted_fraction = 0.0
        for fraction in (1.0, 0.5, 0.25, 0.125):
            trial_atoms = _interpolate_atoms(manifold, atoms, candidate_atoms, fraction)
            trial = geodesic_barycentric_coding(
                manifold,
                adapted,
                trial_atoms,
                ridge=ridge,
                maxiter=coding_maxiter,
                tol=tol,
                reconstruction_maxiter=center_maxiter,
            )
            trial_objective = _weighted_coding_objective(trial, weights, ridge)
            acceptance_tolerance = (
                100.0
                * float(jnp.finfo(jnp.asarray(objective).dtype).eps)
                * max(
                    abs(float(objective)),
                    float(jnp.finfo(jnp.asarray(objective).dtype).tiny),
                )
            )
            if bool(jnp.isfinite(trial_objective)) and float(trial_objective) <= (
                float(objective) + acceptance_tolerance
            ):
                accepted = (trial_atoms, trial, trial_objective)
                accepted_fraction = fraction
                break
        accepted_steps.append(accepted_fraction)
        if accepted is None:
            converged = False
            reason = "atom update stalled without a decreasing accepted step"
            break
        previous = float(objective)
        atoms, coding, objective = accepted
        objective_history.append(objective)
        objective_dtype = jnp.asarray(objective).dtype
        objective_scale = max(
            abs(previous),
            abs(float(objective)),
            float(jnp.finfo(objective_dtype).tiny),
        )
        effective_relative_tol = max(
            tol,
            100.0 * float(jnp.finfo(objective_dtype).eps),
        )
        if abs(previous - float(objective)) <= effective_relative_tol * objective_scale:
            converged = bool(coding.converged and optimizer_converged)
            reason = (
                "objective tolerance reached"
                if converged
                else "objective tolerance reached but an inner solve was incomplete"
            )
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
            "atom_optimizer_converged": jnp.asarray(atom_optimizer_converged),
            "atom_optimizer_tolerances": jnp.asarray(atom_optimizer_tolerances),
        },
    )


__all__ = ["geodesic_barycentric_coding", "manifold_dictionary_learning"]
