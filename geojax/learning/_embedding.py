"""Dense dimension-reduction methods for manifold-valued observations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp

from geojax.geometry import Euclidean
from geojax.geometry._numerics import stable_norm
from geojax.optimization import LBFGS, Minimize

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import EmbeddingResult
from ._statistics import _gradient_tolerances, frechet_mean
from ._utils import (
    as_real_array,
    as_key,
    deterministic_sign_columns,
    flatten_geometry_values,
    integer_control,
    nonnegative_control,
    positive_control,
    require_unbatched,
    stack_points,
    take_point,
    tree_all_finite,
    unflatten_geometry,
    weighted_tangent_sum,
)


def _prepare(manifold: Any, data: Any, method: str) -> ManifoldData:
    adapted = as_manifold_data(manifold, data)
    require_unbatched(adapted, method)
    return adapted


def _descending_eigh(matrix: Any) -> tuple[Any, Any]:
    values, vectors = jnp.linalg.eigh(0.5 * (matrix + matrix.T))
    order = jnp.argsort(values)[::-1]
    return values[order], deterministic_sign_columns(vectors[:, order])


def _validate_n_components(n_components: int, n_samples: int) -> int:
    count = integer_control(n_components, name="n_components")
    if not 1 <= count <= n_samples:
        raise ValueError("n_components must be between 1 and n_samples.")
    return count


def _mds_from_distances(distances: Any, n_components: int) -> tuple[Any, dict[str, Any]]:
    distances = as_real_array(distances, name="distances")
    if distances.ndim != 2:
        raise ValueError("distances must be a two-dimensional square matrix.")
    n_samples = distances.shape[0]
    if n_samples < 1 or distances.shape != (n_samples, n_samples):
        raise ValueError("distances must be square.")
    n_components = _validate_n_components(n_components, n_samples)
    if not bool(jnp.all(jnp.isfinite(distances))):
        raise ValueError("distances must contain only finite values.")
    distance_scale = float(jnp.max(jnp.abs(distances)))
    safe_scale = distance_scale if distance_scale > 0.0 else 1.0
    normalized = distances / safe_scale
    tolerance = 100.0 * float(jnp.finfo(distances.dtype).eps) * n_samples
    if float(jnp.min(normalized)) < -tolerance:
        raise ValueError("distances must be nonnegative.")
    if float(jnp.max(jnp.abs(normalized - normalized.T))) > tolerance:
        raise ValueError("distances must be symmetric.")
    if float(jnp.max(jnp.abs(jnp.diag(normalized)))) > tolerance:
        raise ValueError("distances must have a zero diagonal.")
    normalized = jnp.maximum(0.5 * (normalized + normalized.T), 0.0)
    normalized = normalized.at[jnp.diag_indices(n_samples)].set(0.0)
    centering = jnp.eye(n_samples) - jnp.ones((n_samples, n_samples)) / n_samples
    normalized_gram = -0.5 * centering @ (normalized**2) @ centering
    normalized_eigenvalues, eigenvectors = _descending_eigh(normalized_gram)
    positive = jnp.maximum(normalized_eigenvalues[:n_components], 0.0)
    normalized_coordinates = eigenvectors[:, :n_components] * jnp.sqrt(positive)[None, :]
    coordinates = safe_scale * normalized_coordinates
    reconstructed_normalized = stable_norm(
        normalized_coordinates[:, None, :] - normalized_coordinates[None, :, :],
        axis=-1,
    )
    reconstructed = safe_scale * reconstructed_normalized
    denominator = jnp.sum(normalized**2)
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    stress = jnp.where(
        denominator > 0.0,
        jnp.sqrt(jnp.sum((normalized - reconstructed_normalized) ** 2) / safe_denominator),
        0.0,
    )
    limits = jnp.finfo(distances.dtype)
    if safe_scale > math.sqrt(float(limits.max)):
        raise FloatingPointError(
            "The squared distance scale is not representable in the input dtype; "
            "rescale the observations before classical scaling."
        )
    squared_scale = jnp.asarray(safe_scale, dtype=distances.dtype) ** 2
    eigenvalues = squared_scale * normalized_eigenvalues
    gram = squared_scale * normalized_gram
    return coordinates, {
        "eigenvalues": eigenvalues,
        "normalized_eigenvalues": normalized_eigenvalues,
        "gram_matrix": gram,
        "normalized_gram_matrix": normalized_gram,
        "negative_eigenvalue_mass": jnp.sum(jnp.abs(jnp.minimum(eigenvalues, 0.0))),
        "stress": stress,
        "reconstructed_distances": reconstructed,
        "distance_scale": jnp.asarray(distance_scale, dtype=distances.dtype),
    }


def classical_mds(
    manifold: Any,
    data: Any,
    *,
    n_components: int = 2,
) -> EmbeddingResult:
    """Classical scaling of the exact manifold distance matrix."""
    require_exact_operations(manifold, "classical_mds", "dist")
    adapted = _prepare(manifold, data, "classical_mds")
    distances = pairwise_distances(manifold, adapted)
    coordinates, diagnostics = _mds_from_distances(distances, n_components)
    diagnostics["pairwise_distances"] = distances
    return EmbeddingResult(
        coordinates=coordinates,
        objective=diagnostics["stress"],
        iterations=1,
        converged=True,
        reason="eigendecomposition completed",
        diagnostics=diagnostics,
    )


def _expand_tangent_samples(manifold: Any, tangents: Any, *, left: bool) -> Any:
    factors, leaves = flatten_geometry_values(manifold, tangents, name="tangents")
    out = []
    for factor, leaf in zip(factors, leaves):
        axis = -(len(factor.shape) + (1 if left else 2))
        out.append(jnp.expand_dims(leaf, axis=axis))
    return unflatten_geometry(manifold, out)


@dataclass(frozen=True)
class _PGAModel:
    manifold: Any
    mean: Any
    components: Any
    tangent_mean: Any
    eigenvalues: Any
    n_components: int

    def transform(self, data: Any) -> Any:
        adapted = as_manifold_data(self.manifold, data)
        require_unbatched(adapted, "PGA.transform")
        logs = self.manifold.log(self.mean, adapted.values)
        if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(logs)):
            raise ValueError("PGA transform encountered an undefined logarithm.")
        centered_logs = self.manifold.lincomb(
            self.mean,
            1.0,
            logs,
            -1.0,
            self.tangent_mean,
        )
        left = _expand_tangent_samples(self.manifold, centered_logs, left=True)
        right = _expand_tangent_samples(self.manifold, self.components, left=False)
        return self.manifold.inner(self.mean, left, right)

    def inverse_transform(self, coordinates: Any) -> Any:
        coordinates = jnp.asarray(coordinates)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.n_components:
            raise ValueError(
                "coordinates must have shape (n_samples, n_components); "
                f"expected trailing dimension {self.n_components}."
            )
        if not bool(jnp.all(jnp.isfinite(coordinates))):
            raise ValueError("coordinates must contain only finite values.")
        points = []
        for row in coordinates:
            component_tangent = weighted_tangent_sum(
                self.manifold,
                self.components,
                row,
            )
            tangent = self.manifold.lincomb(
                self.mean,
                1.0,
                self.tangent_mean,
                1.0,
                component_tangent,
            )
            points.append(self.manifold.exp(self.mean, tangent))
        result = stack_points(self.manifold, points)
        if not tree_all_finite(result):
            raise FloatingPointError(
                "PGA inverse_transform left the certified exponential-map domain."
            )
        return result


def principal_geodesic_analysis(
    manifold: Any,
    data: Any,
    *,
    n_components: int = 2,
    mean: Any | None = None,
    maxiter: int = 200,
    tol: float = 1e-7,
) -> EmbeddingResult:
    """Perform tangent PCA using the Riemannian metric at a Fréchet mean."""
    require_exact_operations(manifold, "principal_geodesic_analysis", "dist", "log", "exp")
    adapted = _prepare(manifold, data, "principal_geodesic_analysis")
    n_components = integer_control(n_components, name="n_components")
    if not 1 <= n_components <= min(adapted.n_samples, manifold.dim):
        raise ValueError("n_components exceeds sample count or intrinsic dimension.")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    if mean is None:
        mean_result = frechet_mean(manifold, adapted, maxiter=maxiter, tol=tol)
        center = mean_result.point
    else:
        mean_result = None
        mean_data = as_manifold_data(manifold, stack_points(manifold, [mean]))
        center = take_point(manifold, mean_data.values, 0)
    logs = manifold.log(center, adapted.values)
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(logs)):
        raise ValueError(
            "principal_geodesic_analysis encountered an undefined logarithm; "
            "the data may meet the cut locus of the selected mean."
        )
    tangent_mean = weighted_tangent_sum(
        manifold,
        logs,
        jnp.full((adapted.n_samples,), 1.0 / adapted.n_samples),
    )
    centered_logs = manifold.lincomb(
        center,
        1.0,
        logs,
        -1.0,
        tangent_mean,
    )
    left = _expand_tangent_samples(manifold, centered_logs, left=True)
    right = _expand_tangent_samples(manifold, centered_logs, left=False)
    gram = manifold.inner(center, left, right) / adapted.n_samples
    if not bool(jnp.all(jnp.isfinite(gram))):
        raise FloatingPointError("PGA produced a nonfinite metric Gram matrix.")
    eigenvalues, eigenvectors = _descending_eigh(gram)
    spectral_scale = max(
        float(jnp.max(jnp.abs(eigenvalues))),
        float(jnp.finfo(eigenvalues.dtype).tiny),
    )
    backward_error = (
        100.0
        * float(jnp.finfo(eigenvalues.dtype).eps)
        * max(adapted.n_samples, int(manifold.dim))
        * spectral_scale
    )
    if float(jnp.min(eigenvalues)) < -backward_error:
        raise FloatingPointError(
            "PGA metric Gram matrix is not positive semidefinite within "
            "floating-point backward error."
        )
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    largest = float(eigenvalues[0])
    rank_tolerance = max(
        float(jnp.finfo(eigenvalues.dtype).eps)
        * max(adapted.n_samples, int(manifold.dim))
        * largest,
        backward_error,
    )
    numerical_rank = int(jnp.sum(eigenvalues > rank_tolerance)) if largest > 0.0 else 0
    if n_components > numerical_rank:
        raise ValueError(
            f"n_components={n_components} exceeds the tangent-data numerical rank {numerical_rank}."
        )
    positive = eigenvalues[:n_components]
    components = []
    for component in range(n_components):
        coefficient = eigenvectors[:, component] / jnp.sqrt(positive[component] * adapted.n_samples)
        components.append(weighted_tangent_sum(manifold, centered_logs, coefficient))
    component_tree = stack_points(manifold, components)
    coordinates = eigenvectors[:, :n_components] * jnp.sqrt(
        jnp.maximum(eigenvalues[:n_components] * adapted.n_samples, 0.0)
    )
    model = _PGAModel(
        manifold,
        center,
        component_tree,
        tangent_mean,
        eigenvalues,
        n_components,
    )
    explained = positive / jnp.sum(eigenvalues)
    center_converged = mean_result is None or mean_result.converged
    return EmbeddingResult(
        coordinates=coordinates,
        objective=1.0 - jnp.sum(explained),
        iterations=1,
        converged=center_converged,
        reason=(
            "metric covariance eigendecomposition completed"
            if center_converged
            else "eigendecomposition completed after an unconverged mean fit"
        ),
        model=model,
        diagnostics={
            "mean": center,
            "components": component_tree,
            "tangent_mean": tangent_mean,
            "eigenvalues": eigenvalues,
            "explained_variance_ratio": explained,
            "gram_matrix": gram,
            "numerical_rank": numerical_rank,
            "rank_tolerance": rank_tolerance,
            "mean_result": mean_result,
            "gram_backward_error": backward_error,
        },
    )


@dataclass(frozen=True)
class _KernelPCAModel:
    manifold: Any
    training_data: ManifoldData
    eigenvectors: Any
    eigenvalues: Any
    training_column_mean: Any
    training_global_mean: Any
    bandwidth: float
    kernel: Callable[[Any, float], Any] | None

    def transform(self, data: Any) -> Any:
        queries = as_manifold_data(self.manifold, data)
        require_unbatched(queries, "KernelPCA.transform")
        distances = pairwise_distances(self.manifold, queries, self.training_data)
        matrix = _distance_kernel(distances, self.bandwidth, self.kernel)
        expected = (queries.n_samples, self.training_data.n_samples)
        if jnp.iscomplexobj(matrix):
            raise ValueError("kernel must return a real-valued query matrix.")
        if matrix.shape != expected or not bool(jnp.all(jnp.isfinite(matrix))):
            raise ValueError(
                f"kernel must return a finite query-by-training matrix with shape {expected}."
            )
        centered = (
            matrix
            - self.training_column_mean[None, :]
            - jnp.mean(matrix, axis=1, keepdims=True)
            + self.training_global_mean
        )
        return centered @ self.eigenvectors / jnp.sqrt(self.eigenvalues)


def _distance_kernel(distances: Any, bandwidth: float, kernel: Callable | None) -> Any:
    return (
        jnp.exp(-0.5 * (distances / bandwidth) ** 2)
        if kernel is None
        else jnp.asarray(kernel(distances, bandwidth))
    )


def kernel_pca(
    manifold: Any,
    data: Any,
    *,
    n_components: int = 2,
    bandwidth: float | None = None,
    kernel: Callable[[Any, float], Any] | None = None,
    allow_indefinite: bool = False,
) -> EmbeddingResult:
    """Kernel PCA using an RBF manifold-distance kernel or user callable.

    A genuine kernel-PCA covariance operator requires a positive-semidefinite
    centered Gram matrix. Set ``allow_indefinite=True`` to request the explicit
    positive-spectral-part approximation for an indefinite similarity.
    """
    require_exact_operations(manifold, "kernel_pca", "dist")
    adapted = _prepare(manifold, data, "kernel_pca")
    n_components = _validate_n_components(n_components, adapted.n_samples)
    if kernel is not None and not callable(kernel):
        raise TypeError("kernel must be callable when supplied.")
    if not isinstance(allow_indefinite, bool):
        raise TypeError("allow_indefinite must be a boolean.")
    distances = pairwise_distances(manifold, adapted)
    positive = distances[distances > 0.0]
    scale = float(jnp.median(positive)) if bandwidth is None and positive.size else 1.0
    scale = float(bandwidth) if bandwidth is not None else scale
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("bandwidth must be positive and finite.")
    matrix = _distance_kernel(distances, scale, kernel)
    if jnp.iscomplexobj(matrix):
        raise ValueError("kernel must return a real-valued matrix.")
    if matrix.shape != distances.shape:
        raise ValueError("kernel must return a square sample kernel matrix.")
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError("kernel must return only finite values.")
    asymmetry = jnp.linalg.norm(matrix - matrix.T)
    symmetry_scale = jnp.maximum(jnp.linalg.norm(matrix), jnp.finfo(matrix.dtype).tiny)
    symmetry_tolerance = 100.0 * adapted.n_samples * jnp.finfo(matrix.dtype).eps * symmetry_scale
    if float(asymmetry) > float(symmetry_tolerance):
        raise ValueError("kernel must return a symmetric sample kernel matrix.")
    matrix = 0.5 * (matrix + matrix.T)
    row_mean = jnp.mean(matrix, axis=1, keepdims=True)
    column_mean = jnp.mean(matrix, axis=0)
    global_mean = jnp.mean(matrix)
    centered = matrix - row_mean - column_mean[None, :] + global_mean
    eigenvalues, eigenvectors = _descending_eigh(centered)
    spectral_scale = max(
        float(jnp.max(jnp.abs(eigenvalues))),
        float(jnp.finfo(eigenvalues.dtype).tiny),
    )
    backward_error = (
        100.0 * float(jnp.finfo(eigenvalues.dtype).eps) * adapted.n_samples * spectral_scale
    )
    minimum_eigenvalue = float(jnp.min(eigenvalues))
    if minimum_eigenvalue < -backward_error and not allow_indefinite:
        raise ValueError(
            "The centered kernel matrix is not positive semidefinite; supply a "
            "valid kernel or set allow_indefinite=True to retain only its positive "
            "spectral part."
        )
    largest = max(float(eigenvalues[0]), 0.0)
    rank_tolerance = max(
        float(jnp.finfo(eigenvalues.dtype).eps) * adapted.n_samples * largest,
        backward_error,
    )
    numerical_rank = int(jnp.sum(eigenvalues > rank_tolerance)) if largest > 0.0 else 0
    if n_components > numerical_rank:
        raise ValueError(
            f"n_components={n_components} exceeds the centered-kernel numerical rank "
            f"{numerical_rank}."
        )
    selected_values = eigenvalues[:n_components]
    selected_vectors = eigenvectors[:, :n_components]
    coordinates = selected_vectors * jnp.sqrt(selected_values)[None, :]
    model = _KernelPCAModel(
        manifold,
        adapted,
        selected_vectors,
        selected_values,
        column_mean,
        global_mean,
        scale,
        kernel,
    )
    return EmbeddingResult(
        coordinates=coordinates,
        objective=-jnp.sum(selected_values),
        iterations=1,
        converged=True,
        reason="centered-kernel eigendecomposition completed",
        model=model,
        diagnostics={
            "eigenvalues": eigenvalues,
            "negative_eigenvalue_mass": jnp.sum(jnp.abs(jnp.minimum(eigenvalues, 0.0))),
            "kernel_matrix": matrix,
            "centered_kernel": centered,
            "bandwidth": scale,
            "numerical_rank": numerical_rank,
            "rank_tolerance": rank_tolerance,
            "psd_backward_error": backward_error,
            "allow_indefinite": allow_indefinite,
        },
    )


def _floyd_warshall(graph: Any) -> Any:
    distances = jnp.asarray(graph)
    for pivot in range(distances.shape[0]):
        distances = jnp.minimum(distances, distances[:, pivot, None] + distances[pivot, None, :])
    return distances


def isomap(
    manifold: Any,
    data: Any,
    *,
    n_components: int = 2,
    n_neighbors: int = 7,
    mutual: bool = True,
    disconnected: str = "error",
) -> EmbeddingResult:
    """Isomap with a dense exact-distance neighbor graph."""
    require_exact_operations(manifold, "isomap", "dist")
    adapted = _prepare(manifold, data, "isomap")
    n_neighbors = integer_control(n_neighbors, name="n_neighbors")
    if not 1 <= n_neighbors < adapted.n_samples:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1.")
    if not isinstance(mutual, bool):
        raise TypeError("mutual must be a boolean.")
    if not isinstance(disconnected, str) or disconnected not in {
        "error",
        "largest_component",
        "max_finite",
    }:
        raise ValueError("disconnected must be 'error', 'largest_component', or 'max_finite'.")
    distances = pairwise_distances(manifold, adapted)
    masked = jnp.where(jnp.eye(adapted.n_samples, dtype=bool), jnp.inf, distances)
    neighbors = jnp.argsort(masked, axis=1)[:, :n_neighbors]
    directed = jnp.zeros_like(distances, dtype=bool)
    directed = directed.at[jnp.arange(adapted.n_samples)[:, None], neighbors].set(True)
    adjacency = directed & directed.T if mutual else directed | directed.T
    graph = jnp.where(adjacency, distances, jnp.inf)
    graph = graph.at[jnp.diag_indices(adapted.n_samples)].set(0.0)
    geodesic = _floyd_warshall(graph)
    finite = jnp.isfinite(geodesic)
    selected_indices = jnp.arange(adapted.n_samples)
    if not bool(jnp.all(finite)):
        if disconnected == "error":
            raise ValueError("The k-nearest-neighbor graph is disconnected.")
        if disconnected == "largest_component":
            component_sizes = jnp.sum(finite, axis=1)
            seed = int(jnp.argmax(component_sizes))
            selected_indices = jnp.nonzero(finite[seed], size=adapted.n_samples, fill_value=-1)[0]
            selected_indices = selected_indices[selected_indices >= 0]
            geodesic = geodesic[selected_indices[:, None], selected_indices[None, :]]
        else:
            maximum = jnp.max(jnp.where(finite, geodesic, 0.0))
            replacement = jnp.where(maximum > 0.0, 2.0 * maximum, 1.0)
            geodesic = jnp.where(finite, geodesic, replacement)
    coordinates, diagnostics = _mds_from_distances(geodesic, n_components)
    diagnostics.update(
        {
            "ambient_distances": distances,
            "graph_distances": geodesic,
            "adjacency": adjacency,
            "selected_indices": selected_indices,
        }
    )
    return EmbeddingResult(
        coordinates=coordinates,
        objective=diagnostics["stress"],
        iterations=adapted.n_samples,
        converged=True,
        reason="all-pairs graph distances computed",
        diagnostics=diagnostics,
    )


def sammon_mapping(
    manifold: Any,
    data: Any,
    *,
    n_components: int = 2,
    maxiter: int = 300,
    tol: float = 1e-7,
) -> EmbeddingResult:
    """Optimize Sammon stress from a classical-MDS initialization."""
    require_exact_operations(manifold, "sammon_mapping", "dist")
    adapted = _prepare(manifold, data, "sammon_mapping")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    tol = nonnegative_control(tol, name="tol")
    distances = pairwise_distances(manifold, adapted)
    initial, _ = _mds_from_distances(distances, n_components)
    left_indices, right_indices = jnp.triu_indices(adapted.n_samples, k=1)
    target_distances = distances[left_indices, right_indices]
    distance_scale = max(
        float(jnp.max(target_distances)),
        float(jnp.finfo(distances.dtype).tiny),
    )
    duplicate_tolerance = 64.0 * float(jnp.finfo(distances.dtype).eps) * distance_scale
    if bool(jnp.any(target_distances <= duplicate_tolerance)):
        raise ValueError(
            "Sammon mapping requires distinct observations with positive pairwise distances."
        )
    normalized_targets = target_distances / distance_scale
    normalized_initial = initial / distance_scale
    normalization = jnp.sum(normalized_targets)

    def objective(coordinates: Any) -> Any:
        differences = coordinates[left_indices] - coordinates[right_indices]
        squared = jnp.sum(differences * differences, axis=-1)
        embedded = jnp.sqrt(jnp.where(squared > 0.0, squared, 1.0))
        embedded = jnp.where(squared > 0.0, embedded, 0.0)
        residual = (normalized_targets - embedded) ** 2 / normalized_targets
        return jnp.sum(residual) / normalization

    # Sammon stress is dimensionless after normalization, so its gradient
    # tolerance should not inherit the units of the input distances.
    requested_gradient_tol, effective_tol = _gradient_tolerances(
        normalized_initial,
        1.0,
        tol,
    )
    normalized_solution, value, history = Minimize(
        M=Euclidean(size=normalized_initial.shape),
        cost=objective,
        x0=normalized_initial,
        solver=LBFGS(maxiter=maxiter, tolgradnorm=effective_tol, verbosity=0),
    ).solve()
    solution = distance_scale * normalized_solution
    final = history[-1]
    if (
        not bool(jnp.all(jnp.isfinite(solution)))
        or not bool(jnp.isfinite(value))
        or not math.isfinite(float(final.gradnorm))
    ):
        raise FloatingPointError("Sammon mapping produced a nonfinite solution.")
    return EmbeddingResult(
        coordinates=solution,
        objective=jnp.asarray(value),
        iterations=final.iter,
        converged=final.gradnorm <= effective_tol,
        reason=(
            "gradient tolerance reached at floating-point resolution"
            if final.gradnorm <= effective_tol and final.gradnorm > requested_gradient_tol
            else final.reason
        ),
        diagnostics={
            "history": tuple(history),
            "pairwise_distances": distances,
            "duplicate_tolerance": duplicate_tolerance,
            "distance_scale": distance_scale,
            "normalized_coordinates": normalized_solution,
            "requested_tolerance": tol,
            "requested_gradient_tolerance": requested_gradient_tol,
            "effective_tolerance": effective_tol,
        },
    )


def _tsne_probabilities(distances: Any, perplexity: float, tolerance: float = 1e-5) -> Any:
    distances = as_real_array(distances, name="t-SNE distances")
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("t-SNE distances must be a square matrix.")
    n_samples = distances.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE requires at least two observations.")
    if not bool(jnp.all(jnp.isfinite(distances))) or bool(jnp.any(distances < 0.0)):
        raise ValueError("t-SNE distances must be finite and nonnegative.")
    perplexity = positive_control(perplexity, name="perplexity")
    if not 1.0 <= perplexity <= n_samples - 1:
        raise ValueError("perplexity must lie between 1 and n_samples - 1.")
    tolerance = positive_control(tolerance, name="perplexity_tolerance")
    target = jnp.log(perplexity)
    distance_scale = float(jnp.max(distances))
    safe_scale = distance_scale if distance_scale > 0.0 else 1.0
    normalized_squared = (distances / safe_scale) ** 2
    probabilities = []
    for row in range(n_samples):
        beta = 1.0
        lower, upper = 0.0, jnp.inf
        mask = jnp.arange(n_samples) != row
        row_distances = normalized_squared[row]
        difference = float("inf")
        for _ in range(60):
            scores = jnp.where(mask, -beta * row_distances, -jnp.inf)
            maximum = jnp.max(scores)
            shifted = jnp.where(mask, scores - maximum, -jnp.inf)
            weights = jnp.where(mask, jnp.exp(shifted), 0.0)
            total = jnp.sum(weights)
            conditional = weights / total
            log_conditional = jnp.where(mask, shifted - jnp.log(total), 0.0)
            entropy = -jnp.sum(conditional * log_conditional)
            difference = float(entropy - target)
            if abs(difference) <= tolerance:
                break
            if difference > 0.0:
                lower = beta
                beta = 2.0 * beta if bool(jnp.isinf(upper)) else 0.5 * (beta + upper)
            else:
                upper = beta
                beta = 0.5 * beta if lower == 0.0 else 0.5 * (beta + lower)
        if abs(difference) > max(10.0 * tolerance, 100.0 * float(jnp.finfo(distances.dtype).eps)):
            raise ValueError(
                "The requested perplexity is unattainable for at least one row, "
                "usually because tied or duplicate observations force a larger "
                "minimum neighborhood size."
            )
        probabilities.append(conditional)
    conditional = jnp.stack(probabilities)
    joint = conditional + conditional.T
    off_diagonal = ~jnp.eye(n_samples, dtype=bool)
    floor = jnp.finfo(joint.dtype).tiny
    joint = jnp.where(off_diagonal, jnp.maximum(joint, floor), 0.0)
    return joint / jnp.sum(joint)


def tsne(
    manifold: Any,
    data: Any,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    key: Any | int | None,
    maxiter: int = 1000,
    learning_rate: float | None = None,
    early_exaggeration: float = 12.0,
    exaggeration_iterations: int = 250,
    tol: float = 1e-7,
) -> EmbeddingResult:
    """Dense t-SNE from exact manifold distances with explicit random state."""
    require_exact_operations(manifold, "tsne", "dist")
    adapted = _prepare(manifold, data, "tsne")
    n_components = _validate_n_components(n_components, adapted.n_samples)
    perplexity = positive_control(perplexity, name="perplexity")
    if not 1.0 <= perplexity <= adapted.n_samples - 1:
        raise ValueError("perplexity must lie between 1 and n_samples - 1.")
    maxiter = integer_control(maxiter, name="maxiter", minimum=1)
    rate = (
        positive_control(learning_rate, name="learning_rate")
        if learning_rate is not None
        else max(200.0, adapted.n_samples / 12.0)
    )
    early_exaggeration = positive_control(
        early_exaggeration,
        name="early_exaggeration",
    )
    exaggeration_iterations = integer_control(
        exaggeration_iterations,
        name="exaggeration_iterations",
    )
    tol = nonnegative_control(tol, name="tol")
    if not 0 <= exaggeration_iterations <= maxiter:
        raise ValueError("exaggeration_iterations must lie between 0 and maxiter.")
    distances = pairwise_distances(manifold, adapted)
    effective_tol = max(tol, 100.0 * float(jnp.finfo(distances.dtype).eps))
    probabilities = _tsne_probabilities(distances, perplexity)
    coordinates = 1e-4 * jax.random.normal(as_key(key, "tsne"), (adapted.n_samples, n_components))
    velocity = jnp.zeros_like(coordinates)
    history = []
    converged = False
    previous_objective: float | None = None
    stable_checks = 0
    for iteration in range(maxiter):
        delta = coordinates[:, None, :] - coordinates[None, :, :]
        numerator = 1.0 / (1.0 + jnp.sum(delta * delta, axis=-1))
        numerator = numerator.at[jnp.diag_indices(adapted.n_samples)].set(0.0)
        q = numerator / jnp.sum(numerator)
        q = jnp.where(
            jnp.eye(adapted.n_samples, dtype=bool),
            0.0,
            jnp.maximum(q, jnp.finfo(q.dtype).tiny),
        )
        q = q / jnp.sum(q)
        p = probabilities * (early_exaggeration if iteration < exaggeration_iterations else 1.0)
        gradient = 4.0 * jnp.sum(((p - q) * numerator)[..., None] * delta, axis=1)
        momentum = 0.5 if iteration < exaggeration_iterations else 0.8
        velocity = momentum * velocity - rate * gradient
        coordinates = coordinates + velocity
        coordinates = coordinates - jnp.mean(coordinates, axis=0, keepdims=True)
        if not bool(jnp.all(jnp.isfinite(coordinates))):
            raise FloatingPointError("t-SNE coordinates became nonfinite.")
        if iteration % 10 == 0 or iteration == maxiter - 1:
            final_delta = coordinates[:, None, :] - coordinates[None, :, :]
            final_numerator = 1.0 / (1.0 + jnp.sum(final_delta * final_delta, axis=-1))
            final_numerator = final_numerator.at[jnp.diag_indices(adapted.n_samples)].set(0.0)
            final_q = final_numerator / jnp.sum(final_numerator)
            final_q = jnp.where(
                jnp.eye(adapted.n_samples, dtype=bool),
                0.0,
                jnp.maximum(final_q, jnp.finfo(final_q.dtype).tiny),
            )
            final_q = final_q / jnp.sum(final_q)
            positive = probabilities > 0.0
            objective = jnp.sum(
                jnp.where(
                    positive,
                    probabilities
                    * jnp.log(probabilities / jnp.maximum(final_q, jnp.finfo(final_q.dtype).tiny)),
                    0.0,
                )
            )
            if not bool(jnp.isfinite(objective)):
                raise FloatingPointError("t-SNE produced a nonfinite KL divergence.")
            history.append(objective)
            if iteration >= exaggeration_iterations:
                current = float(objective)
                if previous_objective is not None and abs(
                    previous_objective - current
                ) <= effective_tol * max(
                    abs(previous_objective),
                    abs(current),
                    float(jnp.finfo(distances.dtype).tiny),
                ):
                    stable_checks += 1
                else:
                    stable_checks = 0
                previous_objective = current
                if stable_checks >= 3:
                    converged = True
                    break
    objective = history[-1]
    return EmbeddingResult(
        coordinates=coordinates,
        objective=objective,
        iterations=iteration + 1,
        converged=converged,
        reason=(
            "KL-divergence tolerance reached" if converged else "requested iterations completed"
        ),
        diagnostics={
            "kl_history": jnp.asarray(history),
            "joint_probabilities": probabilities,
            "embedded_probabilities": final_q,
            "pairwise_distances": distances,
            "requested_tolerance": tol,
            "effective_tolerance": effective_tol,
        },
    )


def phate(
    manifold: Any,
    data: Any,
    *,
    n_components: int = 2,
    n_neighbors: int = 5,
    decay: float = 40.0,
    diffusion_time: int | None = None,
    max_diffusion_time: int = 50,
    potential: str = "log",
) -> EmbeddingResult:
    """Compute a dense PHATE-style diffusion-potential embedding.

    This dependency-free implementation follows PHATE's adaptive affinity and
    diffusion-potential construction, followed by classical scaling. The
    reference implementation instead uses metric MDS; classical scaling and
    the deterministic maximum-curvature diffusion-time rule are documented
    approximations in GeoJAX.
    """
    require_exact_operations(manifold, "phate", "dist")
    adapted = _prepare(manifold, data, "phate")
    n_components = _validate_n_components(n_components, adapted.n_samples)
    n_neighbors = integer_control(n_neighbors, name="n_neighbors")
    if not 1 <= n_neighbors < adapted.n_samples:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1.")
    decay = positive_control(decay, name="decay")
    max_diffusion_time = integer_control(
        max_diffusion_time,
        name="max_diffusion_time",
        minimum=1,
    )
    if not isinstance(potential, str) or potential not in {"log", "sqrt"}:
        raise ValueError("potential must be 'log' or 'sqrt'.")
    distances = pairwise_distances(manifold, adapted)
    masked = jnp.where(jnp.eye(adapted.n_samples, dtype=bool), jnp.inf, distances)
    scales = jnp.sort(masked, axis=1)[:, n_neighbors - 1]
    if not bool(jnp.all(jnp.isfinite(scales))) or bool(jnp.any(scales <= 0.0)):
        raise ValueError(
            "PHATE requires a positive kth-neighbor distance for every sample; "
            "remove duplicate observations or increase n_neighbors."
        )
    tiny = jnp.finfo(distances.dtype).tiny
    local = jnp.exp(-((distances / scales[:, None]) ** decay))
    affinity = 0.5 * (local + local.T)
    row_sums = jnp.sum(affinity, axis=1, keepdims=True)
    if not bool(jnp.all(jnp.isfinite(row_sums))) or bool(jnp.any(row_sums <= 0.0)):
        raise FloatingPointError("PHATE produced an invalid affinity normalization.")
    transition = affinity / row_sums
    degrees = jnp.sum(affinity, axis=1)
    inverse_sqrt_degrees = 1.0 / jnp.sqrt(degrees)
    symmetric_diffusion = inverse_sqrt_degrees[:, None] * affinity * inverse_sqrt_degrees[None, :]
    eigenvalues = jnp.linalg.eigvalsh(symmetric_diffusion)
    eigenvalues = jnp.sort(jnp.abs(eigenvalues))[::-1]
    entropies = []
    for time in range(1, max_diffusion_time + 1):
        spectrum = eigenvalues**time
        spectrum_total = jnp.sum(spectrum)
        spectrum = spectrum / spectrum_total
        entropies.append(-jnp.sum(jnp.where(spectrum > 0.0, spectrum * jnp.log(spectrum), 0.0)))
    entropy_array = jnp.asarray(entropies)
    if diffusion_time is None:
        curvature = entropy_array[:-2] - 2.0 * entropy_array[1:-1] + entropy_array[2:]
        selected_time = int(jnp.argmax(jnp.abs(curvature))) + 2 if curvature.size else 1
    else:
        selected_time = integer_control(diffusion_time, name="diffusion_time")
        if not 1 <= selected_time <= max_diffusion_time:
            raise ValueError("diffusion_time must be within max_diffusion_time.")
    diffused = jnp.linalg.matrix_power(transition, selected_time)
    potential_coordinates = (
        -jnp.log(jnp.maximum(diffused, tiny))
        if potential == "log"
        else jnp.sqrt(jnp.maximum(diffused, 0.0))
    )
    potential_distances = stable_norm(
        potential_coordinates[:, None, :] - potential_coordinates[None, :, :],
        axis=-1,
    )
    coordinates, diagnostics = _mds_from_distances(potential_distances, n_components)
    diagnostics.update(
        {
            "pairwise_distances": distances,
            "affinity": affinity,
            "transition": transition,
            "symmetric_diffusion": symmetric_diffusion,
            "diffusion_time": selected_time,
            "von_neumann_entropy": entropy_array,
            "potential_distances": potential_distances,
        }
    )
    return EmbeddingResult(
        coordinates=coordinates,
        objective=diagnostics["stress"],
        iterations=selected_time,
        converged=True,
        reason="diffusion potential embedded",
        diagnostics=diagnostics,
    )


__all__ = [
    "classical_mds",
    "isomap",
    "kernel_pca",
    "phate",
    "principal_geodesic_analysis",
    "sammon_mapping",
    "tsne",
]
