"""Dense dimension-reduction methods for manifold-valued observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from geojax.geometry import Euclidean
from geojax.optimization import LBFGS, Minimize

from ._capabilities import require_exact_operations
from ._data import ManifoldData, as_manifold_data
from ._geometry import pairwise_distances
from ._results import EmbeddingResult
from ._statistics import frechet_mean
from ._utils import (
    as_key,
    flatten_geometry_values,
    require_unbatched,
    stack_points,
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
    return values[order], vectors[:, order]


def _validate_n_components(n_components: int, n_samples: int) -> int:
    count = int(n_components)
    if not 1 <= count <= n_samples:
        raise ValueError("n_components must be between 1 and n_samples.")
    return count


def _mds_from_distances(distances: Any, n_components: int) -> tuple[Any, dict[str, Any]]:
    distances = jnp.asarray(distances)
    n_samples = distances.shape[0]
    if distances.shape != (n_samples, n_samples):
        raise ValueError("distances must be square.")
    n_components = _validate_n_components(n_components, n_samples)
    centering = jnp.eye(n_samples) - jnp.ones((n_samples, n_samples)) / n_samples
    gram = -0.5 * centering @ (distances**2) @ centering
    eigenvalues, eigenvectors = _descending_eigh(gram)
    positive = jnp.maximum(eigenvalues[:n_components], 0.0)
    coordinates = eigenvectors[:, :n_components] * jnp.sqrt(positive)[None, :]
    reconstructed = jnp.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    denominator = jnp.sum(distances**2)
    stress = jnp.sqrt(jnp.sum((distances - reconstructed) ** 2) / jnp.maximum(denominator, 1e-15))
    return coordinates, {
        "eigenvalues": eigenvalues,
        "gram_matrix": gram,
        "negative_eigenvalue_mass": jnp.sum(jnp.abs(jnp.minimum(eigenvalues, 0.0))),
        "stress": stress,
        "reconstructed_distances": reconstructed,
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
    coordinates, diagnostics = _mds_from_distances(distances, int(n_components))
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
    eigenvalues: Any

    def transform(self, data: Any) -> Any:
        adapted = as_manifold_data(self.manifold, data)
        logs = self.manifold.log(self.mean, adapted.values)
        left = _expand_tangent_samples(self.manifold, logs, left=True)
        right = _expand_tangent_samples(self.manifold, self.components, left=False)
        return self.manifold.inner(self.mean, left, right)

    def inverse_transform(self, coordinates: Any) -> Any:
        coordinates = jnp.asarray(coordinates)
        points = []
        for row in coordinates:
            tangent = weighted_tangent_sum(self.manifold, self.components, row)
            points.append(self.manifold.exp(self.mean, tangent))
        return stack_points(self.manifold, points)


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
    n_components = int(n_components)
    if not 1 <= n_components <= min(adapted.n_samples, manifold.dim):
        raise ValueError("n_components exceeds sample count or intrinsic dimension.")
    center = (
        frechet_mean(manifold, adapted, maxiter=maxiter, tol=tol).point
        if mean is None
        else manifold.project(mean)
    )
    logs = manifold.log(center, adapted.values)
    left = _expand_tangent_samples(manifold, logs, left=True)
    right = _expand_tangent_samples(manifold, logs, left=False)
    gram = manifold.inner(center, left, right) / adapted.n_samples
    eigenvalues, eigenvectors = _descending_eigh(gram)
    positive = jnp.maximum(eigenvalues[:n_components], 0.0)
    components = []
    for component in range(n_components):
        coefficient = eigenvectors[:, component] / jnp.sqrt(
            jnp.maximum(positive[component] * adapted.n_samples, 1e-15)
        )
        components.append(weighted_tangent_sum(manifold, logs, coefficient))
    component_tree = stack_points(manifold, components)
    coordinates = eigenvectors[:, :n_components] * jnp.sqrt(
        jnp.maximum(eigenvalues[:n_components] * adapted.n_samples, 0.0)
    )
    model = _PGAModel(manifold, center, component_tree, eigenvalues)
    explained = positive / jnp.maximum(jnp.sum(jnp.maximum(eigenvalues, 0.0)), 1e-15)
    return EmbeddingResult(
        coordinates=coordinates,
        objective=1.0 - jnp.sum(explained),
        iterations=1,
        converged=True,
        reason="metric covariance eigendecomposition completed",
        model=model,
        diagnostics={
            "mean": center,
            "components": component_tree,
            "eigenvalues": eigenvalues,
            "explained_variance_ratio": explained,
            "gram_matrix": gram,
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
        distances = pairwise_distances(self.manifold, queries, self.training_data)
        matrix = _distance_kernel(distances, self.bandwidth, self.kernel)
        expected = (queries.n_samples, self.training_data.n_samples)
        if matrix.shape != expected or not bool(jnp.all(jnp.isfinite(matrix))):
            raise ValueError(
                "kernel must return a finite query-by-training matrix with shape "
                f"{expected}."
            )
        centered = (
            matrix
            - self.training_column_mean[None, :]
            - jnp.mean(matrix, axis=1, keepdims=True)
            + self.training_global_mean
        )
        return centered @ self.eigenvectors / jnp.sqrt(jnp.maximum(self.eigenvalues, 1e-15))


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
) -> EmbeddingResult:
    """Kernel PCA using an RBF manifold-distance kernel or user callable."""
    require_exact_operations(manifold, "kernel_pca", "dist")
    adapted = _prepare(manifold, data, "kernel_pca")
    n_components = _validate_n_components(n_components, adapted.n_samples)
    if kernel is not None and not callable(kernel):
        raise TypeError("kernel must be callable when supplied.")
    distances = pairwise_distances(manifold, adapted)
    positive = distances[distances > 0.0]
    scale = float(jnp.median(positive)) if bandwidth is None and positive.size else 1.0
    scale = float(bandwidth) if bandwidth is not None else scale
    if scale <= 0.0:
        raise ValueError("bandwidth must be positive.")
    matrix = _distance_kernel(distances, scale, kernel)
    if matrix.shape != distances.shape:
        raise ValueError("kernel must return a square sample kernel matrix.")
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError("kernel must return only finite values.")
    asymmetry = jnp.linalg.norm(matrix - matrix.T)
    symmetry_scale = jnp.maximum(jnp.linalg.norm(matrix), 1.0)
    symmetry_tolerance = 100.0 * jnp.finfo(matrix.dtype).eps * symmetry_scale
    if float(asymmetry) > float(symmetry_tolerance):
        raise ValueError("kernel must return a symmetric sample kernel matrix.")
    matrix = 0.5 * (matrix + matrix.T)
    row_mean = jnp.mean(matrix, axis=1, keepdims=True)
    column_mean = jnp.mean(matrix, axis=0)
    global_mean = jnp.mean(matrix)
    centered = matrix - row_mean - column_mean[None, :] + global_mean
    eigenvalues, eigenvectors = _descending_eigh(centered)
    selected_values = jnp.maximum(eigenvalues[:n_components], 0.0)
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
    if not 1 <= int(n_neighbors) < adapted.n_samples:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1.")
    if disconnected not in {"error", "largest_component", "max_finite"}:
        raise ValueError("disconnected must be 'error', 'largest_component', or 'max_finite'.")
    distances = pairwise_distances(manifold, adapted)
    masked = jnp.where(jnp.eye(adapted.n_samples, dtype=bool), jnp.inf, distances)
    neighbors = jnp.argsort(masked, axis=1)[:, : int(n_neighbors)]
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
            replacement = 2.0 * jnp.maximum(maximum, jnp.asarray(1.0, dtype=maximum.dtype))
            geodesic = jnp.where(finite, geodesic, replacement)
    coordinates, diagnostics = _mds_from_distances(geodesic, int(n_components))
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
    distances = pairwise_distances(manifold, adapted)
    initial, _ = _mds_from_distances(distances, int(n_components))
    mask = jnp.triu(jnp.ones_like(distances, dtype=bool), k=1)
    normalization = jnp.sum(jnp.where(mask, distances, 0.0))

    def objective(coordinates: Any) -> Any:
        embedded = jnp.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
        residual = (distances - embedded) ** 2 / jnp.maximum(distances, 1e-12)
        return jnp.sum(jnp.where(mask, residual, 0.0)) / jnp.maximum(normalization, 1e-12)

    solution, value, history = Minimize(
        M=Euclidean(size=initial.shape),
        cost=objective,
        x0=initial,
        solver=LBFGS(maxiter=int(maxiter), tolgradnorm=float(tol), verbosity=0),
    ).solve()
    final = history[-1]
    return EmbeddingResult(
        coordinates=solution,
        objective=jnp.asarray(value),
        iterations=final.iter,
        converged=final.gradnorm <= tol,
        reason=final.reason,
        diagnostics={"history": tuple(history), "pairwise_distances": distances},
    )


def _tsne_probabilities(distances: Any, perplexity: float, tolerance: float = 1e-5) -> Any:
    n_samples = distances.shape[0]
    target = jnp.log(perplexity)
    probabilities = []
    for row in range(n_samples):
        beta = 1.0
        lower, upper = 0.0, jnp.inf
        mask = jnp.arange(n_samples) != row
        row_distances = distances[row] ** 2
        for _ in range(60):
            weights = jnp.where(mask, jnp.exp(-beta * row_distances), 0.0)
            total = jnp.maximum(jnp.sum(weights), 1e-15)
            entropy = jnp.log(total) + beta * jnp.sum(weights * row_distances) / total
            difference = float(entropy - target)
            if abs(difference) <= tolerance:
                break
            if difference > 0.0:
                lower = beta
                beta = 2.0 * beta if bool(jnp.isinf(upper)) else 0.5 * (beta + upper)
            else:
                upper = beta
                beta = 0.5 * beta if lower == 0.0 else 0.5 * (beta + lower)
        probabilities.append(weights / total)
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
) -> EmbeddingResult:
    """Dense t-SNE from exact manifold distances with explicit random state."""
    require_exact_operations(manifold, "tsne", "dist")
    adapted = _prepare(manifold, data, "tsne")
    n_components = _validate_n_components(n_components, adapted.n_samples)
    if not 1.0 <= perplexity < adapted.n_samples:
        raise ValueError("perplexity must be at least 1 and smaller than n_samples.")
    if int(maxiter) < 1:
        raise ValueError("maxiter must be positive.")
    if learning_rate is not None and float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive when supplied.")
    if float(early_exaggeration) <= 0.0:
        raise ValueError("early_exaggeration must be positive.")
    if not 0 <= int(exaggeration_iterations) <= int(maxiter):
        raise ValueError("exaggeration_iterations must lie between 0 and maxiter.")
    distances = pairwise_distances(manifold, adapted)
    probabilities = _tsne_probabilities(distances, float(perplexity))
    rate = float(learning_rate) if learning_rate is not None else max(200.0, adapted.n_samples / 12.0)
    coordinates = 1e-4 * jax.random.normal(
        as_key(key, "tsne"), (adapted.n_samples, n_components)
    )
    velocity = jnp.zeros_like(coordinates)
    history = []
    for iteration in range(int(maxiter)):
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
        if iteration % 10 == 0 or iteration == int(maxiter) - 1:
            positive = probabilities > 0.0
            history.append(
                jnp.sum(
                    jnp.where(
                        positive,
                        probabilities * jnp.log(probabilities / jnp.maximum(q, 1e-30)),
                        0.0,
                    )
                )
            )
    objective = history[-1]
    return EmbeddingResult(
        coordinates=coordinates,
        objective=objective,
        iterations=int(maxiter),
        converged=bool(jnp.isfinite(objective)),
        reason="requested iterations completed",
        diagnostics={
            "kl_history": jnp.asarray(history),
            "joint_probabilities": probabilities,
            "pairwise_distances": distances,
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
    """Dense PHATE using adaptive manifold-distance diffusion affinities."""
    require_exact_operations(manifold, "phate", "dist")
    adapted = _prepare(manifold, data, "phate")
    n_components = _validate_n_components(n_components, adapted.n_samples)
    if not 1 <= int(n_neighbors) < adapted.n_samples:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1.")
    if float(decay) <= 0.0:
        raise ValueError("decay must be positive.")
    if int(max_diffusion_time) < 1:
        raise ValueError("max_diffusion_time must be positive.")
    if potential not in {"log", "sqrt"}:
        raise ValueError("potential must be 'log' or 'sqrt'.")
    distances = pairwise_distances(manifold, adapted)
    masked = jnp.where(jnp.eye(adapted.n_samples, dtype=bool), jnp.inf, distances)
    scales = jnp.sort(masked, axis=1)[:, int(n_neighbors) - 1]
    local = jnp.exp(-((distances / jnp.maximum(scales[:, None], 1e-15)) ** decay))
    affinity = 0.5 * (local + local.T)
    transition = affinity / jnp.maximum(jnp.sum(affinity, axis=1, keepdims=True), 1e-15)
    degrees = jnp.sum(affinity, axis=1)
    inverse_sqrt_degrees = 1.0 / jnp.sqrt(jnp.maximum(degrees, 1e-15))
    symmetric_diffusion = (
        inverse_sqrt_degrees[:, None]
        * affinity
        * inverse_sqrt_degrees[None, :]
    )
    eigenvalues = jnp.linalg.eigvalsh(symmetric_diffusion)
    eigenvalues = jnp.sort(jnp.abs(eigenvalues))[::-1]
    entropies = []
    for time in range(1, int(max_diffusion_time) + 1):
        spectrum = eigenvalues**time
        spectrum = spectrum / jnp.maximum(jnp.sum(spectrum), 1e-15)
        entropies.append(-jnp.sum(spectrum * jnp.log(jnp.maximum(spectrum, 1e-15))))
    entropy_array = jnp.asarray(entropies)
    if diffusion_time is None:
        curvature = entropy_array[:-2] - 2.0 * entropy_array[1:-1] + entropy_array[2:]
        selected_time = int(jnp.argmax(jnp.abs(curvature))) + 2 if curvature.size else 1
    else:
        selected_time = int(diffusion_time)
        if not 1 <= selected_time <= int(max_diffusion_time):
            raise ValueError("diffusion_time must be within max_diffusion_time.")
    diffused = jnp.linalg.matrix_power(transition, selected_time)
    potential_coordinates = (
        -jnp.log(jnp.maximum(diffused, 1e-15))
        if potential == "log"
        else jnp.sqrt(jnp.maximum(diffused, 0.0))
    )
    potential_distances = jnp.linalg.norm(
        potential_coordinates[:, None, :] - potential_coordinates[None, :, :], axis=-1
    )
    coordinates, diagnostics = _mds_from_distances(potential_distances, int(n_components))
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
