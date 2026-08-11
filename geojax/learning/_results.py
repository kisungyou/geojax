"""Immutable result records returned by GeoJAX learning algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class NeighborsResult:
    """Distances and training-set indices for each nearest-neighbor query."""

    distances: Any
    indices: Any


@dataclass(frozen=True)
class FrechetMeanResult:
    """A local Fréchet-mean fit with objective and stationarity diagnostics."""

    point: Any
    objective: Any
    gradient_norm: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrechetMedianResult:
    """A Huber-smoothed intrinsic median fit and its terminal residual."""

    point: Any
    objective: Any
    gradient_norm: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnclosingBallResult:
    """A farthest-point enclosing-ball approximation, not a global certificate."""

    center: Any
    radius: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClusteringResult:
    """Cluster assignments, representatives, objective, and convergence status."""

    labels: Any
    centers: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HierarchicalClusteringResult:
    """Flat labels and the merge table from metric-compatible agglomeration."""

    labels: Any
    linkage: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoresetResult:
    """Sampled observations and normalized importance weights for a coreset."""

    indices: Any
    points: Any
    weights: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelRegressionModel:
    """A fitted distance-kernel model for scalar Euclidean responses."""

    manifold: Any
    training_data: Any
    targets: Any
    bandwidth: float
    kernel: Callable[..., Any] | None

    def predict(self, data: Any) -> Any:
        """Predict responses at canonical or adaptable manifold observations."""

        from ._regression import _predict_kernel_regression

        return _predict_kernel_regression(self, data)


@dataclass(frozen=True)
class KernelCVResult:
    """Selected kernel model, bandwidth, and validation scores."""

    model: KernelRegressionModel
    bandwidth: float
    scores: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NearestCentroidModel:
    """Intrinsic class centroids and their internal mean-fit status.

    ``predict_proba`` returns normalized Gibbs distance scores. They are not
    calibrated posterior probabilities.
    """

    manifold: Any
    classes: Any
    centers: Any
    converged: bool = True
    reason: str = "all class centroids converged"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, data: Any) -> Any:
        """Predict the class of the closest intrinsic centroid."""

        from ._classification import _predict_nearest_centroid

        return _predict_nearest_centroid(self, data)

    def predict_proba(self, data: Any) -> Any:
        """Return normalized, query-scaled Gibbs distance scores."""

        from ._classification import _nearest_centroid_probabilities

        return _nearest_centroid_probabilities(self, data)


@dataclass(frozen=True)
class KNearestNeighborsModel:
    """A fitted geodesic-distance nearest-neighbors classifier."""

    manifold: Any
    training_data: Any
    classes: Any
    encoded_labels: Any
    n_neighbors: int
    weights: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, data: Any) -> Any:
        """Predict labels by uniform or inverse-distance voting."""

        from ._classification import _predict_knn

        return _predict_knn(self, data)

    def predict_proba(self, data: Any) -> Any:
        """Return normalized class vote weights."""

        from ._classification import _knn_probabilities

        return _knn_probabilities(self, data)


@dataclass(frozen=True)
class TangentFeatureMap:
    """A metric-orthonormal coordinate chart at a fitted reference point."""

    manifold: Any
    base_point: Any
    basis: tuple[Any, ...]
    eigenvalues: Any
    converged: bool = True
    reason: str = "reference point supplied"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def transform(self, data: Any) -> Any:
        """Map manifold observations to the retained tangent coordinates."""

        from ._features import transform_tangent_features

        return transform_tangent_features(self, data)


@dataclass(frozen=True)
class TangentSpaceClassifierModel:
    """A logistic, LDA, or QDA classifier in one intrinsic tangent chart."""

    manifold: Any
    classes: Any
    feature_map: TangentFeatureMap
    method: str
    coefficients: Any = None
    intercept: Any = None
    location: Any = None
    scale: Any = None
    class_means: Any = None
    covariances: Any = None
    priors: Any = None
    objective: Any = None
    iterations: int = 0
    converged: bool = True
    reason: str = "closed-form fit"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, data: Any) -> Any:
        """Predict labels after tangent-chart feature extraction."""

        from ._classification import _predict_tangent_classifier

        return _predict_tangent_classifier(self, data)

    def predict_proba(self, data: Any) -> Any:
        """Return normalized class scores from the fitted tangent model."""

        from ._classification import _tangent_classifier_probabilities

        return _tangent_classifier_probabilities(self, data)


@dataclass(frozen=True)
class GeodesicRegressionModel:
    """A fitted one-predictor intrinsic geodesic regression curve."""

    manifold: Any
    intercept: Any
    slope: Any
    predictor_mean: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, predictors: Any) -> Any:
        """Evaluate the fitted geodesic at scalar predictor values."""

        from ._response import _predict_geodesic_regression

        return _predict_geodesic_regression(self, predictors)


@dataclass(frozen=True)
class LocalPolynomialRegressionModel:
    """A local-constant or local-linear manifold-response smoother."""

    manifold: Any
    predictors: Any
    training_data: Any
    bandwidth: float
    degree: int
    kernel: Callable[..., Any] | None
    maxiter: int
    tol: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, predictors: Any) -> Any:
        """Solve the local Fréchet problem at each scalar query."""

        from ._response import _predict_local_polynomial_regression

        return _predict_local_polynomial_regression(self, predictors)


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate, bootstrap replicates, and geodesic confidence radius."""

    estimate: Any
    replicates: Any
    confidence_radius: Any
    confidence_level: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BarycentricCodingResult:
    """Simplex codes, intrinsic reconstructions, and solver diagnostics."""

    codes: Any
    reconstructions: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DictionaryLearningResult:
    """Learned manifold atoms, barycentric codes, and alternating-fit status."""

    atoms: Any
    codes: Any
    reconstructions: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobustLocationResult:
    """A robust intrinsic location estimate and terminal stationarity residual."""

    point: Any
    objective: Any
    gradient_norm: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricRanksResult:
    """Midranks of center distances together with their underlying scores."""

    ranks: Any
    scores: Any
    center: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemiSupervisedResult:
    """Transductive predictions, vertex scores, and graph-solver diagnostics."""

    predictions: Any
    scores: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisTestResult:
    """Observed statistic, calibrated p-value, and simulated null statistics."""

    statistic: Any
    pvalue: Any
    null_distribution: Any
    method: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportResult:
    """Transport distance, powered cost, coupling, and optimality diagnostics."""

    distance: Any
    cost: Any
    plan: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    """Euclidean coordinates and method-specific fit or spectral diagnostics."""

    coordinates: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    model: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricLearningModel:
    """An equivariant embedding followed by a learned positive metric."""

    manifold: Any
    metric: Any
    embedding: Callable[[Any], Any]
    regularization: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def transform(self, x: Any) -> Any:
        """Return Euclidean coordinates whose norm realizes the learned metric."""

        import jax.numpy as jnp

        from ._data import as_manifold_data
        from ._utils import flatten_embedding, require_unbatched

        adapted = as_manifold_data(self.manifold, x)
        require_unbatched(adapted, "MetricLearningModel.transform")
        values = flatten_embedding(self.embedding(adapted.values))
        if values.shape[1] != self.metric.shape[0]:
            raise ValueError("embedding feature dimension does not match the fitted RMML metric.")
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("embedding must return only finite coordinates.")
        factor = jnp.linalg.cholesky(self.metric)
        return values @ factor

    def pairwise_distances(self, x: Any, y: Any | None = None) -> Any:
        """Return distances induced by the fitted embedding metric."""

        from geojax.geometry._numerics import stable_norm

        left = self.transform(x)
        right = left if y is None else self.transform(y)
        delta = left[:, None, :] - right[None, :, :]
        return stable_norm(delta, axis=-1)


__all__ = [
    "BarycentricCodingResult",
    "BootstrapResult",
    "ClusteringResult",
    "CoresetResult",
    "DictionaryLearningResult",
    "EmbeddingResult",
    "EnclosingBallResult",
    "FrechetMeanResult",
    "FrechetMedianResult",
    "GeodesicRegressionModel",
    "HierarchicalClusteringResult",
    "HypothesisTestResult",
    "KNearestNeighborsModel",
    "KernelCVResult",
    "KernelRegressionModel",
    "LocalPolynomialRegressionModel",
    "MetricLearningModel",
    "MetricRanksResult",
    "NearestCentroidModel",
    "NeighborsResult",
    "RobustLocationResult",
    "SemiSupervisedResult",
    "TangentFeatureMap",
    "TangentSpaceClassifierModel",
    "TransportResult",
]
