"""Immutable result records returned by GeoJAX learning algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class NeighborsResult:
    distances: Any
    indices: Any


@dataclass(frozen=True)
class FrechetMeanResult:
    point: Any
    objective: Any
    gradient_norm: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrechetMedianResult:
    point: Any
    objective: Any
    gradient_norm: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnclosingBallResult:
    center: Any
    radius: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClusteringResult:
    labels: Any
    centers: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HierarchicalClusteringResult:
    labels: Any
    linkage: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoresetResult:
    indices: Any
    points: Any
    weights: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelRegressionModel:
    manifold: Any
    training_data: Any
    targets: Any
    bandwidth: float
    kernel: Callable[..., Any] | None

    def predict(self, data: Any) -> Any:
        from ._regression import _predict_kernel_regression

        return _predict_kernel_regression(self, data)


@dataclass(frozen=True)
class KernelCVResult:
    model: KernelRegressionModel
    bandwidth: float
    scores: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NearestCentroidModel:
    manifold: Any
    classes: Any
    centers: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, data: Any) -> Any:
        from ._classification import _predict_nearest_centroid

        return _predict_nearest_centroid(self, data)

    def predict_proba(self, data: Any) -> Any:
        from ._classification import _nearest_centroid_probabilities

        return _nearest_centroid_probabilities(self, data)


@dataclass(frozen=True)
class KNearestNeighborsModel:
    manifold: Any
    training_data: Any
    classes: Any
    encoded_labels: Any
    n_neighbors: int
    weights: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, data: Any) -> Any:
        from ._classification import _predict_knn

        return _predict_knn(self, data)

    def predict_proba(self, data: Any) -> Any:
        from ._classification import _knn_probabilities

        return _knn_probabilities(self, data)


@dataclass(frozen=True)
class TangentFeatureMap:
    manifold: Any
    base_point: Any
    basis: tuple[Any, ...]
    eigenvalues: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def transform(self, data: Any) -> Any:
        from ._features import transform_tangent_features

        return transform_tangent_features(self, data)


@dataclass(frozen=True)
class TangentSpaceClassifierModel:
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
        from ._classification import _predict_tangent_classifier

        return _predict_tangent_classifier(self, data)

    def predict_proba(self, data: Any) -> Any:
        from ._classification import _tangent_classifier_probabilities

        return _tangent_classifier_probabilities(self, data)


@dataclass(frozen=True)
class GeodesicRegressionModel:
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
        from ._response import _predict_geodesic_regression

        return _predict_geodesic_regression(self, predictors)


@dataclass(frozen=True)
class LocalPolynomialRegressionModel:
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
        from ._response import _predict_local_polynomial_regression

        return _predict_local_polynomial_regression(self, predictors)


@dataclass(frozen=True)
class BootstrapResult:
    estimate: Any
    replicates: Any
    confidence_radius: Any
    confidence_level: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BarycentricCodingResult:
    codes: Any
    reconstructions: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DictionaryLearningResult:
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
    point: Any
    objective: Any
    gradient_norm: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricRanksResult:
    ranks: Any
    scores: Any
    center: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemiSupervisedResult:
    predictions: Any
    scores: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisTestResult:
    statistic: Any
    pvalue: Any
    null_distribution: Any
    method: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportResult:
    distance: Any
    cost: Any
    plan: Any
    iterations: int
    converged: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    coordinates: Any
    objective: Any
    iterations: int
    converged: bool
    reason: str
    model: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricLearningModel:
    metric: Any
    embedding: Callable[[Any], Any]
    regularization: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def transform(self, x: Any) -> Any:
        import jax.numpy as jnp

        from ._utils import flatten_embedding

        values = flatten_embedding(self.embedding(x))
        factor = jnp.linalg.cholesky(self.metric)
        return values @ factor

    def pairwise_distances(self, x: Any, y: Any | None = None) -> Any:
        import jax.numpy as jnp

        left = self.transform(x)
        right = left if y is None else self.transform(y)
        delta = left[:, None, :] - right[None, :, :]
        return jnp.linalg.norm(delta, axis=-1)


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
