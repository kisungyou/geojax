from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import Euclidean, FixedRank, Product, Sphere, Torus
from geojax.learning import (
    LearningCapabilityError,
    agglomerative_clustering,
    competitive_quantization,
    frechet_mean,
    frechet_median,
    kernel_regression,
    kmeans,
    kmedoids,
    lightweight_coreset,
    mean_shift,
    minimum_enclosing_ball,
    nearest_neighbors,
    pairwise_distances,
    select_kernel_bandwidth,
    spectral_clustering,
)
from geojax.learning import _clustering as clustering_module
from geojax.learning import _statistics as statistics_module


class _NonfiniteDistanceEuclidean(Euclidean):
    def dist(self, x, y):
        return jnp.full_like(super().dist(x, y), jnp.nan)


class _NonfiniteLogEuclidean(Euclidean):
    def dist(self, x, y):
        return jnp.linalg.norm(jnp.asarray(y) - jnp.asarray(x), axis=-1)

    def squared_dist(self, x, y):
        return jnp.sum((jnp.asarray(y) - jnp.asarray(x)) ** 2, axis=-1)

    def log(self, x, y):
        return jnp.full_like(super().log(x, y), jnp.nan)


class _NonfiniteExpEuclidean(Euclidean):
    def exp(self, x, u):
        return jnp.full_like(super().exp(x, u), jnp.nan)


class _NonfiniteNormEuclidean(Euclidean):
    def dist(self, x, y):
        return jnp.linalg.norm(jnp.asarray(y) - jnp.asarray(x), axis=-1)

    def squared_dist(self, x, y):
        return jnp.sum((jnp.asarray(y) - jnp.asarray(x)) ** 2, axis=-1)

    def norm(self, x, u):
        del x, u
        return jnp.asarray(jnp.nan)


def clustered_euclidean_data():
    return jnp.array([[-1.0, 0.0], [-0.8, 0.1], [-1.2, -0.1], [1.0, 0.0], [0.8, -0.1], [1.2, 0.1]])


def test_pairwise_distances_blocking_nearest_neighbors_and_gradients():
    manifold = Euclidean(2)
    values = clustered_euclidean_data()
    dense = pairwise_distances(manifold, values, squared=True)
    blocked = pairwise_distances(manifold, values, squared=True, block_size=2)

    assert jnp.allclose(dense, blocked)
    assert jnp.allclose(dense, jnp.sum((values[:, None] - values[None, :]) ** 2, axis=-1))
    neighbors = nearest_neighbors(manifold, values, n_neighbors=2)
    assert neighbors.indices.shape == (6, 2)
    gradient = jax.grad(lambda data: jnp.sum(pairwise_distances(manifold, data, squared=True)))(
        values
    )
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_weighted_frechet_mean_and_median_match_euclidean_targets():
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [2.0], [10.0]])
    weights = jnp.array([0.45, 0.45, 0.10])
    mean = frechet_mean(manifold, values, sample_weight=weights, maxiter=50)
    median = frechet_median(manifold, values, sample_weight=weights, maxiter=100)

    assert jnp.allclose(mean.point, jnp.array([1.9]), atol=2e-4)
    assert jnp.allclose(median.point, jnp.array([2.0]), atol=2e-3)
    assert mean.converged


def test_frechet_median_diagnostics_are_evaluated_at_returned_point():
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [2.0], [9.0]])
    smoothing = 1e-3
    result = frechet_median(
        manifold,
        values,
        initial_point=jnp.array([5.0]),
        smoothing=smoothing,
        maxiter=1,
        tol=0.0,
    )
    distances = manifold.dist(result.point, values)
    inverse = (jnp.ones(3) / 3.0) / jnp.maximum(distances, smoothing)
    logs = manifold.log(result.point, values)
    stationarity = jnp.sum(inverse[:, None] * logs, axis=0)
    direction = stationarity / jnp.sum(inverse)
    assert jnp.allclose(result.gradient_norm, manifold.norm(result.point, stationarity))
    assert jnp.allclose(result.diagnostics["update_norm"], manifold.norm(result.point, direction))


def test_smoothed_frechet_median_handles_coincident_samples_monotonically():
    manifold = Euclidean(1)
    smoothing = 0.2
    values = jnp.array([[0.0], [0.0], [10.0]])
    result = frechet_median(
        manifold,
        values,
        initial_point=jnp.array([5.0]),
        smoothing=smoothing,
        maxiter=200,
        tol=1e-7,
    )
    history = result.diagnostics["objective_history"]

    assert result.converged
    assert jnp.allclose(result.point, jnp.array([smoothing / 2.0]), atol=2e-4)
    assert bool(jnp.all(jnp.diff(history) <= 10.0 * jnp.finfo(history.dtype).eps))
    assert jnp.allclose(history[-1], result.objective)


def test_minimum_enclosing_ball_contains_all_euclidean_samples():
    manifold = Euclidean(1)
    values = jnp.array([[-2.0], [0.0], [4.0]])
    result = minimum_enclosing_ball(manifold, values, maxiter=1000, tol=1e-8)
    distances = manifold.dist(result.center, values)
    assert bool(jnp.all(distances <= result.radius + 1e-6))
    assert result.radius <= 3.05


def test_kernel_regression_and_cross_validation_validate_sample_targets():
    manifold = Euclidean(1)
    x = jnp.linspace(-1.0, 1.0, 9)[:, None]
    y = x[:, 0] ** 2
    model = kernel_regression(manifold, x, y, bandwidth=0.3)
    prediction = model.predict(x)
    selected = select_kernel_bandwidth(
        manifold,
        x,
        y,
        jnp.array([0.15, 0.3, 0.8]),
        n_folds=3,
        key=jax.random.key(200),
    )

    assert prediction.shape == y.shape
    assert jnp.mean((prediction - y) ** 2) < 0.03
    assert selected.bandwidth in {0.15, 0.3, 0.8}
    with pytest.raises(ValueError, match="leading sample dimension"):
        kernel_regression(manifold, x, y[:-1], bandwidth=0.3)


@pytest.mark.parametrize(
    "algorithm",
    [
        lambda manifold, data: kmeans(manifold, data, n_clusters=2, key=201, maxiter=20),
        lambda manifold, data: kmedoids(manifold, data, n_clusters=2, key=201),
        lambda manifold, data: spectral_clustering(
            manifold, data, n_clusters=2, key=201, maxiter=20
        ),
        lambda manifold, data: mean_shift(manifold, data, bandwidth=0.4, maxiter=30),
        lambda manifold, data: competitive_quantization(
            manifold, data, n_clusters=2, key=201, epochs=4
        ),
    ],
)
def test_clustering_methods_separate_a_simple_two_cluster_problem(algorithm):
    manifold = Euclidean(2)
    result = algorithm(manifold, clustered_euclidean_data())
    left, right = result.labels[:3], result.labels[3:]
    assert bool(jnp.all(left == left[0]))
    assert bool(jnp.all(right == right[0]))
    assert int(left[0]) != int(right[0])
    assert bool(jnp.isfinite(result.objective))


def test_hierarchical_clustering_linkages_and_invalid_ward_request():
    manifold = Euclidean(2)
    values = clustered_euclidean_data()
    for linkage in ("single", "complete", "average"):
        result = agglomerative_clustering(manifold, values, n_clusters=2, linkage=linkage)
        assert result.linkage.shape == (5, 4)
        assert jnp.unique(result.labels).size == 2
    with pytest.raises(ValueError, match="single"):
        agglomerative_clustering(manifold, values, linkage="ward")


def test_lightweight_coreset_is_reproducible_and_usable_by_weighted_kmeans():
    manifold = Euclidean(2)
    values = clustered_euclidean_data()
    first = lightweight_coreset(manifold, values, size=12, key=202)
    second = lightweight_coreset(manifold, values, size=12, key=202)
    assert jnp.array_equal(first.indices, second.indices)
    assert jnp.isclose(jnp.sum(first.weights), 1.0)
    result = kmeans(
        manifold,
        first.points,
        n_clusters=2,
        key=203,
        sample_weight=first.weights,
        maxiter=20,
    )
    assert bool(jnp.isfinite(result.objective))


def test_center_based_methods_support_nested_product_points():
    manifold = Product({"direction": Sphere(3), "phase": Torus(1)})
    values = manifold.random_point(jax.random.key(204), sample_shape=(5,))
    mean = frechet_mean(manifold, values, maxiter=50)
    result = kmeans(manifold, values, n_clusters=2, key=205, maxiter=5, center_maxiter=30)
    assert bool(jnp.all(manifold.belongs(mean.point)))
    assert bool(jnp.all(manifold.belongs(result.centers)))


def test_proxy_geometry_fails_with_a_precise_capability_error():
    manifold = FixedRank((3, 3), rank=1)
    values = manifold.random_point(jax.random.key(206), sample_shape=(4,))
    for squared in (False, True):
        with pytest.raises(LearningCapabilityError, match="dist=proxy"):
            pairwise_distances(manifold, values, squared=squared)
    with pytest.raises(LearningCapabilityError, match="dist=proxy"):
        kmeans(manifold, values, n_clusters=2, key=207)


def test_kmeans_initialization_contract_and_empty_cluster_recovery():
    manifold = Euclidean(2)
    values = clustered_euclidean_data()
    random_result = kmeans(
        manifold,
        values,
        n_clusters=2,
        key=212,
        init="random",
        n_init=2,
        maxiter=10,
    )
    assert bool(jnp.isfinite(random_result.objective))

    duplicate_centers = jnp.stack([values[0], values[0]])
    recovered = kmeans(
        manifold,
        values,
        n_clusters=2,
        init=duplicate_centers,
        maxiter=1,
    )
    assert recovered.diagnostics["empty_cluster_recoveries"] == 1

    identical = jnp.zeros((4, 2))
    degenerate = kmeans(manifold, identical, n_clusters=3, key=213, maxiter=3)
    assert bool(jnp.all(manifold.belongs(degenerate.centers)))

    weighted = jnp.array([[0.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0]])
    zero_mass_cluster = kmeans(
        manifold,
        weighted,
        n_clusters=2,
        key=213,
        sample_weight=jnp.array([1.0, 0.0, 0.0, 0.0]),
        maxiter=3,
    )
    assert bool(jnp.all(jnp.isfinite(zero_mass_cluster.centers)))
    assert zero_mass_cluster.diagnostics["empty_cluster_recoveries"] >= 1

    with pytest.raises(ValueError, match="between 1"):
        kmeans(manifold, values, n_clusters=0, key=214)
    with pytest.raises(ValueError, match="n_init"):
        kmeans(manifold, values, n_clusters=2, key=214, n_init=0)
    with pytest.raises(ValueError, match="init must"):
        kmeans(manifold, values, n_clusters=2, key=214, init="medoid")
    with pytest.raises(ValueError, match="n_init=1"):
        kmeans(manifold, values, n_clusters=2, init=duplicate_centers, n_init=2)
    with pytest.raises(ValueError, match="exactly n_clusters"):
        kmeans(manifold, values, n_clusters=2, init=values[:1])


def test_converged_kmeans_returns_centers_consistent_with_returned_labels():
    manifold = Euclidean(2)
    values = clustered_euclidean_data()
    result = kmeans(manifold, values, n_clusters=2, key=214, maxiter=30)

    assert result.converged
    assert "assignment_changes" in result.diagnostics
    for cluster in range(2):
        members = values[result.labels == cluster]
        assert jnp.allclose(result.centers[cluster], jnp.mean(members, axis=0), atol=2e-5)


def test_clustering_edge_contracts_and_distance_only_spectral_path():
    manifold = Euclidean(2)
    values = clustered_euclidean_data()
    with pytest.raises(ValueError, match="size must be positive"):
        lightweight_coreset(manifold, values, size=0, key=215)
    with pytest.raises(ValueError, match="between 1"):
        kmedoids(manifold, values, n_clusters=7, key=215)
    with pytest.raises(ValueError, match="between 1"):
        agglomerative_clustering(manifold, values, n_clusters=0)

    singleton = agglomerative_clustering(manifold, values[:1], n_clusters=1)
    assert singleton.linkage.shape == (0, 4)
    assert jnp.array_equal(singleton.labels, jnp.array([0]))

    for laplacian in ("unnormalized", "random_walk"):
        result = spectral_clustering(
            manifold,
            values,
            n_clusters=2,
            key=216,
            affinity="self_tuning",
            n_neighbors=2,
            laplacian=laplacian,
            maxiter=10,
        )
        assert result.diagnostics["laplacian"] == laplacian

    with pytest.raises(ValueError, match="bandwidth"):
        spectral_clustering(manifold, values, n_clusters=2, key=216, bandwidth=0.0)
    with pytest.raises(ValueError, match="n_neighbors"):
        spectral_clustering(
            manifold,
            values,
            n_clusters=2,
            key=216,
            affinity="self_tuning",
            n_neighbors=0,
        )
    with pytest.raises(ValueError, match="affinity"):
        spectral_clustering(manifold, values, n_clusters=2, key=216, affinity="linear")
    with pytest.raises(ValueError, match="laplacian"):
        spectral_clustering(manifold, values, n_clusters=2, key=216, laplacian="directed")

    class DistanceOnlyEuclidean(Euclidean):
        exp_is_exact = False
        log_is_exact = False

    distance_only = DistanceOnlyEuclidean(2)
    result = spectral_clustering(distance_only, values, n_clusters=2, key=217, maxiter=10)
    assert result.centers.shape == (2, 2)
    for cluster in range(2):
        members = values[result.labels == cluster]
        assert bool(jnp.any(jnp.all(members == result.centers[cluster], axis=1)))


def test_random_walk_spectral_embedding_solves_nonsymmetric_eigenproblem():
    result = spectral_clustering(
        Euclidean(2),
        clustered_euclidean_data(),
        n_clusters=2,
        key=217,
        laplacian="random_walk",
        bandwidth=0.8,
        maxiter=10,
    )
    coordinates = result.diagnostics["embedding"]
    eigenvalues = result.diagnostics["eigenvalues"][:2]
    laplacian = result.diagnostics["laplacian_matrix"]
    assert jnp.allclose(
        laplacian @ coordinates,
        coordinates * eigenvalues[None, :],
        atol=2e-5,
    )


def test_mean_shift_and_quantization_validate_iteration_controls():
    manifold = Euclidean(2)
    values = clustered_euclidean_data()
    with pytest.raises(ValueError, match="bandwidth"):
        mean_shift(manifold, values, bandwidth=0.0)
    with pytest.raises(ValueError, match="merge_tol"):
        mean_shift(manifold, values, bandwidth=0.4, merge_tol=-1.0)

    unconverged = mean_shift(
        manifold,
        values,
        bandwidth=0.4,
        maxiter=1,
        tol=0.0,
        merge_tol=0.05,
    )
    assert not unconverged.converged
    assert unconverged.reason == "maximum iterations reached"

    stable = mean_shift(
        Euclidean(1),
        jnp.array([[0.0], [1000.0]]),
        bandwidth=1e-3,
        sample_weight=jnp.array([0.0, 1.0]),
        maxiter=3,
    )
    assert bool(jnp.all(jnp.isfinite(stable.centers)))

    result = competitive_quantization(
        manifold,
        values,
        n_clusters=2,
        key=218,
        epochs=1,
        initial_gain=0.3,
        decay=0.0,
    )
    assert result.diagnostics["updates"] == len(values)


def test_statistics_and_regression_reject_degenerate_controls():
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="smoothing"):
        frechet_median(manifold, values, smoothing=0.0)
    median = frechet_median(
        manifold,
        values,
        initial_point=jnp.array([0.1]),
        maxiter=1,
        tol=0.0,
    )
    assert median.reason == "maximum iterations reached"

    ball = minimum_enclosing_ball(
        manifold,
        values,
        initial_point=jnp.array([1.5]),
        maxiter=1,
    )
    assert ball.reason == "maximum iterations reached"

    with pytest.raises(ValueError, match="finite"):
        kernel_regression(
            manifold,
            values,
            jnp.array([0.0, 1.0, jnp.nan, 3.0]),
            bandwidth=0.5,
        )
    with pytest.raises(ValueError, match="bandwidth"):
        kernel_regression(manifold, values, jnp.arange(4.0), bandwidth=0.0)

    malformed = kernel_regression(
        manifold,
        values,
        jnp.arange(4.0),
        bandwidth=0.5,
        kernel=lambda distances, bandwidth: jnp.ones((1,)),
    )
    with pytest.raises(ValueError, match="one weight"):
        malformed.predict(values)

    nearest = kernel_regression(
        manifold,
        values,
        jnp.arange(4.0),
        bandwidth=0.5,
        kernel=lambda distances, bandwidth: jnp.zeros_like(distances),
    )
    assert jnp.allclose(nearest.predict(values), jnp.arange(4.0))

    for bad_kernel, message in [
        (lambda distances, bandwidth: -jnp.ones_like(distances), "nonnegative"),
        (lambda distances, bandwidth: jnp.full_like(distances, jnp.nan), "finite"),
        (
            lambda distances, bandwidth: jnp.ones_like(distances, dtype=jnp.complex64),
            "real-valued",
        ),
    ]:
        invalid = kernel_regression(
            manifold,
            values,
            jnp.arange(4.0),
            bandwidth=0.5,
            kernel=bad_kernel,
        )
        with pytest.raises(ValueError, match=message):
            invalid.predict(values)

    with pytest.raises(ValueError, match="nonempty vector"):
        select_kernel_bandwidth(
            manifold, values, jnp.arange(4.0), jnp.array([]), n_folds=2, key=219
        )
    with pytest.raises(ValueError, match="n_folds"):
        select_kernel_bandwidth(
            manifold, values, jnp.arange(4.0), jnp.array([0.5]), n_folds=1, key=219
        )
    with pytest.raises(ValueError, match="explicit JAX random key"):
        select_kernel_bandwidth(
            manifold, values, jnp.arange(4.0), jnp.array([0.5]), n_folds=2, key=None
        )


def test_competitive_quantization_does_not_confuse_decay_with_stationarity():
    result = competitive_quantization(
        Euclidean(1),
        jnp.array([[0.0], [10.0]]),
        n_clusters=1,
        key=220,
        epochs=2,
        initial_gain=1e-8,
        decay=1e8,
        tol=1e-3,
    )

    assert not result.converged
    assert (
        jnp.max(result.diagnostics["stationarity_norms"][-1])
        > result.diagnostics["effective_stationarity_tolerance"]
    )


def test_weight_validation_is_shared_across_learning_methods():
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [1.0], [2.0]])
    for weights, message in [
        (jnp.ones(2), "shape"),
        (jnp.array([1.0, jnp.nan, 1.0]), "finite"),
        (jnp.array([1.0, -1.0, 1.0]), "nonnegative"),
        (jnp.zeros(3), "positive total"),
    ]:
        with pytest.raises(ValueError, match=message):
            frechet_mean(manifold, values, sample_weight=weights)

    batched = manifold.random_point(jax.random.key(220), sample_shape=(2, 3))
    with pytest.raises(ValueError, match="unbatched"):
        frechet_mean(manifold, batched)


def test_regression_rejects_remaining_target_kernel_and_cv_contracts():
    manifold = Euclidean(1)
    values = jnp.arange(4.0)[:, None]
    targets = jnp.arange(4.0)

    with pytest.raises(ValueError, match="leading sample dimension"):
        kernel_regression(manifold, values, jnp.asarray(1.0), bandwidth=0.5)
    with pytest.raises(ValueError, match="real-valued"):
        kernel_regression(
            manifold,
            values,
            targets.astype(jnp.complex64),
            bandwidth=0.5,
        )
    with pytest.raises(TypeError, match="callable"):
        kernel_regression(manifold, values, targets, bandwidth=0.5, kernel=1)
    with pytest.raises(TypeError, match="callable"):
        select_kernel_bandwidth(
            manifold,
            values,
            targets,
            jnp.array([0.5]),
            n_folds=2,
            key=221,
            kernel=1,
        )
    with pytest.raises(ValueError, match="between 2 and n_samples"):
        select_kernel_bandwidth(
            manifold,
            values,
            targets,
            jnp.array([0.5]),
            n_folds=5,
            key=221,
        )

    for candidates in (
        jnp.asarray(0.5),
        jnp.array([jnp.nan]),
        jnp.array([-0.5]),
        jnp.ones((1, 1)),
    ):
        with pytest.raises(ValueError, match="nonempty vector"):
            select_kernel_bandwidth(
                manifold,
                values,
                targets,
                candidates,
                n_folds=2,
                key=221,
            )


def test_statistics_fail_loudly_for_nonfinite_solver_and_geometry_outputs(monkeypatch):
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [1.0]])

    with pytest.raises(FloatingPointError, match="data scale"):
        statistics_module._gradient_tolerances(jnp.array([0.0]), jnp.nan, 1e-6)

    outcomes = (
        (jnp.array([jnp.nan]), jnp.asarray(0.0), jnp.asarray(0.0)),
        (jnp.array([0.0]), jnp.asarray(jnp.nan), jnp.asarray(0.0)),
        (jnp.array([0.0]), jnp.asarray(0.0), jnp.asarray(jnp.nan)),
    )
    for point, value, gradnorm in outcomes:
        final = SimpleNamespace(gradnorm=gradnorm, iter=1, reason="")
        monkeypatch.setattr(
            statistics_module,
            "Minimize",
            lambda **kwargs: SimpleNamespace(solve=lambda: (point, value, [final])),
        )
        with pytest.raises(FloatingPointError, match="nonfinite result"):
            frechet_mean(manifold, values, initial_point=jnp.array([0.0]))

    with pytest.raises(ValueError, match="initial_point"):
        frechet_mean(manifold, values, initial_point=jnp.array([jnp.nan]))


def test_intrinsic_location_guards_cover_nonfinite_distance_log_norm_and_exp():
    values = jnp.array([[0.0], [1.0]])
    with pytest.raises(FloatingPointError, match="initial distances"):
        frechet_median(
            _NonfiniteDistanceEuclidean(1),
            values,
            initial_point=jnp.array([0.0]),
        )
    with pytest.raises(FloatingPointError, match="undefined logarithm"):
        frechet_median(
            _NonfiniteLogEuclidean(1),
            values,
            initial_point=jnp.array([0.0]),
        )
    with pytest.raises(FloatingPointError, match="nonfinite gradient"):
        frechet_median(
            _NonfiniteNormEuclidean(1),
            values,
            initial_point=jnp.array([0.0]),
        )
    stalled = frechet_median(
        _NonfiniteExpEuclidean(1),
        values,
        initial_point=jnp.array([0.0]),
        maxiter=1,
        tol=0.0,
    )
    assert stalled.reason == "descent line search failed"

    with pytest.raises(FloatingPointError, match="nonfinite distance"):
        minimum_enclosing_ball(_NonfiniteDistanceEuclidean(1), values)
    with pytest.raises(FloatingPointError, match="undefined logarithm"):
        minimum_enclosing_ball(_NonfiniteLogEuclidean(1), values)
    with pytest.raises(FloatingPointError, match="exponential-map domain"):
        minimum_enclosing_ball(_NonfiniteExpEuclidean(1), values)

    converged = minimum_enclosing_ball(Euclidean(1), jnp.zeros((2, 1)), maxiter=3)
    assert converged.converged
    assert converged.reason == "farthest-point update tolerance reached"


def test_clustering_internal_recovery_and_stationarity_guards(monkeypatch):
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [1.0], [2.0]])
    adapted = clustering_module._prepare(manifold, values, "test")
    centers = [jnp.array([0.0]), jnp.array([2.0])]

    stationarity = clustering_module._kmeans_stationarity(
        manifold,
        adapted,
        centers,
        jnp.array([0, 0, 0]),
        jnp.ones(3) / 3.0,
    )
    assert stationarity.shape == (2,)
    assert stationarity[1] == 0.0
    zero_mass = clustering_module._kmeans_stationarity(
        manifold,
        adapted,
        centers,
        jnp.array([0, 0, 1]),
        jnp.array([0.0, 0.0, 1.0]),
    )
    assert zero_mass[0] == 0.0
    with pytest.raises(FloatingPointError, match="stationarity check"):
        clustering_module._kmeans_stationarity(
            _NonfiniteLogEuclidean(1),
            adapted,
            centers,
            jnp.array([0, 0, 1]),
            jnp.ones(3) / 3.0,
        )
    with pytest.raises(ValueError, match="init"):
        clustering_module._initial_indices(
            manifold,
            adapted,
            2,
            jax.random.key(222),
            "unsupported",
            jnp.ones(3) / 3.0,
        )
    sparse_random = clustering_module._initial_indices(
        manifold,
        adapted,
        2,
        jax.random.key(222),
        "random",
        jnp.array([1.0, 0.0, 0.0]),
    )
    assert jnp.unique(sparse_random).size == 2

    monkeypatch.setattr(clustering_module.jax.random, "choice", lambda *a, **k: jnp.asarray(0))
    collision = clustering_module._initial_indices(
        manifold,
        adapted,
        2,
        jax.random.key(222),
        "kmeans++",
        jnp.ones(3) / 3.0,
    )
    assert jnp.array_equal(collision, jnp.array([0, 2]))
    with pytest.raises(RuntimeError, match="before any nonempty center"):
        clustering_module._centers_from_labels(
            manifold,
            adapted,
            jnp.zeros(3, dtype=int),
            2,
            jnp.zeros(3),
            center_maxiter=2,
            center_tol=0.0,
        )

    failed_fit = SimpleNamespace(
        point=jnp.array([jnp.nan]),
        objective=jnp.asarray(0.0),
        converged=True,
    )
    monkeypatch.setattr(clustering_module, "frechet_mean", lambda *args, **kwargs: failed_fit)
    with pytest.raises(FloatingPointError, match="nonfinite point"):
        clustering_module._centers_from_labels(
            manifold,
            adapted,
            jnp.zeros(3, dtype=int),
            1,
            jnp.ones(3) / 3.0,
            center_maxiter=2,
            center_tol=0.0,
        )
    with pytest.raises(FloatingPointError, match="reference mean"):
        lightweight_coreset(manifold, values, size=2, key=222)


def test_clustering_rejects_nonfinite_geometry_updates_and_remaining_modes():
    values = jnp.array([[0.0], [1.0], [2.0]])
    with pytest.raises(FloatingPointError, match="undefined logarithm"):
        mean_shift(_NonfiniteLogEuclidean(1), values, bandwidth=1.0, maxiter=1)
    with pytest.raises(FloatingPointError, match="exponential-map domain"):
        mean_shift(_NonfiniteExpEuclidean(1), values, bandwidth=1.0, maxiter=1, tol=0.0)
    with pytest.raises(FloatingPointError, match="undefined logarithm"):
        competitive_quantization(_NonfiniteLogEuclidean(1), values, n_clusters=1, key=223, epochs=1)
    with pytest.raises(FloatingPointError, match="exponential-map domain"):
        competitive_quantization(_NonfiniteExpEuclidean(1), values, n_clusters=1, key=223, epochs=1)

    for name, value in (("affinity", 1), ("laplacian", 1)):
        with pytest.raises(TypeError, match=name):
            spectral_clustering(
                Euclidean(1),
                values,
                n_clusters=2,
                key=223,
                **{name: value},
            )
    with pytest.raises(ValueError, match="only defined"):
        spectral_clustering(
            Euclidean(1),
            values,
            n_clusters=2,
            key=223,
            affinity="self_tuning",
            bandwidth=1.0,
        )
    with pytest.raises(ValueError, match="positive kth-neighbor"):
        spectral_clustering(
            Euclidean(1),
            jnp.array([[0.0], [0.0], [1.0]]),
            n_clusters=2,
            key=223,
            affinity="self_tuning",
            n_neighbors=1,
        )
