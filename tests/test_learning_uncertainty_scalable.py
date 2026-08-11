from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

import geojax.learning._scalable as scalable
import geojax.learning._uncertainty as uncertainty
from geojax.geometry import Euclidean, FixedRank, Product, Sphere, Torus
from geojax.learning import (
    LearningCapabilityError,
    bootstrap_frechet_mean,
    energy_two_sample_test,
    kernel_mmd_two_sample_test,
    minibatch_frechet_mean,
    minibatch_kmeans,
    paired_frechet_test,
    streaming_frechet_mean,
)


def test_bootstrap_mean_is_reproducible_and_returns_a_geodesic_ball():
    manifold = Euclidean(1)
    values = jnp.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    first = bootstrap_frechet_mean(
        manifold, values, n_bootstrap=12, confidence_level=0.8, key=310, maxiter=30
    )
    second = bootstrap_frechet_mean(
        manifold, values, n_bootstrap=12, confidence_level=0.8, key=310, maxiter=30
    )

    assert first.replicates.shape == (12, 1)
    assert jnp.array_equal(first.replicates, second.replicates)
    assert first.confidence_radius >= 0.0
    assert jnp.allclose(first.estimate, jnp.array([0.0]), atol=1e-5)


def test_energy_mmd_and_paired_tests_detect_large_location_changes():
    manifold = Euclidean(1)
    left = jnp.linspace(-0.3, 0.3, 8)[:, None]
    right = left + 3.0
    energy = energy_two_sample_test(manifold, left, right, n_permutations=63, key=311)
    mmd = kernel_mmd_two_sample_test(manifold, left, right, n_permutations=63, key=312)
    paired = paired_frechet_test(manifold, left, right, n_permutations=63, key=313)

    assert energy.statistic > 1.0
    assert mmd.statistic > 0.5
    assert paired.statistic > 2.9
    assert energy.pvalue <= 0.05
    assert mmd.pvalue <= 0.05
    assert paired.pvalue <= 0.05
    assert mmd.diagnostics["effective_psd_tolerance"] >= 1e-8
    assert (
        jnp.min(mmd.diagnostics["kernel_eigenvalues"])
        >= -mmd.diagnostics["effective_psd_tolerance"]
    )


def test_uncertainty_methods_validate_keys_sizes_and_bandwidths():
    manifold = Euclidean(1)
    values = jnp.arange(4.0)[:, None]
    with pytest.raises(ValueError, match="explicit JAX random key"):
        bootstrap_frechet_mean(manifold, values, n_bootstrap=2, key=None)
    with pytest.raises(ValueError, match="positive"):
        bootstrap_frechet_mean(manifold, values, n_bootstrap=0, key=314)
    with pytest.raises(ValueError, match="confidence_level must lie"):
        bootstrap_frechet_mean(manifold, values, n_bootstrap=2, confidence_level=1.0, key=314)
    with pytest.raises(ValueError, match="bandwidth"):
        kernel_mmd_two_sample_test(
            manifold, values, values, bandwidth=0.0, n_permutations=2, key=314
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        kernel_mmd_two_sample_test(
            manifold,
            values,
            values,
            kernel=lambda distances: -jnp.eye(distances.shape[0]),
            n_permutations=2,
            key=314,
        )
    with pytest.raises(ValueError, match="finite"):
        kernel_mmd_two_sample_test(
            manifold,
            values,
            values,
            kernel=lambda distances: jnp.full_like(distances, jnp.nan),
            n_permutations=2,
            key=314,
        )
    with pytest.raises(ValueError, match="symmetric"):
        kernel_mmd_two_sample_test(
            manifold,
            values,
            values,
            kernel=lambda distances: jnp.triu(jnp.ones_like(distances)),
            n_permutations=2,
            key=314,
        )
    with pytest.raises(ValueError, match="same size"):
        paired_frechet_test(manifold, values, values[:-1], n_permutations=2, key=314)
    with pytest.raises(ValueError, match="maxiter"):
        paired_frechet_test(manifold, values, values, n_permutations=2, key=314, maxiter=0)


def test_streaming_mean_is_exact_for_weighted_euclidean_data():
    manifold = Euclidean(2)
    values = jnp.array([[0.0, 1.0], [2.0, 3.0], [8.0, -1.0]])
    weights = jnp.array([0.2, 0.3, 0.5])
    result = streaming_frechet_mean(manifold, values, sample_weight=weights)

    assert jnp.allclose(result.point, jnp.sum(weights[:, None] * values, axis=0))
    assert result.reason == "single pass completed"
    assert result.diagnostics["step_sizes"].shape == (3,)


def test_minibatch_mean_is_reproducible_and_improves_a_bad_initial_point():
    manifold = Euclidean(1)
    values = jnp.linspace(-1.0, 1.0, 20)[:, None]
    initial = jnp.array([20.0])
    first = minibatch_frechet_mean(
        manifold,
        values,
        batch_size=5,
        epochs=8,
        key=315,
        initial_point=initial,
        learning_rate=0.6,
        decay=0.2,
    )
    second = minibatch_frechet_mean(
        manifold,
        values,
        batch_size=5,
        epochs=8,
        key=315,
        initial_point=initial,
        learning_rate=0.6,
        decay=0.2,
    )
    assert jnp.array_equal(first.point, second.point)
    assert abs(float(first.point[0])) < 1.0
    assert first.objective < jnp.mean((values[:, 0] - 20.0) ** 2)


def test_minibatch_mean_uses_an_unbiased_unequal_weight_batch_direction():
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [2.0], [10.0], [20.0]])
    weights = jnp.array([0.70, 0.10, 0.10, 0.10])
    initial = jnp.array([5.0])
    result = minibatch_frechet_mean(
        manifold,
        values,
        sample_weight=weights,
        batch_size=4,
        epochs=1,
        key=901,
        initial_point=initial,
        learning_rate=1.0,
        decay=0.0,
        tol=0.0,
    )
    assert jnp.allclose(result.point, jnp.sum(weights[:, None] * values, axis=0))


def test_minibatch_mean_does_not_confuse_decayed_steps_with_stationarity():
    result = minibatch_frechet_mean(
        Euclidean(1),
        jnp.array([[0.0], [10.0]]),
        batch_size=2,
        epochs=2,
        key=902,
        initial_point=jnp.array([100.0]),
        learning_rate=1e-6,
        decay=1e6,
        tol=1e-3,
    )

    assert not result.converged
    assert result.gradient_norm > result.diagnostics["effective_movement_tolerance"]


def test_minibatch_kmeans_respects_global_sample_weight_mass():
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [10.0]])
    result = minibatch_kmeans(
        manifold,
        values,
        n_clusters=1,
        batch_size=1,
        epochs=1,
        sample_weight=jnp.array([1.0 - 1e-9, 1e-9]),
        learning_rate=0.5,
        key=318,
    )

    assert abs(float(result.centers[0, 0])) < 1e-5


def test_minibatch_kmeans_separates_clusters_and_supports_product_points():
    manifold = Euclidean(1)
    values = jnp.array([[-3.0], [-2.8], [-2.6], [2.6], [2.8], [3.0]])
    result = minibatch_kmeans(
        manifold,
        values,
        n_clusters=2,
        batch_size=3,
        epochs=10,
        key=316,
    )
    assert jnp.all(result.labels[:3] == result.labels[0])
    assert jnp.all(result.labels[3:] == result.labels[3])
    assert result.labels[0] != result.labels[3]

    product = Product({"direction": Sphere(3), "phase": Torus(1)})
    product_values = product.random_point(jax.random.key(317), sample_shape=(6,))
    product_result = minibatch_kmeans(
        product,
        product_values,
        n_clusters=2,
        batch_size=3,
        epochs=2,
        key=318,
    )
    assert bool(jnp.all(product.belongs(product_result.centers)))

    zero_mass = minibatch_kmeans(
        Euclidean(1),
        jnp.array([[0.0], [10.0], [11.0], [12.0]]),
        n_clusters=2,
        batch_size=2,
        epochs=3,
        key=318,
        sample_weight=jnp.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert bool(jnp.all(jnp.isfinite(zero_mass.centers)))
    assert bool(jnp.isfinite(zero_mass.objective))


def test_scalable_methods_validate_controls_and_exact_capabilities():
    manifold = Euclidean(1)
    values = jnp.arange(4.0)[:, None]
    with pytest.raises(ValueError, match="batch_size"):
        minibatch_frechet_mean(manifold, values, batch_size=0, epochs=2, key=319)
    with pytest.raises(ValueError, match="epochs"):
        minibatch_kmeans(manifold, values, n_clusters=2, batch_size=2, epochs=0, key=319)
    proxy = FixedRank((3, 3), rank=1)
    proxy_values = proxy.random_point(jax.random.key(320), sample_shape=(3,))
    with pytest.raises(LearningCapabilityError, match="dist=proxy"):
        streaming_frechet_mean(proxy, proxy_values)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"check_psd": 1}, TypeError, "check_psd must be a boolean"),
        ({"kernel": 1}, TypeError, "kernel must be callable"),
        (
            {"kernel": lambda distances: jnp.eye(distances.shape[0]), "bandwidth": 1.0},
            ValueError,
            "bandwidth is only defined",
        ),
        (
            {"kernel": lambda distances: jnp.eye(distances.shape[0], dtype=complex)},
            ValueError,
            "real-valued",
        ),
        (
            {"kernel": lambda distances: jnp.ones((distances.shape[0], 1))},
            ValueError,
            "square matrix matching",
        ),
    ],
)
def test_custom_mmd_kernel_contracts_are_explicit(kwargs, exception, message):
    values = jnp.array([[0.0], [1.0]])
    with pytest.raises(exception, match=message):
        kernel_mmd_two_sample_test(
            Euclidean(1),
            values,
            values,
            n_permutations=1,
            key=820,
            **kwargs,
        )


def test_custom_mmd_kernel_can_explicitly_disable_the_psd_gate():
    values = jnp.array([[0.0], [1.0]])
    result = kernel_mmd_two_sample_test(
        Euclidean(1),
        values,
        values + 0.5,
        kernel=lambda distances: -jnp.eye(distances.shape[0]),
        check_psd=False,
        n_permutations=3,
        key=821,
    )

    assert result.method == "Custom-kernel MMD permutation test"
    assert result.diagnostics["bandwidth"] is None
    assert jnp.min(result.diagnostics["kernel_eigenvalues"]) < 0.0


def test_rbf_mmd_uses_a_finite_default_when_all_distances_are_zero():
    values = jnp.zeros((2, 1))
    result = kernel_mmd_two_sample_test(
        Euclidean(1),
        values,
        values,
        n_permutations=1,
        key=822,
    )

    assert result.diagnostics["bandwidth"] == 1.0
    assert bool(jnp.isfinite(result.statistic))


@pytest.mark.parametrize("method", [energy_two_sample_test, kernel_mmd_two_sample_test])
def test_two_sample_uncertainty_methods_require_two_observations_per_group(method):
    with pytest.raises(ValueError, match="at least two observations"):
        method(
            Euclidean(1),
            jnp.array([[0.0]]),
            jnp.array([[1.0], [2.0]]),
            n_permutations=1,
            key=823,
        )


def test_bootstrap_rejects_original_and_replicate_nonconvergence(monkeypatch):
    values = jnp.array([[0.0], [1.0]])

    monkeypatch.setattr(
        uncertainty,
        "frechet_mean",
        lambda *args, **kwargs: SimpleNamespace(converged=False),
    )
    with pytest.raises(RuntimeError, match="original.*did not converge"):
        bootstrap_frechet_mean(Euclidean(1), values, n_bootstrap=1, key=824)

    fits = iter(
        [
            SimpleNamespace(converged=True, point=jnp.array([0.5])),
            SimpleNamespace(converged=False, point=jnp.array([0.5])),
        ]
    )
    monkeypatch.setattr(uncertainty, "frechet_mean", lambda *args, **kwargs: next(fits))
    with pytest.raises(RuntimeError, match="bootstrap.*did not converge"):
        bootstrap_frechet_mean(Euclidean(1), values, n_bootstrap=1, key=825)


def test_bootstrap_rejects_nonfinite_replicate_points(monkeypatch):
    values = jnp.array([[0.0], [1.0]])
    fits = iter(
        [
            SimpleNamespace(converged=True, point=jnp.array([0.5])),
            SimpleNamespace(converged=True, point=jnp.array([jnp.nan])),
        ]
    )
    monkeypatch.setattr(uncertainty, "frechet_mean", lambda *args, **kwargs: next(fits))

    with pytest.raises(FloatingPointError, match="nonfinite replicates"):
        bootstrap_frechet_mean(Euclidean(1), values, n_bootstrap=1, key=826)


def test_paired_test_rejects_nonconvergence_and_undefined_logarithms(monkeypatch):
    values = jnp.array([[0.0], [1.0]])
    monkeypatch.setattr(
        uncertainty,
        "frechet_mean",
        lambda *args, **kwargs: SimpleNamespace(converged=False),
    )
    with pytest.raises(RuntimeError, match="pooled.*did not converge"):
        paired_frechet_test(Euclidean(1), values, values, n_permutations=1, key=827)

    class UndefinedLogEuclidean(Euclidean):
        def log(self, x, y):
            return jnp.full(jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y)), jnp.nan)

    monkeypatch.setattr(
        uncertainty,
        "frechet_mean",
        lambda *args, **kwargs: SimpleNamespace(converged=True, point=jnp.array([0.0])),
    )
    with pytest.raises(ValueError, match="undefined logarithm"):
        paired_frechet_test(
            UndefinedLogEuclidean(1),
            values,
            values,
            n_permutations=1,
            key=828,
        )


def test_energy_test_rejects_nonfinite_distance_statistics():
    class NonfiniteDistanceEuclidean(Euclidean):
        def dist(self, x, y):
            return jnp.full(jnp.broadcast_shapes(jnp.shape(x)[:-1], jnp.shape(y)[:-1]), jnp.nan)

    values = jnp.array([[0.0], [1.0]])
    with pytest.raises(FloatingPointError, match="nonfinite statistic"):
        energy_two_sample_test(
            NonfiniteDistanceEuclidean(1),
            values,
            values,
            n_permutations=1,
            key=829,
        )


def test_streaming_mean_makes_prior_mass_and_initialization_explicit():
    values = jnp.array([[0.0], [2.0]])
    with pytest.raises(ValueError, match="initial_point is required"):
        streaming_frechet_mean(Euclidean(1), values, initial_weight=1.0)
    with pytest.raises(ValueError, match="initial_point must be a valid"):
        streaming_frechet_mean(Euclidean(1), values, initial_point=jnp.ones(2))

    result = streaming_frechet_mean(
        Euclidean(1),
        values,
        initial_point=jnp.array([10.0]),
        initial_weight=1.0,
    )
    assert jnp.allclose(result.point, jnp.array([5.5]))
    assert result.diagnostics["initial_weight"] == 1.0


def test_streaming_mean_defensively_rejects_zero_mass(monkeypatch):
    monkeypatch.setattr(
        scalable,
        "normalize_weights",
        lambda n_samples, sample_weight: jnp.zeros((n_samples,)),
    )
    with pytest.raises(ValueError, match="positive mass"):
        streaming_frechet_mean(Euclidean(1), jnp.array([[0.0], [1.0]]))


class _UndefinedLogEuclidean(Euclidean):
    def squared_dist(self, x, y):
        return jnp.sum((jnp.asarray(y) - jnp.asarray(x)) ** 2, axis=-1)

    def log(self, x, y):
        return jnp.full(jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y)), jnp.nan)


class _NonfiniteExpEuclidean(Euclidean):
    def exp(self, x, tangent):
        return jnp.full(jnp.broadcast_shapes(jnp.shape(x), jnp.shape(tangent)), jnp.nan)


@pytest.mark.parametrize(
    ("manifold", "message"),
    [
        (_UndefinedLogEuclidean(1), "undefined logarithm"),
        (_NonfiniteExpEuclidean(1), "exponential-map domain"),
    ],
)
def test_streaming_mean_guards_logarithm_and_exponential_domains(manifold, message):
    with pytest.raises(FloatingPointError, match=message):
        streaming_frechet_mean(manifold, jnp.array([[0.0], [1.0]]))


@pytest.mark.parametrize(
    ("manifold", "message"),
    [
        (_UndefinedLogEuclidean(1), "undefined logarithm"),
        (_NonfiniteExpEuclidean(1), "exponential-map domain"),
    ],
)
def test_minibatch_mean_guards_logarithm_and_exponential_domains(manifold, message):
    with pytest.raises(FloatingPointError, match=message):
        minibatch_frechet_mean(
            manifold,
            jnp.array([[0.0], [1.0]]),
            initial_point=jnp.array([0.0]),
            batch_size=2,
            epochs=1,
            key=830,
        )


@pytest.mark.parametrize(
    ("manifold", "message"),
    [
        (_UndefinedLogEuclidean(1), "undefined logarithm"),
        (_NonfiniteExpEuclidean(1), "exponential-map domain"),
    ],
)
def test_minibatch_kmeans_guards_logarithm_and_exponential_domains(manifold, message):
    with pytest.raises(FloatingPointError, match=message):
        minibatch_kmeans(
            manifold,
            jnp.array([[0.0], [1.0]]),
            n_clusters=1,
            batch_size=2,
            epochs=1,
            key=831,
        )


def test_minibatch_mean_skips_batches_with_zero_sample_mass():
    result = minibatch_frechet_mean(
        Euclidean(1),
        jnp.array([[0.0], [10.0], [20.0]]),
        sample_weight=jnp.array([1.0, 0.0, 0.0]),
        initial_point=jnp.array([0.0]),
        batch_size=1,
        epochs=1,
        key=832,
    )

    assert result.diagnostics["updates"] == 1
    assert jnp.allclose(result.point, jnp.array([0.0]))


def test_scalable_mean_diagnostics_reject_undefined_geometry_values():
    data = scalable._prepare(
        _UndefinedLogEuclidean(1),
        jnp.array([[0.0], [1.0]]),
        "test",
    )
    with pytest.raises(FloatingPointError, match="diagnostics.*undefined logarithm"):
        scalable._mean_diagnostics(
            _UndefinedLogEuclidean(1),
            jnp.array([0.0]),
            data,
            jnp.array([0.5, 0.5]),
        )


def test_minibatch_kmeans_rejects_a_nonfinite_initial_objective():
    class NonfiniteSquaredDistance(Euclidean):
        def squared_dist(self, x, y):
            return jnp.full(jnp.broadcast_shapes(jnp.shape(x)[:-1], jnp.shape(y)[:-1]), jnp.nan)

    with pytest.raises(FloatingPointError, match="nonfinite initial objective"):
        minibatch_kmeans(
            NonfiniteSquaredDistance(1),
            jnp.array([[0.0], [1.0]]),
            n_clusters=1,
            batch_size=2,
            epochs=1,
            key=833,
        )
