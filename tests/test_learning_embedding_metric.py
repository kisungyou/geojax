from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import Euclidean, GrassmannProjection, Product, SphereExtrinsic
from geojax.learning import (
    LearningCapabilityError,
    classical_mds,
    isomap,
    kernel_pca,
    pairwise_distances,
    phate,
    principal_geodesic_analysis,
    riemannian_metric_learning,
    sammon_mapping,
    tsne,
)
from geojax.learning._embedding import _mds_from_distances, _tsne_probabilities
from geojax.learning._metric import _default_embedding
import geojax.learning._metric as metric_module


def curved_planar_data():
    parameter = jnp.linspace(-1.0, 1.0, 12)
    return jnp.stack([parameter, parameter**2], axis=-1)


def test_classical_mds_recovers_euclidean_distances_and_reports_spectrum():
    manifold = Euclidean(2)
    values = curved_planar_data()
    result = classical_mds(manifold, values, n_components=2)
    embedded = jnp.linalg.norm(result.coordinates[:, None] - result.coordinates[None, :], axis=-1)
    expected = pairwise_distances(manifold, values)
    assert jnp.allclose(embedded, expected, atol=2e-4, rtol=2e-4)
    assert result.diagnostics["negative_eigenvalue_mass"] < 1e-4


def test_pga_uses_metric_components_and_supports_product_tangents():
    manifold = Euclidean(2)
    values = curved_planar_data()
    result = principal_geodesic_analysis(manifold, values, n_components=2)
    transformed = result.model.transform(values)
    reconstructed = result.model.inverse_transform(transformed)
    assert jnp.allclose(transformed, result.coordinates, atol=2e-4)
    assert reconstructed.shape == values.shape
    component_gram = manifold.inner(
        result.diagnostics["mean"],
        result.diagnostics["components"][:, None, :],
        result.diagnostics["components"][None, :, :],
    )
    assert jnp.allclose(component_gram, jnp.eye(2), atol=2e-4)

    product = Product({"a": Euclidean(1), "b": Euclidean(1)})
    product_values = {"a": values[:, :1], "b": values[:, 1:]}
    product_result = principal_geodesic_analysis(product, product_values, n_components=2)
    assert product_result.coordinates.shape == (12, 2)
    with pytest.raises(ValueError, match="trailing dimension"):
        result.model.inverse_transform(jnp.ones((3, 1)))
    with pytest.raises(ValueError, match="finite"):
        result.model.inverse_transform(jnp.full((3, 2), jnp.nan))


def test_kernel_pca_model_has_consistent_training_transform():
    manifold = Euclidean(2)
    values = curved_planar_data()
    result = kernel_pca(manifold, values, n_components=2, bandwidth=0.5)
    transformed = result.model.transform(values)
    assert jnp.allclose(jnp.abs(transformed), jnp.abs(result.coordinates), atol=2e-4)
    assert bool(jnp.all(jnp.isfinite(result.coordinates)))


def test_isomap_disconnected_policies_are_explicit():
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [0.1], [10.0], [10.1]])
    with pytest.raises(ValueError, match="disconnected"):
        isomap(manifold, values, n_neighbors=1, mutual=True, disconnected="error")
    result = isomap(
        manifold,
        values,
        n_components=1,
        n_neighbors=1,
        mutual=True,
        disconnected="largest_component",
    )
    assert result.coordinates.shape[0] == 2


def test_sammon_tsne_and_phate_return_finite_dense_embeddings():
    manifold = Euclidean(2)
    values = curved_planar_data()
    sammon = sammon_mapping(manifold, values, maxiter=20)
    stochastic = tsne(
        manifold,
        values,
        perplexity=3.0,
        key=300,
        maxiter=20,
        exaggeration_iterations=5,
    )
    diffusion = phate(manifold, values, n_neighbors=3, max_diffusion_time=8)
    for result in (sammon, stochastic, diffusion):
        assert result.coordinates.shape == (12, 2)
        assert bool(jnp.all(jnp.isfinite(result.coordinates)))
        assert bool(jnp.isfinite(result.objective))
    probabilities = stochastic.diagnostics["joint_probabilities"]
    assert jnp.allclose(jnp.diag(probabilities), 0.0)
    assert jnp.allclose(jnp.sum(probabilities), 1.0)

    transition_spectrum = jnp.sort(jnp.linalg.eigvals(diffusion.diagnostics["transition"]).real)
    symmetric_spectrum = jnp.sort(jnp.linalg.eigvalsh(diffusion.diagnostics["symmetric_diffusion"]))
    assert jnp.allclose(transition_spectrum, symmetric_spectrum, atol=2e-5)


def test_mds_rejects_invalid_distance_matrices_and_tsne_is_underflow_stable():
    with pytest.raises(ValueError, match="finite"):
        _mds_from_distances(jnp.array([[0.0, jnp.nan], [1.0, 0.0]]), 1)
    with pytest.raises(ValueError, match="symmetric"):
        _mds_from_distances(jnp.array([[0.0, 1.0], [2.0, 0.0]]), 1)
    with pytest.raises(ValueError, match="nonnegative"):
        _mds_from_distances(jnp.array([[0.0, -1.0], [-1.0, 0.0]]), 1)
    with pytest.raises(ValueError, match="zero diagonal"):
        _mds_from_distances(jnp.array([[1.0, 0.0], [0.0, 1.0]]), 1)

    distances = jnp.array([[0.0, 1e4, 2e4], [1e4, 0.0, 3e4], [2e4, 3e4, 0.0]])
    probabilities = _tsne_probabilities(distances, perplexity=1.5)
    assert bool(jnp.all(jnp.isfinite(probabilities)))
    assert jnp.allclose(jnp.sum(probabilities), 1.0)


def test_mds_scale_normalization_is_finite_and_rejects_unrepresentable_squares():
    dtype = jnp.asarray(1.0).dtype
    limits = jnp.finfo(dtype)
    small = 10.0 * jnp.sqrt(limits.tiny)
    distances = jnp.asarray([[0.0, small], [small, 0.0]], dtype=dtype)
    coordinates, diagnostics = _mds_from_distances(distances, 1)

    assert bool(jnp.all(jnp.isfinite(coordinates)))
    assert bool(jnp.all(jnp.isfinite(diagnostics["normalized_eigenvalues"])))
    assert jnp.allclose(
        diagnostics["normalized_gram_matrix"],
        diagnostics["normalized_gram_matrix"].T,
    )

    huge = 2.0 * jnp.sqrt(limits.max)
    huge_distances = jnp.asarray([[0.0, huge], [huge, 0.0]], dtype=dtype)
    with pytest.raises(FloatingPointError, match="squared distance scale"):
        _mds_from_distances(huge_distances, 1)


def test_rmml_uses_explicit_or_geometry_embedding_and_separates_classes():
    manifold = Euclidean(2)
    values = jnp.array(
        [[-1.0, 0.0], [-0.8, 0.1], [-1.1, -0.1], [1.0, 0.0], [0.8, -0.1], [1.1, 0.1]]
    )
    labels = jnp.array([0, 0, 0, 1, 1, 1])
    model = riemannian_metric_learning(manifold, values, labels)
    eigenvalues = jnp.linalg.eigvalsh(model.metric)
    distances = model.pairwise_distances(values)
    assert bool(jnp.all(eigenvalues > 0.0))
    assert distances.shape == (6, 6)

    sphere = SphereExtrinsic(3)
    sphere_values = sphere.project(jnp.c_[values, jnp.ones(6)])
    sphere_model = riemannian_metric_learning(sphere, sphere_values, labels)
    assert sphere_model.metric.shape == (3, 3)


def test_rmml_matches_the_log_euclidean_closed_form_for_diagonal_scatter():
    manifold = Euclidean(2)
    values = jnp.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
    labels = jnp.array([0, 0, 1, 1])
    regularization = 0.25
    model = riemannian_metric_learning(
        manifold,
        values,
        labels,
        regularization=regularization,
        balance=0.5,
    )
    similar = model.diagnostics["similar_scatter"] + regularization * jnp.eye(2)
    dissimilar = model.diagnostics["dissimilar_scatter"] + regularization * jnp.eye(2)
    # Equation (21) of Zhu et al. has an outer factor of one half.  At equal
    # balance and for commuting diagonal scatter matrices, this is the fourth
    # root of their ratio rather than the square root.
    expected = jnp.diag((jnp.diag(dissimilar) / jnp.diag(similar)) ** 0.25)
    assert jnp.allclose(model.metric, expected, atol=2e-5)

    with pytest.raises(ValueError, match="balance"):
        riemannian_metric_learning(manifold, values, labels, balance=1.1)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_rmml_zero_regularization_repairs_degenerate_scatter(dtype):
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("float64 is disabled in this test matrix entry")
    values = jnp.asarray([[0.0], [0.0], [1.0], [1.0]], dtype=dtype)
    model = riemannian_metric_learning(
        Euclidean(1),
        values,
        jnp.asarray([0, 0, 1, 1]),
        regularization=0.0,
    )

    floor = model.diagnostics["effective_similar_eigenvalue_floor"]
    assert floor >= jnp.finfo(dtype).tiny
    assert bool(jnp.all(jnp.isfinite(model.metric)))
    assert bool(jnp.all(jnp.linalg.eigvalsh(model.metric) > 0.0))


def test_rmml_rejects_a_geometry_without_equivariant_embedding():
    manifold = GrassmannProjection((4, 2))
    values = manifold.random_point(jax.random.key(301), sample_shape=(4,))
    labels = jnp.array([0, 0, 1, 1])
    # GrassmannProjection does expose its projector embedding and should work.
    assert riemannian_metric_learning(manifold, values, labels).metric.shape == (16, 16)

    plain = Product({"embedded": SphereExtrinsic(3), "plain": Euclidean(1)})
    product_values = plain.random_point(jax.random.key(302), sample_shape=(4,))
    with pytest.raises(LearningCapabilityError, match="every Product factor"):
        riemannian_metric_learning(plain, product_values, labels)


def test_embedding_input_contracts_and_alternate_graph_policies():
    manifold = Euclidean(2)
    values = curved_planar_data()
    with pytest.raises(ValueError, match="square"):
        _mds_from_distances(jnp.ones((3, 2)), 2)
    with pytest.raises(ValueError, match="between 1"):
        _mds_from_distances(jnp.eye(3), 4)
    with pytest.raises(ValueError, match="intrinsic dimension"):
        principal_geodesic_analysis(manifold, values, n_components=3)
    with pytest.raises(ValueError, match="numerical rank"):
        principal_geodesic_analysis(manifold, jnp.ones((4, 2)), n_components=1)
    with pytest.raises(ValueError, match="bandwidth"):
        kernel_pca(manifold, values, bandwidth=0.0)
    with pytest.raises(ValueError, match="square"):
        kernel_pca(
            manifold,
            values,
            kernel=lambda distances, bandwidth: jnp.ones((distances.shape[0],)),
        )
    with pytest.raises(ValueError, match="numerical rank"):
        kernel_pca(
            manifold,
            values,
            n_components=1,
            kernel=lambda distances, bandwidth: jnp.ones_like(distances),
        )
    with pytest.raises(ValueError, match="distinct observations"):
        sammon_mapping(manifold, jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="maxiter"):
        sammon_mapping(manifold, values, maxiter=0)
    with pytest.raises(ValueError, match="n_neighbors"):
        isomap(manifold, values, n_neighbors=0)
    with pytest.raises(ValueError, match="disconnected"):
        isomap(manifold, values, n_neighbors=2, disconnected="ignore")

    separated = jnp.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0]])
    filled = isomap(
        manifold,
        separated,
        n_components=1,
        n_neighbors=1,
        mutual=True,
        disconnected="max_finite",
    )
    assert bool(jnp.all(jnp.isfinite(filled.coordinates)))

    with pytest.raises(ValueError, match="perplexity"):
        tsne(manifold, values, perplexity=len(values), key=303, maxiter=1)
    with pytest.raises(ValueError, match="n_components"):
        tsne(manifold, values, n_components=0, perplexity=3.0, key=303)
    with pytest.raises(ValueError, match="maxiter"):
        tsne(manifold, values, perplexity=3.0, key=303, maxiter=0)
    with pytest.raises(ValueError, match="learning_rate"):
        tsne(manifold, values, perplexity=3.0, key=303, learning_rate=0.0)
    with pytest.raises(ValueError, match="early_exaggeration"):
        tsne(manifold, values, perplexity=3.0, key=303, early_exaggeration=0.0)
    with pytest.raises(ValueError, match="exaggeration_iterations"):
        tsne(
            manifold,
            values,
            perplexity=3.0,
            key=303,
            maxiter=2,
            exaggeration_iterations=3,
        )
    with pytest.raises(ValueError, match="n_neighbors"):
        phate(manifold, values, n_neighbors=0)
    with pytest.raises(ValueError, match="n_components"):
        phate(manifold, values, n_components=0, n_neighbors=3)
    with pytest.raises(ValueError, match="decay"):
        phate(manifold, values, n_neighbors=3, decay=0.0)
    with pytest.raises(ValueError, match="max_diffusion_time"):
        phate(manifold, values, n_neighbors=3, max_diffusion_time=0)
    with pytest.raises(ValueError, match="potential"):
        phate(manifold, values, potential="linear")
    with pytest.raises(ValueError, match="diffusion_time"):
        phate(manifold, values, diffusion_time=20, max_diffusion_time=10)

    square_root = phate(
        manifold,
        values,
        n_neighbors=3,
        diffusion_time=2,
        max_diffusion_time=4,
        potential="sqrt",
    )
    assert square_root.iterations == 2


def test_tsne_joint_probabilities_are_symmetric_and_normalized():
    distances = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    probabilities = _tsne_probabilities(distances, perplexity=2.0)
    assert jnp.allclose(probabilities, probabilities.T)
    assert jnp.allclose(jnp.diag(probabilities), 0.0)
    assert jnp.allclose(jnp.sum(probabilities), 1.0)


def test_tsne_rejects_invalid_or_unattainable_perplexity():
    manifold = Euclidean(1)
    values = jnp.arange(4.0)[:, None]
    with pytest.raises(ValueError, match="perplexity"):
        tsne(manifold, values, perplexity=0.5, key=304, maxiter=1)

    tied = jnp.ones((4, 4)) - jnp.eye(4)
    with pytest.raises(ValueError, match="attain"):
        _tsne_probabilities(tied, perplexity=1.5)


def test_rmml_validates_labels_embeddings_and_pair_structure():
    manifold = Euclidean(2)
    values = curved_planar_data()[:4]
    labels = jnp.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="labels must have shape"):
        riemannian_metric_learning(manifold, values, labels[:-1])
    with pytest.raises(ValueError, match="at least two classes"):
        riemannian_metric_learning(manifold, values, jnp.zeros(4, dtype=int))
    with pytest.raises(ValueError, match="nonnegative"):
        riemannian_metric_learning(manifold, values, labels, regularization=-1.0)
    with pytest.raises(ValueError, match="leading sample"):
        riemannian_metric_learning(
            manifold,
            values,
            labels,
            embedding=lambda points: points[:-1],
        )
    with pytest.raises(TypeError, match="callable"):
        riemannian_metric_learning(manifold, values, labels, embedding=3)
    with pytest.raises(ValueError, match="finite coordinates"):
        riemannian_metric_learning(
            manifold,
            values,
            labels,
            embedding=lambda points: points.at[0, 0].set(jnp.nan),
        )
    with pytest.raises(ValueError, match="empty pytree"):
        riemannian_metric_learning(manifold, values, labels, embedding=lambda points: {})
    with pytest.raises(ValueError, match="share their leading"):
        riemannian_metric_learning(
            manifold,
            values,
            labels,
            embedding=lambda points: {"a": points, "b": points[:-1]},
        )
    with pytest.raises(ValueError, match="similar and one dissimilar"):
        riemannian_metric_learning(
            manifold,
            values[:2],
            jnp.array([0, 1]),
        )

    embedded = riemannian_metric_learning(
        manifold,
        values,
        labels,
        embedding=lambda points: {"a": points[:, :1], "b": points[:, 1:]},
    )
    assert embedded.metric.shape == (2, 2)


@pytest.mark.parametrize(
    ("distances", "message"),
    [
        (jnp.array([0.0, 1.0]), "two-dimensional"),
        (jnp.empty((0, 0)), "square"),
        (jnp.eye(2, dtype=jnp.complex64), "real-valued"),
    ],
)
def test_mds_rejects_nonmatrix_empty_and_complex_distances(distances, message):
    with pytest.raises(ValueError, match=message):
        _mds_from_distances(distances, 1)


@pytest.mark.parametrize(
    ("distances", "message"),
    [
        (jnp.zeros((1, 1)), "at least two"),
        (jnp.array([[0.0, -1.0], [-1.0, 0.0]]), "finite and nonnegative"),
        (jnp.array([[0.0, jnp.nan], [jnp.nan, 0.0]]), "finite and nonnegative"),
        (jnp.eye(2, dtype=jnp.complex64), "real-valued"),
    ],
)
def test_tsne_probability_validation_rejects_degenerate_distances(distances, message):
    with pytest.raises(ValueError, match=message):
        _tsne_probabilities(distances, perplexity=1.0)


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (
            lambda distances, bandwidth: jnp.ones_like(distances, dtype=jnp.complex64),
            "real-valued",
        ),
        (
            lambda distances, bandwidth: jnp.ones_like(distances).at[0, 1].set(jnp.nan),
            "finite",
        ),
        (lambda distances, bandwidth: jnp.triu(jnp.ones_like(distances)), "symmetric"),
    ],
)
def test_kernel_pca_rejects_complex_nonfinite_and_asymmetric_kernels(kernel, message):
    with pytest.raises(ValueError, match=message):
        kernel_pca(Euclidean(1), jnp.arange(4.0)[:, None], n_components=1, kernel=kernel)


@pytest.mark.parametrize(
    ("keyword", "value", "exception", "message"),
    [
        ("kernel", 3, TypeError, "callable"),
        ("allow_indefinite", 1, TypeError, "boolean"),
        ("bandwidth", jnp.inf, ValueError, "positive and finite"),
    ],
)
def test_kernel_pca_validates_callable_policy_and_finite_bandwidth(
    keyword, value, exception, message
):
    with pytest.raises(exception, match=message):
        kernel_pca(
            Euclidean(1),
            jnp.arange(4.0)[:, None],
            n_components=1,
            **{keyword: value},
        )


def test_kernel_pca_indefinite_policy_retains_only_the_positive_spectrum():
    values = jnp.arange(4.0)[:, None]

    def indefinite_kernel(distances, bandwidth):
        del bandwidth
        first = jnp.array([1.0, -1.0, 0.0, 0.0], dtype=distances.dtype)
        second = jnp.array([0.0, 0.0, 1.0, -1.0], dtype=distances.dtype)
        return jnp.outer(first, first) - jnp.outer(second, second)

    with pytest.raises(ValueError, match="not positive semidefinite"):
        kernel_pca(Euclidean(1), values, n_components=1, kernel=indefinite_kernel)

    result = kernel_pca(
        Euclidean(1),
        values,
        n_components=1,
        kernel=indefinite_kernel,
        allow_indefinite=True,
    )
    assert result.coordinates.shape == (4, 1)
    assert result.diagnostics["negative_eigenvalue_mass"] > 0.0


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (
            lambda distances, bandwidth: jnp.ones_like(distances, dtype=jnp.complex64),
            "real-valued query",
        ),
        (
            lambda distances, bandwidth: jnp.ones((distances.shape[0], distances.shape[1] - 1)),
            "finite query-by-training",
        ),
        (
            lambda distances, bandwidth: jnp.full_like(distances, jnp.nan),
            "finite query-by-training",
        ),
    ],
)
def test_kernel_pca_model_validates_query_kernel_contract(kernel, message):
    values = jnp.arange(4.0)[:, None]
    model = kernel_pca(Euclidean(1), values, n_components=1, bandwidth=1.0).model
    invalid = replace(model, kernel=kernel)
    with pytest.raises(ValueError, match=message):
        invalid.transform(jnp.array([[0.25], [1.25]]))


class _NonfiniteLogEuclidean(Euclidean):
    def log(self, x, y):
        return jnp.full_like(super().log(x, y), jnp.nan)


class _NonfiniteInnerEuclidean(Euclidean):
    def inner(self, x, u, v):
        return jnp.full_like(super().inner(x, u, v), jnp.nan)


class _NegativeInnerEuclidean(Euclidean):
    def inner(self, x, u, v):
        return -super().inner(x, u, v)


class _NonfiniteExpEuclidean(Euclidean):
    def exp(self, x, u):
        return jnp.full_like(super().exp(x, u), jnp.nan)


@pytest.mark.parametrize(
    ("manifold", "exception", "message"),
    [
        (_NonfiniteLogEuclidean(2), ValueError, "undefined logarithm"),
        (_NonfiniteInnerEuclidean(2), FloatingPointError, "nonfinite metric Gram"),
        (_NegativeInnerEuclidean(2), FloatingPointError, "not positive semidefinite"),
    ],
)
def test_pga_rejects_invalid_logarithms_and_metric_grams(manifold, exception, message):
    with pytest.raises(exception, match=message):
        principal_geodesic_analysis(
            manifold,
            curved_planar_data(),
            n_components=1,
            mean=jnp.zeros(2),
        )


def test_pga_model_guards_transform_logarithms_and_inverse_exponentials():
    values = curved_planar_data()
    fitted = principal_geodesic_analysis(
        Euclidean(2), values, n_components=1, mean=jnp.mean(values, axis=0)
    )

    invalid_transform = replace(fitted.model, manifold=_NonfiniteLogEuclidean(2))
    with pytest.raises(ValueError, match="undefined logarithm"):
        invalid_transform.transform(values)

    invalid_inverse = replace(fitted.model, manifold=_NonfiniteExpEuclidean(2))
    with pytest.raises(FloatingPointError, match="exponential-map domain"):
        invalid_inverse.inverse_transform(jnp.zeros((2, 1)))


@pytest.mark.parametrize(
    ("labels", "exception", "message"),
    [
        (np.array([0.0, 0.0, 1.0, np.nan]), ValueError, "NaN or infinite"),
        (np.array([0, None, 1, 1], dtype=object), ValueError, "missing values"),
        (
            np.array([0, "right", 0, "right"], dtype=object),
            TypeError,
            "mutually comparable",
        ),
    ],
)
def test_rmml_rejects_nonfinite_missing_and_incomparable_labels(labels, exception, message):
    values = curved_planar_data()[:4]
    with pytest.raises(exception, match=message):
        riemannian_metric_learning(Euclidean(2), values, labels)


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        (lambda points: jnp.asarray(0.0), "leading sample dimension"),
        (lambda points: points.astype(jnp.complex64), "real-valued"),
        (lambda points: jnp.zeros((points.shape[0], 0)), "at least one feature"),
    ],
)
def test_rmml_rejects_scalar_complex_and_zero_feature_embeddings(embedding, message):
    values = curved_planar_data()[:4]
    with pytest.raises(ValueError, match=message):
        riemannian_metric_learning(
            Euclidean(2),
            values,
            jnp.array([0, 0, 1, 1]),
            embedding=embedding,
        )


def test_rmml_default_product_embedding_validates_and_preserves_the_factor_tree():
    manifold = Product({"left": SphereExtrinsic(2), "right": SphereExtrinsic(3)})
    values = manifold.random_point(jax.random.key(305), sample_shape=(4,))
    labels = jnp.array([0, 0, 1, 1])
    model = riemannian_metric_learning(manifold, values, labels)
    assert model.metric.shape == (5, 5)

    embedding = _default_embedding(manifold)
    with pytest.raises(ValueError, match="Product factor pytree"):
        embedding((values["left"], values["right"]))
    with pytest.raises(LearningCapabilityError, match="equivariant embedding"):
        _default_embedding(object())


def test_rmml_rejects_nonfinite_scatter_from_extreme_but_finite_features():
    dtype = jnp.asarray(1.0).dtype
    huge = 2.0 * jnp.sqrt(jnp.finfo(dtype).max)
    values = jnp.asarray([[0.0], [huge], [1.0], [2.0]], dtype=dtype)
    with pytest.raises(FloatingPointError, match="pair-scatter matrices"):
        riemannian_metric_learning(Euclidean(1), values, jnp.array([0, 0, 1, 1]))


@pytest.mark.parametrize("replacement", ["zeros", "nan"])
def test_rmml_checks_the_finite_positive_definite_metric_certificate(monkeypatch, replacement):
    def invalid_exponential(matrix):
        if replacement == "zeros":
            return jnp.zeros_like(matrix)
        return jnp.full_like(matrix, jnp.nan)

    monkeypatch.setattr(metric_module, "_spd_exp", invalid_exponential)
    with pytest.raises(FloatingPointError, match="positive-definite metric"):
        riemannian_metric_learning(
            Euclidean(2),
            curved_planar_data()[:4],
            jnp.array([0, 0, 1, 1]),
        )


def test_metric_learning_model_validates_transforms_and_two_sample_distances():
    values = curved_planar_data()[:4]
    model = riemannian_metric_learning(Euclidean(2), values, jnp.array([0, 0, 1, 1]))

    wrong_dimension = replace(model, embedding=lambda points: points[:, :1])
    with pytest.raises(ValueError, match="feature dimension"):
        wrong_dimension.transform(values)
    with pytest.raises(ValueError, match="unbatched"):
        model.transform(jnp.stack([values, values]))

    cross_distances = model.pairwise_distances(values[:2], values[2:])
    assert cross_distances.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(cross_distances)))
