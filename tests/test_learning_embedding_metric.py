from __future__ import annotations

import jax
import jax.numpy as jnp
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
from geojax.learning._embedding import _mds_from_distances


def curved_planar_data():
    parameter = jnp.linspace(-1.0, 1.0, 12)
    return jnp.stack([parameter, parameter**2], axis=-1)


def test_classical_mds_recovers_euclidean_distances_and_reports_spectrum():
    manifold = Euclidean(2)
    values = curved_planar_data()
    result = classical_mds(manifold, values, n_components=2)
    embedded = jnp.linalg.norm(
        result.coordinates[:, None] - result.coordinates[None, :], axis=-1
    )
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

    product = Product({"a": Euclidean(1), "b": Euclidean(1)})
    product_values = {"a": values[:, :1], "b": values[:, 1:]}
    product_result = principal_geodesic_analysis(product, product_values, n_components=2)
    assert product_result.coordinates.shape == (12, 2)


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
    expected = jnp.diag(jnp.sqrt(jnp.diag(dissimilar) / jnp.diag(similar)))
    assert jnp.allclose(model.metric, expected, atol=2e-5)

    with pytest.raises(ValueError, match="balance"):
        riemannian_metric_learning(manifold, values, labels, balance=1.1)


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
    with pytest.raises(ValueError, match="bandwidth"):
        kernel_pca(manifold, values, bandwidth=0.0)
    with pytest.raises(ValueError, match="square"):
        kernel_pca(
            manifold,
            values,
            kernel=lambda distances, bandwidth: jnp.ones((distances.shape[0],)),
        )
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
    with pytest.raises(ValueError, match="n_neighbors"):
        phate(manifold, values, n_neighbors=0)
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
