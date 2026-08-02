from __future__ import annotations

import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

import geojax.learning as learning
from geojax.geometry import (
    CorrelationAffineQuotient,
    CorrelationECM,
    CorrelationLEC,
    Elliptope,
    Euclidean,
    FixedRank,
    GeneralizedGrassmann,
    GeneralizedStiefel,
    Grassmann,
    GrassmannProjection,
    Hyperboloid,
    KendallShape,
    Oblique,
    PoincareBall,
    ProbabilitySimplex,
    Product,
    RankKPSD,
    RankKPSDBuresWasserstein,
    SPDAffineInvariant,
    SPDBuresWasserstein,
    SPDLogEuclidean,
    SpecialEuclidean,
    SpecialOrthogonal,
    Sphere,
    SphereExtrinsic,
    Spectrahedron,
    Stiefel,
    StiefelEuclidean,
    Torus,
)


def adapter_geometries():
    metric = jnp.diag(jnp.array([1.0, 1.5, 2.0, 3.0]))
    return [
        Euclidean(2),
        Oblique((4, 2)),
        ProbabilitySimplex(4),
        PoincareBall(3),
        Sphere(4),
        SphereExtrinsic(4),
        Grassmann((5, 2)),
        GrassmannProjection((5, 2)),
        GeneralizedStiefel((4, 2), metric=metric),
        GeneralizedGrassmann((4, 2), metric=metric),
        SPDLogEuclidean((3, 3)),
        SPDAffineInvariant((3, 3)),
        SPDBuresWasserstein((3, 3)),
        FixedRank((4, 3), rank=2),
        RankKPSD((4, 4), rank=2),
        RankKPSDBuresWasserstein((4, 4), rank=2),
        Elliptope((4, 4), rank=2),
        Spectrahedron((4, 4), rank=2),
        CorrelationECM((3, 3)),
        CorrelationLEC((3, 3)),
        CorrelationAffineQuotient((3, 3)),
        Hyperboloid(3),
        Torus(3),
        SpecialOrthogonal(3),
        SpecialEuclidean(2),
        Stiefel((4, 2)),
        StiefelEuclidean((4, 2)),
        KendallShape((4, 2)),
    ]


@pytest.mark.parametrize("manifold", adapter_geometries())
def test_every_exported_array_geometry_adapts_canonical_samples(manifold):
    values = manifold.random_point(jax.random.key(100), sample_shape=(2, 3))
    data = learning.as_manifold_data(manifold, values)

    assert data.n_samples == 3
    assert data.batch_shape == (2,)
    assert data.report.valid
    assert bool(jnp.all(manifold.belongs(data.values)))


def test_nested_product_adapter_preserves_factor_tree_and_shared_axes():
    manifold = Product(
        {
            "direction": Sphere(3),
            "state": (Torus(2), {"covariance": SPDLogEuclidean((2, 2))}),
        }
    )
    values = manifold.random_point(jax.random.key(101), sample_shape=(4,))
    data = learning.as_manifold_data(manifold, values)

    assert data.n_samples == 4
    assert jax.tree_util.tree_structure(data.values) == jax.tree_util.tree_structure(manifold.factors)
    assert bool(jnp.all(manifold.belongs(data.values)))


def test_adapter_moves_only_an_explicit_sample_axis():
    manifold = Sphere(3)
    canonical = manifold.random_point(jax.random.key(102), sample_shape=(5, 2))
    moved = jnp.moveaxis(canonical, 1, 0)
    data = learning.as_manifold_data(manifold, moved, sample_axis=0)

    assert data.batch_shape == (5,)
    assert data.n_samples == 2
    assert jnp.allclose(data.values, canonical)


def test_point_sequence_requires_explicit_representation():
    manifold = Sphere(3)
    points = [manifold.random_point(jax.random.key(index)) for index in range(3)]
    data = learning.as_manifold_data(manifold, points, representation="point_sequence")
    assert data.values.shape == (3, 3)
    assert data.n_samples == 3


@pytest.mark.parametrize(
    ("manifold", "values", "representation"),
    [
        (Sphere(3), jnp.array([[0.4, 1.2], [1.0, 2.0]]), "hyperspherical"),
        (
            Torus(2),
            jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, -1.0], [-1.0, 0.0]]]),
            "unit_circle",
        ),
        (Hyperboloid(3), jnp.array([[0.1, 0.2], [-0.2, 0.1]]), "poincare"),
        (ProbabilitySimplex(3), jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]), "logits"),
        (
            SPDLogEuclidean((2, 2)),
            jnp.array([[[1.0, 0.0], [0.2, 1.0]], [[2.0, 0.0], [0.0, 0.5]]]),
            "cholesky",
        ),
        (
            CorrelationECM((2, 2)),
            jnp.array([[[2.0, 0.3], [0.3, 1.0]], [[1.0, -0.2], [-0.2, 3.0]]]),
            "covariance",
        ),
        (SpecialOrthogonal(2), jnp.array([0.0, 1.0]), "angle"),
        (
            SpecialOrthogonal(3),
            jnp.array([[1.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.1, 0.3]]),
            "quaternion",
        ),
        (SpecialEuclidean(2), jnp.array([[0.2, 1.0, 2.0], [-0.1, 0.0, 0.5]]), "twist"),
        (KendallShape((4, 2)), jnp.arange(16.0).reshape(2, 4, 2), "raw_landmarks"),
    ],
)
def test_broad_coordinate_adapters_return_valid_canonical_points(
    manifold, values, representation
):
    data = learning.as_manifold_data(manifold, values, representation=representation)
    assert data.n_samples == 2
    assert bool(jnp.all(manifold.belongs(data.values)))


def test_frame_projector_and_low_rank_factor_adapters():
    grassmann = Grassmann((4, 2))
    frames = learning.as_manifold_data(
        grassmann,
        jax.random.normal(jax.random.key(103), (3, 4, 2)),
        representation="basis",
    ).values
    projectors = frames @ jnp.swapaxes(frames, -1, -2)
    recovered = learning.as_manifold_data(grassmann, projectors, representation="projector")
    assert bool(jnp.all(grassmann.belongs(recovered.values)))
    assert jnp.allclose(
        recovered.values @ jnp.swapaxes(recovered.values, -1, -2),
        projectors,
        atol=2e-5,
    )

    manifold = RankKPSD((4, 4), rank=2)
    factors = jax.random.normal(jax.random.key(104), (3, 4, 2))
    matrices = learning.as_manifold_data(manifold, factors, representation="factor")
    assert bool(jnp.all(manifold.belongs(matrices.values)))


def test_remaining_unambiguous_coordinate_adapters_reconstruct_points():
    simplex = ProbabilitySimplex(3)
    probabilities = learning.as_manifold_data(
        simplex,
        jnp.array([[2.0, 3.0, 5.0], [4.0, 1.0, 5.0]]),
        representation="positive",
    )
    assert jnp.allclose(jnp.sum(probabilities.values, axis=-1), 1.0)

    hyperboloid = Hyperboloid(3)
    hyperboloid_points = hyperboloid.random_point(jax.random.key(105), sample_shape=(3,))
    ball = PoincareBall(2)
    ball_points = learning.as_manifold_data(
        ball, hyperboloid_points, representation="hyperboloid"
    )
    assert bool(jnp.all(ball.belongs(ball_points.values)))

    spd = SPDLogEuclidean((2, 2))
    spd_points = spd.random_point(jax.random.key(106), sample_shape=(3,))
    from_logs = learning.as_manifold_data(spd, spd.logm(spd_points), representation="log")
    assert jnp.allclose(from_logs.values, spd_points, atol=2e-5)

    correlation = CorrelationECM((3, 3))
    correlation_points = correlation.random_point(jax.random.key(107), sample_shape=(3,))
    from_cholesky = learning.as_manifold_data(
        correlation,
        jnp.linalg.cholesky(correlation_points),
        representation="cholesky",
    )
    assert jnp.allclose(from_cholesky.values, correlation_points, atol=2e-5)

    rotation = SpecialOrthogonal(3)
    from_axis_angle = learning.as_manifold_data(
        rotation,
        jnp.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.05]]),
        representation="axis_angle",
    )
    assert bool(jnp.all(rotation.belongs(from_axis_angle.values)))


def test_lie_group_component_and_matrix_factor_adapters_reconstruct_points():
    rigid = SpecialEuclidean(2)
    rotations = SpecialOrthogonal(2).random_point(jax.random.key(108), sample_shape=(3,))
    translations = jax.random.normal(jax.random.key(109), (3, 2))
    components = learning.as_manifold_data(
        rigid,
        (rotations, translations),
        representation="rotation_translation",
    )
    assert jnp.allclose(components.values, rigid.from_components(rotations, translations))

    fixed_rank = FixedRank((4, 3), rank=2)
    matrices = fixed_rank.random_point(jax.random.key(110), sample_shape=(3,))
    left, singular_values, right = jnp.linalg.svd(matrices, full_matrices=False)
    reconstructed = learning.as_manifold_data(
        fixed_rank,
        (left[..., :2], singular_values[..., :2], right[..., :2, :]),
        representation="svd",
    )
    assert jnp.allclose(reconstructed.values, matrices, atol=2e-5)


@pytest.mark.parametrize(
    "manifold",
    [
        RankKPSD((4, 4), rank=2),
        RankKPSDBuresWasserstein((4, 4), rank=2),
        Elliptope((4, 4), rank=2),
        Spectrahedron((4, 4), rank=2),
    ],
)
def test_all_low_rank_psd_factor_adapters_return_members(manifold):
    factors = jax.random.normal(jax.random.key(111), (3, 4, 2))
    adapted = learning.as_manifold_data(manifold, factors, representation="factor")
    assert bool(jnp.all(manifold.belongs(adapted.values)))


def test_invalid_data_are_rejected_or_explicitly_repaired():
    manifold = Sphere(3)
    invalid = jnp.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    with pytest.raises(ValueError, match="do not belong"):
        learning.as_manifold_data(manifold, invalid)

    repaired = learning.as_manifold_data(manifold, invalid, repair=True)
    assert repaired.report.repaired_count == 2
    assert bool(jnp.all(manifold.belongs(repaired.values)))

    nonfinite = invalid.at[0, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="NaN or infinite"):
        learning.as_manifold_data(manifold, nonfinite)


def test_adapter_validation_levels_product_layouts_and_failed_repairs():
    manifold = Product({"direction": Sphere(2), "phase": Torus(1)})
    canonical = manifold.random_point(jax.random.key(112), sample_shape=(2, 3))
    moved = jax.tree_util.tree_map(lambda leaf: jnp.moveaxis(leaf, 1, 0), canonical)
    axes = {"direction": 0, "phase": 0}
    restored = learning.as_manifold_data(manifold, moved, sample_axis=axes)
    assert restored.batch_shape == (2,)
    assert restored.n_samples == 3

    with pytest.raises(ValueError, match="factor pytree"):
        learning.as_manifold_data(manifold, canonical, representation=("canonical", "canonical"))
    with pytest.raises(ValueError, match="factor pytree"):
        learning.as_manifold_data(manifold, (canonical["direction"], canonical["phase"]))
    with pytest.raises(ValueError, match="inconsistent sample counts"):
        learning.as_manifold_data(
            manifold,
            {"direction": canonical["direction"], "phase": canonical["phase"][:, :2]},
        )
    with pytest.raises(ValueError, match="inconsistent batch shapes"):
        learning.as_manifold_data(
            manifold,
            {"direction": canonical["direction"], "phase": canonical["phase"][:1]},
        )

    nonfinite = jnp.array([[jnp.nan, 0.0]])
    shape_only = learning.as_manifold_data(Euclidean(2), nonfinite, check="shape")
    assert shape_only.report.valid
    with pytest.raises(ValueError, match="NaN or infinite"):
        learning.as_manifold_data(Euclidean(2), nonfinite, check="finite")

    class NonRepairingGeometry:
        shape = (2,)

        def belongs(self, values):
            return jnp.zeros(values.shape[:-1], dtype=bool)

        def project(self, values):
            return values

    with pytest.raises(ValueError, match="do not belong"):
        learning.as_manifold_data(
            NonRepairingGeometry(), jnp.ones((2, 2)), repair=True
        )


def test_adapter_rejects_ambiguous_and_malformed_alternate_representations():
    with pytest.raises(ValueError, match="hyperspherical"):
        learning.as_manifold_data(Sphere(3), jnp.ones((2, 3)), representation="angles")
    with pytest.raises(ValueError, match="unit-circle"):
        learning.as_manifold_data(
            Torus(2), jnp.ones((2, 2, 3)), representation="unit_circle"
        )
    with pytest.raises(ValueError, match="hyperboloid coordinates"):
        learning.as_manifold_data(PoincareBall(2), jnp.ones((2, 2)), representation="hyperboloid")
    with pytest.raises(ValueError, match="nonnegative"):
        learning.as_manifold_data(
            ProbabilitySimplex(3), jnp.array([[1.0, -1.0, 1.0]]), representation="positive"
        )
    with pytest.raises(ValueError, match="positive row sums"):
        learning.as_manifold_data(
            ProbabilitySimplex(3), jnp.zeros((1, 3)), representation="positive"
        )
    with pytest.raises(ValueError, match="trailing shape"):
        learning.as_manifold_data(
            SpecialOrthogonal(3), jnp.ones((2, 3)), representation="quaternion"
        )
    with pytest.raises(ValueError, match="nonzero"):
        learning.as_manifold_data(
            SpecialOrthogonal(3), jnp.zeros((2, 4)), representation="quaternion"
        )
    with pytest.raises(ValueError, match="axis-angle"):
        learning.as_manifold_data(
            SpecialOrthogonal(3), jnp.ones((2, 2)), representation="axis_angle"
        )
    with pytest.raises(ValueError, match="rotation_translation"):
        learning.as_manifold_data(
            SpecialEuclidean(2), jnp.ones((2, 3)), representation="components"
        )
    with pytest.raises(ValueError, match="twists must end"):
        learning.as_manifold_data(
            SpecialEuclidean(2), jnp.ones((2, 2)), representation="twist"
        )
    rigid_four = SpecialEuclidean(4)
    with pytest.raises(ValueError, match="only for SE"):
        learning.as_manifold_data(
            rigid_four,
            jnp.ones((2, rigid_four.dim)),
            representation="twist",
        )
    with pytest.raises(ValueError, match="SVD factors"):
        learning.as_manifold_data(
            FixedRank((3, 3), rank=1), jnp.ones((2, 3, 3)), representation="svd"
        )
    with pytest.raises(ValueError, match="Unsupported representation"):
        learning.as_manifold_data(Sphere(2), jnp.ones((2, 2)), representation="quaternion")


def test_adapter_contract_rejects_invalid_registration_and_reconversion_options():
    with pytest.raises(TypeError, match="class"):
        learning.register_manifold_data_adapter(3, "scaled", lambda manifold, values: values)
    with pytest.raises(ValueError, match="non-canonical"):
        learning.register_manifold_data_adapter(Euclidean, "canonical", lambda manifold, values: values)
    with pytest.raises(TypeError, match="callable"):
        learning.register_manifold_data_adapter(Euclidean, "invalid", 3)
    with pytest.raises(ValueError, match="check must"):
        learning.as_manifold_data(Euclidean(2), jnp.ones((2, 2)), check="all")
    with pytest.raises(TypeError, match="Python sequence"):
        learning.as_manifold_data(
            Sphere(2), jnp.ones((2, 2)), representation="point_sequence"
        )

    adapted = learning.as_manifold_data(Euclidean(2), jnp.ones((2, 2)))
    assert learning.as_manifold_data(Euclidean(2), adapted) is adapted
    with pytest.raises(ValueError, match="cannot be converted again"):
        learning.as_manifold_data(Euclidean(2), adapted, repair=True)


def test_validation_reports_structural_failures_without_raising():
    report = learning.check_manifold_data(Sphere(3), jnp.zeros((4, 4)))
    assert not report.valid
    assert "event shape" in report.messages[0]


def test_custom_adapter_registration_is_explicit_and_non_overwriting():
    class TaggedEuclidean(Euclidean):
        pass

    learning.register_manifold_data_adapter(
        TaggedEuclidean,
        "halved",
        lambda manifold, values: jnp.asarray(values) / 2.0,
    )
    manifold = TaggedEuclidean(2)
    data = learning.as_manifold_data(
        manifold, jnp.array([[2.0, 4.0], [6.0, 8.0]]), representation="halved"
    )
    assert jnp.allclose(data.values, jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="already registered"):
        learning.register_manifold_data_adapter(
            TaggedEuclidean,
            "halved",
            lambda manifold, values: values,
        )


def test_removed_learning_names_have_no_compatibility_aliases():
    assert not hasattr(learning, "pairwise_squared_dist")
    assert not hasattr(learning, "geodesic_interpolate")
    assert not hasattr(learning, "tangent_map")


def test_learning_import_does_not_load_external_oracle_packages():
    script = """
import sys
import geojax.learning
banned = ('scipy', 'sklearn', 'phate', 'ot')
loaded = sorted(name for name in banned if name in sys.modules)
if loaded:
    raise SystemExit('unexpected runtime imports: ' + ', '.join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
