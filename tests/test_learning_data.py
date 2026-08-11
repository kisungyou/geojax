from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

import geojax.learning as learning
from geojax.learning._data import _rotation_from_axis_angle
from geojax.learning._utils import (
    as_key,
    as_real_array,
    flatten_embedding,
    flatten_geometry_values,
    interval_control,
    normalize_weights,
    require_unbatched,
    scale_tangent_samples,
    stack_points,
    validate_manifold_point,
    validate_tangent_vector,
)
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
    assert jax.tree_util.tree_structure(data.values) == jax.tree_util.tree_structure(
        manifold.factors
    )
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
def test_broad_coordinate_adapters_return_valid_canonical_points(manifold, values, representation):
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
    ball_points = learning.as_manifold_data(ball, hyperboloid_points, representation="hyperboloid")
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


def test_axis_angle_adapter_is_differentiable_at_the_identity():
    vector = jnp.zeros(3)
    rotation = _rotation_from_axis_angle(vector)
    jacobian = jax.jacrev(_rotation_from_axis_angle)(vector)
    assert jnp.allclose(rotation, jnp.eye(3))
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_coordinate_adapters_reject_ambiguous_or_invalid_representations():
    with pytest.raises(ValueError, match="nonzero"):
        learning.as_manifold_data(Torus(1), jnp.zeros((2, 1, 2)), representation="unit_circle")
    with pytest.raises(ValueError, match="open unit ball"):
        learning.as_manifold_data(
            Hyperboloid(3), jnp.array([[1.0, 0.0]]), representation="poincare"
        )
    with pytest.raises(ValueError, match="upper unit hyperboloid"):
        learning.as_manifold_data(
            PoincareBall(2), jnp.array([[1.0, 1.0, 0.0]]), representation="hyperboloid"
        )
    with pytest.raises(ValueError, match="full column rank"):
        learning.as_manifold_data(
            Grassmann((3, 2)),
            jnp.array([[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]]),
            representation="basis",
        )
    with pytest.raises(ValueError, match="idempotent"):
        learning.as_manifold_data(
            Grassmann((3, 1)),
            jnp.array([jnp.diag(jnp.array([0.8, 0.2, 0.0]))]),
            representation="projector",
        )
    with pytest.raises(ValueError, match="lower triangular"):
        learning.as_manifold_data(
            SPDLogEuclidean((2, 2)),
            jnp.array([[[1.0, 0.2], [0.0, 1.0]]]),
            representation="cholesky",
        )
    with pytest.raises(ValueError, match="symmetric"):
        learning.as_manifold_data(
            SPDLogEuclidean((2, 2)),
            jnp.array([[[0.0, 1.0], [0.0, 0.0]]]),
            representation="log_matrix",
        )


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
    if isinstance(manifold, Elliptope):
        factors = factors / jnp.linalg.norm(factors, axis=-1, keepdims=True)
    if isinstance(manifold, Spectrahedron):
        factors = factors / jnp.linalg.norm(factors, axis=(-2, -1), keepdims=True)
    adapted = learning.as_manifold_data(manifold, factors, representation="factor")
    assert bool(jnp.all(manifold.belongs(adapted.values)))


def test_constrained_low_rank_factors_are_not_silently_repaired():
    factors = jax.random.normal(jax.random.key(112), (2, 4, 2))
    with pytest.raises(ValueError, match="unit-norm rows"):
        learning.as_manifold_data(Elliptope((4, 4), rank=2), factors, representation="factor")
    with pytest.raises(ValueError, match="Frobenius norm one"):
        learning.as_manifold_data(Spectrahedron((4, 4), rank=2), factors, representation="factor")


def test_integer_covariance_representation_is_converted_in_float_arithmetic():
    covariance = jnp.array([[[2, 1], [1, 2]]])
    adapted = learning.as_manifold_data(
        CorrelationECM((2, 2)), covariance, representation="covariance"
    )
    assert bool(jnp.all(CorrelationECM((2, 2)).belongs(adapted.values)))


def test_adapter_rank_and_spd_checks_are_relative_to_input_scale():
    tiny_basis = 1e-12 * jnp.array([[[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]])
    grassmann = learning.as_manifold_data(
        Grassmann((3, 2)),
        tiny_basis,
        representation="basis",
    )
    assert bool(jnp.all(Grassmann((3, 2)).belongs(grassmann.values)))

    tiny_covariance = 1e-12 * jnp.array([[[2.0, 0.5], [0.5, 1.0]]])
    correlation = learning.as_manifold_data(
        CorrelationECM((2, 2)),
        tiny_covariance,
        representation="covariance",
    )
    assert bool(jnp.all(CorrelationECM((2, 2)).belongs(correlation.values)))


def test_adapter_casts_integer_data_and_rejects_complex_data_without_loss():
    integer_data = learning.as_manifold_data(Euclidean(2), jnp.array([[1, 2], [3, 4]]))
    assert jnp.issubdtype(integer_data.values.dtype, jnp.floating)

    with pytest.raises(ValueError, match="real-valued"):
        learning.as_manifold_data(
            Euclidean(2),
            jnp.array([[1.0 + 1.0j, 2.0]]),
        )
    with pytest.raises(ValueError, match="requires check='belongs'"):
        learning.as_manifold_data(
            Euclidean(2),
            jnp.ones((2, 2)),
            check="shape",
            repair=True,
        )


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
        learning.as_manifold_data(NonRepairingGeometry(), jnp.ones((2, 2)), repair=True)


def test_adapter_rejects_ambiguous_and_malformed_alternate_representations():
    with pytest.raises(ValueError, match="hyperspherical"):
        learning.as_manifold_data(Sphere(3), jnp.ones((2, 3)), representation="angles")
    with pytest.raises(ValueError, match="unit-circle"):
        learning.as_manifold_data(Torus(2), jnp.ones((2, 2, 3)), representation="unit_circle")
    with pytest.raises(ValueError, match="hyperboloid coordinates"):
        learning.as_manifold_data(PoincareBall(2), jnp.ones((2, 2)), representation="hyperboloid")
    with pytest.raises(ValueError, match="strictly positive"):
        learning.as_manifold_data(
            ProbabilitySimplex(3), jnp.array([[1.0, -1.0, 1.0]]), representation="positive"
        )
    with pytest.raises(ValueError, match="strictly positive"):
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
        learning.as_manifold_data(SpecialEuclidean(2), jnp.ones((2, 2)), representation="twist")
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
        learning.register_manifold_data_adapter(
            Euclidean, "canonical", lambda manifold, values: values
        )
    with pytest.raises(TypeError, match="callable"):
        learning.register_manifold_data_adapter(Euclidean, "invalid", 3)
    with pytest.raises(ValueError, match="check must"):
        learning.as_manifold_data(Euclidean(2), jnp.ones((2, 2)), check="all")
    with pytest.raises(TypeError, match="Python sequence"):
        learning.as_manifold_data(Sphere(2), jnp.ones((2, 2)), representation="point_sequence")

    manifold = Euclidean(2)
    adapted = learning.as_manifold_data(manifold, jnp.ones((2, 2)))
    assert learning.as_manifold_data(manifold, adapted) is adapted
    with pytest.raises(ValueError, match="different geometry instance"):
        learning.as_manifold_data(Euclidean(2), adapted)
    with pytest.raises(ValueError, match="cannot be converted again"):
        learning.as_manifold_data(manifold, adapted, repair=True)


def test_adapted_data_upgrades_validation_before_stronger_reuse():
    manifold = Sphere(2)
    invalid = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    shape_only = learning.as_manifold_data(manifold, invalid, check="shape")

    assert shape_only.report.check == "shape"
    with pytest.raises(ValueError, match="do not belong"):
        learning.as_manifold_data(manifold, shape_only, check="belongs")


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


@pytest.mark.parametrize("representation", [None, 1, ("scaled",)])
def test_adapter_registration_requires_a_string_name(representation):
    class RegisteredEuclidean(Euclidean):
        pass

    with pytest.raises(TypeError, match="representation must be a string"):
        learning.register_manifold_data_adapter(
            RegisteredEuclidean,
            representation,
            lambda manifold, values: values,
        )


def test_adapter_registration_validates_flags_and_supports_explicit_overwrite():
    class RegisteredEuclidean(Euclidean):
        pass

    adapter = lambda manifold, values: jnp.asarray(values)  # noqa: E731
    with pytest.raises(ValueError, match="nonempty"):
        learning.register_manifold_data_adapter(RegisteredEuclidean, "", adapter)
    with pytest.raises(TypeError, match="overwrite must be a boolean"):
        learning.register_manifold_data_adapter(
            RegisteredEuclidean,
            "scaled",
            adapter,
            overwrite=1,
        )

    learning.register_manifold_data_adapter(RegisteredEuclidean, "scaled", adapter)
    learning.register_manifold_data_adapter(
        RegisteredEuclidean,
        "scaled",
        lambda manifold, values: 2.0 * jnp.asarray(values),
        overwrite=True,
    )
    result = learning.as_manifold_data(
        RegisteredEuclidean(1),
        jnp.array([[1.0], [2.0]]),
        representation="scaled",
    )
    assert jnp.array_equal(result.values, jnp.array([[2.0], [4.0]]))


def test_adapter_rejects_invalid_flags_empty_data_and_product_leaf_options():
    with pytest.raises(TypeError, match="repair must be a boolean"):
        learning.as_manifold_data(Euclidean(1), jnp.ones((2, 1)), repair=1)
    with pytest.raises(TypeError, match="representation must be a string"):
        learning.as_manifold_data(Euclidean(1), jnp.ones((2, 1)), representation=1)
    with pytest.raises(ValueError, match="at least one observation"):
        learning.as_manifold_data(Euclidean(1), jnp.empty((0, 1)))

    manifold = Product({"left": Euclidean(1), "right": Sphere(2)})
    values = manifold.random_point(jax.random.key(812), sample_shape=(2,))
    with pytest.raises(TypeError, match="Each Product representation must be a string"):
        learning.as_manifold_data(
            manifold,
            values,
            representation={"left": "canonical", "right": 1},
        )
    with pytest.raises(ValueError, match="factor pytree"):
        flatten_geometry_values(manifold, (values["left"], values["right"]), name="values")


def test_adapter_checks_membership_shape_and_failed_repair_contracts():
    class WrongMembershipShape:
        shape = (1,)

        def belongs(self, values):
            del values
            return jnp.array(True)

    with pytest.raises(ValueError, match="belongs returned shape"):
        learning.as_manifold_data(WrongMembershipShape(), jnp.ones((2, 1)))

    class NonfiniteRepair:
        shape = (1,)

        def belongs(self, values):
            return jnp.zeros(values.shape[:-1], dtype=bool)

        def project(self, values):
            return jnp.full_like(values, jnp.nan)

    with pytest.raises(ValueError, match="project.*NaN"):
        learning.as_manifold_data(NonfiniteRepair(), jnp.ones((2, 1)), repair=True)

    class WrongShapeAfterRepair:
        shape = (1,)

        def __init__(self):
            self.calls = 0

        def belongs(self, values):
            self.calls += 1
            if self.calls == 1:
                return jnp.zeros(values.shape[:-1], dtype=bool)
            return jnp.array(False)

        def project(self, values):
            return values

    with pytest.raises(ValueError, match="after repair"):
        learning.as_manifold_data(WrongShapeAfterRepair(), jnp.ones((2, 1)), repair=True)


@pytest.mark.parametrize(
    ("manifold", "values", "representation", "message"),
    [
        (Grassmann((3, 2)), jnp.ones((2, 2, 2)), "basis", "end in"),
        (
            Grassmann((3, 2)),
            jnp.array([[[jnp.nan, 0.0], [0.0, 1.0], [1.0, 0.0]]]),
            "basis",
            "finite",
        ),
        (SPDLogEuclidean((2, 2)), jnp.ones((1, 2, 3)), "cholesky", "end in"),
        (
            SPDLogEuclidean((2, 2)),
            jnp.array([[[jnp.nan, 0.0], [0.0, 1.0]]]),
            "cholesky",
            "finite",
        ),
        (
            SPDLogEuclidean((2, 2)),
            jnp.array([[[1.0, 0.0], [0.0, 0.0]]]),
            "cholesky",
            "positive diagonal",
        ),
        (SPDLogEuclidean((2, 2)), jnp.ones((1, 2, 3)), "log", "end in"),
        (
            SPDLogEuclidean((2, 2)),
            jnp.array([[[jnp.nan, 0.0], [0.0, 1.0]]]),
            "log",
            "finite",
        ),
        (Hyperboloid(3), jnp.ones((2, 3)), "poincare", "must end in"),
        (
            CorrelationECM((2, 2)),
            jnp.array([[[1.0, 0.0], [0.0, -1.0]]]),
            "covariance",
            "positive definite",
        ),
    ],
)
def test_coordinate_adapter_defensive_shape_and_domain_checks(
    manifold,
    values,
    representation,
    message,
):
    with pytest.raises(ValueError, match=message):
        learning.as_manifold_data(manifold, values, representation=representation)


def test_coordinate_adapter_covers_overflow_safe_simplex_and_se3_twists():
    largest = jnp.finfo(jnp.asarray(1.0).dtype).max
    probabilities = learning.as_manifold_data(
        ProbabilitySimplex(3),
        jnp.array([[largest, largest, largest]]),
        representation="positive",
    )
    assert jnp.allclose(probabilities.values, jnp.full((1, 3), 1.0 / 3.0))

    rigid = SpecialEuclidean(3)
    twists = jnp.array([[0.1, -0.2, 0.05, 1.0, -0.5, 0.25]])
    points = learning.as_manifold_data(rigid, twists, representation="twist")
    assert bool(jnp.all(rigid.belongs(points.values)))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda left, singular, right: (left[..., :-1, :], singular, right), "left factors"),
        (lambda left, singular, right: (left, singular[..., :-1], right), "singular values"),
        (lambda left, singular, right: (left, singular, right[..., :-1]), "right factors"),
        (
            lambda left, singular, right: (left, singular[0], right),
            "identical leading batch",
        ),
        (
            lambda left, singular, right: (left.at[0, 0, 0].set(jnp.nan), singular, right),
            "finite",
        ),
        (
            lambda left, singular, right: (left, singular.at[0, 0].set(0.0), right),
            "strictly positive",
        ),
        (
            lambda left, singular, right: (left.at[0, 0, 0].set(2.0), singular, right),
            "orthonormal columns",
        ),
        (
            lambda left, singular, right: (left, singular, right.at[0, 0, 0].set(2.0)),
            "orthonormal rows",
        ),
    ],
)
def test_fixed_rank_svd_adapter_validates_every_factor_contract(mutator, message):
    manifold = FixedRank((3, 2), rank=1)
    matrices = manifold.random_point(jax.random.key(813), sample_shape=(2,))
    left, singular, right = jnp.linalg.svd(matrices, full_matrices=False)
    factors = mutator(left[..., :1], singular[..., :1], right[..., :1, :])

    with pytest.raises(ValueError, match=message):
        learning.as_manifold_data(manifold, factors, representation="svd")


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (jnp.ones(2), "shape"),
        (jnp.array([1.0, jnp.nan, 1.0]), "finite"),
        (jnp.array([1.0, -1.0, 1.0]), "nonnegative"),
        (jnp.zeros(3), "positive total mass"),
        (jnp.array([1.0 + 1.0j, 1.0, 1.0]), "real-valued"),
    ],
)
def test_weight_normalization_rejects_invalid_public_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        normalize_weights(3, weights)


@pytest.mark.parametrize("key", [True, 1.5, "seed", object()])
def test_random_key_adapter_rejects_nonkeys_and_noninteger_seeds(key):
    with pytest.raises(TypeError, match="integer seed or JAX random key"):
        as_key(key, "method")


def test_random_key_adapter_accepts_seeds_and_rejects_key_batches():
    assert jax.random.key_data(as_key(7, "method")).shape == (2,)
    with pytest.raises(TypeError, match="not a key batch"):
        as_key(jax.random.split(jax.random.key(0), 2), "method")


@pytest.mark.parametrize(
    ("value", "kwargs", "exception", "message"),
    [
        (True, {}, TypeError, "not a boolean"),
        (jnp.inf, {}, ValueError, "finite"),
        (-0.1, {}, ValueError, "must lie"),
        (1.0, {"upper_closed": False}, ValueError, "must lie"),
        (0.0, {"lower_closed": False}, ValueError, "must lie"),
    ],
)
def test_interval_control_validates_real_scalar_domain(value, kwargs, exception, message):
    with pytest.raises(exception, match=message):
        interval_control(value, name="fraction", lower=0.0, upper=1.0, **kwargs)


def test_tree_utilities_validate_shapes_real_values_and_embeddings():
    with pytest.raises(ValueError, match="at least one"):
        stack_points(Euclidean(1), [])
    with pytest.raises(ValueError, match="coefficients must have shape"):
        scale_tangent_samples(Euclidean(1), jnp.ones((3, 1)), jnp.ones(2))
    with pytest.raises(TypeError, match="real numeric array"):
        as_real_array(object(), name="values")
    with pytest.raises(ValueError, match="real-valued"):
        as_real_array(jnp.array([1.0 + 1.0j]), name="values")
    with pytest.raises(ValueError, match="empty pytree"):
        flatten_embedding({})
    with pytest.raises(ValueError, match="leading sample dimension"):
        flatten_embedding(jnp.array(1.0))
    with pytest.raises(ValueError, match="real-valued"):
        flatten_embedding(jnp.ones((2, 1), dtype=complex))
    with pytest.raises(ValueError, match="finite coordinates"):
        flatten_embedding(jnp.array([[jnp.nan]]))
    with pytest.raises(ValueError, match="share their leading"):
        flatten_embedding((jnp.ones((2, 1)), jnp.ones((3, 1))))

    flattened = flatten_embedding((jnp.ones((2, 1)), jnp.zeros((2, 2))))
    assert flattened.shape == (2, 3)


def test_point_and_tangent_validators_wrap_structure_and_domain_errors():
    class BrokenGeometry:
        def belongs(self, point):
            del point
            raise TypeError("bad point tree")

        def is_tangent(self, base_point, tangent):
            del base_point, tangent
            raise ValueError("bad tangent tree")

    with pytest.raises(ValueError, match="contain only finite"):
        validate_manifold_point(Euclidean(1), jnp.array([jnp.nan]), name="point")
    with pytest.raises(ValueError, match="invalid manifold-point structure"):
        validate_manifold_point(BrokenGeometry(), jnp.array([0.0]), name="point")
    with pytest.raises(ValueError, match="must belong"):
        validate_manifold_point(Sphere(2), jnp.array([2.0, 0.0]), name="point")
    with pytest.raises(ValueError, match="contain only finite"):
        validate_tangent_vector(
            Euclidean(1),
            jnp.array([0.0]),
            jnp.array([jnp.nan]),
            name="tangent",
        )
    with pytest.raises(ValueError, match="invalid tangent-vector structure"):
        validate_tangent_vector(
            BrokenGeometry(),
            jnp.array([0.0]),
            jnp.array([0.0]),
            name="tangent",
        )
    with pytest.raises(ValueError, match="tangent space"):
        validate_tangent_vector(
            Sphere(2),
            jnp.array([1.0, 0.0]),
            jnp.array([1.0, 0.0]),
            name="tangent",
        )


def test_require_unbatched_reports_the_observed_batch_shape():
    with pytest.raises(ValueError, match=r"batch shape \(2,\)"):
        require_unbatched(SimpleNamespace(batch_shape=(2,)), "method")


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
