from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from geojax.geometry import (
    Euclidean,
    FixedRank,
    Hyperboloid,
    Product,
    Sphere,
    Stiefel,
    Torus,
)
from geojax.learning import (
    LearningCapabilityError,
    geodesic_interpolation,
    nearest_neighbors,
    pairwise_distances,
    tangent_space_map,
)


def test_pairwise_distances_squared_matches_explicit_sphere_computation():
    M = Sphere(size=3)
    x = M.random_point(jax.random.key(700), sample_shape=(2, 4))
    y = M.random_point(jax.random.key(701), sample_shape=(1, 5))

    actual = jax.jit(lambda left, right: pairwise_distances(M, left, right, squared=True))(x, y)
    expected = jax.vmap(
        lambda batch: jax.vmap(
            lambda left: jax.vmap(lambda right: M.squared_dist(left, right))(y[0])
        )(batch)
    )(x)

    assert actual.shape == (2, 4, 5)
    assert jnp.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    gradient = jax.grad(lambda left: jnp.sum(pairwise_distances(M, left, y, squared=True)))(x)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_pairwise_distances_supports_product_pytrees():
    M = Product({"direction": Sphere(size=3), "phase": Torus(size=2)})
    x = M.random_point(jax.random.key(710), sample_shape=(4,))
    y = M.random_point(jax.random.key(711), sample_shape=(3,))

    distances = pairwise_distances(M, x, y, squared=True)
    expected = pairwise_distances(
        M.factors["direction"], x["direction"], y["direction"], squared=True
    ) + pairwise_distances(M.factors["phase"], x["phase"], y["phase"], squared=True)

    assert distances.shape == (4, 3)
    assert jnp.allclose(distances, expected, atol=1e-6, rtol=1e-6)


def test_pairwise_distances_supports_nested_product_containers():
    M = Product(
        {
            "direction": Sphere(3),
            "state": (Torus(1), {"location": Euclidean(2)}),
        }
    )
    values = M.random_point(jax.random.key(712), sample_shape=(5,))
    distances = pairwise_distances(M, values)

    expected_squared = (
        pairwise_distances(M.factors["direction"], values["direction"], squared=True)
        + pairwise_distances(M.factors["state"][0], values["state"][0], squared=True)
        + pairwise_distances(
            M.factors["state"][1]["location"],
            values["state"][1]["location"],
            squared=True,
        )
    )
    assert distances.shape == (5, 5)
    assert jnp.allclose(distances**2, expected_squared, atol=2e-6, rtol=2e-6)


def test_pairwise_distances_validates_eager_inputs_but_remains_jittable():
    manifold = Sphere(3)
    valid = manifold.random_point(jax.random.key(713), sample_shape=(3,))
    invalid = valid.at[0].set(jnp.array([2.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="belong"):
        pairwise_distances(manifold, invalid)

    compiled = jax.jit(lambda points: pairwise_distances(manifold, points))
    result = compiled(valid)
    assert result.shape == (3, 3)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_pairwise_distances_uses_the_requested_distance_kernel_without_overflow():
    class OrdinaryDistanceOnly(Euclidean):
        def squared_dist(self, x, y):
            raise AssertionError("ordinary distances must not be computed by squaring first")

    class SquaredDistanceOnly(Euclidean):
        def dist(self, x, y):
            raise AssertionError("squared distances must use the squared-distance kernel")

    dtype = jnp.float32
    left = jnp.asarray([[0.0]], dtype=dtype)
    right = jnp.asarray([[1e20]], dtype=dtype)
    ordinary = pairwise_distances(
        OrdinaryDistanceOnly(1),
        left,
        right,
        block_size=1,
    )
    squared = pairwise_distances(
        SquaredDistanceOnly(1),
        jnp.asarray([[0.0], [2.0]], dtype=dtype),
        squared=True,
    )

    assert bool(jnp.all(jnp.isfinite(ordinary)))
    assert jnp.allclose(ordinary, jnp.asarray([[1e20]], dtype=dtype))
    assert jnp.allclose(
        squared,
        jnp.asarray([[0.0, 4.0], [4.0, 0.0]], dtype=dtype),
    )

    class ProxySquaredDistance(Euclidean):
        squared_dist_kind = "proxy"

    with pytest.raises(LearningCapabilityError, match="squared_dist=proxy"):
        pairwise_distances(ProxySquaredDistance(1), left, squared=True)


@pytest.mark.parametrize("M", [Sphere(size=3), Hyperboloid(size=3)])
def test_geodesic_interpolation_is_batched_jittable_and_has_correct_endpoints(M):
    x = M.random_point(jax.random.key(720))
    tangent = M.random_tangent(jax.random.key(721), x, scale=0.4)
    y = M.exp(x, tangent)
    times = jnp.linspace(0.0, 1.0, 9)

    path = jax.jit(lambda values: geodesic_interpolation(M, x, y, values))(times)

    assert path.shape == (9,) + M.shape
    assert bool(jnp.all(M.belongs(path)))
    assert jnp.allclose(path[0], x, atol=2e-6, rtol=2e-6)
    assert jnp.allclose(path[-1], y, atol=2e-6, rtol=2e-6)
    segment_lengths = M.dist(path[:-1], path[1:])
    assert jnp.allclose(segment_lengths, segment_lengths[0], atol=2e-5, rtol=2e-5)


def test_geodesic_interpolation_combines_time_and_endpoint_batch_axes():
    M = Sphere(size=3)
    x = M.random_point(jax.random.key(722), sample_shape=(2,))
    tangent = M.random_tangent(jax.random.key(723), x, scale=0.25)
    y = M.exp(x, tangent)
    times = jnp.linspace(0.0, 1.0, 7)

    path = jax.jit(geodesic_interpolation, static_argnums=0)(M, x, y, times)

    assert path.shape == (7, 2, 3)
    assert bool(jnp.all(M.belongs(path)))
    assert jnp.allclose(path[0], x, atol=2e-6, rtol=2e-6)
    assert jnp.allclose(path[-1], y, atol=2e-6, rtol=2e-6)


def test_geodesic_interpolation_rejects_invalid_points_and_times_eagerly():
    manifold = Sphere(3)
    x = jnp.array([1.0, 0.0, 0.0])
    y = jnp.array([0.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="belong"):
        geodesic_interpolation(manifold, 2.0 * x, y, 0.5)
    with pytest.raises(ValueError, match="finite real"):
        geodesic_interpolation(manifold, x, y, jnp.nan)


def test_learning_helpers_reject_retraction_proxy_geometry():
    M = FixedRank(size=(3, 3), rank=1)
    x = M.random_point(jax.random.key(724), sample_shape=(2,))
    y = M.random_point(jax.random.key(725), sample_shape=(2,))

    with pytest.raises(ValueError, match="requires exact manifold operations"):
        pairwise_distances(M, x, y)
    with pytest.raises(ValueError, match="requires exact manifold operations"):
        geodesic_interpolation(M, x[0], y[0], jnp.linspace(0.0, 1.0, 3))


def test_geodesic_interpolation_rejects_numerical_local_logarithm():
    M = Stiefel(size=(3, 2), log_maxiter=8)
    x = M.random_point(jax.random.key(726))
    y = M.exp(x, M.random_tangent(jax.random.key(727), x, scale=0.05))

    with pytest.raises(ValueError, match="log=numerical-local"):
        geodesic_interpolation(M, x, y, jnp.linspace(0.0, 1.0, 3))


def test_tangent_space_map_keeps_framework_parameters_outside_geojax():
    source = Euclidean(size=2)
    target = Sphere(size=3)
    source_base = jnp.zeros(2)
    target_base = jnp.array([1.0, 0.0, 0.0])
    matrix = jnp.array([[0.0, 0.0], [0.8, -0.2], [0.1, 0.7]])
    points = jnp.array([[0.1, 0.2], [-0.3, 0.4], [0.2, -0.1]])

    mapped = tangent_space_map(
        source,
        target,
        points,
        source_base=source_base,
        target_base=target_base,
        transform=lambda tangent: tangent @ matrix.T,
    )

    assert mapped.shape == (3, 3)
    assert bool(jnp.all(target.belongs(mapped)))
    jacobian = jax.jacrev(
        lambda value: tangent_space_map(
            source,
            target,
            value,
            source_base=source_base,
            target_base=target_base,
            transform=lambda tangent: tangent @ matrix.T,
        )
    )(points[0])
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_tangent_space_map_rejects_a_malformed_transform_output():
    source = Euclidean(2)
    target = Sphere(3)

    with pytest.raises(ValueError, match="target geometry"):
        tangent_space_map(
            source,
            target,
            jnp.array([0.1, 0.2]),
            source_base=jnp.zeros(2),
            target_base=jnp.array([1.0, 0.0, 0.0]),
            transform=lambda tangent: jnp.ones(4),
        )


@pytest.mark.parametrize("squared", [0, 1, None, "yes"])
def test_pairwise_distances_requires_an_actual_boolean_flag(squared):
    values = jnp.array([[0.0], [1.0]])

    with pytest.raises(TypeError, match="squared must be a boolean"):
        pairwise_distances(Euclidean(1), values, squared=squared)


def test_tangent_space_map_requires_a_callable_transform():
    with pytest.raises(TypeError, match="transform must be callable"):
        tangent_space_map(
            Euclidean(1),
            Euclidean(1),
            jnp.array([1.0]),
            source_base=jnp.array([0.0]),
            target_base=jnp.array([0.0]),
            transform=1.0,
        )


def test_nearest_neighbors_validates_flags_layout_and_requested_count():
    manifold = Euclidean(1)
    values = jnp.array([[0.0], [1.0], [3.0]])

    with pytest.raises(TypeError, match="exclude_self must be a boolean"):
        nearest_neighbors(manifold, values, n_neighbors=1, exclude_self=1)
    with pytest.raises(ValueError, match="unbatched"):
        nearest_neighbors(manifold, values[None, ...], n_neighbors=1)
    with pytest.raises(ValueError, match="between 1 and 2"):
        nearest_neighbors(manifold, values, n_neighbors=3)

    queries = jnp.array([[0.2], [2.8]])
    result = nearest_neighbors(
        manifold,
        values,
        queries,
        n_neighbors=1,
        exclude_self=False,
        block_size=2,
    )
    assert jnp.array_equal(result.indices[:, 0], jnp.array([0, 2]))


def test_geometry_helpers_reject_nonfinite_and_nontangent_intermediate_values():
    class NonfiniteLog(Euclidean):
        def log(self, x, y):
            return jnp.full(jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y)), jnp.nan)

    class NontangentLog(Sphere):
        def log(self, x, y):
            return jnp.ones(jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y)))

    with pytest.raises(ValueError, match="finite"):
        geodesic_interpolation(
            NonfiniteLog(1),
            jnp.array([0.0]),
            jnp.array([1.0]),
            0.5,
        )
    with pytest.raises(ValueError, match="tangent space"):
        geodesic_interpolation(
            NontangentLog(2),
            jnp.array([1.0, 0.0]),
            jnp.array([0.0, 1.0]),
            0.5,
        )


def test_geodesic_interpolation_rejects_complex_times():
    with pytest.raises(ValueError, match="finite real"):
        geodesic_interpolation(
            Euclidean(1),
            jnp.array([0.0]),
            jnp.array([1.0]),
            jnp.array(0.5 + 0.25j),
        )


@pytest.mark.parametrize("M", [Sphere(size=3), Hyperboloid(size=3)])
def test_compiled_deterministic_autoencoder_step_decreases_loss_and_preserves_latents(M):
    values = jnp.linspace(-1.0, 1.0, 64)
    data = jnp.stack(
        [
            values,
            values**2,
            jnp.sin(2.5 * values),
            jnp.cos(1.5 * values),
        ],
        axis=-1,
    )
    key_encoder, key_decoder = jax.random.split(jax.random.key(730))
    params = {
        "encoder_weight": 0.15 * jax.random.normal(key_encoder, shape=(4, 2)),
        "encoder_bias": jnp.zeros(2),
        "decoder_weight": 0.15 * jax.random.normal(key_decoder, shape=(3, 4)),
        "decoder_bias": jnp.zeros(4),
    }
    base = jnp.array([1.0, 0.0, 0.0])

    def encode(parameters, inputs):
        raw = inputs @ parameters["encoder_weight"] + parameters["encoder_bias"]
        capped = raw / jnp.sqrt(1.0 + jnp.sum(raw * raw, axis=-1, keepdims=True) / 1.5**2)
        tangent = jnp.concatenate([jnp.zeros(capped.shape[:-1] + (1,)), capped], axis=-1)
        return M.exp(base, tangent)

    def loss(parameters):
        latent = encode(parameters, data)
        reconstruction = latent @ parameters["decoder_weight"] + parameters["decoder_bias"]
        return jnp.mean((reconstruction - data) ** 2)

    @jax.jit
    def train_step(parameters):
        value, gradients = jax.value_and_grad(loss)(parameters)
        updated = jax.tree_util.tree_map(
            lambda parameter, gradient: parameter - 0.25 * gradient,
            parameters,
            gradients,
        )
        return updated, value

    initial_loss = loss(params)
    for _ in range(40):
        params, current_loss = train_step(params)
        assert bool(jnp.isfinite(current_loss))
    final_loss = loss(params)
    latents = encode(params, data)

    assert final_loss < 0.25 * initial_loss
    assert bool(jnp.all(M.belongs(latents)))
    assert bool(jnp.all(jnp.isfinite(jax.grad(loss)(params)["encoder_weight"])))


@pytest.mark.parametrize(
    "M",
    [Euclidean(size=3), Sphere(size=3), Hyperboloid(size=3)],
)
def test_intrinsic_graph_aggregation_is_compiled_differentiable_and_permutation_equivariant(
    M,
):
    base = jnp.array([1.0, 0.0, 0.0])
    coordinates = jnp.array(
        [
            [-0.35, 0.10],
            [-0.10, 0.25],
            [0.05, -0.20],
            [0.28, 0.18],
            [0.40, -0.12],
        ]
    )
    tangents = jnp.concatenate([jnp.zeros((5, 1)), coordinates], axis=-1)
    points = M.exp(base, tangents)
    adjacency = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    weights = adjacency / jnp.sum(adjacency, axis=1, keepdims=True)

    def aggregate(values, matrix):
        logarithms = M.log(values[:, None, :], values[None, :, :])
        messages = jnp.sum(matrix[..., None] * logarithms, axis=1)
        return M.exp(values, 0.35 * messages)

    def objective(values):
        return jnp.sum(M.squared_dist(base, aggregate(values, weights)))

    updated = jax.jit(aggregate)(points, weights)
    gradient = jax.grad(objective)(points)

    permutation = jnp.array([2, 0, 4, 1, 3])
    permuted = aggregate(
        points[permutation],
        weights[permutation][:, permutation],
    )

    assert bool(jnp.all(M.belongs(updated)))
    assert bool(jnp.all(jnp.isfinite(updated)))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert jnp.allclose(permuted, updated[permutation], atol=3e-6, rtol=3e-6)
