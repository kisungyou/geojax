"""Independent mathematical identities covering the September 2026 audit fixes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import (
    CorrelationECM,
    CorrelationLEC,
    Euclidean,
    Grassmann,
    Hyperboloid,
    Oblique,
    PoincareBall,
    ProbabilitySimplex,
    Product,
    Sphere,
    Torus,
)
from geojax.optimization import GaussNewton, LeastSquares, LevenbergMarquardt


def _coincident_cases():
    return [
        (Euclidean(2), jnp.zeros(2), jnp.array([1.0, 0.0])),
        (Sphere(3), jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])),
        (Torus(2), jnp.zeros(2), jnp.array([1.0, 0.0])),
        (Hyperboloid(3), jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])),
        (PoincareBall(2), jnp.zeros(2), jnp.array([0.5, 0.0])),
        (ProbabilitySimplex(2), jnp.array([0.5, 0.5]), jnp.array([0.5, -0.5])),
        (Oblique((2, 2)), jnp.eye(2), jnp.array([[0.0, 1.0], [1.0, 0.0]])),
        (CorrelationECM((2, 2)), jnp.eye(2), jnp.array([[0.0, 1.0], [1.0, 0.0]])),
        (CorrelationLEC((2, 2)), jnp.eye(2), jnp.array([[0.0, 1.0], [1.0, 0.0]])),
        (Grassmann((4, 2)), jnp.eye(4)[:, :2], jnp.eye(4)[:, 2:]),
        (
            Product((Euclidean(2), Torus(2))),
            (jnp.zeros(2), jnp.zeros(2)),
            (jnp.array([1.0, 0.0]), jnp.array([0.0, 2.0])),
        ),
    ]


@pytest.mark.parametrize(
    "manifold,point,tangent", _coincident_cases(), ids=lambda x: type(x).__name__
)
def test_squared_distance_has_the_metric_hessian_at_coincidence(manifold, point, tangent):
    # d(p, Exp_p(t v))^2 = t^2 <v,v> in a normal neighborhood.
    def loss(t):
        displacement = jax.tree_util.tree_map(lambda v: t * v, tangent)
        return manifold.squared_dist(point, manifold.exp(point, displacement))

    expected = 2.0 * manifold.inner(point, tangent, tangent)
    actual = jax.jit(jax.grad(jax.grad(loss)))(jnp.asarray(0.0))
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("angle", [1e-3, 1e-5, 1e-7, 1e-9])
def test_spherical_distances_resolve_small_angles(angle):
    manifold = Sphere(3)
    point = jnp.array([1.0, 0.0, 0.0])
    endpoint = jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
    np.testing.assert_allclose(manifold.dist(point, endpoint), angle, rtol=2e-6, atol=0)
    derivative = jax.grad(
        lambda a: manifold.squared_dist(point, jnp.array([jnp.cos(a), jnp.sin(a), 0.0]))
    )(jnp.asarray(angle))
    np.testing.assert_allclose(derivative, 2 * angle, rtol=2e-6, atol=0)


@pytest.mark.parametrize("geometry", [CorrelationECM, CorrelationLEC])
def test_correlation_identity_gradient_matches_scalar_chart(geometry):
    manifold = geometry((2, 2))
    target = jnp.array([[1.0, 0.2], [0.2, 1.0]])

    def objective(r):
        point = jnp.array([[1.0, r], [r, 1.0]])
        return manifold.squared_dist(point, target)

    # The single flat chart coordinate is r / sqrt(1-r^2).
    expected = -2.0 * 0.2 / np.sqrt(1.0 - 0.2**2)
    np.testing.assert_allclose(jax.grad(objective)(0.0), expected, rtol=2e-6)


@pytest.mark.parametrize("solver", [GaussNewton, LevenbergMarquardt])
def test_least_squares_respects_explicit_adjoint_with_opaque_residual(solver):
    # A callback supplies derivatives of a residual unavailable to autodiff.
    problem = LeastSquares(
        M=Euclidean(1),
        residual=lambda x: jax.lax.stop_gradient(x) - 2.0,
        jacobian_vec=lambda x, u: u,
        adjoint_jacobian=lambda x, z: z,
        x0=jnp.zeros(1),
        solver=solver(verbosity=0, tolgradnorm=1e-5),
    )
    _, gradient = problem.cost_and_grad(jnp.zeros(1))
    np.testing.assert_allclose(gradient, [-2.0])
    point, cost, _ = problem.solve()
    np.testing.assert_allclose(point, [2.0], atol=1e-5)
    assert cost < 1e-10


@pytest.mark.parametrize("product", [False, True])
@pytest.mark.parametrize(
    "method", ["frechet_mean", "frechet_median", "streaming_frechet_mean", "minibatch_frechet_mean"]
)
def test_zero_mass_antipodes_do_not_affect_intrinsic_summaries(product, method):
    from geojax import learning

    sphere = Sphere(3)
    point = jnp.array([1.0, 0.0, 0.0])
    points = jnp.stack([point, -point])
    manifold = Product({"sphere": sphere}) if product else sphere
    data = {"sphere": points} if product else points
    kwargs = {"sample_weight": jnp.array([1.0, 0.0])}
    if method == "minibatch_frechet_mean":
        kwargs.update(batch_size=2, epochs=2, key=0)
    result = getattr(learning, method)(manifold, data, **kwargs)
    fitted = result.point["sphere"] if product else result.point
    np.testing.assert_allclose(fitted, point, atol=1e-6)
    assert np.isfinite(result.gradient_norm)
    assert result.converged


def test_transport_initialization_preserves_mass_smaller_than_solver_tolerance():
    from geojax.learning._transport import _northwest_corner

    supply = jnp.array([0.5004, 0.4996])
    demand = jnp.array([0.5, 0.5])
    plan, _ = _northwest_corner(supply, demand, jnp.zeros((2, 2)), tolerance=0.001)
    np.testing.assert_allclose(plan.sum(axis=1), supply, atol=1e-7, rtol=0)
    np.testing.assert_allclose(plan.sum(axis=0), demand, atol=1e-7, rtol=0)


def test_sample_weights_normalize_at_the_floating_point_limit():
    from geojax.learning._utils import normalize_weights

    largest = jnp.finfo(jnp.asarray(1.0).dtype).max
    weights = normalize_weights(3, jnp.array([largest, largest / 2, 0.0]))
    np.testing.assert_allclose(weights, [2 / 3, 1 / 3, 0.0], rtol=2e-6)


def test_spherical_extrinsic_mean_is_invariant_to_large_weight_scaling():
    from geojax.geometry import SphereExtrinsic

    manifold = SphereExtrinsic(3)
    points = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    largest = jnp.finfo(points.dtype).max
    actual = manifold.extrinsic_mean(points, jnp.array([largest, largest]))
    np.testing.assert_allclose(actual, [1 / np.sqrt(2), 1 / np.sqrt(2), 0], rtol=2e-6)


def test_biswas_ghosh_rejects_nonrepresentable_statistics():
    from geojax.learning import biswas_ghosh_two_sample_test

    scale = 1e200 if jax.config.jax_enable_x64 else 1e20
    with pytest.raises(FloatingPointError, match="nonfinite statistic"):
        biswas_ghosh_two_sample_test(
            Euclidean(1),
            jnp.array([[0.0], [scale]]),
            jnp.array([[2 * scale], [3 * scale]]),
            key=0,
            n_permutations=9,
        )


@pytest.mark.parametrize("response_scale", [1e-5, 1.0, 1e5])
def test_local_linear_regression_reproduces_affine_extrapolation(response_scale):
    from geojax.learning import local_polynomial_regression

    predictors = jnp.linspace(-1.0, 1.0, 7)
    responses = response_scale * (2.0 + predictors)[:, None]
    model = local_polynomial_regression(
        Euclidean(1), predictors, responses, bandwidth=0.8, degree=1, maxiter=20
    )
    np.testing.assert_allclose(model.predict(1.5), [3.5 * response_scale], rtol=2e-5)


def test_transport_matches_independent_linear_programs():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    from geojax.learning._transport import _transportation_simplex

    rng = np.random.default_rng(390)
    for _ in range(12):
        costs = rng.uniform(size=(3, 4))
        a, b = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(4))
        constraints = np.vstack(
            [
                np.kron(np.eye(3), np.ones((1, 4))),
                np.kron(np.ones((1, 3)), np.eye(4)),
            ]
        )
        reference = scipy_optimize.linprog(
            costs.ravel(),
            A_eq=constraints,
            b_eq=np.r_[a, b],
            bounds=(0, None),
            method="highs",
        )
        assert reference.success
        result = _transportation_simplex(
            costs,
            a,
            b,
            tolerance=1e-10,
            max_pivots=1000,
        )
        assert result.converged
        np.testing.assert_allclose(result.cost, reference.fun, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(result.plan.sum(axis=1), a, atol=2e-7, rtol=0)
        np.testing.assert_allclose(result.plan.sum(axis=0), b, atol=2e-7, rtol=0)


def test_distance_kernel_is_invariant_to_large_unit_changes():
    from geojax.learning import kernel_mmd_two_sample_test

    manifold = Euclidean(1)
    x, y = jnp.array([[0.0], [1.0]]), jnp.array([[2.0], [3.0]])
    scale = 1e200 if jax.config.jax_enable_x64 else 1e20
    reference = kernel_mmd_two_sample_test(manifold, x, y, key=0, n_permutations=9)
    rescaled = kernel_mmd_two_sample_test(manifold, scale * x, scale * y, key=0, n_permutations=9)
    np.testing.assert_allclose(rescaled.statistic, reference.statistic, rtol=3e-6)
    np.testing.assert_allclose(
        rescaled.diagnostics["kernel_matrix"], reference.diagnostics["kernel_matrix"], rtol=3e-6
    )


def test_hyperboloid_exponential_acceleration_at_origin():
    manifold = Hyperboloid(3)
    point = jnp.array([1.0, 0.0, 0.0])
    tangent = jnp.array([0.0, 1.0, 0.0])
    acceleration = jax.jacfwd(jax.jacfwd(lambda t: manifold.exp(point, t * tangent)))(0.0)
    np.testing.assert_allclose(acceleration, point, atol=2e-6)


def test_torus_retains_sub_epsilon_angular_displacements():
    manifold = Torus(1)
    angle = jnp.array([1e-9])
    np.testing.assert_allclose(manifold.exp(jnp.zeros(1), angle), angle, atol=0, rtol=1e-6)
    np.testing.assert_allclose(manifold.dist(jnp.zeros(1), angle), angle[0], atol=0, rtol=1e-6)


@pytest.mark.parametrize(
    "solver_name", ["TrustRegions", "AdaptiveRegularizationCubics", "LevenbergMarquardt"]
)
def test_model_based_solvers_recover_from_invalid_trial_steps(solver_name):
    from geojax import optimization

    class LocalRetraction(Euclidean):
        def retr(self, point, tangent, t=1.0):
            step = t * tangent
            return jnp.where(jnp.linalg.norm(step) <= 0.2, point + step, jnp.nan)

    manifold = LocalRetraction(1)
    solver = getattr(optimization, solver_name)(verbosity=0, maxiter=100, tolgradnorm=1e-5)
    if solver_name == "LevenbergMarquardt":
        problem = LeastSquares(
            M=manifold, residual=lambda x: x - 0.5, x0=jnp.zeros(1), solver=solver
        )
    else:
        problem = optimization.Minimize(
            M=manifold, cost=lambda x: jnp.sum((x - 0.5) ** 2), x0=jnp.zeros(1), solver=solver
        )
    point, cost, history = problem.solve()
    np.testing.assert_allclose(point, [0.5], atol=2e-5)
    assert cost < 1e-9
    assert any(entry.extra.get("accepted") is False for entry in history[1:])
