"""Analytic derivatives of spectral repair away from its zero-eigenvalue boundary."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry.spd import _spd_project_differentiable


def _tol():
    return 3e-10 if jax.config.jax_enable_x64 else 4e-5


@pytest.mark.parametrize("other_diagonal", [(-1.0, 2.0), (-1.0, -1.0), (0.0, 2.0)])
def test_mixed_repair_batch_does_not_change_valid_item_hessian(other_diagonal):
    other = jnp.diag(jnp.asarray(other_diagonal))

    def objective(scale):
        matrices = jnp.stack([scale * jnp.eye(2), other])
        return jnp.trace(_spd_project_differentiable(matrices, 0.25)[0])

    gradient = jax.jit(jax.grad(objective))(1.0)
    hessian = jax.jit(jax.jacfwd(jax.grad(objective)))(1.0)
    np.testing.assert_allclose(gradient, 2.0, rtol=_tol(), atol=_tol())
    np.testing.assert_allclose(hessian, 0.0, atol=_tol())


def test_repeated_positive_negative_spectra_have_exact_trace_hessian():
    point = jnp.diag(jnp.array([2.0, 2.0, -1.0, -1.0]))
    direction = jnp.array(
        [[0.1, 0.2, 0.3, -0.2], [0.2, -0.4, 0.5, 0.1], [0.3, 0.5, 0.2, 0.4], [-0.2, 0.1, 0.4, -0.1]]
    )

    def objective(matrix):
        return jnp.trace(_spd_project_differentiable(matrix, 0.25))

    gradient = jax.grad(objective)
    expected_gradient = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0]))
    cross_signs = jnp.array([True, True, False, False])
    expected_hvp = jnp.where(cross_signs[:, None] != cross_signs[None, :], direction / 3.0, 0.0)
    np.testing.assert_allclose(gradient(point), expected_gradient, rtol=_tol(), atol=_tol())
    _, hvp = jax.jit(lambda p, e: jax.jvp(gradient, (p,), (e,)))(point, direction)
    np.testing.assert_allclose(hvp, expected_hvp, rtol=_tol(), atol=_tol())


def test_repaired_rotation_curve_forward_reverse_and_hessian():
    diagonal = jnp.diag(jnp.array([2.0, 2.0, -1.0, -1.0]))
    repaired = jnp.diag(jnp.array([2.0, 2.0, 0.25, 0.25]))
    generator = jnp.zeros((4, 4)).at[0, 3].set(-1.0).at[3, 0].set(1.0)

    def rotation(t):
        return jnp.eye(4) + jnp.sin(t) * generator + (1 - jnp.cos(t)) * (generator @ generator)

    def actual(t):
        frame = rotation(t)
        return _spd_project_differentiable(frame @ diagonal @ frame.T, 0.25)

    def expected(t):
        frame = rotation(t)
        return frame @ repaired @ frame.T

    times = jnp.array([0.0, 0.2])
    np.testing.assert_allclose(
        jax.vmap(actual)(times), jax.vmap(expected)(times), rtol=_tol(), atol=_tol()
    )
    for derivative in (jax.jacfwd, jax.jacrev, lambda f: jax.jacfwd(jax.jacrev(f))):
        computed = jax.jit(jax.vmap(derivative(actual)))(times)
        reference = jax.vmap(derivative(expected))(times)
        np.testing.assert_allclose(computed, reference, rtol=_tol(), atol=_tol())


def test_repeated_negative_spectrum_is_locally_constant_under_vmap():
    points = jnp.stack([-jnp.eye(2), -2.0 * jnp.eye(2)])
    direction = jnp.array([[0.2, 0.3], [0.3, -0.1]])

    def objective(point):
        return jnp.sum(_spd_project_differentiable(point, 0.25))

    def hessian_action(point):
        return jax.jvp(jax.grad(objective), (point,), (direction,))[1]

    np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(objective)))(points), 0.0, atol=_tol())
    np.testing.assert_allclose(jax.jit(jax.vmap(hessian_action))(points), 0.0, atol=_tol())


def test_zero_eigenvalue_preserves_previous_selected_first_derivative():
    point = jnp.diag(jnp.array([0.0, 2.0]))
    direction = jnp.array([[0.3, 0.2], [0.2, -0.1]])
    # At zero the repair is discontinuous. This tests only the existing
    # selected first-derivative convention, not a mathematical Hessian there.
    _, actual = jax.jvp(lambda p: _spd_project_differentiable(p, 0.25), (point,), (direction,))
    loewner = jnp.array([[0.0, 0.875], [0.875, 1.0]])
    np.testing.assert_allclose(actual, loewner * direction, rtol=_tol(), atol=_tol())


def test_mixed_sign_maximum_scale_projection_derivative():
    largest = jnp.finfo(jnp.asarray(1.0).dtype).max
    point = jnp.diag(jnp.array([largest, -largest]))
    direction = jnp.array([[0.3, 0.4], [0.4, -0.2]])
    _, derivative = jax.jit(
        lambda p, e: jax.jvp(lambda matrix: _spd_project_differentiable(matrix, 0.25), (p,), (e,))
    )(point, direction)
    # The positive/negative divided difference tends to 1/2; eps/(2*largest)
    # is far below the rounding resolution of this representable coefficient.
    expected = jnp.array([[0.3, 0.2], [0.2, 0.0]])
    np.testing.assert_allclose(derivative, expected, rtol=_tol(), atol=_tol())
    gradient = jax.jit(jax.grad(lambda p: jnp.trace(_spd_project_differentiable(p, 0.25))))(point)
    np.testing.assert_allclose(gradient, jnp.diag(jnp.array([1.0, 0.0])), atol=_tol())
