"""Regression checks for cancellation in conjugate-gradient directions."""

import jax.numpy as jnp
import numpy as np
import pytest

from geojax.geometry import Euclidean
from geojax.optimization import ConjugateGradient, Minimize
from geojax.optimization._conjugate_gradient import _compute_beta_and_direction


@pytest.mark.parametrize("scale", [1e-5, 1.0, 1e5])
def test_hestenes_stiefel_restarts_a_cancelled_nonstationary_direction(scale):
    m = Euclidean(1)
    old = jnp.array([0.1 * scale])
    new = jnp.array([-0.09 * scale])
    beta, direction = _compute_beta_and_direction(
        M=m,
        options=ConjugateGradient(verbosity=0),
        x=jnp.zeros(1),
        newx=jnp.ones(1),
        grad=old,
        newgrad=new,
        Pgrad=old,
        Pnewgrad=new,
        desc_dir=-old,
        gradPgrad=jnp.sum(old**2),
        newgradPnewgrad=jnp.sum(new**2),
        gradnorm=jnp.linalg.norm(old),
    )
    assert beta == 0.0
    np.testing.assert_allclose(direction, -new, rtol=1e-6)


def test_default_conjugate_gradient_converges_after_overshooting_a_scalar_minimum():
    problem = Minimize(
        M=Euclidean(1),
        cost=lambda x: 0.5 * jnp.sum(x * x) + 0.01 * jnp.sum(x**4),
        x0=jnp.array([0.05]),
        solver=ConjugateGradient(maxiter=100, verbosity=0),
    )
    point, cost, history = problem.solve()
    np.testing.assert_allclose(point, 0.0, atol=2e-6)
    assert cost < 2e-12
    assert history[-1].gradnorm <= 2e-6
