from __future__ import annotations

import jax
import jax.numpy as jnp

from geojax.geometry import Product, Sphere, SPDLogEuclidean, Torus


def test_product_accepts_nested_pytrees():
    M = Product(
        {
            "direction": Sphere(size=3),
            "nested": {"phase": Torus(size=2), "cov": SPDLogEuclidean(size=(2, 2))},
        }
    )
    x = M.random_point(jax.random.key(0))
    u = M.random_tangent(jax.random.key(1), x, scale=0.01)
    y = M.exp(x, u)
    assert bool(M.belongs(y))
    assert bool(M.is_tangent(x, u))
    assert set(x) == {"direction", "nested"}
    assert set(x["nested"]) == {"phase", "cov"}


def test_product_rejects_wrong_tree_structure():
    M = Product({"a": Sphere(size=3), "b": Torus(size=2)})
    x = M.random_point(jax.random.key(0))
    bad = {"a": x["a"]}
    try:
        M.dist(x, bad)
    except ValueError:
        return
    raise AssertionError("Product should reject mismatched pytrees")


def test_product_inner_is_sum_of_factor_inners():
    M = Product({"a": Sphere(size=3), "b": Torus(size=2)})
    x = M.random_point(jax.random.key(0))
    u = M.random_tangent(jax.random.key(1), x)
    val = M.inner(x, u, u)
    expected = M.factors["a"].inner(x["a"], u["a"], u["a"]) + M.factors["b"].inner(
        x["b"], u["b"], u["b"]
    )
    assert jnp.allclose(val, expected)
