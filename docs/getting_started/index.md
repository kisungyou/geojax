# Getting started

## Installation

GeoJAX requires Python 3.11 or newer. Install the package from the project root:

```bash
python -m pip install -e .
```

## First optimization problem

A GeoJAX problem has three pieces:

1. a geometry `M`,
2. a scalar JAX cost function, and
3. a solver.

The example below minimizes a quadratic form on the unit sphere. Constraining
the vector to unit norm removes the otherwise irrelevant scale of an
eigenvector.

```python
import jax.numpy as jnp

from geojax.geometry import Sphere
from geojax.optimization import ConjugateGradient, Minimize

M = Sphere(size=3)
A = jnp.array(
    [[3.0, 1.0, 0.0],
     [1.0, 2.0, 0.0],
     [0.0, 0.0, 0.5]]
)

problem = Minimize(
    M=M,
    cost=lambda x: -x @ A @ x,
    solver=ConjugateGradient(verbosity=0),
    key=7,
)
x_hat, final_cost, history = problem.solve()
```

`Sphere(size=3)` is the two-dimensional sphere embedded in
$\mathbb{R}^3$. The cost is written with `jax.numpy`, so GeoJAX obtains its
ambient gradient through JAX and asks the geometry to convert it to a
Riemannian gradient.

## Geometry primitives

Every geometry exposes the same central operations:

```python
x = M.random_point(key=0)
u = M.random_tangent(key=1, x=x, scale=0.1)
y = M.exp(x, u)
u_recovered = M.log(x, y)
distance = M.dist(x, y)
```

The [geometry guide](../guide/geometry.md) explains the representation and
domain conventions. The [dominant eigenvector tutorial](../tutorials/sphere_eigenvector.md)
develops the optimization example with diagnostics and figures.
