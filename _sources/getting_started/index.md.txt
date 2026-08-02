# Getting started

## Installation

GeoJAX requires Python 3.11 or newer. The distribution name, import package,
repository slug, and PyPI project are all lowercase `geojax`.

### Option 1: stable release from PyPI

PyPI currently provides GeoJAX 0.1.1. Pinning the version makes an experiment
reproducible and distinguishes the stable package from these development
documentation pages.

```bash
python -m pip install "geojax==0.1.1"
```

### Option 2: development version from GitHub

The `main` branch and this website document the unreleased 0.2.0 development
version, including `geojax.learning`.

```bash
git clone https://github.com/kisungyou/geojax.git
cd geojax
python -m pip install .
```

For an editable development installation with the complete test and
documentation toolchain, use

```bash
python -m pip install -e ".[dev,docs,examples]"
```

## First manifold-valued learning workflow

Learning methods begin with a geometry and a collection of observations. The
adapter validates the sample axis, event shape, finite values, and manifold
membership once; the resulting `ManifoldData` object can then be reused.

```python
import jax

from geojax.geometry import Sphere
from geojax.learning import as_manifold_data, frechet_mean, pairwise_distances

M = Sphere(size=3)
observations = M.random_point(jax.random.key(12), sample_shape=(24,))
data = as_manifold_data(M, observations)

summary = frechet_mean(M, data)
distances = pairwise_distances(M, data)

print(summary.point)
print(distances.shape)  # (24, 24)
```

For an array geometry, canonical data have shape
`batch_shape + (n_samples,) + M.shape`. Matrix-valued points keep their matrix
event shape, while Product observations keep their nested pytree. Explicit
adapters convert supported scientific representations such as SPD Cholesky
factors, Grassmann projectors, and Poincaré-ball coordinates.

The [learning guide](../guide/learning.md) explains validation levels,
capability requirements, and the distinction between differentiable
primitives and higher-level statistical algorithms.

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
