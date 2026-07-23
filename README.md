<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/kisungyou/geojax/main/docs/_static/brand/geojax-gj-lockup-dark.png">
    <img src="https://raw.githubusercontent.com/kisungyou/geojax/main/docs/_static/brand/geojax-gj-lockup.png" alt="GeoJAX" width="280">
  </picture>
</p>

# GeoJAX

GeoJAX is a JAX-native toolkit for Riemannian geometry and manifold
optimization. Geometry objects provide manifold primitives, models provide
JAX scalar cost functions, and class-style solvers combine the two.

> [!IMPORTANT]
> GeoJAX is alpha software. The 0.1 series deliberately favors a coherent
> scientific API over backward compatibility.

## Highlights

- Exact geodesic operations where closed forms are available, with
  machine-readable retraction and transport proxies elsewhere.
- Matrix manifolds, Lie groups, hyperbolic spaces, shape spaces, low-rank
  models, and arbitrary pytree products.
- JAX autodifferentiation, JIT-compatible geometry primitives, composable
  batch helpers, and pytree-safe optimization state.
- Executable tutorial pages that place mathematical discussion, code, output,
  and figures in one document.

## Installation

GeoJAX requires Python 3.11 or newer. The human-facing project name is
**GeoJAX**; the Python package, repository slug, and PyPI distribution are all
lowercase **`geojax`**. Install it with:

**PyPI**

```bash
python -m pip install geojax
```

**GitHub source**

```bash
git clone https://github.com/kisungyou/geojax.git
cd geojax
python -m pip install .
```

For development, replace the final command with
`python -m pip install -e ".[dev,docs,examples]"`.

## Quick Start

The following problem minimizes a Rayleigh quotient on the unit sphere. Its
solution is a dominant eigenvector of `A`.

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
    key=0,
)
x_hat, final_cost, history = problem.solve()
```

`Sphere(size=3)` is the unit sphere embedded in `R^3`. `Minimize` obtains the
ambient derivative with JAX, converts it through the geometry, and delegates
the iteration to the selected solver.

## JAX Transformations

Geometry instances are static configuration objects. Their numerical protocol
methods accept array or pytree arguments and compose with `jax.jit`; random
generation takes explicit PRNG keys, and `exp_batch`, `log_batch`, and
`dist_batch` compose `jax.vmap` with JIT compilation. GeoJAX also supports
JAX-derived gradients and Hessian-vector products on array and Product states.

Solver `solve()` methods are deliberately Python drivers rather than
whole-solver JIT kernels. They perform stopping checks, callbacks, timing,
line-search control flow, and conversion of diagnostics to Python scalars.
Costs, derivative callbacks, and geometry operations used inside those drivers
may still be independently JIT compiled.

## Scientific Scope

The public geometry namespace includes Euclidean, sphere, oblique, simplex,
hyperbolic, torus, Grassmann, Stiefel, generalized orthogonality, Lie-group,
SPD, fixed-rank, elliptope, spectrahedron, correlation, Kendall-shape, and
pytree product geometries.

The public solver set is:

- `SteepestDescent`
- `ConjugateGradient`
- `TrustRegions`
- `BarzilaiBorwein`
- `LBFGS`
- `NewtonCG`
- `ParticleSwarm`
- `NelderMead`
- `AdaptiveRegularizationCubics`
- `GaussNewton` and `LevenbergMarquardt` for `LeastSquares`
- `StochasticGradient` for `FiniteSum`
- `AlternatingGradient` for `Product` geometries

Gradient solvers share public fixed-step, Armijo, adaptive Armijo, and
strong-Wolfe line-search strategies.

See the
[geometry guide](https://www.kisungyou.com/geojax/guide/geometry.html),
[optimization guide](https://www.kisungyou.com/geojax/guide/optimization.html),
and
[executable tutorials](https://www.kisungyou.com/geojax/tutorials/)
for the mathematical and computational conventions.

## Documentation

The documentation website is published at
[kisungyou.com/geojax](https://www.kisungyou.com/geojax/).

Install the documentation dependencies and build the complete site locally:

```bash
python -m pip install -e ".[docs,examples]"
make website
python -m http.server 8000 --directory site
```

The build executes every tutorial, fails on Sphinx warnings, and audits the
generated HTML for malformed mathematics and broken local references. Open
`http://127.0.0.1:8000` after the server starts.

## Development

```bash
python -m pip install -e ".[dev,docs,examples]"
make test
make test-float32
ruff check geojax tests
python -m build
python -m twine check dist/*
```

Before a release, `make test-matrix` exercises the supported Python versions,
the declared dependency floor, current dependencies, and both JAX precision
modes. See the
[testing guide](https://www.kisungyou.com/geojax/development/testing.html)
for the exact matrix.

The
[geometry protocol](https://www.kisungyou.com/geojax/development/geometry_protocol.html)
and
[optimization protocol](https://www.kisungyou.com/geojax/development/optimization_protocol.html)
describe the contracts expected from new implementations. Maintainers can
follow the
[release checklist](https://github.com/kisungyou/geojax/blob/main/RELEASING.md)
for manual TestPyPI and PyPI publication.

## Citation

Academic users can cite the project using
[`CITATION.cff`](https://github.com/kisungyou/geojax/blob/main/CITATION.cff).
Release history is recorded in
[`CHANGELOG.md`](https://github.com/kisungyou/geojax/blob/main/CHANGELOG.md).

## License

GeoJAX is released under the
[MIT License](https://github.com/kisungyou/geojax/blob/main/LICENSE).
Licenses and attribution for documentation data and fonts are recorded in
[`THIRD_PARTY_NOTICES.md`](https://github.com/kisungyou/geojax/blob/main/THIRD_PARTY_NOTICES.md).
