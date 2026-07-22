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
- JAX autodifferentiation, `jit`-compatible core operations, batch helpers,
  and pytree-safe optimization state.
- Executable tutorial pages that place mathematical discussion, code, output,
  and figures in one document.

## Installation

GeoJAX requires Python 3.11 or newer. After the first PyPI release, install it
with:

```bash
python -m pip install geojax
```

For development from a local checkout:

```bash
python -m pip install -e ".[dev,docs,examples]"
```

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
- `ParticleSwarm`
- `NelderMead`

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
pytest
ruff check geojax tests
python -m build
python -m twine check dist/*
```

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
