---
html_theme.sidebar_secondary.remove: true
---

# GeoJAX

GeoJAX is a JAX-native toolkit for computations on Riemannian manifolds. It
keeps the model simple: geometries provide exact primitives where available
and clearly labeled retraction proxies otherwise, while optimization algorithms
consume the common manifold protocol.

```python
import jax.numpy as jnp

from geojax.geometry import Sphere
from geojax.optimization import ConjugateGradient, Minimize

M = Sphere(size=3)
A = jnp.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.5]])

problem = Minimize(
    M=M,
    cost=lambda x: -x @ A @ x,
    solver=ConjugateGradient(verbosity=0),
    key=0,
)
eigenvector, value, info = problem.solve()
```

<div class="landing-grid">
  <a class="landing-card" href="getting_started/index.html">
    <div class="landing-card-icon" aria-hidden="true">
      <i class="fa-solid fa-rocket"></i>
    </div>
    <h2>Getting started</h2>
    <p>Install GeoJAX and solve a first optimization problem on a manifold.</p>
  </a>

  <a class="landing-card" href="tutorials/index.html">
    <div class="landing-card-icon" aria-hidden="true">
      <i class="fa-solid fa-flask"></i>
    </div>
    <h2>Tutorials</h2>
    <p>Work through executable mathematical examples with computed figures.</p>
  </a>

  <a class="landing-card" href="guide/index.html">
    <div class="landing-card-icon" aria-hidden="true">
      <i class="fa-solid fa-book-open"></i>
    </div>
    <h2>User guide</h2>
    <p>Understand geometry conventions, representations, and solver choices.</p>
  </a>

  <a class="landing-card" href="api/index.html">
    <div class="landing-card-icon" aria-hidden="true">
      <i class="fa-solid fa-code"></i>
    </div>
    <h2>API reference</h2>
    <p>Look up public geometries, optimization problems, and solver classes.</p>
  </a>
</div>

```{admonition} Scope
:class: note
GeoJAX is alpha software. The API favors explicit geometry and derivative
contracts while its expanding solver set is validated on small scientific
problems and executable tutorials.
```

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
tutorials/index
guide/index
api/index
development/index
```
