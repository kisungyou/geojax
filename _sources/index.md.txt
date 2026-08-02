---
html_theme.sidebar_secondary.remove: true
---

# GeoJAX

GeoJAX is a JAX-native toolkit for Riemannian geometry, manifold optimization,
and statistical and machine learning with manifold-valued data. Geometries
define representations and exact primitives where available; optimization and
learning methods consume that common protocol without flattening every
scientific object into an ordinary vector.

```python
import jax

from geojax.geometry import Sphere
from geojax.learning import as_manifold_data, frechet_mean, pairwise_distances

M = Sphere(size=3)
observations = M.random_point(jax.random.key(0), sample_shape=(32,))
data = as_manifold_data(M, observations)

center = frechet_mean(M, data).point
distances = pairwise_distances(M, data)
```

<div class="landing-grid">
  <a class="landing-card" href="getting_started/index.html">
    <div class="landing-card-icon" aria-hidden="true">
      <i class="fa-solid fa-rocket"></i>
    </div>
    <h2>Getting started</h2>
    <p>Install GeoJAX, validate manifold data, and solve a first geometric learning problem.</p>
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
    <p>Understand geometry conventions, learning workflows, and solver choices.</p>
  </a>

  <a class="landing-card" href="api/index.html">
    <div class="landing-card-icon" aria-hidden="true">
      <i class="fa-solid fa-code"></i>
    </div>
    <h2>API reference</h2>
    <p>Look up public geometries, manifold-learning methods, optimization problems, and solvers.</p>
  </a>
</div>

```{admonition} Scope
:class: note
GeoJAX is alpha software. Geometry capabilities, derivative contracts, and
learning-data validation are explicit. Optimization and learning methods are
tested on analytic cases, representative manifolds, and executable tutorials.
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
