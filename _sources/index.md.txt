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

## Read by task

- **New to manifold optimization?** Start with [Getting started](getting_started/index.md).
- **Want a complete computation?** Open the [tutorials](tutorials/index.md); every result and figure is executed during the documentation build.
- **Need concepts and conventions?** Use the [user guide](guide/index.md).
- **Looking up a symbol?** Go directly to the [API reference](api/index.md).

```{admonition} Scope
:class: note
GeoJAX is alpha software. Geometry and first-order optimization are the most
mature layers; second-order methods and derivative-free solvers should be used
with the numerical limitations documented in this site.
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
