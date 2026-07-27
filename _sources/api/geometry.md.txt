# Geometry API

Every geometry is a directly importable class. Matrix geometries include the
metric in the class name so constructors follow the same convention as
`GrassmannProjection`. Capability metadata distinguishes exact geodesic
operations from numerical-local candidates and retraction proxies, and marks
whether automatic Hessian conversion is mathematically exact.

## Core method contract

For an array geometry with event shape `M.shape`, every point or tangent has
shape `batch_shape + M.shape`. Leading batch dimensions may broadcast; event
dimensions never broadcast and must match exactly. `Product` applies the same
rule independently to every leaf of its factor pytree.

| Member | Inputs | Result and contract |
|---|---|---|
| `belongs(x, atol=None)` | ambient point candidate | Boolean over the inferable batch shape. A malformed event shape returns `False`. Open constraints such as positive definiteness and ball membership remain strict. |
| `project(x)` | ambient array with exactly `M.shape` event axes | A finite manifold point satisfying `belongs(project(x))`. Malformed event axes raise `ValueError`; zero, indefinite, rank-deficient, and noisy values are valid repair inputs. |
| `is_tangent(x, u, atol=None)` | point and tangent candidate | Boolean over broadcast batch axes. Malformed or incompatible shapes return `False`. |
| `tangent_project(x, u)` | point and ambient vector | A tangent satisfying `is_tangent(x, tangent_project(x, u))`; malformed event axes raise `ValueError`. |
| `inner(x, u, v)` | point and two tangent vectors | The Riemannian inner product over broadcast batch axes. |
| `norm(x, u)` | point and tangent vector | The nonnegative metric norm. |
| `retr(x, u, t=1)` | point, tangent, and scalar or batched step | A manifold point obtained from the documented retraction. |
| `exp(x, u)` | point and tangent | Exact exponential, numerical-local operation, or retraction proxy according to `operation_kind("exp")`. |
| `invretr(x, y)` | base and endpoint | Local inverse of the advertised retraction. |
| `log(x, y)` | base and endpoint | Exact, numerical-local, or proxy displacement according to `operation_kind("log")`. Genuine cut loci follow each class's documented branch policy. |
| `squared_dist(x, y)` / `dist(x, y)` | two points | Scalar per batch element with the status reported by `operation_kind("dist")`. |
| `transport(x, y, u)` | endpoints and source tangent | A target tangent. `operation_kind("transport")` reports `parallel`, `isometric`, or `vector`. |
| `egrad_to_rgrad(x, egrad)` | point and ambient gradient | Metric-dual tangent gradient. |
| `random_point(key, sample_shape=())` | JAX key and static shape | Samples shaped `sample_shape + M.shape`. |
| `random_tangent(key, x, scale=1, normalize=False)` | JAX key and point | Tangent samples matching `x`; normalization uses the Riemannian norm before `scale`. |

Numerical repairs use dtype-aware interior margins. A configured `eps` is
never allowed to disappear through float32 rounding, and fixed-rank repairs
place active singular values above the numerical-rank threshold. These repair
conventions do not redefine exact maps at genuine cut loci or manifold
boundaries.

Capability declarations are conservative. `GeometryMixin` certifies no exact
or isometric operation by default; exact geometries opt into
`ExactGeometryMixin`, while retraction geometries use
`RetractionGeometryMixin`. Product metadata is certified only if every factor
provides the corresponding guarantee.

## Shared interface and vector geometries

```{eval-rst}
.. autoclass:: geojax.geometry.base.GeometryProtocol
   :members:

.. autoclass:: geojax.geometry.base.ManifoldProtocol
   :members:

.. autoclass:: geojax.geometry.base.GeometryMixin
   :members:

.. autoclass:: geojax.geometry.base.ExactGeometryMixin
   :members:

.. autoclass:: geojax.geometry.base.RetractionGeometryMixin
   :members:

.. autoclass:: geojax.geometry.Euclidean
   :members:

.. autoclass:: geojax.geometry.Oblique
   :members:

.. autoclass:: geojax.geometry.ProbabilitySimplex
   :members:

.. autoclass:: geojax.geometry.Sphere
   :members:

.. autoclass:: geojax.geometry.SphereExtrinsic
   :members:

.. autoclass:: geojax.geometry.PoincareBall
   :members:

.. autoclass:: geojax.geometry.Grassmann
   :members:

.. autoclass:: geojax.geometry.GrassmannProjection
   :members:

.. autoclass:: geojax.geometry.Stiefel
   :members:
   :inherited-members:

.. autoclass:: geojax.geometry.StiefelEuclidean
   :members:
   :inherited-members:

.. autoclass:: geojax.geometry.StiefelLogInfo
   :members:

.. autoclass:: geojax.geometry.GeneralizedStiefel
   :members:
   :inherited-members:

.. autoclass:: geojax.geometry.GeneralizedGrassmann
   :members:
   :inherited-members:

.. autoclass:: geojax.geometry.Hyperboloid
   :members:

.. autoclass:: geojax.geometry.Torus
   :members:

.. autoclass:: geojax.geometry.Product
   :members:
```

## Matrix Lie groups

```{eval-rst}
.. autoclass:: geojax.geometry.SpecialOrthogonal
   :members:

.. autoclass:: geojax.geometry.SpecialEuclidean
   :members:
```

## Positive-definite matrices

```{eval-rst}
.. autoclass:: geojax.geometry.SPDLogEuclidean
   :members:

.. autoclass:: geojax.geometry.SPDAffineInvariant
   :members:

.. autoclass:: geojax.geometry.SPDBuresWasserstein
   :members:
```

## Fixed-rank and semidefinite matrices

```{eval-rst}
.. autoclass:: geojax.geometry.FixedRank
   :members:
   :inherited-members:

.. autoclass:: geojax.geometry.RankKPSD
   :members:
   :inherited-members:

.. autoclass:: geojax.geometry.RankKPSDBuresWasserstein
   :members:
   :inherited-members:

.. autoclass:: geojax.geometry.Elliptope
   :members:
   :inherited-members:

.. autoclass:: geojax.geometry.Spectrahedron
   :members:
   :inherited-members:
```

## Correlation matrices

### Euclidean-Cholesky metric

```{eval-rst}
.. autoclass:: geojax.geometry.CorrelationECM
   :members:
```

### Log-Euclidean-Cholesky metric

```{eval-rst}
.. autoclass:: geojax.geometry.CorrelationLEC
   :members:
```

### Affine-invariant quotient metric

```{eval-rst}
.. autoclass:: geojax.geometry.CorrelationAffineQuotient
   :members:
   :inherited-members:
```

## Shape spaces

```{eval-rst}
.. autoclass:: geojax.geometry.KendallShape
   :members:
   :inherited-members:
```

## Utility functions

```{eval-rst}
.. autofunction:: geojax.geometry.torus.wrap_angles
```
