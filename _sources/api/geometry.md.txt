# Geometry API

Every geometry is a directly importable class. Matrix geometries include the
metric in the class name so constructors follow the same convention as
`GrassmannProjection`. Capability metadata distinguishes exact geodesic
operations from numerical-local candidates and retraction proxies, and marks
whether automatic Hessian conversion is mathematically exact.

## Shared interface and vector geometries

```{eval-rst}
.. autoclass:: geojax.geometry.base.GeometryProtocol
   :members:

.. autoclass:: geojax.geometry.base.ManifoldProtocol
   :members:

.. autoclass:: geojax.geometry.base.GeometryMixin
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
