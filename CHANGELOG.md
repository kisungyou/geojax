# Changelog

## Unreleased

- Added a dedicated optimization protocol page covering problem, derivative,
  solver-state, Hessian-vector, PyTree, and diagnostic contracts.
- Reworked the website wordmark in Sora, with the `JAX` name in the GeoJAX
  brand green, and removed the former wordmark-font assets.
- Expanded the GitHub-facing README and strengthened source-distribution
  metadata for documentation assets.
- Connected package, citation, README, and documentation metadata to the
  `kisungyou/GeoJAX` repository.
- Expanded the SPD Fréchet-mean tutorial into a visual comparison of the
  log-Euclidean, affine-invariant, and Bures-Wasserstein geometries.
- Added an executable Kendall-shape tutorial using the public SHREC'17 hand
  landmark subset distributed by Geomstats.
- Added `Oblique`, Fisher--Rao `ProbabilitySimplex`, and `PoincareBall`.
- Added `FixedRank`, `RankKPSD`, `RankKPSDBuresWasserstein`, `Elliptope`, and
  `Spectrahedron` with ambient matrix representations.
- Added `GeneralizedStiefel` and `GeneralizedGrassmann` for SPD-weighted
  orthogonality constraints.
- Added the affine-invariant quotient geometry `CorrelationAffineQuotient` and
  regular landmark quotient geometry `KendallShape`.
- Split the geometry contract into `ManifoldProtocol` and `GeometryProtocol`.
  Retraction-only classes expose Manopt-style compatibility proxies with
  machine-readable exactness and transport metadata.
- Added `Stiefel` with the canonical quotient metric and `StiefelEuclidean`
  with the embedded Frobenius metric, both using orthonormal-frame points,
  exact exponential maps, convergence-reporting numerical logarithms, and
  isometric group-action vector transport.
- Added `SpecialOrthogonal` with matrix-group operations, differentiable local
  logarithms, and exact parallel transport for the Frobenius bi-invariant
  metric.
- Added `SpecialEuclidean` with homogeneous-matrix points, canonical product
  geometry, and separate Riemannian and Lie-group exponential/logarithm maps.
- Added `SPDBuresWasserstein`, including the metric, optimal Gaussian transport
  map, geodesic operations, distance, and an isometric optimization transport.
- Added a self-contained visual tutorial for rigid landmark registration in
  `SE(2)`.
## 0.2.0 - 2026-05-02

- Standardized geometry constructors and the class-style optimization API.
- Added geometry protocol helpers, batch helpers, Hyperboloid, and Torus.
- Added pytree Product manifolds, benchmarks, and tests.
- Added release metadata and a concise project README.
- Replaced dotted matrix-geometry namespaces with the direct classes
  `SPDLogEuclidean`, `SPDAffineInvariant`, `CorrelationECM`, and
  `CorrelationLEC`.
- Added `SphereExtrinsic` and changed `GrassmannProjection` to expose
  orthonormal-frame points while evaluating its geometry in projector
  coordinates.

## 0.1.0

- Initial alpha package structure for geometry, optimization, and examples.
