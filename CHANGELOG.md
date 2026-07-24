# Changelog

## 0.2.0 - Unreleased

- Standardized native leading-batch semantics across every public geometry and
  added differentiable `squared_dist` operations.
- Stabilized zero-tangent exponentials, coincident logarithms, repeated SPD
  spectra, matrix divided differences, and numerical Stiefel batching while
  preserving each geometry's documented cut-locus branch policy.
- Added the pure-JAX `geojax.learning` namespace with pairwise squared
  distances, geodesic interpolation, and framework-neutral tangent maps,
  including exact-capability guards and combined time/endpoint batching.
- Classified Stiefel logarithms and distances as numerical-local endpoint
  solutions rather than globally certified shortest geodesics.
- Added exact Hessian capability metadata, the sphere shape-operator
  correction, and explicit errors when a second-order method requires an
  unsupported automatic conversion.
- Added exhaustive numerical stability and nested-batch tests plus compiled
  spherical and hyperbolic autoencoder release contracts.
- Added an executable deterministic-autoencoder tutorial comparing Euclidean,
  spherical, and hyperbolic latent spaces on handwritten digits.

## 0.1.1 - 2026-07-23

- Made the public optimization API class-only and removed the former
  functional solver and options-class exports.
- Defined and tested the JAX transformation contract for geometry primitives,
  batch helpers, automatic differentiation, and Hessian-vector products.
- Added explicit third-party notices for the tutorial data and documentation
  font, and finalized package authorship metadata for Kisung You.
- Added a local Python/JAX/precision test matrix with branch coverage and made
  executed documentation notebooks disposable build products.
- Stabilized Kendall shape distances near coincident shapes and made low-rank
  tangent validation account for dimension- and dtype-scaled roundoff.
- Added precision-aware invariant tests that retain strict float64 tolerances
  while exercising the complete suite under float32.

## 0.1.0 - 2026-07-21

- Added shared `ConstantStep`, `BacktrackingArmijo`, `AdaptiveArmijo`, and
  `StrongWolfe` line-search strategies and migrated the existing gradient
  solvers away from private Armijo implementations.
- Added `AdaptiveRegularizationCubics` and matrix-free `NewtonCG`.
- Added `LeastSquares` with Jacobian/adjoint products, together with
  `GaussNewton` and `LevenbergMarquardt`.
- Added `FiniteSum`, stochastic step schedules, and `StochasticGradient`.
- Added Product-specific `AlternatingGradient` and a visual solver-comparison
  tutorial.
- Added a dedicated optimization protocol page covering problem, derivative,
  solver-state, Hessian-vector, PyTree, and diagnostic contracts.
- Reworked the website wordmark in Sora, with the `JAX` name in the GeoJAX
  brand green, and removed the former wordmark-font assets.
- Expanded the GitHub-facing README and strengthened source-distribution
  metadata for documentation assets.
- Connected package, citation, README, and documentation metadata to the
  `kisungyou/geojax` repository.
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

- Initial alpha package structure for geometry, optimization, and examples.
