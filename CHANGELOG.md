# Changelog

## 0.2.0 - Unreleased

- Added a self-provisioning, release-blocking test matrix for Python
  3.11--3.14, minimum and pinned stable JAX stacks, and both float32 and
  float64 modes, using managed interpreters, isolated coverage data,
  deterministic hash seeds, and a release-contract test preventing metadata
  drift.
- Enforced declared event shapes across every array geometry, added adversarial
  projection tests, and made zero, noisy, indefinite, and rank-deficient
  repairs satisfy the public membership contract in both precision modes.
- Added dtype-aware open-set and spectral margins, including stable float32
  Poincare-ball and hyperboloid conversions.
- Made capability metadata fail closed, validated Product factors against the
  structural geometry protocol, and shipped the `py.typed` marker.
- Corrected L-BFGS curvature reciprocals after non-isometric transport and
  corrected Nelder-Mead simplex scaling and full pairwise diameter reporting.
- Corrected Fisher--Rao simplex tangent projection, stabilized sphere and
  Oblique logarithms and parallel transport near but outside the cut locus,
  and made trust-region residual diagnostics report the terminal inner state.
- Corrected Dubey--Müller Fréchet-ANOVA scaling, weighted trimming by retained
  probability mass, and streaming means so prior mass is always explicit.
- Bound validated learning datasets to their geometry instance, upgraded weak
  validation before stronger reuse, and tightened kernel-PCA, t-SNE, PHATE,
  and disconnected-Isomap numerical contracts.
- Distinguished the stable PyPI package from the unreleased documentation and
  expanded the method-level geometry API contract.
- Standardized native leading-batch semantics across every public geometry and
  added differentiable `squared_dist` operations.
- Stabilized zero-tangent exponentials, coincident logarithms, repeated SPD
  spectra, matrix divided differences, and numerical Stiefel batching while
  preserving each geometry's documented cut-locus branch policy.
- Expanded `geojax.learning` into an adapter-first, manifold-independent layer
  with canonical `ManifoldData`, alternate representation conversion, exact
  capability guards, structured immutable results, geometric primitives,
  intrinsic summaries, kernel regression, inference, clustering, dimension
  reduction, exact empirical transport, optional OTT-JAX Sinkhorn divergence,
  and equivariant-embedding metric learning.
- Added manifold-independent nearest-centroid, k-nearest-neighbor, tangent
  logistic, LDA, and QDA classifiers using metric-orthonormal tangent
  coordinates.
- Added geodesic and local polynomial Fréchet regression for manifold-valued
  responses, bootstrap mean regions, energy and PSD-checked MMD tests, and a
  paired tangent-displacement test.
- Added streaming and mini-batch Fréchet means, mini-batch intrinsic k-means,
  geodesic barycentric coding, and Product-manifold dictionary optimization.
- Added trimmed Fréchet means, Huber/Cauchy/Tukey geodesic M-estimators,
  intrinsic spatial depth, metric midranks, graph label propagation, and
  transductive manifold-regularized regression.
- Made tangent-rank and kernel-PSD certification dtype-aware, kept weighted
  dictionary and trimmed objectives internally consistent, and guarded
  zero-mass cluster updates and duplicate-point graph neighborhoods.
- Replaced `pairwise_squared_dist`, `geodesic_interpolate`, and `tangent_map`
  with the coherent public names `pairwise_distances`,
  `geodesic_interpolation`, and `tangent_space_map`, without compatibility
  aliases.
- Migrated the SPD Fréchet-mean, Kendall-shape, and EEG tutorials from local
  helper implementations to the public learning API.
- Added executable visual tutorials for Product data adaptation, robust
  spherical summaries, broad clustering comparisons, SPD PGA and
  cross-validated regression, two-sample inference, exact circular optimal
  transport, six manifold dimension-reduction methods, and supervised
  equivariant-embedding metric learning.
- Rebalanced visualization tutorials across hyperbolic, toroidal, SPD, and
  Grassmann geometries instead of relying predominantly on spheres.
- Reframed project metadata and first-use documentation around three equal
  scientific layers: Riemannian geometry, manifold optimization, and learning
  with manifold-valued observations.
- Classified Stiefel logarithms and distances as numerical-local endpoint
  solutions rather than globally certified shortest geodesics.
- Added exact Hessian capability metadata, the sphere shape-operator
  correction, and explicit errors when a second-order method requires an
  unsupported automatic conversion.
- Added exhaustive numerical stability and nested-batch tests plus compiled
  spherical and hyperbolic autoencoder release contracts.
- Added an executable deterministic-autoencoder tutorial comparing Euclidean,
  spherical, and hyperbolic latent spaces on handwritten digits.
- Added a real-data geometric-learning tutorial that derives SPD covariance
  descriptors from an attributed PhysioNet motor-imagery EEG subset and
  compares log-Euclidean, affine-invariant, and Bures--Wasserstein prototype
  heads under a recording-run holdout protocol.

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
