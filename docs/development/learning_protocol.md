# Learning protocol

The `geojax.learning` namespace is a manifold-independent statistical layer,
not a neural-network framework. It may depend at runtime on JAX and NumPy only;
OTT-JAX is isolated in the optional `ot` extra. SciPy, scikit-learn, POT, and
PHATE may be test or documentation oracles but cannot be imported by the
runtime package. This boundary follows the separation between geometric
primitives and model-level learning used throughout geometric machine learning
{cite:p}`bronstein2021geometric`.

## Data boundary

Every high-level method starts with `(manifold, data, ...)`. Raw canonical data
are converted through `as_manifold_data`; callers can adapt once and reuse the
immutable result. New methods must not implement private shape conventions or
silently normalize points.

`ManifoldData` is bound by identity to the geometry instance that performed
validation. A method must pass an adapted object through `as_manifold_data`
again so the binding and requested validation level are enforced. It must not
short-circuit merely because the input already has type `ManifoldData`.

An adapter must:

- preserve the mathematical point represented by an alternate coordinate;
- place the sample axis immediately before event dimensions;
- preserve the Product factor pytree and shared batch/sample axes;
- reject nonfinite values before numerical kernels; and
- distinguish representation conversion from explicit `repair=True`.

External representations register through
`register_manifold_data_adapter(geometry_type, name, converter)`. Names are
explicit; automatic representation inference is prohibited.

## Capability boundary

Algorithms declare their required exact operations at entry. A distance-based
method asks for exact `dist`; intrinsic center updates ask for exact `dist`,
`log`, and `exp`; RMML asks for an equivariant embedding. Proxy or
numerical-local operations raise `LearningCapabilityError` and are never
enabled through an undocumented fallback.

Methods that introduce an additional analytic object must validate that object
separately. In particular, a kernel method cannot infer positive
semidefiniteness from the existence of a geodesic distance, a tangent model
must construct coordinates with `M.inner`, and a graph method must distinguish
transductive vertex predictions from an out-of-sample model.

## Transformation boundary

Differentiable primitives such as `pairwise_distances`,
`geodesic_interpolation`, and `tangent_space_map` must compile under `jax.jit`
and retain finite documented derivatives away from cut loci. Statistical
algorithms may use Python orchestration around compiled JAX kernels. Exact
transport pivots, cluster assignments, graph topology, eigenspace selection,
and permutation tests carry no end-to-end gradient guarantee.

Randomized methods receive an explicit JAX key. Public parameter names follow
`n_clusters`, `n_neighbors`, `n_components`, `sample_weight`, `maxiter`, and
`tol`. Iterative immutable results expose `objective`, `iterations`,
`converged`, `reason`, and family-specific diagnostics.

Fitted predictors own the geometry and validated training representation they
need for prediction. A classifier must preserve the user's class labels while
keeping encoded labels inside numerical kernels. A manifold-response model
returns canonical manifold points, not flattened coordinates. Barycentric
codes use explicit simplex constraints, and any routine described as sparse
must have a penalty whose value is not constant on its feasible set.

## Numerical and testing contract

Dense distance algorithms must state their $O(n^2)$ memory use; graph methods
with cubic work must say so. Implementations should use `squared_dist` in
smooth squared-distance objectives and preserve each geometry's cut-locus
branch policy.

A new method is incomplete until tests cover:

- analytic Euclidean behavior and at least one curved geometry;
- nested Product points when its operations support Product;
- deterministic fixed-key results;
- malformed data and missing-capability failures;
- float32 and float64 execution; and
- independent reference outputs or invariant checks.

Inference tests additionally need fixed-seed null checks and explicit
assumptions: negative type for distribution-characterizing energy distances,
PSD kernels for MMD, and exchangeability for sign or label permutations.
Scalable estimates must be compared with their full-batch counterparts on an
analytic Euclidean fixture and identify order dependence on curved spaces.

Runtime dependency tests must demonstrate that importing `geojax.learning`
does not import SciPy, scikit-learn, POT, or PHATE. Executable tutorials remain
self-contained Markdown and may not retain generated notebooks.

The implementation literature is part of the protocol: algorithm docstrings
and scientific tutorials must identify the objective actually computed. For
example, entropically regularized transport must not be presented as the exact
linear program {cite:p}`cuturi2013sinkhorn`.

## References

```{bibliography}
:filter: docname in docnames
```
