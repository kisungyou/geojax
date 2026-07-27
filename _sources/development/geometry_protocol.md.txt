# Geometry protocol

A new geometry should first implement `ManifoldProtocol`: membership and point
projection, tangent operations, the metric, a retraction and inverse
retraction, transport, and random generation. It should also expose `size`,
`shape`, and intrinsic `dim`. The contract follows the computational
abstractions used in standard manifold-optimization references
{cite:p}`absil2008optimization,boumal2023introduction`.

Use `GeometryMixin` when exact or numerical-local geodesic operations are
available. Use `RetractionGeometryMixin` otherwise. The latter provides uniform
compatibility names while advertising them as proxies through `operation_kind`;
do not mark an approximate, endpoint-only, or first-order construction as
globally exact. `GeometryProtocol` is structural and does not replace these
runtime capability checks.

At minimum, test:

```text
belongs(project(x))
is_tangent(x, tangent_project(x, u))
retr(x, 0) == x
invretr(x, retr(x, small_u)) ~= small_u
exp(x, 0) == x                       # when exp is exact
log(x, exp(x, small_u)) ~= small_u   # exact or converged numerical-local log
grad_y squared_dist(x, y) at y=x is finite and zero
dist(x, y) == dist(y, x)             # when dist is exact
norm(y, transport(x, y, u)) ~= norm(x, u)  # when transport is isometric
```

The projection invariant must use genuinely ambient inputs, not only valid
random points. Include zeros, large nonsymmetric or indefinite arrays,
rank-deficient arrays, and both float32 and float64. For array geometries,
`belongs` and `is_tangent` must return `False` on malformed event shapes;
operations that construct values must raise a clear `ValueError` before
performing accidental computations in another dimension.

Random tangent generation must document its normalization and scaling
conventions under the Riemannian metric.

## JAX transformation contract

A public geometry treats its instance as static configuration and its point,
tangent, and PRNG-key arguments as dynamic JAX values. With fixed geometry
dimensions and pytree structure, the following methods must compile under
`jax.jit`:

```text
belongs, project, is_tangent, tangent_project,
inner, norm, exp, log, squared_dist, dist, transport,
random_point, random_tangent
```

`sample_shape` and other shape-defining arguments are static Python values.
Every operation must natively accept `batch_shape + M.shape`, including nested
batches and broadcasting an unbatched base point over batched tangents or
endpoints. `exp_batch`, `log_batch`, and `dist_batch` delegate to that behavior
and must compose with `jax.jit`, including for Product pytrees. Scalar costs
built from geometry operations must remain differentiable with
`jax.value_and_grad`. A geometry advertising an exact Hessian conversion must
test its ambient and Riemannian Hessian-vector products under `jax.jvp`.

The shared transformation test suite compiles the complete numerical protocol
for every public geometry. It additionally tests JIT-plus-vmap composition
across vector, quotient, matrix, Lie-group, numerical-logarithm, retraction,
shape, and Product families. A new geometry belongs in those tests before it
is added to the public namespace.

Every class must set exactness, Hessian-conversion, and transport metadata
correctly. The defaults are deliberately uncertified: use
`ExactGeometryMixin` only after testing the corresponding mathematical
guarantees. If
`transport` is not Levi-Civita parallel transport, its class and guide must name
the construction precisely and state which properties it preserves.
Differentiable routines must be tested at zero tangent vectors and coincident
points. Spectral routines additionally require repeated-eigenvalue tests;
valid matrix functions can otherwise acquire nonfinite autodiff results from
an intermediate eigenbasis. Fill removable analytic limits, but document each
cut-locus branch policy explicitly; nonfinite failure and deterministic branch
selection are both valid only when stated and tested.

## References

```{bibliography}
:filter: docname in docnames
```
