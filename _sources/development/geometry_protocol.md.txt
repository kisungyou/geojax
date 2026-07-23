# Geometry protocol

A new geometry should first implement `ManifoldProtocol`: membership and point
projection, tangent operations, the metric, a retraction and inverse
retraction, transport, and random generation. It should also expose `size`,
`shape`, and intrinsic `dim`.

Use `GeometryMixin` when genuine exponential, logarithm, and distance
operations are available. Use `RetractionGeometryMixin` otherwise. The latter
provides uniform compatibility names while advertising them as proxies through
`operation_kind`; do not mark an approximate or first-order construction as
exact.

At minimum, test:

```text
belongs(project(x))
is_tangent(x, tangent_project(x, u))
retr(x, 0) == x
invretr(x, retr(x, small_u)) ~= small_u
exp(x, 0) == x                       # when exp is exact
log(x, exp(x, small_u)) ~= small_u   # when log is exact
dist(x, y) == dist(y, x)             # when dist is exact
norm(y, transport(x, y, u)) ~= norm(x, u)  # when transport is isometric
```

Random tangent generation must document its normalization and scaling
conventions under the Riemannian metric.

## JAX transformation contract

A public geometry treats its instance as static configuration and its point,
tangent, and PRNG-key arguments as dynamic JAX values. With fixed geometry
dimensions and pytree structure, the following methods must compile under
`jax.jit`:

```text
belongs, project, is_tangent, tangent_project,
inner, norm, exp, log, dist, transport,
random_point, random_tangent
```

`sample_shape` and other shape-defining arguments are static Python values.
`exp_batch`, `log_batch`, and `dist_batch` must compose `jax.vmap` with
`jax.jit`, including for Product pytrees. Scalar costs built from geometry
operations must remain differentiable with `jax.value_and_grad`; ambient and
Riemannian Hessian-vector products must compose with `jax.jvp`.

The shared transformation test suite compiles the complete numerical protocol
for every public geometry. It additionally tests JIT-plus-vmap composition
across vector, quotient, matrix, Lie-group, numerical-logarithm, retraction,
shape, and Product families. A new geometry belongs in those tests before it
is added to the public namespace.

Every class must set the exactness and transport metadata correctly. If
`transport` is not Levi-Civita parallel transport, its class and guide must name
the construction precisely and state which properties it preserves.
Differentiable spectral routines must also be tested at repeated eigenvalues;
valid matrix functions can otherwise acquire nonfinite autodiff results from
an eigenvector-based implementation.
