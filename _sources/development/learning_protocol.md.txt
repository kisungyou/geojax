# Learning protocol

The `geojax.learning` namespace contains small geometry-aware functions, not a
neural-network framework. New functions must remain pure JAX and may not make
Flax, Equinox, Haiku, Optax, or scikit-learn runtime dependencies. This narrow
boundary keeps the package focused on reusable geometric ingredients rather
than prescribing one geometric-deep-learning architecture
{cite:p}`bronstein2021geometric`.

## Functional boundary

A learning primitive receives geometry objects as static configuration and
arrays or pytrees as dynamic values. Trainable parameters belong to the
user-supplied callable:

```python
mapped = tangent_map(
    source,
    target,
    x,
    source_base=source_base,
    target_base=target_base,
    transform=lambda tangent: network(parameters, tangent),
)
```

This boundary lets the same operation compose with plain JAX or any compatible
neural library. GeoJAX should not introduce a second parameter container,
module abstraction, random-state convention, or graph representation.

## Numerical contract

Learning losses should use `squared_dist` rather than `dist(...) ** 2`.
Every new operation must:

- preserve each geometry's event shape and arbitrary leading batch axes;
- support `Product` pytrees when the operation is geometrically meaningful;
- compile under `jax.jit`;
- have finite first derivatives at zero tangents and coincident points;
- require exact capabilities when its public name promises a distance or
  geodesic, and reject proxy or numerical-local operations;
- preserve or explicitly document each geometry's cut-locus branch policy; and
- avoid Python data-dependent control flow over traced values.

Pairwise operations interpret the axis immediately before the manifold event
as the collection axis. Leading batch axes use NumPy broadcasting.
Interpolation parameters add axes before all endpoint batch and event axes, so
tests must include both batched endpoints and vector-valued interpolation
times.

## Release contract

A manifold-learning feature is incomplete until it participates in a compiled
end-to-end test. The deterministic autoencoder test compiles one training step
for spherical and hyperbolic latent spaces, verifies decreasing loss and finite
encoder gradients, and checks manifold membership of every resulting latent.

Tutorial-only dependencies belong in the `docs` or `examples` optional extras.
Executable tutorials must be self-contained Markdown documents and may not
import hidden training scripts or retain generated notebooks in the repository.

## References

```{bibliography}
:filter: docname in docnames
```
