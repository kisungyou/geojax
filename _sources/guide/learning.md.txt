# Manifold-valued learning

GeoJAX supplies geometry-aware operations for neural systems without choosing
a neural-network framework. Parameters owned by Flax, Equinox, Haiku, or plain
JAX functions remain external; geometry objects govern manifold-valued
activations and comparisons. This separates geometric primitives from model
architecture in the broader geometric-deep-learning spirit
{cite:p}`bronstein2021geometric`.

## Batch contract

For a manifold with event shape `M.shape`, points use
`batch_shape + M.shape`. Core pointwise geometry operations preserve leading
batch axes, and scalar pointwise operations such as `squared_dist` return
`batch_shape`. Reducers document their reduced axes separately. Consequently
geometry calls may appear directly inside `jax.jit`, `jax.grad`, and `jax.vmap`.

Use `squared_dist` in smooth losses:

$$
\mathcal L(z,c)=d_{\mathcal M}(z,c)^2.
$$

Unlike `dist`, it does not differentiate a square root at coincident points.
The implementation also fills removable inverse-trigonometric and spectral
singularities. Cut-locus behavior is geometry-specific; for example, a
spherical logarithm is not defined at an antipode, while the torus selects the
half-open angular branch at a component difference of $\pi$.

## Pairwise distances

For collections with shapes `batch_x + (n,) + M.shape` and
`batch_y + (m,) + M.shape`,
{func}`geojax.learning.pairwise_squared_dist` returns
`broadcast(batch_x, batch_y) + (n, m)`. It also accepts `Product` points with
the same pytree as their factors. The helper requires
`operation_kind("dist") == "exact"` and rejects numerical-local or
inverse-retraction proxy distances.

```python
from geojax.geometry import Sphere
from geojax.learning import pairwise_squared_dist

M = Sphere(size=3)
distances = pairwise_squared_dist(M, queries, prototypes)
logits = -distances
```

## Geodesic interpolation

{func}`geojax.learning.geodesic_interpolate` evaluates

$$
\gamma(t)=\operatorname{Exp}_x\!\left(t\operatorname{Log}_x(y)\right).
$$

For endpoint batch shape `batch_shape`, an array of times with shape
`time_shape` produces `time_shape + batch_shape + M.shape`. The helper requires
exact exponential and logarithm capabilities. The selected logarithm must also
be defined, so avoid endpoint pairs at a cut locus.

## Tangent-space maps

{func}`geojax.learning.tangent_map` composes

$$
x
\longmapsto \operatorname{Log}_{b_s}(x)
\longmapsto A_\theta\!\left(\operatorname{Log}_{b_s}(x)\right)
\longmapsto \operatorname{Exp}_{b_t}\!\left(
  \Pi_{b_t}A_\theta(\operatorname{Log}_{b_s}(x))
\right).
$$

The callable $A_\theta$ is supplied by the user and can close over any
framework's parameters. GeoJAX deliberately uses ambient tangent arrays
instead of promising global tangent coordinates, which do not exist smoothly
for every manifold. Exponential/logarithmic tangent-space layers are common in
hyperbolic representation learning
{cite:p}`ganea2018hyperbolic,chami2019hyperbolic`.
`tangent_map` requires an exact source logarithm and exact target exponential;
retraction-only and numerical-local source maps must be composed explicitly by
the user.

## Worked patterns

The geometric-deep-learning tutorials deliberately use plain JAX parameter
pytrees so the geometric part remains visible:

- [deterministic autoencoders](../tutorials/manifold_autoencoder.md) place a
  learned bottleneck in Euclidean, spherical, or hyperbolic space;
- [curvature-aware graph learning](../tutorials/graph_curvature.md) performs
  tangent-space message aggregation and geodesic prototype classification;
  and
- [SPD prototype networks](../tutorials/spd_eeg_classifier.md) transform real
  EEG covariances into learned SPD activations and compare three matrix
  geometries.

Together they cover manifold-valued activations, intrinsic aggregation, and
geometry-aware output heads without coupling GeoJAX to one neural framework.

## Current scope

Version 0.2 establishes differentiable geometry for manifold-valued features.
It does not provide a neural-network framework, graph container, or nominal
coordinate-wise "Riemannian Adam." Ordinary network parameters can use Optax.
For standalone finite-sum manifold objectives, `FiniteSum` and
`StochasticGradient` provide mini-batch updates with transported momentum.
GeoJAX does not yet provide an Optax-compatible functional transformation for
manifold-constrained parameters inside an end-to-end neural training state;
that remains a separate, future addition.

## References

```{bibliography}
:filter: docname in docnames
```
