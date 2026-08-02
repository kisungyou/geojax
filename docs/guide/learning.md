# Manifold-valued learning

GeoJAX separates the geometry of observations from the statistical method
applied to them. A geometry defines point validity, distance, logarithmic and
exponential maps, and optional equivariant embeddings. The learning layer
validates a collection once and then consumes those operations without making
assumptions about whether a point is a vector, matrix, or Product pytree. This
follows the metric-statistics view of random objects while retaining JAX
transformations for differentiable primitives
{cite:p}`frechet1948elements,dubey2019frechet,bronstein2021geometric`.

## Canonical datasets

For an array geometry with event shape `M.shape`, the canonical dataset layout
is

$$
\mathtt{batch\_shape} + (n,) + \mathtt{M.shape}.
$$

The axis immediately before the event dimensions is the sample axis. Product
data use the same nested tuple, list, or dictionary as `M.factors`; every leaf
has the same batch shape and sample count but its own event shape.

```python
import jax

from geojax.geometry import Product, SPDLogEuclidean, Sphere
from geojax.learning import as_manifold_data

M = Product({
    "direction": Sphere(3),
    "covariance": SPDLogEuclidean((2, 2)),
})
values = M.random_point(jax.random.key(0), sample_shape=(32,))
data = as_manifold_data(M, values)
```

`as_manifold_data` defaults to membership validation. It rejects malformed,
nonfinite, or off-manifold observations. Passing `repair=True` explicitly calls
`M.project`; projection is never silent. `sample_axis` must be provided when an
input does not already use the canonical layout, so an adapter never guesses
which axis represents observations.

## Alternate representations

The adapter accepts named representations only when their conversion has a
defined geometric meaning. Examples include hyperspherical angles, Poincaré
coordinates, Grassmann projectors, SPD Cholesky factors, covariance matrices
for correlation geometries, SO(3) quaternions, SE(2)/SE(3) twists, raw Kendall
landmarks, and low-rank factors. Product representations may themselves be
pytrees.

```python
from geojax.geometry import Grassmann
from geojax.learning import as_manifold_data

M = Grassmann((5, 2))
frames = as_manifold_data(M, projectors, representation="projector")
```

A spanning basis represents the same Grassmann point and may be
orthonormalized directly. A nonorthonormal Stiefel frame represents a different
ambient matrix, so it is accepted only through `repair=True`. User-defined
representations can be added with `register_manifold_data_adapter`.

## Exact capabilities

Distance-only methods require `operation_kind("dist") == "exact"`. Methods
that update intrinsic centers additionally require exact logarithmic and
exponential maps. GeoJAX raises `LearningCapabilityError` when a geometry only
offers a retraction proxy or numerical-local logarithm. This prevents an
algorithm named after a geodesic quantity from silently changing its
mathematical objective.

| Required geometry | Learning methods |
|---|---|
| Exact distance | pairwise distances, neighbors, nearest-centroid and k-NN prediction, medoids, hierarchy, spectral and graph learning, kernel regression, MDS, Isomap, kernel PCA, Sammon, t-SNE, PHATE, transport, energy and kernel tests |
| Exact distance, logarithm, exponential | interpolation, Fréchet and robust summaries, geodesic and local Fréchet regression, k-means and mini-batch summaries, mean shift, CLRQ, PGA, barycentric coding, dictionary learning, enclosing balls, Fréchet ANOVA, paired tests |
| Equivariant embedding | Riemannian manifold metric learning |

## Differentiable primitives

{func}`geojax.learning.pairwise_distances` returns ordinary or squared exact
distances. Collections broadcast over leading batch axes, Product values remain
pytrees, and `block_size` limits temporary right-hand blocks while retaining a
dense result.

```python
from geojax.learning import pairwise_distances

squared = pairwise_distances(M, queries, prototypes, squared=True)
logits = -squared
```

Geodesic interpolation evaluates

$$

\gamma(t)=\operatorname{Exp}_x\!\left(t\operatorname{Log}_x(y)\right),

$$

and `tangent_space_map` composes a user transformation between source and
target tangent spaces. These wrappers are compatible with `jax.jit`,
`jax.vmap`, and differentiation whenever the selected geometry operations are.

## Statistical algorithms

Fréchet means minimize weighted squared distance and use GeoJAX's manifold
optimizer. Medians use a guarded Riemannian Weiszfeld iteration. Clustering
provides intrinsic Lloyd updates, medoids, valid metric-space linkage rules,
spectral graph methods, mean shift, and competitive quantization
{cite:p}`karcher1977center,lloyd1982least,zelnik2005self`.

Dimension-reduction methods consume exact pairwise distances. PGA differs: it
forms its covariance Gram matrix with `M.inner` at the Fréchet mean, so the
result remains valid for Product manifolds and metrics that are not ambient
Frobenius metrics {cite:p}`fletcher2004principal`. Classical MDS and kernel
methods report negative eigenvalue mass rather than pretending every manifold
distance is Euclidean.

The exact empirical Wasserstein routine solves the finite weighted transport
problem and reports its transport plan, marginal residuals, reduced costs, and
duality gap. `sinkhorn_divergence` is a separate optional OTT-JAX operation;
regularization is never labeled exact {cite:p}`cuturi2013sinkhorn`.

Inference separates three null hypotheses. Fréchet ANOVA compares object-valued
populations through their means and variances, the Biswas--Ghosh statistic uses
only interpoint distances, and the Wasserstein test compares empirical measures
{cite:p}`dubey2019frechet,biswas2014nonparametric`.

`riemannian_metric_learning` forms similar- and dissimilar-pair scatter
matrices after an equivariant embedding, regularizes both matrices, and uses
their weighted log-Euclidean closed form. The `balance` parameter controls the
relative contributions; the default midpoint matches the core RMML
construction {cite:p}`zhu2018generalized`.

## Supervised prediction

Distance classifiers make the smallest geometric commitment. The nearest-
centroid rule estimates one weighted Fréchet mean $\widehat\mu_c$ per class and
predicts

$$
\widehat c(x)=\arg\min_c d(x,\widehat\mu_c),
$$

while `knn_classifier` votes among the $k$ closest training observations. Both
work with any exact-distance geometry, including nested Product data.

Tangent classifiers first compute a reference Fréchet mean $\widehat\mu$ and
an intrinsic metric Gram matrix

$$
G_{ij}
=\left\langle\operatorname{Log}_{\widehat\mu}(x_i),
\operatorname{Log}_{\widehat\mu}(x_j)\right\rangle_{\widehat\mu}.
$$

Its positive eigenspace gives metric-orthonormal coordinates, rather than an
ambient Frobenius flattening. `tangent_space_logistic_regression` fits a
multinomial softmax model there;
`tangent_space_discriminant_analysis` provides regularized LDA and QDA. These
models are local to the selected logarithm chart and should not be interpreted
across a cut locus.

## Manifold-valued responses

`geodesic_regression` fits the one-predictor curve

$$
\widehat Y(t)
=\operatorname{Exp}_{p}\!\left((t-\overline t)v\right)
$$

by intrinsic least squares, profiling $v\in T_p\mathcal M$ while optimizing
the intercept $p$ {cite:p}`fletcher2013regression`. This is a parametric
geodesic model, so systematic curvature away from one geodesic remains in the
residuals.

`local_polynomial_regression` instead estimates a conditional Fréchet mean.
For local-linear smoothing at $t$, its signed weights are

$$
s_i(t)=K_h(t_i-t)
\frac{S_2-(t_i-t)S_1}{S_0S_2-S_1^2},
\qquad
S_r=\sum_i K_h(t_i-t)(t_i-t)^r,
$$

and the prediction minimizes $\sum_i s_i(t)d^2(p,y_i)$. Degree zero uses
positive Nadaraya--Watson weights. Signed local-linear objectives need not be
globally convex on a general manifold, so GeoJAX initializes them with the
positive local Fréchet mean and reports no global-optimum claim
{cite:p}`petersen2019frechet`.

## Uncertainty and testing

`bootstrap_frechet_mean` resamples the empirical measure, recomputes intrinsic
means, and reports the requested quantile of replicate distances from the
original estimate as a bootstrap geodesic ball. It is an approximate
percentile region, not a curvature-corrected confidence set.

The energy statistic compares between-sample and within-sample distances
{cite:p}`szekely2013energy`. On a completely general metric space, equality
characterization requires an appropriate negative-type condition. The MMD
test similarly requires a positive-semidefinite kernel
{cite:p}`gretton2012kernel`: GeoJAX checks the observed Gram eigenvalues by
default because an RBF of squared geodesic distance is not universally PSD.
The check uses the larger of the requested tolerance and a dtype-aware
eigensolver backward-error bound, so roundoff-scale negative eigenvalues in
float32 are not mistaken for a mathematically indefinite kernel.
`paired_frechet_test` applies sign flips to paired tangent displacements and
therefore assumes exchangeability under those flips.

## Scalable summaries

`streaming_frechet_mean` performs the inductive update

$$
\mu_t
=\operatorname{Exp}_{\mu_{t-1}}
\!\left(\frac{w_t}{\sum_{j\leq t}w_j}
\operatorname{Log}_{\mu_{t-1}}(x_t)\right).
$$

It is exact for a weighted Euclidean mean and order-dependent on a curved
manifold. `minibatch_frechet_mean` and `minibatch_kmeans` use shuffled batches
and decaying log-map steps, following the stochastic Riemannian optimization
view {cite:p}`bonnabel2013stochastic`. They require an explicit key and return
objective and update histories; they are approximations to the corresponding
full-batch estimators.

## Barycentric coding and dictionaries

For atoms $D_1,\ldots,D_m$ and an observation $x$, intrinsic barycentric
coding solves

$$
\min_{w\in\Delta^{m-1}}
\frac12\left\|\sum_{j=1}^m
w_j\operatorname{Log}_x(D_j)\right\|_x^2
+\frac{\lambda}{2}\|w\|_2^2.
$$

`geodesic_barycentric_coding` uses projected gradient steps on the simplex and
reconstructs each observation as the weighted Fréchet mean of its atoms.
`manifold_dictionary_learning` alternates these codes with a Product-manifold
optimization of the fixed-code intrinsic residual. Backtracking accepts only
atom updates that do not increase the reported reconstruction objective
{cite:p}`ho2013dictionary`. These are dense barycentric codes; the simplex
$\ell_1$ norm is constant, so the routine does not claim lasso sparsity. When
sample weights are supplied, atom optimization, backtracking acceptance, and
the returned objective all use the same normalized weighted criterion.

## Robust and graph learning

The trimmed mean repeatedly retains the smallest squared geodesic residuals.
`geodesic_m_estimator` uses Huber, Cauchy, or Tukey residual weights in guarded
iteratively reweighted Fréchet updates. `geodesic_spatial_depth` evaluates

$$
D(x)=1-\left\|
\sum_i w_i
\frac{\operatorname{Log}_x(x_i)}{d(x,x_i)}
\right\|_x,
$$

with the coincident contribution set to zero; distance ranks use midranks
around an intrinsic median {cite:p}`fletcher2009median`. A weighted trimmed
fit reports its objective with weights renormalized over the retained sample.

For partially labeled data, `label_propagation` diffuses class scores over a
geodesic-distance affinity graph while clamping known labels
{cite:p}`zhou2003consistency`. `manifold_regularized_regression` solves the
dense transductive objective

$$
\sum_{i\in L}(f_i-y_i)^2
+\lambda_A\|f\|_2^2
+\lambda_I f^\top L_G f,
$$

where $L_G$ is the graph Laplacian {cite:p}`belkin2006manifold`. The current
result predicts only the supplied graph vertices; it does not claim the RKHS
out-of-sample extension of the full manifold-regularization framework. For a
$k$-nearest-neighbor graph, the diagonal is excluded before sorting, so tied
duplicate observations cannot accidentally select themselves as neighbors.

## Learning roadmap

The adapter-first core now covers the former priority-one roadmap, intrinsic
dictionary learning, robust analysis, and graph semi-supervision. The next
candidate families are deliberately narrower:

| Method family | Candidate public methods | Geometric requirements |
|---|---|---|
| Graph embeddings | diffusion maps and Laplacian eigenmaps | exact distance or a user-supplied affinity |
| Supervised reduction | discriminant tangent components and supervised distance embeddings | exact logarithm and metric or an equivariant embedding |
| Model assessment | intrinsic silhouette, gap statistic, and resampling-based clustering stability | exact distance; randomized procedures require keys |
| Transport statistics | entropic Wasserstein barycenters and transport-based clustering | optional differentiable Sinkhorn backend |

Curves and spatial-analysis routines remain outside this roadmap. Sparse and
approximate-neighbor backends should be designed as separate scalability work
rather than hidden changes to the dense methods documented in this release.

## Computational limits

The learning layer is dense in this release. Pairwise data need
$O(n^2)$ storage, while Floyd-Warshall Isomap and hierarchical updates can need
$O(n^3)$ work. `block_size` lowers temporary pairwise memory but does not make
the returned matrix sparse. High-level assignments, permutations, graph
construction, and exact transport are intentionally not promised to be
end-to-end differentiable.

## References

```{bibliography}
:filter: docname in docnames
```
