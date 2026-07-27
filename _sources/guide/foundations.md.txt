# Mathematical foundations

GeoJAX represents a Riemannian geometry by an object that describes points,
tangent vectors, a metric, and the maps needed to move between them. The public
classes follow a capability-qualified contract. `ManifoldProtocol` contains the
retraction-based operations required by optimizers. `GeometryProtocol` is a
structural interface adding the common `exp`, `log`, and `dist` names, but
satisfying it does not by itself assert that those operations are globally
exact. `operation_kind` carries that mathematical status at runtime.
`GeometryMixin` supplies batching and common helpers. Standard geometric
definitions follow {cite:t}`docarmo1992riemannian`; the computational interface
is informed by
{cite:t}`absil2008optimization` and {cite:t}`boumal2023introduction`.

Let $\mathcal M$ be a smooth manifold. At $x\in\mathcal M$, its tangent space is
$T_x\mathcal M$ and its Riemannian metric is

$$
g_x(u,v)=\langle u,v\rangle_x,
\qquad u,v\in T_x\mathcal M.
$$

The same manifold can carry more than one metric. For example, GeoJAX provides
log-Euclidean and affine-invariant metrics on positive-definite matrices. A
geometry object therefore specifies both the set of valid points and the
metric-dependent operations on that set.

## Representation and dimension

Three attributes describe the numerical representation.

| Member | Mathematical meaning |
|---|---|
| `size` | Constructor-level description of one point, such as `3` or `(5, 2)` |
| `shape` | Array shape of one unbatched point; for `Product`, a matching pytree of shapes |
| `dim` | Intrinsic dimension $\dim(\mathcal M)=\dim(T_x\mathcal M)$ |

The representation may use more numbers than the intrinsic dimension. A point
on $S^{n-1}$, for example, is stored as $n$ coordinates constrained by
$x^\top x=1$, so `shape == (n,)` while `dim == n - 1`.

`Product` is the exception to the single-array representation: its factors,
points, and tangent vectors can be matching tuples, lists, dictionaries, or
nested JAX pytrees.

## Points and tangent vectors

The validation and projection members separate ambient arrays from geometric
objects.

### `belongs(x)` and `project(z)`

`belongs(x)` tests the defining constraints of the manifold. Conceptually,

$$
\operatorname{belongs}(x) = [x\in\mathcal M].
$$

`project(z)` maps ambient data to a valid point,

$$
\Pi_{\mathcal M}:\mathcal E\longrightarrow\mathcal M,
$$

where $\mathcal E$ is the numerical embedding space. It is intended for
initialization and numerical repair. Unless the geometry explicitly says so,
`project` should not be interpreted as the nearest-point projection for a
particular metric.

### `is_tangent(x, u)` and `tangent_project(x, a)`

`is_tangent(x, u)` checks whether $u\in T_x\mathcal M$.
`tangent_project(x, a)` maps an ambient vector to a tangent vector,

$$
\Pi_x:\mathcal E\longrightarrow T_x\mathcal M.
$$

For an embedded manifold defined as a level set
$\mathcal M=\{x:F(x)=0\}$,

$$
T_x\mathcal M=\ker DF_x.
$$

For quotient geometries, the returned tangent representation may instead be a
horizontal vector in a chosen representative space. This is how `Grassmann`
uses $n\times k$ matrices while representing subspaces rather than frames.

## Metric operations

`inner(x, u, v)` evaluates $g_x(u,v)$. `norm(x, u)` is induced by that metric:

$$
\lVert u\rVert_x=\sqrt{g_x(u,u)}.
$$

`lincomb(x, a, u, b, v, ...)` forms a tangent-space linear combination,

$$
a u+b v+\cdots\in T_x\mathcal M.
$$

The generic implementation projects the result back to $T_x\mathcal M$ to
remove numerical drift. Product geometries perform the same operation leaf by
leaf.

## Exponential, logarithm, and distance

For $u\in T_x\mathcal M$, let $\gamma_u$ be the geodesic satisfying

$$
\gamma_u(0)=x,
\qquad
\dot\gamma_u(0)=u.
$$

The exponential map follows that geodesic for one unit of time:

$$
\operatorname{Exp}_x(u)=\gamma_u(1).
$$

`exp(x, u)` evaluates this map. The logarithm is a local inverse:

$$
\operatorname{Exp}_x(\operatorname{Log}_x(y))=y.
$$

When `operation_kind("log") == "exact"`, `log(x, y)` returns one selected
shortest initial velocity where that choice is well defined. The corresponding
geodesic distance is

$$
d(x,y)=\lVert\operatorname{Log}_x(y)\rVert_x.
$$

`squared_dist(x, y)` evaluates $d(x,y)^2$ without differentiating the final
square root. It is the preferred primitive for smooth losses and Fréchet
objectives because

$$
\nabla_y d(x,y)^2\big|_{y=x}=0
$$

is well defined even though the derivative of $d(x,y)$ itself is not defined
at coincidence. Individual geometries use direct coordinate, angle, or
spectral formulas where those are more stable than squaring `dist`.

Globally, a logarithm can be nonunique or undefined at a cut locus. Antipodal
sphere points and Grassmann subspaces with a principal angle $\pi/2$ are
important examples. A closed formula is also not available for every metric:
the two Stiefel classes expose convergence information for their iterative
endpoint-shooting logarithms. Class-specific behavior is documented in the
[Geometry guide](geometry.md).

### Exact operations and proxies

Many useful matrix manifolds have efficient retractions but no practical
closed geodesic formulas. GeoJAX follows the Manopt convention: optimization
depends on `retr`, `invretr`, and `transport`, while geodesic operations are
optional mathematically {cite:p}`boumal2014manopt`. For a coherent compositional API, retraction-only
classes still expose `exp`, `log`, and `dist` as compatibility aliases, with
machine-readable capability metadata:

| Query | Result |
|---|---|
| `operation_kind("exp")` | `"exact"` or `"proxy"` |
| `operation_kind("log")` | `"exact"`, `"numerical-local"`, or `"proxy"` |
| `operation_kind("dist")` | `"exact"`, `"numerical-local"`, or `"proxy"` |
| `operation_kind("transport")` | `"parallel"`, `"isometric"`, or `"vector"` |
| `operation_kind("ehess_to_rhess")` | `"exact"` or `"projection"` |
| `operation_kind("rgrad_jvp")` | `"exact"` or `"projection"` |

For a retraction-only class,

$$
\mathtt{exp}(x,u)=R_x(u),
\qquad
\mathtt{log}(x,y)=R_x^{-1}(y),
\qquad
\mathtt{dist}(x,y)=\lVert R_x^{-1}(y)\rVert_x.
$$

The last expression is local and need not be symmetric, so it must not be
reported as a geodesic distance. This distinction lets generic optimizers and
product manifolds compose uniformly without overstating the available
geometry.

`"numerical-local"` is different from `"proxy"`. It means that an exact
exponential is available and numerical endpoint shooting seeks a local inverse,
but convergence does not certify that the returned geodesic is globally
shortest. The Stiefel and generalized Stiefel logarithms use this status.

## Retractions and means

`retr(x, u, t)` maps a tangent step back to the manifold. A retraction
$R_x:T_x\mathcal M\to\mathcal M$ satisfies

$$
R_x(0)=x,
\qquad
DR_x(0)=\operatorname{id}_{T_x\mathcal M}.
$$

It agrees with the exponential to first order and may be cheaper to evaluate.
The default GeoJAX implementation uses
$R_x(tu)=\operatorname{Exp}_x(tu)$; individual geometries can override it.
Retractions and compatible vector transports are the central abstraction for
manifold optimization {cite:p}`absil2008optimization,boumal2023introduction`.

`invretr(x, y)` is a selected local inverse of the retraction. It satisfies
$R_x(R_x^{-1}(y))\approx y$ near $x$ and provides the displacement needed by
some derivative-free or quasi-Newton constructions. It is not automatically a
Riemannian logarithm.

`pair_mean(x, y)` applies the available named maps:

$$
m(x,y)=\operatorname{Exp}_x\!\left(\tfrac12
\operatorname{Log}_x(y)\right).
$$

It is the selected geodesic midpoint only when `exp` and `log` are exact and the
minimizing logarithm is unique. With proxy or numerical-local operations it is
a midpoint-like local construction instead. A sample Fréchet mean minimizes
$\sum_i w_i d(x,x_i)^2$ and is generally an optimization problem
{cite:p}`frechet1948elements,karcher1977center`.

## Transport

Tangent vectors at different points belong to different vector spaces.
`transport(x, y, u)` maps

$$
\mathcal T_{x\to y}:T_x\mathcal M\longrightarrow T_y\mathcal M.
$$

For exact parallel transport along a geodesic, the Levi-Civita connection
preserves the metric:

$$
g_y(\mathcal T_{x\to y}u,\mathcal T_{x\to y}v)=g_x(u,v).
$$

Transport is used by conjugate-gradient and quasi-Newton methods to compare
directions constructed at successive iterates.

The protocol intentionally says *transport*, not *parallel transport*. Exact
Levi-Civita transport is available when a stable closed formula is known. An
isometric vector transport may be used otherwise, provided the class documents
that distinction. `SPDBuresWasserstein` follows the latter convention because
general Bures-Wasserstein parallel transport is obtained from a differential
equation rather than a closed endpoint formula.

## Matrix Lie groups

A matrix Lie group has algebraic maps in addition to its Riemannian geometry.
For a Lie-algebra element $A$, the group exponential is the matrix exponential

$$
\exp_{\mathrm{grp}}(A)=e^A.
$$

This need not equal the Riemannian exponential. They coincide on
`SpecialOrthogonal` because its metric is bi-invariant. On
`SpecialEuclidean`, GeoJAX uses the canonical product metric on rotations and
translations; its Riemannian geodesic has a straight translation component,
whereas the group exponential couples angular and translational velocity.
Accordingly, the class exposes both `exp`/`log` and
`group_exp`/`group_log`. The distinction between Lie-group and Riemannian
exponentials is reviewed by {cite:t}`hall2015lie`.

## Autodiff and Riemannian derivatives

Suppose a cost $f:\mathcal M\to\mathbb R$ is differentiated through its
ambient array representation. Its Riemannian gradient is defined by

$$
Df(x)[u]=g_x(\operatorname{grad}f(x),u)
\quad\text{for every }u\in T_x\mathcal M.
$$

`egrad_to_rgrad(x, egrad)` converts the ambient Euclidean gradient into this
metric-dual tangent vector. On an isometrically embedded manifold this is often
an orthogonal tangent projection; for a non-Euclidean metric, additional
metric factors are required.

`ehess_to_rhess(x, egrad, ehess_vec, u)` converts an ambient Hessian-vector
product into the Riemannian Hessian action

$$
\operatorname{Hess}f(x)[u]
=\nabla_u\operatorname{grad}f,
$$

including connection or embedding-curvature terms when a geometry supplies
them. The mixin fallback only tangent-projects the ambient Hessian-vector
product and has `operation_kind("ehess_to_rhess") == "projection"`. GeoJAX
second-order solvers reject that fallback: use a geometry advertising `"exact"`
or supply `rhess_vec`. The same rule applies to the directional derivative of a
user-supplied Riemannian gradient through `operation_kind("rgrad_jvp")`. See the
[optimization guide](optimization.md#second-order-models) for the support table.

## Random and batched operations

`random_point(key, sample_shape=())` returns points with shape
`sample_shape + M.shape`. `random_tangent(key, x, scale=...)` samples an ambient
random vector and maps it to $T_x\mathcal M$ according to the geometry. These
routines are reproducible because randomness is controlled by explicit JAX
keys.

Core pointwise protocol operations accept points shaped
`batch_shape + M.shape`. Scalar-valued pointwise operations return
`batch_shape`, while point and tangent operations preserve the event
dimensions. NumPy-style broadcasting applies to compatible leading shapes.
Reducers such as sample means document their reduction axes separately.
`Product` applies the pointwise contract leafwise.

Event axes are part of the manifold definition, not broadcast dimensions.
Consequently, `belongs` and `is_tangent` return `False` for malformed event
shapes, while constructors such as `project` and `tangent_project` raise
`ValueError`. For correctly shaped ambient data, including zero or
rank-deficient matrices, `project` guarantees a finite point accepted by
`belongs`.

`GeometryMixin` retains convenience names for fixed-base collections:

$$
\begin{aligned}
\mathtt{exp\_batch}(x,[u_i]) &= [\operatorname{Exp}_x(u_i)],\\
\mathtt{log\_batch}(x,[y_i]) &= [\operatorname{Log}_x(y_i)],\\
\mathtt{dist\_batch}(x,[y_i]) &= [d(x,y_i)].
\end{aligned}
$$

These methods delegate to the natively batched operations. Users may also
compose the methods with `jax.vmap` when a transformation makes the mapped axis
explicit.

### Differentiability at numerical singularities

Closed geometric formulas often contain removable expressions such as
$\sin(r)/r$, $\sinh(r)/r$, or spectral divided differences with repeated
eigenvalues. GeoJAX evaluates their analytic limits and supplies custom JAX
derivatives where naïve autodiff would otherwise differentiate an undefined
intermediate eigenbasis. Tests cover zero tangents, coincident points, and
repeated SPD spectra under both float32 and float64.

This policy does not conceal genuine singularities. Each geometry documents its
cut-locus convention: sphere, Grassmann, and rotation logarithms return
nonfinite values where no unique branch is selected, while `Torus` deliberately
uses its half-open angular representation to choose one of the two directions
at a component difference of $\pi$. Quotient constructions can likewise select
a representative through a documented alignment rule.

## Equivariant embeddings

An embedding $j:\mathcal M\to\mathbb R^N$ is equivariant when a group action on
$\mathcal M$ is carried to a compatible action in the embedding space. Its
differential maps tangent vectors by

$$
dj_x:T_x\mathcal M\longrightarrow T_{j(x)}j(\mathcal M).
$$

The ambient Euclidean metric can be pulled back as

$$
g_x(u,v)=\langle dj_x(u),dj_x(v)\rangle_F.
$$

An embedding also supplies extrinsic operations such as chordal distance and
projection of an ambient mean back to the manifold. `SphereExtrinsic` uses the
identity embedding, while `GrassmannProjection` uses $j([X])=XX^\top$.

## Product manifolds

For $\mathcal M=\mathcal M_1\times\cdots\times\mathcal M_r$, tangent vectors
split as $u=(u_1,\ldots,u_r)$ and GeoJAX uses the direct-sum metric

$$
g_x(u,v)=\sum_{i=1}^r g_{x_i}^{(i)}(u_i,v_i).
$$

When every factor distance is exact, consequently,

$$
d(x,y)^2=\sum_{i=1}^r d_i(x_i,y_i)^2,
$$

and projection, exponential, logarithm, transport, gradient conversion, and
sampling act independently on the leaves of the matching pytree. Product
capability metadata is exact only when every factor advertises exactness.
Construction rejects leaves that do not satisfy `GeometryProtocol`; absent
third-party capability metadata is never interpreted as a mathematical
guarantee.

## References

```{bibliography}
:filter: docname in docnames
```
