# Mathematical foundations

GeoJAX represents a Riemannian geometry by an object that describes points,
tangent vectors, a metric, and the maps needed to move between them. The public
classes follow a two-level contract. `ManifoldProtocol` contains the
retraction-based operations required by optimizers. `GeometryProtocol` extends
it with genuine geodesic exponential, logarithm, and distance operations.
`GeometryMixin` supplies batching and common helpers.

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

`log(x, y)` returns one selected shortest initial velocity when that choice is
well defined. The corresponding geodesic distance is

$$
d(x,y)=\lVert\operatorname{Log}_x(y)\rVert_x.
$$

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
optional mathematically. For a coherent compositional API, retraction-only
classes still expose `exp`, `log`, and `dist` as compatibility aliases, with
machine-readable capability metadata:

| Query | Result |
|---|---|
| `operation_kind("exp")` | `"exact"` or `"proxy"` |
| `operation_kind("log")` | `"exact"` or `"proxy"` |
| `operation_kind("dist")` | `"exact"` or `"proxy"` |
| `operation_kind("transport")` | `"parallel"`, `"isometric"`, or `"vector"` |

For a retraction-only class,

$$
\texttt{exp}(x,u)=R_x(u),
\qquad
\texttt{log}(x,y)=R_x^{-1}(y),
\qquad
\texttt{dist}(x,y)=\lVert R_x^{-1}(y)\rVert_x.
$$

The last expression is local and need not be symmetric, so it must not be
reported as a geodesic distance. This distinction lets generic optimizers and
product manifolds compose uniformly without overstating the available
geometry.

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

`invretr(x, y)` is a selected local inverse of the retraction. It satisfies
$R_x(R_x^{-1}(y))\approx y$ near $x$ and provides the displacement needed by
some derivative-free or quasi-Newton constructions. It is not automatically a
Riemannian logarithm.

`pair_mean(x, y)` returns the midpoint of the selected geodesic:

$$
m(x,y)=\operatorname{Exp}_x\!\left(\tfrac12
\operatorname{Log}_x(y)\right).
$$

This is a two-point construction. A sample Fréchet mean instead minimizes
$\sum_i w_i d(x,x_i)^2$ and is generally an optimization problem.

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
`group_exp`/`group_log`.

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
product, so second-order solvers on curved spaces should use a geometry-specific
conversion or a user-supplied `rhess_vec`.

## Random and batched operations

`random_point(key, sample_shape=())` returns points with shape
`sample_shape + M.shape`. `random_tangent(key, x, scale=...)` samples an ambient
random vector and maps it to $T_x\mathcal M$ according to the geometry. These
routines are reproducible because randomness is controlled by explicit JAX
keys.

`GeometryMixin` also supplies leading-axis vectorization at a fixed base point:

$$
\begin{aligned}
\texttt{exp_batch}(x,[u_i]) &= [\operatorname{Exp}_x(u_i)],\\
\texttt{log_batch}(x,[y_i]) &= [\operatorname{Log}_x(y_i)],\\
\texttt{dist_batch}(x,[y_i]) &= [d(x,y_i)].
\end{aligned}
$$

They are implemented with `jax.vmap`; geometry methods are therefore written
without Python loops over sample axes.

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
