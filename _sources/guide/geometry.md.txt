# Geometry

Geometry objects own every manifold-dependent operation. A point has shape
`M.shape`, a tangent vector normally uses the same ambient representation, and
leading axes may be used for batches. The formulas below state the conventions
used by the implementation.

## Available spaces

| Geometry | Point representation | Metric |
|---|---|---|
| `Euclidean(size)` | vector, matrix, or tensor | Frobenius |
| `Oblique(size=(n,m))` | $n\times m$ matrix with unit columns | product round metric |
| `ProbabilitySimplex(size=d)` | positive probability vector | Fisher--Rao |
| `Sphere(size=n)` | unit vector in $\mathbb R^n$ | round metric |
| `SphereExtrinsic(size=n)` | unit vector in $\mathbb R^n$ | round metric plus identity-embedding operations |
| `PoincareBall(size=d)` | vector in the open unit ball | hyperbolic metric |
| `Grassmann(size=(n,k))` | $n\times k$ orthonormal frame | canonical quotient metric |
| `GrassmannProjection(size=(n,k))` | $n\times k$ orthonormal frame | normalized projector-embedding metric |
| `Stiefel(size=(n,k))` | $n\times k$ orthonormal frame | canonical quotient metric |
| `StiefelEuclidean(size=(n,k))` | $n\times k$ orthonormal frame | embedded Frobenius metric |
| `GeneralizedStiefel(size=(n,k), metric=B)` | $B$-orthonormal frame | $B$-weighted embedded metric |
| `GeneralizedGrassmann(size=(n,k), metric=B)` | $B$-orthonormal subspace frame | $B$-weighted quotient metric |
| `SpecialOrthogonal(size=n)` | $n\times n$ rotation matrix | Frobenius bi-invariant |
| `SpecialEuclidean(size=n)` | $(n+1)\times(n+1)$ homogeneous rigid motion | canonical product metric |
| `SPDLogEuclidean(size=(n,n))` | SPD matrix | log-Euclidean |
| `SPDAffineInvariant(size=(n,n))` | SPD matrix | affine-invariant |
| `SPDBuresWasserstein(size=(n,n))` | SPD matrix | Bures-Wasserstein |
| `FixedRank(size=(m,n), rank=k)` | rank-$k$ matrix | embedded Frobenius |
| `RankKPSD(size=(n,n), rank=k)` | rank-$k$ PSD matrix | embedded Frobenius |
| `RankKPSDBuresWasserstein(size=(n,n), rank=k)` | rank-$k$ PSD matrix | Bures--Wasserstein quotient |
| `Elliptope(size=(n,n), rank=k)` | rank-$k$ correlation matrix | embedded Frobenius |
| `Spectrahedron(size=(n,n), rank=k)` | rank-$k$ unit-trace PSD matrix | embedded Frobenius |
| `CorrelationECM(size=(n,n))` | correlation matrix | Euclidean-Cholesky pullback |
| `CorrelationLEC(size=(n,n))` | correlation matrix | log-Euclidean-Cholesky pullback |
| `CorrelationAffineQuotient(size=(n,n))` | correlation matrix | affine-invariant quotient |
| `Hyperboloid(size=n)` | upper-sheet Lorentz vector | hyperbolic metric |
| `Torus(size=d)` | angles in $[-\pi,\pi)$ | flat product metric |
| `KendallShape(size=(m,d))` | centered, normalized $m\times d$ landmarks | spherical shape quotient |
| `Product(factors)` | matching pytree | direct-sum metric |

## Core interface

Every public geometry follows the same protocol. This lets optimizers consume a
geometry without knowing its representation.

| Member | Operation |
|---|---|
| `belongs(x)` | Test $x\in\mathcal M$ |
| `project(z)` | Repair ambient data as a point of $\mathcal M$ |
| `is_tangent(x, u)` | Test $u\in T_x\mathcal M$ |
| `tangent_project(x, a)` | Map an ambient vector to $T_x\mathcal M$ |
| `inner(x, u, v)` | Evaluate $g_x(u,v)$ |
| `norm(x, u)` | Evaluate $\sqrt{g_x(u,u)}$ |
| `exp(x, u)` | Evaluate $\operatorname{Exp}_x(u)$ |
| `log(x, y)` | Evaluate a selected $\operatorname{Log}_x(y)$ |
| `dist(x, y)` | Evaluate geodesic distance $d(x,y)$ |
| `retr(x, u, t)` | Retract the step $tu$ from $T_x\mathcal M$ |
| `invretr(x, y)` | Return a local inverse-retraction displacement |
| `transport(x, y, u)` | Move $u\in T_x\mathcal M$ into $T_y\mathcal M$ |
| `egrad_to_rgrad(x, g)` | Convert an ambient gradient to $\operatorname{grad}f(x)$ |
| `random_point(key, sample_shape)` | Generate reproducible manifold points |
| `random_tangent(key, x)` | Generate reproducible tangent vectors |

The shared `exp_batch`, `log_batch`, and `dist_batch` methods apply the scalar
operations over a leading sample axis with `jax.vmap`. See
[Mathematical foundations](foundations.md) for the abstract definitions and the
[geometry API](../api/geometry.md) for signatures.

`operation_kind("exp")`, `operation_kind("log")`, and
`operation_kind("dist")` report `"exact"` or `"proxy"`. A proxy class keeps
the uniform names for composability, but evaluates `retr`, `invretr`, or the
inverse-retraction norm. `operation_kind("transport")` similarly distinguishes
parallel, isometric, and general vector transports.

## Euclidean

`Euclidean(size)` represents a real vector space with the array shape supplied
by `size`. For vectors, matrices, or tensors,

$$
T_x\mathbb R^N=\mathbb R^N,
\qquad
g_x(U,V)=\langle U,V\rangle_F=\sum_i U_iV_i.
$$

There are no constraints, so point and tangent projections are the identity.
The flat operations are

$$
\operatorname{Exp}_X(U)=X+U,
\qquad
\operatorname{Log}_X(Y)=Y-X,
\qquad
d(X,Y)=\lVert Y-X\rVert_F,
$$

and transport leaves the tangent vector unchanged.

## Products of spheres

`Oblique(size=(n, m))` is the oblique manifold

$$
\operatorname{OB}(n,m)
=\{X\in\mathbb R^{n\times m}:\operatorname{diag}(X^\top X)=\mathbf1\}.
$$

It is represented efficiently as a product of $m$ copies of $S^{n-1}$.
Point projection normalizes each column and tangent projection is

$$
\Pi_X(A)=A-X\operatorname{Diag}(X^\top A).
$$

The metric is Frobenius, while `exp`, `log`, distance, and parallel transport
apply the corresponding round-sphere formulas independently to every column.
Its dimension is $m(n-1)$.

## Probability simplex

`ProbabilitySimplex(size=d)` is the open simplex

$$
\Delta^{d-1}_{+}=\{p\in\mathbb R^d:p_i>0,\ \mathbf1^\top p=1\},
\qquad
T_p\Delta^{d-1}_{+}=\{u:\mathbf1^\top u=0\},
$$

with Fisher--Rao metric

$$
g_p(u,v)=\sum_{i=1}^d\frac{u_iv_i}{p_i}.
$$

The square-root map $p\mapsto2\sqrt p$ identifies the simplex with the
positive orthant of a sphere of radius two. GeoJAX uses this isometry for exact
geodesic operations; in particular,

$$
d(p,q)=2\arccos\left(\sum_i\sqrt{p_iq_i}\right).
$$

`retr` uses positive normalized addition for stable optimizer steps. The exact
exponential is local to the positive orthant and reports nonfinite output if a
step crosses that chart boundary.

## Sphere

### Intrinsic round geometry

`Sphere(size=n)` is the unit sphere $S^{n-1}$ embedded in $\mathbb R^n$:

$$
S^{n-1}=\{x\in\mathbb R^n:x^\top x=1\},
\qquad
T_xS^{n-1}=\{u:x^\top u=0\}.
$$

The point and tangent projections are

$$
\Pi_{S}(z)=\frac{z}{\lVert z\rVert},
\qquad
\Pi_x(a)=a-(x^\top a)x,
$$

with a fixed basis vector used when $z=0$. The metric is the ambient dot
product $g_x(u,v)=u^\top v$.

For $r=\lVert u\rVert$,

$$
\operatorname{Exp}_x(u)
=\cos(r)x+\frac{\sin(r)}{r}u.
$$

For $\theta=\arccos(x^\top y)$ and $y\ne -x$,

$$
\operatorname{Log}_x(y)
=\frac{\theta}{\sin\theta}\bigl(y-\cos(\theta)x\bigr),
\qquad
d(x,y)=\theta.
$$

Parallel transport along the unique shortest geodesic is

$$
\mathcal T_{x\to y}(u)
=u-\frac{u^\top y}{1+x^\top y}(x+y).
$$

At $y=-x$, the shortest geodesic and logarithm are not unique; `log` and
`transport` return nonfinite values rather than silently choosing a branch.

### Identity embedding

`SphereExtrinsic` uses the same points, metric, geodesics, and transport as
`Sphere`. It additionally exposes the equivariant embedding

$$
j(x)=x,
\qquad
d_{\mathrm{chord}}(x,y)=\lVert x-y\rVert_2.
$$

Its extrinsic mean projects the ambient mean back to the sphere:

$$
\bar x_{\mathrm{ext}}
=\frac{\sum_i w_i x_i}{\left\lVert\sum_i w_i x_i\right\rVert_2},
$$

provided the numerator is nonzero.

## Grassmann

### Canonical quotient geometry

`Grassmann(size=(n, k))` represents the set of $k$-dimensional subspaces of
$\mathbb R^n$. A public point is an orthonormal frame

$$
X\in\mathbb R^{n\times k},
\qquad
X^\top X=I_k,
$$

but $X$ and $XR$ represent the same point for every $R\in O(k)$. Tangent
vectors use the horizontal representation

$$
T_{[X]}\operatorname{Gr}(k,n)
\cong\{U\in\mathbb R^{n\times k}:X^\top U=0\}.
$$

QR factorization projects an ambient matrix to an orthonormal frame, and

$$
\Pi_X(A)=(I-XX^\top)A,
\qquad
g_X(U,V)=\operatorname{tr}(U^\top V).
$$

If the compact SVD of a tangent is $U=A\Sigma B^\top$, the exponential is
represented by

$$
\operatorname{Exp}_X(U)
=XB\cos(\Sigma)B^\top
+A\sin(\Sigma)B^\top
+X(I-BB^\top).
$$

To compute the logarithm, form

$$
M=(I-XX^\top)Y(X^\top Y)^{-1}.
$$

If $M=A\tan(\Theta)B^\top$, then

$$
\operatorname{Log}_X(Y)=A\Theta B^\top.
$$

The singular values of $X^\top Y$ are $\cos\theta_i$, where $\theta_i$ are the
principal angles, and

$$
d([X],[Y])=\left(\sum_{i=1}^k\theta_i^2\right)^{1/2}.
$$

`transport` applies the corresponding principal-plane rotation obtained from
the SVD of $\operatorname{Log}_X(Y)$. The logarithm requires $X^\top Y$ to be
nonsingular, so it is not defined by this chart when a principal angle equals
$\pi/2$.

### Projection embedding

`GrassmannProjection(size=(n, k))` deliberately keeps the same public
$n\times k$ frame representation. Internally it uses the equivariant embedding

$$
j([X])=P=XX^\top,
$$

where $P=P^\top=P^2$ and $\operatorname{tr}(P)=k$. A horizontal frame tangent
$U$ is embedded as

$$
dj_X(U)=H=UX^\top+XU^\top.
$$

Embedded tangents are symmetric off-diagonal blocks with respect to
$\operatorname{range}(P)$:

$$
H=PH(I-P)+(I-P)HP.
$$

The normalized Frobenius metric satisfies

$$
g_P(H,K)=\tfrac12\operatorname{tr}(HK)
=\operatorname{tr}(U^\top V),
$$

so the embedding is isometric to the canonical Grassmann geometry. For
$\Omega=HP-PH$, the projector geodesic is

$$
P(t)=e^{t\Omega}Pe^{-t\Omega}.
$$

GeoJAX evaluates exponential and transport with this projector formula, then
recovers an $n\times k$ frame from the top $k$ eigenvectors. Orthogonal
Procrustes alignment chooses the representative closest to a reference frame.
The intrinsic `dist` is still the principal-angle distance. The separate
extrinsic distance is

$$
d_{\mathrm{chord}}([X],[Y])
=\frac{\lVert XX^\top-YY^\top\rVert_F}{\sqrt2}.
$$

## Orthonormal frames

`Stiefel(size=(n, k))` and `StiefelEuclidean(size=(n, k))` share the real
Stiefel manifold

$$
\operatorname{St}(n,k)
=\{X\in\mathbb R^{n\times k}:X^\top X=I_k\},
$$

but equip it with different metrics. Unlike `Grassmann`, a Stiefel point is the
frame itself: $X$ and $XR$ are generally different points. The tangent space is

$$
T_X\operatorname{St}(n,k)
=\{U\in\mathbb R^{n\times k}:X^\top U+U^\top X=0\}.
$$

Both classes use the polar factor to project an ambient matrix $Z$ to a point,
and use the Frobenius-orthogonal tangent projection

$$
\Pi_X(Z)=Z-X\operatorname{sym}(X^\top Z).
$$

### Canonical quotient metric

`Stiefel` is the first-class intrinsic geometry. It views
$\operatorname{St}(n,k)$ as the quotient $O(n)/O(n-k)$ and uses

$$
g_X^{\mathrm{can}}(U,V)
=\operatorname{tr}\!\left(U^\top
\left(I-\tfrac12XX^\top\right)V\right).
$$

Write

$$
A=X^\top U,
\qquad
H=(I-XX^\top)U,
$$

where $A$ is skew-symmetric. The horizontal lift generator

$$
\Omega=HX^\top-XH^\top+XAX^\top
$$

is skew-symmetric and satisfies $\Omega X=U$. The exact canonical exponential
implemented by GeoJAX is

$$
\operatorname{Exp}^{\mathrm{can}}_X(U)=e^\Omega X.
$$

For an ambient Euclidean gradient $G$, the canonical Riemannian gradient is

$$
\operatorname{grad}f(X)=G-XG^\top X.
$$

### Embedded Euclidean metric

`StiefelEuclidean` uses the metric induced by the identity embedding into
$\mathbb R^{n\times k}$:

$$
g_X^{\mathrm E}(U,V)=\operatorname{tr}(U^\top V).
$$

Its Riemannian gradient is $\Pi_X(G)$. Its geodesics are not the canonical
geodesics. With $A=X^\top U$ and $S=U^\top U$, the exact exponential is

$$
\operatorname{Exp}^{\mathrm E}_X(U)
=
\begin{bmatrix}X&U\end{bmatrix}
\exp\!\left(
\begin{bmatrix}
A&-S\\
I&A
\end{bmatrix}
\right)
\begin{bmatrix}I\\0\end{bmatrix}
e^{-A}.
$$

This curve satisfies the embedded geodesic equation

$$
\ddot X(t)+X(t)\bigl(\dot X(t)^\top\dot X(t)\bigr)=0.
$$

The two metrics agree on horizontal tangents satisfying $X^\top U=0$. On a
vertical tangent $U=XA$, the canonical squared norm is half the Euclidean
squared norm. For $k=1$, the vertical component is absent and both classes
reduce to the round sphere geometry.

### Logarithms and transport

Neither metric has a general closed-form logarithm. `log_with_info(X, Y)` uses
damped Gauss--Newton endpoint shooting to solve
$\operatorname{Exp}_X(U)=Y$ in a metric-orthonormal tangent basis. It returns
the best tangent together with `converged`, `iterations`, `residual_norm`, and
`step_norm`. `log(X, Y)` returns nonfinite values when shooting does not meet
its tolerance, so a failed local inverse is never silently presented as a
geometric logarithm. `dist` is the norm of the selected converged logarithm.

General Levi-Civita parallel transport for these metrics is described by a
matrix differential equation rather than a simple endpoint formula. GeoJAX
therefore uses an isometric group-action vector transport. Complete $X$ and
$Y$ to orthogonal matrices $Q_X=[X,X_\perp]$ and $Q_Y=[Y,Y_\perp]$; then

$$
\mathcal T_{X\to Y}(U)=Q_YQ_X^\top U.
$$

This map is tangent and preserves either metric exactly, but it is not labeled
as Levi-Civita parallel transport. The formulas follow Edelman, Arias, and
Smith's [geometry of algorithms with orthogonality
constraints](https://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf).
Zimmermann's work gives a dedicated iterative treatment of the [canonical
Stiefel logarithm](https://arxiv.org/abs/1604.05054).

### Generalized orthogonality

Let $B\in\operatorname{SPD}(n)$. `GeneralizedStiefel` represents

$$
\operatorname{St}_B(n,k)=\{X:X^\top B X=I_k\},
$$

with tangent constraint

$$
X^\top B U+U^\top B X=0
$$

and metric $g_X(U,V)=\operatorname{tr}(U^\top B V)$. The map
$X\mapsto B^{1/2}X$ is an isometry to `StiefelEuclidean`; point projection,
exact exponential and numerical exact logarithm are pulled back through this
map.

`GeneralizedGrassmann` quotients these frames by the right action of $O(k)$.
Its horizontal tangents satisfy $X^\top B U=0$, and the same square-root
isometry pulls its exact geodesic operations back from `Grassmann`. These
classes are useful in generalized eigenvalue and constrained subspace
problems, where orthogonality is defined by a mass or covariance matrix rather
than the Euclidean inner product.

## Rotation matrices

`SpecialOrthogonal(size=n)` represents

$$
\operatorname{SO}(n)
=\{R\in\mathbb R^{n\times n}:R^\top R=I,\ \det R=1\}.
$$

Its tangent vectors are ambient matrices $U=R\Omega$ with
$\Omega^\top=-\Omega$. Point projection uses the orientation-preserving polar
factor, while tangent projection and the metric are

$$
\Pi_R(A)=R\operatorname{skew}(R^\top A),
\qquad
g_R(U,V)=\operatorname{tr}(U^\top V).
$$

The Frobenius metric is bi-invariant, so its Riemannian and group exponentials
coincide:

$$
\operatorname{Exp}_R(U)=R\exp(R^\top U),
\qquad
\operatorname{Log}_R(Q)=R\log(R^\top Q).
$$

If $\Omega=\log(R^\top Q)$, exact parallel transport is

$$
\mathcal T_{R\to Q}(U)
=R e^{\Omega/2}(R^\top U)e^{\Omega/2}.
$$

The principal logarithm is not unique when $R^\top Q$ has eigenvalue $-1$,
which includes a relative rotation by $\pi$. `log`, `dist`, and `transport`
return nonfinite values there instead of selecting an arbitrary rotation
plane. Away from that cut locus, GeoJAX supplies a custom JAX derivative for
the matrix logarithm.

## Rigid transformations

`SpecialEuclidean(size=n)` uses homogeneous matrices

$$
G=
\begin{bmatrix}R&t\\0&1\end{bmatrix}
\in\operatorname{SE}(n),
\qquad R\in\operatorname{SO}(n),\quad t\in\mathbb R^n.
$$

A tangent at $G$ is represented as

$$
U=
\begin{bmatrix}R\Omega&u\\0&0\end{bmatrix},
\qquad \Omega^\top=-\Omega,
$$

and GeoJAX equips the manifold with the direct-product metric

$$
g_G(U,V)=\operatorname{tr}((R\Omega_U)^\top R\Omega_V)+u^\top v.
$$

Therefore the Riemannian maps separate rotation and translation:

$$
\begin{aligned}
\operatorname{Exp}_{(R,t)}(R\Omega,u)&=(R e^\Omega,t+u),\\
\operatorname{Log}_{(R,t)}(Q,s)&=(R\log(R^\top Q),s-t),\\
d((R,t),(Q,s))^2&=\lVert\log(R^\top Q)\rVert_F^2+\lVert s-t\rVert_2^2.
\end{aligned}
$$

This Riemannian exponential is generally different from the matrix-group
exponential. If a Lie-algebra element is $(\Omega,v)$, then

$$
\exp_{\mathrm{grp}}(\Omega,v)
=\left(e^\Omega,J(\Omega)v\right),
\qquad
J(\Omega)=\int_0^1 e^{s\Omega}\,ds.
$$

Use `exp` and `log` for manifold optimization and distances. Use
`group_exp`, `group_log`, `compose`, `inverse`, and `apply` for group actions
and rigid-body kinematics.

## Symmetric positive-definite matrices

Both SPD geometries use

$$
\operatorname{SPD}(n)
=\{P\in\mathbb R^{n\times n}:P=P^\top,\ P\succ0\},
\qquad
T_P\operatorname{SPD}(n)=\operatorname{Sym}(n).
$$

`project` symmetrizes an input, eigendecomposes it, and clips eigenvalues below
the configured positive threshold. Tangent projection is symmetrization.

### Log-Euclidean metric

`SPDLogEuclidean` treats $\phi(P)=\log P$ as an isometry from SPD matrices to
symmetric matrices. With $D\log_P[U]$ denoting the Fréchet derivative,

$$
g_P(U,V)
=\left\langle D\log_P[U],D\log_P[V]\right\rangle_F.
$$

The geometry is Euclidean in log coordinates:

$$
\begin{aligned}
\operatorname{Exp}_P(U)
&=\exp\!\left(\log P+D\log_P[U]\right),\\
\operatorname{Log}_P(Q)
&=D\exp_{\log P}[\log Q-\log P],\\
d(P,Q)&=\lVert\log Q-\log P\rVert_F,\\
\mathcal T_{P\to Q}(U)
&=D\exp_{\log Q}\!\left[D\log_P[U]\right].
\end{aligned}
$$

The spectral Fréchet derivatives used here remain valid when $P$ and $U$ do
not commute.

### Affine-invariant metric

`SPDAffineInvariant` uses

$$
g_P(U,V)=\operatorname{tr}(P^{-1}UP^{-1}V).
$$

Its principal operations are

$$
\begin{aligned}
\operatorname{Exp}_P(U)
&=P^{1/2}\exp(P^{-1/2}UP^{-1/2})P^{1/2},\\
\operatorname{Log}_P(Q)
&=P^{1/2}\log(P^{-1/2}QP^{-1/2})P^{1/2},\\
d(P,Q)
&=\left\lVert\log(P^{-1/2}QP^{-1/2})\right\rVert_F.
\end{aligned}
$$

Writing

$$
E=P^{1/2}(P^{-1/2}QP^{-1/2})^{1/2}P^{-1/2},
$$

parallel transport is the congruence action

$$
\mathcal T_{P\to Q}(U)=EUE^\top.
$$

These operations become ill-conditioned near the positive-semidefinite
boundary because inverse square roots and matrix logarithms amplify small
eigenvalues.

### Bures-Wasserstein metric

`SPDBuresWasserstein` is the covariance part of the quadratic Wasserstein
geometry of Gaussian distributions. Let $\mathcal L_P(A)=PA+AP$ and denote its
inverse Sylvester solution by $\mathcal L_P^{-1}$. The metric is

$$
g_P(U,V)=\frac12\operatorname{tr}
\left(\mathcal L_P^{-1}(U)V\right).
$$

Equivalently, if $P=Z\operatorname{diag}(d_i)Z^\top$ and
$\widetilde U=Z^\top UZ$, then

$$
g_P(U,V)=\frac12\sum_{i,j}
\frac{\widetilde U_{ij}\widetilde V_{ij}}{d_i+d_j}.
$$

The optimal map carrying a zero-mean Gaussian covariance $P$ to $Q$ is

$$
T_{P\to Q}=P^{-1/2}
\left(P^{1/2}QP^{1/2}\right)^{1/2}P^{-1/2}.
$$

It gives

$$
\begin{aligned}
\operatorname{Log}_P(Q)
&=(T_{P\to Q}-I)P+P(T_{P\to Q}-I),\\
d(P,Q)^2
&=\operatorname{tr}P+\operatorname{tr}Q
-2\operatorname{tr}\left(P^{1/2}QP^{1/2}\right)^{1/2}.
\end{aligned}
$$

For $A=\mathcal L_P^{-1}(U)$, the local exponential is

$$
\operatorname{Exp}_P(U)=P+U+APA=(I+A)P(I+A).
$$

It remains in the smooth SPD chart while $I+A$ is nonsingular on the selected
positive branch. General Levi-Civita parallel transport has no comparable
closed endpoint expression. GeoJAX's `transport` instead maps through fixed
Euclidean metric coordinates; it is linear and exactly norm-preserving, making
it a valid vector transport for optimization, but it is not labeled as
parallel transport.

The square-root and inverse-square-root routines use custom Fréchet
derivatives. This avoids undefined eigenvector derivatives when an SPD matrix
has repeated eigenvalues, such as an isotropic covariance.

The formulas follow Bhatia, Jain, and Lim's
[Bures-Wasserstein analysis](https://arxiv.org/abs/1712.01504) and Malagò,
Montrucchio, and Pistone's
[Gaussian Riemannian geometry](https://arxiv.org/abs/1801.09269).

## Fixed-rank matrices

`FixedRank(size=(m, n), rank=k)` represents the smooth stratum

$$
\mathcal M_k=\{X\in\mathbb R^{m\times n}:\operatorname{rank}(X)=k\}
$$

with the embedded Frobenius metric. If $X=U\Sigma V^\top$ is a compact SVD,
the tangent projector is

$$
\Pi_X(Z)=UU^\top Z+ZVV^\top-UU^\top ZVV^\top.
$$

`retr` truncates the SVD of $X+\eta$ to rank $k$, and `invretr` projects
$Y-X$ into $T_X\mathcal M_k$. Closed exact geodesic maps are not provided:
`exp`, `log`, and `dist` are explicitly labeled retraction proxies.

## Fixed-rank positive semidefinite matrices

`RankKPSD(size=(n, n), rank=k)` represents

$$
\mathcal S^+_{n,k}=\{P=P^\top\succeq0:\operatorname{rank}(P)=k\}.
$$

For the support projector $S$, its embedded tangent projection is

$$
\Pi_P(Z)=SZ+ZS-SZS
$$

after symmetrizing $Z$. Eigenvalue truncation supplies the retraction, so its
geodesic-named compatibility operations are proxies.

`RankKPSDBuresWasserstein` uses the quotient factorization $P=YY^\top$, with
$Y$ identified with $YO$ for $O\in O(k)$. Procrustes alignment of factors
gives the exact quotient distance

$$
d(P,Q)=\min_{O\in O(k)}\lVert Y_P-Y_QO\rVert_F.
$$

On the fixed-rank stratum the metric uses the pseudoinverse Sylvester solution
$L$ of $PL+LP=U$:

$$
g_P(U,V)=\tfrac12\operatorname{tr}(LV).
$$

The class provides exact local Bures exponential, logarithm, and distance.
Its endpoint projection transport is a general vector transport, not parallel
transport.

### Elliptope and spectrahedron

`Elliptope(size=(n, n), rank=k)` is the rank-$k$ correlation stratum

$$
\{P\in\mathcal S^+_{n,k}:\operatorname{diag}(P)=\mathbf1\}.
$$

It uses unit-row factors $P=YY^\top$; its tangent projection combines the
rank-$k$ tangent constraint with $\operatorname{diag}(U)=0$.
`Spectrahedron(size=(n, n), rank=k)` instead imposes
$\operatorname{tr}(P)=1$ and $\operatorname{tr}(U)=0$. Both use the embedded
Frobenius metric, spectral retractions, and labeled retraction proxies.

## Full-rank correlation matrices

`CorrelationECM` and `CorrelationLEC` share the manifold

$$
\operatorname{Cor}^+(n)
=\{C\in\operatorname{SPD}(n):\operatorname{diag}(C)=\mathbf1\},
$$

with tangent space

$$
T_C\operatorname{Cor}^+(n)
=\{U\in\operatorname{Sym}(n):\operatorname{diag}(U)=0\}.
$$

Point projection first repairs positive definiteness and then rescales by the
diagonal. Tangent projection symmetrizes and zeros the diagonal.

Let $A=\operatorname{chol}(C)$ and define the unit lower-triangular Cholesky
map

$$
\Theta(C)=\operatorname{diag}(A)^{-1}A.
$$

Its inverse normalizes $LL^\top$ to unit diagonal:

$$
\Theta^{-1}(L)
=D^{-1/2}LL^\top D^{-1/2},
\qquad
D=\operatorname{diag}(LL^\top).
$$

### Euclidean-Cholesky metric

For `CorrelationECM`, the flat chart is the strictly lower-triangular part

$$
\phi_{\mathrm E}(C)=\operatorname{sl}(\Theta(C)).
$$

### Log-Euclidean-Cholesky metric

For `CorrelationLEC`, the flat chart is

$$
\phi_{\mathrm L}(C)=\log(\Theta(C)),
$$

which is strictly lower triangular because $\Theta(C)$ is unit lower
triangular.

For either chart $\phi$, the implementation uses the Euclidean pullback:

$$
\begin{aligned}
g_C(U,V)
&=\langle D\phi_C[U],D\phi_C[V]\rangle_F,\\
\operatorname{Exp}_C(U)
&=\phi^{-1}(\phi(C)+D\phi_C[U]),\\
\operatorname{Log}_C(B)
&=D(\phi^{-1})_{\phi(C)}[\phi(B)-\phi(C)],\\
d(C,B)&=\lVert\phi(B)-\phi(C)\rVert_F,\\
\mathcal T_{C\to B}(U)
&=D(\phi^{-1})_{\phi(B)}[D\phi_C[U]].
\end{aligned}
$$

Thus ECM and LEC have the same point set and tangent representation but
different geodesics and distances.

### Affine-invariant quotient metric

`CorrelationAffineQuotient` regards correlation matrices as the quotient of
SPD matrices under positive diagonal congruence. For
$U\in T_C\operatorname{Cor}^+(n)$, its horizontal SPD lift has the form

$$
U^\mathcal H=U+DC+CD,
$$

where diagonal $D$ is chosen so that
$\operatorname{diag}(C^{-1}U^\mathcal H)=0$. The quotient metric is

$$
g_C(U,V)
=\operatorname{tr}\left(C^{-1}U^\mathcal H C^{-1}V^\mathcal H\right).
$$

GeoJAX evaluates this metric and its metric-dual gradient exactly. It uses a
normalized-addition retraction and projected difference inverse retraction;
accordingly `exp`, `log`, and `dist` are documented proxies rather than claims
of closed affine-quotient geodesics.

## Poincare ball

`PoincareBall(size=d)` is the open unit ball
$\mathbb B^d=\{x:\lVert x\rVert<1\}$ with conformal metric

$$
g_x(u,v)=\lambda_x^2 u^\top v,
\qquad
\lambda_x=\frac{2}{1-\lVert x\rVert^2}.
$$

Using Mobius addition $\oplus$, its exact operations include

$$
\begin{aligned}
\operatorname{Exp}_x(u)
&=x\oplus\left(
\tanh\left(\frac{\lambda_x\lVert u\rVert}{2}\right)
\frac{u}{\lVert u\rVert}\right),\\
d(x,y)&=2\operatorname{artanh}\lVert(-x)\oplus y\rVert.
\end{aligned}
$$

Parallel transport uses the associated gyration and conformal-factor ratio.
`PoincareBall(size=d)` and `Hyperboloid(size=d+1)` describe the same
constant-curvature $-1$ geometry in different coordinates.

## Hyperboloid

`Hyperboloid(size=n)` represents $\mathbb H^{n-1}$ in ambient
$\mathbb R^n$. For the Lorentz product

$$
\langle x,y\rangle_L=-x_0y_0+\sum_{i=1}^{n-1}x_i y_i,
$$

the upper sheet and its tangent space are

$$
\mathbb H^{n-1}
=\{x:\langle x,x\rangle_L=-1,\ x_0>0\},
\qquad
T_x\mathbb H^{n-1}=\{u:\langle x,u\rangle_L=0\}.
$$

The point projection retains the spatial coordinates and sets
$x_0=\sqrt{1+\lVert x_{1:}\rVert^2}$. Tangent projection and the metric are

$$
\Pi_x(a)=a+\langle x,a\rangle_Lx,
\qquad
g_x(u,v)=\langle u,v\rangle_L.
$$

For $r=\lVert u\rVert_x$,

$$
\operatorname{Exp}_x(u)
=\cosh(r)x+\frac{\sinh(r)}{r}u.
$$

Writing $\alpha=-\langle x,y\rangle_L\ge1$,

$$
\operatorname{Log}_x(y)
=\frac{\operatorname{arcosh}(\alpha)}{\sqrt{\alpha^2-1}}
(y-\alpha x),
\qquad
d(x,y)=\operatorname{arcosh}(\alpha).
$$

Parallel transport along the unique geodesic is

$$
\mathcal T_{x\to y}(u)
=u+\frac{\langle y,u\rangle_L}{1-\langle x,y\rangle_L}(x+y).
$$

## Torus

`Torus(size=d)` is the flat product of $d$ circles,

$$
\mathbb T^d=(\mathbb R/2\pi\mathbb Z)^d,
$$

represented by its unique angle vector in $[-\pi,\pi)^d$. Define

$$
\operatorname{wrap}(z)=(z+\pi)\bmod 2\pi-\pi
$$

componentwise. Then $T_x\mathbb T^d\cong\mathbb R^d$,
$g_x(u,v)=u^\top v$, and

$$
\begin{aligned}
\operatorname{Exp}_x(u)&=\operatorname{wrap}(x+u),\\
\operatorname{Log}_x(y)&=\operatorname{wrap}(y-x),\\
d(x,y)&=\lVert\operatorname{wrap}(y-x)\rVert_2.
\end{aligned}
$$

Transport is the identity in angular coordinates. If a component differs by
exactly $\pi$, two shortest directions exist and the half-open interval chooses
one branch.

## Kendall shape space

`KendallShape(size=(m, d))` represents configurations of $m$ labeled landmarks
in $\mathbb R^d$ after removing translation, scale, and orientation. Public
points are regular pre-shapes

$$
X\in\mathbb R^{m\times d},
\qquad
\mathbf1^\top X=0,
\qquad
\lVert X\rVert_F=1,
$$

with $X$ and $XR$, $R\in SO(d)$, representing the same shape. Horizontal
tangents satisfy

$$
\mathbf1^\top U=0,
\qquad
\langle X,U\rangle_F=0,
\qquad
X^\top U=U^\top X.
$$

Orientation-preserving Procrustes alignment selects the closest representative
of a second shape. Exact quotient exponential, logarithm, and distance then
follow the great-circle formulas on the pre-shape sphere. The endpoint
projection transport is a general vector transport. Singular pre-shapes are
excluded because they are not part of the regular quotient stratum.

## Product

`Product(factors)` accepts any pytree of geometries. Points and tangent vectors
must have exactly the same tree structure. If leaves are indexed by $i$, then

$$
g_x(u,v)=\sum_i g_{x_i}^{(i)}(u_i,v_i),
\qquad
d(x,y)=\left(\sum_i d_i(x_i,y_i)^2\right)^{1/2}.
$$

Point projection, tangent projection, exponential, logarithm, retraction,
transport, pair means, gradient conversion, and Hessian conversion act leaf by
leaf. Random generation splits the supplied JAX key once per factor while
preserving the pytree structure.
