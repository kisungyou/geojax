# Optimization

Optimization follows a class-style interface:

```python
problem = Minimize(M=M, cost=cost, x0=x0, solver=solver)
solution, final_cost, history = problem.solve()
```

The cost is a scalar JAX function. Supply a Riemannian `grad`, an ambient
`egrad`, or let JAX differentiate the cost and let the geometry convert the
result. The underlying smooth-manifold optimization framework follows
{cite:t}`absil2008optimization` and {cite:t}`boumal2023introduction`.

## JAX transformation boundary

Geometry methods and numerical derivative products can be used inside
`jax.jit`, `jax.vmap`, `jax.grad`, and `jax.jvp` with fixed dimensions and
pytree structure. Random methods take explicit PRNG keys, while batch helper
methods vectorize over a leading sample axis.

The solver driver itself is not a single JIT kernel. `solve()` performs Python
stopping logic, line-search control flow, callbacks, timing, and construction
of the human-readable iteration history. Compile expensive model components
instead:

```python
cost = jax.jit(cost)
egrad = jax.jit(jax.grad(cost))

problem = Minimize(M=M, cost=cost, egrad=egrad, x0=x0, solver=solver)
```

Do not call `jax.jit(problem.solve)`. This explicit boundary keeps diagnostics
and extension hooks flexible while preserving compiled numerical kernels.

## Problem forms

`Minimize` represents a general smooth objective

$$
\min_{x\in\mathcal M} f(x).
$$

For a residual map $r:\mathcal M\to\mathbb R^m$, `LeastSquares` constructs

$$
f(x)=\frac12\lVert r(x)\rVert_2^2
$$

and provides matrix-free products with the residual Jacobian $J_x$ and its
adjoint $J_x^*$. `GaussNewton` and `LevenbergMarquardt` consume this richer
contract directly {cite:p}`nocedal2006numerical,levenberg1944method,marquardt1963algorithm`.

```python
problem = LeastSquares(
    M=M,
    residual=residual,
    x0=x0,
    solver=LevenbergMarquardt(),
)
```

`FiniteSum` represents $f(x)=N^{-1}\sum_{i=1}^N f_i(x)$ without requiring
`StochasticGradient` to evaluate every term at every update. The Riemannian
stochastic-gradient convergence framework is given by
{cite:t}`bonnabel2013stochastic`.

```python
problem = FiniteSum(
    M=M,
    loss=lambda x, i: per_sample_loss(x, data[i]),
    num_terms=N,
    x0=x0,
    solver=StochasticGradient(batch_size=32),
    key=0,
)
```

## Choosing a solver

| Solver | Useful starting point |
|---|---|
| `SteepestDescent` | transparent baseline and debugging |
| `ConjugateGradient` | economical default smooth first-order method |
| `LBFGS` | smooth problems where a short curvature history helps |
| `BarzilaiBorwein` | low-memory spectral step estimates |
| `NewtonCG` | matrix-free Newton steps with reliable Hessian products |
| `TrustRegions` | robust second-order steps and indefinite Hessians |
| `AdaptiveRegularizationCubics` | second-order models globalized by a cubic term |
| `GaussNewton` | well-conditioned nonlinear least squares |
| `LevenbergMarquardt` | nonlinear least squares needing adaptive damping |
| `StochasticGradient` | large finite sums and mini-batch training |
| `AlternatingGradient` | block updates on a `Product` geometry |
| `ParticleSwarm` | derivative-free exploratory search |
| `NelderMead` | small derivative-free problems |

Gauss--Newton and Levenberg--Marquardt require `LeastSquares`.
`StochasticGradient` requires `FiniteSum`, and `AlternatingGradient` requires a
`Product` geometry. All remaining solvers, including the derivative-free
methods, consume `Minimize`.

The table spans classical spectral, quasi-Newton, derivative-free, and
second-order families
{cite:p}`barzilai1988twopoint,liu1989limited,nelder1965simplex,kennedy1995particle,cartis2011adaptive`.

## Line searches

The gradient solvers accept a reusable strategy through `line_search`:

```python
from geojax.optimization import ConjugateGradient, StrongWolfe

solver = ConjugateGradient(line_search=StrongWolfe())
```

| Strategy | Behavior |
|---|---|
| `ConstantStep` | fixed multiplier, useful when a stable scale is known |
| `BacktrackingArmijo` | monotone sufficient decrease |
| `AdaptiveArmijo` | Armijo search initialized from the previous decrease |
| `StrongWolfe` | sufficient decrease plus a curvature condition |

`AdaptiveArmijo` is the default for steepest descent, conjugate gradient,
L-BFGS, and alternating gradient. Barzilai--Borwein uses non-normalized
`BacktrackingArmijo`; stochastic gradient has no line search. Newton-CG and
Gauss--Newton use an unnormalized adaptive Armijo search, so a unit Newton step
is tested first. Every strategy returns common cost/gradient evaluation counts
and its accepted multiplier in `InfoEntry.linesearch`.

`StrongWolfe` pairs the trial gradient with the transported initial direction.
That is the exact derivative of the search curve for a geodesic with parallel
transport. With a general retraction and vector transport, it is the standard
transported-derivative proxy, so classical Wolfe guarantees require the usual
compatibility assumptions. The sufficient-decrease and curvature conditions
trace to {cite:t}`armijo1966minimization` and {cite:t}`wolfe1969convergence`.

## Second-order models

`NewtonCG`, `TrustRegions`, and `AdaptiveRegularizationCubics` use
`problem.rhess_vec(x, u)` rather than forming a Hessian matrix. The cubic method
approximately minimizes

$$
m_x(\eta)
=f(x)+g_x(\operatorname{grad}f(x),\eta)
+\frac12g_x(\eta,\operatorname{Hess}f(x)[\eta])
+\frac{\sigma}{3}\lVert\eta\rVert_x^3.
$$

The regularization parameter $\sigma$ is adapted from the agreement between
predicted and actual decrease. The iteration history records acceptance,
gain ratio, regularization, inner iterations, and curvature events in
`InfoEntry.extra` {cite:p}`cartis2011adaptive`.

Automatic ambient-to-Riemannian Hessian conversion is currently exact for:

| Exact automatic path | Geometries |
|---|---|
| Ambient `egrad` or autodiff cost | `Euclidean`, `Sphere`, `SphereExtrinsic`, `Torus`, `GrassmannProjection`, `GeneralizedStiefel`, `StiefelEuclidean`, `SpecialOrthogonal`, `SpecialEuclidean` |
| JVP of a supplied Riemannian `grad` | `Euclidean`, `Sphere`, `SphereExtrinsic`, `Torus`, `GeneralizedStiefel`, `StiefelEuclidean`, `SpecialOrthogonal`, `SpecialEuclidean` |
| Product geometry | Exact only when every factor supports the selected path |

For every other geometry, supply `rhess_vec`. GeoJAX raises an error instead of
silently substituting tangent projection, because projection alone omits
connection or embedding-curvature terms. `operation_kind("ehess_to_rhess")`
and `operation_kind("rgrad_jvp")` expose these capabilities programmatically.
Gauss--Newton and Levenberg--Marquardt use the `LeastSquares` normal operator
instead and do not require this generic Hessian conversion.

## Product blocks and pytrees

`AlternatingGradient` follows JAX's deterministic leaf order for the geometry
pytree. One outer iteration visits every Product factor, recomputing the full
Riemannian gradient before each block update. The point retains its original
dict/list/tuple nesting throughout.

All other tangent-arithmetic solvers are pytree-safe as well: directions,
gradients, transports, and Hessian-vector products have the same structure as
the point.

See the complete [optimization API](../api/optimization.md) and the executable
[solver comparison](../tutorials/solver_comparison.md).

## References

```{bibliography}
:filter: docname in docnames
```
