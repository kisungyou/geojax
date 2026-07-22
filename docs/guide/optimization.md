# Optimization

Optimization follows a class-style interface:

```python
problem = Minimize(M=M, cost=cost, x0=x0, solver=solver)
solution, final_cost, history = problem.solve()
```

The cost is a scalar JAX function. Supply a Riemannian `grad`, an ambient
`egrad`, or let JAX differentiate the cost and let the geometry convert the
result.

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
contract directly.

```python
problem = LeastSquares(
    M=M,
    residual=residual,
    x0=x0,
    solver=LevenbergMarquardt(),
)
```

`FiniteSum` represents $f(x)=N^{-1}\sum_{i=1}^N f_i(x)$ without requiring
`StochasticGradient` to evaluate every term at every update.

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
`Product` geometry. The remaining gradient solvers consume `Minimize`.

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

`AdaptiveArmijo` is the default for first-order methods. Newton-CG and
Gauss--Newton use its unnormalized full-step form, so a unit Newton step is
tested first. Every strategy returns common cost/gradient evaluation counts and
its accepted multiplier in `InfoEntry.linesearch`.

## Second-order models

`NewtonCG` and `TrustRegions` use `problem.rhess_vec(x, u)` rather than forming
a Hessian matrix. `AdaptiveRegularizationCubics` approximately minimizes

$$
m_x(\eta)
=f(x)+g_x(\operatorname{grad}f(x),\eta)
+\frac12g_x(\eta,\operatorname{Hess}f(x)[\eta])
+\frac{\sigma}{3}\lVert\eta\rVert_x^3.
$$

The regularization parameter $\sigma$ is adapted from the agreement between
predicted and actual decrease. The iteration history records acceptance,
gain ratio, regularization, inner iterations, and curvature events in
`InfoEntry.extra`.

For second-order scientific work, supply `rhess_vec` when the geometry does
not document an exact Hessian conversion. JAX-based fallback products are
convenient for prototypes, but tangent projection alone does not encode every
connection term on every manifold.

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
