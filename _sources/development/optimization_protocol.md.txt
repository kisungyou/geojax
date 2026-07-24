# Optimization protocol

GeoJAX separates four roles:

1. a geometry defines points and tangent operations,
2. a problem defines an objective and derivative products,
3. an optional line search globalizes a search direction, and
4. a solver owns iteration state and stopping rules.

This separation is consistent with established Riemannian optimization
interfaces and algorithmic treatments
{cite:p}`absil2008optimization,boumal2014manopt,boumal2023introduction`.

The public entry point is

```python
problem = Minimize(M=M, cost=cost, x0=x0, solver=solver)
solution, final_cost, history = problem.solve()
```

## General problem contract

Every `Minimize` problem provides:

| Field | Meaning |
|---|---|
| `M` | geometry satisfying the manifold protocol |
| `cost(x)` | scalar JAX objective |
| `x0` | initial point, projected before solving |
| `solver` | class-style object implementing `solve(problem)` |

Optional fields are `grad`, `egrad`, `precon`, `ehess_vec`, `rhess_vec`, and
`key`. If `x0` is omitted, `Minimize` draws a point with `M.random_point`.

For a metric $g_x$, the Riemannian gradient is characterized by

$$
g_x(\operatorname{grad} f(x),u)=\mathrm Df(x)[u]
\qquad\text{for all }u\in T_x\mathcal M.
$$

GeoJAX obtains it in this order:

1. use a supplied Riemannian `grad(x)`,
2. convert a supplied ambient `egrad(x)` with `M.egrad_to_rgrad`, or
3. differentiate `cost` with JAX and convert the ambient result.

## Specialized problem contracts

`LeastSquares` defines

$$
f(x)=\frac12\lVert r(x)\rVert_2^2,
\qquad
\operatorname{grad} f(x)=J_x^*r(x),
$$

where $J_x=\mathrm Dr(x)$ and the adjoint is taken between the Euclidean
residual metric and $g_x$. It adds:

```text
residual_value(x)
residual_norm(x)
jacobian_vec(x, u)
adjoint_jacobian(x, z)
normal_operator(x, u, damping=0)
```

JAX supplies Jacobian-vector and vector-Jacobian products by default, so no
dense Jacobian is formed. User callbacks may replace either product.

`FiniteSum` defines

$$
f(x)=\frac1N\sum_{i=1}^{N} f_i(x)
$$

and adds `sample_batch(key, batch_size)` and
`batch_cost_and_grad(x, indices)`. A stochastic solver should consume those
methods instead of assuming that samples are stored in one array.

## JAX transformation boundary

Costs, residuals, geometry primitives, gradient conversion, Jacobian products,
and Hessian-vector products are numerical JAX kernels. With static geometry
dimensions and pytree structure, they may be differentiated, vectorized, or
compiled independently:

```python
compiled_cost = jax.jit(cost)
compiled_gradient = jax.jit(jax.grad(cost))
compiled_hessian_product = jax.jit(problem.rhess_vec)
```

`solve()` is intentionally outside that boundary. Solver drivers use Python
loops for line-search decisions, stopping callbacks, wall-clock limits,
printing, and `InfoEntry` construction. They also convert accepted diagnostics
to Python scalars. Consequently, do not wrap `problem.solve()` in `jax.jit`.
Passing JIT-compiled costs or derivative callbacks to a problem remains
supported.

Transformation tests cover autodiff and JIT-compiled Hessian products for
ordinary arrays, SPD matrices, and nested Product states. A solver should not
claim whole-solver JIT compatibility unless it supplies a separate pure state
transition expressed with JAX control-flow primitives.

## Solver contract

A public class-style solver follows this structural interface:

```python
class SolverProtocol(Protocol):
    requires_gradient: bool

    def solve(
        self, problem: Minimize
    ) -> tuple[Any, float, list[InfoEntry]]: ...
```

The returned point belongs to `problem.M`; the scalar is a Python `float`; and
the history has one row for the initial point plus one per completed outer
iteration. Solvers expose compatible stopping fields where meaningful:

```text
tolgradnorm, maxiter, maxtime, minstepsize,
verbosity, statsfun, stopfun
```

`statsfun(problem, x, entry)` returns optional values for `entry.extra`.
`stopfun(problem, x, entry)` returns `(stop, reason)` after standard stopping
rules are evaluated.

## Line-search contract

A public strategy implements:

```python
class LineSearchProtocol(Protocol):
    def search(
        self,
        problem,
        x,
        direction,
        cost,
        directional_derivative,
        *,
        state=None,
        initial_alpha=None,
    ) -> LineSearchResult: ...
```

The result contains the accepted point, cost, optional gradient, displacement
norm, tangent multiplier, diagnostics, and state for the next search. A solver
must reuse `result.cost` and `result.gradient` when present rather than
recomputing them.

For a retraction $R_x$ and descent direction $d$, Armijo accepts $\alpha>0$
when

$$
f(R_x(\alpha d))
\leq f(x)+c_1\alpha\,g_x(\operatorname{grad}f(x),d).
$$

`StrongWolfe` also checks a curvature condition by pairing the new gradient
with the transported original direction. This is the exact curve derivative
for geodesics with parallel transport and a standard vector-transport proxy
for general retractions.

Line searches live in `geojax.optimization.linesearch`; solvers must not embed
private Armijo loops.

## Geometry requirements

| Solver family | Geometry and problem operations consumed |
|---|---|
| First-order line search | `egrad_to_rgrad`, `inner`, `norm`, `retr` or `exp` |
| Conjugate gradient, BB, L-BFGS | first-order operations plus `transport` and tangent linear combinations |
| Newton-CG | first-order operations plus `rhess_vec` and optional preconditioning |
| Trust regions and cubic regularization | Newton operations plus tangent model solves and acceptance diagnostics |
| Gauss--Newton and LM | `LeastSquares` Jacobian, adjoint-Jacobian, and normal-operator products |
| Stochastic gradient | `FiniteSum` sampling and mini-batch gradient methods |
| Alternating gradient | first-order operations and Product pytree leaf structure |
| Particle swarm | `random_point`, `random_tangent`, `log`, `retr` or `exp`, `transport`, `norm` |
| Nelder--Mead | `exp`, `log`, `dist`, and midpoint-like averaging |

New implementations should call the helpers in
`geojax.optimization.minimize` for retraction, transport, tangent arithmetic,
derivative evaluation, stopping, and history construction.

## Second-order products

`Minimize.rhess_vec(x, u)` first uses a user callback. Otherwise it
differentiates the available ambient or Riemannian gradient only when the
geometry advertises the corresponding conversion as exact through
`operation_kind("ehess_to_rhess")` or `operation_kind("rgrad_jvp")`.
Unsupported automatic paths raise an error and require an explicit
`rhess_vec`; tangent projection is not a mathematically valid generic fallback.

Newton-CG approximately solves

$$
\operatorname{Hess}f(x)[\eta]=-\operatorname{grad}f(x)
$$

with the shared tangent conjugate-gradient routine. The routine reports
residual convergence and non-positive curvature, and the outer solver falls
back to a descent direction when needed.

Trust regions use the quadratic model

$$
m_x(\eta)=f(x)
+g_x(\operatorname{grad}f(x),\eta)
+\frac12g_x(\eta,\operatorname{Hess}f(x)[\eta]),
$$

subject to a radius. Adaptive cubic regularization adds
$\sigma\lVert\eta\rVert_x^3/3$ and adapts $\sigma$ using actual versus
predicted decrease. Both methods place model diagnostics in `InfoEntry.extra`.

## PyTree tangent state

Points, gradients, search directions, momentum, and Hessian products may be
JAX pytrees. Solver implementations use shared helpers rather than raw array
expressions:

```python
tree_neg(g)
tree_lincomb(1.0, eta, alpha, direction)
tree_zeros_like(x)
```

Every tangent returned by a solver has the same tree structure as its base
point. Product geometries additionally require points to match the factor tree
leaf for leaf. `AlternatingGradient` addresses blocks by deterministic flattened
leaf index but returns the original nested structure.

## Iteration records

`InfoEntry` has common fields

```text
iter, cost, gradnorm, stepsize, time,
linesearch, beta, reason, extra
```

Line-search counts and multipliers belong in `linesearch`. Trust ratios,
regularization or damping values, inner iteration counts, stochastic learning
rates, and block summaries belong in `extra`. A terminating history row always
has a nonempty `reason`.

## Extension checklist

A new solver should:

1. expose a public class with `solve(problem)` and `requires_gradient`,
2. use the most specific problem contract it needs,
3. accept shared stopping and callback options where meaningful,
4. use the public line-search protocol when globalizing a direction,
5. perform PyTree-safe tangent arithmetic,
6. construct history rows with `make_info`,
7. return `(solution, final_cost, history)`, and
8. include focused Euclidean, curved-manifold, and Product tests as applicable.

## References

```{bibliography}
:filter: docname in docnames
```
