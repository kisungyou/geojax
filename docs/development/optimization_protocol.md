# Optimization protocol

GeoJAX optimization separates a problem description from the algorithm that
solves it. The public entry point is

```python
problem = Minimize(M=M, cost=cost, x0=x0, solver=solver)
solution, final_cost, history = problem.solve()
```

`Minimize` owns the manifold, objective, initial point, derivative hooks, and
random key. A solver owns algorithmic options and implements `solve(problem)`.
This boundary lets one solver work with every geometry that supplies the
operations it needs.

## Problem contract

Every problem provides:

| Field | Meaning |
|---|---|
| `M` | Geometry satisfying the manifold protocol |
| `cost(x)` | Scalar JAX objective to minimize |
| `x0` | Initial manifold point; projected before solving |
| `solver` | Class-style solver with `solve(problem)` |

The optional fields are `grad`, `egrad`, `precon`, `ehess_vec`,
`rhess_vec`, and `key`. If `x0` is omitted, `Minimize` draws one with
`M.random_point`.

For a metric $g_x$, the Riemannian gradient is defined by

$$
g_x(\operatorname{grad} f(x), u) = \mathrm{D}f(x)[u]
\qquad\text{for every }u\in T_x\mathcal{M}.
$$

GeoJAX obtains this vector in the following order:

1. Use the supplied Riemannian `grad(x)`.
2. Convert a supplied ambient `egrad(x)` with `M.egrad_to_rgrad`.
3. Differentiate `cost` with JAX and convert the ambient result.

Supplying `grad` therefore takes precedence over both `egrad` and automatic
differentiation.

## Solver contract

A public class-style solver follows this structural interface:

```python
class SolverProtocol(Protocol):
    requires_gradient: bool

    def solve(
        self, problem: Minimize
    ) -> tuple[Any, float, list[InfoEntry]]: ...
```

The returned point must belong to `problem.M`. The scalar is the final objective
value as a Python `float`. The history contains one `InfoEntry` for the initial
state and one for every completed outer iteration.

All solvers expose compatible stopping fields where applicable:

```text
tolgradnorm, maxiter, maxtime, minstepsize,
verbosity, statsfun, stopfun
```

`statsfun(problem, x, entry)` may return additional values for
`entry.extra`. `stopfun(problem, x, entry)` returns `(stop, reason)` and is
evaluated after the standard stopping rules.

## Geometry operations by solver family

The exact set of required operations depends on the algorithm.

| Solver family | Geometry operations consumed |
|---|---|
| First-order line search | `egrad_to_rgrad`, `inner`, `norm`, `retr` or `exp` |
| Conjugate gradient, BB, L-BFGS | First-order operations plus `transport` and tangent linear combinations |
| Trust regions | First-order operations plus `rhess_vec`, preconditioning, and tangent zero/linear combinations |
| Particle swarm | `random_point`, `random_tangent`, `log`, `retr` or `exp`, `transport`, `norm` |
| Nelder-Mead | `exp`, `log`, `dist`, and midpoint-like averaging |

New solvers should call the shared helpers in `geojax.optimization.minimize`
for retraction, transport, tangent arithmetic, derivative evaluation, stopping,
and history construction. This keeps fallback behavior and PyTree handling
consistent across algorithms.

## Second-order derivatives

`Minimize.rhess_vec(x, u)` first uses a user-supplied Riemannian
Hessian-vector product. Otherwise it differentiates the available gradient and,
when the geometry provides `ehess_to_rhess`, performs the geometry-specific
conversion. The final fallback tangent-projects the ambient Hessian-vector
product.

For the quadratic trust-region model,

$$
m_x(\eta)
= f(x)
+ g_x(\operatorname{grad}f(x),\eta)
+ \frac{1}{2}g_x(\eta,\operatorname{Hess}f(x)[\eta]),
$$

the preferred extension point is a supplied `rhess_vec`. The tangent-projected
fallback is useful for prototyping but is not an exact Riemannian Hessian on
every embedded or quotient geometry.

## PyTree states

Points, gradients, search directions, and Hessian-vector products may be JAX
PyTrees. Solver implementations must use the shared tree helpers rather than
raw array expressions:

```python
tree_neg(g)
tree_lincomb(1.0, eta, alpha, direction)
tree_zeros_like(x)
```

Every returned tangent must have the same tree structure as its base point.
Product geometries additionally require each point tree to match the geometry
tree leaf for leaf.

## Iteration records

`InfoEntry` has the common fields

```text
iter, cost, gradnorm, stepsize, time,
linesearch, beta, reason, extra
```

Gradient-based solvers store the Riemannian gradient norm in `gradnorm`.
Derivative-free methods may use `NaN` or an algorithm-specific convergence
quantity, so downstream reporting should also inspect the solver type and
`entry.extra`. Algorithm-specific diagnostics such as trust-region acceptance,
boundary hits, and negative curvature belong in `extra`.

## Extension checklist

A new solver should:

1. Be a public class with `solve(problem)` and `requires_gradient`.
2. Accept the shared stopping and callback options where meaningful.
3. Use manifold retractions, transports, inner products, and norms rather than
   ambient substitutes.
4. Use PyTree-safe tangent arithmetic.
5. Create every history row with `make_info` and store solver-specific values in
   `InfoEntry.extra`.
6. Return `(solution, final_cost, history)` with a nonempty stopping reason.
7. Include a small manifold optimization test and a Product-PyTree test when the
   algorithm performs tangent arithmetic.
