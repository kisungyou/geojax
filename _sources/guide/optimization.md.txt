# Optimization

Optimization follows a class-style interface:

```python
problem = Minimize(M=M, cost=cost, x0=x0, solver=solver)
solution, final_cost, history = problem.solve()
```

## Choosing a solver

| Solver | Useful starting point |
|---|---|
| `SteepestDescent` | transparent baseline and debugging |
| `ConjugateGradient` | default smooth first-order solver |
| `LBFGS` | larger smooth problems with useful curvature history |
| `BarzilaiBorwein` | inexpensive spectral step sizes |
| `TrustRegions` | problems with a trustworthy Riemannian Hessian |
| `ParticleSwarm` | derivative-free exploratory search |
| `NelderMead` | small derivative-free problems; currently heuristic on curved spaces |

Costs should be scalar JAX functions. You may provide a Riemannian `grad`, an
ambient `egrad`, or let JAX differentiate the cost. For second-order work,
provide `rhess_vec` unless the geometry documents an exact conversion.

See the complete [optimization API](../api/optimization.md).

