# Development

GeoJAX separates geometry, learning primitives, problem definition, and solver
implementation through small structural protocols. Geometry objects provide
manifold operations; learning functions compose them inside JAX models;
`Minimize` supplies an objective and its derivatives; solvers consume both and
return a common iteration history.

```{toctree}
:maxdepth: 1

geometry_protocol
learning_protocol
optimization_protocol
testing
documentation
```
