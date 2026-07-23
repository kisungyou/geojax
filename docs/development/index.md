# Development

GeoJAX separates geometry, problem definition, and solver implementation through
small structural protocols. Geometry objects provide manifold primitives;
`Minimize` supplies an objective and its derivatives; solvers consume both and
return a common iteration history.

```{toctree}
:maxdepth: 1

geometry_protocol
optimization_protocol
testing
documentation
```
