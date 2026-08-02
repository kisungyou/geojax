# Development

GeoJAX separates geometry, manifold-data adaptation, learning algorithms,
problem definition, and solver implementation through small structural
protocols. Geometry objects provide mathematical operations; learning methods
consume validated observations and declared capabilities; `Minimize` supplies
an objective and its derivatives; solvers return a common iteration history.

```{toctree}
:maxdepth: 1

geometry_protocol
learning_protocol
optimization_protocol
testing
documentation
```
