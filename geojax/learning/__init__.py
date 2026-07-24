"""Pure-JAX primitives for manifold-valued learning."""

from ._geometry import geodesic_interpolate, pairwise_squared_dist, tangent_map

__all__ = ["geodesic_interpolate", "pairwise_squared_dist", "tangent_map"]
