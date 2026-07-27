"""Public geometry namespace for GeoJAX."""

from __future__ import annotations

from .base import (
    ExactGeometryMixin,
    GeometryMixin,
    GeometryProtocol,
    ManifoldProtocol,
    RetractionGeometryMixin,
)
from .elementary import Oblique, PoincareBall, ProbabilitySimplex
from .euclidean import Euclidean
from .generalized import GeneralizedGrassmann, GeneralizedStiefel
from .grassmann import Grassmann, GrassmannProjection
from .hyperboloid import Hyperboloid
from .lie_groups import SpecialEuclidean, SpecialOrthogonal
from .low_rank import (
    Elliptope,
    FixedRank,
    RankKPSD,
    RankKPSDBuresWasserstein,
    Spectrahedron,
)
from .product import Product
from .shape import KendallShape
from .spd import SPDAffineInvariant, SPDBuresWasserstein, SPDLogEuclidean
from .sphere import Sphere, SphereExtrinsic
from .stiefel import Stiefel, StiefelEuclidean, StiefelLogInfo
from .torus import Torus
from .correlation import CorrelationAffineQuotient, CorrelationECM, CorrelationLEC

__all__ = [
    "Euclidean",
    "ExactGeometryMixin",
    "GeometryProtocol",
    "GeometryMixin",
    "ManifoldProtocol",
    "RetractionGeometryMixin",
    "Oblique",
    "ProbabilitySimplex",
    "PoincareBall",
    "Sphere",
    "SphereExtrinsic",
    "Grassmann",
    "GrassmannProjection",
    "GeneralizedStiefel",
    "GeneralizedGrassmann",
    "Stiefel",
    "StiefelEuclidean",
    "StiefelLogInfo",
    "Hyperboloid",
    "Torus",
    "Product",
    "SPDLogEuclidean",
    "SPDAffineInvariant",
    "SPDBuresWasserstein",
    "FixedRank",
    "RankKPSD",
    "RankKPSDBuresWasserstein",
    "Elliptope",
    "Spectrahedron",
    "CorrelationECM",
    "CorrelationLEC",
    "CorrelationAffineQuotient",
    "KendallShape",
    "SpecialOrthogonal",
    "SpecialEuclidean",
]
