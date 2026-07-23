"""Public optimization namespace for GeoJAX."""

from .minimize import Minimize, InfoEntry, LineSearchStats
from .linesearch import (
    AdaptiveArmijo,
    BacktrackingArmijo,
    ConstantStep,
    LineSearchProtocol,
    LineSearchResult,
    LineSearchState,
    StrongWolfe,
)
from .problems import FiniteSum, LeastSquares
from ._steepest_descent import SteepestDescent
from ._conjugate_gradient import ConjugateGradient
from .trustregions import TrustRegions
from .barzilaiborwein import BarzilaiBorwein
from .lbfgs import LBFGS
from .particleswarm import ParticleSwarm
from .neldermead import NelderMead
from .adaptiveregularizationcubics import AdaptiveRegularizationCubics
from .newtoncg import NewtonCG
from .gaussnewton import GaussNewton
from .levenbergmarquardt import LevenbergMarquardt
from .stochasticgradient import (
    ConstantSchedule,
    CosineDecay,
    PolynomialDecay,
    StepScheduleProtocol,
    StochasticGradient,
)
from .alternatinggradient import AlternatingGradient

__all__ = [
    "Minimize",
    "InfoEntry",
    "LineSearchStats",
    "LineSearchProtocol",
    "LineSearchResult",
    "LineSearchState",
    "ConstantStep",
    "BacktrackingArmijo",
    "AdaptiveArmijo",
    "StrongWolfe",
    "FiniteSum",
    "LeastSquares",
    "SteepestDescent",
    "ConjugateGradient",
    "TrustRegions",
    "BarzilaiBorwein",
    "LBFGS",
    "ParticleSwarm",
    "NelderMead",
    "AdaptiveRegularizationCubics",
    "NewtonCG",
    "GaussNewton",
    "LevenbergMarquardt",
    "StepScheduleProtocol",
    "ConstantSchedule",
    "PolynomialDecay",
    "CosineDecay",
    "StochasticGradient",
    "AlternatingGradient",
]
