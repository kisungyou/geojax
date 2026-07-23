# Optimization API

The optimization namespace separates problem descriptions, globalization
strategies, and solver classes. All public solvers return
`(solution, final_cost, history)`.

## Problem classes

```{eval-rst}
.. autoclass:: geojax.optimization.Minimize
   :members:

.. autoclass:: geojax.optimization.LeastSquares
   :members:

.. autoclass:: geojax.optimization.FiniteSum
   :members:
```

## Line searches

```{eval-rst}
.. autoclass:: geojax.optimization.LineSearchProtocol
   :members:

.. autoclass:: geojax.optimization.ConstantStep
   :members:

.. autoclass:: geojax.optimization.BacktrackingArmijo
   :members:

.. autoclass:: geojax.optimization.AdaptiveArmijo
   :members:

.. autoclass:: geojax.optimization.StrongWolfe
   :members:

.. autoclass:: geojax.optimization.LineSearchState
   :members:

.. autoclass:: geojax.optimization.LineSearchResult
   :members:
```

## Smooth solvers

```{eval-rst}
.. autoclass:: geojax.optimization.SteepestDescent
   :members:

.. autoclass:: geojax.optimization.ConjugateGradient
   :members:

.. autoclass:: geojax.optimization.BarzilaiBorwein
   :members:

.. autoclass:: geojax.optimization.LBFGS
   :members:

.. autoclass:: geojax.optimization.NewtonCG
   :members:

.. autoclass:: geojax.optimization.TrustRegions
   :members:

.. autoclass:: geojax.optimization.AdaptiveRegularizationCubics
   :members:
```

## Structured objectives

```{eval-rst}
.. autoclass:: geojax.optimization.GaussNewton
   :members:

.. autoclass:: geojax.optimization.LevenbergMarquardt
   :members:

.. autoclass:: geojax.optimization.StochasticGradient
   :members:

.. autoclass:: geojax.optimization.StepScheduleProtocol
   :members:

.. autoclass:: geojax.optimization.ConstantSchedule
   :members:

.. autoclass:: geojax.optimization.PolynomialDecay
   :members:

.. autoclass:: geojax.optimization.CosineDecay
   :members:

.. autoclass:: geojax.optimization.AlternatingGradient
   :members:
```

## Derivative-free solvers

```{eval-rst}
.. autoclass:: geojax.optimization.ParticleSwarm
   :members:

.. autoclass:: geojax.optimization.NelderMead
   :members:
```

## Iteration records

```{eval-rst}
.. autoclass:: geojax.optimization.InfoEntry
   :members:

.. autoclass:: geojax.optimization.LineSearchStats
   :members:
```
