# Learning API

The learning namespace provides validated data adaptation, differentiable
geometry primitives, supervised and semi-supervised prediction, intrinsic
statistics, robust and scalable summaries, clustering, inference, transport,
dimension reduction, and metric learning. High-level algorithms operate on
dense distance matrices unless their documentation says otherwise.

## Data and capability contracts

```{eval-rst}
.. currentmodule:: geojax.learning

.. autoclass:: ManifoldData

.. autoclass:: DataValidationReport

.. autoclass:: ManifoldDataAdapterProtocol

.. autoclass:: EquivariantEmbeddingProtocol

.. autoexception:: LearningCapabilityError

.. autofunction:: as_manifold_data

.. autofunction:: check_manifold_data

.. autofunction:: register_manifold_data_adapter
```

## Geometric primitives

```{eval-rst}
.. autofunction:: pairwise_distances

.. autofunction:: geodesic_interpolation

.. autofunction:: tangent_space_map

.. autofunction:: nearest_neighbors

.. autoclass:: NeighborsResult
```

## Statistics and scalar-response regression

```{eval-rst}
.. autofunction:: frechet_mean

.. autofunction:: frechet_median

.. autofunction:: minimum_enclosing_ball

.. autofunction:: kernel_regression

.. autofunction:: select_kernel_bandwidth

.. autoclass:: FrechetMeanResult

.. autoclass:: FrechetMedianResult

.. autoclass:: EnclosingBallResult

.. autoclass:: KernelRegressionModel

.. autoclass:: KernelCVResult
```

## Supervised classification

```{eval-rst}
.. autofunction:: nearest_centroid_classifier

.. autofunction:: knn_classifier

.. autofunction:: tangent_space_logistic_regression

.. autofunction:: tangent_space_discriminant_analysis

.. autoclass:: NearestCentroidModel

.. autoclass:: KNearestNeighborsModel

.. autoclass:: TangentFeatureMap

.. autoclass:: TangentSpaceClassifierModel
```

## Manifold-valued response regression

```{eval-rst}
.. autofunction:: geodesic_regression

.. autofunction:: local_polynomial_regression

.. autoclass:: GeodesicRegressionModel

.. autoclass:: LocalPolynomialRegressionModel
```

## Inference

```{eval-rst}
.. autofunction:: frechet_anova

.. autofunction:: biswas_ghosh_two_sample_test

.. autofunction:: wasserstein_two_sample_test

.. autofunction:: bootstrap_frechet_mean

.. autofunction:: energy_two_sample_test

.. autofunction:: kernel_mmd_two_sample_test

.. autofunction:: paired_frechet_test

.. autoclass:: HypothesisTestResult

.. autoclass:: BootstrapResult
```

## Clustering

```{eval-rst}
.. autofunction:: kmeans

.. autofunction:: lightweight_coreset

.. autofunction:: kmedoids

.. autofunction:: agglomerative_clustering

.. autofunction:: spectral_clustering

.. autofunction:: mean_shift

.. autofunction:: competitive_quantization

.. autoclass:: ClusteringResult

.. autoclass:: HierarchicalClusteringResult

.. autoclass:: CoresetResult
```

## Scalable summaries

```{eval-rst}
.. autofunction:: streaming_frechet_mean

.. autofunction:: minibatch_frechet_mean

.. autofunction:: minibatch_kmeans
```

## Barycentric coding and dictionaries

```{eval-rst}
.. autofunction:: geodesic_barycentric_coding

.. autofunction:: manifold_dictionary_learning

.. autoclass:: BarycentricCodingResult

.. autoclass:: DictionaryLearningResult
```

## Robust analysis

```{eval-rst}
.. autofunction:: trimmed_frechet_mean

.. autofunction:: geodesic_m_estimator

.. autofunction:: geodesic_spatial_depth

.. autofunction:: metric_distance_ranks

.. autoclass:: RobustLocationResult

.. autoclass:: MetricRanksResult
```

## Semi-supervised learning

```{eval-rst}
.. autofunction:: label_propagation

.. autofunction:: manifold_regularized_regression

.. autoclass:: SemiSupervisedResult
```

## Dimension reduction

```{eval-rst}
.. autofunction:: classical_mds

.. autofunction:: principal_geodesic_analysis

.. autofunction:: kernel_pca

.. autofunction:: isomap

.. autofunction:: sammon_mapping

.. autofunction:: tsne

.. autofunction:: phate

.. autoclass:: EmbeddingResult
```

## Transport and metric learning

```{eval-rst}
.. autofunction:: empirical_wasserstein_distance

.. autofunction:: sinkhorn_divergence

.. autoclass:: TransportResult

.. autofunction:: riemannian_metric_learning

.. autoclass:: MetricLearningModel
```
