"""
Unit tests for the clustering module.
"""

import pytest
import pandas as pd
import numpy as np
from suricata_rule_clustering import clustering


class TestRuleClusterer:
    """Tests for RuleClusterer class."""

    def test_initialization_kmeans(self):
        """Test KMeans clusterer initialization."""
        clusterer = clustering.RuleClusterer(algorithm='kmeans', n_clusters=5)

        assert clusterer.algorithm == 'kmeans'
        assert clusterer.params['n_clusters'] == 5
        assert clusterer.model is None
        assert clusterer.labels_ is None

    def test_initialization_dbscan(self):
        """Test DBSCAN clusterer initialization."""
        clusterer = clustering.RuleClusterer(
            algorithm='dbscan',
            eps=0.5,
            min_samples=5
        )

        assert clusterer.algorithm == 'dbscan'
        assert clusterer.params['eps'] == 0.5
        assert clusterer.params['min_samples'] == 5

    def test_initialization_hierarchical(self):
        """Test Hierarchical clusterer initialization."""
        clusterer = clustering.RuleClusterer(
            algorithm='hierarchical',
            n_clusters=8
        )

        assert clusterer.algorithm == 'hierarchical'
        assert clusterer.params['n_clusters'] == 8

    def test_initialization_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        clusterer = clustering.RuleClusterer(algorithm='invalid')

        with pytest.raises(ValueError):
            clusterer.fit(np.random.randn(100, 10))

    def test_fit_kmeans(self, sample_feature_matrix):
        """Test fitting KMeans clusterer."""
        clusterer = clustering.RuleClusterer(algorithm='kmeans', n_clusters=5)
        clusterer.fit(sample_feature_matrix)

        assert clusterer.model is not None
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(sample_feature_matrix)
        assert len(np.unique(clusterer.labels_)) == 5

    def test_fit_dbscan(self, sample_feature_matrix):
        """Test fitting DBSCAN clusterer."""
        clusterer = clustering.RuleClusterer(
            algorithm='dbscan',
            eps=0.5,
            min_samples=3
        )
        clusterer.fit(sample_feature_matrix)

        assert clusterer.model is not None
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(sample_feature_matrix)

    def test_fit_hierarchical(self, sample_feature_matrix):
        """Test fitting Hierarchical clusterer."""
        clusterer = clustering.RuleClusterer(
            algorithm='hierarchical',
            n_clusters=5
        )
        clusterer.fit(sample_feature_matrix)

        assert clusterer.model is not None
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(sample_feature_matrix)
        assert len(np.unique(clusterer.labels_)) == 5

    def test_predict_kmeans(self, sample_feature_matrix):
        """Test predicting with KMeans."""
        clusterer = clustering.RuleClusterer(algorithm='kmeans', n_clusters=5)
        clusterer.fit(sample_feature_matrix)

        # Predict on new data
        new_data = np.random.randn(10, sample_feature_matrix.shape[1])
        predictions = clusterer.predict(new_data)

        assert len(predictions) == 10
        assert all(0 <= p < 5 for p in predictions)

    def test_predict_non_kmeans_raises_error(self, sample_feature_matrix):
        """Test that predict raises error for non-KMeans algorithms."""
        clusterer = clustering.RuleClusterer(algorithm='dbscan')
        clusterer.fit(sample_feature_matrix)

        with pytest.raises(NotImplementedError):
            clusterer.predict(sample_feature_matrix)

    def test_get_cluster_summary(self, sample_feature_matrix, sample_parsed_df):
        """Test getting cluster summary."""
        clusterer = clustering.RuleClusterer(algorithm='kmeans', n_clusters=3)
        clusterer.fit(sample_feature_matrix[:len(sample_parsed_df)])

        summary = clusterer.get_cluster_summary(sample_parsed_df)

        assert isinstance(summary, pd.DataFrame)
        assert 'rule_count' in summary.columns
        assert len(summary) <= 3  # Up to 3 clusters


class TestEvaluateClustering:
    """Tests for evaluate_clustering function."""

    def test_evaluate_clustering_multiple_clusters(self, sample_feature_matrix, sample_labels):
        """Test clustering evaluation with multiple clusters."""
        metrics = clustering.evaluate_clustering(sample_feature_matrix, sample_labels)

        assert 'n_clusters' in metrics
        assert 'n_noise_points' in metrics
        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_score' in metrics
        assert 'calinski_harabasz_score' in metrics

        # Silhouette score should be between -1 and 1
        if metrics['silhouette_score'] is not None:
            assert -1 <= metrics['silhouette_score'] <= 1

    def test_evaluate_clustering_with_noise(self, sample_feature_matrix):
        """Test evaluation with noise points (label -1)."""
        labels = np.array([0, 1, 2, -1, -1, 0, 1, 2] * 12 + [0, 1, 2, -1])

        metrics = clustering.evaluate_clustering(sample_feature_matrix, labels)

        assert metrics['n_noise_points'] == 13
        assert metrics['n_clusters'] == 3

    def test_evaluate_clustering_single_cluster(self, sample_feature_matrix):
        """Test evaluation with single cluster."""
        labels = np.zeros(len(sample_feature_matrix), dtype=int)

        metrics = clustering.evaluate_clustering(sample_feature_matrix, labels)

        assert metrics['n_clusters'] == 1
        # Metrics should be None for single cluster
        assert metrics['silhouette_score'] is None


class TestFindOptimalK:
    """Tests for find_optimal_k function."""

    def test_find_optimal_k_elbow(self, sample_feature_matrix):
        """Test finding optimal k using elbow method."""
        optimal_k, scores = clustering.find_optimal_k(
            sample_feature_matrix,
            k_range=range(2, 6),
            method='elbow'
        )

        assert isinstance(optimal_k, int)
        assert 2 <= optimal_k < 6
        assert len(scores) == 4  # 2, 3, 4, 5
        assert all(isinstance(v, float) for v in scores.values())

    def test_find_optimal_k_silhouette(self, sample_feature_matrix):
        """Test finding optimal k using silhouette score."""
        optimal_k, scores = clustering.find_optimal_k(
            sample_feature_matrix,
            k_range=range(2, 6),
            method='silhouette'
        )

        assert isinstance(optimal_k, int)
        assert 2 <= optimal_k < 6
        assert len(scores) == 4
        # Silhouette scores should be between -1 and 1
        assert all(-1 <= v <= 1 for v in scores.values())


class TestPlotDendrogram:
    """Tests for plot_dendrogram function."""

    def test_plot_dendrogram(self, sample_feature_matrix):
        """Test creating dendrogram plot."""
        fig = clustering.plot_dendrogram(sample_feature_matrix, max_samples=50)

        assert fig is not None

    def test_plot_dendrogram_large_sample(self, sample_feature_matrix):
        """Test dendrogram with max_samples limit."""
        # Create larger dataset
        large_data = np.random.randn(2000, 20)

        fig = clustering.plot_dendrogram(large_data, max_samples=500)

        assert fig is not None


class TestCompareAlgorithms:
    """Tests for compare_algorithms function."""

    def test_compare_algorithms_default(self, sample_feature_matrix, sample_parsed_df):
        """Test comparing algorithms with default configurations."""
        comparison_df = clustering.compare_algorithms(
            sample_feature_matrix[:len(sample_parsed_df)],
            sample_parsed_df
        )

        assert isinstance(comparison_df, pd.DataFrame)
        assert 'algorithm' in comparison_df.columns
        assert 'n_clusters' in comparison_df.columns
        assert len(comparison_df) > 0

    def test_compare_algorithms_custom(self, sample_feature_matrix, sample_parsed_df):
        """Test comparing algorithms with custom configurations."""
        algorithms = [
            {'name': 'KMeans-3', 'algorithm': 'kmeans', 'params': {'n_clusters': 3}},
            {'name': 'KMeans-5', 'algorithm': 'kmeans', 'params': {'n_clusters': 5}},
        ]

        comparison_df = clustering.compare_algorithms(
            sample_feature_matrix[:len(sample_parsed_df)],
            sample_parsed_df,
            algorithms=algorithms
        )

        assert len(comparison_df) == 2
        assert comparison_df.iloc[0]['algorithm'] == 'KMeans-3'
        assert comparison_df.iloc[1]['algorithm'] == 'KMeans-5'


class TestIntegration:
    """Integration tests for clustering module."""

    def test_full_clustering_pipeline(self, sample_feature_matrix, sample_parsed_df):
        """Test complete clustering workflow."""
        # Fit clusterer
        clusterer = clustering.RuleClusterer(algorithm='kmeans', n_clusters=3)
        clusterer.fit(sample_feature_matrix[:len(sample_parsed_df)])

        # Evaluate
        metrics = clustering.evaluate_clustering(
            sample_feature_matrix[:len(sample_parsed_df)],
            clusterer.labels_
        )

        # Get summary
        summary = clusterer.get_cluster_summary(sample_parsed_df)

        # Verify all steps worked
        assert len(clusterer.labels_) == len(sample_parsed_df)
        assert 'n_clusters' in metrics
        assert isinstance(summary, pd.DataFrame)

    def test_multiple_algorithms_comparison(self, sample_feature_matrix, sample_parsed_df):
        """Test comparing multiple clustering approaches."""
        # Try KMeans
        kmeans = clustering.RuleClusterer(algorithm='kmeans', n_clusters=3)
        kmeans.fit(sample_feature_matrix[:len(sample_parsed_df)])

        # Try Hierarchical
        hierarchical = clustering.RuleClusterer(algorithm='hierarchical', n_clusters=3)
        hierarchical.fit(sample_feature_matrix[:len(sample_parsed_df)])

        # Evaluate both
        kmeans_metrics = clustering.evaluate_clustering(
            sample_feature_matrix[:len(sample_parsed_df)],
            kmeans.labels_
        )
        hierarchical_metrics = clustering.evaluate_clustering(
            sample_feature_matrix[:len(sample_parsed_df)],
            hierarchical.labels_
        )

        # Both should have valid metrics
        assert kmeans_metrics['n_clusters'] == 3
        assert hierarchical_metrics['n_clusters'] == 3

    @pytest.mark.slow
    def test_large_dataset_clustering(self):
        """Test clustering on larger dataset."""
        # Create larger synthetic dataset
        np.random.seed(42)
        X = np.random.randn(1000, 50)

        clusterer = clustering.RuleClusterer(algorithm='kmeans', n_clusters=10)
        clusterer.fit(X)

        assert len(clusterer.labels_) == 1000
        assert len(np.unique(clusterer.labels_)) == 10


class TestEdgeCases:
    """Tests for edge cases."""

    def test_clustering_small_dataset(self):
        """Test clustering with very small dataset."""
        X = np.random.randn(5, 10)

        clusterer = clustering.RuleClusterer(algorithm='kmeans', n_clusters=2)
        clusterer.fit(X)

        assert len(clusterer.labels_) == 5

    def test_clustering_high_dimensional(self):
        """Test clustering with high-dimensional data."""
        X = np.random.randn(50, 500)

        clusterer = clustering.RuleClusterer(algorithm='kmeans', n_clusters=5)
        clusterer.fit(X)

        assert len(clusterer.labels_) == 50

    def test_dbscan_all_noise(self):
        """Test DBSCAN when all points are noise."""
        # Create well-separated points
        X = np.random.randn(20, 10) * 100

        clusterer = clustering.RuleClusterer(
            algorithm='dbscan',
            eps=0.1,  # Very small eps
            min_samples=10  # Large min_samples
        )
        clusterer.fit(X)

        # Most or all points should be noise (-1)
        assert -1 in clusterer.labels_
