"""
Clustering Module

This module implements various clustering algorithms for grouping
similar Suricata rules together.
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm


class RuleClusterer:
    """Wrapper class for different clustering algorithms."""

    def __init__(self, algorithm: str = 'kmeans', **kwargs):
        """
        Initialize clusterer with specified algorithm.

        Args:
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm.lower()
        self.model = None
        self.labels_ = None
        self.params = kwargs

    def fit(self, X: np.ndarray) -> 'RuleClusterer':
        """
        Fit the clustering model.

        Args:
            X: Feature matrix

        Returns:
            Self
        """
        if self.algorithm == 'kmeans':
            self.model = self._fit_kmeans(X)
        elif self.algorithm == 'dbscan':
            self.model = self._fit_dbscan(X)
        elif self.algorithm == 'hierarchical':
            self.model = self._fit_hierarchical(X)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        self.labels_ = self.model.labels_
        return self

    def _fit_kmeans(self, X: np.ndarray) -> KMeans:
        """Fit K-Means clustering."""
        n_clusters = self.params.get('n_clusters', 8)
        random_state = self.params.get('random_state', 42)
        max_iter = self.params.get('max_iter', 300)

        print(f"Fitting K-Means with {n_clusters} clusters...")
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=10
        )
        model.fit(X)
        print(f"K-Means converged in {model.n_iter_} iterations")
        return model

    def _fit_dbscan(self, X: np.ndarray) -> DBSCAN:
        """Fit DBSCAN clustering."""
        eps = self.params.get('eps', 0.5)
        min_samples = self.params.get('min_samples', 5)
        metric = self.params.get('metric', 'euclidean')

        print(f"Fitting DBSCAN with eps={eps}, min_samples={min_samples}...")
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            n_jobs=-1
        )
        model.fit(X)

        n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
        n_noise = list(model.labels_).count(-1)
        print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        return model

    def _fit_hierarchical(self, X: np.ndarray) -> AgglomerativeClustering:
        """Fit Hierarchical/Agglomerative clustering."""
        n_clusters = self.params.get('n_clusters', 8)
        linkage_type = self.params.get('linkage', 'ward')
        distance_threshold = self.params.get('distance_threshold', None)

        # If distance_threshold is set, n_clusters must be None
        if distance_threshold is not None:
            n_clusters = None

        print(f"Fitting Hierarchical clustering with linkage={linkage_type}...")
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_type,
            distance_threshold=distance_threshold
        )
        model.fit(X)

        actual_clusters = len(set(model.labels_))
        print(f"Hierarchical clustering found {actual_clusters} clusters")
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Note: Only works for K-Means. Other algorithms don't support prediction.

        Args:
            X: Feature matrix

        Returns:
            Cluster labels
        """
        if self.algorithm == 'kmeans':
            return self.model.predict(X)
        else:
            raise NotImplementedError(
                f"{self.algorithm} does not support prediction on new data"
            )

    def get_cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.

        Args:
            df: Original DataFrame with parsed rules

        Returns:
            DataFrame with cluster summaries
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before getting cluster summary")

        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = self.labels_

        summary = df_with_clusters.groupby('cluster').agg({
            'sid': 'count',
            'classtype': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',
            'priority': 'mean',
            'msg': lambda x: x.iloc[0] if len(x) > 0 else ''  # Sample message
        }).rename(columns={'sid': 'rule_count'})

        return summary


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering quality using multiple metrics.

    Args:
        X: Feature matrix
        labels: Cluster labels

    Returns:
        Dictionary with evaluation metrics
    """
    # Filter out noise points (label -1) for metrics that don't support them
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    n_clusters = len(set(labels_filtered))

    metrics = {
        'n_clusters': n_clusters,
        'n_noise_points': np.sum(labels == -1)
    }

    # Only compute metrics if we have more than 1 cluster
    if n_clusters > 1:
        try:
            print("Calculating silhouette score...")
            metrics['silhouette_score'] = silhouette_score(X_filtered, labels_filtered, sample_size=10000)
        except:
            metrics['silhouette_score'] = None

        try:
            print("Calculating davies bouldin score...")
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_filtered, labels_filtered)
        except:
            metrics['davies_bouldin_score'] = None

        try:
            print("Calculating calinski harabasz score...")
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_filtered, labels_filtered)
        except:
            metrics['calinski_harabasz_score'] = None

    return metrics


def find_optimal_k(
        X: np.ndarray,
        k_range: range = range(2, 11),
        method: str = 'elbow'
) -> Tuple[int, Dict[int, float]]:
    """
    Find optimal number of clusters for K-Means.

    Args:
        X: Feature matrix
        k_range: Range of k values to test
        method: Method to use ('elbow' for inertia, 'silhouette' for silhouette score)

    Returns:
        Tuple of (optimal_k, scores_dict)
    """
    scores = {}

    print(f"Testing K-Means with k in {k_range}...")
    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)

        if method == 'elbow':
            scores[k] = kmeans.inertia_
        elif method == 'silhouette':
            scores[k] = silhouette_score(X, kmeans.labels_, sample_size=10000)

    # Find optimal k
    if method == 'elbow':
        # For elbow method, we'd need to calculate the elbow point
        # For now, return the k with minimum inertia (though this isn't ideal)
        optimal_k = min(scores, key=scores.get)
    elif method == 'silhouette':
        # Higher silhouette score is better
        optimal_k = max(scores, key=scores.get)

    print(f"Optimal k: {optimal_k}")
    return optimal_k, scores


def plot_dendrogram(X: np.ndarray, max_samples: int = 1000, **kwargs):
    """
    Plot dendrogram for hierarchical clustering.

    Args:
        X: Feature matrix
        max_samples: Maximum number of samples to use (for performance)
        **kwargs: Additional arguments for dendrogram plotting
    """
    # Sample data if too large
    if X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    # Compute linkage matrix
    Z = linkage(X_sample, method='ward')

    # Plot dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(Z, **kwargs)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()

    return plt.gcf()


def compare_algorithms(
        X: np.ndarray,
        df: pd.DataFrame,
        algorithms: list = None
) -> pd.DataFrame:
    """
    Compare different clustering algorithms.

    Args:
        X: Feature matrix
        df: Original DataFrame with parsed rules
        algorithms: List of algorithm configurations to compare

    Returns:
        DataFrame comparing algorithm performance
    """
    if algorithms is None:
        algorithms = [
            {'name': 'K-Means (k=5)', 'algorithm': 'kmeans', 'params': {'n_clusters': 5}},
            {'name': 'K-Means (k=8)', 'algorithm': 'kmeans', 'params': {'n_clusters': 8}},
            {'name': 'K-Means (k=10)', 'algorithm': 'kmeans', 'params': {'n_clusters': 10}},
            {'name': 'DBSCAN (eps=0.5)', 'algorithm': 'dbscan', 'params': {'eps': 0.5, 'min_samples': 5}},
            {'name': 'DBSCAN (eps=1.0)', 'algorithm': 'dbscan', 'params': {'eps': 1.0, 'min_samples': 5}},
            {'name': 'Hierarchical (k=8)', 'algorithm': 'hierarchical', 'params': {'n_clusters': 8}},
        ]

    results = []

    for config in algorithms:
        print(f"\nTesting {config['name']}...")
        clusterer = RuleClusterer(
            algorithm=config['algorithm'],
            **config['params']
        )
        clusterer.fit(X)

        metrics = evaluate_clustering(X, clusterer.labels_)
        metrics['algorithm'] = config['name']

        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    return comparison_df
