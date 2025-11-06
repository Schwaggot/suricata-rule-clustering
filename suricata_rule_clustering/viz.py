"""
Visualization Module

This module provides interactive visualizations for clustering results
using Plotly and dimensionality reduction techniques (UMAP, t-SNE).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any
from sklearn.manifold import TSNE
from umap import UMAP


def reduce_dimensions(
    X: np.ndarray,
    method: str = 'umap',
    n_components: int = 2,
    random_state: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Reduce dimensionality of feature matrix for visualization.

    Args:
        X: Feature matrix
        method: Dimensionality reduction method ('umap' or 'tsne')
        n_components: Number of dimensions to reduce to (2 or 3)
        random_state: Random state for reproducibility
        **kwargs: Additional parameters for the reduction method

    Returns:
        Reduced feature matrix
    """
    print(f"Reducing {X.shape[1]} dimensions to {n_components} using {method.upper()}...")

    if method.lower() == 'umap':
        reducer = UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            metric=kwargs.get('metric', 'euclidean')
        )
    elif method.lower() == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=kwargs.get('perplexity', 30),
            n_iter=kwargs.get('n_iter', 1000)
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    X_reduced = reducer.fit_transform(X)
    print(f"Dimensionality reduction complete: {X.shape} -> {X_reduced.shape}")

    return X_reduced


def plot_clusters_2d(
    X_reduced: np.ndarray,
    labels: np.ndarray,
    df: Optional[pd.DataFrame] = None,
    title: str = "Rule Clusters (2D)",
    hover_data: Optional[list] = None
) -> go.Figure:
    """
    Create interactive 2D scatter plot of clusters.

    Args:
        X_reduced: 2D reduced feature matrix
        labels: Cluster labels
        df: Optional DataFrame with rule information for hover text
        title: Plot title
        hover_data: Optional list of column names to include in hover text

    Returns:
        Plotly Figure object
    """
    if X_reduced.shape[1] != 2:
        raise ValueError("X_reduced must have exactly 2 dimensions")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'cluster': labels.astype(str)
    })

    # Add hover data if DataFrame is provided
    if df is not None:
        if hover_data is None:
            hover_data = ['msg', 'classtype', 'priority']

        for col in hover_data:
            if col in df.columns:
                plot_df[col] = df[col].values

    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='cluster',
        hover_data=hover_data if df is not None else None,
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        width=1000,
        height=700,
        hovermode='closest'
    )

    return fig


def plot_clusters_3d(
    X_reduced: np.ndarray,
    labels: np.ndarray,
    df: Optional[pd.DataFrame] = None,
    title: str = "Rule Clusters (3D)",
    hover_data: Optional[list] = None
) -> go.Figure:
    """
    Create interactive 3D scatter plot of clusters.

    Args:
        X_reduced: 3D reduced feature matrix
        labels: Cluster labels
        df: Optional DataFrame with rule information for hover text
        title: Plot title
        hover_data: Optional list of column names to include in hover text

    Returns:
        Plotly Figure object
    """
    if X_reduced.shape[1] != 3:
        raise ValueError("X_reduced must have exactly 3 dimensions")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'z': X_reduced[:, 2],
        'cluster': labels.astype(str)
    })

    # Add hover data if DataFrame is provided
    if df is not None:
        if hover_data is None:
            hover_data = ['msg', 'classtype', 'priority']

        for col in hover_data:
            if col in df.columns:
                plot_df[col] = df[col].values

    # Create 3D scatter plot
    fig = px.scatter_3d(
        plot_df,
        x='x',
        y='y',
        z='z',
        color='cluster',
        hover_data=hover_data if df is not None else None,
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        width=1000,
        height=700,
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        )
    )

    return fig


def plot_cluster_sizes(labels: np.ndarray, title: str = "Cluster Sizes") -> go.Figure:
    """
    Create bar plot of cluster sizes.

    Args:
        labels: Cluster labels
        title: Plot title

    Returns:
        Plotly Figure object
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Create DataFrame
    cluster_df = pd.DataFrame({
        'cluster': unique_labels.astype(str),
        'count': counts
    })

    # Sort by count
    cluster_df = cluster_df.sort_values('count', ascending=False)

    # Create bar plot
    fig = px.bar(
        cluster_df,
        x='cluster',
        y='count',
        title=title,
        labels={'cluster': 'Cluster', 'count': 'Number of Rules'},
        color='count',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        width=800,
        height=500,
        showlegend=False
    )

    return fig


def plot_cluster_characteristics(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature: str = 'priority',
    plot_type: str = 'box'
) -> go.Figure:
    """
    Plot characteristics of each cluster.

    Args:
        df: DataFrame with rule information
        labels: Cluster labels
        feature: Feature to plot
        plot_type: Type of plot ('box', 'violin', 'histogram')

    Returns:
        Plotly Figure object
    """
    plot_df = df.copy()
    plot_df['cluster'] = labels.astype(str)

    if plot_type == 'box':
        fig = px.box(
            plot_df,
            x='cluster',
            y=feature,
            title=f"{feature.title()} Distribution by Cluster",
            labels={'cluster': 'Cluster', feature: feature.title()}
        )
    elif plot_type == 'violin':
        fig = px.violin(
            plot_df,
            x='cluster',
            y=feature,
            title=f"{feature.title()} Distribution by Cluster",
            labels={'cluster': 'Cluster', feature: feature.title()}
        )
    elif plot_type == 'histogram':
        fig = px.histogram(
            plot_df,
            x=feature,
            color='cluster',
            title=f"{feature.title()} Distribution by Cluster",
            barmode='overlay',
            opacity=0.7
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    fig.update_layout(width=1000, height=600)

    return fig


def plot_classtype_distribution(
    df: pd.DataFrame,
    labels: np.ndarray,
    top_n: int = 10
) -> go.Figure:
    """
    Plot distribution of classtypes across clusters.

    Args:
        df: DataFrame with rule information
        labels: Cluster labels
        top_n: Number of top classtypes to show

    Returns:
        Plotly Figure object
    """
    plot_df = df.copy()
    plot_df['cluster'] = labels.astype(str)

    # Get top N classtypes
    top_classtypes = df['classtype'].value_counts().head(top_n).index.tolist()
    plot_df = plot_df[plot_df['classtype'].isin(top_classtypes)]

    # Create grouped bar chart
    classtype_counts = plot_df.groupby(['cluster', 'classtype']).size().reset_index(name='count')

    fig = px.bar(
        classtype_counts,
        x='cluster',
        y='count',
        color='classtype',
        title=f"Top {top_n} Classtypes Distribution by Cluster",
        labels={'cluster': 'Cluster', 'count': 'Number of Rules'},
        barmode='stack'
    )

    fig.update_layout(width=1200, height=600)

    return fig


def create_cluster_dashboard(
    X: np.ndarray,
    X_reduced: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    metrics: Dict[str, Any]
) -> go.Figure:
    """
    Create a comprehensive dashboard with multiple visualizations.

    Args:
        X: Original feature matrix
        X_reduced: Reduced feature matrix (2D)
        labels: Cluster labels
        df: DataFrame with rule information
        metrics: Clustering evaluation metrics

    Returns:
        Plotly Figure object with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cluster Visualization (2D)',
            'Cluster Sizes',
            'Priority Distribution',
            'Evaluation Metrics'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'box'}, {'type': 'table'}]
        ]
    )

    # 1. Cluster visualization
    for label in np.unique(labels):
        mask = labels == label
        fig.add_trace(
            go.Scatter(
                x=X_reduced[mask, 0],
                y=X_reduced[mask, 1],
                mode='markers',
                name=f'Cluster {label}',
                marker=dict(size=3, opacity=0.6)
            ),
            row=1, col=1
        )

    # 2. Cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    fig.add_trace(
        go.Bar(x=unique_labels.astype(str), y=counts, showlegend=False),
        row=1, col=2
    )

    # 3. Priority distribution
    plot_df = df.copy()
    plot_df['cluster'] = labels.astype(str)
    for label in np.unique(labels):
        mask = labels == label
        priorities = df.loc[mask, 'priority'] if 'priority' in df.columns else []
        fig.add_trace(
            go.Box(y=priorities, name=f'Cluster {label}', showlegend=False),
            row=2, col=1
        )

    # 4. Metrics table
    metrics_data = [[k, str(v)] for k, v in metrics.items()]
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value']),
            cells=dict(values=list(zip(*metrics_data)))
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=1000,
        width=1400,
        title_text="Clustering Results Dashboard",
        showlegend=True
    )

    return fig


def save_figure(fig: go.Figure, filename: str, output_dir: str = "outputs"):
    """
    Save Plotly figure to HTML file.

    Args:
        fig: Plotly Figure object
        filename: Output filename (without extension)
        output_dir: Output directory
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{filename}.html"
    fig.write_html(str(filepath))
    print(f"Figure saved to {filepath}")
