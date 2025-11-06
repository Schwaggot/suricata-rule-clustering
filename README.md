# Suricata Rule Clustering

A machine learning project for clustering Suricata IDS/IPS rules to identify groups of similar rules using various
clustering algorithms.

## Overview

This project uses machine learning techniques to analyze and cluster Suricata rules based on their characteristics. It
helps security analysts:

- Identify patterns in large rule sets
- Find groups of similar rules
- Understand rule relationships
- Optimize rule management

## Features

- **Rule Parsing**: Automated parsing of Suricata rule files using `suricata-rule-parser`
- **Feature Engineering**: Extraction of meaningful features from rules including:
    - Basic features (action, protocol, ports, direction)
    - Option-based features (content, pcre, flow patterns)
    - Metadata features (classtype, priority, message characteristics)
    - TF-IDF features from rule messages
- **Multiple Clustering Algorithms**:
    - K-Means
    - DBSCAN
    - Hierarchical/Agglomerative Clustering
- **Interactive Visualizations**:
    - 2D and 3D UMAP projections
    - t-SNE dimensionality reduction
    - Interactive Plotly dashboards
    - Cluster characteristic analysis

## Project Structure

```
suricata-rule-clustering/
├── suricata_rule_clustering/    # Main package
│   ├── __init__.py
│   ├── parser.py                # Rule parsing and loading
│   ├── features.py              # Feature extraction and engineering
│   ├── clustering.py            # Clustering algorithms
│   └── viz.py                   # Visualization utilities
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_visualization.ipynb
├── data/                        # Processed data (generated, ignored by git)
├── outputs/                     # Visualizations and results (generated)
├── rules/                       # Suricata rule files (ignored by git)
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/suricata-rule-clustering.git
cd suricata-rule-clustering
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up pre-commit hooks (to automatically strip notebook outputs):

```bash
pre-commit install
```

5. Place your Suricata rule files in the `rules/` directory

## Usage

### Jupyter Notebooks (Recommended)

The project is designed to be used through Jupyter notebooks for interactive exploration:

1. Start Jupyter:

```bash
jupyter notebook
```

2. Follow the notebooks in order:
    - `01_data_exploration.ipynb` - Explore and understand your rule dataset
    - `02_feature_engineering.ipynb` - Extract and engineer features
    - `03_clustering.ipynb` - Apply clustering algorithms
    - `04_visualization.ipynb` - Create interactive visualizations

### Python API

You can also use the package programmatically:

```python
from suricata_rule_clustering import parser, features, clustering, viz

# Parse rules
df = parser.parse_all_rules(rules_dir='rules')
parser.save_parsed_rules(df, 'data/parsed_rules.pkl')

# Extract features
extractor = features.RuleFeatureExtractor()
X = extractor.create_feature_matrix(df, include_tfidf=True)

# Apply clustering
clusterer = clustering.RuleClusterer(algorithm='kmeans', n_clusters=8)
clusterer.fit(X)

# Visualize results
X_reduced = viz.reduce_dimensions(X, method='umap', n_components=2)
fig = viz.plot_clusters_2d(X_reduced, clusterer.labels_, df=df)
fig.show()
```

## Clustering Algorithms

### K-Means

- Fast and efficient
- Requires specifying number of clusters
- Works well with spherical clusters
- Good for initial exploration

### DBSCAN

- Density-based clustering
- Automatically detects outliers
- Finds clusters of arbitrary shapes
- No need to specify number of clusters

### Hierarchical

- Creates hierarchical cluster relationships
- Useful for understanding rule families
- Can be visualized with dendrograms
- Flexible number of clusters

## Output

The project generates:

- **Parsed rule DataFrames**: Structured data from raw rules
- **Feature matrices**: Numerical representations for clustering
- **Cluster labels**: Rule-to-cluster assignments
- **Interactive HTML visualizations**: Explorable in any browser
- **Cluster analysis reports**: CSV files with representative rules

All outputs are saved in the `data/` and `outputs/` directories.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
