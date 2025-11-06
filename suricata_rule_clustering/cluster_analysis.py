"""
Cluster Analysis Module

This module provides comprehensive analysis and description of clustering results
for Suricata IDS rules, including automatic labeling, feature importance,
and representative rule selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from sklearn.metrics import pairwise_distances
from collections import Counter


class ClusterDescriptor:
    """Generate comprehensive descriptions for rule clusters."""

    def __init__(self, feature_extractor=None):
        """
        Initialize cluster descriptor.

        Args:
            feature_extractor: RuleFeatureExtractor instance with fitted transformers
        """
        self.feature_extractor = feature_extractor

    def describe_cluster(
        self,
        cluster_id: int,
        X: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame,
        cluster_center: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive description for a single cluster.

        Args:
            cluster_id: Cluster ID to describe
            X: Feature matrix
            labels: Cluster labels
            df: DataFrame with parsed rules
            cluster_center: Optional cluster centroid (for K-Means)

        Returns:
            Dictionary with cluster description
        """
        # Get cluster mask
        mask = labels == cluster_id
        cluster_rules = df[mask]
        cluster_features = X[mask]

        if len(cluster_rules) == 0:
            return self._empty_cluster_description(cluster_id)

        # Calculate statistics
        summary = self._calculate_summary_stats(cluster_id, cluster_rules, X, labels)

        # Get dominant characteristics
        characteristics = self._get_dominant_characteristics(cluster_rules)

        # Extract top keywords/terms
        top_terms = self._extract_top_terms(cluster_features, cluster_rules)

        # Find representative rules
        representative_rules = self._get_representative_rules(
            cluster_features, cluster_rules, cluster_center
        )

        # Generate automatic label
        auto_label = self._generate_cluster_label(characteristics, top_terms)

        return {
            'cluster_id': cluster_id,
            'label': auto_label,
            'summary': summary,
            'characteristics': characteristics,
            'top_terms': top_terms,
            'representative_rules': representative_rules
        }

    def describe_all_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame,
        cluster_centers: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate descriptions for all clusters.

        Args:
            X: Feature matrix
            labels: Cluster labels
            df: DataFrame with parsed rules
            cluster_centers: Optional cluster centroids (for K-Means)

        Returns:
            Dictionary mapping cluster_id to description
        """
        unique_labels = np.unique(labels)
        descriptions = {}

        print(f"Generating descriptions for {len(unique_labels)} clusters...")

        for cluster_id in unique_labels:
            center = cluster_centers[cluster_id] if cluster_centers is not None else None
            descriptions[int(cluster_id)] = self.describe_cluster(
                int(cluster_id), X, labels, df, center
            )

        return descriptions

    def _calculate_summary_stats(
        self,
        cluster_id: int,
        cluster_rules: pd.DataFrame,
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate statistical summary for cluster."""
        total_rules = len(labels)
        cluster_size = len(cluster_rules)

        return {
            'size': cluster_size,
            'percentage': (cluster_size / total_rules) * 100,
            'avg_priority': cluster_rules['priority'].mean() if 'priority' in cluster_rules.columns else None,
            'median_priority': cluster_rules['priority'].median() if 'priority' in cluster_rules.columns else None
        }

    def _get_dominant_characteristics(self, cluster_rules: pd.DataFrame) -> Dict[str, Any]:
        """Extract dominant characteristics from cluster rules."""
        characteristics = {}

        # Top classtypes
        if 'classtype' in cluster_rules.columns:
            classtype_counts = cluster_rules['classtype'].value_counts()
            total = len(cluster_rules)
            characteristics['classtypes'] = [
                {
                    'name': ct,
                    'count': int(count),
                    'percentage': (count / total) * 100
                }
                for ct, count in classtype_counts.head(3).items()
            ]

        # Protocol distribution
        if 'protocol' in cluster_rules.columns:
            protocol_counts = cluster_rules['protocol'].value_counts()
            total = len(cluster_rules)
            characteristics['protocols'] = [
                {
                    'name': proto,
                    'count': int(count),
                    'percentage': (count / total) * 100
                }
                for proto, count in protocol_counts.head(5).items()
            ]

        # Action distribution
        if 'action' in cluster_rules.columns:
            action_counts = cluster_rules['action'].value_counts()
            total = len(cluster_rules)
            characteristics['actions'] = [
                {
                    'name': action,
                    'count': int(count),
                    'percentage': (count / total) * 100
                }
                for action, count in action_counts.head(3).items()
            ]

        # Port patterns
        if 'dst_port' in cluster_rules.columns:
            specific_ports = cluster_rules[cluster_rules['dst_port'] != 'any']['dst_port']
            if len(specific_ports) > 0:
                port_counts = specific_ports.value_counts()
                characteristics['common_ports'] = [
                    {'port': str(port), 'count': int(count)}
                    for port, count in port_counts.head(5).items()
                ]

        return characteristics

    def _extract_top_terms(
        self,
        cluster_features: np.ndarray,
        cluster_rules: pd.DataFrame,
        top_n: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Extract top TF-IDF terms for the cluster.

        Args:
            cluster_features: Feature matrix for cluster
            cluster_rules: DataFrame with cluster rules
            top_n: Number of top terms to return

        Returns:
            List of top terms with weights
        """
        if self.feature_extractor is None or self.feature_extractor.tfidf_vectorizer is None:
            return self._extract_terms_from_messages(cluster_rules, top_n)

        # Get TF-IDF feature names
        feature_names = self.feature_extractor.tfidf_vectorizer.get_feature_names_out()

        # Calculate the number of numeric features before TF-IDF
        # We need to extract only the TF-IDF portion of the feature matrix
        tfidf_start_idx = cluster_features.shape[1] - len(feature_names)

        if tfidf_start_idx < 0:
            return self._extract_terms_from_messages(cluster_rules, top_n)

        # Extract TF-IDF features
        tfidf_features = cluster_features[:, tfidf_start_idx:]

        # Calculate mean TF-IDF scores for this cluster
        mean_tfidf = tfidf_features.mean(axis=0)

        # Get top N terms
        top_indices = np.argsort(mean_tfidf)[-top_n:][::-1]

        terms = [
            {
                'term': feature_names[idx],
                'weight': float(mean_tfidf[idx])
            }
            for idx in top_indices if mean_tfidf[idx] > 0
        ]

        return terms

    def _extract_terms_from_messages(
        self,
        cluster_rules: pd.DataFrame,
        top_n: int = 15
    ) -> List[Dict[str, Any]]:
        """Fallback: Extract terms directly from messages."""
        msg_col = 'msg' if 'msg' in cluster_rules.columns else 'message'
        if msg_col not in cluster_rules.columns:
            return []

        # Simple word frequency analysis
        all_words = []
        for msg in cluster_rules[msg_col].dropna():
            words = str(msg).lower().split()
            all_words.extend([w for w in words if len(w) > 3])

        word_counts = Counter(all_words)
        return [
            {'term': word, 'weight': count}
            for word, count in word_counts.most_common(top_n)
        ]

    def _get_representative_rules(
        self,
        cluster_features: np.ndarray,
        cluster_rules: pd.DataFrame,
        cluster_center: Optional[np.ndarray] = None,
        n_representatives: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find representative rules for the cluster.

        Args:
            cluster_features: Feature matrix for cluster
            cluster_rules: DataFrame with cluster rules
            cluster_center: Optional cluster centroid
            n_representatives: Number of representative rules to return

        Returns:
            List of representative rules with metadata
        """
        if len(cluster_rules) == 0:
            return []

        # Calculate cluster center if not provided
        if cluster_center is None:
            cluster_center = cluster_features.mean(axis=0)

        # Calculate distances to center
        distances = pairwise_distances(
            cluster_features,
            cluster_center.reshape(1, -1),
            metric='euclidean'
        ).flatten()

        # Get medoid (closest to center)
        medoid_idx = np.argmin(distances)

        # Get diverse samples (at different distance percentiles)
        representatives = []

        # Add medoid
        medoid_rule = cluster_rules.iloc[medoid_idx]
        representatives.append({
            'type': 'medoid',
            'distance': float(distances[medoid_idx]),
            'message': medoid_rule.get('msg', medoid_rule.get('message', 'N/A')),
            'classtype': medoid_rule.get('classtype', 'N/A'),
            'protocol': medoid_rule.get('protocol', 'N/A'),
            'sid': medoid_rule.get('sid', 'N/A'),
            'raw_rule': medoid_rule.get('raw_rule', 'N/A')[:200] + '...' if len(str(medoid_rule.get('raw_rule', ''))) > 200 else medoid_rule.get('raw_rule', 'N/A')
        })

        # Add diverse samples if cluster is large enough
        if len(cluster_rules) >= n_representatives:
            percentiles = [33, 66] if n_representatives == 3 else [25, 50, 75][:n_representatives-1]
            for percentile in percentiles:
                target_distance = np.percentile(distances, percentile)
                diverse_idx = np.argmin(np.abs(distances - target_distance))

                # Skip if too close to medoid
                if diverse_idx != medoid_idx:
                    diverse_rule = cluster_rules.iloc[diverse_idx]
                    representatives.append({
                        'type': f'diverse (p{percentile})',
                        'distance': float(distances[diverse_idx]),
                        'message': diverse_rule.get('msg', diverse_rule.get('message', 'N/A')),
                        'classtype': diverse_rule.get('classtype', 'N/A'),
                        'protocol': diverse_rule.get('protocol', 'N/A'),
                        'sid': diverse_rule.get('sid', 'N/A'),
                        'raw_rule': diverse_rule.get('raw_rule', 'N/A')[:200] + '...' if len(str(diverse_rule.get('raw_rule', ''))) > 200 else diverse_rule.get('raw_rule', 'N/A')
                    })

        return representatives

    def _generate_cluster_label(
        self,
        characteristics: Dict[str, Any],
        top_terms: List[Dict[str, Any]]
    ) -> str:
        """
        Generate automatic label for cluster based on characteristics.

        Args:
            characteristics: Dominant characteristics
            top_terms: Top TF-IDF terms

        Returns:
            Human-readable cluster label
        """
        label_parts = []

        # Add top classtype
        if 'classtypes' in characteristics and len(characteristics['classtypes']) > 0:
            top_classtype = characteristics['classtypes'][0]['name']
            # Clean up classtype (remove hyphens, capitalize)
            clean_classtype = top_classtype.replace('-', ' ').title()
            label_parts.append(clean_classtype)

        # Add protocol if dominant
        if 'protocols' in characteristics and len(characteristics['protocols']) > 0:
            top_protocol = characteristics['protocols'][0]
            if top_protocol['percentage'] > 70:  # Only if dominant
                label_parts.append(top_protocol['name'].upper())

        # Add key terms (top 2-3)
        if len(top_terms) > 0:
            key_terms = [t['term'].title() for t in top_terms[:2]]
            label_parts.extend(key_terms)

        # Construct label
        if len(label_parts) == 0:
            return "Mixed Rules"

        return " - ".join(label_parts[:3])  # Max 3 parts to keep label concise

    def _empty_cluster_description(self, cluster_id: int) -> Dict[str, Any]:
        """Return empty description for clusters with no rules."""
        return {
            'cluster_id': cluster_id,
            'label': f"Empty Cluster {cluster_id}",
            'summary': {'size': 0, 'percentage': 0.0},
            'characteristics': {},
            'top_terms': [],
            'representative_rules': []
        }

    def export_to_markdown(
        self,
        descriptions: Dict[int, Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        Export cluster descriptions to markdown file.

        Args:
            descriptions: Dictionary of cluster descriptions
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("# Cluster Analysis Report\n\n")
            f.write(f"Total Clusters: {len(descriptions)}\n\n")
            f.write("---\n\n")

            for cluster_id in sorted(descriptions.keys()):
                desc = descriptions[cluster_id]
                self._write_cluster_markdown(f, desc)

        print(f"Markdown report saved to: {output_file}")

    def _write_cluster_markdown(self, file, desc: Dict[str, Any]) -> None:
        """Write single cluster description in markdown format."""
        file.write(f"## Cluster {desc['cluster_id']}: {desc['label']}\n\n")

        # Summary
        summary = desc['summary']
        file.write(f"**Size**: {summary['size']} rules ({summary['percentage']:.2f}%)\n\n")

        if summary.get('avg_priority'):
            file.write(f"**Average Priority**: {summary['avg_priority']:.2f}\n\n")

        # Characteristics
        chars = desc['characteristics']

        if 'classtypes' in chars:
            file.write("**Top Classtypes**:\n")
            for ct in chars['classtypes']:
                file.write(f"- {ct['name']}: {ct['percentage']:.1f}%\n")
            file.write("\n")

        if 'protocols' in chars:
            file.write("**Protocols**: ")
            proto_strs = [f"{p['name']} ({p['percentage']:.1f}%)" for p in chars['protocols']]
            file.write(", ".join(proto_strs))
            file.write("\n\n")

        # Top terms
        if desc['top_terms']:
            file.write("**Key Terms**: ")
            terms = [t['term'] for t in desc['top_terms'][:10]]
            file.write(", ".join(terms))
            file.write("\n\n")

        # Representative rules
        if desc['representative_rules']:
            file.write("**Representative Rules**:\n\n")
            for i, rule in enumerate(desc['representative_rules'], 1):
                file.write(f"{i}. [{rule['type']}] {rule['message']}\n")
                file.write(f"   - Classtype: {rule['classtype']}, Protocol: {rule['protocol']}, SID: {rule['sid']}\n")
            file.write("\n")

        file.write("---\n\n")

    def export_to_csv(
        self,
        descriptions: Dict[int, Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        Export cluster descriptions to CSV file.

        Args:
            descriptions: Dictionary of cluster descriptions
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for cluster_id in sorted(descriptions.keys()):
            desc = descriptions[cluster_id]
            summary = desc['summary']
            chars = desc['characteristics']

            # Safely get top classtype
            classtypes = chars.get('classtypes', [])
            top_classtype = classtypes[0].get('name', 'N/A') if classtypes else 'N/A'

            # Safely get top protocol
            protocols = chars.get('protocols', [])
            top_protocol = protocols[0].get('name', 'N/A') if protocols else 'N/A'

            # Get top terms
            top_terms = ", ".join([t['term'] for t in desc['top_terms'][:10]]) if desc['top_terms'] else 'N/A'
            sample_msg = desc['representative_rules'][0]['message'] if desc['representative_rules'] else 'N/A'

            rows.append({
                'cluster_id': cluster_id,
                'label': desc['label'],
                'size': summary['size'],
                'percentage': f"{summary['percentage']:.2f}%",
                'top_classtype': top_classtype,
                'top_protocol': top_protocol,
                'avg_priority': f"{summary.get('avg_priority', 0):.2f}",
                'key_terms': top_terms,
                'sample_message': sample_msg
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"CSV report saved to: {output_file}")

    def export_to_json(
        self,
        descriptions: Dict[int, Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        Export cluster descriptions to JSON file.

        Args:
            descriptions: Dictionary of cluster descriptions
            output_path: Output file path
        """
        import json

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable_descriptions = convert_to_json_serializable(descriptions)

        with open(output_file, 'w') as f:
            json.dump(serializable_descriptions, f, indent=2)

        print(f"JSON report saved to: {output_file}")


def format_cluster_description(desc: Dict[str, Any]) -> str:
    """
    Format cluster description as readable string.

    Args:
        desc: Cluster description dictionary

    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"\nCLUSTER {desc['cluster_id']}: {desc['label']}")
    lines.append("=" * (len(lines[0]) - 1))

    summary = desc['summary']
    lines.append(f"Size: {summary['size']} rules ({summary['percentage']:.2f}% of dataset)")

    if summary.get('avg_priority'):
        lines.append(f"Priority: {summary['avg_priority']:.2f} average")

    chars = desc['characteristics']

    if 'classtypes' in chars and chars['classtypes']:
        lines.append("\nTop Classtypes:")
        for ct in chars['classtypes']:
            lines.append(f"  • {ct['name']}: {ct['percentage']:.1f}%")

    if 'protocols' in chars and chars['protocols']:
        lines.append("\nProtocols:")
        for proto in chars['protocols']:
            lines.append(f"  • {proto['name']}: {proto['percentage']:.1f}%")

    if desc['top_terms']:
        terms = [t['term'] for t in desc['top_terms'][:10]]
        lines.append(f"\nKey Terms: {', '.join(terms)}")

    if desc['representative_rules']:
        lines.append("\nRepresentative Rules:")
        for i, rule in enumerate(desc['representative_rules'], 1):
            lines.append(f"  {i}. [{rule['type']}] {rule['message']}")

    return "\n".join(lines)
