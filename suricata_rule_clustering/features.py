"""
Feature Engineering Module

This module handles feature extraction and engineering from parsed Suricata rules
to prepare them for clustering.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


class RuleFeatureExtractor:
    """Extract and engineer features from parsed Suricata rules."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None

    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic features from parsed rules.

        Args:
            df: DataFrame with parsed rules

        Returns:
            DataFrame with basic features added
        """
        features_df = df.copy()

        # Action features
        if 'action' in features_df.columns:
            features_df['action_encoded'] = self._encode_categorical(
                features_df['action'], 'action'
            )

        # Protocol features
        if 'protocol' in features_df.columns:
            features_df['protocol_encoded'] = self._encode_categorical(
                features_df['protocol'], 'protocol'
            )

        # Port features
        if 'src_port' in features_df.columns:
            features_df['has_specific_src_port'] = features_df['src_port'].apply(
                lambda x: 0 if x in ['any', None] else 1
            )
        if 'dst_port' in features_df.columns:
            features_df['has_specific_dst_port'] = features_df['dst_port'].apply(
                lambda x: 0 if x in ['any', None] else 1
            )

        # Direction features
        if 'direction' in features_df.columns:
            features_df['bidirectional'] = features_df['direction'].apply(
                lambda x: 1 if x == '<>' else 0
            )

        return features_df

    def extract_option_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from rule options.

        Args:
            df: DataFrame with parsed rules

        Returns:
            DataFrame with option features added
        """
        features_df = df.copy()

        if 'options' not in features_df.columns:
            return features_df

        # Count total options
        features_df['num_options'] = features_df['options'].apply(
            lambda x: len(x) if isinstance(x, (list, dict)) else 0
        )

        # Check for specific option types
        option_types = [
            'content', 'pcre', 'http_uri', 'http_header', 'http_method',
            'flow', 'flowbits', 'threshold', 'detection_filter',
            'fast_pattern', 'noalert', 'sid', 'rev', 'classtype',
            'msg', 'reference', 'priority', 'metadata'
        ]

        for opt_type in option_types:
            features_df[f'has_{opt_type}'] = features_df['options'].apply(
                lambda x: self._has_option(x, opt_type)
            )

        # Count content and pcre patterns
        features_df['num_content'] = features_df['options'].apply(
            lambda x: self._count_option(x, 'content')
        )
        features_df['num_pcre'] = features_df['options'].apply(
            lambda x: self._count_option(x, 'pcre')
        )

        return features_df

    def extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from rule metadata.

        Args:
            df: DataFrame with parsed rules

        Returns:
            DataFrame with metadata features added
        """
        features_df = df.copy()

        # Message/description features
        if 'msg' in features_df.columns or 'message' in features_df.columns:
            msg_col = 'msg' if 'msg' in features_df.columns else 'message'
            features_df['msg_length'] = features_df[msg_col].astype(str).str.len()
            features_df['msg_word_count'] = features_df[msg_col].astype(str).str.split().str.len()

        # Classtype features
        if 'classtype' in features_df.columns:
            features_df['classtype_encoded'] = self._encode_categorical(
                features_df['classtype'], 'classtype'
            )

        # Priority features
        if 'priority' in features_df.columns:
            features_df['priority'] = pd.to_numeric(
                features_df['priority'], errors='coerce'
            ).fillna(3)  # Default priority is 3

        # SID features (can indicate rule families)
        if 'sid' in features_df.columns:
            features_df['sid'] = pd.to_numeric(
                features_df['sid'], errors='coerce'
            )
            features_df['sid_range'] = features_df['sid'].apply(
                lambda x: int(x // 1000000) if pd.notna(x) else -1
            )

        return features_df

    def extract_text_features(
            self,
            df: pd.DataFrame,
            max_features: int = 100,
            ngram_range: tuple = (1, 2)
    ) -> tuple:
        """
        Extract TF-IDF features from rule messages.

        Args:
            df: DataFrame with parsed rules
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF

        Returns:
            Tuple of (features_df, tfidf_matrix)
        """
        msg_col = 'msg' if 'msg' in df.columns else 'message' if 'message' in df.columns else None

        if msg_col is None:
            return df, None

        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )

        # Fit and transform messages
        messages = df[msg_col].fillna('').astype(str)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(messages)

        return df, tfidf_matrix

    def create_feature_matrix(
            self,
            df: pd.DataFrame,
            include_tfidf: bool = True,
            tfidf_max_features: int = 100
    ) -> np.ndarray:
        """
        Create a complete feature matrix for clustering.

        Args:
            df: DataFrame with parsed rules
            include_tfidf: Whether to include TF-IDF features
            tfidf_max_features: Maximum number of TF-IDF features

        Returns:
            numpy array with all features
        """
        # Extract all feature types
        features_df = self.extract_basic_features(df)
        features_df = self.extract_option_features(features_df)
        features_df = self.extract_metadata_features(features_df)

        # Select numeric features for clustering
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove metadata columns that shouldn't be used for clustering
        exclude_cols = ['line_number', 'sid', 'rev']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]

        feature_matrix = features_df[numeric_features].fillna(0).values

        # Add TF-IDF features if requested
        if include_tfidf:
            _, tfidf_matrix = self.extract_text_features(
                features_df,
                max_features=tfidf_max_features
            )
            if tfidf_matrix is not None:
                feature_matrix = np.hstack([feature_matrix, tfidf_matrix.toarray()])

        # Scale features
        feature_matrix = self.scaler.fit_transform(feature_matrix)

        print(f"Created feature matrix with shape: {feature_matrix.shape}")
        return feature_matrix

    def _encode_categorical(self, series: pd.Series, name: str) -> pd.Series:
        """Encode categorical features using LabelEncoder."""
        if name not in self.label_encoders:
            self.label_encoders[name] = LabelEncoder()

        return pd.Series(
            self.label_encoders[name].fit_transform(series.fillna('unknown')),
            index=series.index
        )

    def _has_option(self, options: Any, option_name: str) -> int:
        """Check if a specific option exists in rule options."""
        if not isinstance(options, dict):
            return 0
        return 1 if option_name in options else 0

    def _count_option(self, options: Any, option_name: str) -> int:
        """Count occurrences of a specific option in rule options."""
        if not isinstance(options, dict):
            return 0

        value = options.get(option_name, [])
        if isinstance(value, list):
            return len(value)
        elif value:
            return 1
        return 0


def get_feature_names(extractor: RuleFeatureExtractor) -> List[str]:
    """
    Get names of all extracted features.

    Args:
        extractor: Fitted RuleFeatureExtractor instance

    Returns:
        List of feature names
    """
    feature_names = []

    # Add TF-IDF feature names if available
    if extractor.tfidf_vectorizer:
        feature_names.extend(extractor.tfidf_vectorizer.get_feature_names_out())

    return feature_names
