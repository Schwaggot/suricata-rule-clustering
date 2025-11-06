"""
Unit tests for the features module.
"""

import pytest
import pandas as pd
import numpy as np
from suricata_rule_clustering.features import RuleFeatureExtractor, get_feature_names


class TestRuleFeatureExtractor:
    """Tests for RuleFeatureExtractor class."""

    def test_initialization(self):
        """Test that extractor initializes correctly."""
        extractor = RuleFeatureExtractor()

        assert extractor.label_encoders == {}
        assert extractor.scaler is not None
        assert extractor.tfidf_vectorizer is None

    def test_extract_basic_features(self, sample_parsed_df):
        """Test basic feature extraction."""
        extractor = RuleFeatureExtractor()
        result_df = extractor.extract_basic_features(sample_parsed_df)

        # Check new columns are added
        assert 'action_encoded' in result_df.columns
        assert 'protocol_encoded' in result_df.columns
        assert 'has_specific_src_port' in result_df.columns
        assert 'has_specific_dst_port' in result_df.columns
        assert 'bidirectional' in result_df.columns

        # Check values
        assert result_df['bidirectional'].dtype == int
        assert all(result_df['bidirectional'].isin([0, 1]))

    def test_extract_basic_features_port_detection(self):
        """Test port feature extraction."""
        data = {
            'action': ['alert'] * 3,
            'protocol': ['tcp'] * 3,
            'src_port': ['any', '80', '443'],
            'dst_port': ['any', 'any', '22'],
            'direction': ['->', '->', '->']
        }
        df = pd.DataFrame(data)

        extractor = RuleFeatureExtractor()
        result_df = extractor.extract_basic_features(df)

        assert result_df.iloc[0]['has_specific_src_port'] == 0
        assert result_df.iloc[1]['has_specific_src_port'] == 1
        assert result_df.iloc[2]['has_specific_src_port'] == 1

        assert result_df.iloc[0]['has_specific_dst_port'] == 0
        assert result_df.iloc[1]['has_specific_dst_port'] == 0
        assert result_df.iloc[2]['has_specific_dst_port'] == 1

    def test_extract_option_features(self, sample_parsed_df):
        """Test option-based feature extraction."""
        extractor = RuleFeatureExtractor()
        result_df = extractor.extract_option_features(sample_parsed_df)

        # Check new columns
        assert 'num_options' in result_df.columns
        option_cols = [col for col in result_df.columns if col.startswith('has_') or col.startswith('num_')]
        assert len(option_cols) > 0

    def test_extract_option_features_with_content(self):
        """Test option extraction with content patterns."""
        data = {
            'action': ['alert'] * 3,
            'protocol': ['http'] * 3,
            'options': [
                {'content': ['test'], 'pcre': ['/regex/']},
                {'content': ['a', 'b'], 'pcre': []},
                {}
            ]
        }
        df = pd.DataFrame(data)

        extractor = RuleFeatureExtractor()
        result_df = extractor.extract_option_features(df)

        assert result_df.iloc[0]['num_content'] == 1
        assert result_df.iloc[1]['num_content'] == 2
        assert result_df.iloc[2]['num_content'] == 0

        assert result_df.iloc[0]['num_pcre'] == 1
        assert result_df.iloc[1]['num_pcre'] == 0

    def test_extract_metadata_features(self, sample_parsed_df):
        """Test metadata feature extraction."""
        extractor = RuleFeatureExtractor()
        result_df = extractor.extract_metadata_features(sample_parsed_df)

        # Check for message-related features
        if 'msg' in sample_parsed_df.columns:
            assert 'msg_length' in result_df.columns
            assert 'msg_word_count' in result_df.columns

        # Check for classtype encoding
        if 'classtype' in sample_parsed_df.columns:
            assert 'classtype_encoded' in result_df.columns

    def test_extract_text_features(self, sample_parsed_df):
        """Test TF-IDF text feature extraction."""
        extractor = RuleFeatureExtractor()
        result_df, tfidf_matrix = extractor.extract_text_features(
            sample_parsed_df,
            max_features=10
        )

        assert tfidf_matrix is not None
        assert tfidf_matrix.shape[0] == len(sample_parsed_df)
        assert tfidf_matrix.shape[1] <= 10
        assert extractor.tfidf_vectorizer is not None

    def test_extract_text_features_no_msg_column(self):
        """Test text extraction when msg column is missing."""
        df = pd.DataFrame({'action': ['alert'], 'protocol': ['tcp']})

        extractor = RuleFeatureExtractor()
        result_df, tfidf_matrix = extractor.extract_text_features(df)

        assert tfidf_matrix is None

    def test_create_feature_matrix(self, sample_parsed_df):
        """Test creating complete feature matrix."""
        extractor = RuleFeatureExtractor()
        X = extractor.create_feature_matrix(sample_parsed_df, include_tfidf=False)

        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(sample_parsed_df)
        assert X.shape[1] > 0

    def test_create_feature_matrix_with_tfidf(self, sample_parsed_df):
        """Test creating feature matrix with TF-IDF."""
        extractor = RuleFeatureExtractor()
        X = extractor.create_feature_matrix(
            sample_parsed_df,
            include_tfidf=True,
            tfidf_max_features=20
        )

        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(sample_parsed_df)
        # Should have both numeric and TF-IDF features
        assert X.shape[1] > 20

    def test_create_feature_matrix_scaled(self, sample_parsed_df):
        """Test that feature matrix is properly scaled."""
        extractor = RuleFeatureExtractor()
        X = extractor.create_feature_matrix(sample_parsed_df)

        # Check that features are approximately normalized
        # (mean close to 0, std close to 1)
        assert np.abs(X.mean()) < 1.0
        # Some features might have low variance, so check mean std
        mean_std = np.mean(X.std(axis=0))
        assert 0.5 < mean_std < 1.5


class TestFeatureExtractionEdgeCases:
    """Tests for edge cases in feature extraction."""

    def test_empty_dataframe(self):
        """Test feature extraction on empty DataFrame."""
        df = pd.DataFrame()
        extractor = RuleFeatureExtractor()

        # Should not crash
        result = extractor.extract_basic_features(df)
        assert len(result) == 0

    def test_dataframe_with_missing_columns(self):
        """Test feature extraction with missing expected columns."""
        df = pd.DataFrame({'action': ['alert'], 'protocol': ['tcp']})
        extractor = RuleFeatureExtractor()

        # Should not crash, just skip missing features
        result = extractor.extract_basic_features(df)
        assert len(result) == 1

    def test_dataframe_with_null_values(self):
        """Test feature extraction with null values."""
        data = {
            'action': ['alert', None, 'drop'],
            'protocol': ['tcp', 'http', None],
            'msg': ['Test', None, 'Message'],
            'classtype': [None, 'trojan', None]
        }
        df = pd.DataFrame(data)

        extractor = RuleFeatureExtractor()

        # Should handle nulls gracefully
        result = extractor.extract_basic_features(df)
        assert len(result) == 3

        result = extractor.extract_metadata_features(df)
        assert len(result) == 3


class TestFeatureHelpers:
    """Tests for helper functions."""

    def test_get_feature_names_no_tfidf(self):
        """Test getting feature names without TF-IDF."""
        extractor = RuleFeatureExtractor()
        names = get_feature_names(extractor)

        assert isinstance(names, list)
        assert len(names) == 0  # No TF-IDF features

    def test_get_feature_names_with_tfidf(self, sample_parsed_df):
        """Test getting feature names with TF-IDF."""
        extractor = RuleFeatureExtractor()

        # Extract TF-IDF features first
        extractor.extract_text_features(sample_parsed_df, max_features=10)

        names = get_feature_names(extractor)

        assert isinstance(names, list)
        assert len(names) > 0


class TestIntegration:
    """Integration tests for feature extraction."""

    def test_full_feature_extraction_pipeline(self, sample_parsed_df):
        """Test complete feature extraction workflow."""
        extractor = RuleFeatureExtractor()

        # Extract all feature types
        df = extractor.extract_basic_features(sample_parsed_df)
        df = extractor.extract_option_features(df)
        df = extractor.extract_metadata_features(df)

        # Create feature matrix
        X = extractor.create_feature_matrix(
            sample_parsed_df,
            include_tfidf=True,
            tfidf_max_features=50
        )

        # Verify results
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(sample_parsed_df)
        assert X.shape[1] > 0

        # Check for NaN or inf values
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    @pytest.mark.skipif(not pd.io.common.file_exists("data/parsed_rules.pkl"),
                        reason="Parsed rules file not found")
    def test_real_data_feature_extraction(self):
        """Test feature extraction on real parsed rules."""
        from suricata_rule_clustering import parser

        # Load real data
        df = parser.load_parsed_rules("data/parsed_rules.pkl")

        # Extract features
        extractor = RuleFeatureExtractor()
        X = extractor.create_feature_matrix(df[:100], include_tfidf=True)  # Use first 100 for speed

        assert X.shape[0] == 100
        assert X.shape[1] > 0
        assert not np.isnan(X).any()
