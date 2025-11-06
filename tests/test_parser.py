"""
Unit tests for the parser module.
"""

import pytest
import pandas as pd
from pathlib import Path
from suricata_rule_clustering import parser


class TestLoadRuleFiles:
    """Tests for load_rule_files function."""

    def test_load_rule_files_from_directory(self, sample_rules_dir):
        """Test loading rule files from a directory."""
        rule_files = parser.load_rule_files(str(sample_rules_dir))

        assert len(rule_files) == 3
        assert all(isinstance(f, Path) for f in rule_files)
        assert all(f.suffix == '.rules' for f in rule_files)

    def test_load_rule_files_nonexistent_directory(self):
        """Test loading from non-existent directory raises error."""
        with pytest.raises(FileNotFoundError):
            parser.load_rule_files("nonexistent_directory")

    @pytest.mark.skipif(not Path("rules").exists(), reason="Real rules directory not found")
    def test_load_real_rule_files(self):
        """Test loading real rule files if available."""
        rule_files = parser.load_rule_files("rules")

        assert len(rule_files) > 0
        assert all(f.exists() for f in rule_files)


class TestParseRuleFile:
    """Tests for parse_rule_file function."""

    def test_parse_single_file(self, sample_rules_file):
        """Test parsing a single rule file."""
        rules = parser.parse_rule_file(sample_rules_file)

        assert len(rules) == 5  # 5 sample rules
        assert all(isinstance(r, dict) for r in rules)

    def test_parsed_rule_structure(self, sample_rules_file):
        """Test that parsed rules have expected structure."""
        rules = parser.parse_rule_file(sample_rules_file)

        expected_keys = [
            'raw_rule', 'file_path', 'file_name', 'enabled',
            'action', 'protocol', 'src_ip', 'src_port',
            'direction', 'dst_ip', 'dst_port', 'options',
            'msg', 'sid', 'rev', 'classtype', 'priority'
        ]

        first_rule = rules[0]
        for key in expected_keys:
            assert key in first_rule, f"Missing key: {key}"

    def test_parsed_rule_values(self, sample_rules_file):
        """Test that parsed rules have correct values."""
        rules = parser.parse_rule_file(sample_rules_file)

        first_rule = rules[0]
        assert first_rule['action'] == 'alert'
        assert first_rule['protocol'] == 'tcp'
        assert first_rule['msg'] == 'Test rule 1'
        assert first_rule['sid'] == 1000001
        assert first_rule['rev'] == 1
        assert first_rule['classtype'] == 'trojan-activity'
        assert first_rule['priority'] == 1

    def test_parse_file_with_comments(self, tmp_path):
        """Test that comments and empty lines are skipped."""
        rules_file = tmp_path / "test_comments.rules"
        with open(rules_file, 'w') as f:
            f.write("# Comment line\n")
            f.write("\n")
            f.write("# Another comment\n")
            f.write('alert tcp any any -> any any (msg:"Test"; sid:1;)\n')

        rules = parser.parse_rule_file(rules_file)

        assert len(rules) == 1
        assert rules[0]['sid'] == 1

    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        rules_file = tmp_path / "empty.rules"
        rules_file.touch()

        rules = parser.parse_rule_file(rules_file)

        assert len(rules) == 0


class TestParseAllRules:
    """Tests for parse_all_rules function."""

    def test_parse_all_rules_from_directory(self, sample_rules_dir):
        """Test parsing all rules from a directory."""
        df = parser.parse_all_rules(str(sample_rules_dir))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15  # 3 files * 5 rules each
        assert len(df.columns) == 17

    def test_parse_all_rules_with_max_files(self, sample_rules_dir):
        """Test parsing with max_files limit."""
        df = parser.parse_all_rules(str(sample_rules_dir), max_files=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10  # 2 files * 5 rules each

    def test_parse_all_rules_dataframe_structure(self, sample_rules_dir):
        """Test that resulting DataFrame has correct structure."""
        df = parser.parse_all_rules(str(sample_rules_dir))

        expected_columns = [
            'raw_rule', 'file_path', 'file_name', 'enabled',
            'action', 'protocol', 'src_ip', 'src_port',
            'direction', 'dst_ip', 'dst_port', 'options',
            'msg', 'sid', 'rev', 'classtype', 'priority'
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_parse_all_rules_unique_sids(self, sample_rules_dir):
        """Test that SIDs are unique across files."""
        df = parser.parse_all_rules(str(sample_rules_dir))

        # All SIDs should be unique
        assert len(df['sid'].unique()) == len(df)

    @pytest.mark.skipif(not Path("rules").exists(), reason="Real rules directory not found")
    def test_parse_real_rules_sample(self):
        """Test parsing a sample of real rules."""
        df = parser.parse_all_rules("rules", max_files=1)

        assert len(df) > 0
        assert isinstance(df, pd.DataFrame)
        assert 'msg' in df.columns
        assert 'sid' in df.columns


class TestSaveAndLoadParsedRules:
    """Tests for save_parsed_rules and load_parsed_rules functions."""

    def test_save_parsed_rules(self, sample_parsed_df, tmp_path):
        """Test saving parsed rules."""
        output_path = tmp_path / "test_rules.pkl"

        parser.save_parsed_rules(sample_parsed_df, str(output_path))

        assert output_path.exists()

    def test_load_parsed_rules(self, sample_parsed_df, tmp_path):
        """Test loading saved rules."""
        output_path = tmp_path / "test_rules.pkl"

        # Save first
        parser.save_parsed_rules(sample_parsed_df, str(output_path))

        # Then load
        loaded_df = parser.load_parsed_rules(str(output_path))

        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(sample_parsed_df)
        assert list(loaded_df.columns) == list(sample_parsed_df.columns)

    def test_save_load_round_trip(self, sample_parsed_df, tmp_path):
        """Test that save/load preserves data."""
        output_path = tmp_path / "test_rules.pkl"

        # Save and load
        parser.save_parsed_rules(sample_parsed_df, str(output_path))
        loaded_df = parser.load_parsed_rules(str(output_path))

        # Compare DataFrames
        pd.testing.assert_frame_equal(sample_parsed_df, loaded_df)

    def test_save_creates_directory(self, sample_parsed_df, tmp_path):
        """Test that save creates parent directories if needed."""
        output_path = tmp_path / "subdir" / "test_rules.pkl"

        parser.save_parsed_rules(sample_parsed_df, str(output_path))

        assert output_path.exists()
        assert output_path.parent.exists()


class TestIntegration:
    """Integration tests for the parser module."""

    def test_end_to_end_parsing(self, sample_rules_dir, tmp_path):
        """Test complete parsing workflow."""
        # Parse rules
        df = parser.parse_all_rules(str(sample_rules_dir))

        # Verify parsing worked
        assert len(df) > 0

        # Save to file
        output_path = tmp_path / "parsed.pkl"
        parser.save_parsed_rules(df, str(output_path))

        # Load back
        loaded_df = parser.load_parsed_rules(str(output_path))

        # Verify
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)

    @pytest.mark.skipif(not Path("rules").exists(), reason="Real rules directory not found")
    @pytest.mark.slow
    def test_parse_all_real_rules(self, tmp_path):
        """Test parsing all real rules (slow test)."""
        df = parser.parse_all_rules("rules")

        assert len(df) > 1000  # Should have many rules
        assert 'msg' in df.columns
        assert 'sid' in df.columns
        assert df['sid'].nunique() == len(df)  # All unique SIDs

        # Save and verify
        output_path = tmp_path / "all_rules.pkl"
        parser.save_parsed_rules(df, str(output_path))
        assert output_path.exists()
