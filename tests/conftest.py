"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_rule_lines():
    """Sample Suricata rule lines for testing."""
    return [
        'alert tcp any any -> any any (msg:"Test rule 1"; sid:1000001; rev:1; classtype:trojan-activity; priority:1;)',
        'alert http any any -> any any (msg:"Test HTTP rule"; sid:1000002; rev:2; classtype:web-application-attack; priority:2;)',
        'alert tls $EXTERNAL_NET any -> $HOME_NET any (msg:"SSL Test"; sid:1000003; rev:1; classtype:misc-activity;)',
        'drop tcp $EXTERNAL_NET any -> $HOME_NET 445 (msg:"SMB Attack"; sid:1000004; rev:3; priority:1;)',
        'alert dns any any -> any any (msg:"DNS Query"; sid:1000005; rev:1;)',
    ]


@pytest.fixture
def sample_rules_file(tmp_path, sample_rule_lines):
    """Create a temporary rule file for testing."""
    rules_file = tmp_path / "test.rules"

    # Write sample rules
    with open(rules_file, 'w') as f:
        f.write("# Test rule file\n")
        f.write("# Comment line\n\n")
        for rule in sample_rule_lines:
            f.write(rule + "\n")

    return rules_file


@pytest.fixture
def sample_rules_dir(tmp_path, sample_rule_lines):
    """Create a temporary directory with multiple rule files."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()

    # Create multiple rule files
    for i in range(3):
        rules_file = rules_dir / f"test_{i}.rules"
        with open(rules_file, 'w') as f:
            for j, rule in enumerate(sample_rule_lines):
                # Modify SID to be unique across files
                modified_rule = rule.replace(f"sid:100000{j+1}", f"sid:100{i}{j+1}")
                f.write(modified_rule + "\n")

    yield rules_dir

    # Cleanup
    shutil.rmtree(rules_dir)


@pytest.fixture
def sample_parsed_df():
    """Sample parsed rules DataFrame."""
    data = {
        'raw_rule': [
            'alert tcp any any -> any any (msg:"Rule 1"; sid:1;)',
            'alert http any any -> any any (msg:"Rule 2"; sid:2;)',
            'alert tls any any -> any any (msg:"Rule 3"; sid:3;)',
            'drop tcp any any -> any 445 (msg:"Rule 4"; sid:4;)',
            'alert dns any any -> any any (msg:"Rule 5"; sid:5;)',
        ],
        'file_path': ['test.rules'] * 5,
        'file_name': ['test.rules'] * 5,
        'enabled': [True] * 5,
        'action': ['alert', 'alert', 'alert', 'drop', 'alert'],
        'protocol': ['tcp', 'http', 'tls', 'tcp', 'dns'],
        'src_ip': ['any'] * 5,
        'src_port': ['any'] * 5,
        'direction': ['->'] * 5,
        'dst_ip': ['any'] * 5,
        'dst_port': ['any', 'any', 'any', '445', 'any'],
        'options': [{}] * 5,
        'msg': ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5'],
        'sid': [1, 2, 3, 4, 5],
        'rev': [1, 1, 1, 1, 1],
        'classtype': ['trojan-activity', 'web-application-attack', 'misc-activity', 'attempted-admin', None],
        'priority': [1, 2, 3, 1, None],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_feature_matrix():
    """Sample feature matrix for clustering tests."""
    np.random.seed(42)
    return np.random.randn(100, 20)


@pytest.fixture
def sample_labels():
    """Sample cluster labels."""
    np.random.seed(42)
    return np.random.randint(0, 5, 100)


@pytest.fixture
def real_rules_dir():
    """Path to real rules directory (if available)."""
    rules_path = Path("rules")
    if rules_path.exists():
        return rules_path
    return None
