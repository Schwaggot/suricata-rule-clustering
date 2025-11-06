"""
Rule Parser Module

This module handles parsing Suricata rules from rule files using
the suricata-rule-parser package and converting them to pandas DataFrames.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from suricata_rule_parser import parse_file


def load_rule_files(rules_dir: str = "rules") -> List[Path]:
    """
    Load all .rules files from the specified directory.

    Args:
        rules_dir: Path to the directory containing rule files

    Returns:
        List of Path objects for all .rules files found
    """
    rules_path = Path(rules_dir)
    if not rules_path.exists():
        raise FileNotFoundError(f"Rules directory not found: {rules_dir}")

    rule_files = list(rules_path.rglob("*.rules"))
    print(f"Found {len(rule_files)} rule files in {rules_dir}")
    return rule_files


def parse_rule_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse all rules from a single rule file.

    Args:
        file_path: Path to the rule file

    Returns:
        List of parsed rule dictionaries
    """
    parsed_rules = []
    errors = 0

    try:
        # Use the built-in parse_file function
        rules = parse_file(str(file_path))

        for idx, rule in enumerate(rules):
            try:
                # Convert SuricataRule object to dictionary
                rule_dict_raw = rule.to_dict()

                # Flatten the structure
                rule_dict = {
                    'raw_rule': rule_dict_raw.get('raw', ''),
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'enabled': rule_dict_raw.get('enabled', True),
                }

                # Add header fields
                if 'header' in rule_dict_raw:
                    header = rule_dict_raw['header']
                    rule_dict['action'] = header.get('action')
                    rule_dict['protocol'] = header.get('protocol')
                    rule_dict['src_ip'] = header.get('source_ip')
                    rule_dict['src_port'] = header.get('source_port')
                    rule_dict['direction'] = header.get('direction')
                    rule_dict['dst_ip'] = header.get('dest_ip')
                    rule_dict['dst_port'] = header.get('dest_port')

                # Add options (keep as nested dict for feature extraction)
                rule_dict['options'] = rule_dict_raw.get('options', {})

                # Extract commonly used options to top level for convenience
                options = rule_dict['options']
                rule_dict['msg'] = options.get('msg')
                rule_dict['sid'] = options.get('sid')
                rule_dict['rev'] = options.get('rev')
                rule_dict['classtype'] = options.get('classtype')
                rule_dict['priority'] = options.get('priority')

                parsed_rules.append(rule_dict)
            except Exception as e:
                # Track errors for debugging
                errors += 1
                if errors <= 3:  # Only print first few errors
                    print(f"  Error parsing rule {idx} in {file_path.name}: {e}")
                continue

        if errors > 3:
            print(f"  ... and {errors - 3} more errors in {file_path.name}")

    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        import traceback
        traceback.print_exc()

    return parsed_rules


def parse_all_rules(rules_dir: str = "rules", max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Parse all Suricata rules from the rules directory and return as DataFrame.

    Args:
        rules_dir: Path to the directory containing rule files
        max_files: Optional limit on number of files to process (for testing)

    Returns:
        pandas DataFrame with all parsed rules
    """
    rule_files = load_rule_files(rules_dir)

    if max_files:
        rule_files = rule_files[:max_files]
        print(f"Processing only first {max_files} files for testing")

    all_rules = []
    for i, rule_file in enumerate(rule_files, 1):
        print(f"Processing {i}/{len(rule_files)}: {rule_file.name}", end='\r')
        rules = parse_rule_file(rule_file)
        all_rules.extend(rules)

    print(f"\nSuccessfully parsed {len(all_rules)} rules from {len(rule_files)} files")

    df = pd.DataFrame(all_rules)
    return df


def save_parsed_rules(df: pd.DataFrame, output_path: str = "data/parsed_rules.pkl"):
    """
    Save parsed rules DataFrame to disk.

    Args:
        df: DataFrame with parsed rules
        output_path: Path to save the DataFrame (pickle format)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_pickle(output_path)
    print(f"Saved {len(df)} rules to {output_path}")


def load_parsed_rules(input_path: str = "data/parsed_rules.pkl") -> pd.DataFrame:
    """
    Load previously parsed rules from disk.

    Args:
        input_path: Path to the saved DataFrame

    Returns:
        pandas DataFrame with parsed rules
    """
    df = pd.read_pickle(input_path)
    print(f"Loaded {len(df)} rules from {input_path}")
    return df
