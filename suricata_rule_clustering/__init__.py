"""
Suricata Rule Clustering Package

A machine learning project for clustering Suricata IDS/IPS rules
to find groups of similar rules using various clustering algorithms.
"""

__version__ = "0.1.0"

from . import parser
from . import features
from . import clustering
from . import viz

__all__ = ["parser", "features", "clustering", "viz"]
