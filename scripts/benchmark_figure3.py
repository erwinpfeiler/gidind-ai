#!/usr/bin/env python3
"""Cluster/reviewer entry point: python scripts/benchmark_figure3.py --help."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gidind_ai.figure3_benchmark import main

if __name__ == "__main__":
    main()
