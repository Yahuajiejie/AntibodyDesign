#!/usr/bin/env python3
"""Compatibility wrapper for scripts/data/filter_records.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "data" / "filter_records.py"),
        run_name="__main__",
    )
