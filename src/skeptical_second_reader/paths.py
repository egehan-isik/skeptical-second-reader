"""Project paths, resolved from this file's location (no absolute paths)."""

from pathlib import Path

# src/skeptical_second_reader/paths.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
