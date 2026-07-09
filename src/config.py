"""Central configuration: project paths and placeholder constants.

All paths are derived from the project root so the code is portable
across machines and checkout locations.
"""

from pathlib import Path

# Project root: two levels up from this file (src/config.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories (not committed to git; see .gitignore).
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Output directory for generated reports and figures.
REPORTS_DIR = PROJECT_ROOT / "reports"

# Placeholder constants (to be filled in during later milestones).
CUTOFF_DATE = None
ACTION_GRANULARITY = "product_group_name"
