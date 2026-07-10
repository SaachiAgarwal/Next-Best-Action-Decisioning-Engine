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

# Sampling configuration — defines the working dataset for the whole project.
# We sample a fixed set of customers (and their complete purchase history) so
# development stays tractable locally. The seed lives here so the exact same
# sample regenerates on every run and on every machine.
SAMPLE_CUSTOMERS = 100000
SAMPLE_SEED = 42

# Placeholder constants (to be filled in during later milestones).
CUTOFF_DATE = None

# Granularity of the action space. Product-type granularity balances
# recommendation precision against statistical learnability: it avoids
# article-level sparsity (too many actions, too few observations each) while
# staying more actionable than broad product groups.
ACTION_GRANULARITY = "product_type_name"
