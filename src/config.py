"""Central configuration: project paths and placeholder constants.

All paths are derived from the project root so the code is portable
across machines and checkout locations.
"""

from datetime import date, timedelta
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

# Temporal split (leakage-safe evaluation).
# The last day present in the H&M dataset. The label window is the last
# LABEL_WINDOW_DAYS ending on this date; everything strictly before the cutoff
# is feature history.
DATASET_MAX_DATE = date(2020, 9, 22)  # = max(t_dat) in the sampled event log

# Label window chosen empirically (see reports/week1_data_profile.md). A 7/14/28
# -day sensitivity check showed none of the candidates reach a ~20k evaluable
# customer set; 28 days gives the largest evaluable set (16,895 customers, ~17%
# of active) while staying a realistic 4-week short-horizon task and retaining
# 98.6% of events as features. 7 (5%) and 14 (9%) leave the evaluation set too
# thin, so we deviate from the default of 14 in favor of a healthier eval set.
LABEL_WINDOW_DAYS = 28

# Cutoff computed in code: max(t_dat) - LABEL_WINDOW_DAYS + 1, so the label
# window [CUTOFF_DATE, DATASET_MAX_DATE] is exactly LABEL_WINDOW_DAYS long.
# Resolves to the literal date 2020-08-26.
CUTOFF_DATE = DATASET_MAX_DATE - timedelta(days=LABEL_WINDOW_DAYS - 1)

# Granularity of the action space. Product-type granularity balances
# recommendation precision against statistical learnability: it avoids
# article-level sparsity (too many actions, too few observations each) while
# staying more actionable than broad product groups.
ACTION_GRANULARITY = "product_type_name"
