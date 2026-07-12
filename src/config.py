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

# --- Experiment 3: recency + frequency weighted hybrid ---------------------
# Recency half-life: a purchase HALF_LIFE_DAYS before the reference date counts
# half as much as one at the reference date (exponential decay).
HALF_LIFE_DAYS = 30

# Internal validation window carved from the FEATURE side for weight tuning.
# The last VALID_WINDOW_DAYS of pre-cutoff events are held out as mini-labels;
# the real post-cutoff labels are never used for tuning.
VALID_WINDOW_DAYS = 28

# Blend weights for the hybrid final score:
#   final = ALPHA*personal + BETA*cf + GAMMA*popularity  (components normalized).
# These are defaults; the tuned values live in hybrid_weights_exp3.json (chosen
# by validation grid search) and are authoritative for reporting.
ALPHA_EXP3 = 0.5   # recency x log-frequency own-history signal
BETA_EXP3 = 0.5    # recency-weighted collaborative-filtering signal
GAMMA_EXP3 = 1.0   # global popularity prior (guarantees >= popularity floor)

# --- Experiment 4: content-based + content/CF hybrid (article level) --------
# Blend weights for final = CONTENT_ALPHA*content + CONTENT_BETA*cf (components
# normalized). Defaults; tuned values live in hybrid_weights_exp4.json.
CONTENT_ALPHA_EXP4 = 1.0   # content-similarity signal (article attributes)
CONTENT_BETA_EXP4 = 1.0    # article-level CF signal (Exp B co-occurrence)

# --- Phase 2: LinUCB contextual bandit -------------------------------------
# UCB exploration parameter: p_a = theta_a . x + BANDIT_ALPHA * sqrt(x . A_a^-1 . x).
# 0 = pure exploitation (no exploration); higher = more exploration.
BANDIT_ALPHA = 1.0
SEED = SAMPLE_SEED  # 42 — single stochastic seed for the whole project

# --- Phase 2 revision (v2): fair evaluation --------------------------------
# Split the evaluable customers into a learning stream (the bandit updates on
# these) and a held-out eval set (decisions only, never updated). Multi-pass
# learning lets the weights converge; held-out eval is the honest generalization
# test.
BANDIT_LEARN_FRAC = 0.7   # fraction of evaluable customers used for learning
BANDIT_EPOCHS = 5         # passes over the learning stream
