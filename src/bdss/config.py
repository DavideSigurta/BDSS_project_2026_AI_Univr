"""Single place for tunable values, with the reason for each."""

from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[2]

# Datasets are local and outside the repository.  Running from `repo/` means `../datasets`.
DATASETS_DIR = Path(os.environ.get("BDSS_DATASETS", REPO_ROOT.parent / "datasets"))

# CUDD is the fast backend; tests use `dd.autoref` as the reference implementation.
BDD_BACKEND = os.environ.get("BDSS_BDD_BACKEND", "cudd")

# CONDIZIONI §6: every input bit weighs 1/2, so a weighted count is count / 2**n_x.
VOLUME_WEIGHT = 0.5

# Order-of-magnitude grid plus the uncapped build as a reference.
BUDGETS = (1_000, 10_000, 100_000, 1_000_000, None)
TOY_BUDGETS = (1, 2, 4, None)

# A timeout or a node-cap hit is a failure status, not a data point.  Initial values; frozen
# after the M3 pilot.
TIME_LIMIT_S = 120.0
NODE_CAP = 2_000_000

# 8 GB RAM, one CUDD manager per worker.
N_JOBS = 4

# Random-node selector.
SEED = 0

# Prefix search for forests that never complete.
PREFIX_K = (5, 10)
