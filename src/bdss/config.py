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

# M3 pilot, frozen on 2026-10-27.  The grid is per cell because the feasible budget follows
# the uncapped peak and the number of leaves: B=1 terminates on banknote/5 (12,162 leaves),
# while banknote/10, vehicle/5, ann-thyroid and vehicle/10 time out at 60 s.  vehicle/5 needs
# B>=500k (100k times out); vehicle/10 also times out at B=10k with modal and random, so it is
# the M2 wall again.  Semantic selectors (IG/Shapley/p-value) are exact but too costly on the
# large diagrams: only the cells/budgets below are attempted, the rest are declared.
M3_TIME_LIMIT_S = TIME_LIMIT_S
M3_GRIDS = {
    ("banknote", 5): (1, 10, 100, 1_000),
    ("banknote", 10): (100, 1_000),
    ("ann-thyroid", 5): (5_000, 10_000, 19_000),
    ("ann-thyroid", 10): (50_000, 100_000, 393_000),
    ("vehicle", 5): (500_000, 1_000_000),
    ("vehicle", 10): (10_000,),
}
M3_SEMANTIC = {
    ("banknote", 5): (100, 1_000),
    ("banknote", 10): (100, 1_000),
    ("ann-thyroid", 5): (5_000, 10_000, 19_000),
    ("ann-thyroid", 10): (393_000,),
    ("vehicle", 5): (),
    ("vehicle", 10): (),
}
M3_CHEAP = ("modal", "random")
M3_SEMANTIC_SELECTORS = ("ig", "shapley", "pvalue")
