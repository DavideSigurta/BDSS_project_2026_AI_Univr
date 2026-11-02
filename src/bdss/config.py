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

# M4 uses one completed M3 frontier per cell (modal, a budget with a manageable number of
# leaves) and compares its radii with the uncapped build.
M4_BUILDS = {
    ("banknote", 5): (100, "modal"),
    ("banknote", 10): (1_000, "modal"),
    ("vehicle", 5): (1_000_000, "modal"),
    ("ann-thyroid", 5): (5_000, "modal"),
    ("ann-thyroid", 10): (50_000, "modal"),
}
M4_FRAGILE_TOP = 20

# M5-final: 54 models x 3 budgets x 5 selectors, plus conditional prefixes.  These are the
# only campaign constants; the legacy five-budget campaign is historical (tag m5).
M5_TIMEOUT_S = 120.0
M5_LOCAL_JOBS = 2
M5_CPU = 1.0                 # one physical core per run; CUDD is single-thread
M5_MEMORY_MB = 4096
M5_MAX_USD = 30.0          # hard safety cap (not a target)
M5_SOFT_USD = 10.0         # warning only
# Modal Starter CPU/memory rates, USD per core-hour and per GiB-hour.
M5_CPU_USD_PER_CORE_HOUR = 0.047
M5_MEMORY_USD_PER_GIB_HOUR = 0.008

# M5-final: a separate, definitive run.  It never touches the M5 campaign/result files.
M5_FINAL_BUDGETS = (1_000, 10_000, 100_000)
M5_FINAL_SELECTORS = ("modal", "random", "ig", "shapley", "pvalue")
M5_FINAL_TIMEOUT_S = 300.0
M5_SELECTOR_TIMEOUT_S = 60.0
M5_FINAL_NODE_CAP = 8_000_000
M5_FINAL_MEMORY_MB = 16_384
M5_FINAL_CONTAINERS = 16
M5_FINAL_PREFIX_KS = (50, 30, 20, 10, 5)
M5_FINAL_PREFIX_BUDGETS = (1_000, 10_000, 100_000)
M5_FINAL_PREFIX_SELECTORS = ("modal", "random")
M5_FINAL_PREFIX_CAP = 720
M5_FINAL_MAX_USD = 30.0

# M5 post-build is two products: a cheap readout pass (no test set) and a chunked robustness
# pass.  A worker gets a 600 s working budget inside a 1200 s function so it can persist the
# slice before Modal kills it; short slices keep the first chunk visible and bounded.
M5_READOUT_TIMEOUT_S = 14400.0
M5_READOUT_MEMORY_MB = 8_192
M5_READOUT_CONTAINERS = 32
M5_ROBUSTNESS_WORKER_S = 600.0
M5_ROBUSTNESS_TIMEOUT_S = 1_200.0
M5_ROBUSTNESS_MEMORY_MB = 16_384
M5_ROBUSTNESS_CONTAINERS = 32

# M5 build completion: the final grid leaves 36 models without a full build.  This phase
# searches only their prefixes, with the same 3 budgets and all 5 selectors as the final
# grid.  The fixed range is the coarse search; the bisection locates the adjacent wall.
M5_BUILD_BUDGETS = M5_FINAL_BUDGETS
M5_BUILD_SELECTORS = M5_FINAL_SELECTORS
M5_BUILD_KS = (5, 10, 20, 30, 50)
M5_BUILD_TIMEOUT_S = 300.0
M5_BUILD_NODE_CAP = 4_000_000
M5_BUILD_MEMORY_MB = 8_192
M5_BUILD_CONTAINERS = 16
