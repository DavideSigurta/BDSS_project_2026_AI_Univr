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

# A timeout or a node-cap hit is a failure status, not a data point.
NODE_CAP = 2_000_000

# Random-node selector.
SEED = 0


# Final campaign: 54 models x 3 budgets x 5 selectors, plus build completion for non-full models.
RUN_TIMEOUT_S = 120.0
LOCAL_JOBS = 2
JOB_CPU = 1.0                 # one physical core per run; CUDD is single-thread
JOB_MEMORY_MB = 4096
MAX_USD = 30.0          # hard safety cap (not a target)
SOFT_USD = 10.0         # warning only
# Modal Starter CPU/memory rates, USD per core-hour and per GiB-hour.
CPU_USD_PER_CORE_HOUR = 0.047
MEMORY_USD_PER_GIB_HOUR = 0.008

# Final campaign: a separate, definitive run with its own result files.
FINAL_BUDGETS = (1_000, 10_000, 100_000)
FINAL_SELECTORS = ("modal", "random", "ig", "shapley", "pvalue")
FINAL_TIMEOUT_S = 300.0
SELECTOR_TIMEOUT_S = 60.0
FINAL_NODE_CAP = 8_000_000
FINAL_MEMORY_MB = 16_384
FINAL_CONTAINERS = 16
FINAL_MAX_USD = 30.0

# Post-build is two products: a cheap readout pass (no test set) and a chunked robustness
# pass.  A worker gets a 600 s working budget inside a 1200 s function so it can persist the
# slice before Modal kills it; short slices keep the first chunk visible and bounded.
READOUT_TIMEOUT_S = 14400.0
READOUT_MEMORY_MB = 8_192
READOUT_CONTAINERS = 32
ROBUSTNESS_WORKER_S = 600.0
ROBUSTNESS_TIMEOUT_S = 1_200.0
ROBUSTNESS_MEMORY_MB = 16_384
ROBUSTNESS_CONTAINERS = 32

# Build completion: the final grid leaves 36 models without a full build.  This phase
# searches only their prefixes, with the same 3 budgets and all 5 selectors as the final
# grid.  The fixed range is the coarse search; the bisection locates the adjacent wall.
BUILD_BUDGETS = FINAL_BUDGETS
BUILD_SELECTORS = FINAL_SELECTORS
BUILD_KS = (5, 10, 20, 30, 50)
BUILD_TIMEOUT_S = 300.0
BUILD_NODE_CAP = 4_000_000
BUILD_MEMORY_MB = 8_192
BUILD_CONTAINERS = 16
