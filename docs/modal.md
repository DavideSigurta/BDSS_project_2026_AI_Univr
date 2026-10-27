# M5 on Modal (optional)

The campaign is executable locally without Modal.  This page is only for the optional CPU
run on [Modal](https://modal.com/): the same `bdss.campaign.run_one` is wrapped by
`src/bdss/modal_app.py`, which is the only module that imports `modal`.

## Profile and CLI

The profile is chosen by the user; no name is hardcoded.  Export `MODAL_PROFILE`, or pass
`--profile` to Modal commands (`profile`, `volume`, `token`, and `modal run --profile ...`).
The run examples below use the exported variable.  The `modal` executable may not be in `PATH`: use
`uv run --extra modal modal ...`, or install it once with `uv tool install modal`.

```bash
export MODAL_PROFILE=<profile>            # choose your profile
uv run --extra modal modal profile list
uv run --extra modal modal volume list --profile "$MODAL_PROFILE"
```

If the profile is not authenticated, authenticate it once:

```bash
uv run --extra modal modal token new --profile "$MODAL_PROFILE"
```

## Volumes

Two volumes must exist in the chosen profile.  The app does not create or populate them.

```bash
uv run --extra modal modal volume create bdss-datasets --profile "$MODAL_PROFILE"
uv run --extra modal modal volume create bdss-results  --profile "$MODAL_PROFILE"

# The destination `/` puts the local `datasets/` directory at the volume root.
uv run --extra modal modal volume put bdss-datasets ../datasets / --profile "$MODAL_PROFILE"
```

The dataset volume is mounted at `/data`; the files are therefore at
`/data/datasets/<name>/...` and the app sets `BDSS_DATASETS=/data/datasets`.  The results
volume is mounted at `/results` and receives one JSON file per finished run, keyed by
`(dataset, trees, budget, selector)`.

## Smoke test

```bash
uv run --extra modal modal run -m bdss.modal_app --smoke
```

The smoke test builds `banknote/5` with `modal`.  It fails if `dd.cudd` is not importable;
`dd.autoref` is never accepted for the campaign.

## Campaign

```bash
uv run --extra modal modal run -m bdss.modal_app \
  --max-containers 64 --max-usd 30 --resume
```

The entrypoint prints a worst-case estimate before starting: above 10 USD it warns and
continues; it stops only above `--max-usd` (hard 30).  The default grid is 54 models × 5 budgets × 5 selectors = 1,350 runs; the
prefix search for wall models is capped at 180 runs and the adjacent-prefix bisection at 30,
for a total of at most 1,560.  Use `--limit N` for a small batch.  The function is
`cpu=1, memory=4096, timeout=150, max_containers=64`; one run per container.  The internal
run limit is 120 s (`M5_TIMEOUT_S`), so an over-limit run returns a `timeout` row before the
150 s function timeout.

## Rescue and merge

The base campaign writes `timeout_s=120`; the local fallback writes 90, the rescue 300.  The
rescue re-runs only base rows with status
`timeout` / `memory` / `error` at 300 s, without prefixes or bisection:

```bash
uv run python -m bdss.study m5rescue --base results/m5_campaign.csv \
  --out results/m5_campaign_300.csv --max-usd 30
```

On Modal use `uv run --extra modal modal run -m bdss.modal_app --rescue --max-usd 30`.
A completed rescue row wins over a base timeout; if both fail, the rescue row remains.

| phase | timeout | runs | status | billed | CSV estimate | reason |
|---|---:|---:|---|---:|---:|---|
| base | 120 s | 1,554 | 442 completed / 1,088 timeout / 22 memory / 2 error | 3.32 USD | ~2.97 USD | full grid + prefixes |
| rescue | 300 s | 1,112 | 18 completed / 1,036 timeout / 58 memory / 0 error | 7.32 USD | ~7.06 USD | sensitivity on walls |
| merged | 120/300 s | 1,554 | 460 completed / 1,036 timeout / 58 memory / 0 error | — | — | 30 builds, 24 without |
| post | 600 s | 30 builds | 25 post completed / 5 timeout; 24 missing | ~0.4–0.7 USD | — | readouts/robustness |

Billed ~10.64 USD, CSV estimate ~10.03 USD, overhead ~10 s/run; under the 30 USD hard cap.
`source` is not stored in the merged CSV: reconstruct it from `timeout_s` (120 = base,
300 = rescue).  It is stored in the post summary, coverage, and samples.  Counts and volumes
are over the **certified region**, not full distributions; volume sums can be below 1 (e.g.
appendicitis 0.9985).  The post has 3,695 sample rows.

Merge into the file used by the post phase:

```bash
uv run python -m bdss.study m5merge \
  --base results/m5_campaign.csv --rescue results/m5_campaign_300.csv \
  --out results/m5_campaign_merged.csv
```

## Post phase (local)

The post phase reads the campaign CSV and rebuilds one completed build per model (preferring
100k/modal, otherwise the first completed).  It writes counts, volumes, rankings, and the
robustness distribution over the test set, with the chosen build and coverage recorded:

```bash
uv run python -m bdss.study m5post \
  --rows results/m5_campaign_merged.csv --out results/m5_post.csv --max-usd 30
```

Outputs: `results/m5_post.csv` (one selected build per model; it carries `timeout_s` and
`source` = base/rescue), `results/m5_post_samples.csv`
(one row per test sample with the same timeout/source: own label, certification, radii, witness
bits), and
`results/m5_post_coverage.csv` (one row per model: `build_status` completed/missing,
`post_status` completed/timeout/not_run, plus `timeout_s`/`source`).  Current post: **30 builds
available, 25 post completed, 5 post timeout, 24 missing/not_run**.  Models without a completed
build are reported in coverage, not imputed.  `m5post` reads
`m5_campaign_merged.csv` by default; the post phase is local even when the campaign ran on
Modal, so download the CSVs first.

## Cost

At the Modal Starter rates used in `bdss/config.py` (CPU 0.047 USD/core-hour, memory
0.008 USD/GiB-hour), the 1,560-run grid with prefixes and bisection is about 4–5 USD in the worst case
(52 CPU-hours); the all-prefix variant around 3,668 runs is about 9.66 USD.  The `max_usd`
guard is a hard stop, not a target.  More containers reduce wall time, not the cost; CUDD
is single-threaded, so more cores per container do not help.

## Local vs Modal

| | local | Modal |
|---|---|---|
| command | `python -m bdss.study m5` | `uv run --extra modal modal run -m bdss.modal_app` |
| grid | 3 budgets / 90 s / 2 jobs (990 + ≤30 bisection) | 5 budgets / 120 s / 64 containers (1,560 max) |
| dependency | none beyond `uv sync` | optional extra `modal` |
| results | `results/m5_campaign.csv` | CSV locally + JSON on `/results` (backup; resume from the local CSV) |

Local fallback:

```bash
uv sync
uv run python -m bdss.study m5 \
  --budgets 1000,10000,100000 --selectors modal,random,ig,shapley,pvalue \
  --timeout 90 --jobs 2 --max-usd 0 --resume
```

The local runner also does the prefix search and the ≤30 bisection runs; its maximum is
1,020 runs, not 990, when walls and gaps exist.

## Troubleshooting

- `volume list` fails: check the profile name with `uv run --extra modal modal profile list`; authenticate
  the selected profile if needed.
- `dd.cudd` missing: the image must install the Linux `dd` wheel with CUDD; the smoke test
  refuses to fall back to `dd.autoref`.
- Dataset not found: verify `/data/datasets/<name>/<name>_rf.pkl` and `BDSS_DATASETS`.
- A run times out: the timeout is a recorded status, not a missing row; use `--resume` to
  continue.  The results volume keeps the per-run JSON files.
- Node cap: `NODE_CAP` is checked **after** the build, so a peak above it is recorded as
  `memory`; it is not a streaming memory guard.  The container memory limit (4 GB per run) is
  the real guard, together with the 120 s internal timeout.
