# M5 final campaign on Modal

The campaign is executable locally without Modal. This page documents the optional CPU run
used for the final results: the same `bdss.campaign` code is wrapped by
`src/bdss/modal_app.py`, the only module that imports `modal`.

## Profile and volumes

The profile is chosen by the user; no name is hardcoded. Export `MODAL_PROFILE`:

```bash
export MODAL_PROFILE=<profile>
uv run --extra modal modal profile list
uv run --extra modal modal volume list --profile "$MODAL_PROFILE"
```

Two volumes must exist in the profile:

```bash
uv run --extra modal modal volume create bdss-datasets --profile "$MODAL_PROFILE"
uv run --extra modal modal volume create bdss-results  --profile "$MODAL_PROFILE"
uv run --extra modal modal volume put bdss-datasets ../datasets / --profile "$MODAL_PROFILE"
```

`bdss-datasets` is mounted at `/data`; the app sets `BDSS_DATASETS=/data/datasets`.
`bdss-results` is mounted at `/results` and stores one JSON file per trial.

## Smoke

```bash
uv run --extra modal modal run -m bdss.modal_app --smoke
```

`banknote/5` with `modal`; it fails if `dd.cudd` is not importable. A final-grid smoke is
available with `--final-smoke` (one 8M/16GB grid run and one post).

## Final grid

3 budgets (1k/10k/100k) × 5 selectors × 54 models = **810 grid runs**.  Timeout 300 s,
8M nodes, 16,384 MB, 16 containers, CUDD.

```bash
uv run --extra modal modal run -m bdss.modal_app --final \
  --max-containers 16 --max-usd 30 --resume
```

The raw grid file is `m5_final.csv`; it is an intermediate.  The deliverable is the clean
`m5_final_campaign.csv` after the build-completion phase below.

## Build completion

The grid leaves the models without a completed full-forest row.  The build phase reads the
raw grid, skips every model with a completed `k = full` row, and searches the remaining
models with:

- 3 budgets × 5 selectors (`modal`, `random`, `infogain`, `shapley`, `pvalue`);
- k = 5, 10, 20, 30, 50 (only `k < full`), ascending; stop at the first failure;
- bisection between the last success (`k_ok=0` sentinel) and the first failure; full is a
  known failure and is used only as the upper bound, never rerun;
- no cap.

Resources: 300 s, **4M nodes**, **8 GB**, 16 containers.  The node cap is a post-build
status: a peak above 4M is `memory`, not a build.  No completed row of the final grid
exceeded 365k nodes, so the smaller cap does not drop a completed build.

```bash
uv run --extra modal modal run -m bdss.modal_app --build \
  --max-containers 16 --max-usd 30
```

Output is `m5_build.csv` (`phase=prefix`, `source=build`, `node_cap=4000000`).  The clean
merge drops the old two-selector prefix rows:

```bash
uv run python -m bdss.study m5buildmerge \
  --final results/m5_final.csv --build results/m5_build.csv \
  --out results/m5_final_campaign.csv
```

The clean file has one row per `(phase, dataset, k, budget, selector)` and no raw prefix
rows from the early two-selector search.

## Final post

The post reads `m5_final_campaign.csv`, selects the largest completed k per model (ties:
100k/modal, then smaller peak/leaves), and runs the whole test set with 1800 s, 8M nodes,
16 GB, 16 containers:

```bash
uv run --extra modal modal run -m bdss.modal_app --final-post \
  --max-containers 16 --max-usd 30
```

Outputs:

- `m5_final_post.csv` — one selected build per model;
- `m5_final_post_samples.csv` — one row per test sample;
- `m5_final_post_coverage.csv` — build/post status per model.

The run records a timeout as a status; it does not impute missing samples.

## Output files

The final deliverable is:

```text
results/m5_final_campaign.csv       810 grid + build-completion rows; 54 models
results/m5_final_post.csv           54 selected builds; 22 completed / 32 timeout
results/m5_final_post_samples.csv   full-test sample rows for the 22 completed
results/m5_final_post_coverage.csv  54 build rows; post status per model
```

The raw `m5_final.csv` and `m5_build.csv` are intermediate files and are not part of the
final repo state.

## Cost

At the Modal Starter rates in `bdss/config.py` (CPU 0.047 USD/core-hour, memory
0.008 USD/GiB-hour), the final grid + build + post is below the 30 USD hard cap.  The
`--max-usd` guard is a hard stop; more containers reduce wall time, not cost.

## Troubleshooting

- `volume list` fails: check the profile and authenticate it.
- `dd.cudd` missing: the smoke refuses to fall back to `dd.autoref`.
- Dataset not found: check `/data/datasets/<name>/<name>_rf.pkl` and `BDSS_DATASETS`.
- A timeout is a recorded row, not a missing run; re-run with `--resume` for the grid.
- Node cap: checked after the build; a peak above the cap is `memory`, not a streaming guard.
