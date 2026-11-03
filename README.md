# BDSS Project 2026 — AI Univr

Exact post-hoc analysis of 54 pre-trained random forests for the first half of the BDSS
course. The project does not train or validate a classifier: it reads the trees, turns each
feature into cells, encodes the cell index in binary, composes the per-tree vote counters as
OBDDs, and answers exact queries on the resulting diagram.

The final artifacts are in `results/`.

## What the pipeline does

1. **Cells** — each feature is split into intervals using the thresholds of the forest; a
   sample is mapped to one cell per feature.
2. **Bits** — each cell index is encoded as a fixed number of binary input bits.
3. **Composition** — each tree becomes a small OBDD; counters compose the trees and compute
   the winning label for every input assignment.
4. **Budgeted frontier** — the full forest may be too large. A node budget `B`, a split
   policy and five selectors produce a frontier of certified leaves (or a prefix of trees).
5. **Readouts** — exact certified counts and volumes per label, plus per-bit information
   gain, Shapley and p-value rankings.
6. **Robustness** — for every test sample and every label: targeted radius to make that
   label win, nearest rival, loss of certification, and a witness in input bits, original
   features and destination cells.

The implementation is in `src/bdss/`: `forest`, `discretize`, `compose`, `frontier`,
`readouts`, `robustness`, `campaign`, `post`, `order`, `runtime`, plus the local `study` CLI
and the optional `modal_app`.

## Repository layout

```text
src/bdss/        pipeline, campaign, post-build, local and Modal entry points
tests/           fast tests and independent oracles; slow tests are opt-in
results/         the five delivered CSVs, nothing else
docs/modal.md    optional Modal execution details
```

The provided dataset bundles are included in `datasets/`; the expected layout is
`datasets/<name>/` with:

```text
<name>_rf.pkl     pre-trained forest
<name>_train.csv  training rows
<name>_test.csv   test rows
<name>.csv        full rows, used for the label order
```

Set `BDSS_DATASETS=/path/to/datasets` to use an external copy instead.

## Setup

Python 3.12 and [`uv`](https://docs.astral.sh/uv/):

```bash
cd path/to/repository
uv sync
```

`uv.lock` pins the resolved versions; keep it in the repository so the environment is reproducible.

The bundles were saved with scikit-learn 1.4.2; the pinned dependencies are in
`pyproject.toml`. The pure-Python `dd.autoref` backend works everywhere. For the faster CUDD
backend, on macOS:

```bash
DD_CUDD=1 DD_FETCH=1 uv sync --no-binary-package dd --no-cache --reinstall-package dd
```

On Linux the same variables work if a C compiler and `make` are installed. `bdss.bdd` uses
`dd.cudd` when importable and falls back to the pure-Python `dd.autoref`; local commands
work on either backend. CUDD is optional and is not needed for the demo or the fast suite.
The Modal image explicitly requires CUDD.

## Quick start

A one-model local demonstration, useful for checking the environment or showing the project:

```bash
cd path/to/repository
uv run python -m bdss.study demo --dataset iris --samples 3
```

It prints the selected build, the certified counts and the first test samples with their
own radius, nearest rival and witness. It is read-only: it never writes a canonical CSV
and does not use Modal.

## Command reference

All local commands run from `repo/` and write under `results/` by default. They consume the
clean `results/campaign.csv`; the five canonical files are protected against partial
overwrites.

### `final` — build the campaign

```bash
uv run python -m bdss.study final
```

Runs the complete campaign:

- grid: 54 models × 3 budgets (`1k`, `10k`, `100k`) × 5 selectors (`modal`, `random`, `ig`,
  `shapley`, `pvalue`) = **810 runs**;
- build completion: for the 36 models whose full forest does not close, 3 budgets × 5
  selectors are searched trial by trial, including descending prefixes
  (**2,235 measured runs**);
- joins the rows and writes the clean **`results/campaign.csv`** (**3,045 rows**).

It resumes an existing clean file and skips completed grid/build pairs, so re-running it on
the shipped campaign is a no-op. Useful flags:

```text
--out PATH       destination (default results/campaign.csv)
--jobs N         parallel local child processes
--timeout S      per-run wall clock (default 300 s)
--node-cap N     peak-node threshold recorded in status (post-build check, not a streaming cap)
--limit N        run only the first N grid specs; refused on the canonical file
```

A full local campaign is expensive. The shipped CSVs are the result of the completed runs;
`final` exists to reproduce them, not as a prerequisite for reading the results.

### `readout` — counts, volumes and rankings

```bash
uv run python -m bdss.study readout --resume
```

For the selected build of every model (largest completed `k`; ties prefer `100k/modal`), it
writes one row with:

- `counts_json`: exact count of input assignments certified for each label;
- `volumes_json`: the same count divided by `2^n_bits`;
- `rankings_json`: per-label information gain, Shapley and p-value per input bit, with the
  four rank permutations.

It does not read the test set. The canonical output is **`results/readouts.csv`**
(**54 rows**). Flags: `--rows`, `--out`, `--timeout`, `--jobs`, `--limit`,
`--resume`. A partial selection is refused when the destination is canonical; use `--out`
for an experiment.

### `robustness` — every test sample, every label

```bash
uv run python -m bdss.study robustness
```

For every selected build, every test row and every label it writes:

- **`results/robustness_samples.csv`** — one row per sample: own label, certification,
  own radius, nearest-rival distance and label, loss of certification, and the witness on
  the nearest boundary (**37,950 rows**);
- **`results/robustness_targets.csv`** — one row per sample × label: targeted radius,
  nearest/min-rival flag, witness bits → original features → destination cells
  (**282,654 rows**);
- **`results/post_coverage.csv`** — one row per model with build/readout/robustness status
  (**54 rows, all completed**).

The run is chunked and resumable. `--limit-s` controls the time slice of one worker;
`--chunks`, `--samples`, `--targets`, `--coverage`, `--readouts` override paths, and
`--limit` is refused for canonical outputs. The local run is CPU-heavy; the shipped target
file is the result of the completed run, not a sample.

### `demo` — the fast, read-only demonstration

```bash
uv run python -m bdss.study demo --dataset iris --samples 3
```

One selected model, one readout, the first few robustness samples, and the witness. No
canonical writes, no Modal, typically seconds. This is the command to use for a quick
walkthrough at the oral.

## Final results

The five delivered files are:

```text
results/campaign.csv            3,045 rows: 810 grid + 2,235 build-completion trials
results/readouts.csv            54 rows, all completed
results/robustness_samples.csv  37,950 rows, every test sample
results/robustness_targets.csv  282,654 rows, every sample × label
results/post_coverage.csv       54 rows, build/readout/robustness completed
```

Key facts:

- **54 models**, one selected build per model; all selected builds are at budget `100k`
  (53 `modal`, 1 `pvalue`).
- **18 models** compile the full forest; **36** use a prefix. The prefix is used only where
  the full forest does not complete.
- **54/54 readouts** completed, **54/54 models** have full-test robustness, **37,950/37,950**
  test samples and **282,654** sample × label rows.
- The final campaign has **3,045 rows**: 810 grid + 2,235 build-completion rows.

## Data semantics and limitations

- Readout JSON keys are class indices `0..n-1`; `labels` gives the class names in that order.
- `own_label` is the **hard-vote winner**, not a certificate. `certified` and `own_radius`
  carry the certificate.
- **3,663** samples have a hard-vote tie: `own_label`, `own_radius` and `loss` are blank,
  while `min_rival` is still computed.
- **11** non-certified samples have a hard-vote label at radius 1–2 and `loss=0.0`; they are
  real boundary cases, not missing rows.
- **1,300** target rows have a blank radius because `Win_l = ⊥` (the label cannot win, so
  the distance is infinite). They are not missing measurements.
- Counts and volumes are exact over the **certified region**. Volumes are fractions of the
  input space and need not sum to 1: 35/54 builds leave part of the space uncertified.
- The robustness distance is defined on the hard-vote/certified diagram, not on sklearn
  probabilities. The hard-vote winner can differ from sklearn's probability argmax; this is
  documented, not silently corrected.
- p-values use the exact hypergeometric tail up to 16 input bits and a normal tail above.
  CUDD dynamic reordering is disabled and the input order is fixed per model.

## Tests

```bash
uv run pytest          # fast suite
uv run pytest -m slow  # real dataset bundles and environment
```

The fast suite reports **83 passed, 1 skipped, 2 deselected** with the base install
(the skipped test is the optional Modal import); with `uv sync --extra modal` it reports
**86 passed, 2 deselected**. It covers the toy oracles, witness construction, counts against
the campaign, canonical guards, final artifacts, local build completion and the demo.
CUDD prints “Still N nodes referenced upon shutdown” at interpreter exit; this is a known
library warning and does not affect the result. `dd.autoref` teardown assertions are filtered
by `bdss.bdd`, because they are the same harmless shutdown artefact.

## Modal (optional)

Modal is an **alternative execution surface**, not a hidden prerequisite. The same code is
called from a remote image; local and Modal write the same clean files. Setup and resource
profiles are in [`docs/modal.md`](docs/modal.md).

```bash
# campaign: grid + build completion (prefix search included), same output as local final
uv run --extra modal modal run -m bdss.modal_app --campaign

# readout and robustness, resumable
uv run --extra modal modal run -m bdss.modal_app --readout --resume
uv run --extra modal modal run -m bdss.modal_app --robustness --limit-s 600

# image/dataset smoke
uv run --extra modal modal run -m bdss.modal_app --demo
```

The app uses the `bdss` Modal app name and two Volumes: `bdss-datasets` mounted at
`/data` and containing `datasets/`, and `bdss-results` mounted at `/results`. The app sets
`BDSS_DATASETS=/data/datasets`. The final CSVs are already in `results/`; Modal is only
needed to recompute a full campaign or post pass. `--limit` is valid only for `--readout`
and `--robustness`; `--campaign --limit` is rejected.

Before the first remote run:

```bash
modal setup
modal volume create bdss-datasets
modal volume create bdss-results
modal volume put bdss-datasets datasets /datasets
```

The dataset volume must contain its files under `/datasets/`; the app mounts the volume at
`/data` and reads `/data/datasets/`.

## Expected cost of a full recomputation

The shipped results required about **28 CPU-hours** for readout and **209 CPU-hours** of
robustness worker time (including model loading); the per-sample computation alone is
about **171 CPU-hours**. These are compute-hours, not wall-clock: the actual wall-clock was
longer because of preemption and retries. A full campaign is comparably expensive. This is why the five CSVs
are delivered. For a quick check use `demo`; for a full recomputation use the local
`final`/`readout`/`robustness` chain or the Modal entrypoints. The CSVs are the deliverable.
