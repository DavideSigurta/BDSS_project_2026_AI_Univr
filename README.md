# BDSS Project 2026 — AI Univr

Post-hoc logical analysis of 54 pre-trained random forests, for the first half of the BDSS
course. The project does not train or validate a model: it reads the trees, concentrates each
feature into cells, encodes the cell index in binary, composes the per-tree vote counters as
OBDDs, and answers exact queries (certified region size, bit rankings, budgeted decomposition,
robustness radius).

The numbers live in `results/` (CSV and figures). The PDF report is a separate presentation
and is not part of this repository.

## Setup

Python 3.12 with `uv`:

```bash
uv sync
```

The `.pkl` bundles were saved with scikit-learn 1.4.2; the pinned dependencies are in
`pyproject.toml`.

### Fast BDD backend (CUDD)

`dd==0.6.0` ships CUDD only in its Linux wheels. On macOS the plain `uv sync` installs the
pure-Python `dd.autoref`; to build the CUDD extension as well:

```bash
DD_CUDD=1 DD_FETCH=1 uv sync --no-binary-package dd --no-cache --reinstall-package dd
```

`bdss.bdd` selects `dd.cudd` when it imports and falls back to `dd.autoref` otherwise, so the
pipeline runs either way.

## Datasets

The datasets are expected at `../datasets/<name>/` with `<name>_rf.pkl`, `<name>_train.csv`,
`<name>_test.csv` and the full `<name>.csv`. Override the location with
`BDSS_DATASETS=/path/to/datasets`.

The model's class index is the `pandas.factorize` code of the full target column, so the label
map is the order of first appearance in `<name>.csv` (not the sorted order of the train
labels). `bdss.forest.load_bundle` reads it once and `validate_bundle` checks it against the
bundle's `test_acc`; the soft vote is reproduced from the trees, without
`RandomForestClassifier.predict`.

## Tests

```bash
uv run pytest            # fast tests
uv run pytest -m slow    # real bundles, prefix builds, the predict gate
```

## Toy example

```bash
uv run python examples/toy_forest.py
```

Writes `results/toy_{cells,codes,counts,rankings,frontier,robustness}.csv`.

## M2 study

```bash
uv run python -m bdss.study
```

Validates the 54 bundles, then builds the prefixes `banknote`/5, `banknote`/10,
`vehicle`/5, `ann-thyroid`/5 and `ann-thyroid`/10 with no budget. The input order is fixed
per prefix (features used nearer a root first, then frequency) and the build disables CUDD's
dynamic reordering. `vehicle`/10 is recorded as a timeout: the un-budgeted accumulator does
not complete, which is an M2 result and the reason the budgeted frontier is M3.

Writes `results/m2_models.csv`, `results/m2_builds.csv`, `results/m2_counts.csv` and
`results/m2_rankings.csv`. Counts and volumes are exact; p-values use the exact
hypergeometric tail for at most 16 input bits and the declared normal tail above that.
