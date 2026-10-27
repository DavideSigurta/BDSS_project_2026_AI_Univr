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

## Study

Local and optional Modal CPU runs are described in [docs/modal.md](docs/modal.md); the
campaign uses the same code in both cases.

Nine entry points, all writing CSV in `results/`:

```bash
uv run python -m bdss.study            # bundle validation and prefix readouts
uv run python -m bdss.study frontier   # budgeted frontier over the configured grid
uv run python -m bdss.study robustness # per-sample radii and witnesses
uv run python -m bdss.study pilot      # six-model campaign pilot
uv run python -m bdss.study m5         # local campaign (3 budgets by default)
uv run python -m bdss.study m5rescue   # re-run failed rows at 300 s
uv run python -m bdss.study m5merge    # merge base + rescue for the post
uv run python -m bdss.study m5post     # readouts/robustness from completed builds
uv run python -m bdss.study m5coverage # regenerate post coverage only
```

The label map is validated against `test_acc` for all 54 bundles; the input order is fixed per
model (features used nearer a root first, then frequency) and CUDD dynamic reordering is
disabled. Counts and volumes are exact; p-values use the exact hypergeometric tail up to 16
input bits and a declared normal tail above that.
