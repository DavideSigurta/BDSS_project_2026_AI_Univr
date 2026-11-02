# BDSS Project 2026 — AI Univr

Post-hoc logical analysis of 54 pre-trained random forests, for the first half of the BDSS
course. The project does not train or validate a model: it reads the trees, concentrates each
feature into cells, encodes the cell index in binary, composes the per-tree vote counters as
OBDDs, and answers exact queries (certified region size, bit rankings, budgeted decomposition,
robustness radius).

The final CSV deliverables are in `results/`. The PDF report is a separate presentation and
is not part of this repository.

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

## Final campaign

The study uses 3 budgets (`1k`, `10k`, `100k`) × five selectors (`modal`, `random`, `ig`,
`shapley`, `pvalue`) × 54 models, plus a build-completion phase for the 36 models for which the
grid does not close the full forest. The grid has 810 runs; build completion has 2,235 runs;
the clean campaign has 3,045 rows and 54 models with a completed build. The 18 models whose
full forest completes do not use a prefix; the other 36 use the prefix `k` that completes.

Entry points, all writing CSV in `results/`:

```bash
uv run python -m bdss.study m5final       # grid + descending prefixes
uv run python -m bdss.study m5readout     # counts, volumes, per-bit rankings
uv run python -m bdss.study m5robustness  # every test sample, every label
```

`m5final` covers the grid and the descending prefixes; the build-completion part is already
in the shipped `m5_final_campaign.csv`, which the two post commands consume. The five
delivered files are:

```text
results/m5_final_campaign.csv     clean grid + build-completion rows
results/m5_readouts.csv           one readout per selected build
results/m5_robustness_samples.csv one row per test sample
results/m5_robustness_targets.csv one row per sample and label
results/m5_post_coverage.csv      build/readout/robustness status per model
```

`m5readout` and `m5robustness` write these canonical 54-model CSVs only when the selection is
complete; `--limit` or a missing campaign is rejected before any file is touched. Partial runs
must use a different output path. The Modal commands and resource settings are in
[docs/modal.md](docs/modal.md).

### Data semantics

- Readout `counts_json`, `volumes_json` and `rankings_json` use class indices `0..n-1` as keys;
  the `labels` column gives the class names in that order.
- Counts and volumes are exact over the **certified region**. Volumes are fractions of the
  input space and need not sum to 1 across labels: 35 of the 54 selected builds have sums
  below 1 because part of the space is uncertified.
- `own_label` is the hard-vote winner, not a certificate. A blank `own_label` means a
  hard-vote tie; in that case `own_radius` and `loss` are undefined while `min_rival` is
  still computed. `certified` and `own_radius` carry the certificate. On the final data:
  3,663 tie samples, 11 samples with a hard-vote label at radius 1–2, 34,276 certified and
  3,674 non-certified samples.
- A blank target `radius` means `Win_l = ⊥`: the label cannot win, so the distance is
  infinite. On the final data these are 1,300 rows (wine-quality-white label 9 and
  wine-quality-red label 3); they are not missing measurements.
- The robustness files contain every test row: 37,950 samples and 282,654 sample × label
  target rows. For each target the CSV gives the targeted radius, the nearest rival and its
  label, the loss of certification, and the witness as input bits → original feature →
  destination cell.

p-values use the exact hypergeometric tail up to 16 input bits and a declared normal tail
above that. The label map is validated against `test_acc` for all 54 bundles; the input order
is fixed per model (features used nearer a root first, then frequency) and CUDD dynamic
reordering is disabled.
