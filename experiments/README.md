# Experiment scripts

These are the measurement scripts behind the tables in the root README and
`docs/FINDINGS.md`. They are research code, kept as-run rather than polished.

They expect the competition data under `tr/train/` relative to the working
directory:

```
tr/train/<wellid>__horizontal_well.csv
tr/train/<wellid>__typewell.csv
```

and `src/geosteer.py` importable (copy it in, or run from the repo root with
`PYTHONPATH=src`).

| script | measures | produces |
|---|---|---|
| `tracker_comparison.py A B` | standalone RMSE of surface / DP / PF and of several combination rules, on wells `A:B` of a fixed sample | `cmp3.csv` |
| `cv_ablation.py FOLD [nosurf\|nodp\|noland]` | one GroupKFold fold, optionally dropping a feature family | `oof_<tag>_<fold>.npz` |
| `row_thinning.py STEP FOLD` | accuracy and fit time when training rows are thinned by `STEP` | stdout |
| `reliability_gate.py A B` | the lateral-only self-check — mask the last 250 lateral rows, re-run trackers, score there | `sc2.csv` |

Most take a well range so they can be run in chunks under a time limit, e.g.

```bash
python experiments/tracker_comparison.py 0 45
python experiments/cv_ablation.py 0 full
python experiments/row_thinning.py 3 0
```

Building the feature matrices first (`ds_*.npz`) is a prerequisite for
`cv_ablation.py` and `row_thinning.py`; see `build_well` in `src/geosteer.py`.
