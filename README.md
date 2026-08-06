# Predicting geology along a horizontal wellbore

*[English](README.md) · [Tiếng Việt](README.vi.md)*

Solution for the ROGII Wellbore Geology Prediction competition: predict **TVT**
(true vertical thickness — the well's stratigraphic position inside the rock
column) for the un-logged section of a horizontal well.

**Result: 10.150 public / 9.402 private RMSE — 980 / 6191 (top 16%).**

| stage | private RMSE |
|---|---|
| carry-last baseline (`TVT = last known value`) | ~15.9 |
| first submission (particle filter) | 15.386 |
| GBDT stack on hand-built features | 13.565 |
| + structural-surface prior, DP tracker, leak guard | 12.022 |
| **final: two decorrelated trackers + stack + projection** | **9.402** |

---

## The one equation this is all built on

On every train well checked, exactly (to ±0.05 ft, which is the rounding of the
source data):

```
TVT  =  -Z  +  ANCC(x, y)  +  c_well
```

* `std(TVT + Z - ANCC)` per well: median **0.0066 ft**, max **0.0079 ft** (150 wells)
* the six formation columns are constant offsets of one another *within* a well:
  `std(ASTNU - ANCC)` ≈ **1e-13**

`Z` is given in the test set. So predicting TVT **is** predicting one structural
surface along the wellbore — nothing more. Every feature in `src/geosteer.py` is
a different estimator of that single quantity.

This reframing is what unlocked the run. It also immediately kills the obvious
mistake: summarising a well's formation depth by its **median** throws away the
whole signal, because the surface dips ~150 ft along a *single* lateral. A
constant-`ANCC` predictor scores 106–109 pooled against a 15.2 baseline.

## How hard can this get? Two oracle ladders

**Ladder 1 — how good can a per-well correction be?** (250 wells, oracle-fitted)

| predictor | pooled RMSE |
|---|---|
| carry-last baseline | 16.61 |
| oracle: best **constant** offset per well | **9.07** |
| oracle: best straight line per well | 6.59 |
| oracle: best degree-6 curve per well | 3.00 |

A *perfect* level estimate stops at 9.07. Anything below that must track
**shape**. This single table set the whole strategy: surface features supply
level, GR trackers supply shape.

**Ladder 2 — how few numbers describe a well?** Robust polynomial fit of the
stratigraphic level `U = TVT + Z` against normalised MD:

| degree | RMSE |
|---|---|
| 2 | 3.98 |
| 3 | 3.10 |
| 4 | 2.32 |
| 6 | 1.71 |

A **cubic in normalised MD describes an entire eval-zone trajectory to ~3 ft**.
The answer for a well is essentially four numbers. That is what the projection
layer in section 7 exploits.

---

## Architecture

```
                    ┌─ structural surface transfer ──┐
                    │   (neighbour wells, ~300 ft)   │
per-row features ───┼─ DP (TVT, dip) Viterbi ────────┼──► LGBM ×3 + CatBoost ×2
                    │   deterministic, global        │         │
                    ├─ likelihood-weighted PF ───────┤         ▼
                    │   stochastic, forward          │    Ridge stack
                    └─ GR misfit landscape ──────────┘         │
                        (11 offsets × 4 anchors)                ▼
                                                    blend with trackers
                                                            ▼
                                                    cubic projection of U
                                                            ▼
                                                    shrink to baseline
```

### Three estimators of the surface

**1. Structural surface transfer.** Neighbouring laterals sample the same
surface — median distance to the nearest other well is only **~300 ft**. For
neighbour *j*: `TVT_j(i) = -Z(i) + ANCC_j(nearest point of j) + o_j`, where
`o_j` is fitted on the target's known lateral. That anchor absorbs both the
datum difference *and* the near-constant cross-dip between two parallel
laterals. Neighbours are weighted by their error on a **held-out** slice of the
known zone, so the weights are honest.

Only one formation is used: all six collapse to the same array once `o_j`
absorbs the offset (measured spread **7.7e-5 ft**). The useful spread is across
*neighbours*, not formations.

**2. DP over (TVT, stratigraphic dip).** Viterbi with dip as an *explicit*
state, so the path is piecewise-linear in dip — a handful of segments per
lateral, which is how a human interpreter actually works. Beam search has no
such state, which is why the two are not redundant.

**3. Likelihood-weighted particle filter.** State is `U = TVT + Z` with a
random-walk rate; each of 16 seeds is scored by total log-likelihood and
combined by softmax at several temperatures. Vectorised over
(seed × particle) so it needs no numba.

### Why two GR trackers instead of one

Measured standalone, no model involved:

| | 481-well bank (45 wells) | 773-well bank (30 wells) |
|---|---|---|
| carry-last baseline | 16.168 | 18.384 |
| surface transfer | 18.117 | **14.110** |
| DP (TVT, dip) | 12.785 | 14.967 |
| likelihood-weighted PF | **12.031** | 15.817 |
| **50/50 DP + PF** | **10.587** | **12.415** |

Per-well error correlation between DP and PF: **0.095–0.258**. They fail in
completely different places:

| well | baseline | surf | DP | PF |
|---|---|---|---|---|
| fef8af96 | 44.6 | **4.0** | 18.6 | 49.5 |
| 6d6d93af | 32.2 | **6.1** | 14.3 | 54.7 |
| 197f8a5a | 24.6 | 48.5 | 45.9 | **5.1** |
| a8ed028a | 19.0 | 27.1 | 20.7 | **4.1** |

Averaging beats the best single tracker by ~1.5 ft. **Note the two columns
disagree about which tracker is best** — the surface prior needs dense
neighbours, so its quality depends on how many wells are in the bank. The
notebook therefore *sweeps* the mix weights rather than hardcoding them.

### GR misfit is handed over whole, not as an argmin

GR misfit at the true path is ~3× lower than at a wrong constant offset:

| offset δ (ft) | −20 | −10 | **0** | +10 | +20 |
|---|---|---|---|---|---|
| normalised misfit | 3.20 | 2.69 | **1.05** | 3.13 | 3.48 |

18 of 20 wells put the minimum at δ = 0. But the landscape is badly multi-modal
— constant-offset scans routinely find spurious minima at ±30 ft with cost near
the truth. So rather than committing to an argmin, features `tda*`, `tdbc*`,
`tddp*`, `tdsf*`, `tdpf*` give the model the misfit at 11 offsets around each
anchor and let it pick the mode.

---

## Results

Leave-one-well-out CV, 160 wells, GroupKFold, single sklearn HGB:

| step | pooled RMSE |
|---|---|
| carry-last baseline | 15.192 |
| feature set v1 | 13.312 |
| **+ lateral bug fixes** | **12.920** |
| + tracker blend + cubic projection + shrink | **12.161** |

Fold-to-fold spread is large (10.29 → 15.43) and fold 3 still loses to its own
baseline. Pooled RMSE is dominated by a handful of drift wells.

---

## Two bugs worth reading about

Both had the same root cause: **using the whole known zone instead of only the
lateral part of it.** The known zone contains the build section, where TVT
swings hundreds of feet.

| feature | before | after |
|---|---|---|
| `sc_ens_d` (NCC window match) | median **−92 ft**, 26% of rows beyond ±200, max 1155 | median 1.26, max 8.7 |
| `slp_b_d` (linear extrapolation) | median **+945 ft**, max 4268 | median −28.9, max 57.8 |

The NCC was matching eval GR windows against *build-section* windows; the slope
fit was reading dip off the build curve and extrapolating it 4000 ft. Fixing
both was worth **0.39 ft** pooled — more than most modelling changes.

Neither showed up in aggregate RMSE. What caught them was a feature-range audit
(notebook section 5): every tracker feature is a TVT delta and `TVT − last_known`
has p99 ≈ 56 ft and absmax ≈ 84 ft across the train set, so **anything past ±100
is meaningless by construction**. That check is now a permanent cell.

---

## Negative results

Documented because they cost real time and the reasons generalise.

### Per-well reliability gating does not work

Pooled RMSE is dominated by a few drift wells, so knowing *per well* how far to
trust the model looks like the obvious lever. The headroom is real:

| | pooled RMSE |
|---|---|
| model | 13.312 |
| + global shrink | 13.020 |
| oracle: pick model-or-baseline per well | 11.21 |
| oracle: optimal shrink per well | 10.00 |

But nothing observable predicts it. Correlation between the per-well optimal
shrink and every available statistic peaked at **0.148** (`beam_std_d`). A
corrected lateral-only backtest over 138 wells did no better (best |corr|
**0.164**).

The cause is structural. The known **lateral** is only 350–650 rows long
(measured: 345, 613, 650, 542, 352, 604), so at most ~250 rows can be masked —
about 250 ft against an eval zone of 4000–6000 ft. Over 250 ft the carry
baseline is already near-exact: mean self-check `r_base` was **2.33 ft** against
**15.19 ft** on the real eval zone. The backtest measures a completely different
regime and has no discriminating power. Not fixable by better design.

### A global Bayesian smoother loses to doing nothing

A forward-backward smoother over `U` scored **24.58** against a **16.16**
carry baseline, with the worst 10% of wells producing 56% of squared error.
Shrinking and clipping recovered it to 14.40, but its gain over the stack was
+0.05 ft — below seed noise. Same failure mode as an unconstrained DP: without a
strong prior the GR likelihood walks into spurious minima.

### The train/test overlap is not worth exploiting

The competition ships the three visible test wells inside `train/` as well, with
`TVT` and the formation columns filled in. Reference solutions reconstruct TVT
from formation contacts for those wells and report near-zero error.

**It is dead code on the scored set.** A probe submission adding +100 to every
row and +1000 to overridden wells returned **100.665** — if the scored wells were
the overridden ones the score would have been ~1100. The public leaderboard runs
on different wells.

This repo excludes it deliberately, and guards against it geometrically rather
than by well id:

```python
geometric_exclusions(h, bank, min_dist=50.0)
```

Any bank well whose trajectory sits within 50 ft of the target is dropped. The
separation is unambiguous — a well's own copy sits at ~1 ft, the nearest genuine
offset lateral at 250–360 ft, and nothing lands in between.

Also deliberately excluded: a global −0.40 ft bias term (decoded from that same
probe against the public leaderboard — a fit to public LB noise), and the probe
cell itself.

---

## Repository layout

```
notebooks/rogii_tvt_pipeline.ipynb   end-to-end, Kaggle-ready
src/geosteer.py                      feature builder (same code, importable)
experiments/                         the measurement scripts behind the tables
docs/FINDINGS.md                     condensed log of what was tried
```

`experiments/` scripts expect the competition data under `tr/train/`:

| script | what it measures |
|---|---|
| `cv_ablation.py` | GroupKFold CV and feature-family ablations |
| `tracker_comparison.py` | standalone tracker RMSE and combination rules |
| `row_thinning.py` | training-row thinning vs accuracy and fit time |
| `reliability_gate.py` | the lateral-only self-check (negative result) |

## Running the notebook

Needs a GPU for the CatBoost/LightGBM stack; falls back to CPU automatically.

1. `DRY_RUN_WELLS = 120` (default) → full pass in ~8 minutes. Check that
   section 5 prints `SUSPECT: none` and section 9 shows a low
   `corr(dp_err, pf_err)`.
2. Set `DRY_RUN_WELLS = 0` and **Save & Run All** (~3 h).

Kaggle re-runs the whole notebook on commit, so an interactive full run plus a
commit costs double quota — validate on the subset, then commit once.

Two tuning notes, both measured:

* **Row thinning is free.** Eval rows are 1 ft apart and highly autocorrelated.
  `THIN=1` → 12.097 (141 s fit); `THIN=3` → **11.913** (73 s); `THIN=6` → 12.135
  (33 s). Thinning by 3 is faster *and* slightly better.
* **16 PF seeds is enough.** 32 seeds cost 3.9 s/well against 2.2 s at 16, with
  no measurable accuracy gain.

## Where the remaining headroom is

Oracle ladder 2 says a cubic in normalised MD reaches 2.3–3.1 ft: a well's
trajectory is ~4 numbers. This solution predicts per-row TVT and projects onto
that basis *afterwards*. **Predicting the coefficients directly** — a per-well
regression with 4 outputs instead of a per-row regression with one — is the
obvious next experiment and was never tried.

## Acknowledgements

Two ideas were adapted from publicly shared competition notebooks: the
likelihood-weighted multi-seed particle filter, and the robust-polynomial
projection of `U = TVT + Z`. Their leak-exploitation and leaderboard-probing
layers were deliberately not carried over, for the reasons documented above.
