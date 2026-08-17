---
title: Zero vs. undefined in metrics_from_counts
status: active
updated: 2026-08-11
---

# Zero vs. undefined in `metrics_from_counts`

`cnpix.evaluation.metrics.metrics_from_counts` backs every OFF-label scorer in the
workspace (`offproj.bugnon.manual_validation`, `offproj.evaluation.bugnon_eval`,
`offproj.unit_based.banded_eval`, `offproj.unit_based.head_to_head`,
`samoffs.model_eval`) — all five verified 2026-08-11. None of those modules take a
cross-structure mean themselves — they return tidy per-structure frames and the
averaging happens downstream in notebooks. That is what made the bug below invisible.

## What was wrong

The original kernel returned `nan` for any ratio with a zero denominator:

```python
precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
f1 = (2 * precision * sensitivity / (precision + sensitivity)
      if (precision + sensitivity) > 0 else float("nan"))
```

Two distinct situations collapsed onto `nan`:

1. **`tp == 0, fp > 0`** — the detector fired but overlapped nothing. Precision and
   sensitivity are both *defined zeros*, so F1 is 0, not undefined. The guard returned
   `nan` anyway, contradicting the kernel's own documented contract ("undefined stays
   distinguishable from defined and zero").
2. **`tp == 0, fp == 0`** — the detector emitted nothing. Precision was `nan`; F1
   inherited it only incidentally, via `nan > 0` being `False`.

Both are detector failures with known correct answers whenever `fn > 0`. Because
`pandas` `.mean()` and `.count()` skip `nan` per column, each downstream mean silently
used a different, self-selected denominator — and the excluded structures were never a
random subset, they were exactly the failures. Every affected mean was biased
optimistically, most for the most conservative detectors.

## Current contract

`nan` is reserved for genuinely undefined cases:

| Case | sensitivity | precision | F1 | IoU |
|---|---|---|---|---|
| `tp+fp+fn == 0` (nothing labeled, nothing detected) | `nan` | `nan` | `nan` | `nan` |
| `tp+fn == 0` (no ground truth to recall) | `nan` | 0 if `fp>0` | `nan` | 0 |
| `tp == 0`, labels present | 0 | 0 | **0** | 0 |

Per-structure argmax callers are unaffected: a defined 0 still loses to any positive
score. The convention is safe for argmax and unsafe for averaging; the kernel cannot
tell which the caller is doing, so it now emits the honest value in both.

(The worked example here was `offproj.bugnon.threshold_sweep`, which picked a
per-structure quantile by `idxmax` on IoU. That module was deleted on 2026-08-11 —
it swept candidate quantiles against the manual labels, but the thresholds were
ultimately hand-set in the morphological tuner, so it had no remaining user. No
argmax consumer of these metrics remains in the tree; the caveat is kept because
the constraint applies to any that returns.)

Regression coverage: `tests/test_evaluation_metrics.py::test_complete_miss_scores_zero_not_nan`,
`::test_detector_returning_nothing_scores_zero_not_nan`,
`::test_no_ground_truth_leaves_recall_undefined`,
`::test_defined_zero_still_loses_to_any_positive_score`.

## Stale derived artifacts

The change alters only cells where `tp == 0`; every previously-defined value is
bit-identical. Scored tables written before 2026-08-11 therefore had stale *derived*
columns but valid `TP`/`FP`/`FN`/`TN`, so they were refreshed by re-deriving from the
stored counts — no re-detection needed.

**All four affected tables under
`/Volumes/npx_nfs/nobak/offproj/novel_objects_deprivation/` were refreshed in place on
2026-08-11**, each with a `.parquet.bak-2026-08-11` alongside (that tree is `nobak`):

| file | rows | cells fixed |
|---|---:|---:|
| `manual_vs_banded_and_bugnon_full48h_NREM.parquet` (manuscript Table 1) | 134 | 12 |
| `manual_vs_bugnon.parquet` | 162 | 22 |
| `manual_vs_morphological.parquet` | 168 | 34 |
| `manual_vs_banded_NREM.parquet` | 47 | 2 |

`manual_vs_morphological_full48h.parquet` (174 rows) had no affected cell and was left
alone. Verified after writing: `TP`/`FP`/`FN`/`TN` unchanged, every altered cell was
`nan` before and is finite now, no well-defined value moved, column order preserved.

The refresh procedure, should another such table turn up:

```python
rec = pd.DataFrame([metrics_from_counts(int(r.TP), int(r.FP), int(r.FN), tn=int(r.TN))
                    for r in d.itertuples()])
# assert the already-defined rows are bit-identical BEFORE writing, then:
for c in ["sensitivity", "specificity", "precision", "F1", "IoU"]:
    d[c] = rec[c].to_numpy()
```

`threshold_sweep_{blas,llas}.parquet` (2,900 and 2,700 rows; 542 and 38 affected) were
**deleted** rather than refreshed — zero referencing files, and their producer
`offproj.bugnon.threshold_sweep` was removed the same day. Note they would not have
been wrong for their actual use: `nan` loses an `argmax`, so threshold *selection* was
never biased. Only means were.

### What this does not fix

Numbers already *quoted* from these tables were computed under the old convention and
do not update themselves. For the Table 1 source, under the grouping its own notebook
uses (`batch_manual_vs_banded_and_bugnon.ipynb`, cell 10):

| row | F1 old → new | precision old → new |
|---|---|---|
| `banded-fixed_tiled` | 0.4825 → **0.4492** | 0.4444 → 0.4290 |
| `banded-greedy_fr` | 0.6664 → **0.6294** | 0.6950 → 0.6950 |
| `mua-llas` | 0.6242 → 0.6242 | 0.7313 → 0.7313 |
| `mua-clas` | 0.6234 → **0.5804** | 0.8173 → **0.7609** |
| `mua-blas` | 0.5886 → **0.5480** | 0.8667 → **0.8070** |

**`mua-llas` is the one row that does not move** — it had no affected cell. Every other
row's F1 drops 0.033–0.043, and `clas`/`blas` precision drops ~0.06.

`batch_manual_vs_banded_and_bugnon.ipynb` still carries **stored outputs computed under
the old convention** (cell 10 takes `.mean()` over `F1` directly). Its `df` comes from
`head_to_head.head_to_head_experiment(...)`, which re-derives rather than reading the
parquet, so re-running the notebook regenerates correct values from the fixed kernel —
but it re-detects and needs NFS. Until it is re-run, trust the refreshed parquet over
the notebook's displayed figures.
