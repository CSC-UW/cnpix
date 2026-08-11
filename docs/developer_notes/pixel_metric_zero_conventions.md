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
ultimately hand-set in the mua-bugnon tuner, so it had no remaining user. No
argmax consumer of these metrics remains in the tree; the caveat is kept because
the constraint applies to any that returns.)

Regression coverage: `tests/test_evaluation_metrics.py::test_complete_miss_scores_zero_not_nan`,
`::test_detector_returning_nothing_scores_zero_not_nan`,
`::test_no_ground_truth_leaves_recall_undefined`,
`::test_defined_zero_still_loses_to_any_positive_score`.

## Stale derived artifacts

The change alters only cells where `tp == 0`; every previously-defined value is
bit-identical. Scored tables written before 2026-08-11 therefore have stale *derived*
columns but valid `TP`/`FP`/`FN`/`TN`, so they can be refreshed by re-deriving from the
stored counts — no re-detection needed. Known affected file:

- `/Volumes/npx_nfs/nobak/offproj/novel_objects_deprivation/manual_vs_banded_and_bugnon_full48h_NREM.parquet`
  (source of the manuscript's detector-vs-manual agreement table). Counts re-verified
  2026-08-11: 134 rows, 7 with `nan` F1 and 5 with `nan` precision, and no `nan` in
  `sensitivity` or `IoU`. **Not yet refreshed in place** — the file still predates the
  fix (written 2026-07-13), so any mean taken from its `F1` or `precision` column is
  still the biased one.
