"""Contract tests for the shared confusion-matrix helper.

``metrics_from_counts`` backs both pixel-metric kernels here and the
unit-based agreement analysis in ``offproj``, so its two contracts are
worth pinning: *genuinely* undefined ratios are ``nan`` (not 0) while a
detector's outright misses score a defined 0, and omitting ``tn`` drops the
negative-class columns entirely rather than reporting them as 0.
"""

import math

from cnpix.evaluation.metrics import metrics_from_counts


def test_derived_ratios():
    m = metrics_from_counts(tp=5, fp=2, fn=3, tn=90)
    assert m["sensitivity"] == 5 / 8
    assert m["precision"] == 5 / 7
    assert m["specificity"] == 90 / 92
    assert m["IoU"] == 0.5
    assert m["F1"] == 2 * (5 / 7) * (5 / 8) / ((5 / 7) + (5 / 8))


def test_zero_denominators_are_nan_not_zero():
    # An empty comparison is "undefined", not "perfectly wrong" -- averaging
    # these across conditions must skip them, which only works if they're nan.
    m = metrics_from_counts(tp=0, fp=0, fn=0, tn=0)
    for key in ("sensitivity", "specificity", "precision", "F1", "IoU"):
        assert math.isnan(m[key]), key


def test_complete_miss_scores_zero_not_nan():
    # Regression: F1 used to be nan whenever precision + sensitivity == 0, even
    # though both were *defined* zeros. A detector that fired 9484 pixels with
    # zero overlap has scored 0, not "undefined"; dropping it from a mean
    # silently rewarded the detector for its worst structure.
    m = metrics_from_counts(tp=0, fp=9484, fn=2486, tn=100)
    assert m["precision"] == 0.0
    assert m["sensitivity"] == 0.0
    assert m["F1"] == 0.0
    assert m["IoU"] == 0.0


def test_detector_returning_nothing_scores_zero_not_nan():
    # Regression: precision was nan when the detector emitted no pixels at all,
    # so the structure vanished from the precision mean while still counting 0
    # toward sensitivity -- the same miss penalised one column and was
    # invisible in the other.
    m = metrics_from_counts(tp=0, fp=0, fn=165318, tn=100)
    assert m["precision"] == 0.0
    assert m["F1"] == 0.0
    assert m["sensitivity"] == 0.0
    assert m["IoU"] == 0.0


def test_no_ground_truth_leaves_recall_undefined():
    # Nothing to recall: sensitivity and F1 stay nan even though the detector
    # fired. Precision is still defined and 0 -- every pixel was a false alarm.
    m = metrics_from_counts(tp=0, fp=17, fn=0, tn=100)
    assert math.isnan(m["sensitivity"])
    assert math.isnan(m["F1"])
    assert m["precision"] == 0.0
    assert m["IoU"] == 0.0


def test_defined_zero_still_loses_to_any_positive_score():
    # threshold_sweep.py selects per structure with idxmax, which skips nan.
    # Scoring misses as 0 rather than nan must not make a failed threshold
    # selectable over a working one.
    miss = metrics_from_counts(tp=0, fp=500, fn=500, tn=100)
    hit = metrics_from_counts(tp=1, fp=500, fn=500, tn=100)
    assert miss["F1"] < hit["F1"]


def test_tn_omitted_drops_negative_class_columns():
    m = metrics_from_counts(tp=5, fp=2, fn=3)
    assert "TN" not in m
    assert "specificity" not in m
    # The positive-class metrics are unaffected by the omission.
    full = metrics_from_counts(tp=5, fp=2, fn=3, tn=90)
    for key in ("TP", "FP", "FN", "sensitivity", "precision", "F1", "IoU"):
        assert m[key] == full[key], key


def test_column_order_is_stable():
    # Callers build DataFrames straight from this dict; column order is part
    # of the contract.
    assert list(metrics_from_counts(1, 1, 1, 1)) == [
        "TP", "FP", "FN", "TN",
        "sensitivity", "specificity", "precision", "F1", "IoU",
    ]
    assert list(metrics_from_counts(1, 1, 1)) == [
        "TP", "FP", "FN", "sensitivity", "precision", "F1", "IoU",
    ]
