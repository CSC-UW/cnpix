"""Pixel- and event-level classification metrics for OFF-period labels.

Method-agnostic kernels comparing any predicted OFF label array against a
manual ("ground truth") OFF label array on the same image-stack grid. Both
inputs are instance-segmentation arrays of shape
``(n_chunks, n_rows, n_samples)`` where ``0`` is background and positive
integers are per-event instance IDs; the kernels binarize with ``> 0``.

These were extracted from ``offproj.bugnon.manual_validation`` so that every
detection method (mua-bugnon, sam3, harding, unit_based) can be scored against
the same manual labels. The ``predicted`` argument was historically named
``bugnon``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.ndimage


def compute_pixel_metrics(
    manual: np.ndarray,
    predicted: np.ndarray,
    chunks: np.ndarray,
    row_mask: np.ndarray,
) -> dict:
    """Compute pixel-level classification metrics.

    Parameters
    ----------
    manual
        Manual label array ``(n_chunks, n_rows, n_samples)``.
    predicted
        Predicted label array (same shape) from any detection method.
    chunks
        Indices of chunks to evaluate.
    row_mask
        Boolean mask of shape ``(n_rows,)`` indicating which rows to
        include.

    Returns
    -------
    dict
        Keys: ``TP``, ``FP``, ``FN``, ``TN``, ``sensitivity``,
        ``specificity``, ``precision``, ``F1``, ``IoU``.
    """
    rows = np.where(row_mask)[0]
    m = manual[np.ix_(chunks, rows)] > 0
    p = predicted[np.ix_(chunks, rows)] > 0

    tp = int((m & p).sum())
    fp = int((~m & p).sum())
    fn = int((m & ~p).sum())
    tn = int((~m & ~p).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else float("nan")
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "F1": f1,
        "IoU": iou,
    }


def compute_per_event_pixel_metrics(
    manual: np.ndarray,
    predicted: np.ndarray,
    chunks: np.ndarray,
    row_mask: np.ndarray,
) -> pd.DataFrame:
    """Compute pixel-level metrics for each manual event individually.

    For each manual event, finds all overlapping predicted labels and
    computes pixel metrics over the union of the manual event mask and
    those predicted label masks.

    Parameters
    ----------
    manual
        Manual label array ``(n_chunks, n_rows, n_samples)``.
    predicted
        Predicted label array (same shape).
    chunks
        Indices of chunks to evaluate.
    row_mask
        Boolean mask of shape ``(n_rows,)`` indicating which rows to
        include.

    Returns
    -------
    pd.DataFrame
        One row per manual event with columns: ``label``, ``TP``,
        ``FP``, ``FN``, ``sensitivity``, ``precision``, ``F1``, ``IoU``.
    """
    rows = np.where(row_mask)[0]
    m_sub = manual[np.ix_(chunks, rows)]
    p_sub = predicted[np.ix_(chunks, rows)]

    manual_labels_present = np.unique(m_sub)
    manual_labels_present = manual_labels_present[manual_labels_present > 0]

    records = []
    for lbl in manual_labels_present:
        m_mask = m_sub == lbl
        overlapping_p_labels = set(np.unique(p_sub[m_mask])) - {0}
        if overlapping_p_labels:
            p_mask = np.isin(p_sub, list(overlapping_p_labels))
        else:
            p_mask = np.zeros_like(m_mask)

        tp = int((m_mask & p_mask).sum())
        fp = int((~m_mask & p_mask).sum())
        fn = int((m_mask & ~p_mask).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else float("nan")
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

        records.append(
            {
                "label": int(lbl),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "sensitivity": sensitivity,
                "precision": precision,
                "F1": f1,
                "IoU": iou,
            }
        )

    return pd.DataFrame(records)


def compute_event_metrics(
    manual: np.ndarray,
    predicted: np.ndarray,
    chunks: np.ndarray,
    row_mask: np.ndarray,
    *,
    with_iou: bool = True,
) -> dict:
    """Compute event-level detection metrics.

    A manual event counts as detected if any predicted pixel overlaps it; a
    predicted event counts as matched if any manual pixel overlaps it. Detection
    counts are computed with single labeled reductions
    (``scipy.ndimage.maximum``) rather than a per-label full-array scan.

    Parameters
    ----------
    manual
        Manual label array ``(n_chunks, n_rows, n_samples)``.
    predicted
        Predicted label array (same shape).
    chunks
        Indices of chunks to evaluate.
    row_mask
        Boolean mask of shape ``(n_rows,)`` for row scope.
    with_iou
        If True (default) also compute per-manual-event soft IoU against the
        union of overlapping predicted events (a per-event loop — costly when
        there are many events). Set False to skip it (``per_event_ious`` empty)
        when only detection rates are needed.

    Returns
    -------
    dict
        Event-level metrics including ``n_manual_events``,
        ``n_predicted_events``, ``n_manual_detected``,
        ``n_predicted_unmatched``, ``event_sensitivity``,
        ``event_false_discovery_rate``, and ``per_event_ious``
        (numpy array; empty when ``with_iou`` is False).
    """
    rows = np.where(row_mask)[0]
    m_sub = manual[np.ix_(chunks, rows)]
    p_sub = predicted[np.ix_(chunks, rows)]

    m_binary = m_sub > 0
    p_binary = p_sub > 0

    manual_labels = np.unique(m_sub)
    manual_labels = manual_labels[manual_labels > 0]
    predicted_labels = np.unique(p_sub)
    predicted_labels = predicted_labels[predicted_labels > 0]
    n_manual = int(manual_labels.size)
    n_predicted = int(predicted_labels.size)

    # Detection counts: max of the other side's binary mask over each label's
    # pixels (>0 == overlaps). One labeled reduction each, vs a per-label scan.
    if n_manual:
        m_over = np.atleast_1d(
            scipy.ndimage.maximum(p_binary, labels=m_sub, index=manual_labels)
        )
        manual_detected = m_over > 0
    else:
        manual_detected = np.zeros(0, dtype=bool)
    n_manual_detected = int(manual_detected.sum())

    if n_predicted:
        p_over = np.atleast_1d(
            scipy.ndimage.maximum(m_binary, labels=p_sub, index=predicted_labels)
        )
        n_predicted_matched = int((p_over > 0).sum())
    else:
        n_predicted_matched = 0

    event_sensitivity = n_manual_detected / n_manual if n_manual > 0 else float("nan")
    event_fdr = (
        (n_predicted - n_predicted_matched) / n_predicted
        if n_predicted > 0
        else float("nan")
    )

    # Per-event soft IoU (manual event vs union of overlapping predicted events),
    # only for detected events (undetected -> 0.0). Optional: it is a per-event
    # loop and costly when there are many events.
    if with_iou and n_manual:
        manual_ious = []
        for i, lbl in enumerate(manual_labels):
            if not manual_detected[i]:
                manual_ious.append(0.0)
                continue
            m_mask = m_sub == lbl
            overlapping_p_labels = set(np.unique(p_sub[m_mask])) - {0}
            p_union_mask = np.isin(p_sub, list(overlapping_p_labels))
            intersection = (m_mask & p_union_mask).sum()
            union = (m_mask | p_union_mask).sum()
            manual_ious.append(intersection / union if union > 0 else 0.0)
        per_event_ious = np.array(manual_ious)
    else:
        per_event_ious = np.array([])

    return {
        "n_manual_events": n_manual,
        "n_predicted_events": n_predicted,
        "n_manual_detected": n_manual_detected,
        "n_predicted_unmatched": n_predicted - n_predicted_matched,
        "event_sensitivity": event_sensitivity,
        "event_false_discovery_rate": event_fdr,
        "per_event_ious": per_event_ious,
    }


def summarize_event_ious(ev: dict) -> dict:
    """Replace an event-metrics dict's ``per_event_ious`` array with summaries.

    Returns a flat copy with ``median_event_iou`` / ``mean_event_iou`` instead of
    the raw ``per_event_ious`` array, suitable for a DataFrame row.
    """
    ev = dict(ev)
    ious = ev.pop("per_event_ious")
    ev["median_event_iou"] = (
        float(np.median(ious)) if len(ious) > 0 else float("nan")
    )
    ev["mean_event_iou"] = float(np.mean(ious)) if len(ious) > 0 else float("nan")
    return ev
