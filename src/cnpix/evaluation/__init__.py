"""Method-agnostic evaluation of OFF-period detections against manual labels.

Manual ("ground truth") OFF labels are shared infrastructure: every detection
method — morphological, SAM3, harding, unit-based — is scored against the same
annotated image stacks, and two papers analyzing these rats would be
embarrassed to disagree about what a hit is. So the ground truth, the label QC,
and the metric kernels live here rather than in any one project package.

``config``
    The evaluation conditions (NREM / Wake) and the per-condition
    image-selection convention.
``paths``
    Hive-partitioned label paths (manual ground truth and model predictions).
``labels``
    Manual label loaders, instance-label QC, grid reconciliation, chunk
    selection.
``metrics``
    Pixel- and event-level metric kernels.

Method-specific drivers stay with their methods: ``offproj.evaluation`` keeps
the Bugnon driver and the stack-grid/rasterizer that read offproj detection
outputs, and ``samoffs`` scores SAM3 predictions.
"""

from cnpix.evaluation import config, labels, metrics, paths
from cnpix.evaluation.labels import (
    load_manual_labels,
    qc_and_fix_labels,
    reconcile_to_common_grid,
    select_chunks,
)
from cnpix.evaluation.metrics import (
    compute_event_metrics,
    compute_per_event_pixel_metrics,
    compute_pixel_metrics,
    metrics_from_counts,
    summarize_event_ious,
)

__all__ = [
    "config",
    "labels",
    "metrics",
    "paths",
    "load_manual_labels",
    "qc_and_fix_labels",
    "reconcile_to_common_grid",
    "select_chunks",
    "compute_event_metrics",
    "compute_per_event_pixel_metrics",
    "compute_pixel_metrics",
    "metrics_from_counts",
    "summarize_event_ious",
]
