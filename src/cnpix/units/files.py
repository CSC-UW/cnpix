"""Output paths for the CNPIX unit-metrics and cell-type pipeline.

Everything lives under the project-agnostic ``cnpix`` project so that any
analysis can consume one shared set of unit metrics and cell-type labels,
rather than each paper recomputing its own.

Per-subject intermediates live in the experiment-subject directory; the
cohort-level tables live at the project root.
"""

from pathlib import Path

import wisc_ecephys_tools as wet

__all__ = [
    "PROJECT",
    "Files",
    "get_project",
    "get_subject_file",
    "get_cohort_file",
]

PROJECT = "cnpix"


class Files:
    """Canonical filenames produced by this pipeline."""

    NARROW_ACGS = "narrow_autocorrelograms.zarr"
    WIDE_ACGS = "wide_autocorrelograms.zarr"

    # Per-subject and cohort-level ACG metric tables share a name; they are
    # distinguished by directory.
    ACG_METRICS = "acg_metrics.pqt"

    # Cohort-level only.
    MPS_METRICS = "mps_metrics.pqt"
    AGGREGATED_CELL_METRICS = "aggregated_cell_metrics.pqt"
    CLUSTER_QUALITY = "cluster_quality.pqt"
    CELL_TYPES = "cell_types.pqt"


def get_project():
    return wet.get_sglx_project(PROJECT)


def get_subject_file(experiment: str, subject: str, fname: str) -> Path:
    """Path to a per-subject intermediate."""
    return get_project().get_experiment_subject_file(experiment, subject, fname)


def get_cohort_file(fname: str) -> Path:
    """Path to a cohort-level table at the project root."""
    return get_project().get_project_file(fname)
