"""End-to-end pipeline: spike trains -> ACGs -> metrics -> cell types.

Run the steps in order. Steps 1-2 are per-subject and expensive; steps 3-4 are
cohort-level and cheap.

1. :func:`compute_subject_acgs`         -> narrow/wide autocorrelogram zarrs
2. :func:`compute_subject_acg_metrics`  -> per-subject acg_metrics.pqt
3. :func:`build_cohort_tables`          -> aggregated_cell_metrics / mps_metrics / cluster_quality
4. :func:`assign_cohort_cell_types`     -> cell_types.pqt
"""

import logging
import shutil

import pandas as pd

from cnpix.units import acg, celltypes, files, metrics, sortings

__all__ = [
    "compute_subject_acgs",
    "compute_subject_acg_metrics",
    "build_cohort_tables",
    "assign_cohort_cell_types",
    "run_all",
]

logger = logging.getLogger(__name__)


def compute_subject_acgs(
    subject: str,
    experiment: str = sortings.DEFAULT_EXPERIMENT,
    quality_tier: str = "mua",
    overwrite: bool = False,
) -> None:
    """Compute and store a subject's narrow and wide autocorrelograms."""
    targets = {
        "narrow": files.get_subject_file(experiment, subject, files.Files.NARROW_ACGS),
        "wide": files.get_subject_file(experiment, subject, files.Files.WIDE_ACGS),
    }
    if not overwrite and all(p.exists() for p in targets.values()):
        logger.info("%s: ACGs already exist, skipping", subject)
        return

    hg = sortings.load_full_conservative_hypnogram(subject, experiment)
    mps = sortings.load_multiprobe_sorting(subject, experiment, quality_tier)
    trains = mps.get_cluster_trains()

    for kind, path in targets.items():
        acgs = acg.compute_acgs(trains, hg, kind)
        acgs.encoding["chunks"] = acgs.shape
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.rmtree(path)
        acgs.to_zarr(path)
        logger.info("%s: wrote %s ACGs -> %s", subject, kind, path)


def compute_subject_acg_metrics(
    subject: str, experiment: str = sortings.DEFAULT_EXPERIMENT
) -> pd.DataFrame:
    """Fit a subject's ACGs and derive the per-(cluster, state) metric table."""
    narrow, wide = metrics.load_narrow_and_wide_acgs(subject, experiment)
    table = acg.compute_acg_metrics(narrow, wide)

    path = files.get_subject_file(experiment, subject, files.Files.ACG_METRICS)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)
    logger.info("%s: wrote ACG metrics -> %s", subject, path)
    return table


def build_cohort_tables(
    subjects: list[str] | None = None,
    experiment: str = sortings.DEFAULT_EXPERIMENT,
    quality_tier: str = "mua",
) -> pd.DataFrame:
    """Join ACG metrics to sorting properties and assign quality tiers."""
    if subjects is None:
        subjects = sortings.get_subjects(experiment)

    acg_metrics = metrics.aggregate_acg_metrics(subjects, experiment)
    acg_metrics.to_parquet(files.get_cohort_file(files.Files.ACG_METRICS))

    props = metrics.collect_sorting_properties(subjects, experiment, quality_tier)
    props.to_parquet(files.get_cohort_file(files.Files.MPS_METRICS))

    aggregated = pd.merge(
        acg_metrics,
        props,
        on=["subject", "cluster_id"],
        suffixes=("_acg", None),
        how="left",
    )
    aggregated.to_parquet(files.get_cohort_file(files.Files.AGGREGATED_CELL_METRICS))

    quality = metrics.assign_cluster_quality(aggregated)
    quality.to_parquet(files.get_cohort_file(files.Files.CLUSTER_QUALITY))

    logger.info("Wrote cohort tables for %d subjects", len(subjects))
    return aggregated


def assign_cohort_cell_types(
    experiment: str = sortings.DEFAULT_EXPERIMENT,
) -> pd.DataFrame:
    """Classify cell types and write both the label table and the merged metrics."""
    del experiment  # cohort tables are experiment-agnostic at the project root
    aggregated = pd.read_parquet(
        files.get_cohort_file(files.Files.AGGREGATED_CELL_METRICS)
    )
    aggregated = celltypes.assign_cell_types(aggregated)
    aggregated.to_parquet(files.get_cohort_file(files.Files.AGGREGATED_CELL_METRICS))

    # `cluster_id` is the MULTIPROBE id: MultiSIKS offsets each probe beyond the
    # first by 1e6. Consumers that load a single probe (e.g. offproj) see the raw
    # per-probe Kilosort id, so carry `si_cluster_id` too or their join silently
    # drops every unit on imec1+.
    label_cols = ["narrow_wide_cell_type", "petersen_cell_type"]
    if "si_cluster_id" in aggregated.columns:
        label_cols = ["si_cluster_id"] + label_cols
    labels = aggregated.loc[
        aggregated["state"] == celltypes.CLASSIFICATION_STATE,
        metrics.ID_COLS + label_cols,
    ].reset_index(drop=True)
    labels.to_parquet(files.get_cohort_file(files.Files.CELL_TYPES))

    logger.info("Wrote cell types for %d units", len(labels))
    return labels


def run_all(
    subjects: list[str] | None = None,
    experiment: str = sortings.DEFAULT_EXPERIMENT,
    overwrite: bool = False,
) -> None:
    """Run every step, for every subject with a usable sorting."""
    if subjects is None:
        subjects = sortings.get_subjects(experiment)
    for subject in subjects:
        compute_subject_acgs(subject, experiment, overwrite=overwrite)
        compute_subject_acg_metrics(subject, experiment)
    build_cohort_tables(subjects, experiment)
    assign_cohort_cell_types(experiment)
