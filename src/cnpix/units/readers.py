"""Consumer-facing loaders for the shared cell-type and unit-metric tables.

These are what downstream analyses should use. Generating the tables is the job
of :mod:`cnpix.units.pipeline`.
"""

import pandas as pd

from cnpix.units import files
from cnpix.units.thresholds import tiers_at_least

__all__ = [
    "load_cell_types",
    "load_cluster_quality",
    "load_aggregated_cell_metrics",
]

_ID_COLS = ["subject", "experiment", "probe", "cluster_id"]

_MISSING_MSG = (
    "{path} not found. Generate it with `cnpix-units run-all` "
    "(see cnpix.units.pipeline)."
)


def _read(fname: str) -> pd.DataFrame:
    path = files.get_cohort_file(fname)
    if not path.exists():
        raise FileNotFoundError(_MISSING_MSG.format(path=path))
    return pd.read_parquet(path)


def load_cluster_quality() -> pd.DataFrame:
    """Per-unit strictest passing quality tier (``max_quality``)."""
    return _read(files.Files.CLUSTER_QUALITY)


def load_cell_types(quality_tier: str | None = None) -> pd.DataFrame:
    """Per-unit cell-type labels, optionally restricted by unit quality.

    Parameters
    ----------
    quality_tier
        If given, keep only units whose recorded ``max_quality`` is this tier or
        stricter (e.g. ``"sua_moderate"`` also admits ``"sua_conservative"``).

    Returns
    -------
    pd.DataFrame
        Columns: the unit identifiers, ``si_cluster_id``,
        ``narrow_wide_cell_type``, ``petersen_cell_type``, and — when filtering —
        ``max_quality``. Units in unclassifiable regions carry NaN labels.

        Two unit ids are present and they are NOT interchangeable.
        ``cluster_id`` is the multiprobe id used internally here (MultiSIKS
        offsets each probe past the first by 1e6). ``si_cluster_id`` is the raw
        per-probe Kilosort id. **Join on ``si_cluster_id``** if your sorting was
        loaded one probe at a time.
    """
    labels = _read(files.Files.CELL_TYPES)
    if quality_tier is None:
        return labels
    quality = load_cluster_quality()
    merged = labels.merge(quality, on=_ID_COLS, how="left", validate="one_to_one")
    return merged[merged["max_quality"].isin(tiers_at_least(quality_tier))].reset_index(
        drop=True
    )


def load_aggregated_cell_metrics(quality_tier: str | None = None) -> pd.DataFrame:
    """The full per-(unit, state) metric table, optionally quality-filtered."""
    df = _read(files.Files.AGGREGATED_CELL_METRICS)
    if quality_tier is None:
        return df
    quality = load_cluster_quality()
    merged = df.merge(quality, on=_ID_COLS, how="left")
    return merged[merged["max_quality"].isin(tiers_at_least(quality_tier))].reset_index(
        drop=True
    )
