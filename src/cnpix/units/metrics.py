"""Assemble per-unit metric tables and assign each unit its quality tier."""

import numpy as np
import pandas as pd
import xarray as xr
from ecephys.units import siutils as units_siutils
from ecephys.wne import siutils as wne_siutils

from cnpix.units import files, regions, sortings
from cnpix.units.thresholds import QUALITY_TIERS, get_threshold_kwargs

__all__ = [
    "ID_COLS",
    "load_acgs",
    "load_narrow_and_wide_acgs",
    "aggregate_acg_metrics",
    "collect_sorting_properties",
    "assign_cluster_quality",
]

ID_COLS: list[str] = ["subject", "experiment", "probe", "cluster_id"]


def load_acgs(
    subject: str,
    kind: str,
    experiment: str = sortings.DEFAULT_EXPERIMENT,
    normalize: bool = True,
) -> xr.DataArray:
    """Load a subject's stored ACGs of the given kind ("narrow" or "wide")."""
    from cnpix.units.acg import ACG_PARAMS, normalize_acgs

    if kind not in ACG_PARAMS:
        raise ValueError(f"Unrecognized correlogram kind: {kind!r}")
    fname = {"narrow": files.Files.NARROW_ACGS, "wide": files.Files.WIDE_ACGS}[kind]
    acgs = xr.load_dataarray(files.get_subject_file(experiment, subject, fname))
    return normalize_acgs(acgs) if normalize else acgs


def load_narrow_and_wide_acgs(
    subject: str,
    experiment: str = sortings.DEFAULT_EXPERIMENT,
    normalize: bool = True,
    states: list[str] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    narrow = load_acgs(subject, "narrow", experiment, normalize)
    wide = load_acgs(subject, "wide", experiment, normalize)
    if states is not None:
        narrow = narrow.sel(state=states)
        wide = wide.sel(state=states)
    if not np.array_equal(narrow.cluster_id.values, wide.cluster_id.values):
        raise ValueError(
            f"{subject}: cluster IDs differ between narrow and wide autocorrelograms"
        )
    return narrow, wide


def aggregate_acg_metrics(
    subjects: list[str], experiment: str = sortings.DEFAULT_EXPERIMENT
) -> pd.DataFrame:
    """Concatenate the per-subject ACG metric tables into one cohort table."""
    frames = []
    for subject in subjects:
        path = files.get_subject_file(experiment, subject, files.Files.ACG_METRICS)
        if not path.exists():
            raise FileNotFoundError(
                f"No ACG metrics for {subject} at {path}. Run the acg-metrics step first."
            )
        frame = pd.read_parquet(path)
        frame["subject"] = subject
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def collect_sorting_properties(
    subjects: list[str],
    experiment: str = sortings.DEFAULT_EXPERIMENT,
    quality_tier: str = "mua",
) -> pd.DataFrame:
    """Per-unit sorting properties (quality metrics, waveform metrics, anatomy).

    Anatomical acronyms are collapsed onto Waxholm names and mapped to major
    regions, which is what gates cell-type classification downstream.
    """
    frames = []
    for subject in subjects:
        mps = sortings.load_multiprobe_sorting(subject, experiment, quality_tier)
        frames.append(mps.properties)
    props = pd.concat(frames, ignore_index=True)
    props["acronym"] = props["acronym"].apply(regions.hippocampus_to_waxholm)
    return regions.add_major_regions(props, as_booleans=False)


def assign_cluster_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Record, per unit, the strictest quality tier it passes.

    The tiers are nested, so iterating from most permissive to most restrictive
    and overwriting leaves the strictest passing tier in ``max_quality``. Units
    passing none of them keep NaN.
    """
    missing = set(ID_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[ID_COLS].drop_duplicates(ignore_index=True)
    out["max_quality"] = pd.Series(np.nan, index=out.index, dtype=object)
    index = pd.MultiIndex.from_frame(out[ID_COLS])

    for tier in QUALITY_TIERS:
        simple_filters, callable_filters = wne_siutils.get_quality_metric_filters(
            **get_threshold_kwargs()[tier]
        )
        passing = units_siutils.refine_clusters(
            df, simple_filters, callable_filters, include_nans=True, verbose=False
        )
        passing_index = pd.MultiIndex.from_frame(
            passing[ID_COLS].drop_duplicates(ignore_index=True)
        )
        out.loc[index.isin(passing_index), "max_quality"] = tier

    return out
