"""Putative cell-type classification from waveform width and ACG kinetics.

Implements the scheme of Petersen et al. (CellExplorer): a unit with a narrow
spike is a putative fast-spiking interneuron; among broad-spiking units, a slow
ACG rise separates putative wide interneurons from putative pyramidal cells.

The scheme is defined for cortex and hippocampus only. Units in other regions
(thalamus, claustrum, white matter, ...) are deliberately left unlabeled rather
than forced into a cortical taxonomy.
"""

import numpy as np
import pandas as pd

__all__ = [
    "NARROW_PEAK_TO_VALLEY_S",
    "TAU_RISE_THRESHOLD_MS",
    "CLASSIFIABLE_REGIONS",
    "CELL_TYPES",
    "classify_narrow_wide",
    "classify_petersen",
    "assign_cell_types",
]

# Waveform peak-to-valley duration at or below which a unit is "narrow", in
# SECONDS (template metrics are stored in seconds).
NARROW_PEAK_TO_VALLEY_S: float = 0.000425

# ACG rise-time threshold separating wide interneurons from pyramidal cells,
# in MILLISECONDS. Region-specific, per CellExplorer.
TAU_RISE_THRESHOLD_MS: dict[str, float] = {"cortical": 6.0, "hippocampal": 3.0}

# Only these regions get cell-type labels.
CLASSIFIABLE_REGIONS: tuple[str, ...] = tuple(TAU_RISE_THRESHOLD_MS)

CELL_TYPES: tuple[str, ...] = ("narrow interneuron", "wide interneuron", "pyramidal")

# Classification is performed on this state's metrics and copied to the others,
# so that a unit carries one identity across the recording.
CLASSIFICATION_STATE: str = "NREM"

_ID_COLS: tuple[str, ...] = ("subject", "experiment", "probe", "cluster_id")


def classify_narrow_wide(peak_to_valley: float) -> str | float:
    """Split units into ``narrow`` / ``wide`` on waveform duration.

    Returns NaN when the waveform metric is missing, rather than silently
    assigning a class.
    """
    if not np.isfinite(peak_to_valley):
        return np.nan
    return "narrow" if peak_to_valley <= NARROW_PEAK_TO_VALLEY_S else "wide"


def classify_petersen(peak_to_valley: float, tau_rise: float, region: str) -> str | float:
    """Assign a putative Petersen cell type.

    Parameters
    ----------
    peak_to_valley
        Waveform peak-to-valley duration, in seconds.
    tau_rise
        ACG rise time constant, in milliseconds.
    region
        One of :data:`CLASSIFIABLE_REGIONS`.

    Returns
    -------
    str or NaN
        NaN if the region is unclassifiable or a required metric is missing.
    """
    if region not in TAU_RISE_THRESHOLD_MS:
        return np.nan
    width = classify_narrow_wide(peak_to_valley)
    if width is np.nan or (isinstance(width, float) and np.isnan(width)):
        return np.nan
    if width == "narrow":
        # tau_rise is irrelevant for narrow units, so a failed ACG fit does not
        # cost us the label.
        return "narrow interneuron"
    if not np.isfinite(tau_rise):
        return np.nan
    return "wide interneuron" if tau_rise > TAU_RISE_THRESHOLD_MS[region] else "pyramidal"


def _broadcast_to_all_states(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Copy per-unit labels computed in one state to every state of that unit."""
    id_cols = list(_ID_COLS)
    labels = (
        df.loc[df["state"] == CLASSIFICATION_STATE, id_cols + [column]]
        .dropna(subset=[column])
        .drop_duplicates(subset=id_cols)
    )
    out = df.drop(columns=[column]).merge(labels, on=id_cols, how="left")
    return out


def assign_cell_types(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``narrow_wide_cell_type`` and ``petersen_cell_type`` columns.

    Expects one row per (unit, state) with columns ``state``, ``region``,
    ``peak_to_valley``, ``tau_rise``, plus the unit identifier columns. Labels
    are derived from the :data:`CLASSIFICATION_STATE` rows and broadcast to all
    states of the same unit.
    """
    missing = {"state", "region", "peak_to_valley", "tau_rise", *_ID_COLS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.drop(
        columns=["narrow_wide_cell_type", "petersen_cell_type", "cluster_cell_type"],
        errors="ignore",
    ).copy()

    classifiable = out["state"].eq(CLASSIFICATION_STATE) & out["region"].isin(
        CLASSIFIABLE_REGIONS
    )
    rows = out.loc[classifiable]

    out["narrow_wide_cell_type"] = pd.Series(np.nan, index=out.index, dtype=object)
    out.loc[classifiable, "narrow_wide_cell_type"] = [
        classify_narrow_wide(v) for v in rows["peak_to_valley"]
    ]

    out["petersen_cell_type"] = pd.Series(np.nan, index=out.index, dtype=object)
    out.loc[classifiable, "petersen_cell_type"] = [
        classify_petersen(ptv, tau, reg)
        for ptv, tau, reg in zip(
            rows["peak_to_valley"], rows["tau_rise"], rows["region"], strict=True
        )
    ]

    out = _broadcast_to_all_states(out, "narrow_wide_cell_type")
    out = _broadcast_to_all_states(out, "petersen_cell_type")
    return out
