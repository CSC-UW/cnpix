"""Open CSD, LFP, and hypnogram data sources lazily."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import xarray as xr

import ecephys.hypnogram as hyp
import wisc_ecephys_tools.rats.cnd_hgs as cnd_hgs

from cnpix import f25

from lfp_csd_viewer.data.xarray_source import XarrayDataSource

logger = logging.getLogger(__name__)


@dataclass
class ViewerSources:
    """Lazily-opened data sources for the viewer."""

    csd_source: XarrayDataSource
    lfp_source: XarrayDataSource
    hypnogram: hyp.FloatHypnogram | None


def open_viewer_sources(
    subject: str,
    experiment: str,
    kind: Literal["cortical", "hippocampal"],
    condition: str = "Full.Conservative",
) -> ViewerSources:
    """Open data sources lazily (no bulk data loaded).

    Args:
        subject: Subject name (e.g., "CNPIX12-Santiago").
        experiment: Experiment name
            (e.g., "novel_objects_deprivation").
        kind: Which probe region ("cortical" or "hippocampal").
        condition: Hypnogram condition name for sleep scoring.

    Returns:
        ViewerSources with lazy data sources and hypnogram.
    """
    params = f25.S3.load_experiment_subject_params(
        experiment, subject
    )
    probe = params["spwrProbe"]

    # --- Open CSD lazily ---
    csd_file = f25.NB.get_experiment_subject_file(
        experiment, subject, f"{kind}_kcsd.zarr"
    )
    logger.info("Opening CSD from %s ...", csd_file)
    csd_da = xr.open_dataarray(csd_file, engine="zarr", chunks={})
    csd_source = XarrayDataSource(csd_da)

    # --- Open LFP lazily ---
    logger.info("Opening LFP (%s) ...", kind)
    if kind == "cortical":
        lfp_da = f25.open_cortical_lfps(
            subject, experiment, chunks={}
        )
    else:
        lfp_da = f25.open_hippocampal_lfps(
            subject, experiment, chunks={}
        )
    lfp_source = XarrayDataSource(lfp_da)

    # --- Load hypnogram (small, stays in memory) ---
    hypnogram = None
    try:
        hgs = cnd_hgs.load_statistical_condition_hypnograms(
            subject, experiment, probe
        )
        if condition in hgs:
            hypnogram = hgs[condition]
            logger.info("Loaded hypnogram condition %r", condition)
        else:
            available = list(hgs.keys())
            logger.warning(
                "Condition %r not found. Available: %s",
                condition,
                available,
            )
    except Exception:
        logger.warning(
            "Could not load hypnograms for %s / %s",
            subject,
            experiment,
            exc_info=True,
        )

    return ViewerSources(
        csd_source=csd_source,
        lfp_source=lfp_source,
        hypnogram=hypnogram,
    )
