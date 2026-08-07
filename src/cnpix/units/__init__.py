"""Project-agnostic unit metrics and cell-type classification for CNPIX rats.

The public surface most analyses want:

>>> from cnpix.units import load_cell_types
>>> labels = load_cell_types(quality_tier="sua_moderate")

To (re)generate the underlying tables, use the ``cnpix-units`` CLI or
:mod:`cnpix.units.pipeline`.
"""

from cnpix.units.thresholds import (
    QUALITY_TIERS,
    get_threshold_kwargs,
    tiers_at_least,
)
from cnpix.units.celltypes import (
    CELL_TYPES,
    CLASSIFIABLE_REGIONS,
    NARROW_PEAK_TO_VALLEY_S,
    TAU_RISE_THRESHOLD_MS,
    assign_cell_types,
    classify_narrow_wide,
    classify_petersen,
)
from cnpix.units.readers import load_aggregated_cell_metrics, load_cell_types
from cnpix.units.regions import add_major_regions, hippocampus_to_waxholm
from cnpix.units.sortings import get_sortings, get_subjects

__all__ = [
    "CELL_TYPES",
    "CLASSIFIABLE_REGIONS",
    "NARROW_PEAK_TO_VALLEY_S",
    "QUALITY_TIERS",
    "TAU_RISE_THRESHOLD_MS",
    "add_major_regions",
    "assign_cell_types",
    "classify_narrow_wide",
    "classify_petersen",
    "get_sortings",
    "get_subjects",
    "get_threshold_kwargs",
    "hippocampus_to_waxholm",
    "load_aggregated_cell_metrics",
    "load_cell_types",
    "tiers_at_least",
]
