"""Map Waxholm atlas acronyms onto major brain regions.

Used to decide which cell-type criteria apply to a unit: the Petersen scheme is
defined for cortex and hippocampus only (see :mod:`cnpix.units.celltypes`).
"""

from functools import lru_cache

import numpy as np
import pandas as pd

__all__ = [
    "MAJOR_REGIONS",
    "add_major_regions",
    "hippocampus_to_waxholm",
]

# Region labels emitted into the `region` column, in the order tested.
MAJOR_REGIONS: tuple[str, ...] = ("hippocampal", "thalamic", "cortical", "other")

_ATLAS_NAME = "whs_sd_rat_39um"


@lru_cache(maxsize=1)
def _get_atlas():
    import brainglobe_atlasapi

    return brainglobe_atlasapi.BrainGlobeAtlas(_ATLAS_NAME, check_latest=False)


@lru_cache(maxsize=1)
def _region_acronym_sets() -> dict[str, frozenset[str]]:
    """Acronyms belonging to each atlas super-structure, including the root itself."""
    atlas = _get_atlas()
    return {
        key: frozenset(atlas.get_structure_descendants(key) + [key])
        for key in ("HF", "Thal-D", "Cx")
    }


def hippocampus_to_waxholm(acronym: str) -> str:
    """Collapse subfield-qualified hippocampal acronyms onto Waxholm names.

    Anatomy tables carry labels like ``CA1so`` (stratum oriens); the atlas knows
    only ``CA1``. Non-hippocampal acronyms pass through unchanged.
    """
    for subfield in ("CA1", "CA2", "CA3", "DG"):
        if subfield in acronym:
            return subfield
    return acronym


def add_major_regions(df: pd.DataFrame, as_booleans: bool = True) -> pd.DataFrame:
    """Add major-region information derived from the ``acronym`` column.

    Parameters
    ----------
    df
        Must have an ``acronym`` column of Waxholm atlas region names. Pass it
        through :func:`hippocampus_to_waxholm` first if it may carry subfield
        qualifiers.
    as_booleans
        If True, add ``is_hippocampal`` / ``is_thalamic`` / ``is_cortical`` /
        ``is_other`` columns. If False, add a single ``region`` column.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the region information added.
    """
    sets = _region_acronym_sets()
    out = df.copy()

    is_hippocampal = out["acronym"].isin(sets["HF"])
    is_thalamic = out["acronym"].isin(sets["Thal-D"])
    # A handful of hippocampal acronyms are also Cx descendants in Waxholm, so
    # hippocampus wins the tie -- matching the original findlay2025a behavior.
    is_cortical = ~is_hippocampal & out["acronym"].isin(sets["Cx"])
    is_other = ~is_hippocampal & ~is_thalamic & ~is_cortical

    if as_booleans:
        out["is_hippocampal"] = is_hippocampal
        out["is_thalamic"] = is_thalamic
        out["is_cortical"] = is_cortical
        out["is_other"] = is_other
        return out

    flags = np.column_stack([is_hippocampal, is_thalamic, is_cortical, is_other])
    if not (flags.sum(axis=1) == 1).all():
        raise ValueError(
            "Major region flags are not mutually exclusive; cannot assign a "
            "single `region` label."
        )
    out["region"] = np.select(
        [is_hippocampal, is_thalamic, is_cortical], MAJOR_REGIONS[:3], default="other"
    )
    return out
