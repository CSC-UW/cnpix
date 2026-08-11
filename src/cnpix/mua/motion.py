"""Legacy AP-band motion correction for CNPIX recordings.

The motion vectors were estimated by the legacy sorting pipeline and are stored
alongside each subject's sorting. Applying them is dataset business rather than
paper business: any analysis of these recordings that cares about drift wants
the same vectors.

``lnsp`` (the legacy sorting pipeline) is an optional dependency, imported only
when correction is actually requested::

    uv add "cnpix[motion]"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.sortingcomponents.motion as scm
import wisc_ecephys_tools as wet

from cnpix import constants

if TYPE_CHECKING:
    from ecephys import wne

__all__ = ["apply_legacy_motion_correction"]

_LNSP_MISSING = (
    "Legacy motion correction requires the `lnsp` package "
    "(legacy_npix_sorting_pipeline), which is an optional dependency. "
    'Install it with `uv add "cnpix[motion]"`.'
)


def apply_legacy_motion_correction(
    sglx_subject: wne.sglx.SGLXSubject,
    probe: str,
    recording: si.BaseRecording,
    slices: pd.DataFrame,
    experiment: str = constants.DEFAULT_EXPERIMENT,
) -> si.BaseRecording:
    """Interpolate a recording onto the legacy sorting's motion estimate.

    Returns the recording unchanged if the subject has no legacy motion vector.
    """
    try:
        import lnsp.compat
        from lnsp.sorting_pipeline import SpikeInterfaceSortingPipeline
    except ImportError as err:  # pragma: no cover - depends on install extras
        raise ImportError(_LNSP_MISSING) from err

    s3 = wet.get_sglx_project("shared")
    sorting_pipeline = SpikeInterfaceSortingPipeline.load_from_folder(
        s3,
        sglx_subject,
        experiment,
        "full",
        probe,
        "sorting",
        rerun_existing=False,
    )
    clean_motion_path = sorting_pipeline.clean_motion_path
    if not clean_motion_path.exists():
        print("Subject has no legacy motion vector to apply. Skipping correction.")
        return recording

    _, old_slices = sorting_pipeline.get_raw_si_recording()
    assert slices.equals(old_slices), (
        "Mismatch between legacy motion slice table and provided slice table."
    )

    motion_npz = np.load(clean_motion_path)
    motion, _, _ = lnsp.compat.convert_legacy_motion_npz(motion_npz)
    return scm.InterpolateMotionRecording(
        recording,
        motion,
        border_mode="remove_channels",
        spatial_interpolation_method="nearest",
        sigma_um=20.0,
        p=1,
        num_closest=3,
        dtype="float32",
    )
