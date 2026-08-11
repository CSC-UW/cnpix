"""SpikeInterface lazy preprocessing chain for the MUA amplitude envelope.

Builds a lazy preprocessing chain from raw SpikeGLX recordings that produces
a rectified, resampled (500 Hz) MUA amplitude envelope. No data is saved to
disk — the chain is evaluated on-demand when get_traces() is called.

Design choices, all deliberate:
    - No z-scoring: preserves raw amplitudes for wake baseline subtraction
    - Bandpass to 5 kHz (not 12 kHz): matches Harding et al. (2023)
    - sp.resample instead of sp.decimate: proper anti-aliasing
    - Highpass spatial filter for CSD-like spatial denoising
    - No Gaussian smoothing in chain: applied post-extraction at two scales

The chain is method-agnostic. Every OFF-detection method that starts from a
MUA envelope reads the ``mua_traces.zarr`` this produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.preprocessing as sp

if TYPE_CHECKING:
    from ecephys import wne

__all__ = ["build_preprocessing_chain", "get_channel_subset"]


def build_preprocessing_chain(
    recording: si.BaseRecording,
    bandpass_freq_min: int = 300,
    bandpass_freq_max: int = 5000,
    resample_rate: int = 500,
    detect_bad_channels: bool = True,
    highpass_spatial_filter_enabled: bool = True,
    apply_motion_correction: bool = True,
    sglx_subject: wne.sglx.SGLXSubject | None = None,
    probe: str | None = None,
    slices: pd.DataFrame | None = None,
) -> si.BaseRecording:
    """Build a lazy SpikeInterface preprocessing chain for MUA envelope.

    The chain produces a rectified, anti-aliased, resampled signal at the target
    rate. Both Gaussian smoothing windows are applied in scipy after extracting
    data from this chain, to avoid recomputing the shared preprocessing 3 times.

    Processing order:
        depth_order -> bandpass_filter -> phase_shift ->
        detect/interpolate bad channels -> highpass_spatial_filter ->
        motion_correction (optional) -> rectify -> resample

    Args:
        recording: Raw SpikeGLX recording.
        bandpass_freq_min: Lower bandpass cutoff in Hz.
        bandpass_freq_max: Upper bandpass cutoff in Hz (paper uses 5000).
        resample_rate: Target sampling rate in Hz (paper uses ~498, we use
            500).
        detect_bad_channels: Whether to detect and interpolate bad
            channels. Can be disabled for faster iteration during testing.
        highpass_spatial_filter_enabled: Whether to apply highpass spatial filtering.
            Be aware that this changes per-channel amplitude distributions,
            making each channel's GMM classifier dependent on which channels
            were included in the reference/window. So use this when processing a whole
            probe, but not when processing structures separately.
        apply_motion_correction: Whether to apply legacy motion correction.
            Requires sglx_subject, probe, and slices arguments.
        sglx_subject: SGLXSubject (needed if apply_motion_correction is
            True).
        probe: Probe name (needed if apply_motion_correction is True).
        slices: Slice table (needed if apply_motion_correction is True).

    Returns:
        Lazy SpikeInterface recording at resample_rate Hz, representing the
        rectified MUA amplitude envelope.
    """
    rec = sp.depth_order(recording)
    rec = sp.bandpass_filter(
        rec,
        freq_min=bandpass_freq_min,
        freq_max=bandpass_freq_max,
    )
    rec = sp.phase_shift(rec)

    if detect_bad_channels:
        bad_channel_ids, _ = sp.detect_bad_channels(rec)
        rec = sp.interpolate_bad_channels(rec, bad_channel_ids)

    if highpass_spatial_filter_enabled:
        rec = sp.highpass_spatial_filter(rec)

    if apply_motion_correction:
        if sglx_subject is None or probe is None or slices is None:
            raise ValueError(
                "sglx_subject, probe, and slices are required for motion correction"
            )
        from cnpix.mua.motion import apply_legacy_motion_correction

        rec = apply_legacy_motion_correction(sglx_subject, probe, rec, slices)

    rec = sp.rectify(rec)
    rec = sp.resample(rec, resample_rate=resample_rate, gap_tolerance_ms=0.0)

    return rec

def get_channel_subset(
    recording: si.BaseRecording,
    y_lo: float | None,
    y_hi: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify channels within a depth range.

    Args:
        recording: Depth-ordered SpikeInterface recording.
        y_lo: Lower depth bound in microns, or None for no bound.
        y_hi: Upper depth bound in microns, or None for no bound.

    Returns:
        Tuple of ``(channel_ids, y_coords, mask)`` where mask is a
        boolean array over all channels.
    """
    all_y = recording.get_channel_locations()[:, 1]
    mask = np.ones(len(all_y), dtype=bool)
    if y_lo is not None:
        mask &= all_y >= y_lo
    if y_hi is not None:
        mask &= all_y <= y_hi
    return recording.get_channel_ids()[mask], all_y[mask], mask
