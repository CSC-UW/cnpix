"""Lazy xarray data source for zarr-backed DataArrays."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceMetadata:
    """Metadata extracted from a lazily-opened xarray DataArray."""

    n_time_samples: int
    n_channels: int
    sampling_rate: float
    t_start: float
    t_end: float
    total_duration: float
    depths: np.ndarray  # (n_channels,)


class XarrayDataSource:
    """Lazy wrapper around an xarray DataArray backed by zarr.

    Opens the DataArray with dask chunking so that no bulk data is
    loaded into memory. Provides methods to read arbitrary time slices
    on demand.

    The DataArray is expected to have dimensions ``(time, ...)`` with a
    sorted ``time`` coordinate (in seconds) and a ``y`` coordinate
    giving channel depths.
    """

    def __init__(self, da: xr.DataArray):
        self._da = da
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> SourceMetadata:
        """Extract metadata without loading bulk data."""
        n_time = self._da.sizes["time"]

        # Channel dimension is whichever isn't "time".
        other_dims = [d for d in self._da.dims if d != "time"]
        n_channels = (
            self._da.sizes[other_dims[0]] if other_dims else 1
        )

        # Read only first/last time values and a small window for
        # sampling rate estimation.
        t_start = float(self._da.time.values[0])
        t_end = float(self._da.time.values[-1])

        n_for_sr = min(1000, n_time)
        time_head = self._da.time.values[:n_for_sr].astype(
            np.float64
        )
        if len(time_head) > 1:
            dt = float(np.median(np.diff(time_head)))
            sampling_rate = 1.0 / dt
        else:
            sampling_rate = 1.0

        depths = self._da.y.values.astype(np.float64, copy=False)

        total_duration = t_end - t_start

        logger.info(
            "Source: %d samples x %d channels, %.1f Hz, "
            "%.1f s (%.1f - %.1f)",
            n_time,
            n_channels,
            sampling_rate,
            total_duration,
            t_start,
            t_end,
        )

        return SourceMetadata(
            n_time_samples=n_time,
            n_channels=n_channels,
            sampling_rate=sampling_rate,
            t_start=t_start,
            t_end=t_end,
            total_duration=total_duration,
            depths=depths,
        )

    def read_time_slice(
        self, t_start: float, t_end: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read a time slice from the zarr store.

        Args:
            t_start: Start time in seconds (inclusive).
            t_end: End time in seconds (inclusive).

        Returns:
            Tuple of (values, times) where values has shape
            ``(n_samples, n_channels)`` as float32 and times has shape
            ``(n_samples,)`` as float64.
        """
        chunk = self._da.sel(time=slice(t_start, t_end))
        values = chunk.values.astype(np.float32, copy=False)
        times = chunk.time.values.astype(np.float64, copy=False)
        return values, times

    def estimate_color_range(
        self,
        n_windows: int = 5,
        window_seconds: float = 10.0,
        percentile: float = 99.5,
    ) -> float:
        """Estimate a symmetric color range by sampling sparse windows.

        Reads a few evenly-spaced time windows and computes the
        percentile of the absolute values, avoiding the need to load
        the entire dataset.

        Args:
            n_windows: Number of windows to sample.
            window_seconds: Duration of each sample window in seconds.
            percentile: Percentile for determining max magnitude.

        Returns:
            The estimated vmax (positive float). Color range should be
            set to ``(-vmax, vmax)``.
        """
        md = self.metadata
        usable = md.total_duration - window_seconds
        if usable <= 0:
            # Recording shorter than one window — read it all.
            data, _ = self.read_time_slice(md.t_start, md.t_end)
            vmax = float(
                np.nanpercentile(np.abs(data), percentile)
            )
            return max(vmax, 1e-12)

        offsets = np.linspace(0, usable, n_windows)
        chunks = []
        for offset in offsets:
            t0 = md.t_start + offset
            t1 = t0 + window_seconds
            data, _ = self.read_time_slice(t0, t1)
            chunks.append(data)

        all_data = np.concatenate(chunks, axis=0)
        vmax = float(np.nanpercentile(np.abs(all_data), percentile))
        if np.isnan(vmax) or vmax == 0:
            vmax = 1.0
        return vmax
