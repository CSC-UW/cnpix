"""Tests for autocorrelogram fitting."""

import numpy as np
import pytest

from cnpix.units import acg


@pytest.fixture
def bin_times():
    """Bin right-edge times for a 200-bin, 100 ms window, 0.5 ms ACG."""
    return np.arange(-100, 100) * 0.0005


class TestFitAcg:
    def test_silent_unit_returns_all_nan_rather_than_raising(self):
        # A unit with zero spikes in a state has num_spikes == 0, so
        # normalize_acgs yields all-NaN bins. That must not take down the fit.
        probs = np.full(200, np.nan)
        out = acg.fit_acg(np.arange(-100, 100) * 0.0005, probs)
        assert len(out) == 9
        assert all(np.isnan(v) for v in out)

    def test_partial_nan_also_returns_all_nan(self, bin_times):
        probs = np.ones(200)
        probs[7] = np.nan
        assert all(np.isnan(v) for v in acg.fit_acg(bin_times, probs))

    def test_infinite_bins_return_all_nan(self, bin_times):
        probs = np.ones(200)
        probs[3] = np.inf
        assert all(np.isnan(v) for v in acg.fit_acg(bin_times, probs))

    def test_finite_acg_still_fits(self, bin_times):
        # A plausible decaying ACG should produce a finite fit, so the guard
        # cannot be swallowing real data.
        x = np.abs(bin_times) * 1000  # ms
        probs = 20 * (1 - np.exp(-x / 2)) * np.exp(-x / 30) + 1.0
        out = acg.fit_acg(bin_times, probs)
        assert np.isfinite(out[:8]).all()

    def test_wrong_bin_width_is_rejected(self, bin_times):
        # The CellExplorer fit constants assume 0.5 ms bins; silently accepting
        # another width would give meaningless tau values.
        with pytest.raises(AssertionError):
            acg.fit_acg(bin_times * 2, np.ones(200))


class TestNormalizeAcgs:
    def test_zero_spike_unit_produces_nan_not_an_exception(self):
        import xarray as xr

        da = xr.DataArray(
            np.zeros((2, 4), dtype=np.int64),
            dims=("cluster_id", "time"),
            coords={
                "cluster_id": [1, 2],
                "time": [0.0, 0.0005, 0.001, 0.0015],
                "num_spikes": ("cluster_id", np.array([100, 0])),
            },
            attrs={"bin_ms": 0.5, "window_ms": 100},
        )
        out = acg.normalize_acgs(da)
        assert np.isfinite(out.sel(cluster_id=1)).all()
        assert not np.isfinite(out.sel(cluster_id=2)).all()
