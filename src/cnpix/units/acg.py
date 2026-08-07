"""Autocorrelogram (ACG) computation, fitting, and derived metrics.

An ISI distribution is just the right half of an ACG.

The fitting code is a port of CellExplorer's ``fit_ACG.m``
(https://github.com/petersenpeter/CellExplorer). It is reproduced here
verbatim from the original ``findlay2025a`` implementation so that previously
computed cell-type labels remain reproducible; do not "clean up" the constants.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal
import xarray as xr

__all__ = [
    "ACG_PARAMS",
    "compute_acgs",
    "normalize_acgs",
    "fit_acg",
    "get_fits",
    "get_shoulder",
    "get_shoulders",
    "get_burst_index",
    "compute_acg_metrics",
    "plot_fit",
]

# Narrow ACGs drive the triple-exponential fit (tau_rise is the Petersen
# criterion); wide ACGs supply the burst-index denominator.
ACG_PARAMS: dict[str, dict[str, float]] = {
    "narrow": {"window_ms": 100, "bin_ms": 0.5},
    "wide": {"window_ms": 1000, "bin_ms": 5},
}

# The states over which ACGs are computed separately.
ACG_STATES: tuple[str, ...] = ("Wake", "NREM", "REM", "IS")


def compute_acgs(trains, hg, kind: str, states=ACG_STATES) -> xr.DataArray:
    """Compute per-state ACGs of the given ``kind`` ("narrow" or "wide").

    Thin wrapper over :func:`ecephys.units.compute_autocorrelograms_by_hypnogram_state`,
    which is numba-accelerated, so that the window/bin conventions live in one place.
    """
    from ecephys.units.correlograms import (
        compute_autocorrelograms_by_hypnogram_state,
    )

    if kind not in ACG_PARAMS:
        raise ValueError(f"Unrecognized correlogram kind: {kind!r}")
    params = ACG_PARAMS[kind]
    return compute_autocorrelograms_by_hypnogram_state(
        trains, hg, params["window_ms"], params["bin_ms"], states=list(states)
    )


def normalize_acgs(acgs: xr.DataArray) -> xr.DataArray:
    """Normalize an autocorrelogram whose bins are raw spike counts.

    Each bin is divided by the total number of spikes that cell fired (in that
    state), and then by the bin width. This is the Buzsaki lab's method, and
    their ACG fitting parameters are tuned for this normalization. They usually
    label the y-axis as "spikes per second", but this is not a real
    instantaneous firing rate.
    """
    return acgs / acgs.num_spikes / acgs.bin_ms * 1000


########################
# Functions for fitting ACGs
########################


def _triple_exponential(
    x: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
    h: float,
) -> np.ndarray:
    """
    Triple-exponential function used by CellExplorer to fit ACGs.
    Note that the decay amplitude affects the rise term.

    Parameters:
    x: Bin times, in ms. Expects 0.5ms to 50ms.
    a: tau_decay
    b: tau_rise
    c: decay_amplitude
    d: rise_amplitude
    e: asymptote
    f: refrac
    g: tau_burst
    h: burst_amplitude

    Returns:
    y: max(c*(exp(-(x-f)/a)-d*exp(-(x-f)/b))+h*exp(-(x-f)/g)+e, 0)
    """
    term1 = c * (np.exp(-(x - f) / a) - d * np.exp(-(x - f) / b))
    term2 = h * np.exp(-(x - f) / g)
    result = term1 + term2 + e
    return np.maximum(result, 0)


def fit_acg(
    bin_times: np.ndarray, bin_probs: np.ndarray, plot: bool = False
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Fit a triple-exponential function to a normalized ACG.
    Based on https://github.com/petersenpeter/CellExplorer/blob/cf58ac48935dfee9d04b2f6e975537ecc41b1e2a/calc_CellMetrics/fit_ACG.m#L83

    Parameters:
    ----------
    bin_times: np.ndarray
        Bin right edge times, in seconds.
        Expects a 200-bin ACG with a 100ms window in 0.5ms bins,
        i.e. (-0.05, -0.0495, ..., 0.0495).
        But should be able to handle a 201-bin CellExplorer-style ACG.
    bin_probs: np.ndarray
        Bin spike discharge probabilities.
        Assumes ACG has been normalized by both spike counts and bin width.
    """
    t = bin_times  # Just a convenient alias
    p = bin_probs.copy()  # Because we will set some bins to 0.

    # A unit that fired no spikes in a state has an all-zero ACG, and
    # `normalize_acgs` divides by the spike count, so its bins are NaN. There is
    # nothing to fit; report the same all-NaN result as a failed fit rather than
    # letting curve_fit raise and take the whole subject down. (Reached for the
    # first time by CNPIX7-Giuseppe, which findlay2025a's manifest excluded.)
    if not np.all(np.isfinite(p)):
        return (np.nan,) * 9

    bin_ms = 0.5
    window_ms = 100
    assert np.allclose(np.diff(bin_times), bin_ms / 1000), (
        "Initial guess params may assume 0.5ms bins and 100ms window"
    )

    # Set the autocorrelation around 0 ms to 0
    # In CellExplorer, they use ACGs with 201 bins, because they have a bin that is centered on 0ms
    # Our correlograms have 200 bins, because our central bins use 0ms as an edge, i.e. (-bin_ms, 0) and (0, +bin_ms)
    # This means that they will set 3 bins to zero, whereas we will set 2 bins to zero.
    # However BOTH of us will pass 100 bins to the fitting function, and in both cases the first bin passed will have been set to 0.
    zero_bins = slice((t.size // 2) - 1, (t.size // 2) + 1)
    p[zero_bins] = 0

    # We fit only the right half of the correlogram
    y = p[t.size // 2 :]  # Right half spike counts
    x = (
        np.arange(1, window_ms + 1) * bin_ms
    )  # Right half bin times, starting at 0.5ms and ending at 50ms

    a0 = np.asarray([20, 1, 30, 2, 0.5, 5, 1.5, 2])  # Initial guesses for parameters
    lb = np.asarray([1, 0.1, 0, 0, -30, 0, 0.1, 0])  # Lower bounds for parameters
    ub = np.asarray([500, 50, 500, 15, 50, 20, 5, 100])  # Upper bounds for parameters

    try:
        popt, pcov = scipy.optimize.curve_fit(
            _triple_exponential,
            x,
            y,
            p0=a0,
            method="trf",
            bounds=(lb, ub),
            xtol=1e-6,
            ftol=1e-6,
            maxfev=10000,
        )
    except RuntimeError:
        return (np.nan,) * 9

    (
        tau_decay,
        tau_rise,
        decay_amplitude,
        rise_amplitude,
        asymptote,
        refrac,
        tau_burst,
        burst_amplitude,
    ) = popt

    # Get goodness of fit
    fit_trf = _triple_exponential(x, *popt)
    ss_res = np.sum((y - fit_trf) ** 2)  # residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # total sum of squares
    rsq = 1 - (ss_res / ss_tot)

    if plot:
        plot_fit(t, p, *popt, rsq)

    return (
        tau_decay,
        tau_rise,
        decay_amplitude,
        rise_amplitude,
        asymptote,
        refrac,
        tau_burst,
        burst_amplitude,
        rsq,
    )


def get_fits(narrow_acgs: xr.DataArray) -> xr.Dataset:
    (
        tau_decay,
        tau_rise,
        decay_amplitude,
        rise_amplitude,
        asymptote,
        refrac,
        tau_burst,
        burst_amplitude,
        rsq,
    ) = xr.apply_ufunc(
        fit_acg,
        narrow_acgs.time,
        narrow_acgs,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ],  # Empty dims for each scalar output
        output_dtypes=[float, float, float, float, float, float, float, float, float],
        vectorize=True,
    )
    return xr.Dataset(
        {
            "tau_decay": tau_decay,
            "tau_rise": tau_rise,
            "decay_amplitude": decay_amplitude,
            "rise_amplitude": rise_amplitude,
            "asymptote": asymptote,
            "refrac": refrac,
            "tau_burst": tau_burst,
            "burst_amplitude": burst_amplitude,
            "rsq": rsq,
        }
    )


########################
# Functions for plotting ACG fits
########################


def _acg_decay_func(
    x: np.ndarray, tau_decay: float, decay_amplitude: float, refrac: float
) -> np.ndarray:
    return decay_amplitude * np.exp(-(x - refrac) / tau_decay)


def _acg_rise_func(
    x: np.ndarray, tau_rise: float, rise_amplitude: float, refrac: float
) -> np.ndarray:
    return rise_amplitude * np.exp(-(x - refrac) / tau_rise)


def _acg_burst_func(
    x: np.ndarray, refrac: float, tau_burst: float, burst_amplitude: float
) -> np.ndarray:
    return burst_amplitude * np.exp(-(x - refrac) / tau_burst)


def plot_fit(
    bin_times: np.ndarray,
    bin_probs: np.ndarray,
    tau_decay: float,
    tau_rise: float,
    decay_amplitude: float,
    rise_amplitude: float,
    asymptote: float,
    refrac: float,
    tau_burst: float,
    burst_amplitude: float,
    ax: plt.Axes | None = None,
    plot_terms: bool = True,
) -> None:
    window_ms = 100
    bin_ms = 0.5
    x = (
        np.arange(1, window_ms + 1) * bin_ms
    )  # Right half bin times, starting at 0.5ms and ending at 50ms
    t = bin_times
    y = bin_probs[t.size // 2 :]  # Right half spike counts

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.plot(x, y, label="data", c="grey", lw=3)
    ax.plot(
        x,
        _triple_exponential(
            x,
            tau_decay,
            tau_rise,
            decay_amplitude,
            rise_amplitude,
            asymptote,
            refrac,
            tau_burst,
            burst_amplitude,
        ),
        label="total",
        color="k",
        lw=0.5,
    )
    if plot_terms:
        ax.plot(
            x,
            _acg_decay_func(x, tau_decay, decay_amplitude, refrac),
            label="decay",
            c="r",
            lw=0.5,
        )
        ax.plot(
            x,
            _acg_rise_func(x, tau_rise, rise_amplitude, refrac),
            label="rise",
            c="g",
            lw=0.5,
        )
        ax.plot(
            x,
            _acg_burst_func(x, refrac, tau_burst, burst_amplitude),
            label="burst",
            c="b",
            lw=0.5,
        )
        ax.hlines(asymptote, x.min(), x.max(), label="asymptote", color="pink", lw=0.5)
    ax.set_ylim(np.min([0, asymptote]), np.max(y) * 1.2)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)


########################
# Functions for finding the shoulder of an ISI distribution
########################


def find_first_prominent_peak(
    isi_dist: np.ndarray, time_coords: np.ndarray, prominence_list: list[float] = None
) -> float:
    """Find the earliest peak in a normalized autocorrelogram using a list of prominence thresholds"""
    if prominence_list is None:
        prominence_list = [
            2.0
        ]  # You can go lower if your ISI distributions are more smoothed

    for prominence in prominence_list:
        peaks, properties = scipy.signal.find_peaks(
            isi_dist, prominence=prominence, distance=4
        )
        if len(peaks) > 0:
            # Return the earliest peak (smallest time coordinate)
            earliest_peak_idx = peaks[np.argmin(time_coords[peaks])]
            return time_coords[earliest_peak_idx]

    # If no peaks found with any prominence threshold
    return np.nan


def find_derivative_zero_crossing(
    isi_dist: np.ndarray, time_coords: np.ndarray
) -> float:
    """Find the first zero-crossing after the highest peak in the first derivative.
    Will often miss early peaks when given smoothed ISIs."""
    # Compute first derivative
    derivative = np.gradient(isi_dist, time_coords)

    # Find peaks in the derivative
    peaks, _ = scipy.signal.find_peaks(derivative)
    if len(peaks) == 0:
        return np.nan

    # Find the highest peak in the derivative (maximum value)
    highest_peak_idx = peaks[np.argmax(derivative[peaks])]

    # Look for the first zero-crossing after this peak
    # A zero-crossing occurs where derivative changes from positive to negative
    for i in range(highest_peak_idx, len(derivative) - 1):
        if derivative[i] > 0 and derivative[i + 1] <= 0:
            # Linear interpolation to find more precise zero-crossing
            # Zero occurs at: time_coords[i] + (time_coords[i+1] - time_coords[i]) * (-derivative[i] / (derivative[i+1] - derivative[i]))
            # if derivative[i+1] != derivative[i]:  # Avoid division by zero
            #    zero_crossing_time = time_coords[i] - derivative[i] * (time_coords[i+1] - time_coords[i]) / (derivative[i+1] - derivative[i])
            #    return zero_crossing_time
            # else:
            #    return time_coords[i]
            return time_coords[i]

    # If no zero-crossing found after the peak
    return np.nan


def find_most_prominent_peak(
    isi_dist: np.ndarray, time_coords: np.ndarray, **find_peaks_kwargs
) -> float:
    """Find the most prominent peak in an autocorrelogram. Generally not super useful."""
    peaks, properties = scipy.signal.find_peaks(isi_dist, **find_peaks_kwargs)
    if len(peaks) == 0:
        return np.nan
    # Get the peak with highest prominence
    max_prominence_idx = np.argmax(properties["prominences"])
    peak_idx = peaks[max_prominence_idx]
    return time_coords[peak_idx]


def get_shoulder(bin_times: np.ndarray, bin_probs: np.ndarray) -> float:
    """Get the shoulder of a normalized ISI distribution.

    Parameters:
    ----------
    bin_times: np.ndarray
        Bin right edge times, in seconds.
        Tested on a 100-bin ACG with a 100ms window in 0.5ms bins,
        i.e. (0, 0.0005, ..., 0.0495).
    bin_probs: np.ndarray
        Bin spike discharge probabilities.
        Assumes ACG has been normalized by both spike counts and bin width.

    Returns:
    -------
    float
        Time of the shoulder, in seconds.
    """
    t1 = find_first_prominent_peak(bin_probs, bin_times)
    t2 = find_derivative_zero_crossing(bin_probs, bin_times)
    pk = np.minimum(t1, t2)
    if np.isnan(pk):
        pk = t1
    if np.isnan(pk):
        pk = t2
    if np.isnan(pk):
        pk = bin_times[np.argmax(bin_probs)]
    return pk


def get_shoulders(normalized_isis: xr.DataArray) -> xr.DataArray:
    """Get the left shoulder of a normalized ISI distributon.

    Works best when the normalized ISIs have been smoothed a little.

    Parameters
    ----------
    normalized_isis : xr.DataArray
        Normalized ISI distribution. Must have a `time` dimension (usually the last).
        Probably also has a `cluster_id` dimension, and possibly a `state` dimension.

    Returns
    -------
    xr.DataArray
        Left shoulder of the normalized ISI distribution, in seconds.
    """
    shoulders = xr.apply_ufunc(
        get_shoulder,
        normalized_isis.time,
        normalized_isis,
        input_core_dims=[["time"], ["time"]],
        output_dtypes=[float],
        vectorize=True,
    )
    return shoulders.rename("isi_shoulder")


########################
# Miscellaneous functions for computing ACG metrics
########################


def get_burst_index(narrow_acgs: xr.DataArray, wide_acgs: xr.DataArray) -> float:
    inner = narrow_acgs.sel(time=slice(0.0029, 0.005)).sum(dim="time")
    outer = wide_acgs.sel(time=slice(0.199, 0.300)).sum(dim="time")
    burst_index = inner / outer
    return burst_index.rename("burst_index")


########################
# Aggregating the ACG metrics
########################


def compute_acg_metrics(
    narrow_acgs: xr.DataArray, wide_acgs: xr.DataArray
) -> pd.DataFrame:
    """Derive the per-(cluster, state) ACG metric table.

    Combines the triple-exponential fit parameters, the ISI-distribution
    shoulder, the burst index, and the first moment of the ISI distribution.

    Both inputs must already be normalized (see :func:`normalize_acgs`).
    """
    fit_results = get_fits(narrow_acgs)

    # A little smoothing makes the shoulder finder much more reliable.
    smth_acgs = (
        narrow_acgs.rolling(time=3, center=True).mean().ffill(dim="time").bfill(dim="time")
    )
    shoulders = get_shoulders(smth_acgs.sel(time=slice(0, None)))

    burst_index = get_burst_index(narrow_acgs, wide_acgs)
    moment = narrow_acgs.sel(time=slice(0, None)).mean(dim="time").rename("moment")

    metrics = xr.merge([fit_results, shoulders, burst_index, moment])
    return metrics.to_dataframe().reset_index()
