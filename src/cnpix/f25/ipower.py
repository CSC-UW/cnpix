"""Shared pieces of the instantaneous cortical-power extraction.

Both Findlay et al. (2025) and the local-sleep paper extract instantaneous
band power from these recordings, and both need the same two steps: a
conservative outlier threshold derived from gaps in the value histogram, and
a hypnogram strip with the light/dark overlay. Neither is paper business, so
neither should be forked across the two repositories.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wisc_ecephys_tools as wet
from ecephys import plot as eplt
from ecephys.hypnogram import FloatHypnogram
from ecephys.wne.sglx import SGLXSubject

__all__ = ["plot_hypnogram", "replace_outliers_kd"]


def _find_first_zero_run(a: np.ndarray, n: int = 1) -> int:
    """Get the index of the first run of n consecutive zeros in a numpy array."""
    first_zero_run = None
    for i in range(len(a) - n + 1):
        if not any(a[i : i + n]):  # All zeros in window
            first_zero_run = i
            break
    return first_zero_run


def _get_threshold_table(
    hist: np.ndarray, bin_edges: np.ndarray, compact: bool = True
) -> pd.DataFrame:
    # Get runs of zeros
    zero_runs = np.split(np.arange(len(hist)), np.where(np.diff(hist == 0))[0] + 1)
    # Filter to only the runs of zeros
    zero_runs = [run for run in zero_runs if hist[run[0]] == 0]
    # Get lengths of zero runs
    run_lengths = [len(run) for run in zero_runs]
    # Get thresholds for different run lengths
    thresholds = {}
    for n in range(1, max(run_lengths) + 1):
        ix = _find_first_zero_run(hist, n)
        if ix is not None:
            thresholds[n] = bin_edges[ix]

    df = pd.DataFrame(
        {"run_length": list(thresholds.keys()), "threshold": list(thresholds.values())}
    )
    if compact:
        # Group by threshold and keep first occurrence (minimum run length)
        df = df.groupby("threshold").first().reset_index()
    return df


def _plot_threshold_table(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    tdf: pd.DataFrame,
    ylim_frac: float = 0.01,
    threshold_ix: int = 0,
) -> tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(2, 1, figsize=(4, 5), height_ratios=[10, 1])

    axes[0].hist(bin_edges[:-1], bin_edges, weights=hist)
    axes[0].set_ylim(0, ylim_frac * np.max(hist))
    for ix in tdf.index:
        color = "r" if ix == threshold_ix else "k"
        axes[0].axvline(
            tdf.loc[ix, "threshold"], color=color, linestyle="--", linewidth=0.5
        )

    nz = hist > 0
    axes[1].hist(bin_edges[:-1], bin_edges, weights=nz.astype(int))

    return fig, axes


def replace_outliers_kd(
    x: np.ndarray,
    threshold_ix: int = 0,
    bins: int = 1000,
    plot_distribution: bool = False,
    fill_value: float = np.nan,
    **plot_kwargs,
) -> tuple[float, np.ndarray]:
    """
    Usually more conservative (i.e. yields a higher threshold) than even
    np.nanquantile(x, 0.9999).
    """
    hist, bin_edges = np.histogram(x[~np.isnan(x)], bins=bins)
    tdf = _get_threshold_table(hist, bin_edges)
    if plot_distribution:
        fig, axes = _plot_threshold_table(
            hist, bin_edges, tdf, threshold_ix=threshold_ix, **plot_kwargs
        )
    threshold = tdf.loc[threshold_ix, "threshold"]
    x[x > threshold] = fill_value
    return threshold, x


def plot_hypnogram(
    experiment: str,
    subject: SGLXSubject,
    hg: FloatHypnogram,
    state_colors: dict = eplt.publication_colors,
    show_ticklabels: bool = False,
) -> plt.Axes:
    ax = eplt.plot_hypnogram_overlay(
        hg, xlim="hg", figsize=(16, 1), state_colors=state_colors
    )
    wet.rats.cnd_hgs.plot_lights_overlay(
        *wet.rats.cnd_hgs.get_light_dark_periods(experiment, subject),
        ax=ax,
        ymax=1.04,
    )
    if not show_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return ax
