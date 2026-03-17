"""CSD heatmap + LFP trace overlay + hypnogram state strip."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal

# RGBA colors for displayed sleep states.
STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "Wake": (178, 223, 138, 255),
    "MA": (255, 255, 153, 255),
    "NREM": (166, 206, 227, 255),
    "IS": (255, 127, 0, 255),
    "REM": (253, 191, 111, 255),
    "Artifact": (251, 154, 153, 255),
}

# Default pen for LFP traces.
LFP_PEN = pg.mkPen(color=(0, 0, 0, 180), width=0.5)


class CsdViewWidget(pg.GraphicsLayoutWidget):
    """Displays a CSD heatmap with optional LFP line trace overlay
    and a hypnogram state strip.

    Layout:
        Row 0: Hypnogram state strip (thin, x-linked to main plot)
        Row 1: Main plot with CSD ImageItem + LFP PlotCurveItems
    """

    mouse_moved = Signal(float, float, float)  # time, depth, value

    def __init__(self, parent=None):
        super().__init__(parent)

        self._csd_data: np.ndarray | None = None
        self._csd_times: np.ndarray | None = None
        self._csd_depths: np.ndarray | None = None
        self._lfp_data: np.ndarray | None = None
        self._lfp_times: np.ndarray | None = None
        self._lfp_depths: np.ndarray | None = None

        self._vmin: float = -1.0
        self._vmax: float = 1.0
        self._lfp_gain: float = 1.0
        self._show_csd: bool = True
        self._show_lfp: bool = True
        self._show_state_strip: bool = True
        self._lfp_channel_step: int = 5

        self._lfp_curves: list[pg.PlotCurveItem] = []

        self._setup_plot()

    def _setup_plot(self) -> None:
        """Initialize PyQtGraph components."""
        # --- Row 0: State strip (initially hidden) ---
        self._strip_plot = self.addPlot(row=0, col=0)
        self._strip_plot.hideAxis("left")
        self._strip_plot.hideAxis("bottom")
        self._strip_plot.setMouseEnabled(x=False, y=False)
        self._strip_plot.setMenuEnabled(False)
        self._strip_plot.hideButtons()
        self._strip_plot.setMaximumHeight(20)
        self._strip_plot.setMinimumHeight(12)
        self._strip_plot.disableAutoRange()
        self._strip_plot.setVisible(False)

        self._strip_image = pg.ImageItem()
        self._strip_plot.addItem(self._strip_image)

        # Row sizing: strip stays thin, main plot takes all space.
        self.ci.layout.setRowStretchFactor(0, 0)
        self.ci.layout.setRowStretchFactor(1, 1)

        # --- Row 1: Main plot ---
        self._plot = self.addPlot(row=1, col=0)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Depth", units="um")

        # Link strip x-axis to main plot.
        self._strip_plot.setXLink(self._plot)

        # CSD heatmap image.
        self._csd_image = pg.ImageItem()
        self._plot.addItem(self._csd_image)
        self._csd_image.setZValue(0)

        # Diverging colormap (blue-white-red).
        self._colormap = pg.colormap.get("CET-D1A")
        self._csd_image.setColorMap(self._colormap)

        # Colorbar.
        self._colorbar = pg.ColorBarItem(
            values=(-1, 1),
            colorMap=self._colormap,
            interactive=False,
            width=15,
        )
        self._colorbar.setImageItem(
            self._csd_image, insert_in=self._plot
        )

        # Hover tracking.
        self._csd_image.hoverEvent = self._on_hover

        # Disable auto-range to prevent jumps.
        self._plot.disableAutoRange()

    # --- CSD data ---

    def set_csd_data(
        self,
        data: np.ndarray,
        times: np.ndarray,
        depths: np.ndarray,
    ) -> None:
        """Update the CSD heatmap.

        Args:
            data: Shape (n_samples, n_channels).
            times: Shape (n_samples,).
            depths: Shape (n_channels,).
        """
        self._csd_data = data
        self._csd_times = times
        self._csd_depths = depths

        self._csd_image.setImage(data, autoLevels=False)
        self._csd_image.setVisible(self._show_csd)

        # Map pixel coordinates to time/depth.
        t0 = times[0]
        t1 = times[-1]
        dt = (t1 - t0) / (len(times) - 1) if len(times) > 1 else 1

        y0 = depths.min()
        y1 = depths.max()
        dy = (y1 - y0) / (len(depths) - 1) if len(depths) > 1 else 1

        self._csd_image.setRect(
            t0, y0, (t1 - t0) + dt, (y1 - y0) + dy
        )
        self._csd_image.setLevels([self._vmin, self._vmax])

        self._plot.setXRange(t0, t1 + dt, padding=0)
        self._plot.setYRange(y0, y1 + dy, padding=0)

    # --- LFP traces ---

    def set_lfp_traces(
        self,
        data: np.ndarray,
        times: np.ndarray,
        depths: np.ndarray,
        gain: float,
    ) -> None:
        """Draw LFP line traces overlaid on the CSD heatmap.

        Each channel is mean-subtracted, scaled by gain, and offset
        by its depth coordinate.

        Args:
            data: Shape (n_samples, n_channels).
            times: Shape (n_samples,).
            depths: Shape (n_channels,).
            gain: Multiplier for trace amplitude.
        """
        self._clear_lfp_curves()
        self._lfp_data = data
        self._lfp_times = times
        self._lfp_depths = depths
        self._lfp_gain = gain

        if not self._show_lfp:
            return

        self._draw_lfp_curves(data, times, depths, gain)

    def _draw_lfp_curves(
        self,
        data: np.ndarray,
        times: np.ndarray,
        depths: np.ndarray,
        gain: float,
    ) -> None:
        """Create PlotCurveItem objects for LFP traces."""
        for i in range(0, data.shape[1], self._lfp_channel_step):
            channel_data = data[:, i].astype(np.float64)
            trace = (channel_data - np.nanmean(channel_data)) * gain
            trace = trace + depths[i]

            curve = pg.PlotCurveItem(
                x=times, y=trace, pen=LFP_PEN
            )
            curve.setZValue(5)
            self._plot.addItem(curve)
            self._lfp_curves.append(curve)

    def _clear_lfp_curves(self) -> None:
        """Remove all LFP curve items from the plot."""
        for curve in self._lfp_curves:
            self._plot.removeItem(curve)
        self._lfp_curves.clear()

    def update_lfp_gain(self, gain: float) -> None:
        """Recompute LFP traces with a new gain value.

        Args:
            gain: New gain multiplier.
        """
        self._lfp_gain = gain
        if (
            self._lfp_data is not None
            and self._lfp_times is not None
            and self._lfp_depths is not None
        ):
            self._clear_lfp_curves()
            if self._show_lfp:
                self._draw_lfp_curves(
                    self._lfp_data,
                    self._lfp_times,
                    self._lfp_depths,
                    gain,
                )

    def set_lfp_channel_step(self, step: int) -> None:
        """Set the channel step for LFP traces and redraw.

        Args:
            step: Show every step-th channel (1 = all channels).
        """
        self._lfp_channel_step = max(1, step)
        if (
            self._lfp_data is not None
            and self._lfp_times is not None
            and self._lfp_depths is not None
        ):
            self._clear_lfp_curves()
            if self._show_lfp:
                self._draw_lfp_curves(
                    self._lfp_data,
                    self._lfp_times,
                    self._lfp_depths,
                    self._lfp_gain,
                )

    # --- Visibility toggles ---

    def set_csd_visible(self, visible: bool) -> None:
        """Show or hide the CSD heatmap."""
        self._show_csd = visible
        self._csd_image.setVisible(visible)
        self._colorbar.setVisible(visible)
        self._plot.getViewBox().setBackgroundColor(
            None if visible else "w"
        )

    def set_lfp_visible(self, visible: bool) -> None:
        """Show or hide the LFP traces."""
        self._show_lfp = visible
        if visible:
            if not self._lfp_curves and self._lfp_data is not None:
                self._draw_lfp_curves(
                    self._lfp_data,
                    self._lfp_times,
                    self._lfp_depths,
                    self._lfp_gain,
                )
        else:
            self._clear_lfp_curves()

    # --- Color range ---

    def set_color_range(self, vmin: float, vmax: float) -> None:
        """Set the CSD color scale range (symmetric around 0).

        Args:
            vmin: Minimum value (should be negative).
            vmax: Maximum value (should be positive).
        """
        self._vmin = vmin
        self._vmax = vmax
        self._csd_image.setLevels([vmin, vmax])
        self._colorbar.setLevels(values=(vmin, vmax))

    def set_colormap(self, name: str) -> None:
        """Set the colormap by name.

        Args:
            name: PyQtGraph colormap name (e.g., "CET-D1a").
        """
        try:
            self._colormap = pg.colormap.get(name)
            self._csd_image.setColorMap(self._colormap)
            self._colorbar.setColorMap(self._colormap)
        except Exception:
            pass

    # --- State strip ---

    def set_state_strip(
        self, rgba: np.ndarray, times: np.ndarray
    ) -> None:
        """Update the hypnogram state strip display.

        Args:
            rgba: Shape (n_samples, 4), RGBA colors with dtype
                np.ubyte.
            times: Shape (n_samples,), time values.
        """
        if len(rgba) == 0:
            self.clear_state_strip()
            return

        self._strip_image.setImage(
            rgba.reshape(len(rgba), 1, 4)
        )

        t0 = times[0]
        t1 = times[-1]
        dt = (t1 - t0) / (len(times) - 1) if len(times) > 1 else 1

        self._strip_image.setRect(t0, 0, (t1 - t0) + dt, 1)
        self._strip_plot.setYRange(0, 1, padding=0)

        if self._show_state_strip:
            self._strip_plot.setVisible(True)

    def clear_state_strip(self) -> None:
        """Remove state strip data and hide it."""
        self._strip_image.clear()
        self._strip_plot.setVisible(False)

    def set_state_strip_visible(self, visible: bool) -> None:
        """Show or hide the state strip."""
        self._show_state_strip = visible
        has_data = self._strip_image.image is not None
        self._strip_plot.setVisible(visible and has_data)

    # --- Hover ---

    def _on_hover(self, event) -> None:
        """Handle mouse hover over the CSD heatmap."""
        if event.isExit():
            return

        if self._csd_data is None or self._csd_times is None:
            return

        pos = event.pos()
        x, y = pos.x(), pos.y()

        times = self._csd_times
        depths = self._csd_depths

        # Map to time index.
        t0, t1 = times[0], times[-1]
        if t1 > t0:
            time_idx = int(
                (x - t0) / (t1 - t0) * (len(times) - 1)
            )
            time_idx = max(0, min(time_idx, len(times) - 1))
            time_val = times[time_idx]
        else:
            time_idx = 0
            time_val = t0

        # Map to channel index.
        y0, y1 = depths.min(), depths.max()
        if y1 > y0:
            chan_idx = int(
                (y - y0) / (y1 - y0) * (len(depths) - 1)
            )
            chan_idx = max(0, min(chan_idx, len(depths) - 1))
        else:
            chan_idx = 0

        if (
            0 <= time_idx < self._csd_data.shape[0]
            and 0 <= chan_idx < self._csd_data.shape[1]
        ):
            value = float(self._csd_data[time_idx, chan_idx])
            self.mouse_moved.emit(
                float(time_val), float(depths[chan_idx]), value
            )


def get_state_colors(
    hypnogram, times: np.ndarray
) -> np.ndarray:
    """Map times to RGBA colors based on sleep state.

    Args:
        hypnogram: A FloatHypnogram with per-epoch state labels.
        times: Sorted 1-D array of time values in seconds.

    Returns:
        Array of shape (len(times), 4) with dtype np.ubyte.
    """
    states = hypnogram.get_states(times)
    rgba = np.zeros((len(times), 4), dtype=np.ubyte)
    for state_name, color in STATE_COLORS.items():
        mask = states == state_name
        rgba[mask] = color
    return rgba
