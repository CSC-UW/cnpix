"""Main window for the LFP/CSD Viewer."""

from __future__ import annotations

import logging
import math
from typing import Literal

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QInputDialog,
    QLabel,
    QMainWindow,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from lfp_csd_viewer.data.cache import PageCache, PageData
from lfp_csd_viewer.data.data_loader import DataLoaderThread
from lfp_csd_viewer.data.loader import ViewerSources, open_viewer_sources
from lfp_csd_viewer.widgets.csd_view import CsdViewWidget
from lfp_csd_viewer.widgets.navigation import NavigationBar

logger = logging.getLogger(__name__)

# Diverging colormaps suitable for CSD data.
AVAILABLE_COLORMAPS = [
    "CET-D1a",
    "CET-D1",
    "CET-D9",
]


class MainWindow(QMainWindow):
    """Main application window for the LFP/CSD Viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._sources: ViewerSources | None = None
        self._loader: DataLoaderThread | None = None
        self._cache: PageCache | None = None
        self._page_duration: float = 10.0
        self._current_page: int = 0
        self._n_pages: int = 0
        self._lfp_gain: float = 1.0

        self.setWindowTitle("LFP/CSD Viewer")
        self.resize(1400, 800)

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()

    def _setup_ui(self) -> None:
        """Set up the central widget layout."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._csd_view = CsdViewWidget()
        layout.addWidget(self._csd_view, stretch=1)

        self._nav_bar = NavigationBar()
        layout.addWidget(self._nav_bar)

        # Connect navigation signals.
        self._nav_bar.page_requested.connect(self._request_page)
        self._nav_bar.time_requested.connect(
            self._on_time_requested
        )

        # Connect hover signal for status bar.
        self._csd_view.mouse_moved.connect(self._on_mouse_moved)

        # Status bar.
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

    def _setup_menus(self) -> None:
        """Set up the menu bar."""
        menu_bar = self.menuBar()

        # --- View menu ---
        view_menu = menu_bar.addMenu("&View")

        self._toggle_csd_action = QAction(
            "Show &CSD", self, checkable=True, checked=True
        )
        self._toggle_csd_action.triggered.connect(
            self._on_toggle_csd
        )
        view_menu.addAction(self._toggle_csd_action)

        self._toggle_lfp_action = QAction(
            "Show &LFP", self, checkable=True, checked=True
        )
        self._toggle_lfp_action.triggered.connect(
            self._on_toggle_lfp
        )
        view_menu.addAction(self._toggle_lfp_action)

        self._toggle_strip_action = QAction(
            "Show &State Strip",
            self,
            checkable=True,
            checked=True,
        )
        self._toggle_strip_action.triggered.connect(
            self._on_toggle_state_strip
        )
        view_menu.addAction(self._toggle_strip_action)

        view_menu.addSeparator()

        # Colormap submenu.
        cmap_menu = view_menu.addMenu("Colormap")
        for name in AVAILABLE_COLORMAPS:
            action = QAction(name, self)
            action.triggered.connect(
                lambda checked, n=name: self._csd_view.set_colormap(
                    n
                )
            )
            cmap_menu.addAction(action)

        view_menu.addSeparator()

        color_range_action = QAction(
            "Color Range...", self
        )
        color_range_action.triggered.connect(
            self._on_set_color_range
        )
        view_menu.addAction(color_range_action)

        auto_range_action = QAction(
            "Auto Color Range", self
        )
        auto_range_action.triggered.connect(
            self._on_auto_color_range
        )
        view_menu.addAction(auto_range_action)

    def _setup_toolbar(self) -> None:
        """Set up the toolbar with gain and page duration
        controls."""
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # LFP gain control.
        toolbar.addWidget(QLabel(" LFP Gain: "))
        self._gain_spinbox = QDoubleSpinBox()
        self._gain_spinbox.setRange(0.01, 100000.0)
        self._gain_spinbox.setSingleStep(0.1)
        self._gain_spinbox.setValue(self._lfp_gain)
        self._gain_spinbox.setDecimals(2)
        self._gain_spinbox.setFixedWidth(100)
        self._gain_spinbox.valueChanged.connect(
            self._on_gain_changed
        )
        toolbar.addWidget(self._gain_spinbox)

        toolbar.addSeparator()

        # LFP channel step control.
        toolbar.addWidget(QLabel(" LFP Ch. Step: "))
        self._channel_step_spinbox = QSpinBox()
        self._channel_step_spinbox.setRange(1, 50)
        self._channel_step_spinbox.setValue(5)
        self._channel_step_spinbox.setFixedWidth(60)
        self._channel_step_spinbox.valueChanged.connect(
            self._on_channel_step_changed
        )
        toolbar.addWidget(self._channel_step_spinbox)

        toolbar.addSeparator()

        # Page duration control.
        toolbar.addWidget(QLabel(" Page Duration: "))
        self._page_duration_spinbox = QDoubleSpinBox()
        self._page_duration_spinbox.setRange(1.0, 300.0)
        self._page_duration_spinbox.setSingleStep(1.0)
        self._page_duration_spinbox.setValue(self._page_duration)
        self._page_duration_spinbox.setSuffix(" s")
        self._page_duration_spinbox.setDecimals(1)
        self._page_duration_spinbox.setFixedWidth(100)
        self._page_duration_spinbox.editingFinished.connect(
            self._on_page_duration_changed
        )
        toolbar.addWidget(self._page_duration_spinbox)

    # --- Data loading ---

    def load_data(
        self,
        subject: str,
        experiment: str,
        kind: Literal["cortical", "hippocampal"],
        condition: str = "Full.Conservative",
    ) -> None:
        """Open data sources lazily and start the background loader.

        Args:
            subject: Subject name.
            experiment: Experiment name.
            kind: "cortical" or "hippocampal".
            condition: Hypnogram condition for sleep scoring.
        """
        self.setWindowTitle(
            f"LFP/CSD Viewer - {subject} / {experiment} / {kind}"
        )
        self._status_bar.showMessage("Opening data sources...")

        # Stop any existing loader.
        self._stop_loader()

        self._sources = open_viewer_sources(
            subject, experiment, kind, condition
        )

        # Estimate color range from sparse sampling (fast).
        vmax = self._sources.csd_source.estimate_color_range()
        self._csd_view.set_color_range(-vmax, vmax)

        # Configure paging.
        self._recompute_pages()

        # Create cache and loader thread.
        self._cache = PageCache(max_pages=5)
        self._loader = DataLoaderThread(
            csd_source=self._sources.csd_source,
            lfp_source=self._sources.lfp_source,
            cache=self._cache,
            page_duration=self._page_duration,
            n_pages=self._n_pages,
            hypnogram=self._sources.hypnogram,
        )
        self._loader.page_loaded.connect(self._on_page_loaded)
        self._loader.loading_started.connect(
            self._on_loading_started
        )
        self._loader.error_occurred.connect(
            self._on_loader_error
        )
        self._loader.start()

        # Request first page.
        self._request_page(0)

        md = self._sources.csd_source.metadata
        self._status_bar.showMessage(
            f"Opened: {subject} / {experiment} / {kind} "
            f"({md.n_time_samples} samples, "
            f"{md.total_duration:.0f}s)",
            5000,
        )

    def _recompute_pages(self) -> None:
        """Recalculate page count from current page duration."""
        if self._sources is None:
            return

        total = self._sources.csd_source.metadata.total_duration
        t_offset = self._sources.csd_source.metadata.t_start

        self._n_pages = max(
            1, math.ceil(total / self._page_duration)
        )
        self._nav_bar.configure(
            n_pages=self._n_pages,
            total_duration=total,
            time_offset=t_offset,
        )

    # --- Paging ---

    @Slot(int)
    def _request_page(self, page_index: int) -> None:
        """Request a page from the background loader.

        If the page is already cached, display it immediately.
        Otherwise, queue it for loading.

        Args:
            page_index: Zero-based page index.
        """
        if self._sources is None or self._cache is None:
            return

        page_index = max(0, min(page_index, self._n_pages - 1))
        self._current_page = page_index

        # Eagerly update nav bar so rapid clicks use the new base page.
        self._nav_bar.set_pending_page(page_index)

        # Try cache first for instant display.
        cached = self._cache.get(page_index)
        if cached is not None:
            self._display_page(page_index, cached)
            # Still prefetch neighbors.
            if self._loader is not None:
                self._loader.request_page(
                    page_index + 1, prefetch=True
                )
                self._loader.request_page(
                    page_index - 1, prefetch=True
                )
            return

        # Cancel stale requests before queuing the new one.
        if self._loader is not None:
            self._loader.cancel_pending()
            self._loader.request_page_with_prefetch(page_index)

    @Slot(int, object)
    def _on_page_loaded(
        self, page_index: int, page_data: PageData
    ) -> None:
        """Handle a page loaded by the background thread.

        Only display if this is still the currently requested page.

        Args:
            page_index: The page that was loaded.
            page_data: The loaded page data.
        """
        if page_index == self._current_page:
            self._display_page(page_index, page_data)

    @Slot(int)
    def _on_loading_started(self, page_index: int) -> None:
        """Show loading indicator for the requested page."""
        if page_index == self._current_page:
            self._status_bar.showMessage(
                f"Loading page {page_index + 1}..."
            )

    @Slot(str)
    def _on_loader_error(self, message: str) -> None:
        """Show loader error in status bar."""
        logger.error(message)
        self._status_bar.showMessage(message, 5000)

    def _display_page(
        self, page_index: int, page_data: PageData
    ) -> None:
        """Display a loaded page.

        Args:
            page_index: The page index.
            page_data: The page data to display.
        """
        if self._sources is None:
            return

        csd_depths = self._sources.csd_source.metadata.depths
        lfp_depths = self._sources.lfp_source.metadata.depths

        # CSD heatmap.
        if len(page_data.csd_times) > 0:
            self._csd_view.set_csd_data(
                page_data.csd_values,
                page_data.csd_times,
                csd_depths,
            )

        # LFP traces.
        if len(page_data.lfp_times) > 0:
            self._csd_view.set_lfp_traces(
                page_data.lfp_values,
                page_data.lfp_times,
                lfp_depths,
                self._lfp_gain,
            )

        # State strip.
        if page_data.state_rgba is not None:
            self._csd_view.set_state_strip(
                page_data.state_rgba, page_data.csd_times
            )
        else:
            self._csd_view.clear_state_strip()

        # Update navigation bar.
        t_start = self._sources.csd_source.metadata.t_start
        page_t_start = t_start + page_index * self._page_duration
        actual_end = (
            float(page_data.csd_times[-1])
            if len(page_data.csd_times) > 0
            else page_t_start + self._page_duration
        )
        self._nav_bar.set_current_page(
            page_index, float(page_t_start), actual_end
        )

        self._status_bar.clearMessage()

    @Slot(float)
    def _on_time_requested(self, time_seconds: float) -> None:
        """Jump to a specific time."""
        if self._sources is None:
            return

        offset = (
            time_seconds
            - self._sources.csd_source.metadata.t_start
        )
        page = int(offset / self._page_duration)
        page = max(0, min(page, self._n_pages - 1))
        self._request_page(page)

    # --- Toolbar callbacks ---

    @Slot(float)
    def _on_gain_changed(self, gain: float) -> None:
        """Handle LFP gain spinbox change."""
        self._lfp_gain = gain
        self._csd_view.update_lfp_gain(gain)

    @Slot()
    def _on_page_duration_changed(self) -> None:
        """Handle page duration spinbox change."""
        new_duration = self._page_duration_spinbox.value()
        if new_duration == self._page_duration:
            return

        self._page_duration = new_duration
        self._recompute_pages()

        # Update the loader's paging parameters.
        if self._loader is not None:
            self._loader.set_paging(
                self._page_duration, self._n_pages
            )

        # Re-request at roughly the same time position.
        page = min(self._current_page, self._n_pages - 1)
        self._request_page(page)

    @Slot(int)
    def _on_channel_step_changed(self, step: int) -> None:
        """Handle LFP channel step spinbox change."""
        self._csd_view.set_lfp_channel_step(step)

    # --- Menu callbacks ---

    @Slot(bool)
    def _on_toggle_csd(self, checked: bool) -> None:
        self._csd_view.set_csd_visible(checked)

    @Slot(bool)
    def _on_toggle_lfp(self, checked: bool) -> None:
        self._csd_view.set_lfp_visible(checked)

    @Slot(bool)
    def _on_toggle_state_strip(self, checked: bool) -> None:
        self._csd_view.set_state_strip_visible(checked)

    @Slot()
    def _on_set_color_range(self) -> None:
        """Prompt user to set symmetric color range magnitude."""
        current_mag = self._csd_view._vmax
        mag, ok = QInputDialog.getDouble(
            self,
            "Color Range",
            "Max magnitude (range will be -val to +val):",
            current_mag,
            0.001,
            1e12,
            4,
        )
        if ok:
            self._csd_view.set_color_range(-mag, mag)

    @Slot()
    def _on_auto_color_range(self) -> None:
        """Recompute color range by sampling sparse windows."""
        if self._sources is not None:
            vmax = (
                self._sources.csd_source.estimate_color_range()
            )
            self._csd_view.set_color_range(-vmax, vmax)

    # --- Status bar ---

    @Slot(float, float, float)
    def _on_mouse_moved(
        self, time: float, depth: float, value: float
    ) -> None:
        self._status_bar.showMessage(
            f"Time: {time:.3f}s | Depth: {depth:.0f}um "
            f"| CSD: {value:.4f}"
        )

    # --- Keyboard navigation ---

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_D):
            if self._current_page < self._n_pages - 1:
                self._request_page(self._current_page + 1)
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_A):
            if self._current_page > 0:
                self._request_page(self._current_page - 1)
        elif key == Qt.Key.Key_Home:
            self._request_page(0)
        elif key == Qt.Key.Key_End:
            self._request_page(self._n_pages - 1)
        else:
            super().keyPressEvent(event)

    # --- Cleanup ---

    def _stop_loader(self) -> None:
        """Stop and clean up the loader thread."""
        if self._loader is not None:
            self._loader.stop()
            self._loader.wait()
            self._loader = None

    def closeEvent(self, event) -> None:
        """Stop the loader thread on close."""
        self._stop_loader()
        super().closeEvent(event)
