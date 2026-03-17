"""Navigation bar for paging through data."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QWidget,
)


class NavigationBar(QWidget):
    """Navigation controls for paging through in-memory data.

    Provides a scrollbar, prev/next buttons, page indicator, and
    time jump controls.
    """

    page_requested = Signal(int)
    time_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._n_pages: int = 1
        self._current_page: int = 0
        self._total_duration: float = 0.0

        self._setup_ui()
        self._connect_signals()

        # Debounce timer for scrollbar drags.
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(150)
        self._scroll_timer.timeout.connect(self._emit_scrollbar_page)

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self._prev_button = QPushButton("Prev")
        self._prev_button.setFixedWidth(70)
        layout.addWidget(self._prev_button)

        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrollbar.setMinimum(0)
        self._scrollbar.setMaximum(0)
        self._scrollbar.setPageStep(1)
        self._scrollbar.setSingleStep(1)
        layout.addWidget(self._scrollbar, stretch=1)

        self._next_button = QPushButton("Next")
        self._next_button.setFixedWidth(70)
        layout.addWidget(self._next_button)

        self._page_label = QLabel("Page 0/0")
        self._page_label.setFixedWidth(100)
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._page_label)

        self._time_label = QLabel("0.0 - 0.0 s")
        self._time_label.setFixedWidth(140)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._time_label)

        layout.addWidget(QLabel("Jump to:"))
        self._time_spinbox = QDoubleSpinBox()
        self._time_spinbox.setSuffix(" s")
        self._time_spinbox.setDecimals(1)
        self._time_spinbox.setMinimum(0)
        self._time_spinbox.setMaximum(0)
        self._time_spinbox.setFixedWidth(130)
        layout.addWidget(self._time_spinbox)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._prev_button.clicked.connect(self._on_prev_clicked)
        self._next_button.clicked.connect(self._on_next_clicked)
        self._scrollbar.valueChanged.connect(
            self._on_scrollbar_changed
        )
        self._time_spinbox.editingFinished.connect(
            self._on_time_jump
        )

    def configure(
        self,
        n_pages: int,
        total_duration: float,
        time_offset: float = 0.0,
    ) -> None:
        """Configure navigation for the current dataset.

        Args:
            n_pages: Total number of pages.
            total_duration: Total duration in seconds.
            time_offset: Time in seconds corresponding to sample 0.
        """
        self._n_pages = n_pages
        self._total_duration = total_duration

        self._scrollbar.setMaximum(max(0, n_pages - 1))
        self._time_spinbox.setMinimum(time_offset)
        self._time_spinbox.setMaximum(time_offset + total_duration)

        self._update_display()

    def set_current_page(
        self, page_index: int, start_time: float, end_time: float
    ) -> None:
        """Update display to reflect current page.

        Args:
            page_index: Current page index.
            start_time: Start time of current page in seconds.
            end_time: End time of current page in seconds.
        """
        self._current_page = page_index

        self._scrollbar.blockSignals(True)
        self._scrollbar.setValue(page_index)
        self._scrollbar.blockSignals(False)

        self._page_label.setText(
            f"Page {page_index + 1}/{self._n_pages}"
        )
        self._time_label.setText(
            f"{start_time:.1f} - {end_time:.1f} s"
        )

        self._prev_button.setEnabled(page_index > 0)
        self._next_button.setEnabled(
            page_index < self._n_pages - 1
        )

    def _update_display(self) -> None:
        """Update the display based on current state."""
        self._page_label.setText(
            f"Page {self._current_page + 1}/{self._n_pages}"
        )
        self._prev_button.setEnabled(self._current_page > 0)
        self._next_button.setEnabled(
            self._current_page < self._n_pages - 1
        )

    def _on_prev_clicked(self) -> None:
        if self._current_page > 0:
            self.page_requested.emit(self._current_page - 1)

    def _on_next_clicked(self) -> None:
        if self._current_page < self._n_pages - 1:
            self.page_requested.emit(self._current_page + 1)

    def _on_scrollbar_changed(self, value: int) -> None:
        # Debounce: restart the timer on each tick so we only emit
        # once the user stops dragging.
        self._scroll_timer.start()

    def _emit_scrollbar_page(self) -> None:
        """Emit page_requested for the current scrollbar position."""
        value = self._scrollbar.value()
        if value != self._current_page:
            self.page_requested.emit(value)

    def _on_time_jump(self) -> None:
        self.time_requested.emit(self._time_spinbox.value())

    def set_pending_page(self, page_index: int) -> None:
        """Eagerly update internal state for a requested page.

        Called immediately when a page is requested (before loading
        completes) so that subsequent button clicks use the correct
        base page. ``set_current_page`` will later fill in the full
        time-range display once the data is loaded.

        Args:
            page_index: The page that has been requested.
        """
        self._current_page = page_index

        self._scrollbar.blockSignals(True)
        self._scrollbar.setValue(page_index)
        self._scrollbar.blockSignals(False)

        self._update_display()

    @property
    def current_page(self) -> int:
        return self._current_page
