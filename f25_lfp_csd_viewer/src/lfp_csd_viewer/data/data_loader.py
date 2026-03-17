"""Background thread for loading page data from lazy xarray sources."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field

from PySide6.QtCore import QThread, Signal

import ecephys.hypnogram as hyp

from lfp_csd_viewer.data.cache import PageCache, PageData
from lfp_csd_viewer.data.xarray_source import XarrayDataSource
from lfp_csd_viewer.widgets.csd_view import get_state_colors

logger = logging.getLogger(__name__)


@dataclass(order=True)
class LoadRequest:
    """Request to load a page.

    Ordering is by priority (lower number = higher priority).
    """

    priority: int
    page_index: int = field(compare=False)


class DataLoaderThread(QThread):
    """Background thread that loads CSD + LFP pages on demand.

    Processes load requests from a priority queue, reads time slices
    from two ``XarrayDataSource`` objects (CSD and LFP), bundles the
    results into ``PageData``, caches them, and emits a signal for the
    main thread to display.
    """

    page_loaded = Signal(int, object)  # page_index, PageData
    loading_started = Signal(int)  # page_index
    error_occurred = Signal(str)

    def __init__(
        self,
        csd_source: XarrayDataSource,
        lfp_source: XarrayDataSource,
        cache: PageCache,
        page_duration: float,
        n_pages: int,
        hypnogram: hyp.FloatHypnogram | None,
    ):
        """Initialize the loader thread.

        Args:
            csd_source: Lazy CSD data source.
            lfp_source: Lazy LFP data source.
            cache: The page cache.
            page_duration: Duration of each page in seconds.
            n_pages: Total number of pages.
            hypnogram: Optional hypnogram for state strip coloring.
        """
        super().__init__()
        self._csd_source = csd_source
        self._lfp_source = lfp_source
        self._cache = cache
        self._page_duration = page_duration
        self._n_pages = n_pages
        self._t_start = csd_source.metadata.t_start
        self._hypnogram = hypnogram
        self._request_queue: queue.PriorityQueue[LoadRequest] = (
            queue.PriorityQueue()
        )
        self._stop_flag = threading.Event()
        self._pending_pages: set[int] = set()
        self._pending_lock = threading.Lock()

    def request_page(
        self, page_index: int, prefetch: bool = False
    ) -> None:
        """Queue a page load request.

        No-op if the page is already cached or pending.

        Args:
            page_index: The page to load.
            prefetch: If True, use lower priority.
        """
        if page_index < 0 or page_index >= self._n_pages:
            return

        if self._cache.contains(page_index):
            return

        with self._pending_lock:
            if page_index in self._pending_pages:
                return
            self._pending_pages.add(page_index)

        priority = 1 if prefetch else 0
        self._request_queue.put(
            LoadRequest(priority=priority, page_index=page_index)
        )

    def request_page_with_prefetch(self, page_index: int) -> None:
        """Request a page and prefetch adjacent pages.

        Args:
            page_index: The main page to load.
        """
        self.request_page(page_index, prefetch=False)
        self.request_page(page_index + 1, prefetch=True)
        self.request_page(page_index - 1, prefetch=True)

    def set_paging(
        self, page_duration: float, n_pages: int
    ) -> None:
        """Update page duration and count, invalidating cache.

        Args:
            page_duration: New page duration in seconds.
            n_pages: New total number of pages.
        """
        self._page_duration = page_duration
        self._n_pages = n_pages
        self._cache.invalidate()
        self.cancel_pending()

    def cancel_pending(self) -> None:
        """Cancel all pending requests."""
        with self._pending_lock:
            self._pending_pages.clear()
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                break

    def run(self) -> None:
        """Main thread loop: process load requests."""
        while not self._stop_flag.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            page_index = request.page_index

            with self._pending_lock:
                if page_index not in self._pending_pages:
                    continue

            if self._cache.contains(page_index):
                with self._pending_lock:
                    self._pending_pages.discard(page_index)
                continue

            try:
                self.loading_started.emit(page_index)
                page_data = self._load_page(page_index)
                self._cache.put(page_index, page_data)

                with self._pending_lock:
                    self._pending_pages.discard(page_index)

                self.page_loaded.emit(page_index, page_data)

            except Exception as e:
                with self._pending_lock:
                    self._pending_pages.discard(page_index)
                self.error_occurred.emit(
                    f"Failed to load page {page_index}: {e}"
                )
                logger.exception(
                    "Failed to load page %d", page_index
                )

    def _load_page(self, page_index: int) -> PageData:
        """Load one page of CSD + LFP data.

        Args:
            page_index: Zero-based page index.

        Returns:
            A PageData with all arrays for this page.
        """
        t_start = self._t_start + page_index * self._page_duration
        t_end = t_start + self._page_duration

        csd_values, csd_times = self._csd_source.read_time_slice(
            t_start, t_end
        )
        lfp_values, lfp_times = self._lfp_source.read_time_slice(
            t_start, t_end
        )

        state_rgba = None
        if self._hypnogram is not None and len(csd_times) > 0:
            state_rgba = get_state_colors(
                self._hypnogram, csd_times
            )

        return PageData(
            csd_values=csd_values,
            csd_times=csd_times,
            lfp_values=lfp_values,
            lfp_times=lfp_times,
            state_rgba=state_rgba,
        )

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_flag.set()
        self.cancel_pending()
