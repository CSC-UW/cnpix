"""LRU cache for loaded pages."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class PageData:
    """All data for one viewer page, ready for display."""

    csd_values: np.ndarray  # (n_samples, n_csd_channels)
    csd_times: np.ndarray  # (n_samples,)
    lfp_values: np.ndarray  # (n_samples, n_lfp_channels)
    lfp_times: np.ndarray  # (n_samples,)
    state_rgba: np.ndarray | None  # (n_samples, 4) or None

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        total = (
            self.csd_values.nbytes
            + self.csd_times.nbytes
            + self.lfp_values.nbytes
            + self.lfp_times.nbytes
        )
        if self.state_rgba is not None:
            total += self.state_rgba.nbytes
        return total


@dataclass
class CachedPage:
    """A cached page with access timestamp."""

    page_index: int
    data: PageData
    timestamp: float  # When this was last accessed


class PageCache:
    """LRU cache for loaded pages.

    Thread-safe cache that stores a limited number of pages and evicts
    the least recently used page when capacity is exceeded.
    """

    def __init__(self, max_pages: int = 5):
        """Initialize the cache.

        Args:
            max_pages: Maximum number of pages to cache.
        """
        self._cache: dict[int, CachedPage] = {}
        self._max_pages = max_pages
        self._lock = threading.Lock()

    def get(self, page_index: int) -> PageData | None:
        """Get a cached page's data, updating its access time.

        Args:
            page_index: The page index to retrieve.

        Returns:
            The page data, or None if not in cache.
        """
        with self._lock:
            if page_index in self._cache:
                entry = self._cache[page_index]
                entry.timestamp = time.monotonic()
                return entry.data
            return None

    def put(self, page_index: int, data: PageData) -> None:
        """Cache a page, evicting the oldest if necessary.

        Args:
            page_index: The page index.
            data: The page data to cache.
        """
        with self._lock:
            if page_index in self._cache:
                self._cache[page_index] = CachedPage(
                    page_index=page_index,
                    data=data,
                    timestamp=time.monotonic(),
                )
                return

            while len(self._cache) >= self._max_pages:
                oldest_idx = min(
                    self._cache,
                    key=lambda k: self._cache[k].timestamp,
                )
                del self._cache[oldest_idx]

            self._cache[page_index] = CachedPage(
                page_index=page_index,
                data=data,
                timestamp=time.monotonic(),
            )

    def invalidate(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()

    def contains(self, page_index: int) -> bool:
        """Check if a page is in cache.

        Args:
            page_index: The page index to check.

        Returns:
            True if the page is cached.
        """
        with self._lock:
            return page_index in self._cache

    def __len__(self) -> int:
        """Return number of cached pages."""
        with self._lock:
            return len(self._cache)
