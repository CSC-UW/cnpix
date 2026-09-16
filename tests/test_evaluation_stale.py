import json

import pytest

from cnpix.evaluation import paths


def test_check_not_stale_passes_without_marker(tmp_path):
    paths.check_not_stale(tmp_path)


def test_check_not_stale_raises_with_reason_and_replacement(tmp_path):
    (tmp_path / paths.STALE_MARKER).write_text(
        json.dumps({"reason": "matches hypnogram of 2026-03-09", "replacement": "/new/dir"})
    )
    with pytest.raises(paths.StaleDataError, match="2026-03-09.*\\/new\\/dir"):
        paths.check_not_stale(tmp_path)
    paths.check_not_stale(tmp_path, allow_stale=True)


def test_check_not_stale_handles_unreadable_marker(tmp_path):
    (tmp_path / paths.STALE_MARKER).write_text("{not json")
    with pytest.raises(paths.StaleDataError, match="unreadable"):
        paths.check_not_stale(tmp_path)
