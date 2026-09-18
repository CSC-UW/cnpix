"""Hive-partitioned paths for OFF-period labels.

A deliberately small path builder covering only what label evaluation needs:
the manual ground-truth NPZs and the per-model prediction NPZs. It is *not* a
replacement for ``offproj.files.get_path``, which encodes the full 13-component
schema for detection outputs.

The component order here is a prefix of that schema and must stay consistent
with it::

    project > experiment > subject > method > model > probe > structure > condition

Inside the ``samoffs`` project the ``method`` component is not used (that
project is SAM3-only), so model label paths run straight from ``subject`` to
``model``.

Only ``offproj.files`` may add components; if this list and that schema ever
disagree about ordering, that schema wins.
"""

from __future__ import annotations

import pathlib

import wisc_ecephys_tools as wet

from cnpix import constants

__all__ = [
    "DEFAULT_EXPERIMENT",
    "MANUAL_LABELS_PROJECT",
    "MODEL_LABELS_PROJECT",
    "STALE_MARKER",
    "StaleDataError",
    "check_not_stale",
    "experiment_root",
    "label_dir",
]

DEFAULT_EXPERIMENT: str = constants.DEFAULT_EXPERIMENT

#: Manual ground-truth labels are shared (s3-backed); model predictions are not.
MANUAL_LABELS_PROJECT = "samoffs_s3"
MODEL_LABELS_PROJECT = "samoffs"

# Components that take a ``key=value`` directory, in schema order.
_KEYED = ("method", "model", "probe", "structure", "condition")


def experiment_root(
    project: str, experiment: str = DEFAULT_EXPERIMENT
) -> pathlib.Path:
    """Experiment directory of a WNE project."""
    return wet.get_sglx_project(project).get_experiment_directory(experiment)


def label_dir(
    project: str,
    subject: str,
    *,
    experiment: str = DEFAULT_EXPERIMENT,
    method: str | None = None,
    model: str | None = None,
    probe: str | None = None,
    structure: str | None = None,
    condition: str | None = None,
) -> pathlib.Path:
    """Directory holding labels for one recording, in schema order.

    Components left as ``None`` are omitted, so callers get exactly the depth
    they ask for::

        label_dir("samoffs_s3", "CNPIX15-Claude", probe="imec0",
                  condition="Early.REC.NREM")
        # .../{experiment}/CNPIX15-Claude/probe=imec0/condition=Early.REC.NREM

        label_dir("samoffs", "CNPIX15-Claude", model="2026-05-03_nrem-all",
                  probe="imec0", condition="Early.REC.NREM")
        # .../model=2026-05-03_nrem-all/probe=imec0/condition=...
    """
    d = experiment_root(project, experiment) / subject
    values = {
        "method": method,
        "model": model,
        "probe": probe,
        "structure": structure,
        "condition": condition,
    }
    for key in _KEYED:
        value = values[key]
        if value is not None:
            d = d / f"{key}={value}"
    return d


#: Presence of this file in a stack or label directory marks its contents as
#: aligned to a superseded hypnogram (or otherwise unsafe to combine with current
#: data). Readers refuse such directories unless explicitly told otherwise.
STALE_MARKER = "STALE.json"


class StaleDataError(RuntimeError):
    """Raised when a reader is pointed at a directory carrying a STALE marker."""


def check_not_stale(directory: pathlib.Path, *, allow_stale: bool = False) -> None:
    """Raise :class:`StaleDataError` if ``directory`` holds a ``STALE.json``.

    The marker is written by hand (see the label-organization developer note)
    and holds at least ``reason``; the message quotes it so the caller learns
    what the data actually match. ``allow_stale=True`` bypasses the check for
    code that knowingly works with the stale grid.
    """
    if allow_stale:
        return
    marker = pathlib.Path(directory) / STALE_MARKER
    if not marker.exists():
        return
    import json

    try:
        info = json.loads(marker.read_text())
        reason = info.get("reason", "(no reason recorded)")
        replacement = info.get("replacement")
    except (OSError, ValueError):
        reason, replacement = "(unreadable marker)", None
    msg = f"{directory} is marked stale: {reason}"
    if replacement:
        msg += f" Use {replacement} instead,"
    msg += " or pass allow_stale=True to load it deliberately."
    raise StaleDataError(msg)
