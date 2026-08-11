"""Hive-partitioned paths for OFF-period labels.

A deliberately small path builder covering only what label evaluation needs:
the manual ground-truth NPZs and the per-model prediction NPZs. It is *not* a
replacement for ``offproj.files.get_path``, which encodes the full 13-component
schema for detection outputs.

The component order here is a prefix of that schema and must stay consistent
with it::

    project > experiment > subject > method > model > probe > structure > condition

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
    "experiment_root",
    "label_dir",
]

DEFAULT_EXPERIMENT: str = constants.DEFAULT_EXPERIMENT

#: Manual ground-truth labels are shared (s3-backed); model predictions are not.
MANUAL_LABELS_PROJECT = "offproj_s3"
MODEL_LABELS_PROJECT = "offproj"

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

        label_dir("offproj_s3", "CNPIX15-Claude", probe="imec0",
                  condition="Early.REC.NREM")
        # .../{experiment}/CNPIX15-Claude/probe=imec0/condition=Early.REC.NREM

        label_dir("offproj", "CNPIX15-Claude", method="sam3",
                  model="trained-on-Early.REC.NREM.2026-05-09",
                  probe="imec0", condition="Early.REC.NREM")
        # .../method=sam3/model=trained-on-.../probe=imec0/condition=...
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
