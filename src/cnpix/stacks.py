"""OFF-period image stacks: where they live, their shape, and how to open them.

Stacks are written by ``cnpix_local_sleep.stacks.write`` into the ``samoffs``
project at ``<experiment>/<subject>/probe=<probe>/condition=<condition>/`` as
an OME-Zarr store (``off_stacks.ome.zarr``: series ``0`` AP image, ``1``
spikes, ``2`` LFP, each with its full-resolution array at ``<series>/0``) plus
``timestamps.zarr`` of shape ``(n_chunks, samples_per_chunk)``.

The readers refuse a directory carrying a ``STALE.json`` marker (see
:func:`cnpix.evaluation.paths.check_not_stale`) unless ``allow_stale=True``;
:func:`stack_dir` does not check, so writers can create directories freely.
"""

from __future__ import annotations

import json
import pathlib

from cnpix.evaluation import paths

STACK_NAME = "off_stacks.ome.zarr"
TIMESTAMPS_NAME = "timestamps.zarr"
SERIES = {"ap_data": "0", "spikes": "1", "lfp": "2"}


def stack_dir(
    subject: str,
    probe: str,
    condition: str,
    *,
    structure: str | None = None,
    project: str = paths.MODEL_LABELS_PROJECT,
) -> pathlib.Path:
    """Directory holding one recording's stack and timestamps (no stale check)."""
    return paths.label_dir(
        project, subject, probe=probe, structure=structure, condition=condition
    )


def stack_path(
    subject: str,
    probe: str,
    condition: str,
    *,
    structure: str | None = None,
    allow_stale: bool = False,
) -> pathlib.Path:
    """The ``off_stacks.ome.zarr`` store for one recording."""
    d = stack_dir(subject, probe, condition, structure=structure)
    paths.check_not_stale(d, allow_stale=allow_stale)
    return d / STACK_NAME


def timestamps_path(
    subject: str,
    probe: str,
    condition: str,
    *,
    structure: str | None = None,
    allow_stale: bool = False,
) -> pathlib.Path:
    """The ``timestamps.zarr`` array beside the stack."""
    d = stack_dir(subject, probe, condition, structure=structure)
    paths.check_not_stale(d, allow_stale=allow_stale)
    return d / TIMESTAMPS_NAME


def zarr_shape(store: pathlib.Path) -> tuple[int, ...]:
    """Shape of an OME-Zarr store's series-0 full-resolution array, from metadata only."""
    meta = pathlib.Path(store) / "0" / "0" / "zarr.json"
    if not meta.exists():
        meta = pathlib.Path(store) / "0" / "0" / ".zarray"
    return tuple(json.loads(meta.read_text())["shape"])


def stack_shape(
    subject: str, probe: str, condition: str, *, allow_stale: bool = False
) -> tuple[int, ...]:
    """``(n_chunks, rows, cols)`` of one recording's AP stack, without opening it."""
    return zarr_shape(stack_path(subject, probe, condition, allow_stale=allow_stale))


def open_stack(
    subject: str, probe: str, condition: str, *, allow_stale: bool = False
) -> dict:
    """Open one recording's stack read-only.

    Returns the full-resolution zarr arrays keyed ``ap_data``, ``spikes`` and
    ``lfp`` for whichever series the store holds.
    """
    import zarr

    root = zarr.open(
        str(stack_path(subject, probe, condition, allow_stale=allow_stale)), mode="r"
    )
    out = {}
    for name, series in SERIES.items():
        if series in root:
            group = root[series]
            out[name] = group["0"] if "0" in group else group  # noqa: SIM401
    return out
