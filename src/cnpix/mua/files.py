"""Paths for the whole-probe MUA amplitude envelope.

``mua_traces.zarr`` is written once per (subject, probe) by ``cnpix-mua
write-mua-traces`` and read by every downstream detection method. It sits
outside any method's path system because it has no structure dimension and no
method: it is the shared preprocessed signal.
"""

from __future__ import annotations

import pathlib

import spikeinterface as si
import wisc_ecephys_tools as wet

from cnpix import constants

__all__ = [
    "DEFAULT_MUA_TRACES_PROJECT",
    "get_mua_traces_path",
    "load_mua_traces",
]

#: SpikeGLX project where written ``mua_traces.zarr`` files live.
DEFAULT_MUA_TRACES_PROJECT = "shared_nobak"


def get_mua_traces_path(
    subject: str,
    probe: str,
    mua_traces_project: str = DEFAULT_MUA_TRACES_PROJECT,
    experiment: str = constants.DEFAULT_EXPERIMENT,
) -> pathlib.Path:
    """Path to a pre-saved ``mua_traces.zarr``.

    Args:
        subject: Subject identifier.
        probe: Probe name (e.g., "imec0").
        mua_traces_project: SpikeGLX project containing the zarr file.
        experiment: Experiment name.

    Returns:
        Path to ``{project}/{experiment}/{subject}/{probe}.mua_traces.zarr``.
    """
    return wet.get_sglx_project(mua_traces_project).get_experiment_subject_file(
        experiment,
        subject,
        f"{probe}.mua_traces.zarr",
    )


def load_mua_traces(
    subject: str,
    probe: str,
    *,
    mua_traces_project: str = DEFAULT_MUA_TRACES_PROJECT,
    experiment: str = constants.DEFAULT_EXPERIMENT,
) -> si.BaseRecording:
    """Load the whole-probe MUA envelope for a subject and probe.

    Raises:
        FileNotFoundError: If the zarr has not been written yet.
    """
    path = get_mua_traces_path(subject, probe, mua_traces_project, experiment)
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-saved MUA traces not found at {path}. "
            "Run `cnpix-mua write-mua-traces` first."
        )
    return si.load(path)
