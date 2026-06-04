"""Code for working with subjects from Findlay et al. (2025).

These are freely-moving rats with Neuropixels 1.0 probes inserted through mPPC,
hippocampus, thalamus, and possibly other areas."""

from typing import TYPE_CHECKING

import ecephys.wne.utils

import wisc_ecephys_tools as wet

if TYPE_CHECKING:
    from typing import Literal

    import xarray as xr

NOD = "novel_objects_deprivation"
COW = "conveyor_over_water"
CTN = "conveyor_then_novelty"

S3 = wet.get_sglx_project("shared")
NB = wet.get_sglx_project("shared_nobak")

WHITE_MATTER_THICKNESS = 200  # Thickness of the white matter between hippocampus and
# cortex, in microns.


def _get_manifest() -> dict[str, list[str]]:
    """Get a mapping from subjects to the experiments they participated in."""
    return {
        "CNPIX2-Segundo": [NOD],
        "CNPIX3-Valentino": [NOD],
        "CNPIX4-Doppio": [NOD],
        "CNPIX5-Alessandro": [NOD],
        "CNPIX6-Eugene": [NOD],
        "CNPIX8-Allan": [NOD],
        "CNPIX9-Luigi": [NOD],
        "CNPIX10-Charles": [NOD],
        "CNPIX11-Adrian": [NOD, COW],
        "CNPIX12-Santiago": [NOD, COW],
        "CNPIX14-Francis": [NOD, COW],
        "CNPIX15-Claude": [NOD, COW, CTN],
        "CNPIX17-Hans": [NOD, COW, CTN],
        "CNPIX18-Pier": [NOD, COW, CTN],
        "CNPIX19-Otto": [NOD, COW, CTN],
        "CNPIX20-Ernst": [NOD, COW, CTN],
    }


def get_subjects(experiment: str | None = None) -> list[str]:
    """Get a list of subjects that participated in the given experiment."""
    manifest = _get_manifest()
    if experiment is None:
        return list(manifest.keys())
    return [
        subject
        for subject, experiments in manifest.items()
        if experiment in experiments
    ]


def get_subject_experiment_list(
    experiments=[NOD, COW, CTN],
) -> list[tuple[str, str]]:
    """Get a list of (subject, experiment) pairs for the given experiments."""
    pairs = []
    for experiment in experiments:
        for subject in get_subjects(experiment):
            pairs.append((subject, experiment))
    return pairs


def open_lfps(
    subject: str, experiment: str, drop_bad_channels: bool = False, **kwargs
) -> xr.DataArray:
    params = S3.load_experiment_subject_params(experiment, subject)
    probe = params["spwrProbe"]
    return ecephys.wne.utils.open_lfps(
        NB,
        subject,
        experiment,
        probe,
        anatomy_proj=S3,
        badchan_proj=S3 if drop_bad_channels else None,
        **kwargs,
    )


def open_cortical_lfps(
    subject: str,
    experiment: str,
    drop_bad_channels: bool = False,
    **kwargs,
) -> xr.DataArray:
    """Open LFPs from cortical channels only. Some white matter or surface channels
    may be included."""
    lfps = open_lfps(subject, experiment, drop_bad_channels=drop_bad_channels, **kwargs)
    params = S3.load_experiment_subject_params(experiment, subject)
    probe = params["spwrProbe"]
    [hc_lo, hc_hi] = params["probes"][probe]["structureBounds"]["hippocampus"]
    is_cortical = lfps["y"] > (hc_hi + WHITE_MATTER_THICKNESS)
    return lfps.sel({"channel": is_cortical})


def open_hippocampal_lfps(
    subject: str,
    experiment: str,
    drop_bad_channels: bool = False,
    **kwargs,
) -> xr.DataArray:
    """Open LFPs from hippocampal channels only. Some white matter channels may be
    included."""
    lfps = open_lfps(subject, experiment, drop_bad_channels=drop_bad_channels, **kwargs)
    params = S3.load_experiment_subject_params(experiment, subject)
    probe = params["spwrProbe"]
    [lo, hi] = params["probes"][probe]["structureBounds"]["hippocampus"]
    is_hippocampal = (lfps["y"] >= lo) & (lfps["y"] <= hi)
    return lfps.sel(channel=is_hippocampal)


def _open_kcsd(
    subject: str,
    experiment: str,
    kind: Literal["cortical", "hippocampal"],
    chunks={},  # Use zarr chunks by default.
    **kwargs,
) -> xr.DataArray:
    import xarray as xr

    kcsd_file = NB.get_experiment_subject_file(experiment, subject, f"{kind}_kcsd.zarr")
    return xr.open_dataarray(kcsd_file, engine="zarr", chunks=chunks, **kwargs)


def open_cortical_kcsd(subject: str, experiment: str) -> xr.DataArray:
    """Open the pre-computed cortical kCSD."""
    return _open_kcsd(subject, experiment, "cortical")


def open_hippocampal_kcsd(subject: str, experiment: str) -> xr.DataArray:
    """Open the pre-computed hippocampal kCSD."""
    return _open_kcsd(subject, experiment, "hippocampal")
