import time
from typing import Literal

import click
import numpy as np
from ecephys.xrsig import core as xrc

from cnpix import f25

NB = f25.NB


def do_subject_experiment(
    subject: str, experiment: str, kind: Literal["cortical", "hippocampal"]
):
    t_start = time.time()
    print(f"=== {subject} / {experiment} / {kind} ===")

    # Load kCSD params
    print("  Loading kCSD params...")
    csd_params_file = NB.get_experiment_subject_file(
        experiment, subject, f"{kind}_csd_params.npz"
    )
    csd_params = np.load(csd_params_file)
    kcsd_kwargs = dict(
        drop=csd_params["channels_omitted"],
        do_lcurve=False,
        gdx=csd_params["gdx"],
        R_init=csd_params["R"],
        lambd=csd_params["lambd"],
    )

    # Load LFP
    print("  Loading LFP...")
    t0 = time.time()
    lf = f25.open_lfps(subject, experiment)
    lf = lf.sel(channel=csd_params["channels_used"])
    print(f"  Loaded LFP ({lf.sizes['channel']} channels) in {time.time() - t0:.1f}s")

    # Compute kCSD
    print("  Computing kCSD...")
    t0 = time.time()
    csd = xrc.lazy_mapped_kernel_current_source_density(lf, **kcsd_kwargs)
    print(f"  kCSD graph built in {time.time() - t0:.1f}s")

    # Drop object-dtype coords (anatomy labels from assign_laminar_coordinate)
    obj_coords = [name for name, coord in csd.coords.items() if coord.dtype == object]
    if obj_coords:
        csd = csd.drop_vars(obj_coords)

    # Save to zarr
    zarr_file = NB.get_experiment_subject_file(experiment, subject, f"{kind}_kcsd.zarr")
    print(f"  Saving to {zarr_file}...")
    t0 = time.time()
    csd.to_zarr(zarr_file, compute=True, mode="w")
    print(f"  Saved in {time.time() - t0:.1f}s")

    print(f"  Total: {time.time() - t_start:.1f}s")


def do_experiment(
    experiment: str, kind: Literal["cortical", "hippocampal"] | None = None
):
    subjects = f25.get_subjects(experiment)
    print(f"Running {experiment} for {len(subjects)} subjects")
    t_exp = time.time()
    for i, subject in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] {subject}")
        if kind is None:
            do_subject_experiment(subject, experiment, kind="cortical")
            do_subject_experiment(subject, experiment, kind="hippocampal")
        else:
            do_subject_experiment(subject, experiment, kind=kind)
    print(f"\nAll subjects done in {time.time() - t_exp:.1f}s")


@click.command()
@click.argument("experiment")
@click.option("--subject", default=None, help="Run for a single subject.")
@click.option(
    "--kind",
    type=click.Choice(["cortical", "hippocampal"]),
    default=None,
    help="Run for a single kind. If omitted, run both.",
)
def main(experiment: str, subject: str | None, kind: str | None):
    """Compute full kCSD for an experiment."""
    if subject is not None:
        if kind is None:
            do_subject_experiment(subject, experiment, kind="cortical")
            do_subject_experiment(subject, experiment, kind="hippocampal")
        else:
            do_subject_experiment(subject, experiment, kind=kind)
    else:
        do_experiment(experiment, kind=kind)


if __name__ == "__main__":
    main()
