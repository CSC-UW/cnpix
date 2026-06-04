import time
from typing import Literal

import click
import numpy as np
from ecephys.xrsig import core as xrc
from wisc_ecephys_tools.rats import cnd_hgs

import wisc_ecephys_tools as wet
from cnpix import f25

S3 = wet.get_sglx_project("shared")
NB = wet.get_sglx_project("shared_nobak")


def do_subject_experiment(
    subject: str, experiment: str, kind: Literal["cortical", "hippocampal"]
):
    """Requires the following fields to be defined in experiment_params.json...
    spwrProbe: Use this probe for parameter estimation
    probes->imecX->badChannels: List of bad channels.
    probes->imexX->structureBounds->hippocampus: Borders of hippocampus, in um
    """
    t_start = time.time()
    print(f"=== {subject} / {experiment} / {kind} ===")

    # Get estimation period
    print("  Loading hypnogram...")
    params = S3.load_experiment_subject_params(experiment, subject)
    probe = params["spwrProbe"]
    hg = cnd_hgs.load_statistical_condition_hypnograms(subject, experiment, probe)[
        "Early.REC.NREM"
    ]
    t1 = hg["start_time"].min()
    t2 = hg["end_time"].max()
    print(f"  Estimation window: {t1:.1f} - {t2:.1f} s ({(t2 - t1) / 60:.1f} min)")

    # Get LFP
    print(f"  Loading {kind} LFP...")
    t0 = time.time()
    lf = {
        "cortical": f25.open_cortical_lfps(subject, experiment),
        "hippocampal": f25.open_hippocampal_lfps(subject, experiment),
    }[kind]
    lf = lf.sel(time=slice(t1, t2))
    print(f"  Loaded LFP ({lf.sizes['channel']} channels) in {time.time() - t0:.1f}s")

    # Estimate CSD
    bad_chans = np.array(params["probes"][probe]["badChannels"])
    print(f"  Estimating kCSD (L-curve) with {len(bad_chans)} bad channels dropped...")
    t0 = time.time()
    lambdas = np.logspace(-10, -1, 100, base=10)  # These are NOT default kCSD lambdas,
    #                                               contrary to what the kCSD docs say.
    lcurve_kwargs = {
        "cortical": {"lambdas": lambdas[77:]},  # Constrain to a minimum of 0.001
        "hippocampal": None,
    }[kind]
    csd = xrc.kernel_current_source_density(
        lf, drop=bad_chans, do_lcurve=True, lcurve_kwargs=lcurve_kwargs
    )
    print(f"  kCSD estimation done in {time.time() - t0:.1f}s")

    # Store CSD params
    csd_params_file = NB.get_experiment_subject_file(
        experiment, subject, f"{kind}_csd_params.npz"
    )
    print(f"  Saving params to {csd_params_file}")
    np.savez(
        csd_params_file,
        estm_start_time=t1,
        estm_end_time=t2,
        electrode_pitch=csd.pitch_mm,
        xmin=csd.kcsd.xmin,
        xmax=csd.kcsd.xmax,
        n_estm=csd.kcsd.n_estm,
        gdx=csd.kcsd.gdx,
        lambd=csd.kcsd.lambd,
        R=csd.kcsd.R,
        channels_used=csd.channel.values,
        channels_omitted=bad_chans,
        ele_pos=csd.kcsd.ele_pos,
    )
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
    """Estimate kCSD parameters for an experiment."""
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
