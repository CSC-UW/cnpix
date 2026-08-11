"""``cnpix-mua`` command line interface.

Produces and moves around ``mua_traces.zarr``: the whole-probe, rectified,
resampled MUA amplitude envelope that every MUA-based OFF-detection method
starts from.

Typical whole-probe run, staging through fast local storage::

    cnpix-mua stage-in   CNPIX12-Santiago imec0
    cnpix-mua write-mua-traces CNPIX12-Santiago imec0 --n-jobs 56
    cnpix-mua retrieve   CNPIX12-Santiago imec0
    cnpix-mua cleanup    CNPIX12-Santiago imec0
"""

from __future__ import annotations

import click

from cnpix import constants


@click.group()
def main() -> None:
    """Build and stage the MUA amplitude envelope for the CNPIX dataset."""


# =============================================================================
@main.command("write-mua-traces")
@click.argument("subject")
@click.argument("probe")
@click.option(
    "--source-project",
    default="shared_nobak",
    help="SpikeGLX project to load si_recording.zarr from.",
)
@click.option(
    "--saveto-project",
    default="tmp_nvme",
    help="SpikeGLX project to save mua_traces.zarr to.",
)
@click.option(
    "--no-bad-channel-detection",
    is_flag=True,
    help="Skip bad channel detection.",
)
@click.option(
    "--motion-correction",
    is_flag=True,
    help="Apply legacy motion correction (off by default).",
)
@click.option(
    "--bandpass-freq-max",
    default=5000,
    type=int,
    help="Upper bandpass cutoff in Hz.",
)
@click.option(
    "--resample-rate",
    default=500,
    type=int,
    help="Target resampling rate in Hz.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing output.")
@click.option(
    "--chunk-duration",
    default="30s",
    help="Chunk duration for zarr saving.",
)
@click.option(
    "--n-jobs",
    default=1,
    type=int,
    help="Number of parallel workers for saving.",
)
@click.option(
    "--max-threads-per-worker",
    default=2,
    type=int,
    help="Max threads per parallel worker.",
)
def write_mua_traces_cmd(
    subject: str,
    probe: str,
    source_project: str,
    saveto_project: str,
    no_bad_channel_detection: bool,
    motion_correction: bool,
    bandpass_freq_max: int,
    resample_rate: int,
    overwrite: bool,
    chunk_duration: str,
    n_jobs: int,
    max_threads_per_worker: int,
):
    """Build MUA preprocessing chain and save to zarr.

    Loads a compressed si_recording.zarr, applies the full preprocessing
    chain (bandpass, phase shift, bad channel interpolation, highpass
    spatial filter, motion correction, rectify, resample), and writes the
    result as mua_traces.zarr.

    ~34,000s (~9.4h) for 30s chunks with n_jobs=56 and max_threads_per_worker=2 on
    CNPIX12-Santiago imec0 with motion correction.
    """
    import time

    import numpy as np
    import spikeinterface as si
    import spikeinterface.preprocessing as sp
    import wisc_ecephys_tools as wet
    from ecephys.wne.sglx import spikeinterface as sglx_spikeinterface
    from numcodecs import Delta

    from cnpix.mua import files, preprocess

    t_start = time.perf_counter()

    sglx_subject = wet.get_sglx_subject(subject)

    click.echo(f"Loading recording for {subject} / {probe}")
    recording = si.load(
        wet.get_sglx_project(source_project).get_experiment_subject_file(
            constants.DEFAULT_EXPERIMENT,
            subject,
            f"{probe}.si_recording.zarr",
        )
    )

    preprocess_kwargs: dict = dict(
        bandpass_freq_max=bandpass_freq_max,
        resample_rate=resample_rate,
        detect_bad_channels=not no_bad_channel_detection,
        apply_motion_correction=motion_correction,
    )
    if motion_correction:
        _, slices = sglx_spikeinterface.get_recording(
            wet.get_sglx_project("shared"),
            sglx_subject,
            constants.DEFAULT_EXPERIMENT,
            probe,
        )
        preprocess_kwargs.update(sglx_subject=sglx_subject, probe=probe, slices=slices)

    click.echo("Building preprocessing chain")
    rec = preprocess.build_preprocessing_chain(recording, **preprocess_kwargs)

    if rec.get_dtype() == np.float32:
        click.echo("Casting float32 -> int16 for storage")
        rec = sp.astype(rec, dtype="int16")

    savepath = files.get_mua_traces_path(subject, probe, saveto_project)

    click.echo(f"Saving to {savepath}")
    t_save = time.perf_counter()
    rec.save(
        folder=savepath,
        overwrite=overwrite,
        format="zarr",
        chunk_duration=chunk_duration,
        filters_by_dataset={"times": [Delta(dtype="float64")]},
        n_jobs=n_jobs,
        max_threads_per_worker=max_threads_per_worker,
        progress_bar=True,
        verbose=True,
    )
    save_elapsed = time.perf_counter() - t_save
    total_elapsed = time.perf_counter() - t_start

    click.echo(f"Done. Save: {save_elapsed:.1f}s, Total: {total_elapsed:.1f}s")


# =============================================================================
# Staging: stage-in, retrieve, cleanup
# =============================================================================


@main.command("stage-in")
@click.argument("subject")
@click.argument("probe")
@click.option(
    "--source-project",
    default="shared_nobak",
    help="SpikeGLX project to copy si_recording.zarr from.",
)
@click.option(
    "--staging-project",
    default="tmp_nvme",
    help="SpikeGLX project to copy si_recording.zarr to.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing staged copy.")
@click.option(
    "--dry-run", is_flag=True, help="Print what would happen without copying."
)
@click.option(
    "--n-workers",
    default=16,
    type=int,
    help="Number of parallel file transfers.",
)
def stage_in_cmd(
    subject: str,
    probe: str,
    source_project: str,
    staging_project: str,
    overwrite: bool,
    dry_run: bool,
    n_workers: int,
):
    """Copy si_recording.zarr to fast local storage for processing.

    Stages the compressed recording from NFS to NVME (or other fast
    storage) so that write-mua-traces can read from local disk.
    Uses rclone for parallel bulk copy of zarr chunk files.
    """
    from cnpix.mua import staging

    staging.stage_in(
        subject,
        probe,
        source_project=source_project,
        staging_project=staging_project,
        overwrite=overwrite,
        dry_run=dry_run,
        n_workers=n_workers,
    )


@main.command("retrieve")
@click.argument("subject")
@click.argument("probe")
@click.option(
    "--staging-project",
    default="tmp_nvme",
    help="SpikeGLX project to copy mua_traces.zarr from.",
)
@click.option(
    "--dest-project",
    default="shared_nobak",
    help="SpikeGLX project to copy mua_traces.zarr to.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing destination copy.")
@click.option(
    "--dry-run", is_flag=True, help="Print what would happen without copying."
)
@click.option(
    "--n-workers",
    default=16,
    type=int,
    help="Number of parallel file transfers.",
)
def retrieve_cmd(
    subject: str,
    probe: str,
    staging_project: str,
    dest_project: str,
    overwrite: bool,
    dry_run: bool,
    n_workers: int,
):
    """Copy mua_traces.zarr back from fast local storage to archival NFS.

    Retrieves the processing output from NVME to NFS for long-term
    storage. Uses rclone for parallel bulk copy.
    """
    from cnpix.mua import staging

    staging.retrieve(
        subject,
        probe,
        staging_project=staging_project,
        dest_project=dest_project,
        overwrite=overwrite,
        dry_run=dry_run,
        n_workers=n_workers,
    )


@main.command("cleanup")
@click.argument("subject")
@click.argument("probe")
@click.option(
    "--project",
    default="tmp_nvme",
    help="SpikeGLX project to clean up.",
)
@click.option(
    "--keep-si-recording",
    is_flag=True,
    help="Do not remove the staged si_recording.zarr.",
)
@click.option(
    "--keep-mua-traces",
    is_flag=True,
    help="Do not remove the mua_traces.zarr.",
)
@click.option(
    "--dry-run", is_flag=True, help="Print what would happen without deleting."
)
def cleanup_cmd(
    subject: str,
    probe: str,
    project: str,
    keep_si_recording: bool,
    keep_mua_traces: bool,
    dry_run: bool,
):
    """Remove staged zarr files from fast local storage.

    After processing and retrieval, clean up NVME to free space
    for the next subject-probe.
    """
    from cnpix.mua import staging

    staging.cleanup(
        subject,
        probe,
        project=project,
        remove_si_recording=not keep_si_recording,
        remove_mua_traces=not keep_mua_traces,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
