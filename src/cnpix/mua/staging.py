"""Fast zarr directory staging between storage projects.

Zarr directories contain many small chunk files. Standard ``cp -r`` or
``shutil.copytree`` is slow because of per-file overhead. This module
uses ``rclone`` for parallel multi-file copy, which is ~15x faster than
single-threaded approaches on NFS → NVME transfers.

Benchmarked alternatives (5K files / 15GB, cold NFS → NVME):

- tar-pipe (single-threaded): 55.5s
- ``xargs -P 16 cp`` (parallel, POSIX-only fallback): 3.9s
- ``rclone --transfers=16 --no-check-dest``: 3.6s

``rclone`` was chosen for its speed, built-in progress display, and
automatic directory creation. ``xargs -P cp`` is a viable POSIX-only
fallback if rclone is unavailable.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import time

import wisc_ecephys_tools as wet

from cnpix import constants
from cnpix.mua import files

__all__ = [
    "cleanup",
    "copy_zarr_dir",
    "get_si_recording_path",
    "retrieve",
    "stage_in",
]


def copy_zarr_dir(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
    n_workers: int = 16,
) -> None:
    """Copy a zarr directory using rclone for parallel transfer.

    Uses ``rclone copy`` with ``--no-check-dest`` to skip checksumming
    and ``--transfers=N`` to copy multiple chunk files concurrently.
    This is ~15x faster than single-threaded tar-pipe on NFS.

    Args:
        src: Source zarr directory. Must exist and be a directory.
        dst: Destination zarr directory path. The parent directory will
            be created if it does not exist. If *dst* already exists and
            *overwrite* is False, the copy is skipped.
        overwrite: If True, remove *dst* before copying. If False
            (default), skip if *dst* already exists.
        dry_run: If True, print what would happen but do not copy.
        verbose: If True, print progress messages.
        n_workers: Number of parallel file transfers (default 16).

    Raises:
        FileNotFoundError: If *src* does not exist.
        NotADirectoryError: If *src* exists but is not a directory.
        RuntimeError: If rclone is not installed or the copy fails.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src}")

    if dst.exists():
        if not overwrite:
            if verbose:
                print(f"  Destination already exists, skipping: {dst}")
            return
        if verbose:
            print(f"  Removing existing destination: {dst}")
        if not dry_run:
            shutil.rmtree(dst)

    if verbose:
        print(f"  Copying {src} -> {dst}")
    if dry_run:
        print(f"  [dry run] Would rclone copy with {n_workers} transfers")
        return

    if not shutil.which("rclone"):
        raise RuntimeError(
            "rclone is required for fast parallel zarr copying but was not "
            "found on PATH. Install it: https://rclone.org/install/"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rclone",
        "copy",
        str(src),
        str(dst),
        f"--transfers={n_workers}",
        f"--checkers={n_workers}",
        "--no-check-dest",
    ]
    if verbose:
        cmd.append("--progress")

    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"rclone copy failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}"
        )

    if verbose:
        print(f"  Done in {elapsed:.1f}s")


def get_si_recording_path(
    subject: str,
    probe: str,
    project: str,
) -> pathlib.Path:
    """Get the path to a si_recording.zarr file for a subject/probe.

    Args:
        subject: Subject identifier.
        probe: Probe name (e.g., "imec0").
        project: SpikeGLX project name (e.g., "shared_nobak", "tmp_nvme").

    Returns:
        Path to ``{project_dir}/{experiment}/{subject}/{probe}.si_recording.zarr``.
    """
    return wet.get_sglx_project(project).get_experiment_subject_file(
        constants.DEFAULT_EXPERIMENT,
        subject,
        f"{probe}.si_recording.zarr",
    )


def stage_in(
    subject: str,
    probe: str,
    *,
    source_project: str = "shared_nobak",
    staging_project: str = "tmp_nvme",
    overwrite: bool = False,
    dry_run: bool = False,
    n_workers: int = 16,
) -> pathlib.Path:
    """Copy si_recording.zarr from source project to staging project.

    Args:
        subject: Subject identifier.
        probe: Probe name.
        source_project: Project to copy from (default: ``"shared_nobak"``).
        staging_project: Project to copy to (default: ``"tmp_nvme"``).
        overwrite: Overwrite existing staged copy.
        dry_run: Print what would happen without copying.
        n_workers: Number of parallel file transfers.

    Returns:
        Path to the staged si_recording.zarr.
    """
    src = get_si_recording_path(subject, probe, source_project)
    dst = get_si_recording_path(subject, probe, staging_project)
    print(f"Stage-in: {src} -> {dst}")
    copy_zarr_dir(src, dst, overwrite=overwrite, dry_run=dry_run, n_workers=n_workers)
    return dst


def retrieve(
    subject: str,
    probe: str,
    *,
    staging_project: str = "tmp_nvme",
    dest_project: str = "shared_nobak",
    overwrite: bool = False,
    dry_run: bool = False,
    n_workers: int = 16,
) -> pathlib.Path:
    """Copy mua_traces.zarr from staging project to destination project.

    Args:
        subject: Subject identifier.
        probe: Probe name.
        staging_project: Project to copy from (default: ``"tmp_nvme"``).
        dest_project: Project to copy to (default: ``"shared_nobak"``).
        overwrite: Overwrite existing destination copy.
        dry_run: Print what would happen without copying.
        n_workers: Number of parallel file transfers.

    Returns:
        Path to the retrieved mua_traces.zarr.
    """
    src = files.get_mua_traces_path(subject, probe, staging_project)
    dst = files.get_mua_traces_path(subject, probe, dest_project)
    print(f"Retrieve: {src} -> {dst}")
    copy_zarr_dir(src, dst, overwrite=overwrite, dry_run=dry_run, n_workers=n_workers)
    return dst


def cleanup(
    subject: str,
    probe: str,
    *,
    project: str = "tmp_nvme",
    remove_si_recording: bool = True,
    remove_mua_traces: bool = True,
    dry_run: bool = False,
) -> None:
    """Remove staged zarr files from the staging project.

    Args:
        subject: Subject identifier.
        probe: Probe name.
        project: Project to clean up (default: ``"tmp_nvme"``).
        remove_si_recording: Whether to remove the staged si_recording.zarr.
        remove_mua_traces: Whether to remove the mua_traces.zarr.
        dry_run: Print what would happen without deleting.
    """
    paths_to_remove: list[tuple[str, pathlib.Path]] = []
    if remove_si_recording:
        paths_to_remove.append(
            ("si_recording.zarr", get_si_recording_path(subject, probe, project))
        )
    if remove_mua_traces:
        paths_to_remove.append(
            ("mua_traces.zarr", files.get_mua_traces_path(subject, probe, project))
        )

    for label, path in paths_to_remove:
        if path.exists():
            print(f"  Removing {label}: {path}")
            if not dry_run:
                shutil.rmtree(path)
        else:
            print(f"  {label} not found, skipping: {path}")
