"""Loaders and label-array utilities for OFF-period evaluation.

Method-agnostic ground-truth (manual) label loaders, plus
instance-label QC, channel->stack-row mapping, grid reconciliation, and chunk
selection. Extracted from ``cnpix_local_sleep.morphological.manual_validation`` so all detection
methods share one ground-truth-handling implementation.
"""

from __future__ import annotations

import re

import numpy as np
import scipy.ndimage
import xarray as xr

from cnpix.evaluation import config, paths


_VERSION_RE = re.compile(r"latest|v\d+")


# ======================================================================
# Manual ("ground truth") labels
# ======================================================================


def _manual_labels_filename(version: str) -> str:
    """Map a version label to its on-disk manual-label filename.

    ``"latest"`` resolves to the ``manual_off_labels.npz`` symlink maintained
    by ``cnpix-local-sleep/scripts/update_manual_off_labels_symlinks.py`` (highest ``vN``
    present). Explicit ``"v1"``, ``"v2"``, … pin a specific version file.
    """
    if not _VERSION_RE.fullmatch(version):
        raise ValueError(
            f"version must be 'latest' or 'v<N>' (e.g. 'v1', 'v2'); got {version!r}"
        )
    return (
        "manual_off_labels.npz"
        if version == "latest"
        else f"manual_off_labels_{version}.npz"
    )


def _get_manual_labels_path(
    subject: str, probe: str, condition: str, version: str = "latest"
):
    """Path to a manual-label NPZ for one (subject, probe, condition).

    Files live under the ``samoffs_s3`` project in the Hive-partitioned layout
    ``{subject}/probe={probe}/condition={condition}/{filename}``.
    """
    return paths.label_dir(
        paths.MANUAL_LABELS_PROJECT,
        subject,
        probe=probe,
        condition=condition,
    ) / _manual_labels_filename(version)


def _get_manual_labels_root():
    """Experiment root directory containing manual label files (samoffs_s3)."""
    return paths.experiment_root(paths.MANUAL_LABELS_PROJECT)


def _get_subject_probe_pairs_with_labels(
    condition: str | None = None,
) -> list[tuple[str, str]]:
    """Return (subject, probe) pairs that have manual label files.

    Parameters
    ----------
    condition
        If given, restrict to pairs that have a manual label for that
        condition. If ``None`` (default), any condition counts.
    """
    root = _get_manual_labels_root()
    cond_glob = condition if condition is not None else "*"
    pairs = set()
    for f in root.glob(
        f"*/probe=*/condition={cond_glob}/manual_off_labels_v*.npz"
    ):
        # .../{subject}/probe={probe}/condition={condition}/manual_off_labels_v*.npz
        subject = f.parents[2].name
        probe = f.parents[1].name.removeprefix("probe=")
        pairs.add((subject, probe))
    return sorted(pairs)


def load_manual_labels(
    subject: str,
    probe: str,
    condition: str = config.NREM_CONDITION,
    version: str = "latest",
    *,
    allow_stale: bool = False,
) -> np.ndarray:
    """Load a manual label array from its NPZ file.

    Raises :class:`cnpix.evaluation.paths.StaleDataError` if the label directory
    carries a ``STALE.json`` marker, unless ``allow_stale=True``.

    Returns
    -------
    np.ndarray
        Label array ``(n_chunks, n_rows, samples_per_chunk)``, int32 instance
        IDs (0 = background).
    """
    path = _get_manual_labels_path(subject, probe, condition, version=version)
    paths.check_not_stale(path.parent, allow_stale=allow_stale)
    npz = np.load(path)
    return npz[list(npz.keys())[0]]


# ======================================================================
# Grid reconciliation and chunk selection
# ======================================================================


def reconcile_to_common_grid(*arrays: np.ndarray) -> list[np.ndarray]:
    """Crop label arrays to a common ``(n_chunks, n_rows, n_samples)`` grid.

    Takes the top-left origin (``arr[:c, :r, :s]``): label arrays are
    top-anchored, so a short one is padded at the *bottom* of the y-axis.
    A no-op when shapes already match;
    resilient to off-by-one chunk/sample differences across the model cohort.
    """
    if not arrays:
        raise ValueError("reconcile_to_common_grid requires at least one array")
    c = min(a.shape[0] for a in arrays)
    r = min(a.shape[1] for a in arrays)
    s = min(a.shape[2] for a in arrays)
    return [a[:c, :r, :s] for a in arrays]


def select_chunks(manual: np.ndarray, mode: str) -> np.ndarray:
    """Return the chunk indices to evaluate for a given ground-truth convention.

    Parameters
    ----------
    manual
        Manual label array ``(n_chunks, n_rows, n_samples)``.
    mode
        ``"labeled"`` — only chunks containing a manual label (NREM: unlabeled
        images were not inspected). ``"all"`` — every chunk (Wake: every image
        was inspected, so unlabeled = true negative).
    """
    if mode == "labeled":
        return np.where(np.any(manual > 0, axis=(1, 2)))[0]
    if mode == "all":
        return np.arange(manual.shape[0])
    raise ValueError(f"mode must be 'labeled' or 'all'; got {mode!r}")


# ======================================================================
# Instance-label QC (split non-contiguous labels, drop stray marks)
# ======================================================================


class _UnionFind:
    """Minimal union-find for merging component groups."""

    def __init__(self):
        self._parent: dict = {}

    def find(self, x):
        while self._parent.get(x, x) != x:
            self._parent[x] = self._parent.get(self._parent[x], self._parent[x])
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


_STRUCT_8CONN = np.ones((3, 3), dtype=int)


def chunk_contiguity(stack_times: np.ndarray) -> np.ndarray:
    """Whether each chunk starts one sample after the previous chunk ends.

    ``stack_times`` is ``(n_chunks, samples_per_chunk)``. Returns a boolean array of
    length ``n_chunks - 1`` for :func:`qc_and_fix_labels`'s ``contiguous``.
    """
    dt = np.median(np.diff(stack_times, axis=1))
    gap = stack_times[1:, 0] - stack_times[:-1, -1]
    return np.abs(gap - dt) < 0.5 * dt


def qc_and_fix_labels(
    labels: np.ndarray,
    *,
    min_component_size: int = 50,
    contiguous: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Check label contiguity and fix non-contiguous labels.

    Each unique label should correspond to a single connected component,
    possibly spanning consecutive chunk boundaries. In order:

    1. Labels that have multiple disconnected groups are split into separate
       labels; groups below ``min_component_size`` pixels are removed as stray
       annotation marks.
    2. Different labels that touch across a chunk boundary are merged into the
       smaller label: one event continued into the next chunk under a new id.
    3. Labels below ``min_component_size`` pixels in total are removed as stray
       marks, even when they carry their own id.

    Uses 8-connectivity. Chunk ``i + 1`` continues chunk ``i`` only where
    ``contiguous[i]`` is True (see :func:`chunk_contiguity`); ``None`` treats
    every pair of consecutive chunks as contiguous.

    Returns
    -------
    fixed_labels
        Corrected label array (same shape as input).
    violations
        List of dicts, each with a ``kind`` and the ``original_label``:
        ``"split"`` (keys ``chunks``, ``n_groups_after_merge``, ``n_removed``,
        ``n_split``, ``removed_sizes``), ``"boundary_merge"`` (``merged_into``,
        ``chunks``) and ``"stray"`` (``removed_sizes``).
    """
    n_boundaries = max(labels.shape[0] - 1, 0)
    if contiguous is None:
        contiguous = np.ones(n_boundaries, dtype=bool)
    contiguous = np.asarray(contiguous, dtype=bool)
    if contiguous.shape != (n_boundaries,):
        raise ValueError(f"contiguous must have shape ({n_boundaries},); got {contiguous.shape}")
    fixed = labels.copy()
    # Restrict all per-label work to chunks that actually contain a label, and
    # build a label -> sorted-chunks map in a single pass over those chunks. This
    # avoids an O(n_labels x full_array) cost (a `labels == lbl` full-array scan
    # per label), which otherwise dominates runtime on full-recording
    # (n_chunks, n_rows, n_samples) arrays where only a few chunks are labeled.
    labeled_chunks = np.where(np.any(labels > 0, axis=(1, 2)))[0]
    label_to_chunks: dict[int, list[int]] = {}
    for chunk_ix in labeled_chunks:
        present = np.unique(labels[chunk_ix])
        for lbl in present[present > 0]:
            label_to_chunks.setdefault(int(lbl), []).append(int(chunk_ix))

    unique_labels = sorted(label_to_chunks)
    next_label = (max(unique_labels) + 1) if unique_labels else 1
    violations = []

    for lbl in unique_labels:
        chunks_with_label = np.array(label_to_chunks[lbl])  # sorted ascending

        # Phase 1: collect components per chunk with 8-connectivity.
        components = []  # (chunk_ix, comp_id, component_img, n_pixels)
        comp_imgs = {}  # chunk_ix -> component_img
        for chunk_ix in chunks_with_label:
            chunk_mask = labels[chunk_ix] == lbl
            component_img, n_comp = scipy.ndimage.label(
                chunk_mask,
                structure=_STRUCT_8CONN,
            )
            comp_imgs[chunk_ix] = component_img
            for comp_id in range(1, n_comp + 1):
                n_pixels = int((component_img == comp_id).sum())
                components.append((chunk_ix, comp_id, component_img, n_pixels))

        # Phase 2: merge boundary-connected components across consecutive
        # chunks using union-find.
        uf = _UnionFind()
        comp_keys = [(chunk_ix, comp_id) for chunk_ix, comp_id, _, _ in components]
        for i in range(len(chunks_with_label) - 1):
            ci = chunks_with_label[i]
            cj = chunks_with_label[i + 1]
            if cj != ci + 1 or not contiguous[ci]:
                continue  # non-consecutive or non-contiguous chunks can't connect
            img_i = comp_imgs[ci]
            img_j = comp_imgs[cj]
            # 8-connectivity across the chunk boundary: a pixel at
            # (row, last_col) in chunk i connects to (row-1, 0), (row, 0),
            # and (row+1, 0) in chunk i+1.
            last_col_i = img_i[:, -1]
            first_col_j = img_j[:, 0]
            n_rows = img_i.shape[0]
            for row in range(n_rows):
                ci_comp = last_col_i[row]
                if ci_comp == 0:
                    continue
                for drow in (-1, 0, 1):
                    adj_row = row + drow
                    if 0 <= adj_row < n_rows:
                        cj_comp = first_col_j[adj_row]
                        if cj_comp > 0:
                            uf.union(
                                (ci, ci_comp),
                                (cj, cj_comp),
                            )

        # Build merged groups.
        groups: dict[tuple, list[int]] = {}
        for idx, key in enumerate(comp_keys):
            root = uf.find(key)
            groups.setdefault(root, []).append(idx)

        # Compute total pixel count per group.
        group_sizes = {
            root: sum(components[i][3] for i in members)
            for root, members in groups.items()
        }

        # Phase 3: classify groups. Only one group total — nothing to fix.
        if len(groups) <= 1:
            continue

        # Multiple groups: separate into large (keep/split) and small (remove
        # as stray marks). The largest group always survives.
        sorted_roots = sorted(
            groups,
            key=lambda r: group_sizes[r],
            reverse=True,
        )
        largest_root = sorted_roots[0]
        small_groups = [
            r for r in sorted_roots[1:] if group_sizes[r] < min_component_size
        ]
        large_groups = [
            r
            for r in sorted_roots
            if group_sizes[r] >= min_component_size or r is largest_root
        ]

        n_removed = len(small_groups)
        removed_sizes = sorted(
            [group_sizes[r] for r in small_groups],
            reverse=True,
        )

        # Remove small groups (set to background).
        for root in small_groups:
            for idx in groups[root]:
                chunk_ix, comp_id, component_img, _ = components[idx]
                fixed[chunk_ix][component_img == comp_id] = 0

        # Split large groups: largest keeps original label, rest get new labels.
        n_split = 0
        if len(large_groups) > 1:
            for root in large_groups[1:]:
                for idx in groups[root]:
                    chunk_ix, comp_id, component_img, _ = components[idx]
                    fixed[chunk_ix][component_img == comp_id] = next_label
                n_split += 1
                next_label += 1

        violations.append(
            {
                "kind": "split",
                "original_label": int(lbl),
                "chunks": chunks_with_label.tolist(),
                "n_groups_after_merge": len(groups),
                "n_removed": n_removed,
                "n_split": n_split,
                "removed_sizes": removed_sizes,
            }
        )

    violations += _merge_across_boundaries(fixed, contiguous)
    violations += _remove_stray_labels(fixed, min_component_size)
    return fixed, violations


def _labeled_chunks(labels: np.ndarray) -> np.ndarray:
    return np.where(np.any(labels > 0, axis=(1, 2)))[0]


def _merge_across_boundaries(fixed: np.ndarray, contiguous: np.ndarray) -> list[dict]:
    """Relabel, in place, distinct labels touching across a contiguous chunk boundary."""
    labeled = set(_labeled_chunks(fixed).tolist())
    uf = _UnionFind()
    where: dict[int, set[int]] = {}
    for ci in sorted(labeled):
        cj = ci + 1
        if cj not in labeled or not contiguous[ci]:
            continue
        left, right = fixed[ci][:, -1], fixed[cj][:, 0]
        for drow in (-1, 0, 1):
            a = left[max(-drow, 0): len(left) - max(drow, 0)]
            b = right[max(drow, 0): len(right) - max(-drow, 0)]
            m = (a > 0) & (b > 0) & (a != b)
            for x, y in zip(a[m].tolist(), b[m].tolist()):
                uf.union(x, y)
                where.setdefault(x, set()).update((ci, cj))
                where.setdefault(y, set()).update((ci, cj))
    groups: dict[int, list[int]] = {}
    for lbl in where:
        groups.setdefault(uf.find(lbl), []).append(lbl)
    remap = {lbl: min(members) for members in groups.values() for lbl in members if lbl != min(members)}
    if not remap:
        return []
    src = np.fromiter(remap, dtype=fixed.dtype)
    for ci in sorted(labeled):
        img = fixed[ci]
        hit = np.isin(img, src)
        if hit.any():
            for s in np.unique(img[hit]).tolist():
                img[img == s] = remap[s]
    return [
        {"kind": "boundary_merge", "original_label": int(s), "merged_into": int(d),
         "chunks": sorted(where[s])}
        for s, d in sorted(remap.items())
    ]


def _remove_stray_labels(fixed: np.ndarray, min_component_size: int) -> list[dict]:
    """Zero, in place, labels whose total size across chunks is below the floor."""
    labeled = _labeled_chunks(fixed)
    if not len(labeled):
        return []
    n = int(fixed[labeled].max()) + 1
    sizes = np.zeros(n, dtype=np.int64)
    for ci in labeled:
        sizes += np.bincount(fixed[ci].ravel(), minlength=n)
    stray = np.flatnonzero((sizes > 0) & (sizes < min_component_size))
    stray = stray[stray > 0]
    if not len(stray):
        return []
    for ci in labeled:
        img = fixed[ci]
        img[np.isin(img, stray)] = 0
    return [
        {"kind": "stray", "original_label": int(s), "removed_sizes": [int(sizes[s])]}
        for s in stray
    ]


# ======================================================================
# Channel -> stack-row mapping
# ======================================================================


def _build_channel_maps(
    da_full: xr.DataArray,
    da_struct: xr.DataArray,
    da_det: xr.DataArray,
    n_label_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map structure and detection channels to boolean stack-row masks.

    Converts channel coordinates to flipped stack y-rows (row 0 = shallowest in
    label space) and returns boolean masks indicating which label rows belong to
    each scope.

    Returns
    -------
    det_row_mask, struct_row_mask
        Boolean arrays of shape ``(n_label_rows,)``.
    """
    n_full_channels = da_full.sizes["channel"]
    full_channels = da_full.channel.values
    full_chan_to_idx = {ch: i for i, ch in enumerate(full_channels)}

    def _channels_to_stack_rows(channels):
        return np.array(
            [(n_full_channels - 1) - full_chan_to_idx[ch] for ch in channels],
            dtype=np.intp,
        )

    struct_stack_rows = _channels_to_stack_rows(da_struct.channel.values)
    det_stack_rows = _channels_to_stack_rows(da_det.channel.values)

    struct_in_labels = struct_stack_rows[struct_stack_rows < n_label_rows]
    det_in_labels = det_stack_rows[det_stack_rows < n_label_rows]

    det_row_mask = np.zeros(n_label_rows, dtype=bool)
    det_row_mask[det_in_labels] = True

    struct_row_mask = np.zeros(n_label_rows, dtype=bool)
    struct_row_mask[struct_in_labels] = True

    return det_row_mask, struct_row_mask
