---
title: Porting the unit-metrics and cell-type pipeline out of findlay2025a
updated: 2026-08-06
---

# Porting the unit-metrics and cell-type pipeline into `cnpix`

## Why

Unit quality tiers, ACG metrics, and cell-type labels describe a *unit*, not a
*paper*. They lived in `findlay2025a`, which meant a second analysis either
recomputed them or imported a paper package to get them. Worse, `findlay2025a`
gated its cohort on `core.MANIFEST` — subjects that had a sharp-wave probe —
which silently excluded three subjects whose data are fine.

`cnpix.units` is the project-agnostic home. `findlay2025a` already imported
`get_threshold_kwargs` from `cnpix.units`, so this continues an existing
direction rather than starting a new one.

## What moved

| `findlay2025a` | `cnpix.units` |
|---|---|
| `units/acg.py` | `acg.py` (fitting math verbatim; loaders replaced) |
| `notebooks/pipeline/units/compute_acgs.ipynb` | `acg.compute_acgs` + `pipeline.compute_subject_acgs` |
| `notebooks/pipeline/units/compute_acg_metrics.ipynb` | `acg.compute_acg_metrics` + `pipeline.compute_subject_acg_metrics` |
| `notebooks/pipeline/units/agg_cell_metrics_assign_quality.ipynb` | `metrics.py` + `pipeline.build_cohort_tables` |
| `notebooks/pipeline/units/assign_cell_types.ipynb` | `celltypes.py` + `pipeline.assign_cohort_cell_types` |
| `units/units.py::add_major_regions`, `hippocampus_to_waxholm` | `regions.py` |
| `units/units.py::get_nod_sortings` (manifest-gated) | `sortings.get_sortings` (registry-driven) |

Outputs moved from the `seahorse` project to the new project-agnostic `cnpix`
project (`/Volumes/npx_nfs/nobak/cnpix`), registered in
`wisc_ecephys_tools/.../projects/projects.yaml`.

## Reproduction evidence

Run against the existing `seahorse` artifacts, which were produced by the
original code:

| Stage | Result |
|---|---|
| `celltypes.assign_cell_types` vs `cell_types.pqt` | **4984/4984 identical**, both label columns |
| `regions.add_major_regions` vs `mps_metrics.pqt` `region` | **4984/4984 identical** |
| `metrics.assign_cluster_quality` vs `cluster_quality.pqt` | **4984/4984 identical** |
| ACG computation (CNPIX2-Segundo, recomputed from spike trains) | `isi_shoulder`, `burst_index`, `moment` **bit-identical** (0.0e+00) |
| ACG *fit* parameters (same subject) | **differ slightly** — see below |
| **Full cohort regeneration vs `cell_types.pqt`** | **4984/4984 identical**, both label columns, all 16 shared subjects |

### Cohort-level result: zero label changes

The full 19-subject regeneration reproduces **every** stored label exactly —
4984/4984 on both `narrow_wide_cell_type` and `petersen_cell_type`. The fit drift
described below is real but never moves a unit across a classification boundary.

The regenerated table has **6233** units against the stored 4984. The difference
decomposes cleanly:

- **+897** from the three newly admitted subjects (CNPIX7, CNPIX13, CNPIX16).
- **+352** in the *shared* subjects, caused by `ecephys` commit `9b05ea6`
  (2026-06-12), which lowered the `conservative` required firing-rate floor from
  **0.5 Hz to 0.3 Hz** ("L2/3 neurons can have low FR"). The stored table
  (2025-06-29) predates it. Verified exactly: all 352 added units have a NREM
  firing rate in **[0.300, 0.498] Hz**, and all 4984 retained units are
  **>= 0.501 Hz**. Mostly pyramidal (184), as that commit's rationale predicts,
  plus 14 additional FS cells.

Neither source of difference is attributable to the port. The underlying
`postpro/metrics.csv` files date from 2023 and did not change.

### The fit parameters drift, and that is not the port

With bit-identical ACGs going in, `scipy.optimize.curve_fit` returns slightly
different optima than the stored values: up to ~1.5% relative on `tau_rise` and
~30% on `tau_decay`, while `rsq` differs by only ~3e-4. Equally good fits, a
different local optimum. The most likely cause is `scipy` version drift (the
environment is now 1.17.0); nothing in the ported fitting code differs from the
original, which is why it is reproduced verbatim.

This matters because `tau_rise` is thresholded at 6 ms (cortical) to separate
wide interneurons from pyramidal cells. On CNPIX2-Segundo **zero units changed
class**, though one unit sits right at the boundary (5.989 → 5.949 ms). The
cohort-wide flip count should be recorded here once the full run completes.

## Deliberate behavior changes

1. **Cohort from the registry, not a manifest.** `get_sortings` enumerates every
   subject with a sorting + anatomy + hypnogram: **19**, versus 16 under
   `findlay2025a.core.MANIFEST`. The three additions are `CNPIX7-Giuseppe`,
   `CNPIX13-Al`, `CNPIX16-Walter`.
2. **NaN metrics no longer become "pyramidal".** The original
   `classify_narrow_wide(np.nan)` returned `"wide"` (because `nan <= x` is
   False), and a NaN `tau_rise` then failed the `> 6` test, so a unit with a
   failed ACG fit silently became a putative pyramidal cell. The port returns
   NaN instead. This cannot change any existing label — the current cohort has
   **no** NaN `peak_to_valley` or `tau_rise` in classifiable regions — but ACG
   fit failures do return all-NaN, so the three newly added subjects could have
   hit it.
3. **`assign_cluster_quality` is vectorized.** The original compared each row
   against every other with a per-row mask (O(N²)); this uses a `MultiIndex`.
   Output verified identical.
4. **Correct project type.** `has_sorting` / `has_anatomy` / `has_hypnogram`
   take an `SGLXProject`; the original passed a plain `Project`. Runtime
   behavior was unaffected (only `get_experiment_subject_file` is called).

## A latent bug the manifest was hiding

Removing the cohort manifest immediately surfaced a real defect. On the first
full 19-subject run, `CNPIX7-Giuseppe` — one of the three newly admitted
subjects — died with:

```
ValueError: array must not contain infs or NaNs
```

Cause: cluster `1000074` fired **zero spikes during REM**. `normalize_acgs`
divides every bin by that unit's spike count, so 0/0 gives an all-NaN ACG, and
`scipy.optimize.curve_fit` refuses it. All 4 states are fit independently, so one
silent-in-REM unit killed the entire subject.

Fixed in `fit_acg`, which already contracts to return `(np.nan,) * 9` on a failed
fit — the guard just extends that contract to non-finite input rather than
letting the exception escape. Regression tests in `tests/test_acg.py`.

Two things worth noting:

- **This cannot change any previously computed value.** The guard only triggers
  on non-finite input, which `curve_fit` would have rejected anyway.
- **The unit still gets a cell type.** Labels are derived from NREM, where this
  unit fired 2169 spikes; only its REM-state metrics are NaN. And because
  `classify_petersen` returns NaN on a missing `tau_rise` rather than falling
  through to `"pyramidal"` (see above), a unit that *did* fail its NREM fit would
  be left unlabeled instead of silently mislabeled.

The other 18 subjects completed without incident (~2.2 h wall clock total).

## The old `offproj` cell-type machinery was removed

`offproj/src/offproj/units.py` carried a second, incompatible cell-type scheme:
`CELL_TYPE_FILTERS` splitting FS/RS at **377 µs** peak-to-valley, two-way. With
`cnpix.units` in place that was a live SPOT violation — any unit between 377 and
425 µs was classified differently depending on which module you asked.

It was not load-bearing. `CELL_TYPE_FILTERS` and `assign_cell_type` were
referenced only inside `units.py`; all eight external callers passed
`cell_type="all"`, the no-op `(-inf, inf)` filter. `"FS"` and `"RS"` were never
requested anywhere.

Removed (user-approved 2026-08-06):

- `CELL_TYPE_FILTERS`, `assign_cell_type`, and the `cell_type` parameter of
  `load_sorting` / `load_structure_sorting` / `prepare_unit_raster_cache`.
- The whole ISI chain, whose only purpose was to carry that label:
  `unit_based/pipeline/run_isis.py`, the `unit-based-offs compute-isis` command,
  `files.Files.ISI_MEDIANS`, and `plot_utils.plot_isi_medians` (whose sole caller
  lived in `deprecated/`).
- `offproj/scripts/unit_based/` — **already dead before this change**: it imported
  `run_nrem_detection` and `run_sd_detection`, both deleted in June 2026. Removing
  the ISI step left it with zero working steps.

To get cell types in `offproj` now, join `cnpix.units.load_cell_types()` on
`(subject, experiment, probe, cluster_id)`; `offproj.celltype_firing` does this.

## Gotchas

- **Heavy `ecephys` submodules are not imported by their package `__init__`.**
  Use `from ecephys.units.correlograms import ...`, `from ecephys.wne import
  siutils`, `from ecephys.units import siutils`. Likewise `wisc_ecephys_tools`
  does not auto-import `rats`; use `from wisc_ecephys_tools import rats`.
- **Run commands through the workspace**, not from `cnpix/`. `cnpix` is its own
  uv workspace root, so `uv run` from inside it resolves the standalone
  (git-URL) dependency set and builds a separate `.venv`.
- `offproj` declares `requires-python = ">=3.13"` but depends on `cnpix`, which
  requires `>=3.14`. A standalone `offproj` install on 3.13 cannot resolve. The
  workspace runs 3.14, so this is invisible there. Pre-existing; not introduced
  by this port.

## Cost

Measured on this hardware, per subject, at the `mua` quality tier:

| Subject | units | ACGs | fit |
|---|---|---|---|
| CNPIX2-Segundo | 76 | 81 s | 58 s |
| CNPIX10-Charles | 534 | 315 s | 395 s |

Roughly linear in unit count; the first spike-train read per probe pays an
~80 s NFS penalty. Budget a couple of hours for the full 19-subject cohort.
