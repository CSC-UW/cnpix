# AGENTS.md — cnpix

## What This Is

Project-agnostic code for the **CNPIX chronic Neuropixels rat dataset**. Anything that
describes a *unit* rather than a *paper* belongs here: quality tiers, autocorrelogram
metrics, cell-type labels, anatomical region mapping.

The rule of thumb: if two papers analyzing the same rats would both want it, and would
be embarrassed to disagree about it, it goes in `cnpix`.

Areas: `units/` (quality tiers, ACG metrics, cell-type labels, region mapping), `f25/`,
and `evaluation/` (manual OFF-period ground truth, label QC, and the pixel/event metric
kernels every detection method is scored with — see `cnpix/evaluation/__init__.py`).

## Build & Test

```bash
# Always run through the workspace, so sibling packages resolve to local editable
# sources rather than their pinned git URLs.
cd gfys_workspace
uv run --all-extras --group dev pytest ../cnpix/tests
uv run --all-extras --group dev ruff check ../cnpix/src
```

## Subpackages

| Module | Purpose |
|---|---|
| `cnpix.units` | Unit quality tiers, ACG metrics, cell-type classification |
| `cnpix.f25` | Findlay-2025 LFP/CSD helpers |

## `cnpix.units`

Pipeline, in order. Steps 1-2 are per-subject and expensive; 3-4 are cohort-level.

```bash
cnpix-units list-subjects              # every subject with a sorting + anatomy + hypnogram
cnpix-units acgs                       # 1. narrow + wide autocorrelogram zarrs
cnpix-units acg-metrics                # 2. triple-exponential fits -> acg_metrics.pqt
cnpix-units cohort-tables              # 3. join with sorting properties, assign quality tiers
cnpix-units cell-types                 # 4. Petersen classification -> cell_types.pqt
cnpix-units run-all                    # all of the above
```

Consumers should use the readers, not the pipeline:

```python
from cnpix.units import load_cell_types
labels = load_cell_types(quality_tier="sua_moderate")
```

### Outputs

Everything is written to the **`cnpix` project** (`/Volumes/npx_nfs/nobak/cnpix`),
deliberately *not* to any one paper's project directory.

| File | Level | Contents |
|---|---|---|
| `{narrow,wide}_autocorrelograms.zarr` | per subject | ACGs per (cluster, state) |
| `acg_metrics.pqt` | per subject **and** cohort | fit params, shoulder, burst index, moment |
| `mps_metrics.pqt` | cohort | sorting properties + anatomy + `region` |
| `aggregated_cell_metrics.pqt` | cohort | the above joined, plus cell-type labels |
| `cluster_quality.pqt` | cohort | per unit, the strictest quality tier it passes |
| `cell_types.pqt` | cohort | per unit, `narrow_wide_cell_type` + `petersen_cell_type` |

### Cell-type scheme

Petersen et al. (CellExplorer): a **narrow** waveform (`peak_to_valley <= 425 µs`) marks
a putative fast-spiking interneuron; among broad-spiking units, a slow ACG rise
(`tau_rise > 6 ms` cortical, `> 3 ms` hippocampal) marks a putative wide interneuron,
otherwise pyramidal.

Applied **only** to `region in {cortical, hippocampal}`. Thalamus, claustrum, and white
matter are deliberately left unlabeled rather than forced into a cortical taxonomy.

Labels are computed on **NREM** metrics and broadcast to every state, so a unit has one
identity across the recording.

### Things not to "clean up"

- The ACG fitting constants in `acg.py` (`a0`/`lb`/`ub`, `maxfev`, the zeroing of the
  central bins) are a port of CellExplorer's `fit_ACG.m`. Changing them changes the
  cell type of every previously labeled unit.
- `QUALITY_TIERS` order matters: tiers are nested, and `assign_cluster_quality` relies on
  iterating permissive→strict so the strictest passing tier wins.
- `cnpix.units.get_threshold_kwargs()["sua_moderate"]` must stay identical to
  `offproj.units.QUALITY_FILTERS["su1"]`; a regression test pins this.

### Cohort

The subject list is derived from the **sorting registry** (`cnpix.units.get_sortings`),
not a hard-coded manifest. This is deliberate: `findlay2025a` gated its cohort on having
a sharp-wave probe, which silently excluded three subjects (CNPIX7-Giuseppe, CNPIX13-Al,
CNPIX16-Walter) whose data are perfectly good. Do not reintroduce a manifest.

As of 2026, sortings exist only for the `full` alias of `novel_objects_deprivation`
(19 subjects); `wisc_ecephys_tools.rats.utils.has_sorting` asserts as much.

## Install Pattern

Sibling workspace deps (`ecephys`, `wisc-ecephys-tools`) are declared as **direct git
URLs** so an external user can `uv sync` this package alone. The workspace overrides them
with local editable paths. See the root `AGENTS.md` "Package Dual-Use Install Pattern".

## See Also

- Root `AGENTS.md` — workspace overview, command rules, data rules
- `offproj/src/offproj/celltype_firing.py` — the first downstream consumer
