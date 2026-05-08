# %% [markdown]
# # NOD unit counts by quality filter
#
# Per-subject totals of spike-sorted units that pass each of the four
# threshold sets defined in `rats.units.get_threshold_kwargs`:
#
# - `mua` — `required_threshold="conservative"`, no isolation/false-negative gates.
# - `permissive` — `sua_permissive`: required + permissive isolation + permissive
#   false-negatives.
# - `moderate` — `sua_moderate`.
# - `conservative` — `sua_conservative`.
#
# Subjects are discovered from
# `wisc_ecephys_tools.rats.utils.get_subject_experiment_probe_tuples` so every
# CNPIX subject whose YAML lists `novel_objects_deprivation` is included
# (independent of the hard-coded `rats.f25._get_manifest`). Counts are summed
# across probes for each subject; a subject with no loadable probe sorting is
# shown as `N/A`.

# %%
from __future__ import annotations

import pandas as pd

import wisc_ecephys_tools as wet
import wisc_ecephys_tools.rats.utils  # noqa: F401  (registers wet.rats)
from ecephys.units import siutils as units_siutils
from ecephys.wne import siutils as wne_siutils

from rats.units import get_threshold_kwargs

EXPERIMENT = "novel_objects_deprivation"

FILTER_DISPLAY_TO_KWARGS_KEY = {
    "mua": "mua",
    "permissive": "sua_permissive",
    "moderate": "sua_moderate",
    "conservative": "sua_conservative",
}

# %%
threshold_kwargs = get_threshold_kwargs()
filters = {
    display: threshold_kwargs[key]
    for display, key in FILTER_DISPLAY_TO_KWARGS_KEY.items()
}
filters

# %%
sorting_project = wet.projects.get_sglx_project("shared")
subject_probe_pairs = wet.rats.utils.get_subject_experiment_probe_tuples(
    experiment_filter=lambda x: x == EXPERIMENT,
    expand_probes=True,
)
print(f"Discovered {len(subject_probe_pairs)} (subject, experiment, probe) tuples")
subject_probe_pairs[:5]


# %%
# Cluster-info columns to keep, mirroring PROPERTIES_FROM_SORTING_DIR in
# ecephys/wne/project.py:184-195. Some older cluster_info.tsv files include
# stale `firing_rate`, `amplitude_cutoff`, `isi_violations_ratio`, etc. from
# previous curation rounds; the heavy path drops these via
# extractor.delete_property so the postpro metrics.csv versions win. We do the
# same here, otherwise pandas would append `_x`/`_y` suffixes on merge and the
# filter would silently fail to find the column.
KS_INFO_KEEP = {
    "cluster_id",
    "Amplitude",
    "ContamPct",
    "KSLabel",
    "amp",
    "ch",
    "depth",
    "fr",
    "n_spikes",
    "quality",
    "sh",
}


def load_property_frame(subject: str, probe: str) -> pd.DataFrame:
    """Per-cluster property frame from `cluster_info.tsv` + `postpro/metrics.csv`.

    Bypasses `sglx_sorting_project.get_kilosort_extractor`, which spends ~99%
    of its time inside `spikeinterface.read_kilosort` reading `spike_times.npy`,
    `spike_clusters.npy`, templates, and channel maps that the property-only
    filters don't need. Mirrors SI's rename of the cluster-group column
    (`group` → `quality`; see `ecephys/wne/siutils.py:95`).
    """
    sorting_dir = (
        sorting_project.get_alias_subject_directory(EXPERIMENT, "full", subject)
        / f"sorting.{probe}"
    )
    cluster_info = pd.read_csv(
        sorting_dir / "si_output/sorter_output/cluster_info.tsv", sep="\t"
    ).rename(columns={"group": "quality"})
    cluster_info = cluster_info[
        [c for c in cluster_info.columns if c in KS_INFO_KEEP]
    ]
    metrics = pd.read_csv(sorting_dir / "postpro/metrics.csv")
    return cluster_info.merge(
        metrics, on="cluster_id", how="left", validate="one_to_one"
    )


def count_units_for_probe(subject: str, probe: str) -> dict[str, int | None]:
    """Return a {filter_display_name: passing_unit_count} dict for one probe.

    Returns all-`None` if the probe has no sorting on disk.
    """
    if not wet.rats.utils.has_sorting(subject, EXPERIMENT, probe, sorting_project):
        return {name: None for name in filters}

    properties = load_property_frame(subject, probe)

    counts: dict[str, int | None] = {}
    for display_name, kwargs in filters.items():
        simple_filters, callable_filters = wne_siutils.get_quality_metric_filters(
            **kwargs
        )
        refined = units_siutils.refine_clusters(
            properties,
            simple_filters,
            callable_filters,
            include_nans=True,
            verbose=False,
        )
        counts[display_name] = int(len(refined))
    return counts


# %%
rows = []
for subject, _experiment, probe in subject_probe_pairs:
    counts = count_units_for_probe(subject, probe)
    rows.append({"subject": subject, "probe": probe, **counts})

per_probe = pd.DataFrame(rows)
per_probe = per_probe.sort_values(["subject", "probe"]).reset_index(drop=True)
per_probe

# %% [markdown]
# ## Per-subject totals
#
# Sum across probes. Subjects with no loadable probes show as `N/A`.

# %%
filter_cols = list(filters.keys())


def _aggregate(group: pd.DataFrame) -> pd.Series:
    if group[filter_cols].isna().all(axis=None):
        return pd.Series({col: pd.NA for col in filter_cols})
    return group[filter_cols].sum(min_count=1)


per_subject = (
    per_probe.groupby("subject", sort=True)
    .apply(_aggregate, include_groups=False)
    .reset_index()
)
for col in filter_cols:
    per_subject[col] = per_subject[col].astype("Int64")

# %%
display = per_subject.copy()
for col in filter_cols:
    display[col] = display[col].astype(object).where(display[col].notna(), "N/A")
print(display.to_string(index=False))
