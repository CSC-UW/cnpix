from pathlib import Path

import loupe as lp
import polars as pl
import xarray as xr

MOUNT_POINT = Path("/Volumes/OWC Envoy Ultra")
SUBJECT = "CNPIX12-Santiago"
HYPNOGRAM_FNAME = "Full.Liberal_consensus_hypnogram.htsv"
UNIT_PROBES = ["imec0", "imec1"]
LFP_PROBES = ["imec0", "imec1"]
LOUPE_PROFILE = "simple"  # "simple" or "simple_with_units"

assert LOUPE_PROFILE in ("simple", "simple_with_units")

need_units = LOUPE_PROFILE == "simple_with_units"

data_dir = MOUNT_POINT / SUBJECT

# Top-to-bottom: Hippocampus (CA1-SR), Deep CX (PPC), HippSuperficial CX (PPC)
print("Loading simple scoring LFPs and EMG...")
scoring_lfp = xr.open_dataarray(data_dir / "scoring_lfp.zarr").load()
# Derived/synthetic EMG. Changes slowly. Having right Y axis limits is important
scoring_emg = xr.open_dataarray(data_dir / "scoring_emg.zarr").load()

hypnogram_path = data_dir / HYPNOGRAM_FNAME
hypnogram_schema = lp.IntervalLabelSchema(
    start_col="start_time",
    end_col="end_time",
    label_col="state",
)

keymap = {
    "Wake": ["w", "0"],
    "N1": ["1"],
    "N2": ["2"],
    "NREM": ["n", "3"],
    "IS": ["i", "4"],
    "REM": ["r", "5"],
    "MA": ["m", "`"],
    "Arousal": ["a", "9"],
    "Trans": ["t"],  # Transition from Wake to NREM
    "Artifact": ["x"],
    "None": ["v"],
    "QWK": ["q"],  # Quiet Wake
    "AWK": ["e"],  # Active/exploratory wake
}

state_colors = {
    "Wake": "#98fb98",  # palegreen
    "N1": "#d8bfd8",  # thistle
    "N2": "#dda0dd",  # plum
    "NREM": "#da70d6",  # orchid
    "IS": "#deb887",  # burlywood
    "REM": "#ffe4c4",  # bisque
    "MA": "#afeeee",  # paleturquoise
    "Arousal": "#90ee90",  # lightgreen
    "Trans": "#dcdcdc",  # gainsboro
    "Artifact": "#dc143c",  # crimson
    "None": "#ffffff",  # white
    "QWK": "#228b22",  # forestgreen
    "AWK": "#32cd32",  # limegreen
}

units = {}
if need_units:
    for probe in UNIT_PROBES:
        unit_path = data_dir / f"{probe}.units.parquet"
        if unit_path.exists():
            print(f"Loading {unit_path}...")
            units[probe] = pl.read_parquet(unit_path)

anatomy = {}
anat_probes = sorted(set(UNIT_PROBES) if need_units else [])
for probe in anat_probes:
    anat_path = data_dir / f"{probe}.structures.htsv"
    if anat_path.exists():
        print(f"Loading {anat_path}...")
        anatomy[probe] = pl.read_csv(anat_path, separator="\t")


def _y_to_acronym(y, anat):
    m = anat.filter(pl.col("lo").le(y) & pl.col("hi").ge(y))["acronym"]
    return m[0] if len(m) else "???"


if anatomy:
    print(f"Labeling {list(anatomy)} with anatomy...")
for probe, anat in anatomy.items():
    if probe in units:
        u = units[probe]
        unit_anatomy = [_y_to_acronym(d, anat) for d in u["depth"].to_list()]
        units[probe] = u.with_columns(pl.Series("anatomy", unit_anatomy))

spikes = {}
for probe in units:
    print(f"Exploding spike times for {probe}...")
    spikes[probe] = (
        units[probe]
        .select(["depth", "anatomy", "spike_times"])
        .explode("spike_times")
        .drop_nulls("spike_times")
    )

data_list = [
    lp.TraceConfig(data=scoring_lfp, mode="stacked-subplots"),
    lp.TraceConfig(data=scoring_emg),
] + [
    lp.RasterConfig(
        data=spikes[probe],
        time_col="spike_times",
        order_by="depth",
        split_by="anatomy",
        array_name=probe,
    )
    for probe in spikes
]

print("Launching Loupe...")
w = lp.view(
    data_list,
    keymap=keymap,
    label_colors=state_colors,
    interval_labels=str(hypnogram_path) if hypnogram_path.exists() else None,
    interval_label_schema=hypnogram_schema,
    interval_label_alpha=0.25,
)

# Suggestions:
# - Set label alpha to 0.5
# - View 1000s, autoscale Y-axes, then zoom in to ~10-20s.
