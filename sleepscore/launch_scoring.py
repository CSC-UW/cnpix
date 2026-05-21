from pathlib import Path

import loupe as lp
import polars as pl
import xarray as xr

MOUNT_POINT = Path("/Volumes/OWC Envoy Ultra")
SUBJECT = "CNPIX12-Santiago"
HYPNOGRAM_FNAME = "Full.Liberal_consensus_hypnogram.htsv"
UNIT_PROBES = ["imec0", "imec1"]
LFP_PROBES = ["imec0", "imec1"]

data_dir = MOUNT_POINT / SUBJECT

# Top-to-bottom: Hippocampus (CA1-SR), Deep CX (PPC), HippSuperficial CX (PPC)
print("Loading simple scoring LFPs and EMG...")
scoring_lfp = xr.open_dataarray(data_dir / "scoring_lfp.zarr").load()
# Derived/synthetic EMG. Changes slowly. Having right Y axis limits is important
scoring_emg = xr.open_dataarray(data_dir / "scoring_emg.zarr").load()

hypnogram_path = data_dir / HYPNOGRAM_FNAME
hypnogram_schema = lp.LabelSchema(
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
for probe in UNIT_PROBES:
    unit_path = data_dir / f"{probe}.units.parquet"
    if unit_path.exists():
        print(f"Loading {unit_path}...")
        unit_data = pl.read_parquet(unit_path)
        units[probe] = unit_data

lfps = {}
for probe in LFP_PROBES:
    lfp_path = data_dir / f"{probe}.lf.zarr"
    if lfp_path.exists():
        print(f"Loading {lfp_path}...")
        lfp_data = xr.open_zarr(lfp_path)["lfp"]
        lfps[probe] = lfp_data

anatomy = {}
anat_probes = sorted(list(set(UNIT_PROBES) | set(LFP_PROBES)))
for probe in anat_probes:
    anat_path = data_dir / f"{probe}.structures.htsv"
    if anat_path.exists():
        print(f"Loading {anat_path}...")
        anat_data = pl.read_csv(anat_path, separator="\t")
        anatomy[probe] = anat_data


def _y_to_acronym(y, anat):
    m = anat.filter(pl.col("lo").le(y) & pl.col("hi").ge(y))["acronym"]
    return m[0] if len(m) else "???"


print(f"Labeling {anat_probes} with anatomy...")
for probe, anat in anatomy.items():
    if probe in lfps:
        lf = lfps[probe]
        anatomy_coord = [_y_to_acronym(y, anat) for y in lf["y"].values]
        lfps[probe] = lf.assign_coords(anatomy=("channel", anatomy_coord))
    if probe in units:
        u = units[probe]
        unit_anatomy = [_y_to_acronym(d, anat) for d in u["depth"].to_list()]
        units[probe] = u.with_columns(pl.Series("anatomy", unit_anatomy))

# Build a per-probe spike raster, with units ordered by depth.
matrix_df = None
if UNIT_PROBES:
    print("Pushing units around...")
    spike_dfs = []
    for probe in UNIT_PROBES:
        if probe not in units:
            continue
        u_sorted = units[probe].sort("depth").with_row_index("depth_rank")
        spikes = (
            u_sorted.select(["depth_rank", "spike_times"])
            .explode("spike_times")
            .drop_nulls("spike_times")
            .rename({"spike_times": "time"})
            .with_columns(pl.lit(probe).alias("probe"))
        )
        spike_dfs.append(spikes)
    if spike_dfs:
        matrix_df = pl.concat(spike_dfs)

print("Launching Loupe...")
w = lp.view(
    data=[
        lp.TraceConfig(data=scoring_lfp, mode="stacked-subplots"),
        lp.TraceConfig(data=scoring_emg),
    ],
    matrix_df=matrix_df,
    y_col="depth_rank",
    group_col="probe",
    matrix_name="units",
    keymap=keymap,
    label_colors=state_colors,
    labels=str(hypnogram_path) if hypnogram_path.exists() else None,
    label_schema=hypnogram_schema,
)

# Suggestions:
# - Set label alpha to 0.5
# - View 1000s, autoscale Y-axes, then zoom in to ~10-20s.
