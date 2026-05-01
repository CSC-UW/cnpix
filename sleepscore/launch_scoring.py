# %%
from pathlib import Path

import loupe as lp
import xarray as xr

SUBJECT = "CNPIX12-Santiago"

scoring_lfp = Path(
    f"/Volumes/npx_nfs/nobak/shared/novel_objects_deprivation/{SUBJECT}/scoring_lfp.zarr"
)
scoring_emg = Path(
    f"/Volumes/npx_nfs/nobak/shared/novel_objects_deprivation/{SUBJECT}/scoring_emg.zarr"
)

scoring_lfp = xr.open_dataarray(scoring_lfp).load()
scoring_emg = xr.open_dataarray(scoring_emg).load()

hypnogram_path = Path(
    f"/Volumes/npx_nfs/shared_s3/novel_objects_deprivation/{SUBJECT}/hypnogram.htsv"
)
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


w = lp.view(
    data=[
        lp.TraceConfig(data=scoring_lfp, mode="dense"),
        lp.TraceConfig(data=scoring_emg),
    ],
    keymap=keymap,
    label_colors=state_colors,
    labels=str(hypnogram_path) if hypnogram_path.exists() else None,
    label_schema=hypnogram_schema,
)
