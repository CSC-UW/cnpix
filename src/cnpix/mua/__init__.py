"""The MUA amplitude envelope: build it, store it, stage it.

``mua_traces.zarr`` is the shared substrate for MUA-based OFF-period
detection. One preprocessing chain produces it per (subject, probe), and every
detection method reads the same file, so the signal is not re-derived (or
silently re-derived *differently*) per method.

| Module | Purpose |
|---|---|
| ``preprocess`` | The lazy SpikeInterface chain |
| ``files`` | ``mua_traces.zarr`` paths and loading |
| ``motion`` | Legacy AP-band motion correction (optional ``lnsp``) |
| ``staging`` | Fast rclone copies between NFS and local NVME |
| ``cli`` | The ``cnpix-mua`` entry point |
"""

from cnpix.mua.files import get_mua_traces_path, load_mua_traces
from cnpix.mua.preprocess import build_preprocessing_chain

__all__ = [
    "build_preprocessing_chain",
    "get_mua_traces_path",
    "load_mua_traces",
]
