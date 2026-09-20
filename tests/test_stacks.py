import json

import pytest
from cnpix.evaluation import paths

from cnpix import stacks


def test_paths_live_in_samoffs_without_method_level():
    p = stacks.stack_path("CNPIX2-Segundo", "imec0", "Early.REC.NREM")
    assert p.name == "off_stacks.ome.zarr"
    assert p.parts[-3:-1] == ("probe=imec0", "condition=Early.REC.NREM")
    assert "method=sam3" not in p.parts
    assert "samoffs" in p.parts
    t = stacks.timestamps_path("CNPIX2-Segundo", "imec0", "Early.REC.NREM")
    assert t.parent == p.parent and t.name == "timestamps.zarr"
    assert stacks.stack_dir("CNPIX2-Segundo", "imec0", "Early.REC.NREM") == p.parent


def test_readers_refuse_a_stale_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "label_dir", lambda project, subject, **kw: tmp_path)
    (tmp_path / "STALE.json").write_text(
        json.dumps({"stale": True, "reason": "old hypnogram"})
    )
    with pytest.raises(paths.StaleDataError):
        stacks.stack_path("s", "p", "c")
    assert (
        stacks.stack_path("s", "p", "c", allow_stale=True)
        == tmp_path / "off_stacks.ome.zarr"
    )
    assert stacks.stack_dir("s", "p", "c") == tmp_path


@pytest.mark.parametrize("meta_name", [".zarray", "zarr.json"])
def test_stack_shape_reads_metadata_only(tmp_path, monkeypatch, meta_name):
    meta_dir = tmp_path / "off_stacks.ome.zarr" / "0" / "0"
    meta_dir.mkdir(parents=True)
    (meta_dir / meta_name).write_text(json.dumps({"shape": [101, 384, 1200]}))
    monkeypatch.setattr(paths, "label_dir", lambda project, subject, **kw: tmp_path)
    assert stacks.stack_shape("CNPIX2-Segundo", "imec0", "Early.REC.NREM.Match") == (
        101,
        384,
        1200,
    )
    assert stacks.zarr_shape(tmp_path / "off_stacks.ome.zarr") == (101, 384, 1200)


def test_open_stack_returns_present_series(tmp_path, monkeypatch):
    import zarr

    store = tmp_path / "off_stacks.ome.zarr"
    root = zarr.open_group(str(store), mode="w")
    for series, value in (("0", 0), ("2", 1)):
        root.create_group(series)
        arr = zarr.open_array(
            str(store / series / "0"), mode="w", shape=(2, 3, 4), dtype="uint8"
        )
        arr[:] = value
    monkeypatch.setattr(paths, "label_dir", lambda project, subject, **kw: tmp_path)
    out = stacks.open_stack("s", "p", "c")
    assert set(out) == {"ap_data", "lfp"}
    assert out["ap_data"].shape == (2, 3, 4)
    assert int(out["lfp"][0, 0, 0]) == 1
