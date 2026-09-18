import pathlib

from cnpix.evaluation import paths


def test_projects_are_samoffs():
    assert paths.MANUAL_LABELS_PROJECT == "samoffs_s3"
    assert paths.MODEL_LABELS_PROJECT == "samoffs"


def test_label_dir_has_no_method_level_for_models():
    d = paths.label_dir(
        "samoffs", "CNPIX15-Claude", model="2026-05-03_nrem-all",
        probe="imec0", condition="Early.REC.NREM",
    )
    assert d.parts[-3:] == (
        "model=2026-05-03_nrem-all", "probe=imec0", "condition=Early.REC.NREM",
    )
    assert d.parts[-4] == "CNPIX15-Claude"
    assert pathlib.Path(*d.parts[:-4]).name == "novel_objects_deprivation"
