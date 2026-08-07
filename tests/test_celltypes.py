"""Tests for putative cell-type classification.

These pin the published Petersen criteria. The thresholds are scientific
constants, not tuning knobs -- if a test here fails, the classification of every
previously labeled unit has changed.
"""

import numpy as np
import pandas as pd
import pytest

from cnpix.units import celltypes


NARROW = celltypes.NARROW_PEAK_TO_VALLEY_S  # 425 us


class TestClassifyNarrowWide:
    def test_boundary_is_inclusive_of_narrow(self):
        assert celltypes.classify_narrow_wide(NARROW) == "narrow"
        assert celltypes.classify_narrow_wide(np.nextafter(NARROW, 1)) == "wide"

    def test_typical_values(self):
        assert celltypes.classify_narrow_wide(0.00020) == "narrow"
        assert celltypes.classify_narrow_wide(0.00060) == "wide"

    def test_missing_waveform_is_unlabeled(self):
        assert np.isnan(celltypes.classify_narrow_wide(np.nan))


class TestClassifyPetersen:
    def test_narrow_units_are_fast_spiking_regardless_of_acg(self):
        for tau in (0.0, 5.9, 6.1, 100.0):
            assert (
                celltypes.classify_petersen(0.0002, tau, "cortical")
                == "narrow interneuron"
            )

    @pytest.mark.parametrize(
        "region,threshold", [("cortical", 6.0), ("hippocampal", 3.0)]
    )
    def test_tau_rise_threshold_is_region_specific(self, region, threshold):
        wide = 0.0006
        assert (
            celltypes.classify_petersen(wide, threshold + 0.1, region)
            == "wide interneuron"
        )
        # The comparison is strictly greater-than, so the threshold itself is
        # pyramidal.
        assert celltypes.classify_petersen(wide, threshold, region) == "pyramidal"

    def test_hippocampal_threshold_is_lower_than_cortical(self):
        # tau_rise = 4 ms separates the two regions' verdicts.
        assert celltypes.classify_petersen(0.0006, 4.0, "cortical") == "pyramidal"
        assert (
            celltypes.classify_petersen(0.0006, 4.0, "hippocampal")
            == "wide interneuron"
        )

    def test_unclassifiable_regions_are_unlabeled(self):
        for region in ("thalamic", "other"):
            assert np.isnan(celltypes.classify_petersen(0.0006, 1.0, region))

    def test_broad_unit_with_failed_acg_fit_is_unlabeled(self):
        # A failed fit must not silently become "pyramidal".
        assert np.isnan(celltypes.classify_petersen(0.0006, np.nan, "cortical"))

    def test_missing_waveform_is_unlabeled(self):
        assert np.isnan(celltypes.classify_petersen(np.nan, 1.0, "cortical"))


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "subject",
            "experiment",
            "probe",
            "cluster_id",
            "state",
            "region",
            "peak_to_valley",
            "tau_rise",
        ],
    )


class TestAssignCellTypes:
    def test_labels_are_broadcast_from_nrem_to_every_state(self):
        # Only the NREM row carries usable metrics; the others must still be
        # labeled, because a unit has one identity across the recording.
        df = _frame(
            [
                ("S", "E", "imec0", 1, "NREM", "cortical", 0.0002, 1.0),
                ("S", "E", "imec0", 1, "Wake", "cortical", np.nan, np.nan),
                ("S", "E", "imec0", 1, "REM", "cortical", np.nan, np.nan),
            ]
        )
        out = celltypes.assign_cell_types(df)
        assert set(out["petersen_cell_type"]) == {"narrow interneuron"}
        assert set(out["narrow_wide_cell_type"]) == {"narrow"}

    def test_non_nrem_metrics_do_not_drive_classification(self):
        # The Wake row looks narrow, but NREM is the classification state.
        df = _frame(
            [
                ("S", "E", "imec0", 1, "NREM", "cortical", 0.0006, 1.0),
                ("S", "E", "imec0", 1, "Wake", "cortical", 0.0001, 1.0),
            ]
        )
        out = celltypes.assign_cell_types(df)
        assert set(out["petersen_cell_type"]) == {"pyramidal"}

    def test_unclassifiable_region_stays_null_in_every_state(self):
        df = _frame(
            [
                ("S", "E", "imec0", 7, "NREM", "other", 0.0002, 1.0),
                ("S", "E", "imec0", 7, "Wake", "other", 0.0002, 1.0),
            ]
        )
        out = celltypes.assign_cell_types(df)
        assert out["petersen_cell_type"].isna().all()
        assert out["narrow_wide_cell_type"].isna().all()

    def test_units_are_labeled_independently(self):
        df = _frame(
            [
                ("S", "E", "imec0", 1, "NREM", "cortical", 0.0002, 1.0),
                ("S", "E", "imec0", 2, "NREM", "cortical", 0.0006, 9.0),
                ("S", "E", "imec1", 1, "NREM", "cortical", 0.0006, 1.0),
            ]
        )
        out = celltypes.assign_cell_types(df).set_index(["probe", "cluster_id"])
        assert out.loc[("imec0", 1), "petersen_cell_type"] == "narrow interneuron"
        assert out.loc[("imec0", 2), "petersen_cell_type"] == "wide interneuron"
        assert out.loc[("imec1", 1), "petersen_cell_type"] == "pyramidal"

    def test_row_count_is_preserved(self):
        df = _frame(
            [
                ("S", "E", "imec0", 1, "NREM", "cortical", 0.0002, 1.0),
                ("S", "E", "imec0", 1, "Wake", "cortical", np.nan, np.nan),
            ]
        )
        assert len(celltypes.assign_cell_types(df)) == len(df)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="Missing required columns"):
            celltypes.assign_cell_types(pd.DataFrame({"state": ["NREM"]}))
